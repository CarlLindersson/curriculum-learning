"""
This script is a pygame which implements a simple curriculum learning framework.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4


class Operation(str, Enum):
    """The two hidden transformations participants must learn."""

    SIZE = "size"
    SHAPE = "shape"


class CurriculumMode(str, Enum):
    INTERLEAVED = "interleaved"
    BLOCKED = "blocked"
    PROGRESSIVELY_INTERLEAVED = "progressively_interleaved"
    PROGRESSIVELY_BLOCKED = "progressively_blocked"


TRAINING_SHAPES = ("square", "circle")
NOVEL_SHAPES = ("triangle", "star", "pentagon", "hexagon")
SHAPE_PARTNERS = {
    "square": "circle",
    "circle": "square",
    "triangle": "star",
    "star": "triangle",
    "pentagon": "hexagon",
    "hexagon": "pentagon",
}
LARGE_SYMBOL_RADIUS = 58
SMALL_SYMBOL_RADIUS = 35


@dataclass(frozen=True)
class SymbolState:
    shape: str
    is_large: bool


@dataclass(frozen=True)
class Trial:
    start: SymbolState
    operation: Operation
    correct: SymbolState
    options: tuple[SymbolState, SymbolState]
    correct_option: int


@dataclass(frozen=True)
class CurriculumConfig:
    """Validated settings accepted by :class:`CurriculumGame`."""

    mode: CurriculumMode = CurriculumMode.INTERLEAVED
    mastery_threshold: float = 0.80
    minimum_trials_per_operation: int = 10
    maximum_training_trials: int = 100
    test_trials: int = 40
    block_size: int = 20
    start_ms: int = 1000
    operation_ms: int = 1000
    response_ms: int = 10_000
    feedback_ms: int = 1000

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None = None) -> "CurriculumConfig":
        values = dict(values or {})
        raw_mode = values.pop("mode", values.pop("type", values.pop("name", "interleaved")))
        mode = _normalise_curriculum_mode(str(raw_mode))
        known_fields = {
            "mastery_threshold",
            "minimum_trials_per_operation",
            "maximum_training_trials",
            "test_trials",
            "block_size",
            "start_ms",
            "operation_ms",
            "response_ms",
            "feedback_ms",
        }
        unknown = sorted(set(values) - known_fields)
        if unknown:
            raise ValueError(f"Unknown curriculum setting(s): {', '.join(unknown)}")
        config = cls(mode=mode, **values)
        config._validate()
        return config

    def _validate(self) -> None:
        if not 0 < self.mastery_threshold <= 1:
            raise ValueError("mastery_threshold must be in (0, 1]")
        integer_fields = (
            "minimum_trials_per_operation",
            "maximum_training_trials",
            "test_trials",
            "block_size",
            "start_ms",
            "operation_ms",
            "response_ms",
            "feedback_ms",
        )
        for field_name in integer_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")


def _normalise_curriculum_mode(value: str) -> CurriculumMode:
    normalised = re.sub(r"[\s-]+", "_", value.strip().lower())
    # Accept the spelling used in the original design brief as an input alias.
    normalised = normalised.replace("progressivly", "progressively")
    aliases = {
        "interleaved": CurriculumMode.INTERLEAVED,
        "blocked": CurriculumMode.BLOCKED,
        "progressively_interleaved": CurriculumMode.PROGRESSIVELY_INTERLEAVED,
        "progressive_interleaved": CurriculumMode.PROGRESSIVELY_INTERLEAVED,
        "progressively_blocked": CurriculumMode.PROGRESSIVELY_BLOCKED,
        "progressive_blocked": CurriculumMode.PROGRESSIVELY_BLOCKED,
    }
    try:
        return aliases[normalised]
    except KeyError as exc:
        choices = ", ".join(mode.value for mode in CurriculumMode)
        raise ValueError(f"Unknown curriculum mode {value!r}. Choose from: {choices}") from exc


def apply_operation(state: SymbolState, operation: Operation) -> SymbolState:
    """Apply one transformation to a symbol state."""

    if operation is Operation.SIZE:
        return SymbolState(state.shape, not state.is_large)
    try:
        partner = SHAPE_PARTNERS[state.shape]
    except KeyError as exc:
        raise ValueError(f"Shape {state.shape!r} has no configured partner") from exc
    return SymbolState(partner, state.is_large)


def other_operation(operation: Operation) -> Operation:
    return Operation.SHAPE if operation is Operation.SIZE else Operation.SIZE


def make_trial(
    operation: Operation,
    shapes: Sequence[str],
    rng: random.Random,
) -> Trial:
    """Generate a trial whose foil is the result of the other operation."""

    start = SymbolState(rng.choice(tuple(shapes)), rng.choice((False, True)))
    correct = apply_operation(start, operation)
    foil = apply_operation(start, other_operation(operation))
    correct_option = rng.randrange(2)
    options = (correct, foil) if correct_option == 0 else (foil, correct)
    return Trial(start, operation, correct, options, correct_option)


class CurriculumScheduler:
    """Select operations and decide when the training phase is complete."""

    def __init__(self, config: CurriculumConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.outcomes: dict[Operation, list[bool]] = {
            Operation.SIZE: [],
            Operation.SHAPE: [],
        }
        self.operations: list[Operation] = []
        self.initial_stage_trials: int | None = None

    @property
    def trial_count(self) -> int:
        return len(self.operations)

    def accuracy(self, operation: Operation) -> float | None:
        values = self.outcomes[operation]
        return sum(values) / len(values) if values else None

    def operation_mastered(self, operation: Operation) -> bool:
        values = self.outcomes[operation]
        return (
            len(values) >= self.config.minimum_trials_per_operation
            and sum(values) / len(values) >= self.config.mastery_threshold
        )

    @property
    def mastered(self) -> bool:
        return all(self.operation_mastered(operation) for operation in Operation)

    @property
    def should_end_training(self) -> bool:
        return self.mastered or self.trial_count >= self.config.maximum_training_trials

    @property
    def progressive_block_size(self) -> int | None:
        if self.initial_stage_trials is None:
            return None
        return max(1, math.ceil(self.initial_stage_trials * 0.20))

    def next_operation(self) -> Operation:
        mode = self.config.mode
        if mode is CurriculumMode.INTERLEAVED:
            return self.rng.choice(tuple(Operation))
        if mode is CurriculumMode.BLOCKED:
            block = (self.trial_count // self.config.block_size) % 2
            return (Operation.SIZE, Operation.SHAPE)[block]
        if mode is CurriculumMode.PROGRESSIVELY_INTERLEAVED:
            if self.initial_stage_trials is None:
                return Operation.SIZE
            return self.rng.choice(tuple(Operation))

        # In the progressively blocked curriculum, the first size-only stage
        # establishes the data-dependent block length. Thereafter shape and size
        # alternate, starting with the newly introduced shape operation.
        if self.initial_stage_trials is None:
            return Operation.SIZE
        completed_after_stage = self.trial_count - self.initial_stage_trials
        block = (completed_after_stage // self.progressive_block_size) % 2
        return (Operation.SHAPE, Operation.SIZE)[block]

    @property
    def current_phase(self) -> str:
        """Return the curriculum phase for the next/current training trial."""

        mode = self.config.mode
        if mode is CurriculumMode.INTERLEAVED:
            return "interleaved"
        if mode is CurriculumMode.BLOCKED:
            return f"block{self.trial_count // self.config.block_size + 1}"
        if mode is CurriculumMode.PROGRESSIVELY_INTERLEAVED:
            return "stage1" if self.initial_stage_trials is None else "stage2"
        if self.initial_stage_trials is None:
            return "block1"
        completed_after_stage = self.trial_count - self.initial_stage_trials
        block_number = completed_after_stage // self.progressive_block_size + 2
        return f"block{block_number}"

    def record(self, operation: Operation, correct: bool) -> None:
        self.operations.append(operation)
        self.outcomes[operation].append(bool(correct))
        if (
            self.config.mode
            in (
                CurriculumMode.PROGRESSIVELY_INTERLEAVED,
                CurriculumMode.PROGRESSIVELY_BLOCKED,
            )
            and self.initial_stage_trials is None
            and self.operation_mastered(Operation.SIZE)
        ):
            self.initial_stage_trials = self.trial_count


class SessionRecorder:
    """Append trial and session data to CSV files in the repository's data folder."""

    TRIAL_FIELDS = (
        "session_id",
        "participant_id",
        "curriculum",
        "phase",
        "trial_number",
        "operation",
        "start_shape",
        "start_size",
        "correct_shape",
        "correct_size",
        "top_shape",
        "top_size",
        "bottom_shape",
        "bottom_size",
        "response",
        "is_correct",
        "response_time_ms",
        "score",
        "timestamp_utc",
        "session_number",
        "start_symbol",
        "option_symbols",
        "choice",
        "timeout",
        "curriculum_phase",
        "response_type",
    )
    SUMMARY_FIELDS = (
        "session_id",
        "participant_id",
        "curriculum",
        "score",
        "total_trials",
        "total_accuracy",
        "training_trials",
        "training_accuracy",
        "test_trials",
        "test_accuracy",
        "completed_at_utc",
        "session_number",
    )

    def __init__(
        self,
        participant_id: str,
        curriculum: CurriculumMode,
        data_dir: Path | None = None,
    ) -> None:
        self.participant_id = participant_id
        self.curriculum = curriculum
        self.session_id = uuid4().hex
        self.data_dir = data_dir or Path(__file__).resolve().parents[1] / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_number = _next_session_number(self.data_dir, participant_id)
        self.trials_path = _compatible_csv_path(
            self.data_dir / "curriculum_trials.csv", self.TRIAL_FIELDS
        )
        self.summaries_path = _compatible_csv_path(
            self.data_dir / "curriculum_sessions.csv", self.SUMMARY_FIELDS
        )
        self.records: list[dict[str, Any]] = []

    def record_trial(
        self,
        *,
        phase: str,
        curriculum_phase: str,
        trial_number: int,
        trial: Trial,
        response: int | None,
        response_type: str,
        is_correct: bool,
        response_time_ms: int | None,
        score: int,
    ) -> None:
        record = build_trial_record(
            session_id=self.session_id,
            participant_id=self.participant_id,
            session_number=self.session_number,
            curriculum=self.curriculum,
            phase=phase,
            curriculum_phase=curriculum_phase,
            trial_number=trial_number,
            trial=trial,
            response=response,
            response_type=response_type,
            is_correct=is_correct,
            response_time_ms=response_time_ms,
            score=score,
        )
        self.records.append(record)
        _append_csv(self.trials_path, self.TRIAL_FIELDS, record)

    def finish(self, score: int) -> int:
        training = [row for row in self.records if row["phase"] == "training"]
        test = [row for row in self.records if row["phase"] == "test"]
        summary = {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "curriculum": self.curriculum.value,
            "score": score,
            "total_trials": len(self.records),
            "total_accuracy": _accuracy(self.records),
            "training_trials": len(training),
            "training_accuracy": _accuracy(training),
            "test_trials": len(test),
            "test_accuracy": _accuracy(test),
            "completed_at_utc": _utc_now(),
            "session_number": self.session_number,
        }
        _append_csv(self.summaries_path, self.SUMMARY_FIELDS, summary)
        scores = _read_scores(self.summaries_path)
        return 1 + sum(previous_score > score for previous_score in scores)

    async def pump(self) -> None:
        """Match the asynchronous recorder interface without desktop overhead."""

        return None

    @property
    def is_synced(self) -> bool:
        return True

    @property
    def sync_error(self) -> str:
        return ""


def build_trial_record(
    *,
    session_id: str,
    participant_id: str,
    session_number: int,
    curriculum: CurriculumMode,
    phase: str,
    curriculum_phase: str,
    trial_number: int,
    trial: Trial,
    response: int | None,
    response_type: str,
    is_correct: bool,
    response_time_ms: int | None,
    score: int,
) -> dict[str, Any]:
    """Build the shared desktop/web representation of one completed trial."""

    if response_type not in {"mouse", "arrow", "none", "unknown"}:
        raise ValueError(f"Unknown response type {response_type!r}")
    choice = "" if response is None else _symbol_name(trial.options[response])
    response_position = (
        "timeout" if response is None else ("top" if response == 0 else "bottom")
    )
    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "curriculum": curriculum.value,
        "phase": phase,
        "trial_number": trial_number,
        "operation": trial.operation.value,
        "start_shape": trial.start.shape,
        "start_size": _size_name(trial.start),
        "correct_shape": trial.correct.shape,
        "correct_size": _size_name(trial.correct),
        "top_shape": trial.options[0].shape,
        "top_size": _size_name(trial.options[0]),
        "bottom_shape": trial.options[1].shape,
        "bottom_size": _size_name(trial.options[1]),
        "response": response_position,
        "is_correct": bool(is_correct),
        "response_time_ms": "" if response_time_ms is None else response_time_ms,
        "score": score,
        "timestamp_utc": _utc_now(),
        "session_number": session_number,
        "start_symbol": _symbol_name(trial.start),
        "option_symbols": json.dumps(
            [_symbol_name(option) for option in trial.options], separators=(",", ":")
        ),
        "choice": choice,
        "timeout": response is None,
        "curriculum_phase": curriculum_phase,
        "response_type": response_type,
    }


def _append_csv(path: Path, fieldnames: Iterable[str], row: Mapping[str, Any]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _compatible_csv_path(path: Path, fieldnames: Iterable[str]) -> Path:
    """Avoid mixing records when an existing CSV uses an older schema."""

    expected_header = tuple(fieldnames)
    candidate = path
    version = 1
    while candidate.exists() and candidate.stat().st_size > 0:
        try:
            with candidate.open("r", encoding="utf-8-sig", newline="") as source:
                existing_header = tuple(next(csv.reader(source)))
        except (OSError, StopIteration):
            existing_header = ()
        if existing_header == expected_header:
            return candidate
        version += 1
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
    return candidate


def _next_session_number(data_dir: Path, participant_id: str) -> int:
    """Find the next per-participant session number across old and new CSVs."""

    known_numbers: set[int] = set()
    legacy_session_ids: set[str] = set()
    for path in data_dir.glob("curriculum_*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                for row_number, row in enumerate(csv.DictReader(source), start=1):
                    if row.get("participant_id") != participant_id:
                        continue
                    raw_number = row.get("session_number", "").strip()
                    if raw_number:
                        try:
                            known_numbers.add(int(raw_number))
                        except ValueError:
                            pass
                    else:
                        legacy_id = row.get("session_id") or f"{path.name}:{row_number}"
                        legacy_session_ids.add(legacy_id)
        except (OSError, UnicodeError, csv.Error):
            continue
    highest_known = max(known_numbers, default=0)
    return max(highest_known, len(legacy_session_ids)) + 1


def _read_scores(path: Path) -> list[int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            return [int(row["score"]) for row in csv.DictReader(source)]
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return []


def _accuracy(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return ""
    return f"{sum(bool(row['is_correct']) for row in records) / len(records):.4f}"


def _symbol_name(state: SymbolState) -> str:
    size = "large" if state.is_large else "small"
    return f"{size}-{state.shape}"


def _size_name(state: SymbolState) -> str:
    return "large" if state.is_large else "small"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Screen(str, Enum):
    SUBJECT_ID = "subject_id"
    INSTRUCTIONS = "instructions"
    PLAYING = "playing"
    COMPLETE = "complete"


class TrialPhase(str, Enum):
    START = "start"
    OPERATION = "operation"
    RESPONSE = "response"
    FEEDBACK = "feedback"


class CurriculumGame:
    """Pygame user interface around the testable curriculum and trial logic."""

    WIDTH = 1100
    HEIGHT = 700
    FPS = 60
    INCORRECT_SHAKE_MS = 250

    BACKGROUND = (0, 0, 0)
    TEXT = (245, 247, 250)
    MUTED = (198, 207, 220)
    START_BLUE = (30, 72, 145)
    OPTION_GREEN = (39, 190, 112)
    SIZE_CYAN = (35, 230, 250)
    SHAPE_RED = (241, 78, 85)
    SUCCESS = (46, 207, 119)
    ERROR = (239, 68, 76)
    PANEL = (18, 22, 29)
    BORDER = (76, 88, 105)
    SYMBOL_BACKGROUND = (255, 255, 255)
    SYMBOL_BACKGROUND_RADIUS = LARGE_SYMBOL_RADIUS

    START_POS = (210, 350)
    OPERATION_POS = (540, 350)
    OPTION_POSITIONS = ((860, 245), (860, 465))

    def __init__(
        self,
        curriculum: Mapping[str, Any] | CurriculumConfig | None = None,
        *,
        seed: int | None = None,
        data_dir: Path | None = None,
        recorder_factory: Callable[[str, CurriculumMode, Path | None], Any] | None = None,
    ) -> None:
        self.config = (
            curriculum
            if isinstance(curriculum, CurriculumConfig)
            else CurriculumConfig.from_dict(curriculum)
        )
        self.rng = random.Random(seed)
        self.data_dir = data_dir
        self.recorder_factory = recorder_factory
        self.screen = Screen.SUBJECT_ID
        self.subject_id = ""
        self.input_error = ""
        self.scheduler = CurriculumScheduler(self.config, self.rng)
        self.recorder: Any | None = None
        self.trial: Trial | None = None
        self.trial_phase = TrialPhase.START
        self.phase_started_ms = 0
        self.is_testing = False
        self.test_trial_count = 0
        self.score = 0
        self.selected_option: int | None = None
        self.last_response_correct = False
        self.timed_out = False
        self.ranking: int | None = None
        self._summary_saved = False

    def run(self) -> None:
        """Run the desktop version while retaining an async-capable core."""

        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise SystemExit(
                "pygame is required to run the game. Install it with: pip install pygame"
            ) from exc

        pygame.init()
        pygame.display.set_caption("Curriculum Learning Game")
        surface = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        clock = pygame.time.Clock()
        fonts = {
            "title": pygame.font.Font(None, 58),
            "heading": pygame.font.Font(None, 40),
            "body": pygame.font.Font(None, 30),
            "small": pygame.font.Font(None, 23),
            "feedback": pygame.font.Font(None, 78),
        }
        running = True
        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    self._handle_event(event, now, pygame)

            self._update(now)
            await self._pump_recorder()
            self._draw(surface, fonts, now, pygame)
            pygame.display.flip()
            clock.tick(self.FPS)
            await asyncio.sleep(0)
        pygame.quit()

    async def _pump_recorder(self) -> None:
        if self.recorder is None:
            return
        pump = getattr(self.recorder, "pump", None)
        if pump is not None:
            await pump()
        remote_ranking = getattr(self.recorder, "ranking", None)
        if remote_ranking is not None:
            self.ranking = remote_ranking

    def _handle_event(self, event: Any, now: int, pygame: Any) -> None:
        if self.screen is Screen.SUBJECT_ID and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self._accept_subject_id()
            elif event.key == pygame.K_BACKSPACE:
                self.subject_id = self.subject_id[:-1]
                self.input_error = ""
            elif event.unicode and event.unicode.isprintable() and len(self.subject_id) < 40:
                self.subject_id += event.unicode
                self.input_error = ""
            return

        if self.screen is Screen.INSTRUCTIONS and event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._begin_game(now)
            return

        if self.screen is Screen.COMPLETE and event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                if self.recorder is None or getattr(self.recorder, "is_synced", True):
                    self._reset()
            return

        if self.screen is not Screen.PLAYING or self.trial_phase is not TrialPhase.RESPONSE:
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self._submit_response(0, now, response_type="arrow")
            elif event.key == pygame.K_DOWN:
                self._submit_response(1, now, response_type="arrow")
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for option, centre in enumerate(self.OPTION_POSITIONS):
                if _point_in_circle(event.pos, centre, 86):
                    self._submit_response(option, now, response_type="mouse")
                    break

    def _accept_subject_id(self) -> None:
        cleaned = self.subject_id.strip()
        if not cleaned:
            self.input_error = "Please enter a subject ID."
            return
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", cleaned):
            self.input_error = "Use letters, numbers, hyphens, underscores, or full stops only."
            return
        self.subject_id = cleaned
        if self.recorder_factory is None:
            self.recorder = SessionRecorder(cleaned, self.config.mode, self.data_dir)
        else:
            self.recorder = self.recorder_factory(cleaned, self.config.mode, self.data_dir)
        self.screen = Screen.INSTRUCTIONS

    def _begin_game(self, now: int) -> None:
        self.screen = Screen.PLAYING
        self._start_trial(now)

    def _start_trial(self, now: int) -> None:
        if self.is_testing:
            operation = self.rng.choice(tuple(Operation))
            shapes = NOVEL_SHAPES
        else:
            operation = self.scheduler.next_operation()
            shapes = TRAINING_SHAPES
        self.trial = make_trial(operation, shapes, self.rng)
        self.trial_phase = TrialPhase.START
        self.phase_started_ms = now
        self.selected_option = None
        self.last_response_correct = False
        self.timed_out = False

    def _update(self, now: int) -> None:
        if self.screen is not Screen.PLAYING:
            return
        elapsed = now - self.phase_started_ms
        if self.trial_phase is TrialPhase.START and elapsed >= self.config.start_ms:
            self._set_phase(TrialPhase.OPERATION, now)
        elif self.trial_phase is TrialPhase.OPERATION and elapsed >= self.config.operation_ms:
            self._set_phase(TrialPhase.RESPONSE, now)
        elif self.trial_phase is TrialPhase.RESPONSE and elapsed >= self.config.response_ms:
            self._submit_response(None, now, response_type="none")
        elif self.trial_phase is TrialPhase.FEEDBACK:
            shake_time = (
                self.INCORRECT_SHAKE_MS
                if not self.last_response_correct and not self.timed_out
                else 0
            )
            if elapsed >= self.config.feedback_ms + shake_time:
                self._advance_after_feedback(now)

    def _set_phase(self, phase: TrialPhase, now: int) -> None:
        self.trial_phase = phase
        self.phase_started_ms = now

    def _submit_response(
        self,
        option: int | None,
        now: int,
        *,
        response_type: str = "unknown",
    ) -> None:
        if self.trial_phase is not TrialPhase.RESPONSE or self.trial is None:
            return
        response_time = None if option is None else now - self.phase_started_ms
        is_correct = option == self.trial.correct_option if option is not None else False
        self.selected_option = option
        self.last_response_correct = is_correct
        self.timed_out = option is None
        if is_correct:
            self.score += 1
        phase = "test" if self.is_testing else "training"
        curriculum_phase = "test" if self.is_testing else self.scheduler.current_phase
        trial_number = self.scheduler.trial_count + self.test_trial_count + 1
        if self.is_testing:
            self.test_trial_count += 1
        else:
            self.scheduler.record(self.trial.operation, is_correct)
        if self.recorder is not None:
            self.recorder.record_trial(
                phase=phase,
                curriculum_phase=curriculum_phase,
                trial_number=trial_number,
                trial=self.trial,
                response=option,
                response_type=response_type,
                is_correct=is_correct,
                response_time_ms=response_time,
                score=self.score,
            )
        self._set_phase(TrialPhase.FEEDBACK, now)

    def _advance_after_feedback(self, now: int) -> None:
        if self.is_testing and self.test_trial_count >= self.config.test_trials:
            self._complete_game()
            return
        if not self.is_testing and self.scheduler.should_end_training:
            self.is_testing = True
            self.test_trial_count = 0
        self._start_trial(now)

    def _complete_game(self) -> None:
        if not self._summary_saved and self.recorder is not None:
            ranking = self.recorder.finish(self.score)
            if ranking is not None:
                self.ranking = ranking
            self._summary_saved = True
        self.screen = Screen.COMPLETE

    def _reset(self) -> None:
        self.screen = Screen.SUBJECT_ID
        self.subject_id = ""
        self.input_error = ""
        self.scheduler = CurriculumScheduler(self.config, self.rng)
        self.recorder = None
        self.trial = None
        self.is_testing = False
        self.test_trial_count = 0
        self.score = 0
        self.ranking = None
        self.timed_out = False
        self._summary_saved = False

    def _draw(self, surface: Any, fonts: Mapping[str, Any], now: int, pygame: Any) -> None:
        surface.fill(self.BACKGROUND)
        if self.screen is Screen.SUBJECT_ID:
            self._draw_subject_screen(surface, fonts, pygame)
        elif self.screen is Screen.INSTRUCTIONS:
            self._draw_instruction_screen(surface, fonts, pygame)
        elif self.screen is Screen.PLAYING:
            self._draw_trial_screen(surface, fonts, now, pygame)
        else:
            self._draw_complete_screen(surface, fonts, pygame)

    def _draw_subject_screen(self, surface: Any, fonts: Mapping[str, Any], pygame: Any) -> None:
        _centred_text(surface, fonts["title"], "Curriculum Learning", self.TEXT, 170)
        _centred_text(surface, fonts["body"], "Enter your subject ID", self.TEXT, 275)
        box = pygame.Rect(320, 320, 460, 64)
        pygame.draw.rect(surface, self.PANEL, box, border_radius=10)
        pygame.draw.rect(surface, self.START_BLUE, box, width=2, border_radius=10)
        shown = self.subject_id + ("|" if pygame.time.get_ticks() % 1000 < 500 else "")
        _centred_text(surface, fonts["body"], shown, self.TEXT, 352)
        _centred_text(surface, fonts["small"], "Press Enter to continue", self.MUTED, 425)
        if self.input_error:
            _centred_text(surface, fonts["small"], self.input_error, self.ERROR, 470)

    def _draw_instruction_screen(self, surface: Any, fonts: Mapping[str, Any], pygame: Any) -> None:
        panel = pygame.Rect(170, 115, 760, 470)
        pygame.draw.rect(surface, self.PANEL, panel, border_radius=20)
        pygame.draw.rect(surface, self.BORDER, panel, width=2, border_radius=20)
        _centred_text(surface, fonts["title"], "Instructions", self.TEXT, 185)
        lines = (
            "Your goal is to earn as many points as possible.",
            "Select the correct option on each trial.",
            "You will need to work out the rules by trial and error.",
            "Use the up/down arrow keys or click an option.",
        )
        for index, line in enumerate(lines):
            _centred_text(surface, fonts["body"], line, self.TEXT, 285 + index * 48)
        _centred_text(surface, fonts["small"], "Press Enter or Space to start", self.MUTED, 535)

    def _draw_trial_screen(
        self,
        surface: Any,
        fonts: Mapping[str, Any],
        now: int,
        pygame: Any,
    ) -> None:
        if self.trial is None:
            return
        pygame.draw.line(surface, self.BORDER, (350, 165), (350, 565), width=2)
        pygame.draw.line(surface, self.BORDER, (700, 165), (700, 565), width=2)
        _draw_symbol(
            surface,
            self.trial.start,
            self.START_POS,
            self.START_BLUE,
            pygame,
            background_colour=self.SYMBOL_BACKGROUND,
            background_radius=self.SYMBOL_BACKGROUND_RADIUS,
        )

        if self.trial_phase in (TrialPhase.OPERATION, TrialPhase.RESPONSE, TrialPhase.FEEDBACK):
            operation_colour = (
                self.SIZE_CYAN if self.trial.operation is Operation.SIZE else self.SHAPE_RED
            )
            pygame.draw.circle(surface, operation_colour, self.OPERATION_POS, 49)
            pygame.draw.circle(surface, self.PANEL, self.OPERATION_POS, 49, width=4)

        if self.trial_phase in (TrialPhase.RESPONSE, TrialPhase.FEEDBACK):
            for index, (state, centre) in enumerate(zip(self.trial.options, self.OPTION_POSITIONS)):
                draw_centre = centre
                if (
                    self.trial_phase is TrialPhase.FEEDBACK
                    and not self.last_response_correct
                    and self.selected_option == index
                    and now - self.phase_started_ms < self.INCORRECT_SHAKE_MS
                ):
                    draw_centre = (centre[0] + round(math.sin((now - self.phase_started_ms) / 20) * 9), centre[1])
                _draw_symbol(
                    surface,
                    state,
                    draw_centre,
                    self.OPTION_GREEN,
                    pygame,
                    background_colour=self.SYMBOL_BACKGROUND,
                    background_radius=self.SYMBOL_BACKGROUND_RADIUS,
                )
            _draw_arrow_hint(
                surface,
                (970, self.OPTION_POSITIONS[0][1]),
                "up",
                self.MUTED,
                pygame,
            )
            _draw_arrow_hint(
                surface,
                (970, self.OPTION_POSITIONS[1][1]),
                "down",
                self.MUTED,
                pygame,
            )

        _text(surface, fonts["small"], f"Score  {self.score}", self.TEXT, (40, 35))
        phase_name = "Test" if self.is_testing else "Training"
        _text(surface, fonts["small"], phase_name, self.MUTED, (980, 35), anchor="topright")

        show_feedback = (
            self.last_response_correct
            or self.timed_out
            or now - self.phase_started_ms >= self.INCORRECT_SHAKE_MS
        )
        if self.trial_phase is TrialPhase.FEEDBACK and show_feedback:
            feedback = "+1" if self.last_response_correct else "0"
            colour = self.SUCCESS if self.last_response_correct else self.ERROR
            if self.timed_out:
                _centred_text(
                    surface,
                    fonts["body"],
                    "The trial timed out",
                    self.ERROR,
                    575,
                )
            _centred_text(surface, fonts["feedback"], feedback, colour, 625)

    def _draw_complete_screen(self, surface: Any, fonts: Mapping[str, Any], pygame: Any) -> None:
        panel = pygame.Rect(210, 105, 680, 490)
        pygame.draw.rect(surface, self.PANEL, panel, border_radius=20)
        pygame.draw.rect(surface, self.BORDER, panel, width=2, border_radius=20)
        _centred_text(surface, fonts["title"], "Game complete", self.TEXT, 185)
        _centred_text(surface, fonts["body"], "Thank you for taking part.", self.TEXT, 265)
        _centred_text(surface, fonts["heading"], f"Your score: {self.score}", self.SUCCESS, 350)
        is_remote = self.recorder is not None and hasattr(self.recorder, "is_synced")
        sync_error = getattr(self.recorder, "sync_error", "") if self.recorder else ""
        if self.ranking is not None:
            rank_text = f"Pilot ranking: #{self.ranking}"
        elif is_remote and sync_error:
            rank_text = "Saving interrupted — please keep this page open"
        elif is_remote and not self.recorder.is_synced:
            rank_text = "Saving results..."
        else:
            rank_text = "Ranking unavailable"
        _centred_text(surface, fonts["body"], rank_text, self.TEXT, 420)
        completion_hint = (
            "Press Enter to return to the start"
            if self.recorder is None or getattr(self.recorder, "is_synced", True)
            else "Please keep this page open while the results are saved"
        )
        _centred_text(surface, fonts["small"], completion_hint, self.MUTED, 535)


def _draw_symbol(
    surface: Any,
    state: SymbolState,
    centre: tuple[int, int],
    colour: tuple[int, int, int],
    pygame: Any,
    *,
    background_colour: tuple[int, int, int] | None = None,
    background_radius: int = 72,
) -> None:
    if background_colour is not None:
        pygame.draw.circle(surface, background_colour, centre, background_radius)
    radius = LARGE_SYMBOL_RADIUS if state.is_large else SMALL_SYMBOL_RADIUS
    x, y = centre
    if state.shape == "circle":
        pygame.draw.circle(surface, colour, centre, radius)
        return
    if state.shape == "square":
        rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        pygame.draw.rect(surface, colour, rect, border_radius=max(4, radius // 8))
        return
    sides = {"triangle": 3, "pentagon": 5, "hexagon": 6}.get(state.shape)
    if sides is not None:
        points = _regular_polygon(centre, radius, sides, rotation=-math.pi / 2)
        pygame.draw.polygon(surface, colour, points)
        return
    if state.shape == "star":
        points = []
        for point in range(10):
            angle = -math.pi / 2 + point * math.pi / 5
            point_radius = radius if point % 2 == 0 else radius * 0.43
            points.append((x + math.cos(angle) * point_radius, y + math.sin(angle) * point_radius))
        pygame.draw.polygon(surface, colour, points)
        return
    raise ValueError(f"Cannot draw unknown shape {state.shape!r}")


def _draw_arrow_hint(
    surface: Any,
    centre: tuple[int, int],
    direction: str,
    colour: tuple[int, int, int],
    pygame: Any,
) -> None:
    """Draw a font-independent up/down key hint beside an option."""

    x, y = centre
    if direction == "up":
        points = (
            (x, y - 18),
            (x - 14, y - 3),
            (x - 6, y - 3),
            (x - 6, y + 17),
            (x + 6, y + 17),
            (x + 6, y - 3),
            (x + 14, y - 3),
        )
    elif direction == "down":
        points = (
            (x, y + 18),
            (x - 14, y + 3),
            (x - 6, y + 3),
            (x - 6, y - 17),
            (x + 6, y - 17),
            (x + 6, y + 3),
            (x + 14, y + 3),
        )
    else:
        raise ValueError(f"Unknown arrow direction {direction!r}")
    pygame.draw.polygon(surface, colour, points)


def _regular_polygon(
    centre: tuple[int, int],
    radius: float,
    sides: int,
    *,
    rotation: float = 0,
) -> list[tuple[float, float]]:
    return [
        (
            centre[0] + math.cos(rotation + 2 * math.pi * point / sides) * radius,
            centre[1] + math.sin(rotation + 2 * math.pi * point / sides) * radius,
        )
        for point in range(sides)
    ]


def _point_in_circle(point: tuple[int, int], centre: tuple[int, int], radius: int) -> bool:
    return (point[0] - centre[0]) ** 2 + (point[1] - centre[1]) ** 2 <= radius**2


def _text(
    surface: Any,
    font: Any,
    message: str,
    colour: tuple[int, int, int],
    position: tuple[int, int],
    *,
    anchor: str = "topleft",
) -> None:
    rendered = font.render(message, True, colour)
    rect = rendered.get_rect()
    setattr(rect, anchor, position)
    surface.blit(rendered, rect)


def _centred_text(
    surface: Any,
    font: Any,
    message: str,
    colour: tuple[int, int, int],
    y: int,
) -> None:
    _text(surface, font, message, colour, (surface.get_width() // 2, y), anchor="center")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the curriculum learning pygame.")
    parser.add_argument(
        "--curriculum",
        default=CurriculumMode.INTERLEAVED.value,
        help="interleaved, blocked, progressively_interleaved, or progressively_blocked",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    return parser


def main(curriculum: Mapping[str, Any] | None = None) -> None:
    """Run from Python with a curriculum dictionary, or parse command-line flags."""

    if curriculum is None:
        arguments = build_argument_parser().parse_args()
        curriculum = {"mode": arguments.curriculum}
        seed = arguments.seed
    else:
        seed = None
    CurriculumGame(curriculum, seed=seed).run()


if __name__ == "__main__":
    main()
