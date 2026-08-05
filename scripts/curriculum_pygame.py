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
from dataclasses import dataclass, replace
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
NOVEL_SHAPES = ("triangle", "star", "pentagon", "plus")
SHAPE_PARTNERS = {
    "square": "circle",
    "circle": "square",
    "triangle": "star",
    "star": "triangle",
    "pentagon": "plus",
    "plus": "pentagon",
    # Retain support for trials created by versions that used an X or hexagon.
    "x": "pentagon",
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
    minimum_trials_per_operation: int = 15
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


@dataclass(frozen=True)
class PrivacyNotice:
    """Public controller details shown before the browser creates a session."""

    controller_name: str
    contact_email: str
    retention_period: str
    version: str

    def __post_init__(self) -> None:
        values = {
            "controller_name": self.controller_name,
            "contact_email": self.contact_email,
            "retention_period": self.retention_period,
            "version": self.version,
        }
        for field_name, value in values.items():
            if not value.strip() or value.startswith("__"):
                raise ValueError(f"{field_name} must be configured for the web build")
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", self.contact_email):
            raise ValueError("contact_email must be a valid email address")


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
        if self.config.mode in (
            CurriculumMode.BLOCKED,
            CurriculumMode.PROGRESSIVELY_INTERLEAVED,
        ):
            # The two online curricula use the same rolling mastery window.
            window_size = self.config.minimum_trials_per_operation
            recent_values = values[-window_size:]
            return (
                len(recent_values) == window_size
                and sum(recent_values) / window_size >= self.config.mastery_threshold
            )
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
            return (
                Operation.SHAPE
                if self.operation_mastered(Operation.SIZE)
                else Operation.SIZE
            )
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
            return (
                "block2"
                if self.operation_mastered(Operation.SIZE)
                else "block1"
            )
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
    def is_ready(self) -> bool:
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


def _display_alias(alias: str) -> str:
    return " ".join(part.capitalize() for part in alias.split("-"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Screen(str, Enum):
    PRIVACY = "privacy"
    SUBJECT_ID = "subject_id"
    INSTRUCTIONS = "instructions"
    PLAYING = "playing"
    COMPLETE = "complete"
    DECLINED = "declined"


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
    ALIAS_BUTTON_RECTS = (
        (250, 260, 600, 70),
        (250, 355, 600, 70),
        (250, 450, 600, 70),
    )
    INSTRUCTION_START_RECT = (430, 480, 240, 60)
    PRIVACY_AGE_RECT = (105, 514, 28, 28)
    PRIVACY_PARTICIPATION_RECT = (105, 558, 28, 28)
    PRIVACY_AGE_HIT_RECT = (95, 506, 620, 42)
    PRIVACY_PARTICIPATION_HIT_RECT = (95, 550, 650, 42)
    PRIVACY_LEAVE_RECT = (160, 620, 240, 56)
    PRIVACY_ACCEPT_RECT = (670, 620, 270, 56)

    def __init__(
        self,
        curriculum: Mapping[str, Any] | CurriculumConfig | None = None,
        *,
        seed: int | None = None,
        data_dir: Path | None = None,
        recorder_factory: Callable[[str, CurriculumMode, Path | None], Any] | None = None,
        anonymous_participant_id: str | None = None,
        new_anonymous_participant: Callable[[], str] | None = None,
        curriculum_for_participant: Callable[[str], CurriculumMode | str] | None = None,
        privacy_notice: PrivacyNotice | None = None,
        require_privacy_acceptance: bool = False,
    ) -> None:
        self.config = (
            curriculum
            if isinstance(curriculum, CurriculumConfig)
            else CurriculumConfig.from_dict(curriculum)
        )
        self.rng = random.Random(seed)
        self.data_dir = data_dir
        self.recorder_factory = recorder_factory
        self.anonymous_participant_id = anonymous_participant_id
        self.new_anonymous_participant = new_anonymous_participant
        self.curriculum_for_participant = curriculum_for_participant
        self.privacy_notice = privacy_notice
        self.require_privacy_acceptance = require_privacy_acceptance
        if self.require_privacy_acceptance and self.privacy_notice is None:
            raise ValueError("privacy_notice is required when acceptance is required")
        if anonymous_participant_id is not None and not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,40}", anonymous_participant_id
        ):
            raise ValueError("anonymous_participant_id has an invalid format")
        self.screen = (
            Screen.PRIVACY if self.require_privacy_acceptance else Screen.SUBJECT_ID
        )
        self.subject_id = anonymous_participant_id or ""
        self.leaderboard_name = ""
        self.input_error = ""
        self.privacy_age_confirmed = False
        self.privacy_participation_confirmed = False
        self.privacy_accepted = False
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
        if self.anonymous_participant_id is not None:
            if self.recorder_factory is None:
                raise ValueError("anonymous browser mode requires a recorder_factory")
            if not self.require_privacy_acceptance:
                self._create_anonymous_recorder()

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
            "hud": pygame.font.Font(None, 42),
            "notice": pygame.font.Font(None, 22),
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
        remote_name = getattr(self.recorder, "leaderboard_name", None)
        if remote_name and self.anonymous_participant_id is not None:
            self.leaderboard_name = remote_name
            ready = bool(getattr(self.recorder, "is_ready", False))
            if self.screen is Screen.SUBJECT_ID and ready:
                self.input_error = ""
                self.screen = Screen.INSTRUCTIONS

    def _handle_event(self, event: Any, now: int, pygame: Any) -> None:
        if self.screen is Screen.PRIVACY:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.privacy_age_confirmed = not self.privacy_age_confirmed
                    self.input_error = ""
                elif event.key == pygame.K_2:
                    self.privacy_participation_confirmed = (
                        not self.privacy_participation_confirmed
                    )
                    self.input_error = ""
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._accept_privacy_notice()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(*self.PRIVACY_AGE_HIT_RECT).collidepoint(event.pos):
                    self.privacy_age_confirmed = not self.privacy_age_confirmed
                    self.input_error = ""
                elif pygame.Rect(*self.PRIVACY_PARTICIPATION_HIT_RECT).collidepoint(
                    event.pos
                ):
                    self.privacy_participation_confirmed = (
                        not self.privacy_participation_confirmed
                    )
                    self.input_error = ""
                elif pygame.Rect(*self.PRIVACY_ACCEPT_RECT).collidepoint(event.pos):
                    self._accept_privacy_notice()
                elif pygame.Rect(*self.PRIVACY_LEAVE_RECT).collidepoint(event.pos):
                    self.input_error = ""
                    self.screen = Screen.DECLINED
            return

        if self.screen is Screen.DECLINED:
            return

        if self.screen is Screen.SUBJECT_ID:
            if self.anonymous_participant_id is not None:
                if event.type == pygame.KEYDOWN:
                    number_keys = {
                        pygame.K_1: 0,
                        pygame.K_2: 1,
                        pygame.K_3: 2,
                    }
                    if event.key in number_keys:
                        self._choose_alias(number_keys[event.key])
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for index, dimensions in enumerate(self.ALIAS_BUTTON_RECTS):
                        if pygame.Rect(*dimensions).collidepoint(event.pos):
                            self._choose_alias(index)
                            break
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self._accept_subject_id()
                elif event.key == pygame.K_BACKSPACE:
                    self.subject_id = self.subject_id[:-1]
                    self.input_error = ""
                elif event.unicode and event.unicode.isprintable() and len(self.subject_id) < 40:
                    self.subject_id += event.unicode
                    self.input_error = ""
            return

        if self.screen is Screen.INSTRUCTIONS:
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_RETURN,
                pygame.K_SPACE,
            ):
                self._begin_game(now)
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and pygame.Rect(*self.INSTRUCTION_START_RECT).collidepoint(event.pos)
            ):
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

    def _create_anonymous_recorder(self) -> None:
        if self.recorder is not None:
            return
        if self.anonymous_participant_id is None or self.recorder_factory is None:
            raise ValueError("anonymous browser mode is not configured")
        self.recorder = self.recorder_factory(
            self.anonymous_participant_id, self.config.mode, self.data_dir
        )

    def _accept_privacy_notice(self) -> None:
        if not (
            self.privacy_age_confirmed and self.privacy_participation_confirmed
        ):
            self.input_error = "Please confirm both statements before continuing."
            return
        self.privacy_accepted = True
        self.input_error = ""
        if self.anonymous_participant_id is not None:
            self._create_anonymous_recorder()
        self.screen = Screen.SUBJECT_ID

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

    def _choose_alias(self, index: int) -> None:
        if self.recorder is None:
            return
        options = getattr(self.recorder, "alias_options", [])
        if not 0 <= index < len(options):
            return
        try:
            self.recorder.choose_alias(options[index])
            self.input_error = ""
        except ValueError as exc:
            self.input_error = str(exc)

    def _begin_game(self, now: int) -> None:
        if self.recorder is not None and not getattr(self.recorder, "is_ready", True):
            self.input_error = "Please wait while your session is prepared."
            return
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
        if (
            self.anonymous_participant_id is not None
            and self.new_anonymous_participant is not None
        ):
            participant_id = self.new_anonymous_participant()
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", participant_id):
                raise ValueError("The new anonymous participant code has an invalid format")
            self.anonymous_participant_id = participant_id
            if self.curriculum_for_participant is not None:
                assigned_mode = self.curriculum_for_participant(participant_id)
                mode = (
                    assigned_mode
                    if isinstance(assigned_mode, CurriculumMode)
                    else _normalise_curriculum_mode(assigned_mode)
                )
                self.config = replace(self.config, mode=mode)
        self.screen = (
            Screen.PRIVACY if self.require_privacy_acceptance else Screen.SUBJECT_ID
        )
        self.subject_id = self.anonymous_participant_id or ""
        self.leaderboard_name = ""
        self.input_error = ""
        self.privacy_age_confirmed = False
        self.privacy_participation_confirmed = False
        self.privacy_accepted = False
        self.scheduler = CurriculumScheduler(self.config, self.rng)
        self.recorder = None
        self.trial = None
        self.is_testing = False
        self.test_trial_count = 0
        self.score = 0
        self.ranking = None
        self.timed_out = False
        self._summary_saved = False
        if (
            self.anonymous_participant_id is not None
            and self.recorder_factory is not None
            and not self.require_privacy_acceptance
        ):
            self._create_anonymous_recorder()

    def _draw(self, surface: Any, fonts: Mapping[str, Any], now: int, pygame: Any) -> None:
        surface.fill(self.BACKGROUND)
        if self.screen is Screen.PRIVACY:
            self._draw_privacy_screen(surface, fonts, pygame)
        elif self.screen is Screen.SUBJECT_ID:
            self._draw_subject_screen(surface, fonts, pygame)
        elif self.screen is Screen.INSTRUCTIONS:
            self._draw_instruction_screen(surface, fonts, pygame)
        elif self.screen is Screen.PLAYING:
            self._draw_trial_screen(surface, fonts, now, pygame)
        elif self.screen is Screen.COMPLETE:
            self._draw_complete_screen(surface, fonts, pygame)
        else:
            self._draw_declined_screen(surface, fonts, pygame)

    def _draw_privacy_screen(
        self, surface: Any, fonts: Mapping[str, Any], pygame: Any
    ) -> None:
        assert self.privacy_notice is not None
        notice = self.privacy_notice
        notice_font = fonts.get("notice", fonts["small"])
        _centred_text(
            surface,
            fonts["heading"],
            "Pilot information and privacy",
            self.TEXT,
            48,
        )
        panel = pygame.Rect(60, 78, 980, 418)
        pygame.draw.rect(surface, self.PANEL, panel, border_radius=18)
        pygame.draw.rect(surface, self.BORDER, panel, width=2, border_radius=18)

        left_x, right_x = 92, 566
        _text(surface, fonts["body"], "About this pilot", self.TEXT, (left_x, 108))
        _draw_wrapped_text(
            surface,
            notice_font,
            (
                "This pilot evaluates learning across curricula."
            ),
            self.MUTED,
            (left_x, 146),
            420,
            24,
        )
        _text(surface, fonts["body"], "What is recorded", self.TEXT, (left_x, 210))
        _draw_wrapped_text(
            surface,
            notice_font,
            (
                "Only task data (choices, response times, scores and trial details), "
                "a random task ID (e.g., a3f2cc9d1...) and game alias (e.g., "
                "fantastic-otter) are saved."
            ),
            self.MUTED,
            (left_x, 248),
            420,
            24,
        )
        _text(surface, fonts["body"], "How the data is used", self.TEXT, (left_x, 374))
        _draw_wrapped_text(
            surface,
            notice_font,
            (
                "To estimate learning rates across curricula and improve task design."
            ),
            self.MUTED,
            (left_x, 412),
            420,
            24,
        )

        _text(
            surface,
            fonts["body"],
            "Page and database services",
            self.TEXT,
            (right_x, 108),
        )
        _draw_wrapped_text(
            surface,
            notice_font,
            (
                "GitHub Pages hosts the task and Supabase stores the task data. Both "
                "providers use your IP address to operate their services and may keep "
                "a short-lived record of it as part of their normal service."
            ),
            self.MUTED,
            (right_x, 146),
            438,
            24,
        )
        _text(surface, fonts["body"], "Opt out at any time", self.TEXT, (right_x, 292))
        _draw_wrapped_text(
            surface,
            notice_font,
            (
                "Taking part is voluntary and you may stop at any time. Saved trial "
                f"data remains {notice.retention_period} unless you ask for it to be "
                f"deleted. To ask for data deletion, email me at {notice.contact_email} "
                "and include your game alias."
            ),
            self.MUTED,
            (right_x, 330),
            438,
            24,
        )

        self._draw_checkbox(
            surface,
            fonts,
            pygame,
            self.PRIVACY_AGE_RECT,
            self.privacy_age_confirmed,
            "1. I confirm that I am aged 20 or over.",
        )
        self._draw_checkbox(
            surface,
            fonts,
            pygame,
            self.PRIVACY_PARTICIPATION_RECT,
            self.privacy_participation_confirmed,
            "2. I have read this notice and voluntarily agree to take part.",
        )

        leave_button = pygame.Rect(*self.PRIVACY_LEAVE_RECT)
        pygame.draw.rect(surface, self.PANEL, leave_button, border_radius=12)
        pygame.draw.rect(surface, self.BORDER, leave_button, width=2, border_radius=12)
        _text(
            surface,
            fonts["body"],
            "Leave pilot",
            self.TEXT,
            leave_button.center,
            anchor="center",
        )

        accepted = self.privacy_age_confirmed and self.privacy_participation_confirmed
        accept_button = pygame.Rect(*self.PRIVACY_ACCEPT_RECT)
        accept_colour = self.START_BLUE if accepted else self.BORDER
        pygame.draw.rect(surface, accept_colour, accept_button, border_radius=12)
        _text(
            surface,
            fonts["body"],
            "Accept and continue",
            self.TEXT if accepted else self.MUTED,
            accept_button.center,
            anchor="center",
        )
        if self.input_error:
            _centred_text(surface, fonts["small"], self.input_error, self.ERROR, 604)
        _text(
            surface,
            fonts["small"],
            f"Notice version {notice.version}",
            self.MUTED,
            (1025, 68),
            anchor="bottomright",
        )

    def _draw_checkbox(
        self,
        surface: Any,
        fonts: Mapping[str, Any],
        pygame: Any,
        dimensions: tuple[int, int, int, int],
        checked: bool,
        label: str,
    ) -> None:
        box = pygame.Rect(*dimensions)
        pygame.draw.rect(surface, self.PANEL, box, border_radius=4)
        pygame.draw.rect(
            surface,
            self.START_BLUE if checked else self.BORDER,
            box,
            width=3,
            border_radius=4,
        )
        if checked:
            pygame.draw.line(
                surface,
                self.TEXT,
                (box.left + 6, box.centery),
                (box.left + 12, box.bottom - 7),
                width=3,
            )
            pygame.draw.line(
                surface,
                self.TEXT,
                (box.left + 12, box.bottom - 7),
                (box.right - 5, box.top + 6),
                width=3,
            )
        _text(surface, fonts["small"], label, self.TEXT, (box.right + 12, box.centery), anchor="midleft")

    def _draw_declined_screen(
        self, surface: Any, fonts: Mapping[str, Any], pygame: Any
    ) -> None:
        panel = pygame.Rect(220, 190, 660, 300)
        pygame.draw.rect(surface, self.PANEL, panel, border_radius=20)
        pygame.draw.rect(surface, self.BORDER, panel, width=2, border_radius=20)
        _centred_text(surface, fonts["title"], "You have not joined the pilot", self.TEXT, 270)
        _centred_text(
            surface,
            fonts["body"],
            "No task session was created and no task responses were sent.",
            self.MUTED,
            360,
        )
        _centred_text(surface, fonts["small"], "You may now close this page.", self.MUTED, 420)

    def _draw_subject_screen(self, surface: Any, fonts: Mapping[str, Any], pygame: Any) -> None:
        if self.anonymous_participant_id is not None:
            self._draw_alias_screen(surface, fonts, pygame)
            return
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

    def _draw_alias_screen(self, surface: Any, fonts: Mapping[str, Any], pygame: Any) -> None:
        _centred_text(surface, fonts["title"], "Choose your leaderboard name", self.TEXT, 130)
        options = getattr(self.recorder, "alias_options", []) if self.recorder else []
        if options:
            _centred_text(
                surface,
                fonts["small"],
                "Click a name or press 1, 2, or 3",
                self.MUTED,
                205,
            )
            for index, (alias, dimensions) in enumerate(
                zip(options, self.ALIAS_BUTTON_RECTS)
            ):
                button = pygame.Rect(*dimensions)
                pygame.draw.rect(surface, self.PANEL, button, border_radius=14)
                pygame.draw.rect(surface, self.START_BLUE, button, width=3, border_radius=14)
                _centred_text(
                    surface,
                    fonts["heading"],
                    f"{index + 1}. {_display_alias(alias)}",
                    self.TEXT,
                    button.centery,
                )
        else:
            _centred_text(
                surface,
                fonts["body"],
                "Creating your anonymous session...",
                self.MUTED,
                350,
            )

        status = getattr(self.recorder, "alias_status", "") if self.recorder else ""
        sync_error = getattr(self.recorder, "sync_error", "") if self.recorder else ""
        message = self.input_error or status
        colour = self.ERROR if self.input_error else self.MUTED
        if not message and sync_error:
            message = "Unable to connect. Retrying automatically..."
            colour = self.ERROR
        if message:
            _centred_text(surface, fonts["small"], message, colour, 575)

    def _draw_instruction_screen(self, surface: Any, fonts: Mapping[str, Any], pygame: Any) -> None:
        panel = pygame.Rect(170, 115, 760, 470)
        pygame.draw.rect(surface, self.PANEL, panel, border_radius=20)
        pygame.draw.rect(surface, self.BORDER, panel, width=2, border_radius=20)
        _centred_text(surface, fonts["title"], "Instructions", self.TEXT, 185)
        lines = (
            "Your goal is to earn as many points as possible.",
            "The two green symbols are your options on each trial.",
            "Select the option you think is correct.",
            "You will need to work out the rules by trial and error.",
            "Use the up/down arrow keys or click an option.",
        )
        for index, line in enumerate(lines):
            _centred_text(surface, fonts["body"], line, self.TEXT, 255 + index * 42)

        start_button = pygame.Rect(*self.INSTRUCTION_START_RECT)
        pygame.draw.rect(surface, self.START_BLUE, start_button, border_radius=12)
        pygame.draw.rect(surface, self.TEXT, start_button, width=2, border_radius=12)
        _text(
            surface,
            fonts["heading"],
            "Start",
            self.TEXT,
            start_button.center,
            anchor="center",
        )
        _centred_text(
            surface,
            fonts["small"],
            "Or press Enter or Space",
            self.MUTED,
            560,
        )

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

        _text(surface, fonts["hud"], f"Score  {self.score}", self.TEXT, (40, 35))
        participant_name = (
            _display_alias(self.leaderboard_name)
            if self.leaderboard_name
            else self.subject_id
        )
        if participant_name:
            _text(
                surface,
                fonts["hud"],
                participant_name,
                self.TEXT,
                (self.WIDTH // 2, 35),
                anchor="midtop",
            )
        phase_name = "Test" if self.is_testing else "Training"
        _text(surface, fonts["hud"], phase_name, self.MUTED, (980, 35), anchor="topright")

        show_feedback = (
            self.last_response_correct
            or self.timed_out
            or now - self.phase_started_ms >= self.INCORRECT_SHAKE_MS
        )
        if self.trial_phase is TrialPhase.FEEDBACK and show_feedback:
            feedback = "+1" if self.last_response_correct else "0"
            colour = self.SUCCESS if self.last_response_correct else self.ERROR
            _text(
                surface,
                fonts["feedback"],
                feedback,
                colour,
                (self.OPERATION_POS[0], 140),
                anchor="center",
            )
            if self.timed_out:
                _text(
                    surface,
                    fonts["body"],
                    "The trial timed out",
                    self.ERROR,
                    (self.OPERATION_POS[0], 210),
                    anchor="center",
                )

    def _draw_complete_screen(self, surface: Any, fonts: Mapping[str, Any], pygame: Any) -> None:
        panel = pygame.Rect(210, 105, 680, 490)
        pygame.draw.rect(surface, self.PANEL, panel, border_radius=20)
        pygame.draw.rect(surface, self.BORDER, panel, width=2, border_radius=20)
        _centred_text(surface, fonts["title"], "Game complete", self.TEXT, 185)
        _centred_text(surface, fonts["body"], "Thank you for taking part.", self.TEXT, 265)
        if self.leaderboard_name:
            _centred_text(
                surface,
                fonts["body"],
                f"Leaderboard name: {_display_alias(self.leaderboard_name)}",
                self.TEXT,
                310,
            )
        _centred_text(surface, fonts["heading"], f"Your score: {self.score}", self.SUCCESS, 365)
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
        _centred_text(surface, fonts["body"], rank_text, self.TEXT, 435)
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
    if state.shape == "plus":
        offset = round(radius * 0.72)
        width = max(12, round(radius * 0.30))
        pygame.draw.line(
            surface,
            colour,
            (x - offset, y),
            (x + offset, y),
            width=width,
        )
        pygame.draw.line(
            surface,
            colour,
            (x, y - offset),
            (x, y + offset),
            width=width,
        )
        return
    if state.shape == "x":  # Legacy support for an older loaded page.
        offset = round(radius * 0.58)
        width = max(12, round(radius * 0.30))
        pygame.draw.line(
            surface,
            colour,
            (x - offset, y - offset),
            (x + offset, y + offset),
            width=width,
        )
        pygame.draw.line(
            surface,
            colour,
            (x + offset, y - offset),
            (x - offset, y + offset),
            width=width,
        )
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


def _draw_wrapped_text(
    surface: Any,
    font: Any,
    message: str,
    colour: tuple[int, int, int],
    position: tuple[int, int],
    max_width: int,
    line_height: int,
) -> int:
    """Draw word-wrapped text and return the y coordinate after its last line."""

    lines: list[str] = []
    current = ""
    for word in message.split():
        candidate = word if not current else f"{current} {word}"
        if current and font.size(candidate)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    x, y = position
    for line in lines:
        _text(surface, font, line, colour, (x, y))
        y += line_height
    return y


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
