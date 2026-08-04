"""Asynchronous browser recorder used by the Pygbag build."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4

from curriculum_pygame import (
    CurriculumMode,
    SessionRecorder,
    Trial,
    build_trial_record,
)


PostJson = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]


class PygbagJsonTransport:
    """POST JSON through the browser Fetch API without third-party Python wheels."""

    _FETCH_BRIDGE = r"""
window.CurriculumFetch = window.CurriculumFetch || {};
window.CurriculumFetch.POST = function* (url, data) {
    let result;
    const request = new Request(url, {
        method: "POST",
        headers: {"Accept": "application/json", "Content-Type": "application/json"},
        body: data
    });
    fetch(request)
        .then(async response => {
            result = JSON.stringify({
                ok: response.ok,
                status: response.status,
                body: await response.text()
            });
        })
        .catch(error => {
            result = JSON.stringify({ok: false, status: 0, body: String(error)});
        });
    while (result === undefined) yield;
    yield result;
};
"""

    def __init__(self, endpoint: str) -> None:
        if not endpoint.startswith("https://"):
            raise ValueError("The Supabase function URL must start with https://")
        if sys.platform != "emscripten":
            raise RuntimeError("PygbagJsonTransport is only available in the browser build")
        import platform

        self.platform = platform
        self.endpoint = endpoint
        platform.window.eval(self._FETCH_BRIDGE)

    async def post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, separators=(",", ":"))
        raw_envelope = await self.platform.jsiter(
            self.platform.window.CurriculumFetch.POST(self.endpoint, encoded)
        )
        envelope = json.loads(str(raw_envelope))
        if not envelope.get("ok"):
            raise RuntimeError(
                f"Data service returned HTTP {envelope.get('status')}: {envelope.get('body')}"
            )
        body = json.loads(envelope.get("body") or "{}")
        if not body.get("ok"):
            raise RuntimeError(body.get("error", "Data service rejected the request"))
        return body


class RemoteSessionRecorder:
    """Queue trial rows and synchronize them without blocking the Pygame loop."""

    TRIAL_FIELDS = SessionRecorder.TRIAL_FIELDS
    MAX_BATCH_SIZE = 20
    MAX_RETRY_SECONDS = 30.0

    def __init__(
        self,
        participant_id: str,
        curriculum: CurriculumMode,
        endpoint: str | None = None,
        *,
        post_json: PostJson | None = None,
    ) -> None:
        if post_json is None:
            if endpoint is None:
                raise ValueError("endpoint is required when post_json is not supplied")
            post_json = PygbagJsonTransport(endpoint).post
        self.participant_id = participant_id
        self.curriculum = curriculum
        self.session_id = str(uuid4())
        self.session_number = 0
        self.records: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []
        self._post_json = post_json
        self._session_started = False
        self._finish_requested = False
        self._completion_acknowledged = False
        self._final_score = 0
        self._task: asyncio.Task[dict[str, Any]] | None = None
        self._task_kind = ""
        self._task_count = 0
        self._retry_count = 0
        self._retry_at = 0.0
        self.ranking: int | None = None
        self._sync_error = ""

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
        self._pending.append(record)

    def finish(self, score: int) -> None:
        self._final_score = score
        self._finish_requested = True
        return None

    @property
    def is_synced(self) -> bool:
        if self._finish_requested:
            return self._completion_acknowledged
        return self._session_started and not self._pending and self._task is None

    @property
    def sync_error(self) -> str:
        return self._sync_error

    async def pump(self) -> None:
        """Advance at most one background request; call once per rendered frame."""

        if self._task is not None and self._task.done():
            self._finish_task()
        if self._task is not None or asyncio.get_running_loop().time() < self._retry_at:
            await asyncio.sleep(0)
            return

        if not self._session_started:
            self._start_task(
                "start",
                {
                    "action": "start_session",
                    "session_id": self.session_id,
                    "participant_id": self.participant_id,
                    "curriculum": self.curriculum.value,
                },
            )
        elif self._pending:
            batch = self._pending[: self.MAX_BATCH_SIZE]
            serializable_batch = []
            for row in batch:
                converted = dict(row)
                converted["session_number"] = self.session_number
                converted["option_symbols"] = json.loads(converted["option_symbols"])
                serializable_batch.append(converted)
            self._start_task(
                "trials",
                {
                    "action": "save_trials",
                    "session_id": self.session_id,
                    "trials": serializable_batch,
                },
                count=len(batch),
            )
        elif self._finish_requested and not self._completion_acknowledged:
            self._start_task(
                "complete",
                {
                    "action": "complete_session",
                    "session_id": self.session_id,
                    "score": self._final_score,
                },
            )
        await asyncio.sleep(0)

    def _start_task(self, kind: str, payload: Mapping[str, Any], count: int = 0) -> None:
        self._task_kind = kind
        self._task_count = count
        self._task = asyncio.create_task(self._post_json(payload))

    def _finish_task(self) -> None:
        assert self._task is not None
        try:
            result = self._task.result()
            if self._task_kind == "start":
                self.session_number = int(result["session_number"])
                self._session_started = True
                for row in self._pending:
                    row["session_number"] = self.session_number
            elif self._task_kind == "trials":
                del self._pending[: self._task_count]
            elif self._task_kind == "complete":
                self.ranking = int(result["ranking"])
                self._completion_acknowledged = True
            self._retry_count = 0
            self._retry_at = 0.0
            self._sync_error = ""
        except Exception as exc:
            self._retry_count += 1
            delay = min(2 ** (self._retry_count - 1), self.MAX_RETRY_SECONDS)
            self._retry_at = asyncio.get_running_loop().time() + delay
            self._sync_error = str(exc)
        finally:
            self._task = None
            self._task_kind = ""
            self._task_count = 0
