"""Asynchronous browser recorder used by the Pygbag build."""

from __future__ import annotations

import asyncio
import json
import re
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
RememberPending = Callable[[str, list[Mapping[str, Any]]], None]
ALIAS_RE = re.compile(r"^[a-z]+-[a-z]+$")
BALANCED_CURRICULA = {
    CurriculumMode.BLOCKED,
    CurriculumMode.PROGRESSIVELY_INTERLEAVED,
}


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
window.CurriculumFetch.pending = window.CurriculumFetch.pending || {};
window.CurriculumFetch.requestOptions = function(data, keepalive) {
    return {
        method: "POST",
        headers: {"Accept": "application/json", "Content-Type": "application/json"},
        body: data,
        keepalive: keepalive
    };
};
window.CurriculumFetch.remember = function(sessionId, url, payloadsJson) {
    try {
        const payloads = JSON.parse(payloadsJson);
        if (payloads.length) {
            window.CurriculumFetch.pending[sessionId] = {url: url, payloads: payloads};
        } else {
            delete window.CurriculumFetch.pending[sessionId];
        }
    } catch (error) {
        console.warn("Could not retain unsaved curriculum trials in page memory", error);
    }
};
window.CurriculumFetch.flushOnExit = function() {
    for (const pending of Object.values(window.CurriculumFetch.pending)) {
        for (const payload of pending.payloads) {
            try {
                fetch(
                    pending.url,
                    window.CurriculumFetch.requestOptions(JSON.stringify(payload), true)
                ).catch(() => {});
            } catch (error) {
                // Page-exit upload is best effort and may be terminated by the browser.
            }
        }
    }
};
if (!window.CurriculumFetch.exitHandlerInstalled) {
    window.CurriculumFetch.exitHandlerInstalled = true;
    window.addEventListener("pagehide", window.CurriculumFetch.flushOnExit);
}
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

    def remember_pending(
        self, session_id: str, payloads: list[Mapping[str, Any]]
    ) -> None:
        """Retain unsent batches in page memory for a best-effort exit upload."""

        encoded = json.dumps(payloads, separators=(",", ":"))
        self.platform.window.CurriculumFetch.remember(
            session_id, self.endpoint, encoded
        )


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
        remember_pending: RememberPending | None = None,
    ) -> None:
        if post_json is None:
            if endpoint is None:
                raise ValueError("endpoint is required when post_json is not supplied")
            transport = PygbagJsonTransport(endpoint)
            post_json = transport.post
            remember_pending = transport.remember_pending
        self.participant_id = participant_id
        # This is provisional until start_session returns the authoritative
        # server allocation. Trials cannot begin before that response.
        self.curriculum = curriculum
        self.assigned_curriculum: CurriculumMode | None = None
        self.session_id = str(uuid4())
        self.session_number = 0
        self.records: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []
        self._post_json = post_json
        self._remember_pending = remember_pending
        self._identity_checked = False
        self._start_requested = False
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
        self.alias_options: list[str] = []
        self.leaderboard_name: str | None = None
        self.alias_status = ""
        self._selected_alias: str | None = None
        self._sync_error = ""

    def choose_alias(self, alias: str) -> None:
        """Queue one of the server-offered aliases for an atomic claim."""

        if self.leaderboard_name is not None:
            return
        if self._selected_alias is not None:
            return
        if alias not in self.alias_options:
            raise ValueError("Choose one of the offered leaderboard names")
        self._selected_alias = alias
        self._start_requested = True
        self.alias_status = "Reserving your leaderboard name..."

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
        self._persist_pending()

    def finish(self, score: int) -> None:
        self._final_score = score
        self._finish_requested = True
        return None

    @property
    def is_synced(self) -> bool:
        if self._finish_requested:
            return self._completion_acknowledged
        return self.is_ready and not self._pending and self._task is None

    @property
    def is_ready(self) -> bool:
        """Trials must not begin until the server has created the session row."""

        return (
            self._session_started
            and self.assigned_curriculum is not None
            and self.leaderboard_name is not None
        )

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

        if not self._identity_checked:
            self._start_task(
                "identity",
                {
                    "action": "identify_participant",
                    "participant_id": self.participant_id,
                },
            )
        elif self._start_requested and not self._session_started:
            self._start_task(
                "start",
                {
                    "action": "start_session",
                    "session_id": self.session_id,
                    "participant_id": self.participant_id,
                },
            )
        elif self.leaderboard_name is None:
            if self._selected_alias is not None:
                self._start_task(
                    "alias",
                    {
                        "action": "claim_alias",
                        "session_id": self.session_id,
                        "alias": self._selected_alias,
                    },
                )
        elif self._pending:
            batch = self._pending[: self.MAX_BATCH_SIZE]
            self._start_task(
                "trials",
                {
                    "action": "save_trials",
                    "session_id": self.session_id,
                    "trials": self._serializable_batch(batch),
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
            if self._task_kind == "identity":
                self._identity_checked = True
                self._read_alias_offer(result)
                self.alias_status = ""
            elif self._task_kind == "start":
                self.session_number = int(result["session_number"])
                try:
                    assigned_curriculum = CurriculumMode(result["curriculum"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Data service returned an invalid curriculum allocation"
                    ) from exc
                if assigned_curriculum not in BALANCED_CURRICULA:
                    raise RuntimeError(
                        "Data service returned an invalid curriculum allocation"
                    )
                self.curriculum = assigned_curriculum
                self.assigned_curriculum = assigned_curriculum
                self._session_started = True
                self._read_alias_offer(result)
                self.alias_status = ""
                for row in self._pending:
                    row["session_number"] = self.session_number
                    row["curriculum"] = self.curriculum.value
                self._persist_pending()
            elif self._task_kind == "alias":
                if result.get("claimed") is True:
                    leaderboard_name = result.get("leaderboard_name")
                    if not isinstance(leaderboard_name, str) or not ALIAS_RE.fullmatch(
                        leaderboard_name
                    ):
                        raise RuntimeError("Data service returned an invalid leaderboard name")
                    self.leaderboard_name = leaderboard_name
                    self.alias_options = []
                    self._selected_alias = None
                    self.alias_status = ""
                else:
                    self._selected_alias = None
                    self._read_alias_offer(result)
                    self.alias_status = (
                        "That name was just taken. Please choose another one."
                    )
            elif self._task_kind == "trials":
                del self._pending[: self._task_count]
                self._persist_pending()
            elif self._task_kind == "complete":
                self.ranking = int(result["ranking"])
                self._completion_acknowledged = True
                self._persist_pending()
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

    def _read_alias_offer(self, result: Mapping[str, Any]) -> None:
        leaderboard_name = result.get("leaderboard_name")
        if leaderboard_name is not None:
            if not isinstance(leaderboard_name, str) or not ALIAS_RE.fullmatch(
                leaderboard_name
            ):
                raise RuntimeError("Data service returned an invalid leaderboard name")
            self.leaderboard_name = leaderboard_name
            self.alias_options = []
            self._selected_alias = None
            return

        raw_options = result.get("alias_options")
        if not isinstance(raw_options, list) or not raw_options:
            raise RuntimeError("Data service did not offer any leaderboard names")
        options: list[str] = []
        for option in raw_options[:3]:
            if not isinstance(option, str) or not ALIAS_RE.fullmatch(option):
                raise RuntimeError("Data service returned an invalid leaderboard-name option")
            if option not in options:
                options.append(option)
        if not options:
            raise RuntimeError("Data service did not offer any leaderboard names")
        self.alias_options = options

    def _serializable_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        serializable = []
        for row in batch:
            converted = dict(row)
            converted["session_number"] = self.session_number
            converted["option_symbols"] = json.loads(converted["option_symbols"])
            serializable.append(converted)
        return serializable

    def _persist_pending(self) -> None:
        if self._remember_pending is None or not self._session_started:
            return
        payloads: list[Mapping[str, Any]] = []
        for offset in range(0, len(self._pending), self.MAX_BATCH_SIZE):
            batch = self._pending[offset : offset + self.MAX_BATCH_SIZE]
            payloads.append(
                {
                    "action": "save_trials",
                    "session_id": self.session_id,
                    "trials": self._serializable_batch(batch),
                }
            )
        try:
            self._remember_pending(self.session_id, payloads)
        except Exception:
            # Normal network synchronization remains available if the JS bridge fails.
            pass
