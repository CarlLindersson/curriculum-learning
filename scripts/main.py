"""Pygbag browser entry point; the desktop entry point remains curriculum_pygame.py."""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

from curriculum_pygame import CurriculumGame, CurriculumMode, PrivacyNotice
from remote_recorder import RemoteSessionRecorder
from web_config import (
    DATA_CONTROLLER_EMAIL,
    DATA_CONTROLLER_NAME,
    DATA_RETENTION_PERIOD,
    DEFAULT_CURRICULUM,
    PRIVACY_NOTICE_VERSION,
    SUPABASE_FUNCTION_URL,
)


def new_anonymous_participant_id() -> str:
    """Create a fresh in-memory code for one play; never persist it in the browser."""

    return str(uuid4())


def remote_recorder_factory(
    participant_id: str,
    curriculum: CurriculumMode,
    _data_dir: object = None,
) -> RemoteSessionRecorder:
    return RemoteSessionRecorder(
        participant_id,
        curriculum,
        endpoint=SUPABASE_FUNCTION_URL,
    )


async def main() -> None:
    if SUPABASE_FUNCTION_URL.startswith("__"):
        raise RuntimeError(
            "Set SUPABASE_FUNCTION_URL in web_config.py or through the Pages build."
        )
    curriculum = os.environ.get("CURRICULUM", DEFAULT_CURRICULUM).lower()
    if sys.platform == "emscripten":
        import platform

        query = str(platform.window.location.search).lstrip("?")
        for item in query.split("&"):
            key, separator, value = item.partition("=")
            if separator and key.lower() == "curriculum":
                curriculum = value.lower()
                break
    game = CurriculumGame(
        {"mode": curriculum},
        recorder_factory=remote_recorder_factory,
        anonymous_participant_id=new_anonymous_participant_id(),
        new_anonymous_participant=new_anonymous_participant_id,
        privacy_notice=PrivacyNotice(
            controller_name=DATA_CONTROLLER_NAME,
            contact_email=DATA_CONTROLLER_EMAIL,
            retention_period=DATA_RETENTION_PERIOD,
            version=PRIVACY_NOTICE_VERSION,
        ),
        require_privacy_acceptance=True,
    )
    await game.run_async()


asyncio.run(main())
