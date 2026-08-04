"""Pygbag browser entry point; the desktop entry point remains curriculum_pygame.py."""

from __future__ import annotations

import asyncio
import os
import sys

from curriculum_pygame import CurriculumGame, CurriculumMode
from remote_recorder import RemoteSessionRecorder
from web_config import DEFAULT_CURRICULUM, SUPABASE_FUNCTION_URL


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
    )
    await game.run_async()


asyncio.run(main())
