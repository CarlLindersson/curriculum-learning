"""Insert required public configuration into a GitHub Pages build checkout."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent / "scripts" / "web_config.py"


def main() -> None:
    values = {
        "__SUPABASE_FUNCTION_URL__": os.environ.get("SUPABASE_FUNCTION_URL", "").strip(),
        "__DATA_CONTROLLER_NAME__": os.environ.get("DATA_CONTROLLER_NAME", "").strip(),
        "__DATA_CONTROLLER_EMAIL__": os.environ.get("DATA_CONTROLLER_EMAIL", "").strip(),
        "__DATA_RETENTION_PERIOD__": os.environ.get("DATA_RETENTION_PERIOD", "").strip(),
    }
    if not values["__SUPABASE_FUNCTION_URL__"].startswith("https://"):
        raise SystemExit("SUPABASE_FUNCTION_URL must be a complete https:// URL")
    if not values["__DATA_CONTROLLER_NAME__"]:
        raise SystemExit("Set the DATA_CONTROLLER_NAME repository variable")
    if not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+", values["__DATA_CONTROLLER_EMAIL__"]
    ):
        raise SystemExit("DATA_CONTROLLER_EMAIL must be a valid public contact email")
    if not values["__DATA_RETENTION_PERIOD__"]:
        raise SystemExit("Set the DATA_RETENTION_PERIOD repository variable")

    source = CONFIG_PATH.read_text(encoding="utf-8")
    for placeholder, value in values.items():
        source = source.replace(json.dumps(placeholder), json.dumps(value))
    CONFIG_PATH.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
