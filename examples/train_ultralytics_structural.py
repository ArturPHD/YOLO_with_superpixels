"""Compatibility forwarder for the single versioned SP-11 pilot launcher.

Use ``scripts/run_sp11_pilot.py`` directly in new commands. This entry point
intentionally forwards all arguments so it cannot introduce another set of
training defaults.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    script = ROOT / "scripts" / "run_sp11_pilot.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
