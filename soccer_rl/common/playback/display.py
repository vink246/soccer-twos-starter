"""Headless display helpers for Unity watch builds and video capture."""

import os
import shutil
import subprocess
import time
from typing import Optional


def start_virtual_display_if_needed(headless: bool, display: str, size: str) -> Optional[subprocess.Popen]:
    """If headless, start Xvfb on DISPLAY. Returns the process or None."""
    if not headless:
        return None
    xvfb_path = shutil.which("Xvfb")
    if xvfb_path is None:
        raise RuntimeError(
            "Headless video requested but Xvfb was not found on PATH. "
            "Install Xvfb or run with xvfb-run."
        )
    os.environ.setdefault("DISPLAY", display)
    cmd = [xvfb_path, os.environ["DISPLAY"], "-screen", "0", size, "-nolisten", "tcp"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    return proc
