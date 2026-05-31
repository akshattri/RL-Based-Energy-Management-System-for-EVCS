"""
matlab_runner.py
================
Helper for triggering MATLAB from Streamlit on macOS.

Workflow when the Streamlit button is clicked:
  1. Write a temporary .m "trigger" script with the commands we want to run.
  2. Use AppleScript to tell MATLAB to evaluate that script.
  3. If MATLAB isn't running, launch it first with `open -a MATLAB_R2024b`.

This avoids the matlab.engine Python package, which is fiddly to install
on macOS arm64.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Where MATLAB lives on macOS.  R2024b is the default; override via env var
# MATLAB_APP if you have a different version installed.
# ---------------------------------------------------------------------------
DEFAULT_MATLAB_APP = "MATLAB_R2024b"


def _matlab_app_name() -> str:
    return os.environ.get("MATLAB_APP", DEFAULT_MATLAB_APP)


def matlab_is_running() -> bool:
    """True if there's a process named MATLAB on the system."""
    try:
        out = subprocess.run(
            ["pgrep", "-x", "MATLAB"],
            capture_output=True, text=True, timeout=5,
        )
        return out.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def trigger_demo(project_dir: str,
                 stop_time: float = 540.0,
                 model_name: str = "EVCS_model_v2") -> dict:
    """
    Tell MATLAB to run the demo.

    Returns a dict with keys:
      ok          : bool
      message     : human-readable status
      script_path : str path of the trigger .m we wrote
    """
    project_dir = str(Path(project_dir).expanduser().resolve())
    mat_file = os.path.join(project_dir, "data", "ems_schedule.mat")

    if not os.path.isfile(mat_file):
        return {
            "ok": False,
            "message": (f"Schedule not found at {mat_file}. "
                        f"Click 'Run pipeline' first."),
            "script_path": None,
        }

    # The .m we want MATLAB to evaluate
    matlab_cmd = textwrap.dedent(f"""\
        fprintf('=== Streamlit trigger received ===\\n');
        cd('{project_dir}');
        fprintf('cd to: %s\\n', pwd);
        addpath(pwd);
        try
            demo({stop_time});
            fprintf('=== demo() finished ===\\n');
        catch ME
            fprintf('=== ERROR ===\\n');
            fprintf('%s\\n', getReport(ME));
        end
    """).strip()

    # Write to a stable path so MATLAB can read it more than once
    script_path = os.path.join(project_dir, "data", "_run_demo_trigger.m")
    with open(script_path, "w") as f:
        f.write(matlab_cmd + "\n")

    # AppleScript: focus MATLAB, paste the command, hit Enter.
    # We use `do script` via System Events because the MATLAB.app
    # AppleScript dictionary is limited.
    apple_script = textwrap.dedent(f"""\
        tell application "{_matlab_app_name()}" to activate
        delay 0.5
        tell application "System Events"
            keystroke "run('{script_path}')"
            key code 36
        end tell
    """)

    # If MATLAB isn't running, launch it first; needs a longer delay.
    if not matlab_is_running():
        subprocess.run(["open", "-a", _matlab_app_name()],
                       capture_output=True, text=True, timeout=10)
        # Wait for MATLAB to be ready before sending keystrokes
        apple_script = "delay 8\n" + apple_script

    proc = subprocess.run(
        ["osascript", "-e", apple_script],
        capture_output=True, text=True, timeout=30,
    )

    if proc.returncode != 0:
        return {
            "ok": False,
            "message": (
                f"AppleScript failed (rc={proc.returncode}).\n"
                f"stderr: {proc.stderr.strip()}\n\n"
                f"You may need to grant Terminal/Streamlit access to "
                f"control MATLAB:\n"
                f"  System Settings → Privacy & Security → Accessibility\n"
                f"  Add the app you launched Streamlit from (Terminal/iTerm).\n\n"
                f"Workaround: in MATLAB, run:\n"
                f"  >> run('{script_path}')"
            ),
            "script_path": script_path,
        }

    return {
        "ok": True,
        "message": (
            "Triggered MATLAB to run the demo. "
            "Switch to MATLAB to watch the scopes."
        ),
        "script_path": script_path,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    res = trigger_demo(os.getcwd(), stop_time=90.0)
    print(res)