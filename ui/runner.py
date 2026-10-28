"""Subprocess runner for run_simulation.py / run_experiments.py."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from ui.state import CODE_DIR

SIM_SCRIPT = os.path.join(CODE_DIR, "run_simulation.py")
EXP_SCRIPT = os.path.join(CODE_DIR, "run_experiments.py")


def _read_script_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def discover_simulation_flags() -> dict[str, Any]:
    """
    Inspect run_simulation.py for optional CLI flags that concurrent roadmap
    work may add (dataset, non-IID, backend). Always supports --rounds.
    """
    text = _read_script_text(SIM_SCRIPT)
    flags = {
        "rounds": True,
        "dataset": bool(
            re.search(r"--dataset|dataset\s*=|choices=.*pima|heart", text, re.I)
        ),
        "non_iid": bool(
            re.search(r"--non-?iid|--iid|non_iid|label.?skew", text, re.I)
        ),
        "backend": bool(
            re.search(r"--backend|CHAIN_BACKEND|get_chain\(", text, re.I)
        ),
        "exists": os.path.isfile(SIM_SCRIPT),
    }
    # Dataset choices: prefer explicit mentions, else offer pima only
    datasets = ["pima"]
    if re.search(r"heart", text, re.I):
        datasets.append("heart")
    flags["dataset_choices"] = datasets
    flags["experiments_exists"] = os.path.isfile(EXP_SCRIPT)
    return flags


@dataclass
class SimulationRunner:
    """Track a single background simulation / experiment subprocess."""

    process: subprocess.Popen | None = None
    kind: str = "simulation"  # simulation | experiments
    log_tail: list[str] = field(default_factory=list)
    last_cmd: list[str] = field(default_factory=list)
    last_error: str | None = None

    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def poll_logs(self, max_lines: int = 40) -> list[str]:
        if self.process is None or self.process.stdout is None:
            return self.log_tail
        try:
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                self.log_tail.append(line.rstrip())
                if len(self.log_tail) > max_lines:
                    self.log_tail = self.log_tail[-max_lines:]
        except (OSError, ValueError):
            pass
        return self.log_tail

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def start_simulation(
        self,
        rounds: int = 10,
        dataset: str | None = None,
        non_iid: bool | None = None,
        backend: str = "inprocess",
        python_exe: str | None = None,
    ) -> bool:
        if self.is_running():
            self.last_error = "A process is already running."
            return False
        if not os.path.isfile(SIM_SCRIPT):
            self.last_error = f"Missing {SIM_SCRIPT}"
            return False

        flags = discover_simulation_flags()
        py = python_exe or sys.executable
        cmd = [py, SIM_SCRIPT, "--rounds", str(int(rounds))]

        if flags["dataset"] and dataset:
            cmd.extend(["--dataset", dataset])
        if flags["non_iid"] and non_iid is not None:
            if non_iid:
                cmd.append("--non-iid")
            else:
                # Prefer --iid when present; otherwise omit
                text = _read_script_text(SIM_SCRIPT)
                if re.search(r"--iid\b", text):
                    cmd.append("--iid")
        if flags["backend"] and backend:
            text = _read_script_text(SIM_SCRIPT)
            if re.search(r"--backend", text):
                cmd.extend(["--backend", backend])

        env = os.environ.copy()
        env["CHAIN_BACKEND"] = backend or "inprocess"
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=CODE_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.kind = "simulation"
            self.last_cmd = cmd
            self.last_error = None
            self.log_tail = [f"$ {' '.join(cmd)}"]
            return True
        except OSError as exc:
            self.last_error = str(exc)
            self.process = None
            return False

    def start_experiments(self, python_exe: str | None = None) -> bool:
        if self.is_running():
            self.last_error = "A process is already running."
            return False
        if not os.path.isfile(EXP_SCRIPT):
            self.last_error = "run_experiments.py not found yet."
            return False
        py = python_exe or sys.executable
        cmd = [py, EXP_SCRIPT]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=CODE_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.kind = "experiments"
            self.last_cmd = cmd
            self.last_error = None
            self.log_tail = [f"$ {' '.join(cmd)}"]
            return True
        except OSError as exc:
            self.last_error = str(exc)
            self.process = None
            return False
