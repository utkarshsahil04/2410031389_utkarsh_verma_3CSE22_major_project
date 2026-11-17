"""Subprocess runner for run_simulation.py / run_experiments.py."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ui.state import CODE_DIR

SIM_SCRIPT = os.path.join(CODE_DIR, "run_simulation.py")
EXP_SCRIPT = os.path.join(CODE_DIR, "run_experiments.py")


def discover_simulation_flags() -> dict[str, Any]:
    """CLI flags run_simulation.py supports (keep in sync with its parse_args)."""
    return {
        "rounds": True,
        "dataset": True,
        "non_iid": True,
        "backend": True,  # via CHAIN_BACKEND env, not a CLI flag
        "exists": os.path.isfile(SIM_SCRIPT),
        "dataset_choices": ["pima", "heart", "mimic"],
        "experiments_exists": os.path.isfile(EXP_SCRIPT),
    }


@dataclass
class SimulationRunner:
    """Track a single background simulation / experiment subprocess."""

    process: subprocess.Popen | None = None
    kind: str = "simulation"  # simulation | experiments
    log_tail: deque = field(default_factory=lambda: deque(maxlen=200))
    last_cmd: list[str] = field(default_factory=list)
    last_error: str | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def poll_logs(self, max_lines: int = 40) -> list[str]:
        # Lines are collected by a reader thread, so this never blocks the Streamlit script.
        return list(self.log_tail)[-max_lines:]

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.log_tail.append("[stopped by user]")
        self.process = None

    def _spawn(self, cmd: list[str], kind: str, extra_env: dict[str, str] | None = None) -> bool:
        env = os.environ.copy()
        env.update(extra_env or {})
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=CODE_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self.last_error = str(exc)
            self.process = None
            return False
        self.process, self.kind, self.last_cmd, self.last_error = proc, kind, cmd, None
        self.log_tail = deque([f"$ {' '.join(cmd)}"], maxlen=200)
        tail = self.log_tail
        threading.Thread(
            target=lambda: [tail.append(line.rstrip()) for line in proc.stdout], daemon=True
        ).start()
        return True

    def start_simulation(
        self,
        rounds: int = 10,
        dataset: str | None = None,
        non_iid: bool | None = None,
        backend: str = "inprocess",
        python_exe: str | None = None,
        keep_data: bool = False,
    ) -> bool:
        if self.is_running():
            self.last_error = "A process is already running."
            return False
        if not os.path.isfile(SIM_SCRIPT):
            self.last_error = f"Missing {SIM_SCRIPT}"
            return False
        cmd = [python_exe or sys.executable, SIM_SCRIPT, "--rounds", str(int(rounds))]
        if dataset:
            cmd.extend(["--dataset", dataset])
        if non_iid is not None:
            cmd.extend(["--split", "noniid" if non_iid else "iid"])
        if keep_data:
            cmd.append("--keep-data")
        return self._spawn(cmd, "simulation", {"CHAIN_BACKEND": backend or "inprocess"})

    def start_experiments(
        self, dataset: str = "pima", split: str = "iid", python_exe: str | None = None
    ) -> bool:
        if self.is_running():
            self.last_error = "A process is already running."
            return False
        if not os.path.isfile(EXP_SCRIPT):
            self.last_error = "run_experiments.py not found yet."
            return False
        cmd = [python_exe or sys.executable, EXP_SCRIPT, "--dataset", dataset, "--split", split]
        return self._spawn(cmd, "experiments")
