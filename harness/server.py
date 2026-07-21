"""Manage a vLLM OpenAI-compatible server as a background subprocess.

Running vLLM as a real subprocess (rather than embedding the `LLM` class in the
calling process) avoids environment-specific stdout/file-descriptor issues seen
in notebook environments (see docs/PHASE0_RESULTS.md), and matches how vLLM is
actually deployed in production — which is also what the reproducibility
measurements are meant to characterize.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional, Tuple

import requests


class ServerStartupError(RuntimeError):
    pass


@dataclass
class VLLMServer:
    model: str
    port: int = 8000
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    seed: int = 0
    extra_args: Tuple[str, ...] = field(default_factory=tuple)
    log_path: Path = Path("vllm_server.log")

    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _log_file: Optional[IO] = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def start(self, timeout_s: int = 600, poll_interval_s: float = 5.0) -> None:
        self._log_file = open(self.log_path, "w")
        cmd = [
            "vllm", "serve", self.model,
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--port", str(self.port),
            "--seed", str(self.seed),
            *self.extra_args,
        ]
        self._proc = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerStartupError(
                    f"vLLM server exited early with code {self._proc.returncode}; "
                    f"see {self.log_path}"
                )
            try:
                r = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(poll_interval_s)
        raise ServerStartupError(
            f"Server did not become healthy within {timeout_s}s; see {self.log_path}"
        )

    def stop(self, timeout_s: int = 30) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Graceful shutdown (CUDA context + NCCL teardown) can take a
            # while; force kill rather than hang the caller.
            self._proc.kill()
            self._proc.wait(timeout=timeout_s)
        if self._log_file:
            self._log_file.close()
        self._proc = None

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
