# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compressionenv Environment Implementation.

Environment where the agent proposes compression/decompression algorithms for a
Paul Graham essay. The environment validates round-trip correctness and scores
compressed size relative to the agent's prior attempts and baseline compressors.
"""

import base64
import json
import os
import re
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import bz2
import lzma
import zlib

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import CompressionenvAction, CompressionenvObservation


def _uses_forbidden_compression(compression_code: str, decompression_code: str) -> bool:
    """True if code uses zlib, bz2, lzma, gzip - reward will be set to 0."""
    combined = compression_code + "\n" + decompression_code
    return bool(re.search(r"\b(zlib|bz2|lzma|gzip)\b", combined))


@dataclass(frozen=True)
class _Essay:
    essay_id: str
    text: str


class CompressionenvEnvironment(Environment):
    """
    Compression algorithm search environment.

    - On `reset()`, selects a PG essay (from `../essays/*.txt`) and returns it.
    - On `step()`, executes agent-provided Python code defining:
        compress(text: str) -> bytes
        decompress(data: bytes) -> str
      Validates that decompress(compress(essay)) == essay.

    Rewards (per spec):
    - If algorithms fail or don't round-trip: -1 reward.
    - If compressed size is lower than average of previous successful sizes for
      this essay in the episode: +1 reward.
    - Compare against baselines (zlib, bz2, lzma):
        - If agent achieves smaller size than at least one baseline: +10 reward.
        - If agent achieves smaller size than the best baseline: +20 reward.
    """

    # Episode ends after this many steps.
    MAX_STEPS: int = 20

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the compressionenv environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._essay: _Essay | None = None
        self._successful_sizes: list[int] = []
        self._baselines: dict[str, int] = {}

    def reset(self) -> CompressionenvObservation:
        """
        Reset the environment.

        Returns:
            CompressionenvObservation containing a selected essay
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._essay = self._pick_essay()
        self._successful_sizes = []
        self._baselines = self._compute_baselines(self._essay.text)
        self._update_state()

        return CompressionenvObservation(
            essay_id=self._essay.essay_id,
            essay_text=self._essay.text,
            valid=True,
            error=None,
            compressed_size_bytes=None,
            avg_prev_compressed_size_bytes=None,
            improved_over_avg=None,
            baselines_size_bytes=self._baselines,
            best_baseline_size_bytes=min(self._baselines.values()) if self._baselines else None,
            beat_any_baseline=None,
            beat_best_baseline=None,
            done=False,
            reward=0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "num_baselines": len(self._baselines),
            },
        )

    def step(self, action: CompressionenvAction) -> CompressionenvObservation:  # type: ignore[override]
        """
        Execute a step: run agent algorithms, validate, score compression size.
        """
        if self._essay is None:
            # Defensive: ensure reset called.
            self._essay = self._pick_essay()
            self._baselines = self._compute_baselines(self._essay.text)
            self._successful_sizes = []

        self._state.step_count += 1

        essay_text = self._essay.text
        baselines = self._baselines
        best_baseline = min(baselines.values()) if baselines else None

        reward = 0.0
        error: str | None = None
        valid = False
        compressed_size: int | None = None
        improved_over_avg: bool | None = None
        beat_any_baseline: bool | None = None
        beat_best_baseline: bool | None = None
        avg_prev: float | None = None

        try:
            compressed_bytes = self._run_agent_codec(
                essay_text=essay_text,
                compression_code=action.compression_code,
                decompression_code=action.decompression_code,
            )
            compressed_size = len(compressed_bytes)
            valid = True
        except Exception as e:
            error = str(e)
            reward = -1.0

        if valid and compressed_size is not None:
            if self._successful_sizes:
                avg_prev = sum(self._successful_sizes) / len(self._successful_sizes)
                improved_over_avg = compressed_size < avg_prev
                if improved_over_avg:
                    reward += 1.0
            else:
                avg_prev = None
                improved_over_avg = None

            self._successful_sizes.append(compressed_size)

            if baselines:
                beat_any_baseline = any(compressed_size < s for s in baselines.values())
                beat_best_baseline = best_baseline is not None and compressed_size < best_baseline
                if beat_best_baseline:
                    reward += 20.0
                elif beat_any_baseline:
                    reward += 10.0

        if _uses_forbidden_compression(action.compression_code, action.decompression_code):
            reward = 0.0

        observation = CompressionenvObservation(
            essay_id=self._essay.essay_id,
            essay_text=essay_text,
            valid=valid,
            error=error,
            compressed_size_bytes=compressed_size,
            avg_prev_compressed_size_bytes=avg_prev,
            improved_over_avg=improved_over_avg,
            baselines_size_bytes=baselines,
            best_baseline_size_bytes=best_baseline,
            beat_any_baseline=beat_any_baseline,
            beat_best_baseline=beat_best_baseline,
            done=self._state.step_count >= self.MAX_STEPS,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "algo_name": action.algo_name,
                "num_successful_attempts": len(self._successful_sizes),
            },
        )
        self._update_state()
        return observation

    def _update_state(self) -> None:
        """Sync internal state (State) from current episode and step data."""
        if self._essay is not None:
            self._state.essay_id = self._essay.essay_id  # type: ignore[attr-defined]
            self._state.essay_text = self._essay.text  # type: ignore[attr-defined]
            self._state.baselines_size_bytes = self._baselines  # type: ignore[attr-defined]
            self._state.num_successful_attempts = len(self._successful_sizes)  # type: ignore[attr-defined]
            if self._successful_sizes:
                self._state.best_compressed_size_bytes = min(self._successful_sizes)  # type: ignore[attr-defined]
                self._state.last_compressed_size_bytes = self._successful_sizes[-1]  # type: ignore[attr-defined]
            else:
                self._state.best_compressed_size_bytes = None  # type: ignore[attr-defined]
                self._state.last_compressed_size_bytes = None  # type: ignore[attr-defined]
            if self._baselines:
                self._state.best_baseline_size_bytes = min(self._baselines.values())  # type: ignore[attr-defined]
            else:
                self._state.best_baseline_size_bytes = None  # type: ignore[attr-defined]
        return None

    @property
    def state(self) -> State:
        """
        Get the current environment **state**.

        In RL terms, the State is a (Markov) description of the underlying
        environment that is at least as informative as any single Observation.
        Here we include all information needed to reconstruct what any call to
        `reset()` or `step()` would expose in an observation for this episode.
        """
        return self._state

    def _pick_essay(self) -> _Essay:
        # Expected layout:
        #   compression-openenv/
        #     essays/
        #     compressionenv/
        #       server/
        #         compressionenv_environment.py  (this file)
        essays_dir = Path(__file__).resolve().parents[2] / "essays"
        if not essays_dir.exists():
            # Try repo-level essays directory (if running from different cwd/layout).
            essays_dir = Path(os.getcwd()).resolve() / "essays"
        paths = sorted(essays_dir.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(
                f"No essays found in {essays_dir}. Expected PG essay .txt files."
            )
        path = random.choice(paths)
        essay_id = path.stem
        text = path.read_text(encoding="utf-8")
        return _Essay(essay_id=essay_id, text=text)

    def _compute_baselines(self, text: str) -> dict[str, int]:
        data = text.encode("utf-8")
        # Deterministic settings.
        baselines: dict[str, bytes] = {
            "zlib": zlib.compress(data, level=9),
            "bz2": bz2.compress(data, compresslevel=9),
            "lzma": lzma.compress(data, preset=9),
        }
        return {k: len(v) for k, v in baselines.items()}

    def _run_agent_codec(
        self,
        essay_text: str,
        compression_code: str,
        decompression_code: str,
    ) -> bytes:
        """
        Execute agent code in a subprocess and return compressed bytes.

        Security note: this is not a hardened sandbox. It's a best-effort isolation
        to avoid contaminating the server process, with a timeout.
        """
        runner = r"""
import base64
import json
import sys

payload = json.loads(sys.stdin.read())
essay_text = payload["essay_text"]
compression_code = payload["compression_code"]
decompression_code = payload["decompression_code"]

ns = {}
exec(compression_code, ns, ns)
exec(decompression_code, ns, ns)

compress = ns.get("compress")
decompress = ns.get("decompress")
if compress is None or decompress is None:
    raise RuntimeError("Expected functions compress(text: str)->bytes and decompress(data: bytes)->str")

compressed = compress(essay_text)
if not isinstance(compressed, (bytes, bytearray)):
    raise RuntimeError(f"compress() must return bytes, got {type(compressed)}")
compressed = bytes(compressed)

round_trip = decompress(compressed)
if not isinstance(round_trip, str):
    raise RuntimeError(f"decompress() must return str, got {type(round_trip)}")
if round_trip != essay_text:
    raise RuntimeError("Round-trip failed: decompress(compress(essay)) != essay")

sys.stdout.write(base64.b64encode(compressed).decode("ascii"))
"""
        payload = {
            "essay_text": essay_text,
            "compression_code": compression_code,
            "decompression_code": decompression_code,
        }
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, "-c", runner],
                input=json.dumps(payload).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=td,
                timeout=3.0,
                env={
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(stderr or f"Agent codec subprocess failed with code {proc.returncode}")
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        try:
            return base64.b64decode(out.encode("ascii"), validate=True)
        except Exception as e:
            raise RuntimeError(f"Failed to decode compressed output: {e}") from e
