# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compressionenv Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import CompressionenvAction, CompressionenvObservation


class CompressionenvEnv(
    EnvClient[CompressionenvAction, CompressionenvObservation, State]
):
    """
    Client for the Compressionenv Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CompressionenvEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.essay_id, len(result.observation.essay_text))
        ...
        ...     result = client.step(CompressionenvAction(
        ...         compression_code="def compress(t): return t.encode()",
        ...         decompression_code="def decompress(b): return b.decode()",
        ...     ))
        ...     print(result.observation.valid, result.observation.compressed_size_bytes)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CompressionenvEnv.from_docker_image("compressionenv-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CompressionenvAction(
        ...         compression_code="def compress(t): return t.encode()",
        ...         decompression_code="def decompress(b): return b.decode()",
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CompressionenvAction) -> Dict:
        """
        Convert CompressionenvAction to JSON payload for step message.

        Args:
            action: CompressionenvAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "compression_code": action.compression_code,
            "decompression_code": action.decompression_code,
            "algo_name": action.algo_name,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CompressionenvObservation]:
        """
        Parse server response into StepResult[CompressionenvObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CompressionenvObservation
        """
        obs_data = payload.get("observation", {})
        observation = CompressionenvObservation(
            essay_id=obs_data.get("essay_id", ""),
            essay_text=obs_data.get("essay_text", ""),
            valid=obs_data.get("valid", False),
            error=obs_data.get("error"),
            compressed_size_bytes=obs_data.get("compressed_size_bytes"),
            avg_prev_compressed_size_bytes=obs_data.get("avg_prev_compressed_size_bytes"),
            improved_over_avg=obs_data.get("improved_over_avg"),
            baselines_size_bytes=obs_data.get("baselines_size_bytes") or {},
            best_baseline_size_bytes=obs_data.get("best_baseline_size_bytes"),
            beat_any_baseline=obs_data.get("beat_any_baseline"),
            beat_best_baseline=obs_data.get("beat_best_baseline"),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            done=obs_data.get("done", payload.get("done", False)),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request (may include extra fields
                e.g. essay_id, essay_text, baselines_size_bytes).

        Returns:
            State object with episode_id, step_count, and any extra fields.
        """
        return State(**payload)
