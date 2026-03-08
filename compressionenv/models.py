# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Compressionenv Environment.

The compressionenv environment gives the agent a Paul Graham essay and asks it to
propose compression + decompression algorithms (as Python code).
"""

from typing import Any, Dict, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class CompressionenvAction(Action):
    """
    Agent-provided compression/decompression algorithms.

    The environment expects `compression_code` and `decompression_code` to define:

    - compress(text: str) -> bytes
    - decompress(data: bytes) -> str
    """

    compression_code: str = Field(
        ...,
        description="Python code defining compress(text: str) -> bytes",
        min_length=1,
    )
    decompression_code: str = Field(
        ...,
        description="Python code defining decompress(data: bytes) -> str",
        min_length=1,
    )
    algo_name: str = Field(
        default="agent_algo",
        description="Optional name/label for this algorithm variant",
    )


class CompressionenvObservation(Observation):
    """Observation from the Compressionenv environment."""

    essay_id: str = Field(..., description="Selected essay slug/id for this episode")
    essay_text: str = Field(
        ...,
        description="Full essay text for the agent to compress",
    )

    valid: bool = Field(
        default=False,
        description="Whether the submitted algorithms successfully round-tripped",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the algorithms failed validation/execution",
    )

    compressed_size_bytes: Optional[int] = Field(
        default=None,
        description="Size of compressed bytes produced by the agent algorithm",
        ge=0,
    )
    avg_prev_compressed_size_bytes: Optional[float] = Field(
        default=None,
        description="Average compressed size over previous successful steps for this essay",
        ge=0,
    )
    improved_over_avg: Optional[bool] = Field(
        default=None,
        description="True if current compressed size < avg of previous sizes",
    )

    baselines_size_bytes: Dict[str, int] = Field(
        default_factory=dict,
        description="Baseline compressor sizes for this essay (zlib/bz2/lzma)",
    )
    best_baseline_size_bytes: Optional[int] = Field(
        default=None,
        description="Best (smallest) baseline size in bytes",
        ge=0,
    )
    beat_any_baseline: Optional[bool] = Field(
        default=None,
        description="True if current compressed size is smaller than at least one baseline",
    )
    beat_best_baseline: Optional[bool] = Field(
        default=None,
        description="True if current compressed size is smaller than the best baseline",
    )

    reward: float = Field(default=0.0, description="Reward for this step")
    done: bool = Field(default=False, description="Whether episode is done")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra info")
