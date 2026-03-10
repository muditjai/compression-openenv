# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compressionenv Environment."""

from .client import CompressionenvEnv
from .models import CompressionenvAction, CompressionenvObservation

__all__ = [
    "CompressionenvAction",
    "CompressionenvObservation",
    "CompressionenvEnv",
    "ensure_essays",
]


def ensure_essays():
    """Ensure essays/ exists (extract from essays.tar.gz if needed). For remote Jupyter."""
    import os

    from .essays_utils import ensure_essays_extracted

    path = ensure_essays_extracted()
    n = len(os.listdir(path))
    print(f"essays/ ready ({n} files)")
    return path
