# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for essays directory (e.g. extract from tar for remote Jupyter)."""

import os
import tarfile


def ensure_essays_extracted(essays_dir: str = "essays", archive_path: str = "essays.tar.gz") -> str:
    """
    Ensure essays/ exists with .txt files. If not, extract from essays.tar.gz.

    Returns:
        Path to essays directory (absolute).

    Raises:
        FileNotFoundError: If essays/ is empty and essays.tar.gz not found.
    """
    if os.path.isdir(essays_dir) and os.listdir(essays_dir):
        return os.path.abspath(essays_dir)
    if os.path.isfile(archive_path):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(".")
        return os.path.abspath(essays_dir)
    raise FileNotFoundError(
        f"{archive_path} not found. Upload it to the same directory as this notebook. "
        "It is located in the repo root alongside notebook.ipynb."
    )
