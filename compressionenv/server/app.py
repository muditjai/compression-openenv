# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Compressionenv Environment.

This module creates an HTTP server that exposes the CompressionenvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from ..models import CompressionenvAction, CompressionenvObservation
from .compressionenv_environment import CompressionenvEnvironment


# Create the app with web interface and README integration
app = create_app(
    CompressionenvEnvironment,
    CompressionenvAction,
    CompressionenvObservation,
    env_name="compressionenv",
    max_concurrent_envs=4,  # allow 4 concurrent WebSocket sessions for parallel workers
)


def _show_port_processes(port: int = 8000) -> None:
    """Show all processes attached to port."""
    import subprocess

    for cmd in [["lsof", "-i", f":{port}"], ["ss", "-tlnp", f"sport = :{port}"]]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                print(f"Processes on port {port}:\n{r.stdout.strip()}\n")
                return
        except FileNotFoundError:
            continue
    print(f"No processes found on port {port} (or lsof/ss not available)")


def _kill_port(port: int = 8000) -> None:
    """Kill any existing process on port."""
    import os
    import signal
    import subprocess

    for cmd in [["lsof", "-ti", f"tcp:{port}"], ["fuser", "-t", f"{port}/tcp"]]:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            continue
        for tok in out.split():
            try:
                pid = int(tok)
                os.kill(pid, signal.SIGKILL)
                print(f"Killed process {pid} on port {port}")
            except (ValueError, ProcessLookupError):
                pass
        return


def main(host: str = "0.0.0.0", port: int = 8000, kill_existing: bool = False):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m compressionenv.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
        kill_existing: If True, kill any process on port before starting

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn compressionenv.server.app:app --workers 4
    """
    import time
    import uvicorn

    if kill_existing:
        _show_port_processes(port)
        _kill_port(port)
        time.sleep(1)

    uvicorn.run(app, host=host, port=port)

    if kill_existing:
        time.sleep(1)
        _show_port_processes(port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--kill-existing", action="store_true", help="Kill process on port before starting")
    args = parser.parse_args()
    main(port=args.port, kill_existing=args.kill_existing)
