# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Worker that interacts with the Compressionenv server (one uvicorn, multiple sessions)."""

import argparse
import json
import multiprocessing
import os
from typing import Any, Dict, List, Optional

from .client import CompressionenvEnv
from .models import CompressionenvAction


# Trivial identity codec used for trajectory collection
IDENTITY_ACTION = CompressionenvAction(
    compression_code="def compress(text: str) -> bytes:\n    return text.encode('utf-8')",
    decompression_code="def decompress(data: bytes) -> str:\n    return data.decode('utf-8')",
    algo_name="identity",
)


OBSERVATION_EXCLUDE = {"essay_text"}


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    """Serialize observation, excluding bulky fields already available in essays/."""
    return obs.model_dump(exclude=OBSERVATION_EXCLUDE)


def _run_one_trajectory(worker_id: int, base_url: str) -> Dict[str, Any]:
    """
    Run one episode until done.
    Returns a dict with essay_id (look up text in essays/) and a list of steps.
    """
    steps: List[Dict[str, Any]] = []
    with CompressionenvEnv(base_url=base_url) as client:
        result = client.reset()
        essay_id = result.observation.essay_id
        steps.append({
            "step": 0,
            "observation": _obs_to_dict(result.observation),
            "action": None,
            "reward": getattr(result, "reward", 0.0),
            "done": getattr(result.observation, "done", False),
        })
        step_index = 1
        while not steps[-1]["done"]:
            result = client.step(IDENTITY_ACTION)
            steps.append({
                "step": step_index,
                "observation": _obs_to_dict(result.observation),
                "action": IDENTITY_ACTION.model_dump(),
                "reward": result.reward,
                "done": result.done,
            })
            step_index += 1
    return {"essay_id": essay_id, "steps": steps}


def _print_baselines(essay_id: str, baselines: Dict[str, int], best_baseline: Optional[int]) -> None:
    """Print baseline compression sizes for an essay."""
    print(f"essay_id: {essay_id}  baselines: {baselines}  best: {best_baseline} bytes")


def _print_step(step: Dict[str, Any]) -> None:
    """Print a single step summary."""
    o = step["observation"]
    print(
        f"  step={step['step']:>2}  valid={o.get('valid')}  "
        f"compressed={o.get('compressed_size_bytes')}  reward={step['reward']}  done={step['done']}"
    )


def _run_trajectory_llm(
    base_url: str,
    trajectory_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one episode with LLM-generated actions.
    Returns dict with essay_id and steps.
    """
    from .llm_agent import generate_compression_action

    steps: List[Dict[str, Any]] = []
    with CompressionenvEnv(base_url=base_url) as client:
        result = client.reset()
        steps.append({
            "step": 0,
            "observation": _obs_to_dict(result.observation),
            "action": None,
            "reward": 0.0,
            "done": False,
        })
        _print_baselines(
            steps[0]["observation"]["essay_id"],
            steps[0]["observation"].get("baselines_size_bytes", {}),
            steps[0]["observation"].get("best_baseline_size_bytes"),
        )
        _print_step(steps[-1])

        if trajectory_file:
            with open(trajectory_file, "w") as f:
                f.write(f"essay_id: {steps[0]['observation']['essay_id']}\n\n")

        step_index = 1
        while not steps[-1]["done"]:
            prev_obs = steps[-1]["observation"]
            essay_text = result.observation.essay_text
            action = generate_compression_action(prev_obs, step=step_index, essay_text=essay_text)
            result = client.step(action)
            steps.append({
                "step": step_index,
                "observation": _obs_to_dict(result.observation),
                "action": action.model_dump(),
                "reward": result.reward,
                "done": result.done,
            })
            if trajectory_file:
                with open(trajectory_file, "a") as f:
                    f.write(
                        f"=== step {step_index} "
                        f"(valid={result.observation.valid} "
                        f"compressed={result.observation.compressed_size_bytes} "
                        f"reward={result.reward}) ===\n"
                    )
                    f.write("--- compression_code ---\n")
                    f.write(action.compression_code)
                    f.write("\n--- decompression_code ---\n")
                    f.write(action.decompression_code)
                    f.write("\n\n")
            _print_step(steps[-1])
            step_index += 1

    return {"essay_id": steps[0]["observation"]["essay_id"], "steps": steps}


def run_worker_process(args: tuple[int, str]) -> tuple[int, Dict[str, Any]]:
    """Wrapper for multiprocessing: (worker_id, base_url) -> (worker_id, trajectory)."""
    worker_id, base_url = args
    return (worker_id, _run_one_trajectory(worker_id, base_url))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 4 workers in parallel against one server (port 8000); each gets its own session."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (all workers connect to this port)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host where the server is running",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output/trajectories.json",
        help="Output file path for trajectories JSON",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single worker (no multiprocessing, for testing)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-generated actions (Qwen 8B) instead of identity codec",
    )
    parser.add_argument(
        "--trajectory-file",
        type=str,
        default=None,
        help="Write step-by-step compression/decompression code to this file (for --llm)",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if args.llm:
        traj = _run_trajectory_llm(base_url, trajectory_file=args.trajectory_file)
        out_data = {"worker_0": traj}
        print(f"essay_id: {traj['essay_id']}, steps: {len(traj['steps'])}")
        if args.trajectory_file:
            print(f"\nTo download the trajectory file locally, run:")
            print(
                f"npx @northflank/cli download service file --projectId hackathon "
                f"--serviceId jupyter-pytorch --localPath {os.path.basename(args.trajectory_file)} "
                f"--remotePath {args.trajectory_file}"
            )
    elif args.single:
        traj = _run_one_trajectory(0, base_url)
        out_data = {"worker_0": traj}
    else:
        with multiprocessing.Pool(4) as pool:
            results = pool.map(
                run_worker_process,
                [(i, base_url) for i in range(4)],
            )
        out_data = {f"worker_{wid}": traj for wid, traj in results}

    output_path = args.output
    if output_path and not args.llm:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Wrote trajectories to {output_path}")
    elif not args.llm:
        print(json.dumps(out_data, indent=2))


if __name__ == "__main__":
    main()
