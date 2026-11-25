"""
Plans the hipBLASLt Python bindings' Component CI lanes.

Outputs two JSON matrices via GITHUB_OUTPUT, one entry per lane:
  - linux_gpu: gfx942 always; gfx950 with the ci:run-gfx950 label or on a manual run
  - windows: the CPU lane always; gfx1151 when all lanes are asked for

GPU runner labels come from ROCm/therock-ci-config's runner-config-v2.json, since they can
change without notice. A GPU lane names its device wheel; the CPU lane leaves it empty.

Usage:
  python hipblaslt_python_lanes.py --config ci-config/runner-config-v2.json

Environment: REPO, PR (empty outside a pull request), EVENT (github.event_name),
ALL_LANES ("true" or "false"), WINDOWS_RUNNER.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci_utils import set_github_output

GFX950_LABEL = "ci:run-gfx950"


def gpu_runner(config: Mapping[str, Any], family: str, os_name: str) -> str:
    """The test runner label for a GPU family on one OS, from runner-config-v2.json."""
    try:
        label = config["gpu_runner_labels"][family][os_name]["test-runs-on"]
    except KeyError as e:
        raise SystemExit(
            f"runner-config-v2.json has no {family}/{os_name} test runner"
        ) from e
    if not isinstance(label, str) or not label:
        raise SystemExit(
            f"runner-config-v2.json's {family}/{os_name} test runner is empty"
        )
    return label


def plan_lanes(
    labels: Sequence[str],
    event: str,
    all_lanes: bool,
    windows_runner: str,
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """The Linux GPU and Windows matrices for one run."""
    linux_gpu = [
        {
            "name": "gfx942",
            "runner": gpu_runner(config, "gfx94x", "linux"),
            "device": "gfx942",
        }
    ]
    # gfx950 machines are scarce: only the label or a manual run asks for one, not a change to
    # the lanes.
    if event == "workflow_dispatch" or GFX950_LABEL in labels:
        linux_gpu.append(
            {
                "name": f"gfx950 ({GFX950_LABEL})",
                "runner": gpu_runner(config, "gfx950", "linux"),
                "device": "gfx950",
            }
        )
    windows = [
        {
            "name": "CPU (non-blocking)",
            "runner": windows_runner,
            "device": "",
            "require_gpu": "",
        }
    ]
    if all_lanes:
        windows.append(
            {
                "name": "gfx1151 (opt-in)",
                "runner": gpu_runner(config, "gfx1151", "windows"),
                "device": "gfx1151",
                "require_gpu": "1",
            }
        )
    return {"linux_gpu": linux_gpu, "windows": windows}


def pr_labels(repo: str, pr: str) -> list[str]:
    """The pull request's labels now, not from the event, so a re-run after adding one sees it."""
    if not pr:
        return []
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/issues/{pr}/labels", "--jq", ".[].name"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.split()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to therock-ci-config's runner-config-v2.json",
    )
    args = parser.parse_args()
    env = os.environ
    lanes = plan_lanes(
        labels=pr_labels(env.get("REPO", ""), env.get("PR", "")),
        event=env.get("EVENT", ""),
        all_lanes=env.get("ALL_LANES", "false") == "true",
        windows_runner=env["WINDOWS_RUNNER"],
        config=json.loads(args.config.read_text(encoding="utf-8")),
    )
    for name, matrix in lanes.items():
        print(
            name
            + ": "
            + ", ".join(f"{lane['name']} on {lane['runner']}" for lane in matrix)
        )
    set_github_output({name: json.dumps(matrix) for name, matrix in lanes.items()})


if __name__ == "__main__":
    main()
