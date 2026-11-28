# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write build-runners and gpu-runners step outputs from the therock-ci-config checkout."""

import json
import os
import random
import sys

sys.path.insert(0, "ci-config")
from ci_config_api import load_config


def pick(runners: list[dict]) -> str:
    """Pick a label at random, as TheRock's select_weighted_label does.

    Entries are weighted by count (number of runners) when they have one,
    otherwise by weight.
    """
    key = "count" if "count" in runners[0] else "weight"
    return random.choices(
        [runner["label"] for runner in runners],
        weights=[runner[key] for runner in runners],
    )[0]


def resolve(entries: dict) -> dict:
    """Pick one label from each runner list; keep strings and skip other values."""
    return {
        key: pick(value) if isinstance(value, list) else value
        for key, value in entries.items()
        if isinstance(value, (list, str))
    }


config = load_config()
build_runners = {
    platform: resolve(pools) for platform, pools in config.build_runners.items()
}
gpu_runners = {
    family: {platform: resolve(entries) for platform, entries in platforms.items()}
    for family, platforms in config.gpu_runner_labels.items()
}
print(
    json.dumps({"build-runners": build_runners, "gpu-runners": gpu_runners}, indent=2)
)
with open(os.environ["GITHUB_OUTPUT"], "a") as output:
    output.write(f"build-runners={json.dumps(build_runners)}\n")
    output.write(f"gpu-runners={json.dumps(gpu_runners)}\n")
