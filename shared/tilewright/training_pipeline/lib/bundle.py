# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The files of a stage05 bundle. A round that retrains nothing carries the
previous round's bundle forward: its stage05 folder holds a copy of the
model and `carried_forward.json`, which names the stage05 folder the bundle
came from. Rounds carried before every file of a bundle was copied lack
some (`metrics.json` is never copied), so a reader follows the chain."""
from __future__ import annotations

import json
from pathlib import Path


def bundle_file(train_dir: Path, name: str) -> Path:
    """The file `name` of the bundle in `train_dir`: its own, or for a
    bundle a round carried forward without it the one of the stage05
    folder it came from, through every carry (a path that does not exist
    when none holds it)."""
    d, seen = Path(train_dir), set()
    while not (d / name).is_file() and d not in seen:
        seen.add(d)
        try:
            source = json.loads((d / "carried_forward.json").read_text())["source"]
        except (OSError, ValueError, KeyError, TypeError):
            break
        d = Path(str(source))
    return d / name
