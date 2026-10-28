# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How a written `model.bin` stores its trees, for tests of the flags that choose it."""
from __future__ import annotations

from pathlib import Path

#: The flags of `train` and `generate`, and the (encoding, leaf type) each writes.
ENCODING_FLAGS = [
    ([], ("split_leaf", "float32")),
    (["--leaf-precision", "float64"], ("split_leaf", "float64")),
    (["--model-encoding", "node_arrays"], ("node_arrays", None)),
]


def encoding_of(path: Path) -> tuple[str, str | None]:
    """("node_arrays", None), or ("split_leaf", "float32" | "float64") from its leaves.

    Every tree, of every layer, must be stored the same way.
    """
    import uhd_gen  # noqa: F401  puts _generated/ on sys.path
    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT
    from hipdnn_flatbuffers_sdk.data_objects.RealArray import RealArray

    model = GbdtModelT.InitFromPackedBuf(bytearray(Path(path).read_bytes()), 0)
    trees = list(model.trees or [])
    for group in model.groups or []:
        trees += list(group.trees or [])
    names = {
        RealArray.NONE: ("node_arrays", None),
        RealArray.Float32Array: ("split_leaf", "float32"),
        RealArray.Float64Array: ("split_leaf", "float64"),
    }
    stored = {names[tree.leavesType] for tree in trees}
    assert len(stored) == 1, stored
    return stored.pop()
