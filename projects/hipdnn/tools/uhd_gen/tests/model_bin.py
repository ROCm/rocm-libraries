# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`model.bin` in tests: writing a GbdtModelT, and reading how a written one stores its trees."""
from __future__ import annotations

from pathlib import Path


def packed(model) -> bytes:
    """A GbdtModelT as `model.bin` bytes, with the file identifier the runtime checks."""
    import flatbuffers

    from uhd_gen.artifact import GBDT_MODEL_IDENTIFIER

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=GBDT_MODEL_IDENTIFIER)
    return bytes(builder.Output())


def leaf_model(
    features_hash: str,
    num_features: int,
    *,
    leaf: float = 1.0,
    grouped: bool = False,
    training_arches: list[str] | None = None,
) -> bytes:
    """The smallest `tree_data` artifact the runtime would load: one tree of one leaf."""
    from uhd_gen.lgbm_to_flatbuffer import build_gbdt_model

    ensemble = {
        "tree_info": [{"tree_structure": {"leaf_value": leaf}}],
        "max_feature_idx": num_features - 1,
        "objective": "regression",
    }
    return build_gbdt_model(
        ensemble,
        features_hash,
        training_arches=training_arches,
        group_by_feature_index=0 if grouped else -1,
        groups=[(0.0, ensemble)] if grouped else None,
    )


def write_model(model, path: Path) -> Path:
    """`packed(model)` written to `path`; returns `path`."""
    path.write_bytes(packed(model))
    return path


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
