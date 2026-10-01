# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The runtime's `tree_data` artifact checks, applied before any offline tool trusts the bytes.

`TreeDataAdapter::loadFromBuffer` refuses an artifact whose declared digest differs, whose
file identifier is not `HGBM`, which the FlatBuffers verifier rejects, or whose trees fail
`prepareTrees`. An offline evaluator or promote that decodes the same file with fewer
checks reports numbers for -- or installs -- a model the engine will never use, and the
engine's only trace of that is one log line before it silently ranks by static order.
"""
from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

#: `TreeDataAdapter::load` reads at most this many bytes (and refuses an empty file).
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
#: `loadFromBuffer`'s floor: a root offset and a file identifier.
_MIN_ARTIFACT_BYTES = 4 + 4
GBDT_MODEL_IDENTIFIER = b"HGBM"

# What a malformed buffer raises from the generated Python accessors, which have no
# verifier: an offset past the end, a vector length the buffer cannot hold, a string that is
# not UTF-8. Each is the runtime verifier's refusal, reached a different way.
_DECODE_ERRORS = (
    struct.error,
    IndexError,
    ValueError,
    TypeError,
    UnicodeDecodeError,
    OverflowError,
)


def artifact_digest(path: Path) -> str:
    """`tree_data.hash` as the runtime compares it: bare lowercase hex SHA-256 of the file.

    `sha256(buffer, size)` in Sha256.hpp returns no `sha256:` prefix, so neither does this;
    a prefixed value would refuse every artifact it guards.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_trees(trees, num_features: int, where: str) -> None:
    """`TreeDataAdapter::prepareTrees`, check for check, with its reasons."""
    for index, tree in enumerate(trees or []):

        def reject(reason: str):
            raise ValueError(f"{where} tree {index}: {reason}")

        if (
            tree is None
            or tree.leftChildren is None
            or len(tree.leftChildren) == 0
            or tree.rightChildren is None
            or tree.featureIndices is None
            or tree.thresholds is None
            or tree.leafValues is None
        ):
            reject("missing nodes or a required node array")
        count = len(tree.leftChildren)
        if (
            len(tree.rightChildren) != count
            or len(tree.featureIndices) != count
            or len(tree.thresholds) != count
        ):
            reject("node-parallel arrays have different lengths")
        incoming = [0] * count
        children: list[tuple[int, int] | None] = []
        for node in range(count):
            left = int(tree.leftChildren[node])
            if left == -1:
                if node >= len(tree.leafValues):
                    reject("leaf has no prediction")
                if not math.isfinite(float(tree.leafValues[node])):
                    reject("leaf prediction is not finite")
                children.append(None)
                continue
            right = int(tree.rightChildren[node])
            if left < 0 or right < 0 or left >= count or right >= count:
                reject("child index outside the tree")
            if not math.isfinite(float(tree.thresholds[node])):
                reject("split threshold is not finite")
            feature = int(tree.featureIndices[node])
            if feature < 0 or feature >= num_features:
                reject("split feature outside the declared feature count")
            incoming[left] += 1
            incoming[right] += 1
            children.append((left, right))
        # Kahn's algorithm over every node, reached or not, as the runtime does.
        ready = [node for node in range(count) if incoming[node] == 0]
        position = 0
        while position < len(ready):
            pair = children[ready[position]]
            position += 1
            for child in pair or ():
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
        if len(ready) != count:
            reject("cycle in child indices")


def verify_tree_artifact(path: Path, declared_hash: str | None) -> bytes:
    """The bytes of a `tree_data` artifact the runtime would load, or ValueError saying why not.

    Same order as `TreeDataAdapter::loadFromBuffer`: size, declared digest, identifier,
    structure. The features-hash comparison is left to the caller, which holds the
    descriptor's signature digest.
    """
    path = Path(path)
    data = path.read_bytes()
    if not data or len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"{path}: artifact size {len(data)} is outside (0, 256 MiB]")
    if len(data) < _MIN_ARTIFACT_BYTES:
        raise ValueError(f"{path}: artifact is too short to be a FlatBuffer")
    if declared_hash is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual != declared_hash:
            raise ValueError(
                f"{path}: model hash mismatch - declared {declared_hash!r}, actual {actual!r}"
            )
    if data[4:8] != GBDT_MODEL_IDENTIFIER:
        raise ValueError(
            f"{path}: file identifier {bytes(data[4:8])!r} is not {GBDT_MODEL_IDENTIFIER!r}"
        )

    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT

    try:
        model = GbdtModelT.InitFromPackedBuf(bytearray(data), 0)
    except _DECODE_ERRORS as error:
        raise ValueError(
            f"{path}: artifact does not decode as a GbdtModel: {error}"
        ) from error
    if model.numFeatures < 0:
        raise ValueError(f"{path}: negative feature count")
    if not math.isfinite(model.baseScore):
        raise ValueError(f"{path}: base score is not finite")
    _check_trees(model.trees, model.numFeatures, str(path))
    if model.groups:
        if not 0 <= model.groupByFeatureIndex < model.numFeatures:
            raise ValueError(
                f"{path}: grouped model's group_by_feature_index {model.groupByFeatureIndex} "
                f"is outside the declared feature count {model.numFeatures}"
            )
        for index, group in enumerate(model.groups):
            if group is None:
                raise ValueError(f"{path}: null group")
            _check_trees(group.trees, model.numFeatures, f"{path} group {index}")
    return data


def is_grouped_tree(data: bytes) -> bool:
    """Whether a verified artifact is two-layer: the runtime's `groups() && !groups()->empty()`."""
    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModel

    return GbdtModel.GetRootAs(bytearray(data), 0).GroupsLength() > 0
