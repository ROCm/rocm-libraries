# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The runtime's `tree_data` artifact checks, for offline tools that read the same bytes.

Mirrors `TreeDataAdapter::loadFromBuffer` and the feature-count admission, so an offline
tool never reports on or installs a model the engine would refuse.
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


class _Unverifiable(ValueError):
    """The FlatBuffers verifier's refusal, with the check that failed."""


class _GbdtModelVerifier:
    """`flatbuffers::Verifier` with its default options, running `VerifyGbdtModelBuffer`.

    The generated Python accessors do no bounds checks, so this ports verifier.h and the
    generated `Verify` methods check for check. Positions are offsets from buffer start,
    as in C++ (`p - buf_`), so alignment matches.
    """

    _MAX_DEPTH = 64
    _MAX_TABLES = 1_000_000
    #: FLATBUFFERS_MAX_BUFFER_SIZE: the largest soffset_t.
    _MAX_SIZE = 0x7FFFFFFF
    #: FLATBUFFERS_MIN_BUFFER_SIZE: root offset, vtable offset and two vtable fields.
    _MIN_SIZE = 4 + 4 + 2 + 2

    def __init__(self, data: bytes):
        self._data = data
        self._size = len(data)
        self._depth = 0
        self._tables = 0

    @staticmethod
    def _check(ok: bool, what: str) -> None:
        if not ok:
            raise _Unverifiable(what)

    def _range(self, at: int, length: int, what: str) -> None:
        # `Verify(elem, elem_len)`; C++ positions are size_t, so negative is out of range.
        self._check(
            0 <= at and length < self._size and at <= self._size - length,
            f"{what} at {at} (+{length}) lies outside the {self._size}-byte buffer",
        )

    def _scalar(self, at: int, width: int, what: str) -> None:
        # `Verify<T>(elem)`: aligned to its own width, then in range.
        self._check(at % width == 0, f"{what} at {at} is not {width}-byte aligned")
        self._range(at, width, what)

    def _u32(self, at: int) -> int:
        return struct.unpack_from("<I", self._data, at)[0]

    def _u16(self, at: int) -> int:
        return struct.unpack_from("<H", self._data, at)[0]

    def _offset(self, at: int, what: str) -> int:
        """`VerifyOffset<uoffset_t>`: the position a verified uoffset at `at` points to."""
        self._scalar(at, 4, what)
        offset = self._u32(at)
        self._check(offset != 0, f"{what} at {at} points to itself")
        self._check(offset <= self._MAX_SIZE, f"{what} at {at} wraps around")
        self._range(at + offset, 1, what)
        return at + offset

    def _table(self, at: int, what: str) -> int:
        """`VerifyTableStart`: the verified vtable position of the table at `at`."""
        self._scalar(at, 4, f"{what} vtable offset")
        vtable = at - struct.unpack_from("<i", self._data, at)[0]
        self._depth += 1
        self._tables += 1
        self._check(
            self._depth <= self._MAX_DEPTH and self._tables <= self._MAX_TABLES,
            f"{what} exceeds the verifier's depth or table-count limit",
        )
        self._scalar(vtable, 2, f"{what} vtable")
        size = self._u16(vtable)
        self._check(size % 2 == 0, f"{what} vtable size {size} is odd")
        self._range(vtable, size, f"{what} vtable")
        return vtable

    def _field(self, at: int, vtable: int, field: int) -> int:
        """`GetOptionalFieldOffset`: 0 when the field is absent."""
        return self._u16(vtable + field) if field < self._u16(vtable) else 0

    def _scalar_field(
        self, at: int, vtable: int, field: int, width: int, what: str
    ) -> None:
        """`Table::VerifyField<T>(verifier, field, sizeof(T))`."""
        offset = self._field(at, vtable, field)
        if offset:
            self._scalar(at + offset, width, what)

    def _pointer_field(self, at: int, vtable: int, field: int, what: str):
        """`Table::VerifyOffset` then the accessor: the referenced position, or None."""
        offset = self._field(at, vtable, field)
        return self._offset(at + offset, what) if offset else None

    def _vector(self, at: int | None, element: int, what: str) -> int:
        """`VerifyVectorOrString`: the verified element count (0 when absent)."""
        if at is None:
            return 0
        self._scalar(at, 4, f"{what} length")
        count = self._u32(at)
        self._check(
            count < self._MAX_SIZE // element, f"{what} length {count} overflows"
        )
        self._range(at, 4 + element * count, what)
        return count

    def _string(self, at: int | None, what: str) -> None:
        """`VerifyString`: a verified byte vector followed by a NUL terminator."""
        if at is None:
            return
        end = at + 4 + self._vector(at, 1, what)
        self._range(end, 1, f"{what} terminator")
        self._check(self._data[end] == 0, f"{what} is not NUL-terminated")

    def _elements(self, at: int | None, count: int):
        """`Vector<Offset<T>>::Get(i)` for each element: where it points (unverified)."""
        for index in range(count):
            slot = at + 4 + 4 * index
            yield slot + self._u32(slot)

    def _tree(self, at: int, what: str) -> None:
        vtable = self._table(at, what)
        for field, width, name in (
            (4, 4, "feature_indices"),
            (6, 8, "thresholds"),
            (8, 4, "left_children"),
            (10, 4, "right_children"),
            (12, 8, "leaf_values"),
            (14, 1, "default_left"),
            (16, 1, "decision_lte"),
        ):
            where = f"{what}.{name}"
            self._vector(self._pointer_field(at, vtable, field, where), width, where)
        self._depth -= 1

    def _trees(self, at: int, vtable: int, field: int, what: str) -> None:
        trees = self._pointer_field(at, vtable, field, what)
        count = self._vector(trees, 4, what)
        for index, tree in enumerate(self._elements(trees, count)):
            self._tree(tree, f"{what}[{index}]")

    def _group(self, at: int, what: str) -> None:
        vtable = self._table(at, what)
        self._scalar_field(at, vtable, 4, 8, f"{what}.value")
        self._trees(at, vtable, 6, f"{what}.trees")
        self._depth -= 1

    def _model(self, at: int) -> None:
        vtable = self._table(at, "model")
        self._trees(at, vtable, 4, "trees")
        self._scalar_field(at, vtable, 6, 4, "num_features")
        self._string(
            self._pointer_field(at, vtable, 8, "features_hash"), "features_hash"
        )
        self._scalar_field(at, vtable, 10, 8, "base_score")
        self._scalar_field(at, vtable, 12, 8, "learning_rate")
        self._string(self._pointer_field(at, vtable, 14, "framework"), "framework")
        self._string(
            self._pointer_field(at, vtable, 16, "training_date"), "training_date"
        )
        self._scalar_field(at, vtable, 18, 8, "num_training_samples")
        self._string(
            self._pointer_field(at, vtable, 20, "training_objective"),
            "training_objective",
        )
        arches = self._pointer_field(at, vtable, 22, "training_arches")
        count = self._vector(arches, 4, "training_arches")
        for index, arch in enumerate(self._elements(arches, count)):
            self._string(arch, f"training_arches[{index}]")
        self._string(
            self._pointer_field(at, vtable, 24, "model_version"), "model_version"
        )
        self._scalar_field(at, vtable, 26, 4, "group_by_feature_index")
        groups = self._pointer_field(at, vtable, 28, "groups")
        count = self._vector(groups, 4, "groups")
        for index, group in enumerate(self._elements(groups, count)):
            self._group(group, f"groups[{index}]")
        self._depth -= 1

    def verify(self) -> None:
        """`VerifyBufferFromStart<GbdtModel>("HGBM", 0)`, or _Unverifiable naming the check."""
        self._check(
            self._size >= self._MIN_SIZE,
            f"{self._size} bytes is below the smallest FlatBuffer",
        )
        self._check(
            self._data[4:8] == GBDT_MODEL_IDENTIFIER,
            f"file identifier {bytes(self._data[4:8])!r} is not {GBDT_MODEL_IDENTIFIER!r}",
        )
        self._model(self._offset(0, "root offset"))


def verify_gbdt_buffer(data: bytes) -> None:
    """Raise ValueError unless `VerifyGbdtModelBuffer` would accept `data`."""
    try:
        _GbdtModelVerifier(data).verify()
    except _Unverifiable as error:
        raise ValueError(
            f"the FlatBuffers verifier refuses the artifact: {error}"
        ) from None


def verify_feature_count(num_features: int, signature_length: int, where) -> None:
    """The runtime's arity admission: `num_features` must equal the signature length.

    A matching features_hash does not prove this; the engine checks it separately.
    """
    if num_features != signature_length:
        raise ValueError(
            f"{where}: artifact num_features {num_features} differs from the "
            f"{signature_length}-slot features_signature; the engine refuses the model"
        )


def artifact_digest(path: Path) -> str:
    """`tree_data.hash` as the runtime compares it: bare lowercase hex SHA-256, no prefix."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_contained_relative_path(payload: str) -> bool:
    """Whether `payload` names a file inside the descriptor's directory, as UhdParser reads it.

    Portable names are '/'-separated on every platform: no leading '/', no '\\', ':' or
    NUL, and no '..' segment anywhere. The final segment must name a file (neither empty
    nor '.'); './x' and 'a//b' are admitted.
    """
    if (
        not payload
        or payload.startswith("/")
        or any(character in payload for character in "\\:\0")
    ):
        return False
    segments = payload.split("/")
    return ".." not in segments and segments[-1] not in ("", ".")


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


def verify_tree_artifact(
    path: Path, declared_hash: str | None, *, feature_count: int | None = None
) -> bytes:
    """The bytes of a `tree_data` artifact the runtime would load, or ValueError saying why not.

    Checks run in `TreeDataAdapter::loadFromBuffer` order. The caller compares the features
    hash; `feature_count`, when given, must equal the artifact's `num_features`.
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
    try:
        verify_gbdt_buffer(data)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None

    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT

    # Verified, so every accessor below reads inside the buffer.
    model = GbdtModelT.InitFromPackedBuf(bytearray(data), 0)
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
        values = set()
        for index, group in enumerate(model.groups):
            if group is None:
                raise ValueError(f"{path}: null group")
            # A set of floats treats 0.0 and -0.0 as one value, as the runtime's std::set does.
            value = float(group.value)
            if not math.isfinite(value) or value in values:
                raise ValueError(
                    f"{path}: group {index} value {value!r} must be finite and unique"
                )
            values.add(value)
            _check_trees(group.trees, model.numFeatures, f"{path} group {index}")
    if feature_count is not None:
        verify_feature_count(model.numFeatures, feature_count, path)
    return data


def is_grouped_tree(data: bytes) -> bool:
    """Whether a verified artifact is two-layer: the runtime's `groups() && !groups()->empty()`."""
    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModel

    return GbdtModel.GetRootAs(bytearray(data), 0).GroupsLength() > 0
