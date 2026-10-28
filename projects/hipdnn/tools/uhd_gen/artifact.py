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

import numpy as np

import uhd_gen  # noqa: F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.IntegerArray import IntegerArray
from hipdnn_flatbuffers_sdk.data_objects.RealArray import RealArray

#: `TreeDataAdapter::load` reads at most this many bytes (and refuses an empty file).
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
#: `loadFromBuffer`'s floor: a root offset and a file identifier.
_MIN_ARTIFACT_BYTES = 4 + 4
GBDT_MODEL_IDENTIFIER = b"HGBM"

#: The element type of each IntegerArray / RealArray union member, by tag (gbdt_model.fbs).
#: A tag missing here is one this reader, like the runtime, cannot decode.
INTEGER_ARRAY_DTYPES = {
    IntegerArray.Int8Array: np.int8,
    IntegerArray.UInt8Array: np.uint8,
    IntegerArray.Int16Array: np.int16,
    IntegerArray.UInt16Array: np.uint16,
    IntegerArray.Int32Array: np.int32,
}
REAL_ARRAY_DTYPES = {
    RealArray.Float32Array: np.float32,
    RealArray.Float64Array: np.float64,
}
#: Element width of each member, by tag.
INTEGER_ARRAY_BYTES = {t: np.dtype(d).itemsize for t, d in INTEGER_ARRAY_DTYPES.items()}
REAL_ARRAY_BYTES = {t: np.dtype(d).itemsize for t, d in REAL_ARRAY_DTYPES.items()}


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

    def _typed_array(self, at: int, vtable: int, field: int, widths, what: str) -> None:
        """A union field (type at `field`, value at `field + 2`), then `Verify<Union>`: a known
        member is a table of one `values` vector; NONE or an unknown tag verifies, as the
        generated code does, and is refused by the decoder instead."""
        self._scalar_field(at, vtable, field, 1, f"{what}_type")
        value = self._pointer_field(at, vtable, field + 2, what)
        offset = self._field(at, vtable, field)
        tag = self._data[at + offset] if offset else 0
        if value is None or tag not in widths:
            return
        table = self._table(value, what)
        where = f"{what}.values"
        self._vector(self._pointer_field(value, table, 4, where), widths[tag], where)
        self._depth -= 1

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
        for field, name in (
            (18, "split_features"),
            (22, "split_thresholds"),
            (26, "split_left"),
            (30, "split_right"),
        ):
            self._typed_array(at, vtable, field, INTEGER_ARRAY_BYTES, f"{what}.{name}")
        for field, name in ((34, "split_default_left"), (36, "split_decision_lte")):
            where = f"{what}.{name}"
            self._vector(self._pointer_field(at, vtable, field, where), 1, where)
        self._scalar_field(at, vtable, 38, 1, f"{what}.default_left_fill")
        self._typed_array(at, vtable, 40, REAL_ARRAY_BYTES, f"{what}.leaves")
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
        self._typed_array(at, vtable, 30, REAL_ARRAY_BYTES, "threshold_table")
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


def _values(array):
    """A decoded typed array's values, or None when the member or its values are absent."""
    return None if array is None or array.values is None else array.values


def _decoded(tag: int, array, widths, reject, name: str):
    """The runtime's decodeIntegers / decodeReals: the values, or a rejection naming why."""
    if tag == 0:  # NONE in either union
        reject(f"{name} is absent")
    if tag not in widths:
        reject(
            f"{name} is stored as {'integer' if widths is INTEGER_ARRAY_BYTES else 'real'} "
            f"array type {tag}, which this runtime cannot read"
        )
    values = _values(array)
    if values is None:
        reject(f"{name} has no values")
    return values


def _length(tag: int, array) -> int:
    values = _values(array) if tag in REAL_ARRAY_BYTES else None
    return 0 if values is None else len(values)


def _check_split(threshold: float, feature: int, num_features: int, reject) -> None:
    """Either encoding's split, once its children are in range, as `prepareTrees` checks it."""
    if not math.isfinite(threshold):
        reject("split threshold is not finite")
    if feature < 0 or feature >= num_features:
        reject("split feature outside the declared feature count")


def _check_leaf(value: float, reject) -> None:
    """Either encoding's leaf, as `prepareTrees` checks it."""
    if not math.isfinite(value):
        reject("leaf prediction is not finite")


def _split_leaf_arrays(tree, table, num_features: int, reject):
    """A split/leaf tree as node arrays (splits first, then leaves; -1 children mark a leaf),
    checked as `prepareTrees` checks it."""
    leaves = _decoded(tree.leavesType, tree.leaves, REAL_ARRAY_BYTES, reject, "leaves")
    if len(leaves) == 0:
        reject("leaves are empty")
    splits = len(leaves) - 1
    if splits > 0:
        features, thresholds, lefts, rights = (
            _decoded(tag, array, INTEGER_ARRAY_BYTES, reject, name)
            for tag, array, name in (
                (tree.splitFeaturesType, tree.splitFeatures, "split_features"),
                (tree.splitThresholdsType, tree.splitThresholds, "split_thresholds"),
                (tree.splitLeftType, tree.splitLeft, "split_left"),
                (tree.splitRightType, tree.splitRight, "split_right"),
            )
        )
    elif any(
        tag != IntegerArray.NONE
        for tag in (
            tree.splitFeaturesType,
            tree.splitThresholdsType,
            tree.splitLeftType,
            tree.splitRightType,
        )
    ):
        reject("a single-leaf tree has split arrays")
    default_left = tree.splitDefaultLeft
    decision_lte = tree.splitDecisionLte
    # The object API reads an absent bool vector as None and a present empty one as [].
    if (
        splits > 0
        and any(len(a) != splits for a in (features, thresholds, lefts, rights))
    ) or any(
        flags is not None and len(flags) != splits
        for flags in (default_left, decision_lte)
    ):
        reject(
            "split arrays do not hold one entry per split (one fewer than the leaves)"
        )

    count = splits + len(leaves)
    feature_out = np.full(count, -1, dtype=np.int64)
    threshold_out = np.zeros(count, dtype=np.float64)
    left_out = np.full(count, -1, dtype=np.int64)
    right_out = np.full(count, -1, dtype=np.int64)
    leaf_out = np.zeros(count, dtype=np.float64)
    default_out = np.zeros(count, dtype=bool)
    lte_out = np.ones(count, dtype=bool)

    def node(child: int) -> int:
        child = int(child)
        if child >= 0:
            if child >= splits:
                reject("child index outside the tree")
            return child
        if ~child >= len(leaves):
            reject("child index outside the tree")
        return splits + ~child

    for i in range(splits):
        left_out[i] = node(lefts[i])
        right_out[i] = node(rights[i])
        index = int(thresholds[i])
        if index < 0 or index >= len(table):
            reject("split threshold index outside the threshold table")
        threshold = float(table[index])
        feature = int(features[i])
        _check_split(threshold, feature, num_features, reject)
        feature_out[i] = feature
        threshold_out[i] = threshold
        default_out[i] = (
            bool(default_left[i])
            if default_left is not None
            else bool(tree.defaultLeftFill)
        )
        lte_out[i] = True if decision_lte is None else bool(decision_lte[i])
    for j, leaf in enumerate(leaves):
        _check_leaf(float(leaf), reject)
        leaf_out[splits + j] = float(leaf)
    return (
        feature_out,
        threshold_out,
        left_out,
        right_out,
        leaf_out,
        default_out,
        lte_out,
    )


def _node_arrays(tree, reject):
    """A node-array tree as checked by `prepareTrees`, with routing flags read as it does."""
    if (
        tree.rightChildren is None
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
    return count


def tree_arrays(tree, table, num_features: int, where: str = "tree"):
    """Node arrays of either encoding, as the runtime decodes and checks them:
    (feature, threshold, left, right, leaf, default_left, decision_lte), one entry per node,
    node 0 the root, -1 children marking a leaf. ValueError names what the runtime refuses.
    """

    def reject(reason: str):
        raise ValueError(f"{where}: {reason}")

    if tree.leavesType != RealArray.NONE:
        return _split_leaf_arrays(tree, table, num_features, reject)
    count = _node_arrays(tree, reject)
    feature = np.asarray(tree.featureIndices, dtype=np.int64)
    threshold = np.asarray(tree.thresholds, dtype=np.float64)
    left = np.asarray(tree.leftChildren, dtype=np.int64)
    right = np.asarray(tree.rightChildren, dtype=np.int64)
    leaf_values = np.asarray(tree.leafValues, dtype=np.float64)
    leaf = np.zeros(count, dtype=np.float64)
    leaf[: min(count, len(leaf_values))] = leaf_values[:count]

    def routing(values, absent: bool) -> np.ndarray:
        """Supplied entries, then `absent`, as `prepareTrees` reads a short flag vector."""
        flags = np.full(count, absent, dtype=bool)
        supplied = np.asarray([] if values is None else values, dtype=bool)[:count]
        flags[: len(supplied)] = supplied
        return flags

    lte = tree.decisionLte
    for node in range(count):
        if int(left[node]) == -1:
            if node >= len(leaf_values):
                reject("leaf has no prediction")
            _check_leaf(float(leaf_values[node]), reject)
            continue
        if (
            left[node] < 0
            or right[node] < 0
            or left[node] >= count
            or right[node] >= count
        ):
            reject("child index outside the tree")
        _check_split(float(threshold[node]), int(feature[node]), num_features, reject)
    return (
        feature,
        threshold,
        left,
        right,
        leaf,
        routing(tree.defaultLeft, False),
        # Absent or empty `decision_lte` is `<=` everywhere; a nonempty short one is `<`.
        routing(lte, lte is None or len(lte) == 0),
    )


def threshold_table(model, where: str) -> np.ndarray:
    """The model's split threshold table as the runtime decodes it (empty when absent)."""
    if model.thresholdTableType == RealArray.NONE:
        return np.zeros(0, dtype=np.float64)

    def reject(reason: str):
        raise ValueError(f"{where}: {reason}")

    values = _decoded(
        model.thresholdTableType,
        model.thresholdTable,
        REAL_ARRAY_BYTES,
        reject,
        "threshold_table",
    )
    return np.asarray(values, dtype=np.float64)


def _check_trees(
    trees, table, num_features: int, where: str, budget: int, max_bytes: int
) -> int:
    """`TreeDataAdapter::prepareTrees`, check for check, with its reasons; returns `budget`
    plus the bytes these trees' nodes claim. Every layer shares one node store, so
    `max_bytes` bounds them all.
    """
    trees = list(trees or [])
    for index, tree in enumerate(trees):

        def reject(reason: str):
            raise ValueError(f"{where} tree {index}: {reason}")

        if tree is None:
            reject("missing nodes or a required node array")
        if tree.leavesType != RealArray.NONE:
            leaf_bytes = REAL_ARRAY_BYTES.get(tree.leavesType, 0)
            child_bytes = INTEGER_ARRAY_BYTES.get(tree.splitLeftType, 0)
            if leaf_bytes == 0:
                reject("leaves are stored in a type this runtime cannot read")
            leaves = _length(tree.leavesType, tree.leaves)
            splits = max(leaves - 1, 0)
            if splits > 0 and child_bytes == 0:
                reject("split children are stored in a type this runtime cannot read")
            count, used = leaves + splits, leaves * leaf_bytes + splits * child_bytes
        else:
            if tree.leftChildren is None or len(tree.leftChildren) == 0:
                reject("missing nodes or a required node array")
            count = len(tree.leftChildren)
            used = 4 * count
        if count == 0:
            reject("missing nodes or a required node array")
        # Entries may share arrays, so the count is bounded by the buffer, not by them.
        if used > max_bytes - budget:
            reject("ensemble has more nodes than the model buffer can hold")
        budget += used

    for index, tree in enumerate(trees):
        feature, threshold, left, right, *_ = tree_arrays(
            tree, table, num_features, f"{where} tree {index}"
        )
        count = len(left)
        incoming = [0] * count
        children: list[tuple[int, int] | None] = []
        for node in range(count):
            if int(left[node]) < 0:
                children.append(None)
                continue
            incoming[int(left[node])] += 1
            incoming[int(right[node])] += 1
            children.append((int(left[node]), int(right[node])))
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
            raise ValueError(f"{where} tree {index}: cycle in child indices")
    return budget


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
    table = threshold_table(model, str(path))
    # Each distinct node needs its child entry in the buffer, as TreeDataAdapter bounds it.
    max_bytes = min(len(data), (2**32 - 1) * 4)
    total = _check_trees(model.trees, table, model.numFeatures, str(path), 0, max_bytes)
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
            total = _check_trees(
                group.trees,
                table,
                model.numFeatures,
                f"{path} group {index}",
                total,
                max_bytes,
            )
    if feature_count is not None:
        verify_feature_count(model.numFeatures, feature_count, path)
    return data


def is_grouped_tree(data: bytes) -> bool:
    """Whether a verified artifact is two-layer: the runtime's `groups() && !groups()->empty()`."""
    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModel

    return GbdtModel.GetRootAs(bytearray(data), 0).GroupsLength() > 0
