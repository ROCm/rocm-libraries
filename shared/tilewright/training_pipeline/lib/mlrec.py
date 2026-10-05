# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Model file format (MLREC) writer, reader and v1 -> v2 converter.

MLREC_v2 (what the engine loads; all little-endian):

    char[8]  "MLREC_v2"
    u32      0x01020304
    u32      header_size          bytes before the first payload byte
    u64      payload_size
    u32      payload_crc32        CRC-32/ISO-HDLC of the payload
    u8       weight_dtype         0 fp32, 1 bf16, 2 int8, 3 int4
    u8[3]    0
    u32      q_dim, i_dim, x_dim
    u32      n_cells, n_splits
    u16+str  feature_catalog_hash
    u16+str  arch
    f64      parallel_mi_cu, bw_c0, bw_c1, bw_c2, mi_default_cycles
    u32      n_mi, then n_mi x {u32 mi_m, mi_n, mi_k; i32 dtype; f64 cycles}
             sorted by (mi_m, mi_n, mi_k, dtype)
    payload  n_splits split records, then n_cells cell records
    char[8]  "MLRECEND"

MLREC_v1 had `u32 version | u32 endian | u8 weight_dtype | u32+str hash |
u32+str arch | u32 dims[3] | u32 n_cells, n_splits` before the same payload
and an optional trailer. Payload records:

    split  u16+str parent | char axis, '\\0' | i32 threshold (value <=
           threshold goes to lo) | u16+str lo | u16+str hi
    cell   u16+str label | u32 embed, hidden, inter | f32 temperature |
           f32 q_mean[q], q_std[q], i_mean[i], i_std[i], x_mean[x], x_std[x] |
           u32 n_sig | n_sig x i32[8] | the 14 tensors of WEIGHT_ORDER

Weight matrices use the weight dtype (bf16: u16 holding the top half of the
fp32 bits, round to nearest even; int8: f32 scale + i8[n], value = scale*q;
int4: f32 scale + ceil(n/2) bytes, nibble = q + 8 with the even index in the
low nibble, value = scale*(nibble - 8)); biases, statistics and the
temperature are always f32.

`write_model` and `convert_v1_to_v2` refuse (ValueError) anything the
engine's loader (shared/tilewright/src/tilewright/format.cpp) rejects: labels
and arch that are not 1..65535 printable ASCII characters without spaces,
layer widths outside 1..65536, non-finite values (also after bf16 rounding),
int8/int4 scales the loader bounds, non-positive or non-finite constants and
MI cycles, MI dtypes outside the DataType range or fnuz, more than 1024 or
duplicate MI entries, duplicate cells or split parents, cyclic split trees
and feature dimensions or catalog hash other than the catalog's.
"""
from __future__ import annotations

import math
import re
import struct
import zlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import features as fs
from .hardware import DATATYPE_NAME, DATATYPE_VALUE, ArchConstants
from .subcells import ROUTING_AXES, SplitRule, split_tree_from_labels

MAGIC_V1 = b"MLREC_v1"
MAGIC_V2 = b"MLREC_v2"
TRAILER = b"MLRECEND"
ENDIAN_MARKER = 0x01020304

MAX_LAYER_DIM = 1 << 16
MAX_MI_ENTRIES = 1024
FNUZ_DATATYPES = frozenset(
    DATATYPE_VALUE[n]
    for n in ("float8_fnuz", "bfloat8_fnuz", "float8bfloat8_fnuz", "bfloat8float8_fnuz")
)
_FLT_MAX = float(np.finfo(np.float32).max)
# Largest |level| the loader multiplies an int8 / int4 scale by.
_QUANT_LEVEL_BOUND = {2: 128.0, 3: 8.0}
_LABEL_RE = re.compile(r"[\x21-\x7e]{1,65535}")

WEIGHT_DTYPES: Dict[str, int] = {"fp32": 0, "bf16": 1, "int8": 2, "int4": 3}
WEIGHT_DTYPE_NAMES: Dict[int, str] = {v: k for k, v in WEIGHT_DTYPES.items()}

WEIGHT_ORDER: Tuple[str, ...] = (
    "q_proj.0.weight",
    "q_proj.0.bias",
    "q_proj.2.weight",
    "q_proj.2.bias",
    "q_proj.4.weight",
    "q_proj.4.bias",
    "i_proj.0.weight",
    "i_proj.0.bias",
    "i_proj.2.weight",
    "i_proj.2.bias",
    "inter_mlp.0.weight",
    "inter_mlp.0.bias",
    "inter_mlp.2.weight",
    "inter_mlp.2.bias",
)

NORM_KEYS: Tuple[str, ...] = ("q_mean", "q_std", "i_mean", "i_std", "x_mean", "x_std")

_V2_FIXED = struct.Struct("<8sIIQIB3xIIIII")
_V2_RESERVED = slice(29, 32)
_V1_FIXED = struct.Struct("<8sIIB")
_MI_ENTRY = struct.Struct("<IIIid")


class FormatError(ValueError):
    """Malformed or unsupported model file."""


def weight_dtype_code(weight_dtype: Union[str, int]) -> int:
    if isinstance(weight_dtype, str):
        code = WEIGHT_DTYPES.get(weight_dtype.strip().lower())
    else:
        code = int(weight_dtype) if int(weight_dtype) in WEIGHT_DTYPE_NAMES else None
    if code is None:
        raise ValueError(
            f"unknown weight dtype {weight_dtype!r}; expected one of "
            f"{sorted(WEIGHT_DTYPES)}"
        )
    return code


def tensor_shapes(
    q_dim: int, i_dim: int, x_dim: int, embed: int, hidden: int, inter: int
) -> Dict[str, Tuple[int, ...]]:
    """Shape of every WEIGHT_ORDER tensor of one cell."""
    return {
        "q_proj.0.weight": (hidden, q_dim),
        "q_proj.0.bias": (hidden,),
        "q_proj.2.weight": (hidden, hidden),
        "q_proj.2.bias": (hidden,),
        "q_proj.4.weight": (embed, hidden),
        "q_proj.4.bias": (embed,),
        "i_proj.0.weight": (hidden, i_dim),
        "i_proj.0.bias": (hidden,),
        "i_proj.2.weight": (embed, hidden),
        "i_proj.2.bias": (embed,),
        "inter_mlp.0.weight": (inter, x_dim),
        "inter_mlp.0.bias": (inter,),
        "inter_mlp.2.weight": (1, inter),
        "inter_mlp.2.bias": (1,),
    }


# ── tensor encoding ──────────────────────────────────────────────────────────


def _as_f32(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(t, dtype=np.float32))


def bf16_bits(a32: np.ndarray) -> np.ndarray:
    """Top 16 bits of each fp32 value, rounded to nearest even (the rounding
    torch's float32 -> bfloat16 conversion uses for finite values)."""
    u = np.ascontiguousarray(a32, dtype=np.float32).reshape(-1).view(np.uint32)
    u = u.astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _scale_f32(amax: float, levels: float, weight_dtype: int) -> float:
    """Quantization scale as the f32 the file stores; ValueError when the
    loader would reject it."""
    scale = float(np.float32((amax / levels) if amax > 0 else 1.0))
    if not math.isfinite(scale) or abs(scale) * _QUANT_LEVEL_BOUND[
        weight_dtype
    ] > float(_FLT_MAX):
        raise ValueError(f"quantization scale {scale!r} is out of the engine's range")
    return scale


def encode_weight(t: Any, weight_dtype: int) -> bytes:
    """Encode one weight matrix (row-major) in `weight_dtype`. Raises
    ValueError for values the engine would reject."""
    a32 = _as_f32(t).reshape(-1)
    _require_finite(a32, "weight")
    if weight_dtype == WEIGHT_DTYPES["fp32"]:
        return a32.tobytes(order="C")
    if weight_dtype == WEIGHT_DTYPES["bf16"]:
        bits = bf16_bits(a32)
        if np.any((bits & 0x7F80) == 0x7F80):
            raise ValueError("weight rounds to a non-finite bf16 value")
        return bits.tobytes(order="C")
    if weight_dtype == WEIGHT_DTYPES["int8"]:
        amax = float(np.abs(a32).max()) if a32.size else 0.0
        scale = _scale_f32(amax, 127.0, weight_dtype)
        q = np.clip(np.round(a32 / scale), -127, 127).astype(np.int8)
        return struct.pack("<f", scale) + q.tobytes(order="C")
    if weight_dtype == WEIGHT_DTYPES["int4"]:
        amax = float(np.abs(a32).max()) if a32.size else 0.0
        scale = _scale_f32(amax, 7.0, weight_dtype)
        q = np.clip(np.round(a32 / scale), -7, 7).astype(np.int32) + 8
        n = q.size
        packed = np.zeros((n + 1) // 2, dtype=np.uint8)
        packed[: (n + 1) // 2] = q[0::2].astype(np.uint8)
        if n > 1:
            hi = q[1::2].astype(np.uint8)
            packed[: hi.size] |= hi << 4
        return struct.pack("<f", scale) + packed.tobytes(order="C")
    raise ValueError(f"unknown weight dtype code {weight_dtype}")


def encoded_weight_size(n: int, weight_dtype: int) -> int:
    if weight_dtype == WEIGHT_DTYPES["fp32"]:
        return 4 * n
    if weight_dtype == WEIGHT_DTYPES["bf16"]:
        return 2 * n
    if weight_dtype == WEIGHT_DTYPES["int8"]:
        return 4 + n
    if weight_dtype == WEIGHT_DTYPES["int4"]:
        return 4 + (n + 1) // 2
    raise ValueError(f"unknown weight dtype code {weight_dtype}")


def decode_weight(buf: bytes, n: int, weight_dtype: int) -> np.ndarray:
    """fp32 values of an encoded weight matrix, computed like the engine."""
    if weight_dtype == WEIGHT_DTYPES["fp32"]:
        return np.frombuffer(buf, dtype="<f4", count=n).astype(np.float32)
    if weight_dtype == WEIGHT_DTYPES["bf16"]:
        u16 = np.frombuffer(buf, dtype="<u2", count=n).astype(np.uint32)
        return (u16 << 16).view(np.float32)
    scale = np.float32(struct.unpack_from("<f", buf, 0)[0])
    if weight_dtype == WEIGHT_DTYPES["int8"]:
        q = np.frombuffer(buf, dtype=np.int8, count=n, offset=4)
        return (scale * q.astype(np.float32)).astype(np.float32)
    if weight_dtype == WEIGHT_DTYPES["int4"]:
        packed = np.frombuffer(buf, dtype=np.uint8, count=(n + 1) // 2, offset=4)
        nib = np.empty(n, dtype=np.int32)
        nib[0::2] = packed[: (n + 1) // 2] & 0x0F
        nib[1::2] = packed[: n // 2] >> 4
        return (scale * (nib - 8).astype(np.float32)).astype(np.float32)
    raise ValueError(f"unknown weight dtype code {weight_dtype}")


# ── writer ───────────────────────────────────────────────────────────────────


def _lenstr(s: str, width: str = "<H") -> bytes:
    b = str(s).encode("ascii")
    if len(b) > (0xFFFF if width == "<H" else 0xFFFFFFFF):
        raise ValueError(f"string too long ({len(b)} bytes): {s[:40]!r}...")
    return struct.pack(width, len(b)) + b


def check_label(text: Any, what: str) -> str:
    """`text` when the engine accepts it as a label or arch: 1..65535
    printable ASCII characters without spaces; else ValueError."""
    s = str(text)
    if not _LABEL_RE.fullmatch(s):
        raise ValueError(
            f"{what} {s[:60]!r} must be 1 to 65535 printable ASCII characters "
            f"without spaces"
        )
    return s


def _label(text: Any, what: str) -> bytes:
    return _lenstr(check_label(text, what))


def _check_dims(label: str, dims: Iterable[int]) -> None:
    for d in dims:
        if not 1 <= int(d) <= MAX_LAYER_DIM:
            raise ValueError(f"{label}: layer width {d} is outside 1..{MAX_LAYER_DIM}")


def _check_acyclic(rules: Sequence[SplitRule]) -> None:
    children = {r.cell: (r.lo_label, r.hi_label) for r in rules}
    state: Dict[str, int] = {}
    for root in children:
        if state.get(root):
            continue
        state[root] = 1
        stack = [(root, iter(children[root]))]
        while stack:
            node, it = stack[-1]
            child = next(it, None)
            if child is None:
                state[node] = 2
                stack.pop()
                continue
            if state.get(child) == 1:
                raise ValueError(f"split tree has a cycle through {child!r}")
            if not state.get(child):
                state[child] = 1
                stack.append((child, iter(children.get(child, ()))))


def _require_finite(values: np.ndarray, what: str) -> None:
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite value in {what}")


def _temperature(state_dict: Mapping[str, Any]) -> float:
    t = state_dict.get("temperature")
    if t is None:
        return 1.0
    a = _as_f32(t).reshape(-1)
    return float(a[0]) if a.size > 0 else 1.0


def _split_bytes(rule: SplitRule) -> bytes:
    if rule.axis not in ROUTING_AXES:
        raise ValueError(f"unsupported split axis {rule.axis!r} for {rule.cell!r}")
    if rule.lo_label == rule.hi_label or rule.cell in (rule.lo_label, rule.hi_label):
        raise ValueError(f"degenerate split rule for {rule.cell!r}")
    if not -(2**31) <= int(rule.threshold) < 2**31:
        raise ValueError(f"split threshold of {rule.cell!r} does not fit an i32")
    return (
        _label(rule.cell, "split parent")
        + struct.pack("<cc", rule.axis.encode("ascii"), b"\x00")
        + struct.pack("<i", int(rule.threshold))
        + _label(rule.lo_label, "split lo label")
        + _label(rule.hi_label, "split hi label")
    )


def _cell_bytes(
    label: str,
    entry: Mapping[str, Any],
    dims: Tuple[int, int, int],
    weight_dtype: int,
) -> bytes:
    q_dim, i_dim, x_dim = dims
    embed = int(entry["embed_dim"])
    hidden = int(entry["hidden_dim"])
    inter = int(entry["inter_hidden"])
    _check_dims(label, (embed, hidden, inter))
    sd = entry["state_dict"]
    norms = entry.get("feature_norms") or {}
    defaults = {"q": q_dim, "i": i_dim, "x": x_dim}
    out: List[bytes] = [
        _label(label, "cell label"),
        struct.pack("<III", embed, hidden, inter),
    ]
    temperature = np.asarray([_temperature(sd)], dtype=np.float32)
    _require_finite(temperature, f"{label}: temperature")
    out.append(temperature.tobytes())
    for key in NORM_KEYS:
        n = defaults[key[0]]
        fill = 1.0 if key.endswith("_std") else 0.0
        vals = np.asarray(norms.get(key, [fill] * n), dtype=np.float32).reshape(-1)
        if vals.size != n:
            raise ValueError(f"{label}: {key} has {vals.size} values, expected {n}")
        _require_finite(vals, f"{label}: {key}")
        out.append(vals.astype("<f4").tobytes())
    sigs = [tuple(int(v) for v in s) for s in (entry.get("smart_k_signatures") or [])]
    out.append(struct.pack("<I", len(sigs)))
    for sig in sigs:
        if len(sig) != 8:
            raise ValueError(f"{label}: signature must have 8 ints: {sig!r}")
        out.append(struct.pack("<8i", *sig))
    shapes = tensor_shapes(q_dim, i_dim, x_dim, embed, hidden, inter)
    for name in WEIGHT_ORDER:
        if name not in sd:
            raise KeyError(f"{label}: missing tensor {name!r}")
        a32 = _as_f32(sd[name])
        if tuple(a32.shape) != shapes[name]:
            raise ValueError(
                f"{label}: {name} has shape {tuple(a32.shape)}, "
                f"expected {shapes[name]}"
            )
        _require_finite(a32, f"{label}: {name}")
        if name.endswith(".weight"):
            try:
                out.append(encode_weight(a32, weight_dtype))
            except ValueError as e:
                raise ValueError(f"{label}: {name}: {e}") from None
        else:
            out.append(a32.astype("<f4").tobytes(order="C"))
    return b"".join(out)


def _constants_tail(fhash: str, arch: str, constants: ArchConstants) -> bytes:
    scalars = [float(constants.parallel_mi_cu), *map(float, constants.bw)]
    scalars.append(float(constants.mi_default))
    if len(scalars) != 5 or not all(math.isfinite(v) for v in scalars):
        raise ValueError("arch constants must be 5 finite numbers")
    if scalars[0] <= 0:
        raise ValueError("parallel_mi_cu must be > 0")
    if scalars[4] <= 0:
        raise ValueError("mi_default cycles must be > 0")
    if len(constants.mi_table) > MAX_MI_ENTRIES:
        raise ValueError(f"MI table has more than {MAX_MI_ENTRIES} entries")
    entries = []
    for (m, n, k, dt), cycles in constants.mi_table.items():
        key = (m, n, k, dt)
        if dt not in DATATYPE_VALUE:
            raise ValueError(f"MI table dtype {dt!r} has no DataType value")
        if DATATYPE_VALUE[dt] in FNUZ_DATATYPES:
            raise ValueError(
                f"MI table entry {key}: the engine looks fnuz dtypes up under "
                f"their non-fnuz name ({str(dt)[: -len('_fnuz')]!r})"
            )
        if not all(0 <= int(v) < 2**32 for v in (m, n, k)):
            raise ValueError(f"MI table entry {key}: shape does not fit a u32")
        if not (math.isfinite(float(cycles)) and float(cycles) > 0):
            raise ValueError(f"MI table entry {key}: cycles must be finite and > 0")
        entries.append((int(m), int(n), int(k), DATATYPE_VALUE[dt], float(cycles)))
    if len({e[:4] for e in entries}) != len(entries):
        raise ValueError("MI table has duplicate entries")
    entries.sort(key=lambda e: e[:4])
    tail = [_lenstr(fhash), _label(arch, "arch"), struct.pack("<ddddd", *scalars)]
    tail.append(struct.pack("<I", len(entries)))
    tail.extend(_MI_ENTRY.pack(*e) for e in entries)
    return b"".join(tail)


def _assemble_v2(
    *,
    weight_dtype: int,
    fhash: str,
    arch: str,
    dims: Tuple[int, int, int],
    n_cells: int,
    n_splits: int,
    constants: ArchConstants,
    payload: bytes,
) -> bytes:
    tail = _constants_tail(fhash, arch, constants)
    header_size = _V2_FIXED.size + len(tail)
    head = _V2_FIXED.pack(
        MAGIC_V2,
        ENDIAN_MARKER,
        header_size,
        len(payload),
        zlib.crc32(payload) & 0xFFFFFFFF,
        weight_dtype,
        *dims,
        n_cells,
        n_splits,
    )
    return head + tail + payload + TRAILER


def write_model(
    bundle: Mapping[str, Any],
    splits: Optional[Union[Mapping[str, SplitRule], Iterable[SplitRule]]],
    arch: str,
    constants: ArchConstants,
    weight_dtype: Union[str, int],
) -> bytes:
    """MLREC_v2 bytes of a trained bundle (the `models.pt` dict: `models`
    label -> entry with state_dict / feature_norms / dims /
    smart_k_signatures, plus `q_names` / `i_names` / `x_names`).

    `splits` None rebuilds the tree from the cell labels, which is what a
    deployed model must carry. Cells and splits are written sorted by label.
    Raises ValueError when the bundle does not match the current feature
    catalog or holds values the engine would reject."""
    wdt = weight_dtype_code(weight_dtype)
    names = (
        list(bundle.get("q_names", fs.query_feature_names())),
        list(bundle.get("i_names", fs.item_feature_names())),
        list(bundle.get("x_names", fs.interaction_feature_names())),
    )
    expected = (
        fs.query_feature_names(),
        fs.item_feature_names(),
        fs.interaction_feature_names(),
    )
    if names[0] != expected[0] or names[1] != expected[1] or names[2] != expected[2]:
        raise ValueError(
            "bundle feature names differ from the current catalog "
            f"(hash {fs.feature_names_hash()}); the engine would reject the model"
        )
    dims = (len(names[0]), len(names[1]), len(names[2]))
    models: Mapping[str, Any] = bundle["models"]
    if not models:
        raise ValueError("bundle has no cells")
    if splits is None:
        rules = list(split_tree_from_labels(models.keys()).values())
    elif isinstance(splits, Mapping):
        rules = list(splits.values())
    else:
        rules = list(splits)
    rules.sort(key=lambda r: r.cell)
    parents = [r.cell for r in rules]
    if len(set(parents)) != len(parents):
        raise ValueError("duplicate split parents")
    _check_acyclic(rules)
    payload = b"".join(_split_bytes(r) for r in rules) + b"".join(
        _cell_bytes(label, models[label], dims, wdt) for label in sorted(models)
    )
    return _assemble_v2(
        weight_dtype=wdt,
        fhash=fs.feature_names_hash(),
        arch=str(arch),
        dims=dims,
        n_cells=len(models),
        n_splits=len(rules),
        constants=constants,
        payload=payload,
    )


# ── reader ───────────────────────────────────────────────────────────────────


class _Cursor:
    def __init__(self, data: bytes, pos: int, end: int) -> None:
        self.data, self.pos, self.end = data, pos, end

    def take(self, n: int, what: str) -> bytes:
        if n < 0 or self.pos + n > self.end:
            raise FormatError(f"truncated at {what} (offset {self.pos})")
        b = self.data[self.pos : self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt: str, what: str) -> Tuple[Any, ...]:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt), what))

    def lenstr(self, width: str, what: str) -> str:
        (n,) = self.unpack(width, what)
        raw = self.take(n, what)
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError as e:
            raise FormatError(f"non-ASCII {what}") from e

    def floats(self, n: int, what: str) -> np.ndarray:
        return np.frombuffer(self.take(4 * n, what), dtype="<f4").astype(np.float32)


def _parse_payload(
    cur: _Cursor,
    n_splits: int,
    n_cells: int,
    dims: Tuple[int, int, int],
    weight_dtype: int,
) -> Tuple[List[SplitRule], List[Dict[str, Any]]]:
    q_dim, i_dim, x_dim = dims
    splits: List[SplitRule] = []
    for s in range(n_splits):
        parent = cur.lenstr("<H", f"split {s} parent")
        axis_pair = cur.take(2, f"split {s} axis")
        axis = chr(axis_pair[0])
        if axis not in ROUTING_AXES or axis_pair[1] != 0:
            raise FormatError(f"split {s}: bad axis {axis_pair!r}")
        (thr,) = cur.unpack("<i", f"split {s} threshold")
        lo = cur.lenstr("<H", f"split {s} lo label")
        hi = cur.lenstr("<H", f"split {s} hi label")
        splits.append(
            SplitRule(cell=parent, axis=axis, threshold=thr, lo_label=lo, hi_label=hi)
        )
    cells: List[Dict[str, Any]] = []
    for c in range(n_cells):
        label = cur.lenstr("<H", f"cell {c} label")
        embed, hidden, inter = cur.unpack("<III", f"{label} dims")
        (temperature,) = cur.unpack("<f", f"{label} temperature")
        cell: Dict[str, Any] = {
            "label": label,
            "embed_dim": embed,
            "hidden_dim": hidden,
            "inter_hidden": inter,
            "temperature": float(temperature),
        }
        sizes = {"q": q_dim, "i": i_dim, "x": x_dim}
        cell["feature_norms"] = {
            key: cur.floats(sizes[key[0]], f"{label} {key}") for key in NORM_KEYS
        }
        (n_sig,) = cur.unpack("<I", f"{label} n_sig")
        if 32 * n_sig > cur.end - cur.pos:
            raise FormatError(f"{label}: signature count exceeds the file")
        sig_arr = np.frombuffer(cur.take(32 * n_sig, f"{label} signatures"), "<i4")
        cell["smart_k_signatures"] = [
            tuple(int(v) for v in row) for row in sig_arr.reshape(n_sig, 8)
        ]
        state: Dict[str, np.ndarray] = {}
        for name, shape in tensor_shapes(
            q_dim, i_dim, x_dim, embed, hidden, inter
        ).items():
            n = math.prod(shape)
            if name.endswith(".weight"):
                buf = cur.take(encoded_weight_size(n, weight_dtype), f"{label} {name}")
                state[name] = decode_weight(buf, n, weight_dtype).reshape(shape)
            else:
                state[name] = cur.floats(n, f"{label} {name}").reshape(shape)
        state["temperature"] = np.asarray(temperature, dtype=np.float32)
        cell["state_dict"] = state
        cells.append(cell)
    if cur.pos != cur.end:
        raise FormatError(f"{cur.end - cur.pos} unparsed payload bytes")
    return splits, cells


def _read_v2(data: bytes) -> Dict[str, Any]:
    if len(data) < _V2_FIXED.size + len(TRAILER):
        raise FormatError("file too short")
    (
        _magic,
        endian,
        header_size,
        payload_size,
        crc,
        wdt,
        q_dim,
        i_dim,
        x_dim,
        n_cells,
        n_splits,
    ) = _V2_FIXED.unpack_from(data, 0)
    if endian != ENDIAN_MARKER:
        raise FormatError("bad endian marker")
    if data[_V2_RESERVED] != b"\x00\x00\x00":
        raise FormatError("reserved header bytes are not zero")
    if wdt not in WEIGHT_DTYPE_NAMES:
        raise FormatError(f"bad weight dtype {wdt}")
    if header_size + payload_size + len(TRAILER) != len(data):
        raise FormatError("header/payload sizes do not match the file size")
    if data[-len(TRAILER) :] != TRAILER:
        raise FormatError("missing MLRECEND trailer")
    cur = _Cursor(data, _V2_FIXED.size, header_size)
    fhash = cur.lenstr("<H", "feature hash")
    arch = cur.lenstr("<H", "arch")
    pmc, c0, c1, c2, mi_default = cur.unpack("<ddddd", "arch constants")
    (n_mi,) = cur.unpack("<I", "n_mi")
    table: Dict[Tuple[int, int, int, str], float] = {}
    for e in range(n_mi):
        m, n, k, dt, cycles = cur.unpack(_MI_ENTRY.format, f"mi entry {e}")
        if dt not in DATATYPE_NAME:
            raise FormatError(f"mi entry {e}: unknown dtype {dt}")
        table[(m, n, k, DATATYPE_NAME[dt])] = cycles
    if cur.pos != header_size:
        raise FormatError("header_size does not match the header contents")
    payload = data[header_size : header_size + payload_size]
    if zlib.crc32(payload) & 0xFFFFFFFF != crc:
        raise FormatError("payload CRC mismatch")
    dims = (q_dim, i_dim, x_dim)
    splits, cells = _parse_payload(
        _Cursor(data, header_size, header_size + payload_size),
        n_splits,
        n_cells,
        dims,
        wdt,
    )
    return {
        "format": MAGIC_V2.decode(),
        "version": 2,
        "weight_dtype": wdt,
        "weight_dtype_name": WEIGHT_DTYPE_NAMES[wdt],
        "feature_catalog_hash": fhash,
        "arch": arch,
        "dims": dims,
        "constants": ArchConstants(
            parallel_mi_cu=pmc, bw=(c0, c1, c2), mi_table=table, mi_default=mi_default
        ),
        "splits": splits,
        "cells": cells,
        "header_size": header_size,
        "payload_crc32": crc,
        "has_trailer": True,
    }


def _v1_header(data: bytes) -> Dict[str, Any]:
    if len(data) < _V1_FIXED.size:
        raise FormatError("file too short")
    _magic, version, endian, wdt = _V1_FIXED.unpack_from(data, 0)
    if version != 1 or endian != ENDIAN_MARKER:
        raise FormatError("bad MLREC_v1 version or endian marker")
    if wdt not in WEIGHT_DTYPE_NAMES:
        raise FormatError(f"bad weight dtype {wdt}")
    has_trailer = data[-len(TRAILER) :] == TRAILER
    cur = _Cursor(data, _V1_FIXED.size, len(data))
    fhash = cur.lenstr("<I", "feature hash")
    arch = cur.lenstr("<I", "arch")
    q_dim, i_dim, x_dim, n_cells, n_splits = cur.unpack("<IIIII", "dims")
    return {
        "weight_dtype": wdt,
        "feature_catalog_hash": fhash,
        "arch": arch,
        "dims": (q_dim, i_dim, x_dim),
        "n_cells": n_cells,
        "n_splits": n_splits,
        "payload_start": cur.pos,
        "payload_end": len(data) - (len(TRAILER) if has_trailer else 0),
        "has_trailer": has_trailer,
    }


def _read_v1(data: bytes) -> Dict[str, Any]:
    h = _v1_header(data)
    splits, cells = _parse_payload(
        _Cursor(data, h["payload_start"], h["payload_end"]),
        h["n_splits"],
        h["n_cells"],
        h["dims"],
        h["weight_dtype"],
    )
    payload = data[h["payload_start"] : h["payload_end"]]
    return {
        "format": MAGIC_V1.decode(),
        "version": 1,
        "weight_dtype": h["weight_dtype"],
        "weight_dtype_name": WEIGHT_DTYPE_NAMES[h["weight_dtype"]],
        "feature_catalog_hash": h["feature_catalog_hash"],
        "arch": h["arch"],
        "dims": h["dims"],
        "constants": None,
        "splits": splits,
        "cells": cells,
        "header_size": h["payload_start"],
        "payload_crc32": zlib.crc32(payload) & 0xFFFFFFFF,
        "has_trailer": h["has_trailer"],
    }


def read_header(data: bytes) -> Dict[str, Any]:
    """Format version, weight dtype, hash, arch and counts of a model file
    without parsing the payload."""
    data = bytes(data[:4096]) if len(data) > 4096 else bytes(data)
    magic = data[:8]
    if magic == MAGIC_V2:
        if len(data) < _V2_FIXED.size:
            raise FormatError("file too short")
        fields = _V2_FIXED.unpack_from(data, 0)
        cur = _Cursor(data, _V2_FIXED.size, len(data))
        return {
            "version": 2,
            "weight_dtype": fields[5],
            "feature_catalog_hash": cur.lenstr("<H", "feature hash"),
            "arch": cur.lenstr("<H", "arch"),
            "dims": tuple(fields[6:9]),
            "n_cells": fields[9],
            "n_splits": fields[10],
        }
    if magic == MAGIC_V1:
        h = _v1_header(data)
        return {"version": 1} | {
            k: h[k]
            for k in (
                "weight_dtype",
                "feature_catalog_hash",
                "arch",
                "dims",
                "n_cells",
                "n_splits",
            )
        }
    raise FormatError(f"not an MLREC file (magic {magic!r})")


def read_model(data: bytes) -> Dict[str, Any]:
    """Parse an MLREC_v1 or MLREC_v2 file. Weight matrices come back as the
    fp32 values the engine computes from them. Raises FormatError."""
    data = bytes(data)
    magic = data[:8]
    if magic == MAGIC_V2:
        return _read_v2(data)
    if magic == MAGIC_V1:
        return _read_v1(data)
    raise FormatError(f"not an MLREC file (magic {magic!r})")


def bundle_from_model(parsed: Mapping[str, Any]) -> Dict[str, Any]:
    """A `write_model` bundle holding the decoded cells of `read_model`."""
    models = {}
    for c in parsed["cells"]:
        models[c["label"]] = {
            "state_dict": dict(c["state_dict"]),
            "feature_norms": {k: list(v) for k, v in c["feature_norms"].items()},
            "embed_dim": c["embed_dim"],
            "hidden_dim": c["hidden_dim"],
            "inter_hidden": c["inter_hidden"],
            "smart_k_signatures": [list(s) for s in c["smart_k_signatures"]],
        }
    return {
        "q_names": fs.query_feature_names(),
        "i_names": fs.item_feature_names(),
        "x_names": fs.interaction_feature_names(),
        "models": models,
    }


def _check_parsed_payload(
    splits: Sequence[SplitRule], cells: Sequence[Mapping[str, Any]]
) -> None:
    """The loader's payload rules on a parsed payload."""
    if not cells:
        raise ValueError("model has no cells")
    for rule in splits:
        _split_bytes(rule)
    if len({r.cell for r in splits}) != len(splits):
        raise ValueError("duplicate split parents")
    _check_acyclic(splits)
    labels = [check_label(c["label"], "cell label") for c in cells]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate cell labels")
    for c in cells:
        _check_dims(c["label"], (c["embed_dim"], c["hidden_dim"], c["inter_hidden"]))
        arrays = [np.float32([c["temperature"]]), *c["feature_norms"].values()]
        arrays += list(c["state_dict"].values())
        for a in arrays:
            _require_finite(np.asarray(a, dtype=np.float32), c["label"])


def convert_v1_to_v2(data: bytes, constants: ArchConstants) -> bytes:
    """MLREC_v2 file with the v1 payload byte for byte and `constants` (the
    ones the model was trained with) in the header. Raises ValueError when
    the engine would reject the result."""
    data = bytes(data)
    if data[:8] != MAGIC_V1:
        raise FormatError(f"not an MLREC_v1 file (magic {data[:8]!r})")
    parsed = _read_v1(data)
    h = _v1_header(data)
    catalog = (
        len(fs.query_feature_names()),
        len(fs.item_feature_names()),
        len(fs.interaction_feature_names()),
    )
    if tuple(h["dims"]) != catalog:
        raise ValueError(
            f"feature dims {h['dims']} differ from the catalog's {catalog}"
        )
    if h["feature_catalog_hash"] != fs.feature_names_hash():
        raise ValueError(
            f"feature catalog hash {h['feature_catalog_hash']!r} differs from the "
            f"catalog's {fs.feature_names_hash()!r}"
        )
    _check_parsed_payload(parsed["splits"], parsed["cells"])
    return _assemble_v2(
        weight_dtype=h["weight_dtype"],
        fhash=h["feature_catalog_hash"],
        arch=h["arch"],
        dims=h["dims"],
        n_cells=h["n_cells"],
        n_splits=h["n_splits"],
        constants=constants,
        payload=data[h["payload_start"] : h["payload_end"]],
    )


__all__: Sequence[str] = (
    "FormatError",
    "WEIGHT_DTYPES",
    "WEIGHT_ORDER",
    "bundle_from_model",
    "check_label",
    "convert_v1_to_v2",
    "decode_weight",
    "encode_weight",
    "read_model",
    "weight_dtype_code",
    "write_model",
)
