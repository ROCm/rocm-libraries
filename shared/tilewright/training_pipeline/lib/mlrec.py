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
    u8       serving_flags        bit 0: served without the non-temporal rule
    u16      cascade_keys         0, or the view keys of a tier without a whitelist
                                  that get the full score after the embedding term
    u32      q_dim, i_dim, x_dim
    u32      n_cells, n_splits
    u16+str  feature_catalog_hash
    u16+str  arch
    f64      parallel_mi_cu, bw_c0, bw_c1, bw_c2, mi_default_cycles
    u32      n_mi, then n_mi x {u32 mi_m, mi_n, mi_k; i32 dtype; f64 cycles}
             sorted by (mi_m, mi_n, mi_k, dtype)
    payload  n_splits split records, then n_cells cell records
    char[8]  "MLRECEND"

MLREC_v3 keeps the fixed header (magic "MLREC_v3", bytes 29-31 zero) and
replaces the header tail with:

    u16+str  feature_catalog_hash  features.v3_catalog_hash()
    u64      feature_spec_hash     feature_spec_hash(spec)
    u16+str  arch
    f64      parallel_mi_cu, mi_default_cycles
    u32      n_mi, then the MI table as in v2
    u32      serving_flags         bit 0 small-batched one-tile rule, bit 1 Dot2
                                   M<3 rule, bit 2 non-temporal rule (soft rules,
                                   applied when set), bit 3 no whitelist tier,
                                   bit 4 launch keys (the pool is grouped by
                                   the launch model's launch_key), bit 5
                                   family-wise non-temporal rule (only with
                                   bit 2; lib/feasibility.py), bit 6 trained
                                   on the table plus the bank, bit 7
                                   trained-key gate, bit 8 CU floor, bit 9
                                   alignment veto (never with bit 2)
    u32      cascade_keys
    f64      min_cu_fraction, max_cu_fraction
                                   usable CUs / device CUs the model supports,
                                   0 < min <= max <= 1
    u8       schedules             bit per Schedule (default, dynamic, auto)
    u8       clip                  1: cells carry raw feature bounds
    u16      n_attributes, then per entry u16+str name | i64 default |
             u8 transform (ATTRIBUTE_TRANSFORMS index) | onehot only:
             u16 n_values, i64 values[n_values]
    with bit 7 only: u32 n_trained_keys (>= 1), u64 view_key_hash[n] strictly
             ascending: the view keys with training data; a pool's other keys
             are scored only for a problem none of them suits
    with bit 8 only: u32 min_effective_cus (>= 1): the fewest usable CUs the
             model supports on any device, whatever its CU count
    with bit 9 only: u32 veto_k_min (>= 1), u32 veto_align_bits (a power of
             two, >= 8): the alignment veto (lib/feasibility.py
             `alignment_vetoed`), a serving rule stage06 adds at deploy
    optional provenance, present when header bytes remain:
    u16      n_provenance (1..5), then per entry u16+str key | u16+str value,
             each key of PROVENANCE_KEYS at most once

and its cell records add, after x_std, the bounds when clip is set (f32
q_lo[q], q_hi[q], i_lo[i], i_hi[i], x_lo[x], x_hi[x]; +-inf allowed) and,
after the signatures, `u32 n_view_keys | u64 view_key_hash[n_view_keys]`.
i_dim is the base item dim plus the spec's features.

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
from .hardware import DATATYPE_NAME, DATATYPE_VALUE, ArchConstants, DeviceHardware
from .subcells import ROUTING_AXES, SplitRule, split_tree_from_labels

MAGIC_V1 = b"MLREC_v1"
MAGIC_V2 = b"MLREC_v2"
MAGIC_V3 = b"MLREC_v3"
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

_V2_FIXED = struct.Struct("<8sIIQIBBHIIIII")
NO_NON_TEMPORAL_RULE = 0x01
_SERVING_FLAGS_KNOWN = NO_NON_TEMPORAL_RULE
V3_SMALL_BATCHED_RULE = 0x01
V3_DOT2_RULE = 0x02
V3_NON_TEMPORAL_RULE = 0x04
V3_NO_WHITELIST_TIER = 0x08
V3_LAUNCH_KEYS = 0x10
V3_NON_TEMPORAL_FAMILY_RULE = 0x20
V3_BANK = 0x40
V3_TRAINED_KEY_GATE = 0x80
V3_CU_FLOOR = 0x100
V3_ALIGNMENT_VETO = 0x200
_V3_SERVING_FLAGS_KNOWN = 0x3FF
SCHEDULE_BITS: Dict[str, int] = {"default": 1, "dynamic": 2, "auto": 4}
BOUND_KEYS: Tuple[str, ...] = ("q_lo", "q_hi", "i_lo", "i_hi", "x_lo", "x_hi")
# Where an MLREC_v3 model came from: the training config's id, a hash of the
# training data, the revision of the pipeline's sources, the training date
# (ISO 8601) and the id of the launch model whose plans it trained with (the
# engine refuses to rank with another). Written in this order.
PROVENANCE_KEYS: Tuple[str, ...] = (
    "config_id",
    "training_data_hash",
    "source_revision",
    "date",
    "launch_model",
)
_ISO_DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?"
)
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


def _bound_bytes(
    label: str, entry: Mapping[str, Any], dims: Tuple[int, int, int]
) -> bytes:
    bounds = entry.get("feature_bounds") or {}
    sizes = {"q": dims[0], "i": dims[1], "x": dims[2]}
    out = []
    for key in BOUND_KEYS:
        n = sizes[key[0]]
        fill = -math.inf if key.endswith("_lo") else math.inf
        vals = np.asarray(bounds.get(key, [fill] * n), dtype=np.float32).reshape(-1)
        if vals.size != n:
            raise ValueError(f"{label}: {key} has {vals.size} values, expected {n}")
        if np.any(np.isnan(vals)):
            raise ValueError(f"{label}: NaN in {key}")
        out.append(vals.astype("<f4").tobytes())
    for side in "qix":
        lo = np.frombuffer(out[BOUND_KEYS.index(f"{side}_lo")], "<f4")
        hi = np.frombuffer(out[BOUND_KEYS.index(f"{side}_hi")], "<f4")
        if np.any(lo > hi):
            raise ValueError(f"{label}: a {side} lower bound exceeds its upper bound")
    return b"".join(out)


def _cell_bytes(
    label: str,
    entry: Mapping[str, Any],
    dims: Tuple[int, int, int],
    weight_dtype: int,
    version: int = 2,
    clip: bool = False,
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
    if version == 3 and clip:
        out.append(_bound_bytes(label, entry, dims))
    sigs = [tuple(int(v) for v in s) for s in (entry.get("smart_k_signatures") or [])]
    out.append(struct.pack("<I", len(sigs)))
    for sig in sigs:
        if len(sig) != 8:
            raise ValueError(f"{label}: signature must have 8 ints: {sig!r}")
        out.append(struct.pack("<8i", *sig))
    if version == 3:
        keys = sorted(
            {int(h) & 0xFFFFFFFFFFFFFFFF for h in entry.get("view_key_hashes") or []}
        )
        out.append(struct.pack("<I", len(keys)))
        out.extend(struct.pack("<Q", h) for h in keys)
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


def _mi_table_bytes(constants: ArchConstants) -> bytes:
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
    return struct.pack("<I", len(entries)) + b"".join(
        _MI_ENTRY.pack(*e) for e in entries
    )


def _constants_tail(fhash: str, arch: str, constants: ArchConstants) -> bytes:
    scalars = [float(constants.parallel_mi_cu), *map(float, constants.bw)]
    scalars.append(float(constants.mi_default))
    if len(scalars) != 5 or not all(math.isfinite(v) for v in scalars):
        raise ValueError("arch constants must be 5 finite numbers")
    if scalars[0] <= 0:
        raise ValueError("parallel_mi_cu must be > 0")
    if scalars[4] <= 0:
        raise ValueError("mi_default cycles must be > 0")
    tail = [_lenstr(fhash), _label(arch, "arch"), struct.pack("<ddddd", *scalars)]
    tail.append(_mi_table_bytes(constants))
    return b"".join(tail)


def _assemble_v2(
    *,
    weight_dtype: int,
    serving_flags: int = 0,
    cascade_keys: int = 0,
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
        serving_flags,
        cascade_keys,
        *dims,
        n_cells,
        n_splits,
    )
    return head + tail + payload + TRAILER


def _split_rules(
    models: Mapping[str, Any],
    splits: Optional[Union[Mapping[str, SplitRule], Iterable[SplitRule]]],
) -> List[SplitRule]:
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
    return rules


def v3_serving_flags(serving: Mapping[str, Any]) -> int:
    """MLREC_v3 serving flags of a bundle's `serving` (soft rules, launch
    keys, the bank, the trained-key gate and the alignment veto off and the
    whitelist tier on unless set). ValueError for the family-wise
    non-temporal rule without the non-temporal rule, and for the alignment
    veto with it."""
    flags = 0
    if serving.get("small_batched_rule", False):
        flags |= V3_SMALL_BATCHED_RULE
    if serving.get("dot2_rule", False):
        flags |= V3_DOT2_RULE
    if serving.get("non_temporal_rule", False):
        flags |= V3_NON_TEMPORAL_RULE
    if not serving.get("whitelist_tier", True):
        flags |= V3_NO_WHITELIST_TIER
    if serving.get("launch_keys", False):
        flags |= V3_LAUNCH_KEYS
    if serving.get("non_temporal_family_rule", False):
        if not flags & V3_NON_TEMPORAL_RULE:
            raise ValueError(
                "the family-wise non-temporal rule refines the non-temporal rule; "
                "serve with both"
            )
        flags |= V3_NON_TEMPORAL_FAMILY_RULE
    if serving.get("bank", False):
        flags |= V3_BANK
    if serving.get("trained_key_gate", False):
        flags |= V3_TRAINED_KEY_GATE
    if serving.get("alignment_veto"):
        if flags & V3_NON_TEMPORAL_RULE:
            raise ValueError(
                "the alignment veto is a rule of models served without the "
                "non-temporal rule"
            )
        flags |= V3_ALIGNMENT_VETO
    return flags


def trained_keys_bytes(keys: Iterable[int]) -> bytes:
    """The trained-key section of the view-key hashes `keys` (sorted and
    deduplicated). ValueError when there is none."""
    hashes = sorted({int(h) & 0xFFFFFFFFFFFFFFFF for h in keys})
    if not hashes:
        raise ValueError("the trained-key gate needs at least one trained key")
    return struct.pack(f"<I{len(hashes)}Q", len(hashes), *hashes)


def cu_floor_bytes(cus: Any) -> bytes:
    """The CU-floor section of a model that supports no fewer than `cus`
    usable CUs. ValueError unless `cus` is a whole number from 1 to
    2**32 - 1."""
    if (
        isinstance(cus, bool)
        or not isinstance(cus, (int, np.integer))
        or not 1 <= int(cus) < 2**32
    ):
        raise ValueError(
            f"the CU floor must be a whole number of CUs from 1 to 2**32 - 1, "
            f"got {cus!r}"
        )
    return struct.pack("<I", int(cus))


def alignment_veto_bytes(veto: Mapping[str, Any]) -> bytes:
    """The alignment-veto section of a serving `alignment_veto` ({k_min,
    align_bits}). ValueError unless k_min is a whole number from 1 to
    2**32 - 1 and align_bits a power of two from 8 to 2**31."""
    k_min, bits = veto.get("k_min"), veto.get("align_bits")
    for name, value in (("k_min", k_min), ("align_bits", bits)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"the alignment veto's {name} must be a whole number, got {value!r}"
            )
    if not 1 <= int(k_min) < 2**32:
        raise ValueError(
            f"the alignment veto's k_min {k_min} is not from 1 to 2**32 - 1"
        )
    bits = int(bits)
    if bits < 8 or bits & (bits - 1) or bits >= 2**32:
        raise ValueError(
            f"the alignment veto's line of {bits} bits is not a power of two "
            f"from 8 to 2**31"
        )
    return struct.pack("<II", int(k_min), bits)


def schedule_mask(schedules: Iterable[str]) -> int:
    mask = 0
    for name in schedules:
        if name not in SCHEDULE_BITS:
            raise ValueError(f"unknown schedule {name!r}")
        mask |= SCHEDULE_BITS[name]
    if mask == 0:
        raise ValueError("a model must support at least one schedule")
    return mask


# Fixed inputs whose v3 features `feature_spec_hash` covers, with fixed arch
# constants, launch plans and execution contexts: (problem, config,
# (n_cu, lds_bytes, l2_bytes), (cu_budget, schedule), launch plan). The engine
# holds the same cases (features.cpp).
_GOLDEN_CONSTANTS = ArchConstants(
    parallel_mi_cu=4.0,
    bw=(0.0, 0.0, 0.0),
    mi_table={
        (16, 16, 32, "bfloat16"): 16.0,
        (16, 16, 128, "float8"): 16.0,
        (16, 16, 4, "xfloat32"): 32.0,
    },
    mi_default=64.0,
)


def _golden_problem(m, n, k, batch, ta, tb, ab, cd, mi):
    return dict(
        m=m, n=n, k=k, batch=batch, a_transpose=ta, b_transpose=tb,
        a_dtype=ab, b_dtype=ab, c_dtype=cd, d_dtype=cd, mi_dtype=mi,
    )  # fmt: skip


def _golden_config(mt, mi, hints):
    return dict(
        mt_m=mt[0], mt_n=mt[1], mt_k=mt[2], mi_m=mi[0], mi_n=mi[1], mi_k=mi[2],
        cache_hints_a=hints[0], cache_hints_b=hints[1],
    )  # fmt: skip


_GOLDEN_CASES = (
    (
        _golden_problem(4096, 4096, 4096, 1, "T", "N", "bfloat16", "bfloat16", "bfloat16"),
        _golden_config((256, 256, 64), (16, 16, 32), (0, 0)),
        (256, 163840, 4194304), (0, "default"), fs.LaunchPlan(),
    ),
    (
        _golden_problem(777, 12345, 1000, 3, "N", "T", "float8", "bfloat16", "float8"),
        _golden_config((128, 64, 128), (16, 16, 128), (4, 0)),
        (256, 163840, 4194304), (64, "dynamic"), fs.LaunchPlan(True, 192, "parallel"),
    ),
    (
        _golden_problem(1, 9000, 16, 1, "N", "N", "half", "half", "half"),
        _golden_config((16, 64, 32), (1, 1, 64), (0, 4)),
        (80, 65536, 8388608), (0, "auto"), fs.LaunchPlan(True, 0, "none"),
    ),
    (
        _golden_problem(300, 200, 70000, 8, "T", "T", "float", "float", "xfloat32"),
        _golden_config((64, 256, 16), (16, 16, 4), (1, 1)),
        (304, 65536, 4194304), (100, "default"), fs.LaunchPlan(False, 304, "atomic"),
    ),
)  # fmt: skip


def feature_spec_hash(spec: fs.AttributeSpec) -> int:
    """The feature_spec_hash of an MLREC_v3 file with attribute spec `spec`:
    FNV-1a 64 of a text with the v3 catalog hash and version, the names of the
    query, item (with the spec's) and interaction features, the spec, and the
    float32 bits of the features of `_GOLDEN_CASES`. A change to any feature's
    name or computation changes it, so the engine rejects a model whose
    features differ from its own. The text is the engine's
    (feature_spec_hash in features.cpp)."""
    text = fs.v3_catalog_hash() + "\nversion\t3\n"
    for tag, names in (
        ("query", fs.v3_query_feature_names()),
        ("item", fs.v3_item_feature_names(spec)),
        ("interaction", fs.v3_interaction_feature_names()),
    ):
        text += tag + "\t" + ",".join(names) + "\n"
    for a in spec:
        text += f"{a.name}\t{a.default}\t{a.transform}"
        if a.transform == "onehot":
            text += "\t" + ",".join(str(v) for v in a.values)
        text += "\n"
    for c, (problem, config, hw, ctx, plan) in enumerate(_GOLDEN_CASES):
        attrs = {
            a.name: (
                a.values[(j + c) % len(a.values)]
                if a.transform == "onehot"
                else (7 * j + 3 * c) % 17 - 4
            )
            for j, a in enumerate(spec)
        }
        q, i, x = fs.v3_feature_vectors(
            problem,
            config,
            attrs,
            hardware=DeviceHardware(*hw),
            constants=_GOLDEN_CONSTANTS,
            context=fs.ExecutionContext(*ctx),
            spec=spec,
            plan=plan,
        )
        bits = np.asarray(q + i + x, dtype=np.float32).view(np.uint32)
        text += f"golden\t{c}\t" + ",".join(f"{int(b):08x}" for b in bits) + "\n"
    return fs._fnv1a(text.encode("ascii"))


def check_provenance(provenance: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The provenance entries an MLREC_v3 file records, in PROVENANCE_KEYS
    order: keys of PROVENANCE_KEYS with values that are 1 to 65535 printable
    ASCII characters without spaces (`date` an ISO 8601 date or date and
    time); None or "" values are left out. ValueError otherwise."""
    if not provenance:
        return {}
    unknown = sorted(set(provenance) - set(PROVENANCE_KEYS))
    if unknown:
        raise ValueError(
            f"unknown provenance keys {unknown}; allowed: {list(PROVENANCE_KEYS)}"
        )
    out: Dict[str, str] = {}
    for key in PROVENANCE_KEYS:
        value = provenance.get(key)
        if value is None or value == "":
            continue
        text = check_label(value, f"provenance {key}")
        if key == "date" and not _ISO_DATE_RE.fullmatch(text):
            raise ValueError(f"provenance date {text!r} is not an ISO 8601 date")
        out[key] = text
    return out


def _provenance_bytes(provenance: Mapping[str, str]) -> bytes:
    if not provenance:
        return b""
    return struct.pack("<H", len(provenance)) + b"".join(
        _lenstr(k) + _lenstr(v) for k, v in provenance.items()
    )


def _v3_tail(
    spec: fs.AttributeSpec,
    arch: str,
    constants: ArchConstants,
    serving: Mapping[str, Any],
    contexts: Mapping[str, Any],
    clip: bool,
    provenance: Optional[Mapping[str, Any]] = None,
    trained_keys: Optional[Iterable[int]] = None,
) -> bytes:
    pmc, mi_default = float(constants.parallel_mi_cu), float(constants.mi_default)
    if not (math.isfinite(pmc) and pmc > 0):
        raise ValueError("parallel_mi_cu must be finite and > 0")
    if not (math.isfinite(mi_default) and mi_default > 0):
        raise ValueError("mi_default cycles must be finite and > 0")
    lo = float(contexts.get("min_cu_fraction", 1.0))
    hi = float(contexts.get("max_cu_fraction", 1.0))
    if not (0.0 < lo <= hi <= 1.0):
        raise ValueError("CU fraction range must satisfy 0 < min <= max <= 1")
    cascade = int(serving.get("cascade_keys", 32) or 0)
    if not 0 <= cascade < 2**32:
        raise ValueError(f"cascade_keys {cascade} out of range")
    floor = contexts.get("min_effective_cus")
    flags = v3_serving_flags(serving) | (V3_CU_FLOOR if floor is not None else 0)
    fhash = fs.v3_catalog_hash()
    tail = [
        _lenstr(fhash),
        struct.pack("<Q", feature_spec_hash(spec)),
        _label(arch, "arch"),
        struct.pack("<dd", pmc, mi_default),
        _mi_table_bytes(constants),
        struct.pack("<II", flags, cascade),
        struct.pack("<dd", lo, hi),
        struct.pack(
            "<BB",
            schedule_mask(contexts.get("schedules", ("default",))),
            1 if clip else 0,
        ),
        struct.pack("<H", len(spec)),
    ]
    for a in spec:
        tail.append(_label(a.name, "attribute name"))
        tail.append(
            struct.pack("<qB", a.default, fs.ATTRIBUTE_TRANSFORMS.index(a.transform))
        )
        if a.transform == "onehot":
            tail.append(struct.pack(f"<H{len(a.values)}q", len(a.values), *a.values))
    if serving.get("trained_key_gate", False):
        tail.append(trained_keys_bytes(trained_keys or ()))
    if floor is not None:
        tail.append(cu_floor_bytes(floor))
    if serving.get("alignment_veto"):
        tail.append(alignment_veto_bytes(serving["alignment_veto"]))
    tail.append(_provenance_bytes(check_provenance(provenance)))
    return b"".join(tail)


def write_model_v3(
    bundle: Mapping[str, Any],
    splits: Optional[Union[Mapping[str, SplitRule], Iterable[SplitRule]]],
    arch: str,
    constants: ArchConstants,
    weight_dtype: Union[str, int],
    provenance: Optional[Mapping[str, Any]] = None,
) -> bytes:
    """MLREC_v3 bytes of a trained v3 bundle: as `write_model`, plus
    `attribute_spec` (list of AttributeFeature dicts), `contexts`
    {min_cu_fraction, max_cu_fraction, schedules, and optionally
    min_effective_cus: the CU floor, written with serving flag bit 8},
    `clip` and per cell
    `feature_bounds` {q_lo, ..., x_hi} and `view_key_hashes`; `serving`
    holds small_batched_rule / dot2_rule / non_temporal_rule /
    non_temporal_family_rule, launch_keys, bank and trained_key_gate
    (default off), whitelist_tier (default on), cascade_keys (default
    32) and alignment_veto ({k_min, align_bits}, or absent or None for
    none); with trained_key_gate, `trained_view_keys` lists the view-key
    hashes of the keys with training data. The arch's
    bandwidth coefficients are not written. `provenance` (else the bundle's
    `provenance`) is recorded as `check_provenance` describes."""
    wdt = weight_dtype_code(weight_dtype)
    spec = fs.parse_attribute_spec(bundle.get("attribute_spec") or [])
    expected = (
        fs.v3_query_feature_names(),
        fs.v3_item_feature_names(spec),
        fs.v3_interaction_feature_names(),
    )
    names = (
        list(bundle.get("q_names", expected[0])),
        list(bundle.get("i_names", expected[1])),
        list(bundle.get("x_names", expected[2])),
    )
    if [list(n) for n in names] != [list(e) for e in expected]:
        raise ValueError(
            "bundle feature names differ from the v3 catalog "
            f"(hash {fs.v3_catalog_hash()}) with its attribute spec"
        )
    dims = (len(expected[0]), len(expected[1]), len(expected[2]))
    models: Mapping[str, Any] = bundle["models"]
    rules = _split_rules(models, splits)
    clip = bool(bundle.get("clip", False))
    payload = b"".join(_split_bytes(r) for r in rules) + b"".join(
        _cell_bytes(label, models[label], dims, wdt, version=3, clip=clip)
        for label in sorted(models)
    )
    tail = _v3_tail(
        spec,
        str(arch),
        constants,
        bundle.get("serving") or {},
        bundle.get("contexts") or {},
        clip,
        provenance if provenance is not None else bundle.get("provenance"),
        bundle.get("trained_view_keys"),
    )
    header_size = _V2_FIXED.size + len(tail)
    head = _V2_FIXED.pack(
        MAGIC_V3,
        ENDIAN_MARKER,
        header_size,
        len(payload),
        zlib.crc32(payload) & 0xFFFFFFFF,
        wdt,
        0,
        0,
        *dims,
        len(models),
        len(rules),
    )
    return head + tail + payload + TRAILER


def write_model(
    bundle: Mapping[str, Any],
    splits: Optional[Union[Mapping[str, SplitRule], Iterable[SplitRule]]],
    arch: str,
    constants: ArchConstants,
    weight_dtype: Union[str, int],
    provenance: Optional[Mapping[str, Any]] = None,
) -> bytes:
    """MLREC_v2 bytes of a trained bundle (the `models.pt` dict: `models`
    label -> entry with state_dict / feature_norms / dims /
    smart_k_signatures, plus `q_names` / `i_names` / `x_names`).

    `splits` None rebuilds the tree from the cell labels, which is what a
    deployed model must carry. Cells and splits are written sorted by label.
    Raises ValueError when the bundle does not match the current feature
    catalog or holds values the engine would reject.

    A bundle with `catalog` "v3" is written as MLREC_v3 (`write_model_v3`)
    with `provenance`; MLREC_v2 files record none."""
    if bundle.get("catalog", "v2") == "v3":
        return write_model_v3(bundle, splits, arch, constants, weight_dtype, provenance)
    if bundle.get("catalog", "v2") != "v2":
        raise ValueError(f"unknown bundle catalog {bundle.get('catalog')!r}")
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
    rules = _split_rules(models, splits)
    payload = b"".join(_split_bytes(r) for r in rules) + b"".join(
        _cell_bytes(label, models[label], dims, wdt) for label in sorted(models)
    )
    serving = bundle.get("serving") or {}
    cascade = int(serving.get("cascade_keys", 0) or 0)
    if not 0 <= cascade <= 0xFFFF:
        raise ValueError(f"cascade_keys {cascade} out of range")
    return _assemble_v2(
        weight_dtype=wdt,
        serving_flags=(
            0 if serving.get("non_temporal_rule", True) else NO_NON_TEMPORAL_RULE
        ),
        cascade_keys=cascade,
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
    version: int = 2,
    clip: bool = False,
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
        if version == 3 and clip:
            cell["feature_bounds"] = {
                key: cur.floats(sizes[key[0]], f"{label} {key}") for key in BOUND_KEYS
            }
        (n_sig,) = cur.unpack("<I", f"{label} n_sig")
        if 32 * n_sig > cur.end - cur.pos:
            raise FormatError(f"{label}: signature count exceeds the file")
        sig_arr = np.frombuffer(cur.take(32 * n_sig, f"{label} signatures"), "<i4")
        cell["smart_k_signatures"] = [
            tuple(int(v) for v in row) for row in sig_arr.reshape(n_sig, 8)
        ]
        if version == 3:
            (n_vk,) = cur.unpack("<I", f"{label} n_view_keys")
            if 8 * n_vk > cur.end - cur.pos:
                raise FormatError(f"{label}: view key count exceeds the file")
            vk = np.frombuffer(cur.take(8 * n_vk, f"{label} view keys"), "<u8")
            cell["view_key_hashes"] = [int(h) for h in vk]
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
        flags,
        cascade,
        q_dim,
        i_dim,
        x_dim,
        n_cells,
        n_splits,
    ) = _V2_FIXED.unpack_from(data, 0)
    if endian != ENDIAN_MARKER:
        raise FormatError("bad endian marker")
    if flags & ~_SERVING_FLAGS_KNOWN:
        raise FormatError(f"unknown serving flags {flags}")
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
        "serving": {
            "non_temporal_rule": not flags & NO_NON_TEMPORAL_RULE,
            "cascade_keys": cascade,
        },
        "header_size": header_size,
        "payload_crc32": crc,
        "has_trailer": True,
    }


def _v3_header(data: bytes, end: int) -> Dict[str, Any]:
    if len(data) < _V2_FIXED.size:
        raise FormatError("file too short")
    fields = _V2_FIXED.unpack_from(data, 0)
    (_m, endian, header_size, payload_size, crc, wdt, f29, f30) = fields[:8]
    if endian != ENDIAN_MARKER:
        raise FormatError("bad endian marker")
    if f29 or f30:
        raise FormatError("reserved header bytes 29-31 must be zero")
    if wdt not in WEIGHT_DTYPE_NAMES:
        raise FormatError(f"bad weight dtype {wdt}")
    cur = _Cursor(data, _V2_FIXED.size, min(header_size, end))
    fhash = cur.lenstr("<H", "feature hash")
    (spec_hash,) = cur.unpack("<Q", "feature spec hash")
    arch = cur.lenstr("<H", "arch")
    pmc, mi_default = cur.unpack("<dd", "arch constants")
    (n_mi,) = cur.unpack("<I", "n_mi")
    table: Dict[Tuple[int, int, int, str], float] = {}
    for e in range(n_mi):
        m, n, k, dt, cycles = cur.unpack(_MI_ENTRY.format, f"mi entry {e}")
        if dt not in DATATYPE_NAME:
            raise FormatError(f"mi entry {e}: unknown dtype {dt}")
        table[(m, n, k, DATATYPE_NAME[dt])] = cycles
    serving_offset = cur.pos
    flags, cascade = cur.unpack("<II", "serving")
    lo, hi = cur.unpack("<dd", "CU fraction range")
    schedules, clip = cur.unpack("<BB", "schedules")
    if flags & ~_V3_SERVING_FLAGS_KNOWN:
        raise FormatError(f"unknown serving flags {flags}")
    if flags & V3_NON_TEMPORAL_FAMILY_RULE and not flags & V3_NON_TEMPORAL_RULE:
        raise FormatError(
            f"serving flags {flags} ask for the family-wise non-temporal rule "
            f"without the non-temporal rule"
        )
    if flags & V3_ALIGNMENT_VETO and flags & V3_NON_TEMPORAL_RULE:
        raise FormatError(
            f"serving flags {flags} ask for the alignment veto with the "
            f"non-temporal rule"
        )
    if not 0.0 < lo <= hi <= 1.0:
        raise FormatError("CU fraction range must satisfy 0 < min <= max <= 1")
    if schedules == 0 or schedules & ~0x07:
        raise FormatError(f"invalid schedule mask {schedules}")
    if clip > 1:
        raise FormatError(f"invalid clip flag {clip}")
    (n_attr,) = cur.unpack("<H", "n_attributes")
    raw = []
    for a in range(n_attr):
        name = cur.lenstr("<H", f"attribute {a} name")
        default, transform = cur.unpack("<qB", f"attribute {a}")
        if transform >= len(fs.ATTRIBUTE_TRANSFORMS):
            raise FormatError(f"attribute {name}: unknown transform {transform}")
        entry: Dict[str, Any] = {
            "name": name,
            "default": default,
            "transform": fs.ATTRIBUTE_TRANSFORMS[transform],
        }
        if entry["transform"] == "onehot":
            (nv,) = cur.unpack("<H", f"attribute {name} values")
            entry["values"] = list(cur.unpack(f"<{nv}q", f"attribute {name} values"))
        raw.append(entry)
    trained: Optional[List[int]] = None
    if flags & V3_TRAINED_KEY_GATE:
        (n_trained,) = cur.unpack("<I", "n_trained_keys")
        if n_trained == 0:
            raise FormatError("the trained-key gate lists no trained key")
        if 8 * n_trained > cur.end - cur.pos:
            raise FormatError("trained key count exceeds the header")
        trained = [
            int(h)
            for h in np.frombuffer(cur.take(8 * n_trained, "trained keys"), "<u8")
        ]
        if any(b <= a for a, b in zip(trained, trained[1:])):
            raise FormatError("trained keys are not strictly ascending")
    floor: Optional[int] = None
    if flags & V3_CU_FLOOR:
        (floor,) = cur.unpack("<I", "CU floor")
        if floor == 0:
            raise FormatError("the CU floor must be at least 1")
    veto: Optional[Dict[str, int]] = None
    if flags & V3_ALIGNMENT_VETO:
        veto_k_min, veto_bits = cur.unpack("<II", "alignment veto")
        if veto_k_min == 0:
            raise FormatError("the alignment veto's K must be at least 1")
        if veto_bits < 8 or veto_bits & (veto_bits - 1):
            raise FormatError(
                f"the alignment veto's line of {veto_bits} bits is not a power "
                f"of two of at least 8 bits"
            )
        veto = {"k_min": veto_k_min, "align_bits": veto_bits}
    provenance: Dict[str, str] = {}
    if cur.pos < cur.end:
        (n_prov,) = cur.unpack("<H", "n_provenance")
        if not 1 <= n_prov <= len(PROVENANCE_KEYS):
            raise FormatError(f"provenance count {n_prov} out of range")
        for p in range(n_prov):
            key = cur.lenstr("<H", f"provenance {p} key")
            value = cur.lenstr("<H", f"provenance {p} value")
            if key not in PROVENANCE_KEYS:
                raise FormatError(f"unknown provenance key {key!r}")
            if key in provenance:
                raise FormatError(f"duplicate provenance key {key!r}")
            if not _LABEL_RE.fullmatch(value):
                raise FormatError(f"invalid provenance {key} value {value[:60]!r}")
            provenance[key] = value
    try:
        spec = fs.parse_attribute_spec(raw)
    except ValueError as e:
        raise FormatError(str(e)) from e
    if fhash != fs.v3_catalog_hash():
        raise FormatError(f"feature catalog hash {fhash!r} is not the v3 catalog's")
    if feature_spec_hash(spec) != spec_hash:
        raise FormatError(
            "feature_spec_hash does not match the attribute spec and the v3 features"
        )
    names = (
        fs.v3_query_feature_names(),
        fs.v3_item_feature_names(spec),
        fs.v3_interaction_feature_names(),
    )
    dims = tuple(fields[8:11])
    if dims != tuple(len(n) for n in names):
        raise FormatError(f"feature dims {dims} do not match the v3 catalog and spec")
    contexts: Dict[str, Any] = {
        "min_cu_fraction": lo,
        "max_cu_fraction": hi,
        "schedules": [n for n, b in SCHEDULE_BITS.items() if schedules & b],
    }
    if floor is not None:
        contexts["min_effective_cus"] = floor
    serving: Dict[str, Any] = {
        "small_batched_rule": bool(flags & V3_SMALL_BATCHED_RULE),
        "dot2_rule": bool(flags & V3_DOT2_RULE),
        "non_temporal_rule": bool(flags & V3_NON_TEMPORAL_RULE),
        "whitelist_tier": not flags & V3_NO_WHITELIST_TIER,
        "launch_keys": bool(flags & V3_LAUNCH_KEYS),
        "non_temporal_family_rule": bool(flags & V3_NON_TEMPORAL_FAMILY_RULE),
        "bank": bool(flags & V3_BANK),
        "trained_key_gate": bool(flags & V3_TRAINED_KEY_GATE),
        "cascade_keys": cascade,
    }
    if veto is not None:
        serving["alignment_veto"] = veto
    return {
        "header_end": cur.pos,
        "serving_offset": serving_offset,
        "header_size": header_size,
        "payload_size": payload_size,
        "crc": crc,
        "weight_dtype": wdt,
        "feature_catalog_hash": fhash,
        "feature_spec_hash": spec_hash,
        "attribute_spec": spec,
        "names": names,
        "arch": arch,
        "dims": dims,
        "n_cells": fields[11],
        "n_splits": fields[12],
        "constants": ArchConstants(
            parallel_mi_cu=pmc,
            bw=(0.0, 0.0, 0.0),
            mi_table=table,
            mi_default=mi_default,
        ),
        "serving": serving,
        "trained_view_keys": trained,
        "contexts": contexts,
        "clip": bool(clip),
        "provenance": {k: provenance[k] for k in PROVENANCE_KEYS if k in provenance},
    }


def _read_v3(data: bytes) -> Dict[str, Any]:
    if len(data) < _V2_FIXED.size + len(TRAILER):
        raise FormatError("file too short")
    h = _v3_header(data, len(data))
    header_size, payload_size = h["header_size"], h["payload_size"]
    if header_size + payload_size + len(TRAILER) != len(data):
        raise FormatError("header/payload sizes do not match the file size")
    if data[-len(TRAILER) :] != TRAILER:
        raise FormatError("missing MLRECEND trailer")
    if h["header_end"] != header_size:
        raise FormatError("header_size does not match the header contents")
    payload = data[header_size : header_size + payload_size]
    if zlib.crc32(payload) & 0xFFFFFFFF != h["crc"]:
        raise FormatError("payload CRC mismatch")
    splits, cells = _parse_payload(
        _Cursor(data, header_size, header_size + payload_size),
        h["n_splits"],
        h["n_cells"],
        h["dims"],
        h["weight_dtype"],
        version=3,
        clip=h["clip"],
    )
    return {
        "format": MAGIC_V3.decode(),
        "version": 3,
        "weight_dtype": h["weight_dtype"],
        "weight_dtype_name": WEIGHT_DTYPE_NAMES[h["weight_dtype"]],
        "feature_catalog_hash": h["feature_catalog_hash"],
        "feature_spec_hash": h["feature_spec_hash"],
        "attribute_spec": h["attribute_spec"],
        "names": h["names"],
        "arch": h["arch"],
        "dims": h["dims"],
        "constants": h["constants"],
        "splits": splits,
        "cells": cells,
        "serving": h["serving"],
        "trained_view_keys": h["trained_view_keys"],
        "contexts": h["contexts"],
        "clip": h["clip"],
        "provenance": h["provenance"],
        "header_size": header_size,
        "payload_crc32": h["crc"],
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


V3_SOFT_RULE_BITS = {
    "small_batched_rule": V3_SMALL_BATCHED_RULE,
    "dot2_rule": V3_DOT2_RULE,
    "non_temporal_rule": V3_NON_TEMPORAL_RULE,
    "non_temporal_family_rule": V3_NON_TEMPORAL_FAMILY_RULE,
}


def _with_flags(data: bytes, edit: Any) -> bytes:
    """`data` (MLREC_v3) with its serving flags replaced by `edit(flags)`."""
    out = bytearray(data)
    h = _v3_header(bytes(out), len(out))
    at = h["serving_offset"]
    (flags,) = struct.unpack_from("<I", out, at)
    flags = edit(flags)
    if flags & V3_NON_TEMPORAL_FAMILY_RULE and not flags & V3_NON_TEMPORAL_RULE:
        raise ValueError(
            "the family-wise non-temporal rule refines the non-temporal rule; "
            "serve with both"
        )
    if flags & V3_ALIGNMENT_VETO and flags & V3_NON_TEMPORAL_RULE:
        raise ValueError(
            "the alignment veto is a rule of models served without the "
            "non-temporal rule"
        )
    struct.pack_into("<I", out, at, flags)
    return bytes(out)


def with_soft_rules(data: bytes, rules: Mapping[str, bool]) -> bytes:
    """An MLREC_v3 model file that serves with `rules` (`passes_gates` rule
    keywords; unnamed rules keep their flag) and is otherwise `data`. The
    leading-free rule is not a v3 serving option; the family-wise
    non-temporal rule needs the non-temporal rule (ValueError)."""
    if bytes(data[:8]) != MAGIC_V3:
        raise FormatError("soft-rule flags are an MLREC_v3 serving option")
    unknown = sorted(set(rules) - set(V3_SOFT_RULE_BITS) - {"leading_free_rule"})
    if unknown:
        raise ValueError(f"unknown soft rules {unknown}")
    if rules.get("leading_free_rule"):
        raise ValueError("MLREC_v3 does not serve with the leading-free rule")

    def edit(flags: int) -> int:
        for name, bit in V3_SOFT_RULE_BITS.items():
            if name in rules:
                flags = flags | bit if rules[name] else flags & ~bit
        return flags

    return _with_flags(data, edit)


def with_bank(data: bytes, bank: bool) -> bytes:
    """An MLREC_v3 model file that says whether it was trained on the bank
    as well as the table (serving flag bit 6) and is otherwise `data`."""
    if bytes(data[:8]) != MAGIC_V3:
        raise FormatError("the bank flag is an MLREC_v3 serving option")
    return _with_flags(data, lambda f: f | V3_BANK if bank else f & ~V3_BANK)


def with_trained_keys(data: bytes, keys: Optional[Iterable[int]]) -> bytes:
    """An MLREC_v3 model file served with the trained-key gate over the
    view-key hashes `keys` (without the gate when None) and otherwise
    `data`: the header changes size, the payload and its CRC do not."""
    data = bytes(data)
    if data[:8] != MAGIC_V3:
        raise FormatError("the trained-key gate is an MLREC_v3 serving option")
    h = _v3_header(data, len(data))
    header_size = h["header_size"]
    flags = struct.unpack_from("<I", data, h["serving_offset"])[0]
    # The section sits between the attribute spec and the provenance.
    cur = _Cursor(data, _V2_FIXED.size, header_size)
    spec_end = _attribute_spec_end(cur, h["serving_offset"])
    section_end = spec_end
    if flags & V3_TRAINED_KEY_GATE:
        (n,) = struct.unpack_from("<I", data, spec_end)
        section_end = spec_end + 4 + 8 * n
    section = b"" if keys is None else trained_keys_bytes(keys)
    head = bytearray(data[:spec_end] + section + data[section_end:header_size])
    flags = (
        flags | V3_TRAINED_KEY_GATE
        if keys is not None
        else flags & ~V3_TRAINED_KEY_GATE
    )
    struct.pack_into("<I", head, h["serving_offset"], flags)
    struct.pack_into("<I", head, 12, len(head))
    return bytes(head) + data[header_size:]


def with_cu_floor(data: bytes, cus: Optional[int]) -> bytes:
    """An MLREC_v3 model file that supports no fewer than `cus` usable CUs
    on any device (serving flag bit 8; no CU floor when None) and is
    otherwise `data`: the header changes size, the payload and its CRC do
    not. An engine that predates the floor refuses a file that has one."""
    data = bytes(data)
    if data[:8] != MAGIC_V3:
        raise FormatError("the CU floor is an MLREC_v3 header field")
    h = _v3_header(data, len(data))
    header_size = h["header_size"]
    flags = struct.unpack_from("<I", data, h["serving_offset"])[0]
    # The section follows the trained keys and precedes the provenance.
    at = _attribute_spec_end(
        _Cursor(data, _V2_FIXED.size, header_size), h["serving_offset"]
    )
    if flags & V3_TRAINED_KEY_GATE:
        (n,) = struct.unpack_from("<I", data, at)
        at += 4 + 8 * n
    end = at + 4 if flags & V3_CU_FLOOR else at
    section = b"" if cus is None else cu_floor_bytes(cus)
    head = bytearray(data[:at] + section + data[end:header_size])
    flags = flags | V3_CU_FLOOR if cus is not None else flags & ~V3_CU_FLOOR
    struct.pack_into("<I", head, h["serving_offset"], flags)
    struct.pack_into("<I", head, 12, len(head))
    return bytes(head) + data[header_size:]


def with_alignment_veto(data: bytes, k_min: int, align_bits: int) -> bytes:
    """An MLREC_v3 model file served with the alignment veto of K above
    `k_min` and `align_bits`-bit lines (serving flag bit 9; lib/feasibility.py
    `alignment_vetoed`), replacing any it has, and otherwise `data`: the
    header changes size, the payload and its CRC do not, so a model can carry
    the veto without retraining. ValueError for a model served with the
    non-temporal rule and for values `alignment_veto_bytes` refuses."""
    section = alignment_veto_bytes({"k_min": k_min, "align_bits": align_bits})
    return _with_alignment_veto_section(data, section)


def without_alignment_veto(data: bytes) -> bytes:
    """An MLREC_v3 model file without the alignment veto and otherwise
    `data`, which it returns unchanged when `data` has none."""
    return _with_alignment_veto_section(data, None)


def _with_alignment_veto_section(data: bytes, section: Optional[bytes]) -> bytes:
    data = bytes(data)
    if data[:8] != MAGIC_V3:
        raise FormatError("the alignment veto is an MLREC_v3 serving option")
    h = _v3_header(data, len(data))
    header_size = h["header_size"]
    flags = struct.unpack_from("<I", data, h["serving_offset"])[0]
    if section is None and not flags & V3_ALIGNMENT_VETO:
        return data
    if section is not None and flags & V3_NON_TEMPORAL_RULE:
        raise ValueError(
            "the alignment veto is a rule of models served without the "
            "non-temporal rule"
        )
    # The section follows the trained keys and the CU floor and precedes the
    # provenance.
    at = _attribute_spec_end(
        _Cursor(data, _V2_FIXED.size, header_size), h["serving_offset"]
    )
    if flags & V3_TRAINED_KEY_GATE:
        (n,) = struct.unpack_from("<I", data, at)
        at += 4 + 8 * n
    if flags & V3_CU_FLOOR:
        at += 4
    end = at + 8 if flags & V3_ALIGNMENT_VETO else at
    head = bytearray(data[:at] + (section or b"") + data[end:header_size])
    flags = (
        flags | V3_ALIGNMENT_VETO if section is not None else flags & ~V3_ALIGNMENT_VETO
    )
    struct.pack_into("<I", head, h["serving_offset"], flags)
    struct.pack_into("<I", head, 12, len(head))
    return bytes(head) + data[header_size:]


def _attribute_spec_end(cur: _Cursor, serving_offset: int) -> int:
    """Offset just past an MLREC_v3 header's attribute spec."""
    cur.pos = serving_offset + struct.calcsize("<IIddBB")
    (n_attr,) = cur.unpack("<H", "n_attributes")
    for a in range(n_attr):
        cur.lenstr("<H", f"attribute {a} name")
        _default, transform = cur.unpack("<qB", f"attribute {a}")
        if fs.ATTRIBUTE_TRANSFORMS[transform] == "onehot":
            (nv,) = cur.unpack("<H", f"attribute {a} values")
            cur.take(8 * nv, f"attribute {a} values")
    return cur.pos


def read_header(data: bytes) -> Dict[str, Any]:
    """Format version, weight dtype, hash, arch and counts of a model file
    without parsing the payload."""
    magic = bytes(data[:8])
    if magic == MAGIC_V3:
        data = bytes(data)
        h = _v3_header(data, len(data))
        return {"version": 3} | {
            k: h[k]
            for k in (
                "weight_dtype",
                "feature_catalog_hash",
                "feature_spec_hash",
                "attribute_spec",
                "arch",
                "dims",
                "n_cells",
                "n_splits",
                "serving",
                "trained_view_keys",
                "contexts",
                "clip",
                "provenance",
            )
        }
    data = bytes(data[:4096]) if len(data) > 4096 else bytes(data)
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
            "dims": tuple(fields[8:11]),
            "n_cells": fields[11],
            "n_splits": fields[12],
            "serving": {
                "non_temporal_rule": not fields[6] & NO_NON_TEMPORAL_RULE,
                "cascade_keys": fields[7],
            },
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
    """Parse an MLREC_v1, v2 or v3 file. Weight matrices come back as the
    fp32 values the engine computes from them. Raises FormatError."""
    data = bytes(data)
    magic = data[:8]
    if magic == MAGIC_V3:
        return _read_v3(data)
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
        if "view_key_hashes" in c:
            models[c["label"]]["view_key_hashes"] = list(c["view_key_hashes"])
        if "feature_bounds" in c:
            models[c["label"]]["feature_bounds"] = {
                k: list(v) for k, v in c["feature_bounds"].items()
            }
    if parsed.get("version") == 3:
        q, i, x = parsed["names"]
        out: Dict[str, Any] = {
            "catalog": "v3",
            "attribute_spec": [a.to_dict() for a in parsed["attribute_spec"]],
            "contexts": dict(parsed["contexts"]),
            "clip": parsed["clip"],
            "q_names": list(q),
            "i_names": list(i),
            "x_names": list(x),
            "models": models,
        }
        if parsed.get("provenance"):
            out["provenance"] = dict(parsed["provenance"])
        if parsed.get("trained_view_keys") is not None:
            out["trained_view_keys"] = list(parsed["trained_view_keys"])
    else:
        out = {
            "q_names": fs.query_feature_names(),
            "i_names": fs.item_feature_names(),
            "x_names": fs.interaction_feature_names(),
            "models": models,
        }
    if "serving" in parsed:
        out["serving"] = dict(parsed["serving"])
    return out


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
    "PROVENANCE_KEYS",
    "WEIGHT_DTYPES",
    "WEIGHT_ORDER",
    "bundle_from_model",
    "check_label",
    "check_provenance",
    "convert_v1_to_v2",
    "cu_floor_bytes",
    "decode_weight",
    "encode_weight",
    "feature_spec_hash",
    "read_model",
    "trained_keys_bytes",
    "weight_dtype_code",
    "with_bank",
    "with_cu_floor",
    "with_soft_rules",
    "with_trained_keys",
    "write_model",
    "write_model_v3",
)
