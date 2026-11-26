# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write the generated C++ headers of the native engine ``hipkernel:AttentionGeneralBwd``.

The engine's C++ lowering, predicate and plan restate rocKE's Python backward
(``dispatch/attention_bwd/hipdnn.py`` over ``kernels/common/attention_bwd_plan.py``).
Every constant and table they read comes from Python through this script, never by
hand:

* ``AttentionGeneralBwdConstants.generated.hpp``: the reason codes in predicate
  order (plus the engine's ``LOWERING``), the limits, the head sizes and dtypes, the
  feature groups and the engine policy's declared set, the tuning-key vocabularies
  and buckets, the per-arch default geometry and aux rows, the token encodings and
  names, and the kernarg ABI of every launch kind.
* ``AttentionGeneralBwdCatalog.generated.hpp``: the per-arch tuned-table keys with the
  runtime values the plan derives from them, every catalog row (one main kernel and
  the prep / convert / split-dQ kernels its plan launches) and one representative
  request per row.

Usage::

    python3 general_bwd_constants.py --out-dir <kernel_ingestor_engine/packs> [--check]

The output is formatted with ``clang-format --style=file`` (the provider's style
file), so ``--check`` compares formatted text. ``--clang-format`` or the environment
variable ``HKP_CLANG_FORMAT`` names the binary; otherwise ``clang-format-18`` or
``clang-format`` on PATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HKP = Path(__file__).resolve().parents[2]
for _sub in ("rocke/platform/python", "rocke/library"):
    _p = str(_HKP / _sub)
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
os.environ.setdefault("ROCKE_BACKEND", "python")

from dispatch.attention_bwd import hipdnn as H  # noqa: E402
from dispatch.attention_bwd import hipdnn_catalog as C  # noqa: E402
from dispatch.sdpa.hipdnn import HIPDNN_DTYPES  # noqa: E402
from kernels.common import attention_bwd_plan as P  # noqa: E402
from kernels.common._attention_bwd_worklist import WL_HEADER_WORDS  # noqa: E402
from kernels.common.attention_bwd import (  # noqa: E402
    ATTN_BWD_ABI,
    BWD_DTYPES,
    BWD_HEAD_SIZES,
    STAGE_VECS,
    STATIC_SCALES,
    attn_bwd_arch_facts,
    attn_bwd_default_geometry,
    attn_bwd_params,
)
from kernels.common.attention_bwd_aux import CONVERT_VECS  # noqa: E402
from rocke.helpers.attention_band import I32_FAR  # noqa: E402

CONSTANTS_HEADER = "AttentionGeneralBwdConstants.generated.hpp"
CATALOG_HEADER = "AttentionGeneralBwdCatalog.generated.hpp"
NAMESPACE = "hip_kernel_provider::kernel_ingestor_engine::general_bwd::generated"
TOOL = "dnn-providers/hip-kernel-provider/descriptor-packaging/tools/general_bwd_constants.py"
STAGES = ("prep", "main", "convert", "dq")
SEQ_MODES = ("batched", "thd")
# The head size of the HEAD_DIM_256_BWD decline (a literal of the predicate).
HEAD_DIM_256 = 256
# Full compute-unit count per arch: the representatives' device fill hint.
FULL_CUS = {"gfx942": 304, "gfx950": 256}


# ---------------------------------------------------------------------------
# C++ spelling
# ---------------------------------------------------------------------------


def _str(s: str) -> str:
    if not all(32 <= ord(c) < 127 for c in s) or "\\" in s:
        raise ValueError(f"{s!r} is not a plain ASCII literal")
    return json.dumps(s)


def _bool(v) -> str:
    return "true" if v else "false"


def _int(v) -> str:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{v!r} is not an int")
    return f"INT64_C({v})" if abs(v) >= 2**31 else str(v)


def _enum_name(s: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in s).upper()
    return out if not out[0].isdigit() else f"V{out}"


def _array(ctype: str, name: str, items: list[str]) -> str:
    body = "".join(f"    {x},\n" for x in items)
    text = f"inline constexpr std::array<{ctype}, {len(items)}> {name} = {{{{\n{body}}}}};\n"
    if ctype in _SCALAR_TYPES:
        return text
    # one record per line: clang-format would align aggregate fields in columns
    return f"// clang-format off\n{text}// clang-format on\n"


_SCALAR_TYPES = ("std::string_view", "int64_t", "bool", "double")


def _str_array(name: str, values) -> str:
    return _array("std::string_view", name, [_str(v) for v in values])


def _int_array(name: str, values) -> str:
    return _array("int64_t", name, [_int(v) for v in values])


def _enum(name: str, values: dict, base: str = "int64_t") -> str:
    body = "".join(f"    {_enum_name(k)} = {v},\n" for k, v in values.items())
    return f"enum class {name} : {base}\n{{\n{body}}};\n"


def _struct(name: str, fields: list[tuple[str, str]]) -> str:
    body = "".join(f"    {t} {n};\n" for t, n in fields)
    return f"struct {name}\n{{\n{body}}};\n"


def _init(values: list[str]) -> str:
    return "{" + ", ".join(values) + "}"


def _wrap(body: str, includes: tuple[str, ...], what: str) -> str:
    inc = "".join(f"#include <{i}>\n" for i in includes)
    return (
        "// Copyright © Advanced Micro Devices, Inc., or its affiliates.\n"
        "// SPDX-License-Identifier:  MIT\n\n"
        f"// Generated by {TOOL}; do not edit.\n"
        f"// {what}\n\n"
        "#pragma once\n\n"
        f"{inc}\n"
        f"namespace {NAMESPACE}\n{{\n\n{body}\n}} // namespace {NAMESPACE}\n"
    )


# ---------------------------------------------------------------------------
# constants header
# ---------------------------------------------------------------------------

_KIND_TYPE = {"Q": "pointer", "q": "i64", "f": "f32", "i": "i32"}


def constants_header() -> str:
    reasons = (*P.REASON_CODES, H.LOWERING)
    policy = H.engine_policy()
    parts = []
    parts.append(
        f"inline constexpr std::string_view ENGINE_NAME = {_str(H.ENGINE_NAME)};\n"
    )
    parts.append(f"inline constexpr std::string_view ABI = {_str(ATTN_BWD_ABI)};\n")
    parts.append(_str_array("ENGINE_ARCHS", H.ENGINE_ARCHS))
    parts.append(
        "\n// Decline codes in the order the checks run (first failure wins); LOWERING is the\n"
        "// engine's own code for a graph its lowering cannot read.\n"
    )
    parts.append(_enum("Reason", {r: i for i, r in enumerate(reasons)}, "uint8_t"))
    parts.append(_str_array("REASON_NAMES", reasons))
    parts.append(_str_array("FEATURE_DECLINES", P.FEATURE_DECLINES))

    parts.append("\n// Limits and numeric constants.\n")
    scalars = [
        ("double", "LOG2E", repr(P.LOG2E)),
        ("int64_t", "I32_FAR", _int(I32_FAR)),
        ("int64_t", "I32_MAX", _int(P.I32_MAX)),
        ("int64_t", "GRID_YZ_MAX", _int(P._GRID_YZ_MAX)),
        ("int64_t", "WORKSPACE_ALIGN", _int(P._WS_ALIGN)),
        ("int64_t", "OVERLAP_BUDGET", _int(P._OVERLAP_BUDGET)),
        ("int64_t", "HEAD_DIM_256", _int(HEAD_DIM_256)),
        ("int64_t", "LSE_MIN_ALIGNMENT", _int(H._LSE_MIN_ALIGNMENT)),
        ("int64_t", "DEFAULT_ALIGNMENT", _int(H.DEFAULT_ALIGNMENT)),
        ("int64_t", "WORKLIST_HEADER_WORDS", _int(WL_HEADER_WORDS)),
        ("bool", "MIRROR_PAIRING_BUILT", _bool(P.MIRROR_PAIRING_BUILT)),
    ]
    for t, n, v in scalars:
        parts.append(f"inline constexpr {t} {n} = {v};\n")
    parts.append(_int_array("BWD_HEAD_SIZES", BWD_HEAD_SIZES))
    parts.append(_str_array("BWD_DTYPES", BWD_DTYPES))
    parts.append(_int_array("STAGE_VECS", STAGE_VECS))
    parts.append(_int_array("CONVERT_VECS", CONVERT_VECS))

    parts.append(
        "\n// hipDNN DataType JSON names and the rocKE dtype each lowers to.\n"
    )
    parts.append(
        _struct(
            "DtypeName", [("std::string_view", "hipdnn"), ("std::string_view", "rocke")]
        )
    )
    parts.append(
        _array(
            "DtypeName",
            "HIPDNN_DTYPES",
            [
                _init([_str(k), _str("unset" if v is None else v)])
                for k, v in HIPDNN_DTYPES.items()
            ],
        )
    )

    parts.append(
        "\n// Functional feature groups in order; DECLARED marks the engine policy's set.\n"
    )
    parts.append(_str_array("FEATURE_GROUPS", P.FEATURE_GROUPS))
    parts.append(
        _array(
            "bool",
            "FEATURE_GROUP_DECLARED",
            [_bool(g in policy.declared_groups) for g in P.FEATURE_GROUPS],
        )
    )

    parts.append("\n// Tuning-key vocabularies and buckets.\n")
    parts.append(_str_array("MASK_KINDS", P.MASK_KINDS))
    parts.append(_str_array("HEAD_MODES", P.HEAD_MODES))
    parts.append(_str_array("LENGTH_RELATIONS", P.LENGTH_RELATIONS))
    parts.append(_str_array("LENGTH_MODES", P.LENGTH_MODES))
    parts.append(_int_array("KV_TILE_BUCKETS", P.KV_TILE_BUCKETS))
    parts.append(_str_array("FILL_BUCKETS", P.FILL_BUCKETS))
    parts.append(
        f"inline constexpr std::string_view ANY_BUCKET = {_str(P.ANY_BUCKET)};\n"
    )
    parts.append(_str_array("SCALE_PLACEMENTS", P.SCALE_PLACEMENTS))
    parts.append(_str_array("STATIC_SCALES", STATIC_SCALES))
    parts.append(_enum("LbOrder", dict(P.LB_ORDERS)))

    parts.append("\n// Token encodings of the KMD fields.\n")
    enc_names = {
        "dtype": "Dtype",
        "seq_mode": "SeqMode",
        "dkv_mode": "DkvMode",
        "mask_class": "MaskClass",
        "dq_mode": "DqMode",
    }
    for field, cname in enc_names.items():
        parts.append(_enum(cname, dict(H.ENCODINGS[field])))
    if dict(H.ENCODINGS["main_static_scale"]) != {
        s: i for i, s in enumerate(STATIC_SCALES)
    }:
        raise AssertionError("main_static_scale must encode as the STATIC_SCALES index")

    parts.append("\n// Graph-match token names.\n")
    parts.append(
        f"inline constexpr std::string_view PREP_TOKEN_PREFIX = {_str('attention_general_bwd.prep.')};\n"
    )
    parts.append(
        f"inline constexpr std::string_view MAIN_TOKEN_PREFIX = {_str('attention_general_bwd.main.')};\n"
    )
    parts.append(_str_array("PREP_TOKENS", H.PREP_TOKENS))
    parts.append(_str_array("MAIN_TOKENS", H.MAIN_TOKENS))
    parts.append(
        _struct(
            "WorkspaceRegion",
            [("std::string_view", "plan"), ("std::string_view", "region")],
        )
    )
    parts.append(
        _array(
            "WorkspaceRegion",
            "WS_REGIONS",
            [_init([_str(k), _str(v)]) for k, v in H.WS_REGIONS.items()],
        )
    )

    parts.append(
        "\n// Per-arch facts: wave size and the default rows per CTA of prep and convert.\n"
    )
    parts.append(
        _struct("ArchFacts", [("std::string_view", "arch"), ("int64_t", "waveSize")])
    )
    parts.append(
        _array(
            "ArchFacts",
            "ARCH_FACTS",
            [
                _init([_str(a), _int(attn_bwd_arch_facts(a).wave_size)])
                for a in H.ENGINE_ARCHS
            ],
        )
    )
    parts.append(
        _struct(
            "AuxRows",
            [
                ("std::string_view", "arch"),
                ("int64_t", "headSize"),
                ("int64_t", "rowsPerCta"),
            ],
        )
    )
    parts.append(
        _array(
            "AuxRows",
            "AUX_ROWS_PER_CTA",
            [
                _init([_str(a), _int(d), _int(P._rows_per_cta(a, d))])
                for a in H.ENGINE_ARCHS
                for d in BWD_HEAD_SIZES
            ],
        )
    )

    parts.append(
        "\n// Default main geometry per (arch, head size, dtype, stage_vec, seq_mode).\n"
    )
    parts.append(
        _struct(
            "DefaultGeometry",
            [
                ("std::string_view", "arch"),
                ("int64_t", "headSize"),
                ("Dtype", "dtype"),
                ("int64_t", "stageVec"),
                ("SeqMode", "seqMode"),
                ("int64_t", "blockM"),
                ("int64_t", "blockN"),
                ("int64_t", "waves"),
            ],
        )
    )
    geo = []
    for a in H.ENGINE_ARCHS:
        for d in BWD_HEAD_SIZES:
            for dt in BWD_DTYPES:
                for sv in STAGE_VECS:
                    for seq in SEQ_MODES:
                        g = attn_bwd_default_geometry(a, d, dt, sv, seq)
                        geo.append(
                            _init(
                                [
                                    _str(a),
                                    _int(d),
                                    f"Dtype::{_enum_name(dt)}",
                                    _int(sv),
                                    f"SeqMode::{_enum_name(seq)}",
                                    _int(g["block_m"]),
                                    _int(g["block_n"]),
                                    _int(g["waves"]),
                                ]
                            )
                        )
    parts.append(_array("DefaultGeometry", "DEFAULT_GEOMETRY", geo))

    parts.append(
        "\n// Kernarg ABI per launch kind: kind is the struct format letter (Q pointer, q i64,\n"
        "// f f32, i i32).\n"
    )
    parts.append(_struct("AbiParam", [("std::string_view", "name"), ("char", "kind")]))
    for stage in STAGES:
        params = attn_bwd_params(stage)
        for _n, k in params:
            if k not in _KIND_TYPE:
                raise AssertionError(f"unknown kernarg kind {k!r}")
        parts.append(
            _array(
                "AbiParam",
                f"ABI_{stage.upper()}",
                [_init([_str(n), f"'{k}'"]) for n, k in params],
            )
        )
    return _wrap(
        "".join(parts),
        ("array", "cstdint", "string_view"),
        "Constants of the rocKE backward predicate, plan and kernel ABI.",
    )


# ---------------------------------------------------------------------------
# catalog header
# ---------------------------------------------------------------------------


def _code(table: tuple, value: str, what: str) -> int:
    if value not in table:
        raise AssertionError(f"{what} {value!r} not in {table}")
    return table.index(value)


def _tuned_key_record(arch: str, kid: int, key: tuple, s: str) -> dict:
    entry = dict(H.engine_policy().tuned_table[key])
    _a, _r, d, dtype, mask, head, rel, mode, sv, kvb, fill = key
    seq = "thd" if mode == "thd" else "batched"
    geo = attn_bwd_default_geometry(arch, d, dtype, sv, seq)
    placement = entry.get("scale_placement", "fold_ds")
    narrow = entry.get("narrow_vec", 0)
    pin = entry.get("head_pack")
    return {
        "arch": arch,
        "id": kid,
        "key": s,
        "headSize": d,
        "dtype": dtype,
        "maskKind": _code(P.MASK_KINDS, mask, "mask kind"),
        "headMode": _code(P.HEAD_MODES, head, "head mode"),
        "lengthRelation": _code(P.LENGTH_RELATIONS, rel, "length relation"),
        "lengthMode": _code(P.LENGTH_MODES, mode, "length mode"),
        "stageVec": sv,
        "kvBucket": kvb,
        "fillBucket": fill,
        "blockM": entry.get("block_m", geo["block_m"]),
        "blockN": entry.get("block_n", geo["block_n"]),
        "gSplit": int(entry.get("g_split", 1)),
        "lbOrder": P.LB_ORDERS[entry.get("lb_order", "natural")],
        "scalePlacement": _code(P.SCALE_PLACEMENTS, placement, "scale placement"),
        "staticScale": _code(
            STATIC_SCALES, entry.get("static_scale", placement), "static scale"
        ),
        "useWorklist": bool(entry.get("use_worklist", False)),
        "xcdChunk": entry.get("xcd_chunk", 0),
        "dkvSink": entry.get("dkv_sink", "atomic"),
        "dqMode": entry.get("dq_mode", "atomic"),
        "narrowVecCap": int(narrow or 0),
        "headPackPin": -1 if pin is None else int(bool(pin)),
    }


def _check_tuned_record(rec: dict, rows: list) -> None:
    mine = [r for r in rows if r.tuning_key_id == rec["id"]]
    if rec["xcdChunk"] not in (0, None):
        raise AssertionError(f"{rec['key']}: an XCD chunk needs num_xcds")
    if rec["dkvSink"] != "atomic":
        raise AssertionError(
            f"{rec['key']}: dkv_sink {rec['dkvSink']!r} needs per-slice dK/dV workspace"
            " the engine does not size or reduce"
        )
    for r in mine:
        want = {
            "main_block_m": rec["blockM"],
            "main_block_n": r.main_block_n,
            "dq_mode": rec["dqMode"],
            "main_static_scale": STATIC_SCALES[rec["staticScale"]],
            "main_worklist": rec["useWorklist"],
            "lb_interleave": r.lb_interleave,
        }
        got = {k: getattr(r, k) for k in want}
        if got != want:
            raise AssertionError(
                f"{rec['key']}: row disagrees with the key: {got} != {want}"
            )
        if (rec["lbOrder"] == P.LB_ORDERS["mirror"]) != r.lb_mirror:
            raise AssertionError(f"{rec['key']}: lb_mirror does not follow lb_order")
        if (rec["lbOrder"] == P.LB_ORDERS["interleave"]) and not r.lb_interleave:
            raise AssertionError(
                f"{rec['key']}: lb_interleave does not follow lb_order"
            )


def _layout_of(req) -> str:
    if req.is_thd:
        return "thd"
    if req.strides is not None:
        raise AssertionError(f"representative with explicit strides: {req}")
    return "bhsd"


def _representative(row, arch: str) -> dict:
    """The row's representative as a compact request, at the arch's full CU count
    when the plan there is still this row, else at the representative's own."""
    rep = row.representative
    candidates = [dataclasses.replace(rep, num_cus=FULL_CUS[arch])]
    if rep.num_cus is not None:
        candidates.append(rep)
    chosen, reproduces = candidates[-1], False
    for req in candidates:
        plan = H.engine_verdict(req, arch)[2]
        if plan is not None and plan.launches[1].kernel == row.kernels["main"]:
            mf = H.main_fields(plan)
            if mf == row.main_kmd:
                chosen, reproduces = req, True
                break
    left, right, br = chosen.resolved_band()
    return {
        "b": chosen.b,
        "hQ": chosen.h_q,
        "hK": chosen.h_k,
        "hV": chosen.h_v,
        "sQ": chosen.s_q,
        "sKv": chosen.s_kv,
        "headSize": chosen.d_qk,
        "dtype": chosen.dtype,
        "seqMode": "thd" if chosen.is_thd else "batched",
        "left": left,
        "right": right,
        "bottomRight": bool(br),
        "alignment": chosen.tensor_alignment,
        "numCus": chosen.num_cus or 0,
        "reproduces": reproduces,
        "layout": _layout_of(chosen),
    }


def catalog_data(arch: str) -> dict:
    rows = C.catalog_rows(arch)
    keys = [_tuned_key_record(arch, i, k, s) for i, k, s in H.tuned_keys(arch)]
    for rec in keys:
        _check_tuned_record(rec, rows)
    reps = [_representative(r, arch) for r in rows]
    return {"rows": rows, "keys": keys, "reps": reps}


_ROW_FIELDS = [
    ("std::string_view", "arch"),
    ("std::string_view", "pack"),
    ("Dtype", "dtype"),
    ("int64_t", "headSize"),
    ("SeqMode", "seq"),
    ("int64_t", "stageVec"),
    ("DkvMode", "dkv"),
    ("MaskClass", "mask"),
    ("DqMode", "dq"),
    ("bool", "headPack"),
    ("bool", "lbMirror"),
    ("bool", "lbInterleave"),
    ("int64_t", "tuningKeyId"),
    ("int64_t", "mainBlockM"),
    ("int64_t", "mainBlockN"),
    ("int64_t", "mainWaves"),
    ("int64_t", "mainThreads"),
    ("bool", "mainWorklist"),
    ("int64_t", "mainStaticScale"),
    ("int64_t", "prepRowsPerCta"),
    ("int64_t", "prepThreads"),
    ("int64_t", "prepNarrowVec"),
    ("bool", "prepWorklist"),
    ("int64_t", "convertRowsPerCta"),
    ("int64_t", "convertThreads"),
    ("int64_t", "convertVec"),
    ("int64_t", "dqBlockM"),
    ("int64_t", "dqBlockN"),
    ("int64_t", "dqThreads"),
    ("bool", "dqHeadLoopGroup"),
    ("std::string_view", "mainKernel"),
    ("std::string_view", "prepKernel"),
    ("std::string_view", "convertKernel"),
    ("std::string_view", "dqKernel"),
]


def _enc(cname: str, field: str, value) -> str:
    if value not in H.ENCODINGS[field]:
        raise AssertionError(f"{field} {value!r} has no encoding")
    return f"{cname}::{_enum_name(value)}"


def _row_init(row) -> str:
    dq = row.dq_spec
    vals = [
        _str(row.arch),
        _str(row.pack),
        _enc("Dtype", "dtype", row.dtype),
        _int(row.head_size),
        _enc("SeqMode", "seq_mode", row.seq_mode),
        _int(row.stage_vec),
        _enc("DkvMode", "dkv_mode", row.dkv_mode),
        _enc("MaskClass", "mask_class", row.mask_class),
        _enc("DqMode", "dq_mode", row.dq_mode),
        _bool(row.head_pack),
        _bool(row.lb_mirror),
        _bool(row.lb_interleave),
        _int(row.tuning_key_id),
        _int(row.main_block_m),
        _int(row.main_block_n),
        _int(row.main_waves),
        _int(row.main_threads),
        _bool(row.main_worklist),
        _int(STATIC_SCALES.index(row.main_static_scale)),
        _int(row.prep_rows_per_cta),
        _int(row.prep_threads),
        _int(row.prep_narrow_vec),
        _bool(row.prep_worklist),
        _int(row.convert_rows_per_cta),
        _int(row.convert_threads),
        _int(row.convert_vec),
        _int(row.dq_block_m),
        _int(row.dq_block_n),
        _int(row.dq_threads),
        _bool(dq is not None and dq.head_loop == "group"),
        _str(row.kernels["main"]),
        _str(row.kernels["prep"]),
        _str(row.kernels["convert_dk"]),
        _str(row.kernels["dq"] if dq is not None else ""),
    ]
    if len(vals) != len(_ROW_FIELDS):
        raise AssertionError("row initialiser out of sync with CatalogRow")
    return _init(vals)


_KEY_FIELDS = [
    ("std::string_view", "arch"),
    ("int64_t", "id"),
    ("std::string_view", "key"),
    ("int64_t", "headSize"),
    ("Dtype", "dtype"),
    ("int64_t", "maskKind"),
    ("int64_t", "headMode"),
    ("int64_t", "lengthRelation"),
    ("int64_t", "lengthMode"),
    ("int64_t", "stageVec"),
    ("std::string_view", "kvBucket"),
    ("std::string_view", "fillBucket"),
    ("int64_t", "blockM"),
    ("int64_t", "blockN"),
    ("int64_t", "gSplit"),
    ("int64_t", "lbOrder"),
    ("int64_t", "scalePlacement"),
    ("int64_t", "staticScale"),
    ("bool", "useWorklist"),
    ("int64_t", "xcdChunk"),
    ("DqMode", "dqMode"),
    ("int64_t", "narrowVecCap"),
    ("int64_t", "headPackPin"),
]


def _key_init(rec: dict) -> str:
    vals = []
    for ctype, name in _KEY_FIELDS:
        v = rec[name]
        if name == "dtype":
            vals.append(_enc("Dtype", "dtype", v))
        elif name == "dqMode":
            vals.append(_enc("DqMode", "dq_mode", v))
        elif ctype == "std::string_view":
            vals.append(_str(v))
        elif ctype == "bool":
            vals.append(_bool(v))
        else:
            vals.append(_int(int(v)))
    return _init(vals)


_REP_FIELDS = [
    ("size_t", "row"),
    ("int64_t", "b"),
    ("int64_t", "hQ"),
    ("int64_t", "hK"),
    ("int64_t", "hV"),
    ("int64_t", "sQ"),
    ("int64_t", "sKv"),
    ("int64_t", "headSize"),
    ("Dtype", "dtype"),
    ("SeqMode", "seqMode"),
    ("int64_t", "left"),
    ("int64_t", "right"),
    ("bool", "bottomRight"),
    ("int64_t", "alignment"),
    ("int64_t", "numCus"),
    ("bool", "reproduces"),
]


def _rep_init(index: int, rep: dict) -> str:
    vals = [str(index)]
    for ctype, name in _REP_FIELDS[1:]:
        v = rep[name]
        if name == "dtype":
            vals.append(_enc("Dtype", "dtype", v))
        elif name == "seqMode":
            vals.append(_enc("SeqMode", "seq_mode", v))
        elif ctype == "bool":
            vals.append(_bool(v))
        else:
            vals.append(_int(int(v)))
    return _init(vals)


def catalog_header() -> str:
    data = {a: catalog_data(a) for a in H.ENGINE_ARCHS}
    parts = []
    parts.append(
        "// A tuned-table key of the general family and the plan values its entry sets.\n"
        "// maskKind / headMode / lengthRelation / lengthMode index the constants header's\n"
        "// vocabularies; lbOrder is an LbOrder value; scalePlacement indexes\n"
        "// SCALE_PLACEMENTS and staticScale STATIC_SCALES; narrowVecCap 0 means no cap;\n"
        "// headPackPin -1 means the plan decides. Ids are 1-based per arch (0 = class\n"
        "// default).\n"
    )
    parts.append(_struct("TunedKey", _KEY_FIELDS))
    keys = [k for a in H.ENGINE_ARCHS for k in data[a]["keys"]]
    parts.append(_array("TunedKey", "TUNED_KEYS", [_key_init(k) for k in keys]))
    parts.append(
        "\n// One main kernel and the kernels its plan launches. Selection: (arch, class,\n"
        "// tuningKeyId, headPack) names one row. convertKernel serves the dQ launch when\n"
        "// dq is ATOMIC; dqKernel (empty otherwise) is the split dQ kernel.\n"
    )
    parts.append(_struct("CatalogRow", _ROW_FIELDS))
    rows = [r for a in H.ENGINE_ARCHS for r in data[a]["rows"]]
    parts.append(_array("CatalogRow", "CATALOG", [_row_init(r) for r in rows]))
    parts.append(
        "\n// One request per catalog row (dense rows: BHSD packed strides; THD rows: packed\n"
        "// token-major tensors). numCus 0 means no device fill hint; reproduces is false\n"
        "// when the request plans to another row at numCus.\n"
    )
    parts.append(_struct("Representative", _REP_FIELDS))
    reps, base = [], 0
    for a in H.ENGINE_ARCHS:
        for i, rep in enumerate(data[a]["reps"]):
            reps.append(_rep_init(base + i, rep))
        base += len(data[a]["rows"])
    parts.append(_array("Representative", "REPRESENTATIVES", reps))
    return _wrap(
        "".join(parts),
        ("array", "cstddef", "cstdint", "string_view"),
        "Tuned keys and kernel catalog of the engine; include after the constants header.",
    ).replace(
        "#include <array>",
        f'#include "{CONSTANTS_HEADER}"\n\n#include <array>',
        1,
    )


def catalog_summary() -> dict:
    """Per-arch counts of the catalog (rows per pack, distinct binaries per kind)."""
    out = {}
    for a in H.ENGINE_ARCHS:
        rows = C.catalog_rows(a)
        packs: dict[str, int] = {}
        for r in rows:
            packs[r.pack] = packs.get(r.pack, 0) + 1
        packs[C.pack_of(a, "prep")] = len(C.prep_rows(a))
        binaries = {
            "main": len({r.kernels["main"] for r in rows}),
            "prep": len({p.kernel for p in C.prep_rows(a)}),
            "convert": len({r.kernels["convert_dk"] for r in rows}),
            "dq": len({r.kernels["dq"] for r in rows if r.dq_spec is not None}),
        }
        binaries["total"] = sum(binaries.values())
        out[a] = {"rows_per_pack": packs, "binaries": binaries}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def find_clang_format(explicit: str | None = None) -> str | None:
    for cand in (
        explicit,
        os.environ.get("HKP_CLANG_FORMAT"),
        "clang-format-18",
        "clang-format",
    ):
        if not cand:
            continue
        path = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if path:
            return path
    return None


def clang_format(text: str, binary: str, assume: Path) -> str:
    out = subprocess.run(
        [binary, "--style=file", f"--assume-filename={assume}"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        cwd=assume.parent,
    )
    return out.stdout


def render(out_dir: Path, binary: str) -> dict[str, str]:
    return {
        name: clang_format(text, binary, out_dir / name)
        for name, text in (
            (CONSTANTS_HEADER, constants_header()),
            (CATALOG_HEADER, catalog_header()),
        )
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--check", action="store_true", help="fail when a committed header differs"
    )
    ap.add_argument("--clang-format", default=None)
    ap.add_argument(
        "--summary", action="store_true", help="print the catalog counts as JSON"
    )
    args = ap.parse_args(argv)
    if args.summary:
        print(json.dumps(catalog_summary(), indent=2, sort_keys=True))
        return 0
    binary = find_clang_format(args.clang_format)
    if binary is None:
        print("general_bwd_constants: clang-format not found", file=sys.stderr)
        return 2
    texts = render(args.out_dir.resolve(), binary)
    stale = []
    for name, text in texts.items():
        path = args.out_dir / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(str(path))
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        print(
            "stale generated headers (rerun general_bwd_constants.py):", file=sys.stderr
        )
        for s in stale:
            print(f"  {s}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
