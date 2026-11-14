# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write the device cases of the ``hipkernel:AttentionGeneralBwd`` GPU integration tests.

The engine selects its kernels from the graph and the device's compute-unit count, so
a catalog row is reachable on a device only through a shape whose tuning key lands on
it at that device's count. For every catalog row of each arch this script finds the
cheapest graph that selects it on the arch's full device, plus named cases the tests
use for features (THD head modes, windows, scales, layouts, stats forms, declines,
capture). Each case is lowered exactly as the C++ test builds it
(:func:`case_graph`), so the expected kernels, workspace and verdict come from the
engine's own Python reference (``dispatch/attention_bwd/hipdnn.py``).

Usage::

    python3 general_bwd_gpu_cases.py --out-dir <src/integration_tests/kernel_ingestor_engine> [--check]

The header is formatted with ``clang-format --style=file``; ``--check`` compares the
formatted text.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_HKP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _sub in ("rocke/platform/python", "rocke/library"):
    _p = str(_HKP / _sub)
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
os.environ.setdefault("ROCKE_BACKEND", "python")

from dispatch.attention_bwd import hipdnn as H  # noqa: E402
from dispatch.attention_bwd import hipdnn_catalog as C  # noqa: E402
from general_bwd_constants import clang_format, find_clang_format  # noqa: E402

HEADER = "AttentionGeneralBwdGpuCases.generated.hpp"
TOOL = "dnn-providers/hip-kernel-provider/descriptor-packaging/tools/general_bwd_gpu_cases.py"
NAMESPACE = "hip_kernel_provider::kernel_ingestor_engine::integration::general_bwd_gpu"
BUNDLE = (
    _HKP / "src/engines/kernel_ingestor_engine/descriptors/rocKE/attention_general_bwd"
)
# Full compute-unit count of the device each arch's suite runs on.
FULL_CUS = {"gfx942": 304, "gfx950": 256}
# Batch sizes the row search tries on each candidate shape (the fill bucket follows B).
BATCHES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
# Shapes above this many multiply-adds per GEMM are not searched.
MAX_COST = 4 * 10**10
QUICK_CAP = 40
UIDS = {
    "q": 1,
    "k": 2,
    "v": 3,
    "o": 4,
    "do": 5,
    "lse": 6,
    "dq": 7,
    "dk": 8,
    "dv": 9,
    "offsets_q": 10,
    "offsets_kv": 11,
}
_Q_SIDE = ("q", "o", "do", "dq")
_HIPDNN_DTYPE = {"fp16": "half", "bf16": "bfloat16"}


@dataclass(frozen=True)
class Case:
    """One graph of the GPU suite, in the C++ builder's terms."""

    b: int
    h_q: int
    h_k: int
    h_v: int
    s_q: int
    s_kv: int
    d: int
    dtype: str
    layout: str  # "bhsd", "bshd" or "thd" (token-major, ragged on the two tables)
    # "dense" [B,H,S,1]; on Q's table: "ragged" token-major [T,H,1] or
    # "ragged_head_major" [H,T]
    stats: str = "dense"
    left: int = -1  # hipDNN left_bound, -1 = none
    right: int = -1  # hipDNN right_bound, -1 = none
    bottom_right: bool = False
    alignment: int = 16
    scale: float | None = None  # None: no scale on the graph (hipDNN's 1.0)

    @property
    def cost(self) -> int:
        return self.b * self.h_q * self.s_q * self.s_kv * self.d


def _strides(c: Case, role: str) -> tuple[int, int, int, int]:
    q_side = role in _Q_SIDE
    h = c.h_q if q_side else (c.h_k if role in ("k", "dk") else c.h_v)
    s = c.s_q if q_side else c.s_kv
    if c.layout == "bhsd":
        return (h * s * c.d, s * c.d, c.d, 1)
    return (s * h * c.d, c.d, h * c.d, 1)


def case_graph(c: Case) -> dict:
    """The hipDNN graph JSON of ``c``, as the C++ test builds it."""
    thd = c.layout == "thd"
    tensors = []

    def add(name, uid, dtype, dims, strides, **extra):
        entry = {
            "uid": uid,
            "name": name,
            "data_type": dtype,
            "dims": list(dims),
            "strides": list(strides),
            "virtual": False,
            "is_runtime_pass_by_value": False,
            "alignment": 16,
            "ragged_offset_multiplier": 1,
        }
        entry.update(extra)
        tensors.append(entry)

    for role in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        q_side = role in _Q_SIDE
        h = c.h_q if q_side else (c.h_k if role in ("k", "dk") else c.h_v)
        s = c.s_q if q_side else c.s_kv
        st = _strides(c, role)
        extra = {"alignment": c.alignment}
        if thd:
            extra.update(
                ragged_offset_tensor_uid=UIDS["offsets_q" if q_side else "offsets_kv"],
                ragged_offset_multiplier=st[2],
            )
        add(role, UIDS[role], _HIPDNN_DTYPE[c.dtype], (c.b, h, s, c.d), st, **extra)
    lse_dims = (c.b, c.h_q, c.s_q, 1)
    if c.stats == "ragged":
        add(
            "stats",
            UIDS["lse"],
            "float",
            lse_dims,
            (c.s_q * c.h_q, 1, c.h_q, 1),
            ragged_offset_tensor_uid=UIDS["offsets_q"],
            ragged_offset_multiplier=c.h_q,
        )
    elif c.stats == "ragged_head_major":
        add(
            "stats",
            UIDS["lse"],
            "float",
            lse_dims,
            (c.s_q, c.b * c.s_q, 1, 1),
            ragged_offset_tensor_uid=UIDS["offsets_q"],
            ragged_offset_multiplier=1,
        )
    else:
        add("stats", UIDS["lse"], "float", lse_dims, (c.h_q * c.s_q, c.s_q, 1, 1))
    if thd:
        for table in ("offsets_q", "offsets_kv"):
            add(table, UIDS[table], "int32", (c.b + 1, 1, 1, 1), (1, 1, 1, 1))
    inputs = {k: None for k in H._OPTIONAL_INPUTS}
    inputs.update(
        q_tensor_uid=UIDS["q"],
        k_tensor_uid=UIDS["k"],
        v_tensor_uid=UIDS["v"],
        o_tensor_uid=UIDS["o"],
        do_tensor_uid=UIDS["do"],
        stats_tensor_uid=UIDS["lse"],
    )
    params = {
        "alibi_mask": False,
        "padding_mask": False,
        "causal_mask": False,
        "causal_mask_bottom_right": False,
        "dropout_probability": None,
        "attn_scale_value": c.scale,
        "left_bound": None if c.left == -1 else c.left,
        "right_bound": None if c.right == -1 else c.right,
        "diagonal_alignment": "BOTTOM_RIGHT" if c.bottom_right else "TOP_LEFT",
    }
    node = {
        "name": "SdpaBwd_0",
        "type": H.NODE_TYPE,
        "compute_data_type": "float",
        "inputs": inputs,
        "outputs": {
            "dq_tensor_uid": UIDS["dq"],
            "dk_tensor_uid": UIDS["dk"],
            "dv_tensor_uid": UIDS["dv"],
            "dbias_tensor_uid": None,
        },
        "parameters": params,
    }
    return {"name": "SdpaBackward", "nodes": [node], "tensors": tensors}


def verdict(c: Case, arch: str):
    """``(code, plan)`` of the engine for ``c`` on ``arch``'s full device."""
    try:
        req = H.from_hipdnn_graph(case_graph(c), arch, num_cus=FULL_CUS[arch])
    except H.HipdnnLoweringError:
        return H.LOWERING, None
    code, _detail, plan = H.engine_verdict(req, arch)
    return code, plan


def _f32(x: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", x))[0]


def case_of(req) -> Case | None:
    """The suite case of a catalog request (``None``: not a layout the suite builds)."""
    if req.is_thd:
        layout = "thd"
    elif req.strides is None:
        layout = "bhsd"
    else:
        st = req.tensor_strides("q")
        layout = "bshd" if st[1] == req.d_qk and st[2] == req.h_q * req.d_qk else None
        if layout is None:
            return None
    left, right, br = req.resolved_band()
    return Case(
        b=req.b,
        h_q=req.h_q,
        h_k=req.h_k,
        h_v=req.h_v,
        s_q=req.s_q,
        s_kv=req.s_kv,
        d=req.d_qk,
        dtype=req.dtype,
        layout=layout,
        stats="ragged" if layout == "thd" else "dense",
        left=-1 if left == -1 else left + 1,
        right=right,
        bottom_right=br,
        alignment=req.tensor_alignment,
        scale=_f32(req.resolved_scale()),
    )


def _row_index(arch: str, plan) -> int:
    mf = H.main_fields(plan)
    key = (
        tuple(mf[f] for f in C.CLASS_FIELDS),
        H.tuning_key_id(arch, H.plan_table_key(plan)),
    )
    for i, row in enumerate(C.catalog_rows(arch)):
        if (row.class_key, row.tuning_key_id) == key:
            return i
    raise AssertionError(f"{arch}: plan selects no catalog row: {key}")


def main_kernel_name(row) -> str:
    tk = "default" if row.tuning_key_id == 0 else f"k{row.tuning_key_id}"
    return (
        f"main.{row.arch}.{row.dtype}_d{row.head_size}_{row.seq_mode}"
        f"_sv{row.stage_vec}_{row.dkv_mode}_{row.mask_class}_{row.dq_mode}"
        f"_hp{int(row.head_pack)}_{tk}"
    )


def prep_kernel_name(arch: str, plan) -> str:
    f = H.prep_fields(plan)
    return (
        f"prep.{arch}.{f['dtype']}_d{f['head_size']}_{f['seq_mode']}"
        f"_sv{f['stage_vec']}_wl{int(f['worklist'])}_r{f['rows_per_cta']}"
        f"_nv{f['narrow_vec']}"
    )


def kernel_ids(bundle: Path = BUNDLE) -> dict[str, str]:
    """KDP kernel name -> kernel id over the committed bundle."""
    out = {}
    roots = [bundle, bundle / "artifacts"]
    for root in roots:
        for path in sorted(root.glob("*.kdp.json")) if root.is_dir() else []:
            for k in json.loads(path.read_text(encoding="utf-8"))["kernelDescriptors"]:
                out[k["name"]] = k["id"]
    return out


@dataclass(frozen=True)
class Entry:
    name: str
    case: Case
    code: str
    row: int = -1
    main_name: str = ""
    main_id: str = ""
    prep_id: str = ""
    workspace: int = 0
    launches: int = 0
    quick: bool = False
    tags: tuple = ()


def _entry(name: str, c: Case, arch: str, ids: dict, quick=False, tags=()) -> Entry:
    code, plan = verdict(c, arch)
    if plan is None:
        return Entry(name, c, code, quick=quick, tags=tags)
    row = C.catalog_rows(arch)[_row_index(arch, plan)]
    main = main_kernel_name(row)
    prep = prep_kernel_name(arch, plan)
    for n in (main, prep):
        if n not in ids:
            raise AssertionError(f"{arch}: kernel {n} is not in the committed KDPs")
    return Entry(
        name,
        c,
        "OK",
        _row_index(arch, plan),
        main,
        ids[main],
        ids[prep],
        int(plan.workspace_bytes),
        len(plan.launches),
        quick,
        tags,
    )


def row_cases(arch: str) -> dict[int, Case]:
    """The cheapest case selecting each catalog row on ``arch``'s full device."""
    rows = C.catalog_rows(arch)
    bases = [r.representative for r in rows] + C.default_requests(arch)
    bases += [q for _k, q in C.tuned_requests(arch)]
    seen, best = set(), {}
    for req in bases:
        base = case_of(req)
        if base is None:
            continue
        for b in BATCHES:
            c = dataclasses.replace(base, b=b)
            if c in seen or c.cost > MAX_COST:
                continue
            seen.add(c)
            _code, plan = verdict(c, arch)
            if plan is None:
                continue
            i = _row_index(arch, plan)
            known = best.get(i)
            if known is None or (c.cost, repr(c)) < (known.cost, repr(known)):
                best[i] = c
    missing = [i for i in range(len(rows)) if i not in best]
    if missing:
        raise AssertionError(
            f"{arch}: no searched shape selects rows {missing} at "
            f"{FULL_CUS[arch]} CUs"
        )
    return best


def _features(arch: str, row_index: int) -> set:
    r = C.catalog_rows(arch)[row_index]
    out = {(f, getattr(r, f)) for f in C.CLASS_FIELDS}
    out.add(("tuned", r.tuning_key_id != 0))
    out.add(("d_seq", (r.head_size, r.seq_mode)))
    out.add(("d_dtype", (r.head_size, r.dtype)))
    return out


def _quick_rows(arch: str, cases: dict[int, Case]) -> set[int]:
    """A small set of rows covering every class value, cheapest first."""
    want = set().union(*(_features(arch, i) for i in cases))
    chosen: set[int] = set()
    covered: set = set()
    order = sorted(cases, key=lambda i: (cases[i].cost, i))
    while covered != want and len(chosen) < QUICK_CAP:
        best = max(order, key=lambda i: len(_features(arch, i) - covered))
        gain = _features(arch, best) - covered
        if not gain:
            break
        chosen.add(best)
        covered |= gain
    return chosen


def _named(arch: str) -> list[tuple[str, Case, tuple]]:
    """Feature cases beyond the rows (both arches)."""
    s = _f32(0.125)
    thd = dict(layout="thd", stats="ragged", alignment=16, scale=s)
    return [
        (
            "thd_gqa_per_side_multipliers",
            Case(3, 8, 2, 2, 96, 160, 64, "bf16", **thd),
            ("thd",),
        ),
        ("thd_mqa", Case(2, 8, 1, 1, 128, 128, 64, "fp16", **thd), ("thd",)),
        ("thd_hk_ne_hv", Case(2, 8, 4, 2, 80, 192, 64, "bf16", **thd), ("thd",)),
        (
            "thd_stats_head_major",
            Case(
                2, 8, 2, 2, 64, 96, 64, "fp16", **dict(thd, stats="ragged_head_major")
            ),
            ("thd",),
        ),
        (
            "thd_window_bottom_right",
            Case(
                2,
                4,
                2,
                2,
                96,
                200,
                128,
                "bf16",
                left=33,
                right=0,
                bottom_right=True,
                **thd,
            ),
            ("thd",),
        ),
        (
            "thd_causal_top_left_d32",
            Case(3, 4, 4, 4, 64, 64, 32, "fp16", right=0, **thd),
            ("thd",),
        ),
        (
            "thd_stage_vec1",
            Case(2, 4, 2, 2, 64, 96, 64, "bf16", **dict(thd, alignment=8)),
            ("thd",),
        ),
        (
            "window_top_left",
            Case(2, 4, 4, 4, 128, 256, 64, "bf16", "bhsd", left=40, right=16, scale=s),
            (),
        ),
        (
            "window_bottom_right_unequal",
            Case(
                2,
                4,
                2,
                2,
                64,
                300,
                128,
                "fp16",
                "bshd",
                left=50,
                right=7,
                bottom_right=True,
                scale=s,
            ),
            (),
        ),
        (
            "causal_bottom_right_short_query",
            Case(
                2,
                8,
                2,
                2,
                48,
                257,
                64,
                "bf16",
                "bhsd",
                right=0,
                bottom_right=True,
                scale=s,
            ),
            (),
        ),
        (
            "bshd_layout_hk_ne_hv",
            Case(2, 8, 4, 2, 100, 100, 128, "bf16", "bshd", scale=s),
            (),
        ),
        (
            "stage_vec1_mqa",
            Case(2, 8, 1, 1, 64, 130, 64, "fp16", "bhsd", alignment=8, scale=s),
            (),
        ),
        (
            "explicit_scale",
            Case(2, 4, 4, 4, 64, 96, 64, "bf16", "bhsd", scale=_f32(0.3)),
            (),
        ),
        ("no_scale_is_unit_scale", Case(2, 4, 4, 4, 64, 96, 64, "fp16", "bhsd"), ()),
        (
            "capture_batched",
            Case(2, 4, 2, 2, 64, 128, 64, "bf16", "bhsd", scale=s),
            ("capture",),
        ),
        (
            "capture_thd",
            Case(4, 8, 4, 2, 96, 160, 64, "bf16", **thd),
            ("capture", "thd"),
        ),
        (
            "misaligned_pointer",
            Case(1, 2, 2, 2, 64, 64, 64, "bf16", "bhsd", scale=s),
            ("misaligned",),
        ),
        (
            "thd_dense_stats_declines",
            Case(2, 4, 4, 4, 64, 64, 64, "bf16", **dict(thd, stats="dense")),
            ("decline",),
        ),
    ]


def arch_entries(arch: str) -> list[Entry]:
    ids = kernel_ids()
    cases = row_cases(arch)
    quick = _quick_rows(arch, cases)
    out = []
    for i in sorted(cases):
        e = _entry(f"row{i:03d}", cases[i], arch, ids, quick=i in quick)
        if e.row != i:
            raise AssertionError(f"{arch}: row {i} case selects row {e.row}")
        out.append(e)
    for name, c, tags in _named(arch):
        e = _entry(name, c, arch, ids, quick=True, tags=tags)
        if ("decline" in tags) == (e.code == "OK"):
            raise AssertionError(f"{arch}: {name} verdict {e.code}")
        out.append(e)
    # Prep builds a device work list that main reads: replay those plans under capture too.
    for i in sorted(cases):
        _code, plan = verdict(cases[i], arch)
        if H.prep_fields(plan)["worklist"]:
            name = f"capture_row{i:03d}_worklist"
            out.append(
                _entry(name, cases[i], arch, ids, quick=True, tags=("capture", "thd"))
            )
    return out


# ---------------------------------------------------------------------------
# C++
# ---------------------------------------------------------------------------


def _s(x: str) -> str:
    return json.dumps(x)


def _b(x: bool) -> str:
    return "true" if x else "false"


def _record(e: Entry) -> str:
    c = e.case
    scale = "-1.0F" if c.scale is None else f"{c.scale!r}F"
    if c.scale is not None and float(scale[:-1]) != c.scale:
        raise AssertionError(f"{e.name}: scale {c.scale} does not print exactly")
    vals = [
        _s(e.name),
        str(c.b),
        str(c.h_q),
        str(c.h_k),
        str(c.h_v),
        str(c.s_q),
        str(c.s_kv),
        str(c.d),
        _b(c.dtype == "bf16"),
        f"Layout::{c.layout.upper()}",
        f"Stats::{c.stats.upper()}",
        str(c.left),
        str(c.right),
        _b(c.bottom_right),
        str(c.alignment),
        scale,
        _s(e.code),
        str(e.row),
        _s(e.main_name),
        _s(e.main_id),
        _s(e.prep_id),
        str(e.workspace),
        str(e.launches),
        _b(e.quick),
        _s(",".join(e.tags)),
    ]
    return "{" + ", ".join(vals) + "}"


def header() -> str:
    parts = []
    for arch in ("gfx942", "gfx950"):
        entries = arch_entries(arch)
        body = "".join(f"    {_record(e)},\n" for e in entries)
        parts.append(
            f"// clang-format off\n"
            f"inline constexpr std::array<GpuCase, {len(entries)}> "
            f"CASES_{arch.upper()} = {{{{\n{body}}}}};\n// clang-format on\n"
        )
    cus = "".join(
        f"inline constexpr int64_t FULL_CUS_{a.upper()} = {n};\n"
        for a, n in FULL_CUS.items()
    )
    return (
        "// Copyright © Advanced Micro Devices, Inc., or its affiliates.\n"
        "// SPDX-License-Identifier:  MIT\n\n"
        f"// Generated by {TOOL}; do not edit.\n"
        "// Device cases of the hipkernel:AttentionGeneralBwd GPU tests.\n\n"
        "#pragma once\n\n"
        "#include <array>\n#include <cstdint>\n#include <string_view>\n\n"
        f"namespace {NAMESPACE}\n{{\n\n"
        "enum class Layout\n{\n    BHSD,\n    BSHD,\n    THD,\n};\n\n"
        "enum class Stats\n{\n    DENSE,\n    RAGGED,\n    RAGGED_HEAD_MAJOR,\n};\n\n"
        "// One graph of the suite. Bounds are the graph's (-1 = none); scale < 0 sets no\n"
        "// scale. code is the expected verdict; for an accept, row is the catalog row of\n"
        "// the arch, mainKernel its KDP kernel name, the ids the kernels each slot\n"
        "// selects, workspaceBytes the workspace query and launches the kernels one\n"
        "// execute launches, on a device with the arch's full compute-unit count.\n"
        "struct GpuCase\n{\n"
        "    std::string_view name;\n"
        "    int64_t b;\n    int64_t hQ;\n    int64_t hK;\n    int64_t hV;\n"
        "    int64_t sQ;\n    int64_t sKv;\n    int64_t d;\n    bool bf16;\n"
        "    Layout layout;\n    Stats stats;\n"
        "    int64_t leftBound;\n    int64_t rightBound;\n    bool bottomRight;\n"
        "    int64_t alignment;\n    float scale;\n"
        "    std::string_view code;\n    int64_t row;\n"
        "    std::string_view mainKernel;\n    std::string_view mainKernelId;\n"
        "    std::string_view prepKernelId;\n    int64_t workspaceBytes;\n"
        "    int64_t launches;\n"
        "    bool quick;\n    std::string_view tags;\n};\n\n"
        f"{cus}\n" + "\n".join(parts) + f"\n}} // namespace {NAMESPACE}\n"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clang-format", default=None)
    args = ap.parse_args(argv)
    binary = find_clang_format(args.clang_format)
    if binary is None:
        print("general_bwd_gpu_cases: clang-format not found", file=sys.stderr)
        return 2
    path = args.out_dir / HEADER
    text = clang_format(header(), binary, path.resolve())
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            print(f"stale {path} (rerun general_bwd_gpu_cases.py)", file=sys.stderr)
            return 1
        return 0
    path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
