# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build the CDNA SDPA backward code objects from ``Spec`` strings (``_sdpa_bwd_spec``).

Two entry points share one compile:

    python -m generators._sdpa_bwd_cdna_build --specs specs.csv --arch gfx950 --out-dir /path/to/empty
    python -m generators.gen_sdpa_bwd_<kind> --arch gfx950 --out-dir /path/to/content-root

The first builds the Specs of a CSV (``spec`` column). The second is the generator of
one backward op (``run_op``): it builds the op's rows of ``_instances`` and writes
``<content-root>/<op>/gfx950/`` with the objects, ``manifest.json`` and ``SOURCE.md``,
the layout of the other generators. Each spec is compiled once, in its own process slot
(``_parallel``: ``FLYDSL_GEN_JOBS`` sets the worker count), and the object is recovered
from the IR dump. A compile prints the SGPR greedy-allocation MIR (``print-after=greedy``)
into a file of its own, and ``tools/sdpa_bwd_gate.py`` gates the delivered object and
that dump in a subprocess: the resource row, the #1087 detectors and the R-MIR scan.
Per spec the outcome is:

* ``ok``          the object was extracted, checked against its declared kernarg
                  layout and gated;
* ``dual_object`` the dump holds two different objects and the extractor refused to choose;
* ``error``       anything else (an invalid spec, a build, compile, check or gate failure).

Rebuilding from identical sources can yield a different object, or a dump holding two
different objects: LLVM's AMDGPU backend is not deterministic for some kernels
(the dq-causal family and five known bimodal specs, ``tests/data/sdpa_bwd_known_flaky.csv``). Those outcomes are
reported, never retried; one build per spec. Comparing an object against a pin, and
deciding which outcomes are red, is the caller's job (``tools/sdpa_bwd_compare.py``).

The ``bias`` rows of ``sdpa_bwd_merged`` build one object with two kernels: the merged kernel and the
dBias reduce kernel (``reduce_symbol``). Its per-kernel gate (``tools/sdpa_bwd_gate.py``) holds the merged
kernel's function bytes and resource row to the standalone merged pin and the reduce kernel's resource row
to the reduce pin. ``generate`` for that op ships pins (``--reference-dir``): a non-bias row ships its pin's
bytes and its rebuild only verifies it; a bias row ships the combined rebuild. A row that does not build
(``error``; for a combined row also ``dual_object``) or a red per-kernel gate stops the run, and nothing
is shipped.

The kernels are the ones under ``kernels_src/kernels/attention_bwd_cdna/``. Each is
traced with no tensors: the launchers take raw pointers and integers, so the arguments
are synthesised from the launcher's own signature. Every build constructs a fresh
launcher (a reused JitFunction is a cache hit) and compiles into a fresh dump directory.

The exit status is 0 when every spec is ``ok`` or ``dual_object`` (a spill finding is
recorded in the manifest and judged by the compare tool), and 1 when any is ``error``,
a compile worker dies, or the arch is refused.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _flydsl_env as env
from ._codeobject import (
    GeneratorError,
    arg_records,
    describe,
    hkp_pack_module,
    kernel_names,
    verify_arch,
)
from ._extract_hsaco import HsacoExtractionError, hsaco_from_dump
from ._instances import SDPA_BWD_MERGED_REDUCE_SYMBOL, instances_for
from ._manifest import FLYDSL_DIR, refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all
from ._sdpa_bwd_spec import (
    KERNELS,
    LADDER,
    Spec,
    rung_below,
    to_spec,
)  # noqa: F401  (LADDER is part of this module's surface)

# The arches each op is built for. One entry per op, by `sdpa_bwd_<kernel>`.
_OP_ARCHES = {f"sdpa_bwd_{kernel}": ("gfx950",) for kernel in KERNELS}

# `_extract_hsaco.extract_bin_attr` words a dump with two different objects this way.
_DUAL_OBJECT_MARK = "embedded objects are not identical"

_RESULT_COLUMNS = (
    "hash",
    "spec",
    "status",
    "sha256",
    "detail",
    "spill_findings",
    "other_findings",
)

# The gates of one delivered object (resources, #1087, R-MIR), run as a subprocess
# so that editing them restamps no object.
_GATE_TOOL = FLYDSL_DIR / "tools" / "sdpa_bwd_gate.py"

# One kernel argument is its pointer (8 bytes, `global_buffer`) or a by-value scalar.
_ARG_KINDS = {
    "ptr": ("global_buffer", 8),
    "i32": ("by_value", 4),
    "f32": ("by_value", 4),
    "i64": ("by_value", 8),
}


def _signature(params: tuple[tuple[str, str], ...]) -> tuple[tuple, tuple]:
    """The (kind, size, offset) layout and the argument names of a parameter list."""
    layout, names, offset = [], [], 0
    for name, kind in params:
        value_kind, size = _ARG_KINDS[kind]
        offset = (offset + size - 1) // size * size
        layout.append((value_kind, size, offset))
        names.append(name)
        offset += size
    return tuple(layout), tuple(names)


# The kernels' own parameter lists, in order: the `@flyc.kernel` signatures under
# `kernels_src/kernels/attention_bwd_cdna/`. The prep objects with `sink=1` are the
# `fmha_bwd_prep_sink_gfx950_kernel`, which adds `sink` and `sinkterm`. The native
# dispatch marshals by position, so a parameter list that changes is a change of the
# declaration here, of the dispatch and of the descriptors together.
_PARAMS: dict[str, tuple[tuple[str, str], ...]] = {
    "prep": (
        ("O", "ptr"),
        ("DO", "ptr"),
        ("LSE", "ptr"),
        ("LSE_ws", "ptr"),
        ("Delta", "ptr"),
        ("seqinfo_q0", "ptr"),
        ("seqinfo_q1", "ptr"),
        ("varlen_bits", "i32"),
        ("num_seqlens", "i32"),
        ("max_seqlen_q", "i32"),
        ("num_head_q", "i32"),
        ("hdim_vo", "i32"),
        ("stride_o_batch", "i64"),
        ("stride_o_head", "i64"),
        ("stride_o_seq", "i64"),
        ("stride_do_batch", "i64"),
        ("stride_do_head", "i64"),
        ("stride_do_seq", "i64"),
    ),
    "prep_sink": (
        ("O", "ptr"),
        ("DO", "ptr"),
        ("LSE", "ptr"),
        ("LSE_ws", "ptr"),
        ("Delta", "ptr"),
        ("seqinfo_q0", "ptr"),
        ("seqinfo_q1", "ptr"),
        ("varlen_bits", "i32"),
        ("num_seqlens", "i32"),
        ("max_seqlen_q", "i32"),
        ("num_head_q", "i32"),
        ("hdim_vo", "i32"),
        ("stride_o_batch", "i64"),
        ("stride_o_head", "i64"),
        ("stride_o_seq", "i64"),
        ("stride_do_batch", "i64"),
        ("stride_do_head", "i64"),
        ("stride_do_seq", "i64"),
        ("sink", "ptr"),
        ("sinkterm", "ptr"),
    ),
    "dbias_reduce": (
        ("DB", "ptr"),
        ("DBias", "ptr"),
        ("num_batch", "i32"),
        ("num_head", "i32"),
        ("seqlen_q", "i32"),
        ("seqlen_k", "i32"),
        ("reduce_batch", "i32"),
        ("reduce_head", "i32"),
        ("stride_ws_batch", "i64"),
        ("stride_ws_head", "i64"),
        ("stride_ws_seq", "i64"),
        ("stride_out_batch", "i64"),
        ("stride_out_head", "i64"),
        ("stride_out_seq", "i64"),
    ),
    "dsink_reduce": (
        ("SinkTerm", "ptr"),
        ("DSink", "ptr"),
        ("batch_size", "i32"),
        ("seqinfo_q0", "ptr"),
        ("seqinfo_q1", "ptr"),
        ("varlen_bits", "i32"),
        ("num_seqlens", "i32"),
        ("max_seqlen_q", "i32"),
        ("num_head_q", "i32"),
    ),
    "dkdv": (
        ("Q", "ptr"),
        ("K", "ptr"),
        ("V", "ptr"),
        ("B", "ptr"),
        ("DO", "ptr"),
        ("DK", "ptr"),
        ("DV", "ptr"),
        ("LSE", "ptr"),
        ("Delta", "ptr"),
        ("seqinfo_q0", "ptr"),
        ("seqinfo_q1", "ptr"),
        ("seqinfo_k0", "ptr"),
        ("seqinfo_k1", "ptr"),
        ("varlen_bits", "i32"),
        ("num_seqlens", "i32"),
        ("max_seqlen_q", "i32"),
        ("max_seqlen_k", "i32"),
        ("window_left", "i32"),
        ("window_right", "i32"),
        ("philox_seed_ptr", "ptr"),
        ("philox_offset1", "ptr"),
        ("philox_offset2", "i64"),
        ("idropout_p", "i32"),
        ("dropout_scale", "f32"),
        ("num_head_q", "i32"),
        ("num_head_k", "i32"),
        ("hdim_qk", "i32"),
        ("hdim_vo", "i32"),
        ("sm_scale", "f32"),
        ("stride_q_batch", "i64"),
        ("stride_q_head", "i64"),
        ("stride_q_seq", "i64"),
        ("stride_k_batch", "i64"),
        ("stride_k_head", "i64"),
        ("stride_k_seq", "i64"),
        ("stride_v_batch", "i64"),
        ("stride_v_head", "i64"),
        ("stride_v_seq", "i64"),
        ("stride_do_batch", "i64"),
        ("stride_do_head", "i64"),
        ("stride_do_seq", "i64"),
        ("stride_dk_batch", "i64"),
        ("stride_dk_head", "i64"),
        ("stride_dk_seq", "i64"),
        ("stride_dv_batch", "i64"),
        ("stride_dv_head", "i64"),
        ("stride_dv_seq", "i64"),
        ("stride_b_batch", "i64"),
        ("stride_b_head", "i64"),
        ("stride_b_seq_q", "i64"),
        ("rt_flags", "i32"),
    ),
    "dq": (
        ("Q", "ptr"),
        ("K", "ptr"),
        ("V", "ptr"),
        ("B", "ptr"),
        ("DO", "ptr"),
        ("DQ", "ptr"),
        ("DB", "ptr"),
        ("LSE", "ptr"),
        ("Delta", "ptr"),
        ("seqinfo_q0", "ptr"),
        ("seqinfo_q1", "ptr"),
        ("seqinfo_k0", "ptr"),
        ("seqinfo_k1", "ptr"),
        ("varlen_bits", "i32"),
        ("num_seqlens", "i32"),
        ("max_seqlen_q", "i32"),
        ("max_seqlen_k", "i32"),
        ("window_left", "i32"),
        ("window_right", "i32"),
        ("philox_seed_ptr", "ptr"),
        ("philox_offset1", "ptr"),
        ("philox_offset2", "i64"),
        ("idropout_p", "i32"),
        ("dropout_scale", "f32"),
        ("num_head_q", "i32"),
        ("num_head_k", "i32"),
        ("hdim_qk", "i32"),
        ("hdim_vo", "i32"),
        ("sm_scale", "f32"),
        ("stride_q_batch", "i64"),
        ("stride_q_head", "i64"),
        ("stride_q_seq", "i64"),
        ("stride_k_batch", "i64"),
        ("stride_k_head", "i64"),
        ("stride_k_seq", "i64"),
        ("stride_v_batch", "i64"),
        ("stride_v_head", "i64"),
        ("stride_v_seq", "i64"),
        ("stride_do_batch", "i64"),
        ("stride_do_head", "i64"),
        ("stride_do_seq", "i64"),
        ("stride_dq_batch", "i64"),
        ("stride_dq_head", "i64"),
        ("stride_dq_seq", "i64"),
        ("stride_b_batch", "i64"),
        ("stride_b_head", "i64"),
        ("stride_b_seq_q", "i64"),
        ("stride_db_batch", "i64"),
        ("stride_db_head", "i64"),
        ("stride_db_seq_q", "i64"),
        ("rt_flags", "i32"),
    ),
    "merged": (
        ("Q", "ptr"),
        ("K", "ptr"),
        ("V", "ptr"),
        ("B", "ptr"),
        ("O", "ptr"),
        ("DO", "ptr"),
        ("DK", "ptr"),
        ("DV", "ptr"),
        ("DQ", "ptr"),
        ("DB", "ptr"),
        ("LSE", "ptr"),
        ("batch_size", "i32"),
        ("seqinfo_q0", "ptr"),
        ("seqinfo_q1", "ptr"),
        ("seqinfo_k0", "ptr"),
        ("seqinfo_k1", "ptr"),
        ("varlen_bits", "i32"),
        ("num_seqlens", "i32"),
        ("max_seqlen_q", "i32"),
        ("max_seqlen_k", "i32"),
        ("window_left", "i32"),
        ("window_right", "i32"),
        ("philox_seed_ptr", "ptr"),
        ("philox_offset1", "ptr"),
        ("philox_offset2", "i64"),
        ("idropout_p", "i32"),
        ("dropout_scale", "f32"),
        ("num_head_q", "i32"),
        ("num_head_k", "i32"),
        ("hdim_qk", "i32"),
        ("hdim_vo", "i32"),
        ("sm_scale", "f32"),
        ("stride_q_batch", "i64"),
        ("stride_q_head", "i64"),
        ("stride_q_seq", "i64"),
        ("stride_k_batch", "i64"),
        ("stride_k_head", "i64"),
        ("stride_k_seq", "i64"),
        ("stride_v_batch", "i64"),
        ("stride_v_head", "i64"),
        ("stride_v_seq", "i64"),
        ("stride_o_batch", "i64"),
        ("stride_o_head", "i64"),
        ("stride_o_seq", "i64"),
        ("stride_do_batch", "i64"),
        ("stride_do_head", "i64"),
        ("stride_do_seq", "i64"),
        ("stride_dk_batch", "i64"),
        ("stride_dk_head", "i64"),
        ("stride_dk_seq", "i64"),
        ("stride_dv_batch", "i64"),
        ("stride_dv_head", "i64"),
        ("stride_dv_seq", "i64"),
        ("stride_dq_batch", "i64"),
        ("stride_dq_head", "i64"),
        ("stride_dq_seq", "i64"),
        ("stride_b_batch", "i64"),
        ("stride_b_head", "i64"),
        ("stride_b_seq_q", "i64"),
        ("stride_db_batch", "i64"),
        ("stride_db_head", "i64"),
        ("stride_db_seq_q", "i64"),
        ("rt_flags", "i32"),
    ),
}

# Threads per workgroup. The prep and reduce kernels fix it; the dK/dV, dQ and merged
# kernels run `num_waves * 64`, which the policy resolves unless the Spec sets it.
_FIXED_BLOCK = {"prep": 256, "dbias_reduce": 256, "dsink_reduce": 64}

# Launch grid, as the launchers' `.launch(grid=...)` compute it (`seqs` is `num_seqlens`
# when non-zero, else the batch size):
#   prep          (ceil(max_seqlen_q / rows_per_block), num_head_q, seqs)
#   dkdv          (num_head_k, ceil(max_seqlen_k / BLOCK_KV), seqs)
#   dq            (num_head_q, ceil(max_seqlen_q / BLOCK_M), seqs)
#   merged        (seqs * (num_head_k * kv_blocks + num_head_q * q_blocks), 1, 1)
#   dbias_reduce  (blocks over the plane, planes, 1)
#   dsink_reduce  (num_head_q, 1, 1)
_GRID_RULE = {
    "prep": "qblocks_by_heads_by_seqs",
    "dkdv": "kheads_by_kvblocks_by_seqs",
    "dq": "qheads_by_qblocks_by_seqs",
    "merged": "linear_roles",
    "dbias_reduce": "blocks_by_planes",
    "dsink_reduce": "heads",
}

# LDS one workgroup may allocate on gfx950. The compiler reports an object over it as
# an error, yet still emits the object, which then fails to launch.
_LDS_LIMIT_BYTES = 163840


def op_of(spec: Spec) -> str:
    return f"sdpa_bwd_{spec.kernel}"


def reduce_symbol_of(spec: Spec) -> str:
    """The dBias reduce kernel the object of @p spec carries beside its own: the merged ``bias`` objects only."""
    return (
        SDPA_BWD_MERGED_REDUCE_SYMBOL
        if spec.kernel == "merged" and spec.cls == "bias"
        else ""
    )


def geometry_of(built: Built) -> dict:
    """The launch geometry the engine reads from a merged record (``block_threads``, ``block_kv``, ``block_m``)."""
    if built.traits is None:
        return {}
    kt, qt = built.traits
    return {
        "block_threads": kt.BLOCK_SIZE,
        "block_kv": kt.BLOCK_KV,
        "block_m": qt.BLOCK_M,
    }


def hdv_of(kernel: str, tile: int) -> int:
    """dQ meta carries head_dim_v = the rung below the tile; dK/dV ignores it."""
    return rung_below(tile) if (kernel == "dq" and tile >= 64) else 0


def floor_value(spec: Spec):
    if spec.floor == "g1":
        return None
    if spec.floor == "tail":
        return rung_below(spec.tile)
    return int(spec.floor)


@dataclass
class Built:
    spec: Spec
    jf: object
    module: object
    meta: object
    knobs: object
    hints: dict
    traits: object = None  # the merged launcher's (dK/dV, dQ) role traits


def _single_jit(launch):
    """The one JitFunction a module's launcher closes over."""
    from flydsl.compiler.jit_function import JitFunction  # noqa: PLC0415

    jfs = [
        c.cell_contents
        for c in (launch.__closure__ or ())
        if isinstance(c.cell_contents, JitFunction)
    ]
    assert len(jfs) == 1, f"{len(jfs)} JitFunctions"
    return jfs[0]


def _hints_of(mod, spec: Spec) -> dict:
    if not spec.hints:
        return {}
    return {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in mod._COMPILE_HINTS.items()
    }


def _pin_dkdv(spec, mk, meta):
    """dK/dV knobs: the policy resolves what the Spec leaves unset. `num_waves` / `block_q` pin the four-field geometry set,
    so the rest of it (BLOCK_KV = mfma_rows * num_waves / shards, granule, waves_per_eu) is taken from the policy resolution
    of the same Spec without them."""
    kn = spec.knobs
    plain = {k: v for k, v in kn.items() if k not in ("num_waves", "block_q")}
    plain = {
        **plain,
        **(
            {"tight_registers": bool(plain["tight_registers"])}
            if "tight_registers" in plain
            else {}
        ),
    }
    knobs = mk(**plain).resolve(meta)
    if "num_waves" not in kn and "block_q" not in kn:
        return knobs
    nw = kn.get("num_waves", knobs.num_waves)
    shards, rows = knobs.dkv_shards, knobs.mfma_rows
    if nw % shards:
        raise ValueError(f"num_waves {nw} is not a multiple of dkv_shards {shards}")
    return mk(
        **{
            **plain,
            "num_waves": nw,
            "block_q": kn.get("block_q", knobs.block_q),
            "block_kv": rows * (nw // shards),
            "head_dim_granule": knobs.head_dim_granule,
            "dkv_shards": shards,
            "mfma_rows": rows,
            "waves_per_eu": knobs.waves_per_eu,
        }
    ).resolve(meta)


def _pin_dq(spec, mk, meta):
    """dQ knobs: as `_pin_dkdv`; `num_waves` / `block_m` / `block_n` pin the four-field set with `head_dim_granule`."""
    kn = spec.knobs
    plain = {
        k: v
        for k, v in kn.items()
        if k not in ("num_waves", "block_m", "block_n", "store_db", "bias_stage")
    }
    knobs = mk(**plain).resolve(meta)
    if not any(k in kn for k in ("num_waves", "block_m", "block_n")):
        return knobs
    nw = kn.get("num_waves", knobs.num_waves)
    return mk(
        **{
            **plain,
            "num_waves": nw,
            "block_m": kn.get("block_m", knobs.block_m),
            "block_n": kn.get("block_n", knobs.block_n),
            "head_dim_granule": knobs.head_dim_granule,
            "mfma_rows": knobs.mfma_rows,
            "waves_per_eu": knobs.waves_per_eu,
        }
    ).resolve(meta)


def build(spec: Spec) -> Built:
    """A fresh launcher for @p spec; nothing is compiled yet."""
    if spec.kernel == "prep":
        from kernels.attention_bwd_cdna import (
            fmha_bwd_prep_gfx950 as mod,
        )  # noqa: PLC0415

        launch = mod.build_fmha_bwd_prep_gfx950_module_primary(
            spec.dtype,
            spec.tile,
            bool(spec.padded),
            **({"sink": True} if spec.sink else {}),
        )
        return Built(spec, _single_jit(launch), mod, None, None, _hints_of(mod, spec))
    if spec.kernel == "dbias_reduce":
        from kernels.attention_bwd_cdna import (
            fmha_bwd_dbias_reduce_gfx950 as mod,
        )  # noqa: PLC0415

        launch = mod.build_fmha_bwd_dbias_reduce_gfx950_module_primary(spec.dtype)
        return Built(
            spec, _single_jit(launch), mod, None, None, dict(mod._COMPILE_HINTS)
        )
    if spec.kernel == "dsink_reduce":
        from kernels.attention_bwd_cdna import (
            fmha_bwd_dsink_reduce_gfx950 as mod,
        )  # noqa: PLC0415

        launch = mod.build_fmha_bwd_dsink_reduce_gfx950_module_primary(spec.dtype)
        return Built(
            spec, _single_jit(launch), mod, None, None, dict(mod._COMPILE_HINTS)
        )
    if spec.kernel == "merged":
        return _build_merged(spec)
    cz = {"plain": 0, "causal": 3, "bias": 0}[spec.cls]
    bias = spec.cls == "bias"
    hdv = spec.hdv or spec.tile
    fl = floor_value(spec)
    if spec.kernel == "dkdv":
        from kernels.attention_bwd_cdna import (
            fmha_bwd_dkdv_gfx950 as mod,
        )  # noqa: PLC0415
        from kernels.attention_bwd_cdna.fmha_tuning_bwd_dkdv_gfx950 import (  # noqa: PLC0415
            BwdDkDvInputMetadata,
            bwd_dkdv_knobs,
        )

        meta = BwdDkDvInputMetadata(
            num_heads=1,
            head_dim=spec.tile,
            head_dim_v=hdv,
            causal=cz != 0,
            window=cz != 0,
            dtype_str=spec.dtype,
            bias=bias,
            dropout=bool(spec.dropout),
        )
        kw = dict(
            block_dmodel=spec.tile,
            block_dmodel_v=spec.tile,
            padded_head=bool(spec.padded),
            strides_constexpr=False,
        )
        if fl is not None:
            kw["hdim_qk_floor"] = fl
        knobs = _pin_dkdv(spec, lambda **o: bwd_dkdv_knobs("gfx950", **kw, **o), meta)
        builder = mod.build_fmha_bwd_dkdv_gfx950_module_primary
    else:
        from kernels.attention_bwd_cdna import (
            fmha_bwd_dq_gfx950 as mod,
        )  # noqa: PLC0415
        from kernels.attention_bwd_cdna.fmha_tuning_bwd_dq_gfx950 import (
            bwd_dq_knobs,
        )  # noqa: PLC0415
        from kernels.attention_bwd_cdna.fmha_tuning_gfx950 import (
            FmhaInputMetadata,
        )  # noqa: PLC0415

        meta = FmhaInputMetadata(
            num_heads=1,
            head_dim=spec.tile,
            head_dim_v=hdv,
            causal=cz != 0,
            window=cz != 0,
            dtype_str=spec.dtype,
            bias=bias,
            dropout=bool(spec.dropout),
        )
        kw = dict(
            block_dmodel=spec.tile,
            padded_head=bool(spec.padded),
            paged=False,
            num_kv_splits=1,
            store_db=bias and spec.store_db is None,
        )
        if spec.bias_stage is not None:
            kw["bias_stage"] = spec.bias_stage
        if fl is not None:
            kw["hdim_qk_floor"] = fl
        knobs = _pin_dq(spec, lambda **o: bwd_dq_knobs("gfx950", **kw, **o), meta)
        builder = mod.build_fmha_bwd_dq_gfx950_module_primary
    rt = ("dropout",) if spec.dform == "rt" else ()
    launch = builder(meta, knobs, runtime_axes=rt)
    return Built(spec, _single_jit(launch), mod, meta, knobs, _hints_of(mod, spec))


def _build_merged(spec: Spec) -> Built:
    """The merged object: the dK/dV and dQ roles each resolved as their standalone Spec would be (policy defaults), except that the dQ
    role takes the dK/dV role's wave count (one workgroup size): BLOCK_M = 32 * num_waves, BLOCK_N and the rest from the dQ policy.
    A ``bias`` object also launches the dBias reduce kernel.
    """
    from kernels.attention_bwd_cdna import (
        fmha_bwd_merged_gfx950 as mod,
    )  # noqa: PLC0415

    common = dict(
        dtype=spec.dtype,
        tile=spec.tile,
        cls=spec.cls,
        dform=spec.dform,
        dropout=spec.dropout,
        padded=spec.padded,
        floor=spec.floor,
        hints=spec.hints,
    )
    bd = build(
        Spec("dkdv", hdv=0, **({"gqa_pack": 1} if spec.gqa_pack else {}), **common)
    )
    nw = bd.knobs.num_waves
    bq0 = build(Spec("dq", hdv=hdv_of("dq", spec.tile), **common))
    bq = build(
        Spec(
            "dq",
            hdv=hdv_of("dq", spec.tile),
            num_waves=nw,
            block_m=32 * nw,
            block_n=bq0.knobs.block_n,
            **({"store_db": 0} if spec.store_db == 0 else {}),
            **common,
        )
    )
    launch = mod.build_fmha_bwd_merged_gfx950_module_primary(
        bd.meta,
        bd.knobs,
        bq.meta,
        bq.knobs,
        spec.role_order or "concat",
        dbias_reduce=bool(reduce_symbol_of(spec)),
    )
    return Built(
        spec,
        launch.jit,
        mod,
        (bd.meta, bq.meta),
        (bd.knobs, bq.knobs),
        _hints_of(mod, spec),
        launch.traits,
    )


def alignment_for(name):
    if name in {
        "Q",
        "K",
        "V",
        "Bias",
        "O",
        "DO",
        "DK",
        "DV",
        "DQ",
        "DB",
        "DBias",
        "LSE",
        "Delta",
        "seqinfo_q0",
        "seqinfo_q1",
        "seqinfo_k0",
        "seqinfo_k1",
    }:
        return 16
    return None


def synth_compile_args(jf):
    """Arguments that trace @p jf: null pointers (aligned as the kernel assumes), zero scalars, the null stream."""
    import flydsl.compiler as flyc  # noqa: PLC0415
    import flydsl.expr as fx  # noqa: PLC0415
    from flydsl.compiler.jit_argument import resolve_signature  # noqa: PLC0415

    args = []
    for p in resolve_signature(jf.func).parameters.values():
        n = getattr(p.annotation, "__name__", str(p.annotation))
        if n == "Pointer":
            args.append(
                flyc.from_c_void_p(fx.Uint8, 0, assumed_align=alignment_for(p.name))
            )
        elif n in ("Int32", "Int64"):
            args.append(0)
        elif n == "Float32":
            args.append(0.0)
        elif n == "Stream":
            args.append(fx.Stream(None))
        else:
            raise NotImplementedError(f"annotation {n} for {p.name}")
    return args


def hints_ctx(built: Built):
    """The module's compile hints, applied as its own launcher applies them."""
    from flydsl.compiler.kernel_function import CompilationContext  # noqa: PLC0415

    return (
        CompilationContext.compile_hints(built.hints)
        if built.hints
        else contextlib.nullcontext()
    )


@dataclass
class Outcome:
    """One spec's build: its status, the object, why it failed, and the gates of the object."""

    status: str
    blob: bytes | None = None
    detail: str = ""
    gate: dict | None = None
    geometry: dict | None = (
        None  # the merged launch geometry, known once the launcher is built
    )


@contextlib.contextmanager
def _stderr_to(path: Path):
    """Redirect file descriptor 2 to @p path: LLVM prints its pass dumps there, not through ``sys.stderr``."""
    sys.stderr.flush()
    saved = os.dup(2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.dup2(fd, 2)
        os.close(fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


_print_after_set = False


def _print_after_greedy() -> None:
    """Ask LLVM for the MIR after the greedy register allocator, once per process."""
    global _print_after_set
    if not _print_after_set:
        from flydsl.compiler.llvm_options import _get_fly_module  # noqa: PLC0415

        _get_fly_module().set_llvm_option_str("print-after", "greedy")
        _print_after_set = True


def _run_gate(
    obj: Path,
    greedy: Path | None,
    spec: Spec,
    reduce_symbol: str = "",
    reference_dir: Path | None = None,
) -> dict:
    """The gates of the object file @p obj, scanned with the dump of the compile that produced it (none: R-MIR is not scanned)."""
    command = [
        sys.executable,
        str(_GATE_TOOL),
        "--hsaco",
        str(obj),
        "--spec",
        spec.key(),
    ]
    if greedy is not None:
        command += ["--greedy-dump", str(greedy)]
    if reduce_symbol:
        command += ["--reduce-symbol", reduce_symbol]
        if reference_dir is not None:
            command += ["--reference-dir", str(reference_dir)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode:
        raise GeneratorError(f"gate: {done.stderr.strip()}")
    gate = json.loads(done.stdout)
    if gate["object_sha256"] != sha256(obj.read_bytes()):
        raise GeneratorError("gate: scanned an object other than the delivered one")
    return gate


def _gate(blob: bytes, greedy: Path, spec: Spec, reference_dir: Path | None) -> dict:
    """The gates of @p blob, scanned with the dump of the compile that produced it."""
    obj = greedy.with_name("obj.hsaco")
    obj.write_bytes(blob)
    return _run_gate(obj, greedy, spec, reduce_symbol_of(spec), reference_dir)


def _compile_one(task: tuple[str, str, str]) -> Outcome:
    """Compile and gate one spec. Never raises for a build failure."""
    spec_s, _, reference = task
    geometry = None
    try:
        spec = Spec.parse(spec_s)
        built = build(spec)
        geometry = geometry_of(built) or None
        dump_dir = Path(tempfile.mkdtemp(prefix=f"sdpa_bwd_{spec.hash()}_"))
        try:
            env.set_dump_dir(dump_dir / "ir")
            greedy = dump_dir / "greedy.txt"
            with _stderr_to(greedy):
                _print_after_greedy()
                with hints_ctx(built):
                    built.jf(*synth_compile_args(built.jf))
            blob, _ = hsaco_from_dump(dump_dir / "ir")
            gate = _gate(blob, greedy, spec, Path(reference) if reference else None)
        finally:
            shutil.rmtree(dump_dir, ignore_errors=True)
    except HsacoExtractionError as exc:
        status = "dual_object" if _DUAL_OBJECT_MARK in str(exc) else "error"
        return Outcome(
            status, detail=" ".join(str(exc).split())[:500], geometry=geometry
        )
    except Exception as exc:  # noqa: BLE001 - reported per spec, never retried
        return Outcome(
            "error",
            detail=f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500],
            geometry=geometry,
        )
    return Outcome("ok", blob, gate=gate, geometry=geometry)


def compile_specs(
    entries: list[Spec], arch: str, reference_dir: Path | None = None
) -> list[Outcome]:
    """One build per spec, in order (``FLYDSL_GEN_JOBS`` workers); a dead worker raises ``GeneratorError``.

    With @p reference_dir, a combined object is gated against the pins there.
    """
    reference = str(reference_dir) if reference_dir else ""
    return compile_all(
        _compile_one,
        [(spec.key(), spec.hash(), reference) for spec in entries],
        arch,
        lambda task: task[1],
    )


def _max_flat_workgroup_size(blob: bytes, where: str, symbol: str) -> int:
    hkp = hkp_pack_module()
    document = hkp._metadata_document(hkp.amdgcn_object(blob, where), where)
    kernel = next(
        k
        for k in document["amdhsa.kernels"]
        if k[".symbol"].removesuffix(".kd") == symbol
    )
    return int(kernel[".max_flat_workgroup_size"])


def _params_key(spec: Spec) -> str:
    return "prep_sink" if spec.kernel == "prep" and spec.sink else spec.kernel


def _verify_kernel(
    blob: bytes,
    described: dict,
    params_key: str,
    wanted_threads: int | None,
    arch: str,
    where: str,
) -> int:
    """Check one kernel's arch, LDS, kernarg layout and block size against the declared ones; returns its threads per workgroup."""
    verify_arch(described, arch, where)
    if described["group_segment_fixed_size"] > _LDS_LIMIT_BYTES:
        raise GeneratorError(
            f"{where}: needs {described['group_segment_fixed_size']} bytes of LDS; "
            f"{arch} allows {_LDS_LIMIT_BYTES} per workgroup"
        )
    expected, _ = _signature(_PARAMS[params_key])
    if described["signature"] != expected:
        raise GeneratorError(
            f"{where}: kernarg layout is\n  {described['signature']}\nbut {params_key} is declared as\n"
            f"  {expected}\nThe kernel's parameter list changed; update _PARAMS, the native "
            "dispatch and its signature together."
        )
    size = expected[-1][2] + expected[-1][1]
    if described["kernarg_segment_size"] != size:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, expected {size}"
        )
    threads = _max_flat_workgroup_size(blob, where, described["symbol"])
    if wanted_threads is not None and threads != wanted_threads:
        raise GeneratorError(
            f"{where}: built for {threads} threads per workgroup, declared {wanted_threads}"
        )
    if threads % 64 or not 64 <= threads <= 1024:
        raise GeneratorError(f"{where}: {threads} threads per workgroup")
    return threads


def verify(
    blob: bytes, spec: Spec, arch: str, where: str, reduce_symbol: str = ""
) -> dict:
    """The facts of @p blob, or ``GeneratorError``: its arch, kernarg layout, LDS and block size must be the declared ones.

    With @p reduce_symbol the object holds exactly two kernels, the spec's own and the dBias reduce kernel
    of that symbol, and each is checked against its declaration; the reduce kernel's facts are under
    ``reduce``.
    """
    wanted = _FIXED_BLOCK.get(spec.kernel) or (spec.num_waves or 0) * 64 or None
    if not reduce_symbol:
        described = describe(blob, where)
        threads = _verify_kernel(
            blob, described, _params_key(spec), wanted, arch, where
        )
        return {**described, "block_threads": threads}
    found = kernel_names(blob, where)
    if len(found) != 2 or reduce_symbol not in found:
        raise GeneratorError(
            f"{where}: expected a {spec.kernel} kernel and {reduce_symbol!r}, found {found}"
        )
    main_symbol = next(name for name in found if name != reduce_symbol)
    described = describe(blob, where, main_symbol)
    reduce = describe(blob, where, reduce_symbol)
    threads = _verify_kernel(blob, described, _params_key(spec), wanted, arch, where)
    reduce_threads = _verify_kernel(
        blob,
        reduce,
        "dbias_reduce",
        _FIXED_BLOCK["dbias_reduce"],
        arch,
        f"{where} ({reduce_symbol})",
    )
    return {
        **described,
        "block_threads": threads,
        "reduce": {**reduce, "block_threads": reduce_threads},
    }


def _read_specs(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "spec" not in reader.fieldnames:
            raise GeneratorError(f"{path}: no `spec` column")
        return [row["spec"] for row in reader if row["spec"].strip()]


def _refuse(ops: set[str], arch: str) -> str | None:
    """The message refusing @p arch for any of @p ops, if one does."""
    refused = sorted(op for op in ops if arch not in _OP_ARCHES[op])
    if not refused:
        return None
    allowed = sorted({a for op in refused for a in _OP_ARCHES[op]})
    return f"{', '.join(refused)} cannot be built for {arch!r}; allowed: {', '.join(allowed)}"


def _empty_dir(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--specs", required=True, type=Path, help="CSV with a `spec` column"
    )
    parser.add_argument(
        "--arch", required=True, help="GPU target; one arch per invocation"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="empty directory for <hash>.hsaco and results.csv",
    )
    args = parser.parse_args(argv)

    try:
        spec_strings = _read_specs(args.specs)
    except (OSError, GeneratorError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # One entry per distinct spec, in file order; a spec that does not parse is an `error` row.
    entries: list[tuple[str, str, Spec | None, str]] = []
    seen: set[str] = set()
    for text in spec_strings:
        try:
            spec = Spec.parse(text.strip())
        except (ValueError, TypeError) as exc:
            entries.append(("", text, None, f"{type(exc).__name__}: {exc}"[:500]))
            continue
        digest = spec.hash()
        if digest not in seen:
            seen.add(digest)
            entries.append((digest, spec.key(), spec, ""))

    # Refused before compiling.
    refusal = _refuse({op_of(s) for _, _, s, _ in entries if s is not None}, args.arch)
    if refusal:
        print(f"error: {refusal}", file=sys.stderr)
        return 1

    if not _empty_dir(args.out_dir):
        print(f"error: --out-dir {args.out_dir} is not empty", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        env.assert_flydsl_version()
        env.prepare(args.arch)
        todo = [e for e in entries if e[2] is not None]
        outcomes = compile_specs([e[2] for e in todo], args.arch)
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    by_hash = {entry[0]: outcome for entry, outcome in zip(todo, outcomes)}
    failed = False
    with (args.out_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(_RESULT_COLUMNS)
        for digest, key, spec, problem in entries:
            outcome = (
                by_hash[digest]
                if spec is not None
                else Outcome("error", detail=problem)
            )
            if outcome.blob is not None:
                (args.out_dir / f"{digest}.hsaco").write_bytes(outcome.blob)
            failed |= outcome.status == "error"
            rmir = (outcome.gate or {}).get("rmir", {})
            writer.writerow(
                (
                    digest,
                    key,
                    outcome.status,
                    sha256(outcome.blob) if outcome.blob is not None else "",
                    outcome.detail,
                    rmir.get("spill_findings", ""),
                    rmir.get("other_findings", ""),
                )
            )
    counts = {
        s: sum(1 for o in by_hash.values() if o.status == s)
        for s in ("ok", "dual_object", "error")
    }
    print(
        f"{counts} (+{sum(1 for e in entries if e[2] is None)} unparsable); results in {args.out_dir / 'results.csv'}"
    )
    return 1 if failed else 0


# The ops that ship the sweep's pinned objects, not their rebuilds (D14).
_SHIP_PINS_OPS = ("sdpa_bwd_merged",)

_REFERENCE_CSV = FLYDSL_DIR / "tests" / "data" / "sdpa_bwd_reference.csv"


@dataclass(frozen=True)
class Pin:
    """A pinned object: the sweep's bytes for a Spec, by the hash and sha256 its reference row gives."""

    hash: str
    sha256: str
    blob: bytes | None = None  # read only for a pin that ships


def read_reference(path: Path = _REFERENCE_CSV) -> dict[str, dict]:
    """The reference rows by Spec hash."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["hash"]: row for row in csv.DictReader(handle)}


def load_pin(
    spec: Spec, reference_dir: Path, reference: dict[str, dict], *, ships: bool
) -> Pin:
    """The pin of @p spec, its object checked against its row; the bytes are read when @p ships."""
    row = reference.get(spec.hash())
    if row is None or row["spec"] != spec.key():
        raise GeneratorError(f"the reference has no row for {spec.key()}")
    if not ships:
        return Pin(spec.hash(), row["sha256"])
    path = reference_dir / f"{spec.hash()}.hsaco"
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise GeneratorError(f"{path}: {exc.strerror}") from exc
    if sha256(blob) != row["sha256"]:
        raise GeneratorError(f"{path} is not the pin its reference row names")
    return Pin(spec.hash(), row["sha256"], blob)


def _record(
    instance,
    spec: Spec,
    outcome: Outcome,
    op_dir: Path,
    arch: str,
    pin: Pin | None = None,
) -> dict:
    """The manifest record of one row: its build status, and for an ``ok`` row the object and its gates.

    With @p pin the row ships pinned objects (O10): a non-bias row ships ``pin.blob`` and its rebuild only
    verifies it; a bias row (``pin`` is its merged pin, bytes unread) ships the combined rebuild. The record
    adds ``pin_sha256``, ``rebuilt_sha256`` and ``ship`` (``pin`` | ``combined``); ``sha256`` is the shipped
    bytes' and the resources are analysed from them. The R-MIR verdict is the rebuild's, and only when the
    rebuilt object is the pin (else ``not scanned``).
    """
    record = instance.to_record()
    record.update(
        {
            "build_status": outcome.status,
            "spec": spec.key(),
            "spec_hash": spec.hash(),
            "file": "",
            "sha256": "",
        }
    )
    record["knobs"].update(outcome.geometry or {})
    reduce_symbol = instance.knobs.get("reduce_symbol", "")
    if reduce_symbol != reduce_symbol_of(spec):
        raise GeneratorError(
            f"{instance.name}: the row names reduce symbol {reduce_symbol!r}, its build has {reduce_symbol_of(spec)!r}"
        )
    shipped_pin = pin is not None and pin.blob is not None
    if outcome.status != "ok" and not shipped_pin:
        record["detail"] = outcome.detail
        return record
    if outcome.status != "ok":
        record["detail"] = outcome.detail
    blob = pin.blob if shipped_pin else outcome.blob
    filename = f"{instance.name}.hsaco"
    where = f"{arch}/{instance.op}/{filename}"
    described = verify(blob, spec, arch, where, reduce_symbol)
    if (
        outcome.geometry
        and outcome.geometry["block_threads"] != described["block_threads"]
    ):
        raise GeneratorError(
            f"{where}: built for {outcome.geometry['block_threads']} threads per workgroup, "
            f"the object says {described['block_threads']}"
        )
    (op_dir / filename).write_bytes(blob)
    delivered = sha256((op_dir / filename).read_bytes())
    rebuilt = sha256(outcome.blob) if outcome.blob is not None else ""
    if shipped_pin and rebuilt != pin.sha256:
        # The pin is not the object the rebuild's gates saw: analyse the shipped bytes (R-MIR: the sweep's scan stands).
        gate = _run_gate(op_dir / filename, None, spec)
    else:
        gate = outcome.gate
    _, names = _signature(_PARAMS[_params_key(spec)])
    rmir = gate["rmir"]
    record.update(
        {
            "block": [described["block_threads"], 1, 1],
            "grid_rule": _GRID_RULE[spec.kernel],
            "args": arg_records(names, described["signature"]),
            "file": filename,
            "symbol": described["symbol"],
            "bytes": len(blob),
            "sha256": delivered,
            "kernarg_segment_size": described["kernarg_segment_size"],
            "lds_bytes": described["group_segment_fixed_size"],
            "vgprs": described["vgpr_count"],
            "vgpr_spills": described.get("vgpr_spill_count", 0),
            "named_args": described["named_args"],
            "resources": gate["resources"],
            "gates": {
                "object_sha256": gate["object_sha256"],
                "object_matches_delivered": gate["object_sha256"] == delivered,
                "rmir": {
                    k: rmir[k]
                    for k in ("verdict", "spill_findings", "other_findings", "kernels")
                },
                "static_1087": gate["static_1087"],
            },
        }
    )
    if reduce_symbol:
        if "pin_gate" not in gate:
            raise GeneratorError(f"{where}: the build ran no per-kernel gate")
        reduce = described["reduce"]
        _, reduce_names = _signature(_PARAMS["dbias_reduce"])
        record.update(
            {
                "reduce_block": [reduce["block_threads"], 1, 1],
                "reduce_args": arg_records(reduce_names, reduce["signature"]),
                "reduce_resources": gate["per_kernel"][reduce_symbol]["resources"],
            }
        )
        record["gates"]["pin_gate"] = gate["pin_gate"]
    if pin is not None:
        record.update(
            {
                "pin_sha256": pin.sha256,
                "rebuilt_sha256": rebuilt,
                "ship": "pin" if shipped_pin else "combined",
            }
        )
        if shipped_pin and outcome.gate is not None:
            record["rebuilt_resources"] = outcome.gate["resources"]
    return record


def _stop_reasons(records: list[dict]) -> list[str]:
    """Why a ship-pins run ships nothing (O9): a row that did not build and has no pin to ship, or a red per-kernel gate."""
    reasons = []
    for record in records:
        combined = bool(record["knobs"].get("reduce_symbol"))
        status = record["build_status"]
        if status == "error" or (combined and status != "ok"):
            reasons.append(f"{record['name']}: {status}: {record.get('detail', '')}")
        elif combined and record["gates"]["pin_gate"]["verdict"] != "ok":
            problems = "; ".join(record["gates"]["pin_gate"]["problems"])
            reasons.append(f"{record['name']}: per-kernel gate red: {problems}")
    return reasons


def generate(
    op: str,
    arch: str,
    out_root: Path,
    only: set[str] | None = None,
    reference_dir: Path | None = None,
) -> int:
    """Build every row of @p op (or the rows whose Spec hash is in @p only) for @p arch.

    Writes ``<out_root>/<op>/<arch>/``; returns the number of ``error`` rows. The
    directory must not hold files already.

    An op in ``_SHIP_PINS_OPS`` needs @p reference_dir (the sweep's pinned objects, checked
    against ``tests/data/sdpa_bwd_reference.csv``) and ships pins (``_record``). It raises
    ``GeneratorError``, with nothing left in the op directory, when a row has to stop the run
    (``_stop_reasons``).
    """
    refusal = _refuse({op}, arch)
    if refusal:
        raise GeneratorError(refusal)
    ship_pins = op in _SHIP_PINS_OPS
    if ship_pins != (reference_dir is not None):
        raise GeneratorError(
            f"{op} ships its pinned objects and needs --reference-dir"
            if ship_pins
            else f"{op} takes no reference directory"
        )
    instances = instances_for(op, arch)
    if only is not None:
        unknown = only - {to_spec(i).hash() for i in instances}
        if unknown:
            raise GeneratorError(f"{op} has no row with hash {sorted(unknown)}")
        instances = [i for i in instances if to_spec(i).hash() in only]
    op_dir = out_root / op / arch
    if not _empty_dir(op_dir):
        raise GeneratorError(f"{op_dir} is not empty")
    specs = [to_spec(instance) for instance in instances]
    pins: list[Pin | None] = [None] * len(specs)
    if ship_pins:
        reference = read_reference()
        pins = [
            load_pin(spec, reference_dir, reference, ships=not reduce_symbol_of(spec))
            for spec in specs
        ]
    env.assert_flydsl_version()
    env.prepare(arch)
    outcomes = compile_specs(specs, arch, reference_dir)

    op_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for instance, spec, outcome, pin in zip(instances, specs, outcomes, pins):
        try:
            records.append(_record(instance, spec, outcome, op_dir, arch, pin))
        except GeneratorError as exc:
            failed = Outcome(
                "error",
                detail=" ".join(str(exc).split())[:500],
                geometry=outcome.geometry,
            )
            records.append(_record(instance, spec, failed, op_dir, arch))
    if ship_pins:
        reasons = _stop_reasons(records)
        if reasons:
            for written in op_dir.iterdir():
                written.unlink()
            raise GeneratorError(
                f"{len(reasons)} row(s) stop the run, nothing is shipped:\n  "
                + "\n  ".join(reasons)
            )
    # None of these kernels comes from AITER, so its pin is no part of how the bytes came to be.
    provenance = env.provenance(arch)
    provenance.pop("aiter_kernels_commit", None)
    write_op_manifest(op_dir, arch, op, provenance, records)
    source_md = refresh_source_md(op_dir)

    status = {
        s: sum(r["build_status"] == s for r in records)
        for s in ("ok", "dual_object", "error")
    }
    red = sum(
        1 for r in records if r.get("gates", {}).get("rmir", {}).get("verdict") == "red"
    )
    print(f"{op}: {status}, {red} with an R-MIR spill finding; wrote {op_dir}")
    if ship_pins:
        shipped = {
            kind: sum(r["ship"] == kind for r in records if "ship" in r)
            for kind in ("pin", "combined")
        }
        differing = sum(
            1
            for r in records
            if r.get("ship") == "pin" and r["rebuilt_sha256"] != r["pin_sha256"]
        )
        print(
            f"shipped {shipped}; {differing} pinned row(s) rebuilt to other bytes (T2)"
        )
    print(f"refreshed {source_md}")
    return status["error"]


def run_op(op: str, description: str, argv: list[str] | None = None) -> int:
    """The command line of one backward op's generator (``gen_<op>.py``)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--arch", required=True, help="GPU target; one arch per invocation"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="content root to write <op>/<arch>/ under; <op>/<arch>/ must be empty",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated Spec hashes: build just those rows (debugging)",
    )
    if op in _SHIP_PINS_OPS:
        parser.add_argument(
            "--reference-dir",
            type=Path,
            default=None,
            help="the sweep's pinned objects, <hash>.hsaco (required): the rows ship them",
        )
    args = parser.parse_args(argv)
    refusal = _refuse({op}, args.arch)
    if refusal:
        print(f"error: {refusal}", file=sys.stderr)
        return 1
    only = set(args.only.split(",")) if args.only else None
    try:
        errors = generate(
            op, args.arch, args.out_dir, only, getattr(args, "reference_dir", None)
        )
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(run())
