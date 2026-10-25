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
from the compile's cache artifact (cache mode, not FlyDSL's IR-dump mode). A compile prints the SGPR greedy-allocation MIR (``print-after=greedy``)
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
reported, never retried (one build per spec), except for a combined row (below), whose unstable build is rebuilt
in a fresh process, at most three builds in all. Comparing an object against a pin, and
deciding which outcomes are red, is the caller's job (``tools/sdpa_bwd_compare.py``).

The dB-storing ``bias`` rows of ``sdpa_bwd_merged`` and ``sdpa_bwd_dq`` (``store_db`` unset; every fp32 bias dQ row)
build one object with two kernels: the merged or dQ kernel and the dBias reduce kernel (``reduce_symbol``); a
``store_db=0`` object never writes dB, so it never launches the reduce and holds its kernel alone. A combined
object's per-kernel gate (``tools/sdpa_bwd_gate.py``) holds the main kernel's function bytes and resource row to its
standalone pin and the reduce kernel's resource row to the reduce pin; the reduce is compiled under the main
module's LLVM options, so its bytes are not compared. For an fp32 dQ object those options differ from the
standalone fp32 reduce's, so its reduce is gated against a standalone fp32 reduce built under the dQ module's
options (``_fp32_reduce_reference``). ``generate`` for the merged, prep, dK/dV and dQ ops ships pins
(``--reference-dir``): every other row ships its pin's bytes and its rebuild only verifies it; a combined
row ships the combined rebuild. A row that does not build
(``error``; for a combined row also ``dual_object``) or a red per-kernel gate stops the run, and nothing
is shipped.

The kernels are the ones under ``kernels_src/kernels/attention_bwd_cdna/``. Each is
traced with no tensors: the launchers take raw pointers and integers, so the arguments
are synthesised from the launcher's own signature. Every build constructs a fresh
launcher (a reused JitFunction is a cache hit) and compiles into a fresh cache directory, in a worker process of its own (``_parallel``).

The exit status is 0 when every spec is ``ok`` or ``dual_object`` (a spill finding is
recorded in the manifest and judged by the compare tool), and 1 when any is ``error``,
a compile worker dies, or the arch is refused.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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
from ._extract_hsaco import HsacoExtractionError, extract_bin_attr
from ._instances import SDPA_BWD_REDUCE_SYMBOL, instances_for
from ._manifest import FLYDSL_DIR, refresh_source_md, sha256, write_op_manifest
from . import _parallel
from ._sdpa_bwd_spec import (  # noqa: F401  (LADDER is part of this module's surface)
    FP32,
    KERNELS,
    LADDER,
    Spec,
    dbias_reduce_spec,
    rung_below,
    to_spec,
)

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

_REFERENCE_CSV = FLYDSL_DIR / "tests" / "data" / "sdpa_bwd_reference.csv"

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
    """The dBias reduce kernel the object of @p spec carries beside its own: the dB-storing ``bias`` objects of the merged and dQ kernels.

    An fp32 bias dQ object always stores dB (``store_db`` is not an fp32 knob), so it always carries the fp32 reduce.
    """
    return (
        SDPA_BWD_REDUCE_SYMBOL[spec.dtype]
        if spec.kernel in ("merged", "dq") and spec.cls == "bias" and spec.store_db != 0
        else ""
    )


def geometry_of(built: Built) -> dict:
    """The launch geometry the engine reads from a row: ``block_threads`` and ``block_kv`` (dK/dV), ``block_m`` (dQ), all three (merged).

    The resolved values: the fields replace the rows' ``-1`` knobs. The prep and reduce kernels have a fixed block; the engine derives it.
    """
    spec = built.spec
    if built.traits is not None:
        kt, qt = built.traits
        return {
            "block_threads": kt.BLOCK_SIZE,
            "block_kv": kt.BLOCK_KV,
            "block_m": qt.BLOCK_M,
        }
    if spec.kernel not in ("dkdv", "dq"):
        return {}
    if spec.dtype == FP32:
        family = built.knobs.family()
        if spec.kernel == "dkdv":
            return {"block_threads": family.block, "block_kv": built.knobs.block_kv}
        return {"block_threads": family.block, "block_m": family.bq}
    traits = built.knobs.build_traits(built.meta)
    if spec.kernel == "dkdv":
        return {"block_threads": traits.BLOCK_SIZE, "block_kv": traits.BLOCK_KV}
    return {"block_threads": traits.BLOCK_SIZE, "block_m": traits.BLOCK_M}


def hdv_of(kernel: str, tile: int, dtype: str | None = None) -> int:
    """dQ meta carries head_dim_v = the rung below the tile on the dtype's ladder; dK/dV ignores it."""
    return rung_below(tile, dtype) if (kernel == "dq" and tile >= 64) else 0


def floor_value(spec: Spec):
    if spec.floor == "g1":
        return None
    if spec.floor == "tail":
        return rung_below(spec.tile, spec.dtype)
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
    if spec.dtype == FP32:
        # The fp32 table fixes schedule, waves and the derived set per (tile, geometry): the only pin is the geometry (`block_q`).
        return mk(
            **plain, **({"block_q": kn["block_q"]} if "block_q" in kn else {})
        ).resolve(meta)
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
    if spec.dtype == FP32:
        # The fp32 table fixes waves and the derived set per (tile, geometry): the only pin is the geometry (`block_m`).
        return mk(
            **plain, **({"block_m": kn["block_m"]} if "block_m" in kn else {})
        ).resolve(meta)
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


def _family(spec: Spec, kernel: str):
    """(module, builder) of @p kernel's family for the spec's dtype: ``fmha_bwd_<kernel>_gfx950`` for 16-bit, ``fmha_bwd_<kernel>_f32_gfx950`` for fp32.

    Static imports only: the inputs digest follows import statements.
    """
    if spec.dtype == FP32:
        if kernel == "prep":
            from kernels.attention_bwd_cdna import fmha_bwd_prep_f32_gfx950 as mod

            return mod, mod.build_fmha_bwd_prep_f32_gfx950_module_primary
        if kernel == "dkdv":
            from kernels.attention_bwd_cdna import fmha_bwd_dkdv_f32_gfx950 as mod

            return mod, mod.build_fmha_bwd_dkdv_f32_gfx950_module_primary
        if kernel == "dq":
            from kernels.attention_bwd_cdna import fmha_bwd_dq_f32_gfx950 as mod

            return mod, mod.build_fmha_bwd_dq_f32_gfx950_module_primary
        from kernels.attention_bwd_cdna import fmha_bwd_dbias_reduce_f32_gfx950 as mod

        return mod, mod.build_fmha_bwd_dbias_reduce_f32_gfx950_module_primary
    if kernel == "prep":
        from kernels.attention_bwd_cdna import fmha_bwd_prep_gfx950 as mod

        return mod, mod.build_fmha_bwd_prep_gfx950_module_primary
    if kernel == "dkdv":
        from kernels.attention_bwd_cdna import fmha_bwd_dkdv_gfx950 as mod

        return mod, mod.build_fmha_bwd_dkdv_gfx950_module_primary
    if kernel == "dq":
        from kernels.attention_bwd_cdna import fmha_bwd_dq_gfx950 as mod

        return mod, mod.build_fmha_bwd_dq_gfx950_module_primary
    from kernels.attention_bwd_cdna import fmha_bwd_dbias_reduce_gfx950 as mod

    return mod, mod.build_fmha_bwd_dbias_reduce_gfx950_module_primary


def _tuning(spec: Spec, kernel: str):
    """(tuning module, input-metadata class) of the dK/dV or dQ family (the same exported names for both dtypes)."""
    if spec.dtype == FP32:
        if kernel == "dkdv":
            from kernels.attention_bwd_cdna import (
                fmha_tuning_bwd_dkdv_f32_gfx950 as tuning,
            )

            return tuning, tuning.BwdDkDvInputMetadata
        from kernels.attention_bwd_cdna import fmha_tuning_bwd_dq_f32_gfx950 as tuning

        return tuning, tuning.FmhaInputMetadata
    if kernel == "dkdv":
        from kernels.attention_bwd_cdna import fmha_tuning_bwd_dkdv_gfx950 as tuning

        return tuning, tuning.BwdDkDvInputMetadata
    from kernels.attention_bwd_cdna import fmha_tuning_bwd_dq_gfx950 as tuning
    from kernels.attention_bwd_cdna.fmha_tuning_gfx950 import FmhaInputMetadata

    return tuning, FmhaInputMetadata


# fp32 stage-1 dQ objects that are not built: SGPR spill to scratch after 5 repair rounds; the block_m=32 twin serves them.
FP32_DQ_REFUSED = frozenset(
    f"dq,fp32,128,bias,bk,0,1,{floor},96,1;block_m=16;block_n=16"
    for floor in ("tail", "g1")
)


def build(spec: Spec) -> Built:
    """A fresh launcher for @p spec; nothing is compiled yet."""
    if spec.key() in FP32_DQ_REFUSED:
        raise NotImplementedError(
            f"{spec.key()} is not built: SGPR spill to scratch after 5 repair rounds; served by the block_m=32 twin"
        )
    if spec.kernel == "prep":
        mod, builder = _family(spec, "prep")
        launch = builder(
            spec.dtype,
            spec.tile,
            bool(spec.padded),
            **({"sink": True} if spec.sink else {}),
        )
        return Built(spec, _single_jit(launch), mod, None, None, _hints_of(mod, spec))
    if spec.kernel == "dbias_reduce":
        mod, builder = _family(spec, "dbias_reduce")
        launch = builder(spec.dtype)
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
    extra = {}
    if spec.kernel == "dkdv":
        mod, builder = _family(spec, "dkdv")
        tuning, metadata = _tuning(spec, "dkdv")
        meta = metadata(
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
        knobs = _pin_dkdv(
            spec, lambda **o: tuning.bwd_dkdv_knobs("gfx950", **kw, **o), meta
        )
    else:
        mod, builder = _family(spec, "dq")
        tuning, metadata = _tuning(spec, "dq")
        meta = metadata(
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
        knobs = _pin_dq(
            spec, lambda **o: tuning.bwd_dq_knobs("gfx950", **kw, **o), meta
        )
        extra = {"dbias_reduce": bool(reduce_symbol_of(spec))}
    rt = ("dropout",) if spec.dform == "rt" else ()
    launch = builder(meta, knobs, runtime_axes=rt, **extra)
    return Built(spec, _single_jit(launch), mod, meta, knobs, _hints_of(mod, spec))


def _build_merged(spec: Spec) -> Built:
    """The merged object: the dK/dV and dQ roles each resolved as their standalone Spec would be (policy defaults, plus the merged Spec's
    ``num_waves`` / ``mfma_rows`` / ``dkv_shards`` on the dK/dV role), except that the dQ role takes the dK/dV role's wave count (one
    workgroup size): BLOCK_M = 32 * num_waves, BLOCK_N and the rest from the dQ policy.
    Per-tile role pins are defaults that apply only when the merged Spec does not set the knob: tile 224 dK/dV ``mfma_rows=16`` (the stock
    role spills); tile 256 dQ ``mfma_rows=32, block_n=64`` (the stock geometry is not addressable). An explicit Spec knob wins over the pin.
    A dB-storing ``bias`` object (``reduce_symbol_of``) also launches the dBias reduce kernel; a ``store_db=0`` bias row builds the dQ role without the dB store and no reduce.
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
    grid = {
        k: v
        for k, v in spec.knobs.items()
        if k in ("num_waves", "mfma_rows", "dkv_shards")
    }
    if spec.tile == 224:
        grid.setdefault("mfma_rows", 16)
    bd = build(
        Spec(
            "dkdv",
            hdv=0,
            **grid,
            **({"gqa_pack": 1} if spec.gqa_pack else {}),
            **common,
        )
    )
    nw = bd.knobs.num_waves
    bq0 = build(Spec("dq", hdv=hdv_of("dq", spec.tile), **common))
    dq_pin = (
        {"mfma_rows": 32, "block_n": 64}
        if spec.tile == 256
        else {"block_n": bq0.knobs.block_n}
    )
    bq = build(
        Spec(
            "dq",
            hdv=hdv_of("dq", spec.tile),
            num_waves=nw,
            block_m=32 * nw,
            **dq_pin,
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
    attempts: int = (
        1  # builds made: a combined row whose gate is red is rebuilt in a fresh process
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
    reference_csv: Path = _REFERENCE_CSV,
) -> dict:
    """The gates of the object file @p obj, scanned with the dump of the compile that produced it (none: R-MIR is not scanned).

    With a @p reduce_symbol and a @p reference_dir, the pins are the objects there, checked against the rows of @p reference_csv.
    """
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
            command += [
                "--reference-dir",
                str(reference_dir),
                "--reference-csv",
                str(reference_csv),
            ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode:
        raise GeneratorError(f"gate: {done.stderr.strip()}")
    gate = json.loads(done.stdout)
    if gate["object_sha256"] != sha256(obj.read_bytes()):
        raise GeneratorError("gate: scanned an object other than the delivered one")
    return gate


def _gate_reference(
    spec: Spec,
    reference_dir: Path,
    reference_csv: Path,
    fp32_reduce_dir: Path | None,
    work: Path,
) -> tuple[Path, Path]:
    """The (directory, CSV) the per-kernel gate of a combined @p spec reads its pins from.

    An fp32 combined object's reduce kernel is compiled under the dQ module's LLVM options, so it is gated against a standalone fp32 reduce
    built under the same options (``_fp32_reduce_reference``) in place of the standalone fp32 reduce pin: a directory under @p work that
    holds the dQ pin and that reduce object, and a CSV with their two rows. Every other combined object is gated against @p reference_dir.
    """
    if spec.dtype != FP32:
        return reference_dir, reference_csv
    if fp32_reduce_dir is None:
        raise GeneratorError(f"{spec.key()}: no fp32 reduce reference was built")
    reduce_spec = dbias_reduce_spec(FP32)
    overlay = work / "gate-reference"
    overlay.mkdir()
    (overlay / f"{spec.hash()}.hsaco").symlink_to(
        reference_dir / f"{spec.hash()}.hsaco"
    )
    blob = (fp32_reduce_dir / "reduce.hsaco").read_bytes()
    (overlay / f"{reduce_spec.hash()}.hsaco").write_bytes(blob)
    resources = json.loads((fp32_reduce_dir / "reduce.json").read_text())
    with reference_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        rows = [row for row in reader if row["hash"] == spec.hash()]
    rows.append(
        {
            **dict.fromkeys(columns, ""),
            **resources,
            "hash": reduce_spec.hash(),
            "spec": reduce_spec.key(),
            "sha256": sha256(blob),
        }
    )
    path = overlay / "reference.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return overlay, path


def _gate(
    blob: bytes,
    greedy: Path,
    spec: Spec,
    reference_dir: Path | None,
    reference_csv: Path,
    fp32_reduce_dir: Path | None,
) -> dict:
    """The gates of @p blob, scanned with the dump of the compile that produced it."""
    obj = greedy.with_name("obj.hsaco")
    obj.write_bytes(blob)
    reduce_symbol = reduce_symbol_of(spec)
    if reduce_symbol and reference_dir is not None:
        reference_dir, reference_csv = _gate_reference(
            spec, reference_dir, reference_csv, fp32_reduce_dir, greedy.parent
        )
    return _run_gate(obj, greedy, spec, reduce_symbol, reference_dir, reference_csv)


def _compile_all(compile_one, tasks, arch: str, label) -> list:
    """``compile_one(task)`` for every task, in order, each in a worker process of its own.

    ``_parallel.compile_all`` with one compile per worker: FlyDSL restores a scoped LLVM option to its default value, which
    LLVM reads as an explicit setting, so a reused worker would carry the previous compile's options (the fp32 dQ module's
    ``unroll-max-count``) into the next object. ``FLYDSL_GEN_JOBS`` sets the worker count; 1 compiles in this process.
    """
    jobs = _parallel._jobs(len(tasks))
    if jobs == 1:
        return _parallel.compile_all(compile_one, tasks, arch, label)
    print(f"compiling {len(tasks)} object(s) on {jobs} worker(s)", flush=True)
    results: list = [None] * len(tasks)
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_parallel._prepare_worker,
        initargs=(arch,),
        max_tasks_per_child=1,
    ) as pool:
        futures = {
            pool.submit(compile_one, task): index for index, task in enumerate(tasks)
        }
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                try:
                    results[index] = future.result()
                except BrokenProcessPool as exc:
                    raise GeneratorError(
                        f"{label(tasks[index])}: a compile worker died ({exc}); "
                        "retry with fewer workers (FLYDSL_GEN_JOBS)"
                    ) from exc
                print(f"[{done}/{len(tasks)}] {label(tasks[index])}", flush=True)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    return results


# A compile task: (spec key, spec hash, reference dir, reference CSV, fp32 reduce reference dir, hints kernel). The empty string is none.
# A non-empty hints kernel names the module of the spec's dtype whose compile hints replace the spec's own: the one object of
# ``_fp32_reduce_reference``, a standalone fp32 reduce built under the dQ module's options.
_Task = tuple[str, str, str, str, str, str]


def _compile_one(task: _Task) -> Outcome:
    """Compile and gate one spec. Never raises for a build failure."""
    spec_s, _, reference, reference_csv, fp32_reduce, hints_kernel = task
    geometry = None
    try:
        spec = Spec.parse(spec_s)
        built = build(spec)
        if hints_kernel:
            built.hints = _hints_of(_family(spec, hints_kernel)[0], spec)
        geometry = geometry_of(built) or None
        dump_dir = Path(tempfile.mkdtemp(prefix=f"sdpa_bwd_{spec.hash()}_"))
        try:
            # Cache mode, as the sweep built the pins: FlyDSL's IR-dump mode runs the pass pipeline stage by stage and
            # changes the code generated for some fp32 dQ objects. The object is read back from the one artifact this
            # compile wrote to a fresh cache directory.
            os.environ.pop("FLYDSL_DUMP_IR", None)
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(dump_dir / "cache")
            greedy = dump_dir / "greedy.txt"
            with _stderr_to(greedy):
                _print_after_greedy()
                with hints_ctx(built):
                    built.jf(*synth_compile_args(built.jf))
            ((_, artifact),) = built.jf._mem_cache.items()
            blob = extract_bin_attr(artifact._ir_text, f"{spec.key()} cache artifact")
            if blob[:4] != b"\x7fELF":
                raise GeneratorError(
                    f"{spec.key()}: the extracted object is not an ELF"
                )
            gate = _gate(
                blob,
                greedy,
                spec,
                Path(reference) if reference else None,
                Path(reference_csv) if reference_csv else _REFERENCE_CSV,
                Path(fp32_reduce) if fp32_reduce else None,
            )
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


def _fp32_reduce_reference(arch: str, out: Path) -> Path:
    """Build the standalone fp32 dBias reduce under the fp32 dQ module's compile options into @p out (``reduce.hsaco``, ``reduce.json``: its resource row).

    The reduce kernel of a combined fp32 dQ object is compiled under the dQ module's LLVM options (one set per gpu module), so its bytes and
    resource row are those of this object, not of the standalone fp32 reduce pin.
    """
    reduce_spec = dbias_reduce_spec(FP32)
    (outcome,) = _compile_all(
        _compile_one,
        [(reduce_spec.key(), reduce_spec.hash(), "", "", "", "dq")],
        arch,
        lambda task: f"{task[1]} (fp32 reduce under the dQ options)",
    )
    if outcome.status != "ok":
        raise GeneratorError(
            f"the fp32 reduce reference did not build: {outcome.status}: {outcome.detail}"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "reduce.hsaco").write_bytes(outcome.blob)
    (out / "reduce.json").write_text(json.dumps(outcome.gate["resources"]))
    return out


# Builds a combined row may take: the first, then fresh-process rebuilds while its per-kernel gate is red (LLVM backend nondeterminism).
_COMBINED_ATTEMPTS = 3


def _combined_unstable(outcome: Outcome) -> bool:
    """A combined row's build that a rebuild may change: its dump held two objects, or its per-kernel gate is red."""
    if outcome.status == "dual_object":
        return True
    return (
        outcome.status == "ok"
        and (outcome.gate or {}).get("pin_gate", {}).get("verdict") == "red"
    )


def compile_specs(
    entries: list[Spec],
    arch: str,
    reference_dir: Path | None = None,
    reference_csv: Path = _REFERENCE_CSV,
) -> list[Outcome]:
    """One build per spec, in order (``FLYDSL_GEN_JOBS`` workers); a dead worker raises ``GeneratorError``.

    With @p reference_dir, a combined object is gated against the pins there (rows of @p reference_csv). A combined row whose build is
    unstable (``_combined_unstable``) is rebuilt in a fresh worker process, up to ``_COMBINED_ATTEMPTS`` builds in all; its outcome is the last.
    """
    reference = str(reference_dir) if reference_dir else ""
    csv_path = str(reference_csv) if reference_dir else ""
    with tempfile.TemporaryDirectory(prefix="sdpa_bwd_fp32_reduce_") as scratch:
        fp32_reduce = ""
        if reference_dir and any(
            spec.dtype == FP32 and reduce_symbol_of(spec) for spec in entries
        ):
            fp32_reduce = str(_fp32_reduce_reference(arch, Path(scratch)))

        def tasks(specs: list[Spec]) -> list[_Task]:
            return [
                (spec.key(), spec.hash(), reference, csv_path, fp32_reduce, "")
                for spec in specs
            ]

        outcomes = _compile_all(_compile_one, tasks(entries), arch, lambda t: t[1])
        for attempt in range(2, _COMBINED_ATTEMPTS + 1):
            again = [
                i
                for i, (spec, outcome) in enumerate(zip(entries, outcomes))
                if reference_dir
                and reduce_symbol_of(spec)
                and _combined_unstable(outcome)
            ]
            if not again:
                break
            print(f"attempt {attempt}: rebuilding {len(again)} combined row(s)")
            rebuilt = _compile_all(
                _compile_one,
                tasks([entries[i] for i in again]),
                arch,
                lambda t: t[1],
            )
            for i, outcome in zip(again, rebuilt):
                outcome.attempts = attempt
                outcomes[i] = outcome
    return outcomes


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


# The ops that ship the sweep's pinned objects, not their rebuilds. The combined rows (a dB-storing ``bias`` row of the merged or dQ op) are
# the exception: they ship their rebuild, gated per kernel.
_SHIP_PINS_OPS = (
    "sdpa_bwd_prep",
    "sdpa_bwd_dkdv",
    "sdpa_bwd_dq",
    "sdpa_bwd_merged",
)


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

    With @p pin the row ships pinned objects: a row without a reduce symbol ships ``pin.blob`` and its rebuild only
    verifies it; a row with one (a dB-storing bias row; ``pin`` is its merged pin, bytes unread) ships the combined rebuild. The record
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
    """Why a ship-pins run ships nothing: any ``error`` row, a row carrying the reduce kernel that did not build ``ok``, or such a row with a red per-kernel gate or a red R-MIR verdict."""
    reasons = []
    for record in records:
        combined = bool(record["knobs"].get("reduce_symbol"))
        status = record["build_status"]
        if status == "error" or (combined and status != "ok"):
            reasons.append(f"{record['name']}: {status}: {record.get('detail', '')}")
        elif combined and record["gates"]["pin_gate"]["verdict"] != "ok":
            problems = "; ".join(record["gates"]["pin_gate"]["problems"])
            reasons.append(f"{record['name']}: per-kernel gate red: {problems}")
        elif combined and record["gates"]["rmir"]["verdict"] == "red":
            reasons.append(f"{record['name']}: R-MIR spill finding")
    return reasons


def generate(
    op: str,
    arch: str,
    out_root: Path,
    only: set[str] | None = None,
    reference_dir: Path | None = None,
    reference_csv: Path = _REFERENCE_CSV,
) -> int:
    """Build every row of @p op (or the rows whose Spec hash is in @p only) for @p arch.

    Writes ``<out_root>/<op>/<arch>/``; returns the number of ``error`` rows. The
    directory must not hold files already.

    An op in ``_SHIP_PINS_OPS`` needs @p reference_dir (the sweep's pinned objects, checked
    against the rows of @p reference_csv, ``tests/data/sdpa_bwd_reference.csv``) and ships pins (``_record``). It raises
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
        reference = read_reference(reference_csv)
        pins = [
            load_pin(spec, reference_dir, reference, ships=not reduce_symbol_of(spec))
            for spec in specs
        ]
    env.assert_flydsl_version()
    env.prepare(arch)
    outcomes = compile_specs(specs, arch, reference_dir, reference_csv)

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
    for spec, outcome in zip(specs, outcomes):
        if outcome.attempts > 1:
            print(f"{spec.hash()}: built in {outcome.attempts} attempts")
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
            f"shipped {shipped}; {differing} pinned row(s) rebuilt to other bytes (LLVM backend nondeterminism)"
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
