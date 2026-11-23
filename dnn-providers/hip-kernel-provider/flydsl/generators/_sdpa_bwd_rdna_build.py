# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build the RDNA-family SDPA backward code objects from ``Spec`` strings (``_sdpa_bwd_spec``).

    python -m generators.gen_sdpa_bwd_<prep|dkdv|dq> --arch gfx11-generic --out-dir <content root> --reference-dir <S1 reference dir>

The gfx11 counterpart of ``_sdpa_bwd_cdna_build``, which owns the command line, the compile worker pool and the
manifest of the gfx950 objects; the ``gen_sdpa_bwd_{prep,dkdv,dq}`` modules hand ``gfx11-generic`` to ``generate`` here. The
rows are the tables of ``_instances`` (``SDPA_BWD_GFX11_TABLES``, ...), each the pin of the arch's S1 catalogue
(``tests/data/sdpa_bwd_gfx11_reference.csv``, ...). The kernels are the ones under ``kernels_src/kernels/attention_bwd_rdna/``
(16-bit and fp32 families); they take the knob record of their tuning module and are built as the sweep built them:
``SWEEP_ARCH``-free, through the FlyDSL generic-target shim (``_flydsl_env.prepare``), in cache mode, under the module's own
compile hints.

``ARCHES`` are the arches whose rows ``family(arch, dtype, kernel) == "rdna"`` this module builds: gfx11-generic (every row) and
the 16-bit rows of gfx942 and gfx90a (the MFMA16 layer, wave64) with their fp32 ``dbias_reduce`` (the f32 dB reduce of a 16-bit row,
the gfx11 kernel). ``_sdpa_bwd_split_build`` builds those arches' other fp32 rows with the CDNA family and writes one manifest.

What differs from gfx950:

* the launch grid of dK/dV is ``(kv_blocks, num_head_k, seqs)`` (``grid`` of the row), or ``(num_head_k, kv_blocks, seqs)`` when the
  Spec's ``grid_order`` is ``head_fastest`` (gfx942's default); gfx950's is the latter;
* the workgroup is a multiple of the wave (32 lanes, 64 on gfx942 and gfx90a), and the LDS limit is 64 KiB;
* the gfx950 gates (resource row, #1087 detectors, R-MIR scan) read gfx950 objects. A row of gfx942 or gfx90a carries the R-MIR verdict
  and the #1087 facts of ``_sdpa_bwd_cdna_build._run_gate``; a row of a generic arch records ``gates: n/a`` and its pinned resource
  row (R-MIR may be scanned for the pin evidence only, ``_compile_one``'s ``scan``, never in a manifest). What replaces the gates there
  is the pin check: the build must be the S1 object, byte for byte;
* the dBias reduce kernel is not part of a dQ object. A bias dQ row names the standalone reduce kernel of its dtype
  (``reduce_symbol``) and the manifest lists the three standalone reduce objects under ``companions`` of ``sdpa_bwd_dq``, then the
  three tail-zero objects (``build_tailzero``, ``gen_sdpa_bwd_dq.py``): they have no pin and no reference row, the tail-zero generator is their provenance.

A row that does not build ``ok`` or is not its pin stops the run, and nothing is shipped.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _flydsl_env as env
from . import _sdpa_bwd_cdna_build as cdna
from . import _sdpa_bwd_nondet as nondet
from ._codeobject import (
    GeneratorError,
    arg_records,
    describe,
    verify_arch,
    verify_generic,
)
from ._extract_hsaco import HsacoExtractionError, extract_bin_attr
from ._instances import (
    SDPA_BWD_CDNA9_KNOBS,
    SDPA_BWD_CDNA9_TARGETS,
    SDPA_BWD_GFX11_KNOBS,
    SDPA_BWD_GFX11_SPLIT_KINDS,
    SDPA_BWD_GFX11_TARGET,
    _sdpa_bwd_tables,
    instances_for,
    sdpa_bwd_perf_knobs,
)
from ._manifest import (
    FLYDSL_DIR,
    refresh_source_md,
    sha256,
    write_op_manifest,
)
from ._sdpa_bwd_spec import (
    FP32,
    GRID_ORDER_DEFAULT,
    LDS_LIMIT_BYTES,
    UNSET_INT,
    UNSET_WORD,
    WAVE_SIZE,
    Spec,
    family,
)

# The arches whose RDNA-family rows this module builds (every row of gfx11-generic, the 16-bit rows of gfx942 and gfx90a).
ARCHES = (SDPA_BWD_GFX11_TARGET, *SDPA_BWD_CDNA9_TARGETS)
# The arches whose every row this module builds alone (gfx12-generic joins with its rows).
GENERIC_ARCHES = (SDPA_BWD_GFX11_TARGET,)
# The arches whose rows pass the CDNA gates (R-MIR verdict, #1087 facts; ``_sdpa_bwd_cdna_build._run_gate``). The other arches record
# ``gates: "n/a"``: byte identity with the S1 pin is their only check; their R-MIR is scanned only on request (``_compile_one``).
GATED_ARCHES = SDPA_BWD_CDNA9_TARGETS

# The ops that build here.
OPS = tuple(f"sdpa_bwd_{kind}" for kind in SDPA_BWD_GFX11_SPLIT_KINDS)

# The S1 catalogue's pins: ``hash, spec, sha256`` and the resource row of each object.
_REFERENCE_CSVS = {
    arch: FLYDSL_DIR / "tests" / "data" / name
    for arch, name in {
        SDPA_BWD_GFX11_TARGET: "sdpa_bwd_gfx11_reference.csv",
        "gfx942": "sdpa_bwd_gfx942_reference.csv",
        "gfx90a": "sdpa_bwd_gfx90a_reference.csv",
    }.items()
}

# Launch grid, as the launchers' `.launch(grid=...)` compute it (`seqs` is `num_seqlens` when non-zero, else the batch size):
#   prep          (ceil(max_seqlen_q / rows_per_block), num_head_q, seqs)
#   dkdv          (ceil(max_seqlen_k / BLOCK_KV), num_head_k, seqs); head_fastest: (num_head_k, ceil(max_seqlen_k / BLOCK_KV), seqs)
#   dq            (num_head_q, ceil(max_seqlen_q / BLOCK_M), seqs)
#   dbias_reduce  (blocks over the plane, planes, 1)
_GRID_RULE = {
    "prep": "qblocks_by_heads_by_seqs",
    "dkdv": "kvblocks_by_kheads_by_seqs",
    "dq": "qheads_by_qblocks_by_seqs",
    "dbias_reduce": "blocks_by_planes",
}
_DKDV_GRID_HEAD_FASTEST = "kheads_by_kvblocks_by_seqs"


def _grid_rule(spec: Spec) -> str:
    """The grid rule of @p spec's row: dK/dV's follows the Spec's ``grid_order`` (the default of its arch when not spelled)."""
    if (
        spec.kernel == "dkdv"
        and (spec.grid_order or GRID_ORDER_DEFAULT[spec.arch]) == "head_fastest"
    ):
        return _DKDV_GRID_HEAD_FASTEST
    return _GRID_RULE[spec.kernel]


# The dV-only object (``u2=dv``) runs one workgroup column per V head.
_GRID_RULE_DVONLY = "kvblocks_by_vheads_by_seqs"


def grid_rule_of(spec: Spec) -> str:
    """The launch grid spelling of @p spec's object (``_grid_rule``, the V-head one for the dV-only object)."""
    return (
        _GRID_RULE_DVONLY
        if spec.kernel == "dkdv" and spec.u2 == "dv"
        else _grid_rule(spec)
    )


RESOURCE_COLUMNS = (
    "vgpr",
    "agpr",
    "vgpr_total",
    "sgpr",
    "sgpr_spill",
    "vgpr_spill",
    "scratch_bytes",
    "lds_bytes",
    "occupancy",
)


@dataclass(frozen=True)
class Meta:
    """The functional inputs of a dK/dV or dQ build: what the knob resolution and the builder are given."""

    dtype: str
    tile: int
    cls: str
    dropout: int
    padded: int
    hdim_qk: int
    hdim_vo: int


def specs_for(kind: str, arch: str) -> list[Spec]:
    """The Specs of the RDNA-family rows of @p arch's table of @p kind, in table order (``_instances._sdpa_bwd_instances`` spells the same rows).

    On gfx942 and gfx90a only the 16-bit rows and the fp32 ``dbias_reduce`` are this module's; the other fp32 rows are the CDNA family's.
    """
    names = (
        SDPA_BWD_CDNA9_KNOBS if arch in SDPA_BWD_CDNA9_TARGETS else SDPA_BWD_GFX11_KNOBS
    )[kind]
    specs = []
    for (
        dtype,
        tile,
        cls,
        dform,
        dropout,
        padded,
        floor,
        hdv,
        hints,
        *perf,
    ) in _sdpa_bwd_tables(arch)[kind]:
        if family(arch, dtype, kind) != "rdna":
            continue
        pins = {
            name: value
            for name, value in sdpa_bwd_perf_knobs(names, perf).items()
            if value not in (UNSET_INT, UNSET_WORD)
        }
        specs.append(
            Spec(
                kind,
                dtype,
                tile,
                cls,
                dform,
                dropout,
                padded,
                floor,
                hdv,
                hints,
                arch=arch,
                **pins,
            )
        )
    return specs


def _family(spec: Spec, kernel: str):
    """(module, builder) of @p kernel's family for the spec's dtype: ``fmha_bwd_<kernel>_gfx11`` for 16-bit, ``fmha_bwd_<kernel>_f32_gfx11`` for fp32.

    Static imports only: the inputs digest follows import statements.
    """
    if spec.dtype == FP32:
        if kernel == "prep":
            from kernels.attention_bwd_rdna import fmha_bwd_prep_f32_gfx11 as mod

            return mod, mod.build_fmha_bwd_prep_f32_gfx11_module_primary
        if kernel == "dkdv":
            from kernels.attention_bwd_rdna import fmha_bwd_dkdv_f32_gfx11 as mod

            return mod, mod.build_fmha_bwd_dkdv_f32_gfx11_module_primary
        if kernel == "dq":
            from kernels.attention_bwd_rdna import fmha_bwd_dq_f32_gfx11 as mod

            return mod, mod.build_fmha_bwd_dq_f32_gfx11_module_primary
        from kernels.attention_bwd_rdna import fmha_bwd_dbias_reduce_f32_gfx11 as mod

        return mod, mod.build_fmha_bwd_dbias_reduce_f32_gfx11_module_primary
    if kernel == "prep":
        from kernels.attention_bwd_rdna import fmha_bwd_prep_gfx11 as mod

        return mod, mod.build_fmha_bwd_prep_gfx11_module_primary
    if kernel == "dkdv":
        from kernels.attention_bwd_rdna import fmha_bwd_dkdv_gfx11 as mod

        return mod, mod.build_fmha_bwd_dkdv_gfx11_module_primary
    if kernel == "dq":
        from kernels.attention_bwd_rdna import fmha_bwd_dq_gfx11 as mod

        return mod, mod.build_fmha_bwd_dq_gfx11_module_primary
    from kernels.attention_bwd_rdna import fmha_bwd_dbias_reduce_gfx11 as mod

    return mod, mod.build_fmha_bwd_dbias_reduce_gfx11_module_primary


def _tuning(spec: Spec, kernel: str):
    """The tuning module of the dK/dV or dQ family (the same call shape for both dtypes)."""
    if spec.dtype == FP32:
        if kernel == "dkdv":
            from kernels.attention_bwd_rdna import (
                fmha_tuning_bwd_dkdv_f32_gfx11 as tuning,
            )

            return tuning
        from kernels.attention_bwd_rdna import fmha_tuning_bwd_dq_f32_gfx11 as tuning

        return tuning
    if kernel == "dkdv":
        from kernels.attention_bwd_rdna import fmha_tuning_bwd_dkdv_gfx11 as tuning

        return tuning
    from kernels.attention_bwd_rdna import fmha_tuning_bwd_dq_gfx11 as tuning

    return tuning


def build(spec: Spec) -> cdna.Built:
    """A fresh launcher for @p spec; nothing is compiled yet.

    prep takes ``(dtype, tile, padded)`` and the dBias reduce ``(dtype)``; dK/dV and dQ take the ``Meta`` and the knob record of
    ``bwd_<kernel>_knobs(arch, dtype, tile, cls, dropout, padded, hdim_qk, hdim_vo, **the Spec's knobs)``. ``hdim_vo`` is the
    tile unless the Spec spells a V head dim.
    """
    if spec.arch not in ARCHES or family(spec.arch, spec.dtype, spec.kernel) != "rdna":
        raise GeneratorError(f"{spec.key()} is not an RDNA-family Spec")
    mod, builder = _family(spec, spec.kernel)
    if spec.kernel == "prep":
        launch = builder(
            spec.dtype,
            spec.tile,
            bool(spec.padded),
            **({"vec_tail": True} if spec.vec_tail else {}),
        )
        return cdna.Built(
            spec, cdna._single_jit(launch), mod, None, None, cdna._hints_of(mod, spec)
        )
    if spec.kernel == "dbias_reduce":
        launch = builder(spec.dtype)
        return cdna.Built(
            spec,
            cdna._single_jit(launch),
            mod,
            None,
            None,
            dict(mod._COMPILE_HINTS),
        )
    meta = Meta(
        spec.dtype,
        spec.tile,
        spec.cls,
        spec.dropout,
        spec.padded,
        spec.tile,
        spec.hdv or spec.tile,
    )
    knobs = getattr(_tuning(spec, spec.kernel), f"bwd_{spec.kernel}_knobs")(
        spec.arch,
        meta.dtype,
        meta.tile,
        meta.cls,
        meta.dropout,
        meta.padded,
        meta.hdim_qk,
        meta.hdim_vo,
        **spec.knobs,
    )
    if not knobs.legal:
        raise ValueError(f"{spec.key()}: the knob record is not legal: {knobs}")
    launch = builder(
        meta, knobs, runtime_axes=("dropout",) if spec.dform == "rt" else ()
    )
    return cdna.Built(
        spec, cdna._single_jit(launch), mod, meta, knobs, cdna._hints_of(mod, spec)
    )


def geometry_of(built: cdna.Built) -> dict:
    """The schedule and launch geometry the knob record resolved, which replaces the row's ``-1`` knobs.

    dK/dV: ``num_waves``, ``block_q``, ``waves_per_eu`` (``-1`` where the record leaves it unset), ``block_threads``, ``block_kv``;
    dQ: ``num_waves``, ``block_m``, ``block_n``, ``waves_per_eu``, ``block_threads``, ``bias_stage_resolved`` (``direct`` with a bias, ``na``
    without: gfx11 never stages the bias). The prep and reduce kernels fix their block; the engine derives it.
    """
    spec, knobs = built.spec, built.knobs
    if spec.kernel not in ("dkdv", "dq"):
        return {}
    waves_per_eu = knobs.waves_per_eu or -1
    threads = knobs.num_waves * WAVE_SIZE[spec.arch]
    if spec.kernel == "dkdv":
        # The 16-bit family's record resolves the order; the fp32 (VALU) family's has none and launches kv_fastest.
        order = spec.grid_order or GRID_ORDER_DEFAULT[spec.arch]
        launched = getattr(knobs, "grid_order", "kv_fastest")
        if launched != order:
            raise GeneratorError(
                f"{spec.key()}: the knob record launches dK/dV {launched}, the Spec says {order}"
            )
        return {
            "num_waves": knobs.num_waves,
            "block_q": knobs.block_q,
            "waves_per_eu": waves_per_eu,
            "block_threads": threads,
            "block_kv": knobs.block_kv,
        }
    return {
        "num_waves": knobs.num_waves,
        "block_m": knobs.block_m,
        "block_n": knobs.block_n,
        "waves_per_eu": waves_per_eu,
        "block_threads": threads,
        "bias_stage_resolved": "direct" if spec.cls == "bias" else "na",
    }


def _compile_one(task: tuple[str, ...]) -> cdna.Outcome:
    """Compile one spec. Never raises for a build failure.

    @p task is ``(spec key,)`` or ``(spec key, scan)``. Cache mode, as the sweep built the pins: no IR dump is requested (FlyDSL skips
    its disk cache when one is), and the object is read back from the one artifact this compile wrote to a fresh cache directory.

    R-MIR: a row of ``GATED_ARCHES`` is built with the SGPR greedy dump captured and goes through the CDNA gates (``Outcome.gate``).
    On another arch the dump is captured only when @p task asks to ``scan`` (a non-empty second element), and ``Outcome.gate`` then holds the
    R-MIR scan alone (``sdpa_bwd_gate.py --scan-only``): evidence for the pin run, which ``_record`` never writes into a manifest.
    Capturing the dump changes no byte of the object.
    """
    spec_s, *options = task
    scan = bool(options and options[0])
    geometry = None
    try:
        spec = Spec.parse(spec_s)
        built = build(spec)
        geometry = geometry_of(built) or None
        gated = spec.arch in GATED_ARCHES
        work = Path(tempfile.mkdtemp(prefix=f"sdpa_bwd_{spec.arch}_{spec.hash()}_"))
        gate = None
        try:
            os.environ.pop("FLYDSL_DUMP_IR", None)
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(work / "cache")
            greedy = work / "greedy.txt"
            if gated or scan:
                with cdna._stderr_to(greedy):
                    cdna._print_after_greedy()
                    with cdna.hints_ctx(built):
                        built.jf(*cdna.synth_compile_args(built.jf))
            else:
                with cdna.hints_ctx(built):
                    built.jf(*cdna.synth_compile_args(built.jf))
            ((_, artifact),) = built.jf._mem_cache.items()
            blob = extract_bin_attr(artifact._ir_text, f"{spec.key()} cache artifact")
            if blob[:4] != b"\x7fELF":
                raise GeneratorError(
                    f"{spec.key()}: the extracted object is not an ELF"
                )
            if gated or scan:
                obj = work / "obj.hsaco"
                obj.write_bytes(blob)
                gate = cdna._run_gate(obj, greedy, spec, scan_only=not gated)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    except HsacoExtractionError as exc:
        status = "dual_object" if cdna._DUAL_OBJECT_MARK in str(exc) else "error"
        return cdna.Outcome(
            status, detail=" ".join(str(exc).split())[:500], geometry=geometry
        )
    except Exception as exc:  # noqa: BLE001 - reported per spec, never retried
        return cdna.Outcome(
            "error",
            detail=f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500],
            geometry=geometry,
        )
    return cdna.Outcome("ok", blob, gate=gate, geometry=geometry)


def verify(blob: bytes, spec: Spec, where: str) -> dict:
    """The facts of @p blob, or ``GeneratorError``: its target (name and ELF header), kernarg layout, LDS and block size must be the declared ones."""
    described = describe(blob, where)
    verify_arch(described, spec.arch, where)
    verify_generic(blob, spec.arch, where)
    cdna.verify_u2_symbol(spec, described["symbol"], where)
    if described["group_segment_fixed_size"] > LDS_LIMIT_BYTES[spec.arch]:
        raise GeneratorError(
            f"{where}: needs {described['group_segment_fixed_size']} bytes of LDS; "
            f"{spec.arch} allows {LDS_LIMIT_BYTES[spec.arch]} per workgroup"
        )
    expected, _ = cdna._signature(cdna._PARAMS[cdna._params_key(spec)])
    if described["signature"] != expected:
        raise GeneratorError(
            f"{where}: kernarg layout is\n  {described['signature']}\nbut {spec.kernel} is declared as\n"
            f"  {expected}\nThe kernel's parameter list changed; update _PARAMS, the native "
            "dispatch and its signature together."
        )
    size = expected[-1][2] + expected[-1][1]
    if described["kernarg_segment_size"] != size:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, expected {size}"
        )
    threads = cdna._max_flat_workgroup_size(blob, where, described["symbol"])
    wanted = cdna._FIXED_BLOCK.get(spec.kernel)
    if wanted is not None and threads != wanted:
        raise GeneratorError(
            f"{where}: built for {threads} threads per workgroup, declared {wanted}"
        )
    warp = WAVE_SIZE[spec.arch]
    if threads % warp or not warp <= threads <= 1024:
        raise GeneratorError(f"{where}: {threads} threads per workgroup")
    return {**described, "block_threads": threads}


def _number(text: str):
    """A resource-row cell of the reference CSV: an int, or null where the pin has none."""
    return int(text) if text else None


def _gates_of(spec: Spec, outcome: cdna.Outcome, delivered: str, where: str):
    """The ``gates`` of a row: the CDNA gate record on ``GATED_ARCHES``, ``"n/a"`` elsewhere (a scan-only R-MIR result stays out of the manifest)."""
    if spec.arch not in GATED_ARCHES:
        return "n/a"
    gate = outcome.gate
    if gate is None:
        raise GeneratorError(f"{where}: the build ran no gate")
    return {
        "object_sha256": gate["object_sha256"],
        "object_matches_delivered": gate["object_sha256"] == delivered,
        "rmir": {
            k: gate["rmir"][k]
            for k in ("verdict", "spill_findings", "other_findings", "kernels")
        },
        "static_1087": gate["static_1087"],
    }


def _record(
    instance, spec: Spec, outcome: cdna.Outcome, pin: dict, op_dir: Path
) -> dict:
    """The manifest record of one row: its build status, and for an ``ok`` row the object, its facts and its pinned resource row.

    The record has the fields of a gfx950 row of the op, ``grid`` last.
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
    if spec.kernel in cdna.SPLIT_KERNELS:
        record["knobs"]["grid"] = grid_rule_of(spec)
    if outcome.status != "ok":
        record["detail"] = outcome.detail
        return record
    filename = f"{instance.name}.hsaco"
    where = f"{spec.arch}/{instance.op}/{filename}"
    blob = outcome.blob
    if sha256(blob) != pin["sha256"]:
        raise GeneratorError(
            f"{where}: the build is not the S1 object {pin['hash']} "
            f"({sha256(blob)[:16]} != {pin['sha256'][:16]})"
        )
    described = verify(blob, spec, where)
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
    _, names = cdna._signature(cdna._PARAMS[cdna._params_key(spec)])
    record.update(
        {
            "block": [described["block_threads"], 1, 1],
            "grid_rule": grid_rule_of(spec),
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
            "resources": {c: _number(pin[c]) for c in RESOURCE_COLUMNS},
            "gates": _gates_of(spec, outcome, delivered, where),
        }
    )
    return record


def _check_rebuild(spec: Spec, outcome: cdna.Outcome, listed: nondet.Row) -> None:
    """``GeneratorError`` when the rebuild of a listed row exists and is neither its pin nor a recorded variant (``nondet.tolerates``)."""
    if outcome.blob and not nondet.tolerates(listed, sha256(outcome.blob)):
        raise GeneratorError(
            f"{spec.key()}: the rebuild {sha256(outcome.blob)[:8]} is neither its pin {listed.pin_sha256[:8]} nor a recorded variant of it "
            "(tests/data/sdpa_bwd_nondeterministic.csv): sample and list it"
        )


def _pinned_record(
    instance,
    spec: Spec,
    outcome: cdna.Outcome,
    pin: dict,
    op_dir: Path,
    reference_dir: Path,
    listed: nondet.Row,
) -> dict:
    """The record of a row of the nondeterministic list: it ships its pin, whatever its build did, but a rebuild must be its pin or a recorded
    variant (``_check_rebuild``); the build is recorded as ``rebuilt_sha256`` (and its R-MIR verdict as ``rebuilt_rmir`` on a gated arch).

    On a ``GATED_ARCHES`` arch the gates are those of the shipped pin, not of the rebuild: the rebuild's gate saw another object, or none
    when it did not build. The pin is not scanned for R-MIR; the gate tool accepts it only where its reference row records a clean scan.
    """
    _check_rebuild(spec, outcome, listed)
    blob = (reference_dir / f"{spec.hash()}.hsaco").read_bytes()
    gate = None
    if spec.arch in GATED_ARCHES:
        shipped = op_dir / f"{instance.name}.hsaco"
        shipped.write_bytes(blob)
        gate = cdna._run_gate(
            shipped, None, spec, reference_csv=_REFERENCE_CSVS[spec.arch]
        )
    record = _record(
        instance,
        spec,
        cdna.Outcome("ok", blob, gate=gate, geometry=outcome.geometry),
        pin,
        op_dir,
    )
    record.update(
        {
            "build_status": outcome.status,
            "pin_sha256": pin["sha256"],
            "ship": "pin",
            "pin_reason": cdna._NONDETERMINISTIC_REASON,
            "rebuilt_sha256": sha256(outcome.blob) if outcome.blob else "",
            "rebuilt_rmir": (outcome.gate or {}).get("rmir", {}).get("verdict", ""),
        }
    )
    return record


def _companion(spec: Spec, outcome: cdna.Outcome, pin: dict, op_dir: Path) -> dict:
    """The manifest entry of one standalone reduce object, written to ``<hash>.hsaco``; ``GeneratorError`` for one that is not its pin."""
    filename = f"{spec.hash()}.hsaco"
    where = f"{spec.arch}/companion/{filename}"
    if outcome.status != "ok":
        raise GeneratorError(f"{where}: {outcome.status}: {outcome.detail}")
    # A stored pin (a listed companion) carries no gate: it is not scanned, its recorded scan is checked by ``build_rows``.
    if (
        spec.arch in GATED_ARCHES
        and (outcome.gate or {}).get("rmir", {}).get("verdict") == "red"
    ):
        raise GeneratorError(f"{where}: R-MIR spill finding")
    blob = outcome.blob
    if sha256(blob) != pin["sha256"]:
        raise GeneratorError(
            f"{where}: the build is not the S1 object {pin['hash']} "
            f"({sha256(blob)[:16]} != {pin['sha256'][:16]})"
        )
    described = verify(blob, spec, where)
    (op_dir / filename).write_bytes(blob)
    return {
        "hash": spec.hash(),
        "spec": spec.key(),
        "file": filename,
        "symbol": described["symbol"],
        "sha256": sha256((op_dir / filename).read_bytes()),
    }


def build_rows(
    op: str,
    arch: str,
    instances: list,
    specs: list[Spec],
    companions: list[Spec],
    reference: dict,
    op_dir: Path,
    reference_dir: Path,
    nondeterministic_csv: Path = nondet.NONDETERMINISTIC_CSV,
) -> tuple[list[dict], list[dict], list[str]]:
    """Build the RDNA-family rows @p specs (spelled by @p instances) and @p companions of @p op on @p arch.

    Returns (manifest records in row order, companion entries, the reasons a row stops the run); the objects are written to
    @p op_dir. Every object must be its pin in @p reference (rows of the arch's reference CSV, the objects in @p reference_dir), except
    a row of the nondeterministic list @p nondeterministic_csv (``_sdpa_bwd_cdna_build``, "Nondeterministic builds"), which ships its pin
    whatever its build does (``ship: pin``) but stops the run when its rebuild is neither the pin nor a recorded variant; a ``dual_object``
    build of any other row is rebuilt up to ``cdna._ATTEMPTS`` builds in all.
    """
    everything = [*specs, *companions]
    listed = cdna.read_nondeterministic(
        arch, reference, everything, nondeterministic_csv
    )

    def compile_rows(rows: list[Spec]) -> list[cdna.Outcome]:
        return cdna._compile_all(
            _compile_one,
            [(spec.key(),) for spec in rows],
            arch,
            lambda task: Spec.parse(task[0]).hash(),
        )

    outcomes = compile_rows(everything)
    for _ in range(2, cdna._ATTEMPTS + 1):
        again = [
            i
            for i, (spec, outcome) in enumerate(zip(everything, outcomes))
            if outcome.status == "dual_object" and spec.hash() not in listed
        ]
        if not again:
            break
        for i, outcome in zip(again, compile_rows([everything[i] for i in again])):
            outcomes[i] = outcome

    op_dir.mkdir(parents=True, exist_ok=True)
    records = []
    reasons = []
    for instance, spec, outcome in zip(instances, specs, outcomes):
        try:
            if spec.hash() in listed:
                records.append(
                    _pinned_record(
                        instance,
                        spec,
                        outcome,
                        reference[spec.hash()],
                        op_dir,
                        reference_dir,
                        listed[spec.hash()],
                    )
                )
            else:
                records.append(
                    _record(instance, spec, outcome, reference[spec.hash()], op_dir)
                )
        except GeneratorError as exc:
            failed = cdna.Outcome(
                "error",
                detail=" ".join(str(exc).split())[:500],
                geometry=outcome.geometry,
            )
            records.append(
                _record(instance, spec, failed, reference[spec.hash()], op_dir)
            )
        if records[-1]["build_status"] != "ok" and records[-1].get("ship") != "pin":
            reasons.append(
                f"{instance.name}: {records[-1]['build_status']}: {records[-1].get('detail', '')}"
            )
    entries = []
    for spec, outcome in zip(companions, outcomes[len(specs) :]):
        if spec.hash() in listed:
            # A listed companion ships its pin like a listed row; on a gated arch the pin needs a recorded clean R-MIR scan.
            try:
                _check_rebuild(spec, outcome, listed[spec.hash()])
                scan = reference[spec.hash()].get("rmir_scan") or ""
                if spec.arch in GATED_ARCHES and scan not in cdna._RMIR_SCANS:
                    raise GeneratorError(
                        f"{spec.key()}: the pin has no recorded clean R-MIR scan (rmir_scan {scan!r} in the reference)"
                    )
            except GeneratorError as exc:
                reasons.append(" ".join(str(exc).split())[:500])
            outcome = cdna.Outcome(
                "ok", (reference_dir / f"{spec.hash()}.hsaco").read_bytes()
            )
        try:
            entries.append(_companion(spec, outcome, reference[spec.hash()], op_dir))
        except GeneratorError as exc:
            reasons.append(" ".join(str(exc).split())[:500])
    if not reasons and entries:
        symbols = {
            Spec.parse(entry["spec"]).dtype: entry["symbol"] for entry in entries
        }
        for record in records:
            wanted = record["knobs"].get("reduce_symbol", "")
            dtype = "fp32" if record["knobs"].get("bias_f32") == 1 else record["dtype"]
            if wanted and wanted != symbols.get(dtype):
                reasons.append(
                    f"{record['name']}: names reduce symbol {wanted!r}, the {dtype} bias companion is {symbols.get(dtype)!r}"
                )
    return records, entries, reasons


def rows_of(
    op: str, arch: str, only: set[str] | None = None
) -> tuple[list, list[Spec], list[Spec]]:
    """(instances, specs, companion specs) of the RDNA-family rows of @p op on @p arch, in table order, checked against each other."""
    kind = op.removeprefix("sdpa_bwd_")
    instances = [
        i for i in instances_for(op, arch) if family(arch, i.dtype, kind) == "rdna"
    ]
    specs = specs_for(kind, arch)
    for instance, spec in zip(instances, specs, strict=True):
        knobs = instance.knobs
        if (
            instance.dtype,
            knobs["head_dim"],
            knobs["has_dropout"],
            knobs["padded"],
        ) != (
            spec.dtype,
            spec.tile,
            spec.dropout,
            spec.padded,
        ) or (
            knobs["causal"],
            knobs["has_bias"],
        ) != (
            int(spec.cls == "causal"),
            int(spec.cls == "bias"),
        ):
            raise GeneratorError(f"{instance.name} is not the row of {spec.key()}")
    if only is not None:
        unknown = only - {spec.hash() for spec in specs}
        if unknown:
            raise GeneratorError(f"{op} has no row with hash {sorted(unknown)}")
        keep = [i for i, spec in enumerate(specs) if spec.hash() in only]
        instances = [instances[i] for i in keep]
        specs = [specs[i] for i in keep]
    companions = specs_for("dbias_reduce", arch) if kind == "dq" else []
    return instances, specs, companions


def generate(
    op: str,
    arch: str,
    out_root: Path,
    only: set[str] | None = None,
    reference_dir: Path | None = None,
    reference_csv: Path | None = None,
    build_tailzero: cdna.CompanionBuilder | None = None,
    nondeterministic_csv: Path = nondet.NONDETERMINISTIC_CSV,
    stored_dir: Path | None = None,
) -> int:
    """Build every row of @p op (or the rows whose Spec hash is in @p only) for the generic arch @p arch (gfx11-generic).

    Writes ``<out_root>/<op>/<arch>/``; returns the number of ``error`` rows. The directory must not hold files already.
    @p reference_dir holds the S1 objects (``<hash>.hsaco``): every built object must be its sha256, which the rows of @p reference_csv
    record, and the run raises ``GeneratorError`` with nothing left in the op directory when one is not or does not build, except a
    row of the nondeterministic list (``build_rows``). The dQ op
    also builds the standalone dBias reduce objects of every dtype (its companions, ``manifest.json`` ``companions``) and checks
    that every bias row names the reduce symbol of its bias dtype's companion (``Spec.bias_dtype``). @p build_tailzero supplies the op's
    tail-zero companions (``(arch, op_dir) -> manifest entries``); they have no pin and no reference row. @p stored_dir is accepted so that
    one ``--stored-dir`` serves every op's command line; a generic arch has no combined row and reads nothing from it.
    """
    if op not in OPS or arch not in GENERIC_ARCHES:
        raise GeneratorError(
            f"{op} cannot be built for {arch!r} by this module alone; allowed: {', '.join(OPS)} on {', '.join(GENERIC_ARCHES)} "
            "(gfx942 and gfx90a have rows of both families: _sdpa_bwd_split_build)"
        )
    if reference_dir is None:
        raise GeneratorError(
            f"{op} checks its objects against the S1 pins and needs --reference-dir"
        )
    if env.rocm_version() == "unknown":
        raise GeneratorError(
            "the ROCm version is unknown: set ROCM_PATH to the ROCm install the build compiles against"
        )
    instances, specs, companions = rows_of(op, arch, only)
    op_dir = out_root / op / arch
    if not cdna._empty_dir(op_dir):
        raise GeneratorError(f"{op_dir} is not empty")
    reference = cdna.read_reference(reference_csv or _REFERENCE_CSVS[arch])
    for spec in (*specs, *companions):
        row = reference.get(spec.hash())
        if row is None or row["spec"] != spec.key():
            raise GeneratorError(f"the reference has no row for {spec.key()}")
        path = reference_dir / f"{spec.hash()}.hsaco"
        if not path.is_file() or sha256(path.read_bytes()) != row["sha256"]:
            raise GeneratorError(f"{path} is not the pin its reference row names")
    env.assert_flydsl_version()
    env.prepare(arch)
    records, entries, reasons = build_rows(
        op,
        arch,
        instances,
        specs,
        companions,
        reference,
        op_dir,
        reference_dir,
        nondeterministic_csv,
    )
    # The tail-zero objects are no row of this op: they have no pin and no reference row (the generator is their provenance).
    tailzero = []
    if not reasons and build_tailzero is not None:
        try:
            tailzero = build_tailzero(arch, op_dir)
        except GeneratorError as exc:
            reasons.append(" ".join(str(exc).split())[:500])
    if reasons:
        for written in op_dir.iterdir():
            written.unlink()
        raise GeneratorError(
            f"{len(reasons)} row(s) stop the run, nothing is shipped:\n  "
            + "\n  ".join(reasons)
        )
    provenance = env.provenance(arch)
    provenance.pop("aiter_kernels_commit", None)
    write_op_manifest(op_dir, arch, op, provenance, records)
    source_md = cdna._complete_source_md(refresh_source_md(op_dir), op, arch, records)
    if entries or tailzero:
        cdna.write_companions(op_dir, [*entries, *tailzero])
    print(
        f"{op}: {len(records)} ok, {len(entries) + len(tailzero)} companion(s); wrote {op_dir}"
    )
    print(f"refreshed {source_md}")
    return 0
