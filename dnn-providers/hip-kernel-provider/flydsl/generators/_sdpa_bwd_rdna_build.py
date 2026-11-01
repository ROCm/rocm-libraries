# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build the gfx11-generic SDPA backward code objects from ``Spec`` strings (``_sdpa_bwd_spec``).

    python -m generators.gen_sdpa_bwd_<prep|dkdv|dq> --arch gfx11-generic --out-dir <content root> --reference-dir <S1 reference dir>

The gfx11 counterpart of ``_sdpa_bwd_cdna_build``, which owns the command line, the compile worker pool and the
manifest of the gfx950 objects; the ``gen_sdpa_bwd_{prep,dkdv,dq}`` modules hand ``gfx11-generic`` to ``generate`` here.
The rows are the gfx11 tables of ``_instances`` (``SDPA_BWD_GFX11_TABLES``), each the pin of the S1 catalogue
(``tests/data/sdpa_bwd_gfx11_reference.csv``). The kernels are the ones under ``kernels_src/kernels/attention_bwd_rdna/``
(16-bit and fp32 families); they take the knob record of their tuning module and are built as the sweep built them:
``SWEEP_ARCH``-free, through the FlyDSL generic-target shim (``_flydsl_env.prepare``), in cache mode, under the module's own
compile hints.

What differs from gfx950:

* the launch grid of dK/dV is ``(kv_blocks, num_head_k, seqs)`` (``grid`` of the row; gfx950's is ``(num_head_k, kv_blocks, seqs)``);
* the workgroup is a multiple of the 32-lane wave, and the LDS limit is 64 KiB;
* the gfx950 gates (resource row, #1087 detectors, R-MIR scan) read gfx950 objects, so a row records ``gates: n/a``
  and its pinned resource row instead. What replaces them is the pin check: the build must be the S1 object, byte for byte;
* the dBias reduce kernel is not part of a dQ object. A bias dQ row names the standalone reduce kernel of its dtype
  (``reduce_symbol``) and the manifest lists the three standalone reduce objects under ``companions`` of ``sdpa_bwd_dq``.

A row that does not build ``ok`` or is not its pin stops the run, and nothing is shipped.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _flydsl_env as env
from . import _sdpa_bwd_cdna_build as cdna
from ._codeobject import (
    GeneratorError,
    arg_records,
    describe,
    verify_arch,
    verify_generic,
)
from ._extract_hsaco import HsacoExtractionError, extract_bin_attr
from ._instances import (
    SDPA_BWD_GFX11_KNOBS,
    SDPA_BWD_GFX11_SPLIT_KINDS,
    SDPA_BWD_GFX11_TABLES,
    SDPA_BWD_GFX11_TARGET,
    instances_for,
)
from ._manifest import (
    FLYDSL_DIR,
    MANIFEST_NAME,
    _dumps,
    refresh_source_md,
    source_md_path,
    sha256,
    write_op_manifest,
)
from ._sdpa_bwd_spec import FP32, UNSET_INT, UNSET_WORD, Spec

ARCH = SDPA_BWD_GFX11_TARGET

# The ops that build for gfx11-generic.
OPS = tuple(f"sdpa_bwd_{kind}" for kind in SDPA_BWD_GFX11_SPLIT_KINDS)

# The S1 catalogue's pins: ``hash, spec, sha256`` and the resource row of each object.
_REFERENCE_CSV = FLYDSL_DIR / "tests" / "data" / "sdpa_bwd_gfx11_reference.csv"

# Lanes of the gfx11 wave: a workgroup is `num_waves * WARP_SIZE` threads.
WARP_SIZE = 32

# LDS one workgroup may allocate on gfx11 (64 KiB). The compiler reports an object over it as an error, yet still emits the
# object, which then fails to launch.
_LDS_LIMIT_BYTES = 65536

# Launch grid, as the launchers' `.launch(grid=...)` compute it (`seqs` is `num_seqlens` when non-zero, else the batch size):
#   prep          (ceil(max_seqlen_q / rows_per_block), num_head_q, seqs)
#   dkdv          (ceil(max_seqlen_k / BLOCK_KV), num_head_k, seqs)
#   dq            (num_head_q, ceil(max_seqlen_q / BLOCK_M), seqs)
#   dbias_reduce  (blocks over the plane, planes, 1)
_GRID_RULE = {
    "prep": "qblocks_by_heads_by_seqs",
    "dkdv": "kvblocks_by_kheads_by_seqs",
    "dq": "qheads_by_qblocks_by_seqs",
    "dbias_reduce": "blocks_by_planes",
}

_RESOURCE_COLUMNS = (
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


def specs_for(kind: str) -> list[Spec]:
    """The Specs of the gfx11 table of @p kind, in table order (``_instances._sdpa_bwd_gfx11_instances`` spells the same rows)."""
    names = SDPA_BWD_GFX11_KNOBS[kind]
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
    ) in SDPA_BWD_GFX11_TABLES[kind]:
        pins = {
            name: value
            for name, value in zip(names, perf, strict=True)
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
                arch=ARCH,
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
    if spec.arch != ARCH:
        raise GeneratorError(f"{spec.key()} is not a {ARCH} Spec")
    mod, builder = _family(spec, spec.kernel)
    if spec.kernel == "prep":
        launch = builder(spec.dtype, spec.tile, bool(spec.padded))
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
    dQ: ``num_waves``, ``block_m``, ``block_n``, ``waves_per_eu``, ``block_threads``. The prep and reduce kernels fix their
    block; the engine derives it.
    """
    spec, knobs = built.spec, built.knobs
    if spec.kernel not in ("dkdv", "dq"):
        return {}
    waves_per_eu = knobs.waves_per_eu or -1
    threads = knobs.num_waves * WARP_SIZE
    if spec.kernel == "dkdv":
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
    }


def _compile_one(task: tuple[str]) -> cdna.Outcome:
    """Compile one spec. Never raises for a build failure.

    Cache mode, as the sweep built the pins: no IR dump is requested (FlyDSL skips its disk cache when one is), and the object is
    read back from the one artifact this compile wrote to a fresh cache directory.
    """
    (spec_s,) = task
    geometry = None
    try:
        spec = Spec.parse(spec_s)
        built = build(spec)
        geometry = geometry_of(built) or None
        work = Path(tempfile.mkdtemp(prefix=f"sdpa_bwd_gfx11_{spec.hash()}_"))
        try:
            os.environ.pop("FLYDSL_DUMP_IR", None)
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(work / "cache")
            with cdna.hints_ctx(built):
                built.jf(*cdna.synth_compile_args(built.jf))
            ((_, artifact),) = built.jf._mem_cache.items()
            blob = extract_bin_attr(artifact._ir_text, f"{spec.key()} cache artifact")
            if blob[:4] != b"\x7fELF":
                raise GeneratorError(
                    f"{spec.key()}: the extracted object is not an ELF"
                )
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
    return cdna.Outcome("ok", blob, geometry=geometry)


def verify(blob: bytes, spec: Spec, where: str) -> dict:
    """The facts of @p blob, or ``GeneratorError``: its target (name and ELF header), kernarg layout, LDS and block size must be the declared ones."""
    described = describe(blob, where)
    verify_arch(described, ARCH, where)
    verify_generic(blob, ARCH, where)
    if described["group_segment_fixed_size"] > _LDS_LIMIT_BYTES:
        raise GeneratorError(
            f"{where}: needs {described['group_segment_fixed_size']} bytes of LDS; "
            f"{ARCH} allows {_LDS_LIMIT_BYTES} per workgroup"
        )
    expected, _ = cdna._signature(cdna._PARAMS[spec.kernel])
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
    if threads % WARP_SIZE or not WARP_SIZE <= threads <= 1024:
        raise GeneratorError(f"{where}: {threads} threads per workgroup")
    return {**described, "block_threads": threads}


def _number(text: str):
    """A resource-row cell of the reference CSV: an int, or null where the pin has none."""
    return int(text) if text else None


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
        record["knobs"]["grid"] = _GRID_RULE[spec.kernel]
    if outcome.status != "ok":
        record["detail"] = outcome.detail
        return record
    filename = f"{instance.name}.hsaco"
    where = f"{ARCH}/{instance.op}/{filename}"
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
    _, names = cdna._signature(cdna._PARAMS[spec.kernel])
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
            "resources": {c: _number(pin[c]) for c in _RESOURCE_COLUMNS},
            "gates": "n/a",
        }
    )
    return record


def _write_companions(op_dir: Path, companions: list[dict]) -> None:
    """List the companion objects in the manifest of @p op_dir (``companions``, after ``instances``) and in its SOURCE.md."""
    path = op_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["companions"] = companions
    path.write_text(_dumps(manifest) + "\n", encoding="utf-8")
    source_md = source_md_path(op_dir)
    lines = [
        "",
        "## Companion objects",
        "",
        "Objects the instances launch beside their own, named by Spec hash.",
        "",
        "| Object | Symbol | SHA256 |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| `{c['file']}` (`{c['spec']}`) | `{c['symbol']}` | `{c['sha256'][:16]}` |"
        for c in companions
    ]
    text = source_md.read_text(encoding="utf-8").rstrip("\n")
    source_md.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _companion(spec: Spec, outcome: cdna.Outcome, pin: dict, op_dir: Path) -> dict:
    """The manifest entry of one standalone reduce object, written to ``<hash>.hsaco``; ``GeneratorError`` for one that is not its pin."""
    filename = f"{spec.hash()}.hsaco"
    where = f"{ARCH}/companion/{filename}"
    if outcome.status != "ok":
        raise GeneratorError(f"{where}: {outcome.status}: {outcome.detail}")
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


def generate(
    op: str,
    out_root: Path,
    only: set[str] | None = None,
    reference_dir: Path | None = None,
    reference_csv: Path = _REFERENCE_CSV,
) -> int:
    """Build every row of @p op (or the rows whose Spec hash is in @p only) for gfx11-generic.

    Writes ``<out_root>/<op>/gfx11-generic/``; returns the number of ``error`` rows. The directory must not hold files already.
    @p reference_dir holds the S1 objects (``<hash>.hsaco``): every built object must be its sha256, which the rows of @p reference_csv
    record, and the run raises ``GeneratorError`` with nothing left in the op directory when one is not or does not build. The dQ op
    also builds the standalone dBias reduce objects of every dtype (its companions, ``manifest.json`` ``companions``) and checks
    that every bias row names the reduce symbol of its dtype's companion.
    """
    if op not in OPS:
        raise GeneratorError(
            f"{op} cannot be built for {ARCH!r}; allowed: {', '.join(OPS)}"
        )
    if reference_dir is None:
        raise GeneratorError(
            f"{op} checks its objects against the S1 pins and needs --reference-dir"
        )
    if env.rocm_version() == "unknown":
        raise GeneratorError(
            "the ROCm version is unknown: set ROCM_PATH to the ROCm install the build compiles against"
        )
    kind = op.removeprefix("sdpa_bwd_")
    instances = instances_for(op, ARCH)
    specs = specs_for(kind)
    for instance, spec in zip(instances, specs, strict=True):
        knobs = instance.knobs
        if (instance.dtype, knobs["head_dim"], knobs["dropout"], knobs["padded"]) != (
            spec.dtype,
            spec.tile,
            spec.dropout,
            spec.padded,
        ) or (knobs["causal"], knobs["has_bias"]) != (
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
    companions = specs_for("dbias_reduce") if kind == "dq" else []
    op_dir = out_root / op / ARCH
    if not cdna._empty_dir(op_dir):
        raise GeneratorError(f"{op_dir} is not empty")
    reference = cdna.read_reference(reference_csv)
    for spec in (*specs, *companions):
        row = reference.get(spec.hash())
        if row is None or row["spec"] != spec.key():
            raise GeneratorError(f"the reference has no row for {spec.key()}")
        path = reference_dir / f"{spec.hash()}.hsaco"
        if not path.is_file() or sha256(path.read_bytes()) != row["sha256"]:
            raise GeneratorError(f"{path} is not the pin its reference row names")
    env.assert_flydsl_version()
    env.prepare(ARCH)
    everything = [*specs, *companions]
    outcomes = cdna._compile_all(
        _compile_one,
        [(spec.key(),) for spec in everything],
        ARCH,
        lambda task: Spec.parse(task[0]).hash(),
    )

    op_dir.mkdir(parents=True, exist_ok=True)
    records = []
    reasons = []
    for instance, spec, outcome in zip(instances, specs, outcomes):
        try:
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
        if records[-1]["build_status"] != "ok":
            reasons.append(
                f"{instance.name}: {records[-1]['build_status']}: {records[-1].get('detail', '')}"
            )
    entries = []
    for spec, outcome in zip(companions, outcomes[len(specs) :]):
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
            dtype = record["dtype"]
            if wanted and wanted != symbols.get(dtype):
                reasons.append(
                    f"{record['name']}: names reduce symbol {wanted!r}, the {dtype} companion is {symbols.get(dtype)!r}"
                )
    if reasons:
        for written in op_dir.iterdir():
            written.unlink()
        raise GeneratorError(
            f"{len(reasons)} row(s) stop the run, nothing is shipped:\n  "
            + "\n  ".join(reasons)
        )
    provenance = env.provenance(ARCH)
    provenance.pop("aiter_kernels_commit", None)
    write_op_manifest(op_dir, ARCH, op, provenance, records)
    source_md = cdna._complete_source_md(refresh_source_md(op_dir), op, ARCH, records)
    if companions:
        _write_companions(op_dir, entries)
    print(f"{op}: {len(records)} ok, {len(entries)} companion(s); wrote {op_dir}")
    print(f"refreshed {source_md}")
    return 0
