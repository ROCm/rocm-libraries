# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build the RDNA SDPA backward smoke object (``sdpa_bwd_rdna_dbias_reduce``) from a ``Spec``.

    python -m generators.gen_sdpa_bwd_rdna_dbias_reduce --arch gfx11-generic --out-dir <content root>

The op is the backward's landing place on ``gfx11-generic``: the dBias reduction of
``kernels/attention_bwd_rdna/fmha_bwd_dbias_reduce_rdna.py``, built through the same
generator machinery as the CDNA backward ops (a ``_sdpa_bwd_spec.Spec`` per row, one
compile per spec, ``ok`` / ``dual_object`` / ``error`` outcomes, the manifest and
SOURCE.md of ``_manifest``). What differs is the target: the arch allow-list below is
checked before anything compiles, the FlyDSL generic-target shim is installed, the
object's ELF header must name the generic target, and the compile carries the gfx11
LLVM options. The CDNA gates (resource row, #1087 detectors, R-MIR scan) are gfx950
checks and do not apply; the manifest records them as ``n/a``.

Nothing is retried: one build per spec, and ``dual_object`` is reported as such. The
exit status is 0 when every row is ``ok`` or ``dual_object`` and 1 on any ``error``
or a refused arch.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _flydsl_env as env
from . import _generic_targets
from ._codeobject import (
    GeneratorError,
    arg_records,
    describe,
    hkp_pack_module,
    verify_arch,
    verify_generic,
)
from ._extract_hsaco import HsacoExtractionError, hsaco_from_dump
from ._instances import instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all
from ._sdpa_bwd_spec import Spec, to_spec

OP = "sdpa_bwd_rdna_dbias_reduce"

# The arches each op is built for; enforced before compiling.
_OP_ARCHES = {OP: ("gfx11-generic",)}

# `_extract_hsaco.extract_bin_attr` words a dump with two different objects this way.
_DUAL_OBJECT_MARK = "embedded objects are not identical"

# LDS one workgroup may allocate on gfx11 (64 KiB). The compiler reports an object over
# it as an error, yet still emits the object, which then fails to launch.
_LDS_LIMIT_BYTES = {"gfx11-generic": 65536}

# LLVM codegen options per target, added to the kernel's own compile hints. Copied from
# the RDNA prefill build: the AMDGPU trackers' tighter register accounting is what fits
# gfx11-generic kernels without a spill.
_LLVM_OPTIONS = {"gfx11-generic": {"amdgpu-use-amdgpu-trackers": True}}

# One kernel argument is its pointer (8 bytes, `global_buffer`) or a by-value scalar.
_ARG_KINDS = {
    "ptr": ("global_buffer", 8),
    "i32": ("by_value", 4),
    "i64": ("by_value", 8),
}

# The kernel's own parameter list, in order: the `@flyc.kernel` signature of
# `fmha_bwd_dbias_reduce_rdna.py`. The native dispatch marshals by position, so a
# parameter list that changes is a change of this declaration, of the dispatch and of the
# descriptors together.
_PARAMS: tuple[tuple[str, str], ...] = (
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
)

_BLOCK_THREADS = 256
_GRID_RULE = "blocks_by_planes"

# Pointers the kernel assumes 16-byte aligned.
_ALIGNED_POINTERS = {"DB", "DBias"}


def _signature() -> tuple[tuple, tuple]:
    """The (kind, size, offset) layout and the argument names of ``_PARAMS``."""
    layout, names, offset = [], [], 0
    for name, kind in _PARAMS:
        value_kind, size = _ARG_KINDS[kind]
        offset = (offset + size - 1) // size * size
        layout.append((value_kind, size, offset))
        names.append(name)
        offset += size
    return tuple(layout), tuple(names)


def _refuse(arch: str) -> str | None:
    """The message refusing @p arch for this op, if it does."""
    if arch in _OP_ARCHES[OP]:
        return None
    return f"{OP} cannot be built for {arch!r}; allowed: {', '.join(_OP_ARCHES[OP])}"


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


def _synth_compile_args(jf):
    """Arguments that trace @p jf: null pointers, zero scalars, the null stream."""
    import flydsl.compiler as flyc  # noqa: PLC0415
    import flydsl.expr as fx  # noqa: PLC0415
    from flydsl.compiler.jit_argument import resolve_signature  # noqa: PLC0415

    args = []
    for p in resolve_signature(jf.func).parameters.values():
        n = getattr(p.annotation, "__name__", str(p.annotation))
        if n == "Pointer":
            align = 16 if p.name in _ALIGNED_POINTERS else None
            args.append(flyc.from_c_void_p(fx.Uint8, 0, assumed_align=align))
        elif n in ("Int32", "Int64"):
            args.append(0)
        elif n == "Stream":
            args.append(fx.Stream(None))
        else:
            raise NotImplementedError(f"annotation {n} for {p.name}")
    return args


@dataclass
class Outcome:
    """One spec's build: its status, the object and why it failed."""

    status: str
    blob: bytes | None = None
    detail: str = ""


def _compile_one(task: tuple[str, str]) -> Outcome:
    """Compile one spec for ``arch``. Never raises for a build failure."""
    spec_s, arch = task
    try:
        spec = Spec.parse(spec_s)
        if spec.kernel != "dbias_reduce":
            raise ValueError(f"{OP} builds dbias_reduce specs, not {spec.kernel}")
        from flydsl.compiler.kernel_function import (  # noqa: PLC0415
            CompilationContext,
        )
        from kernels.attention_bwd_rdna import (  # noqa: PLC0415
            fmha_bwd_dbias_reduce_rdna as mod,
        )

        launch = mod.build_fmha_bwd_dbias_reduce_rdna_module_primary(spec.dtype)
        jf = _single_jit(launch)
        # FlyDSL merges hint layers shallowly: `llvm_options` passed here would replace
        # the kernel's own, not extend it. Merge onto them explicitly.
        hints = {k: v for k, v in mod._COMPILE_HINTS.items()}
        options = dict(hints.get("llvm_options") or {})
        options.update(_LLVM_OPTIONS[arch])
        hints["llvm_options"] = options
        dump_dir = Path(tempfile.mkdtemp(prefix=f"sdpa_bwd_rdna_{spec.hash()}_"))
        try:
            env.set_dump_dir(dump_dir / "ir")
            with CompilationContext.compile_hints(hints):
                jf(*_synth_compile_args(jf))
            blob, _ = hsaco_from_dump(dump_dir / "ir")
        finally:
            shutil.rmtree(dump_dir, ignore_errors=True)
    except HsacoExtractionError as exc:
        status = "dual_object" if _DUAL_OBJECT_MARK in str(exc) else "error"
        return Outcome(status, detail=" ".join(str(exc).split())[:500])
    except Exception as exc:  # noqa: BLE001 - reported per spec, never retried
        return Outcome(
            "error",
            detail=f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500],
        )
    return Outcome("ok", blob)


def _max_flat_workgroup_size(blob: bytes, where: str, symbol: str) -> int:
    hkp = hkp_pack_module()
    document = hkp._metadata_document(hkp.amdgcn_object(blob, where), where)
    kernel = next(
        k
        for k in document["amdhsa.kernels"]
        if k[".symbol"].removesuffix(".kd") == symbol
    )
    return int(kernel[".max_flat_workgroup_size"])


def verify(blob: bytes, arch: str, where: str) -> dict:
    """The facts of @p blob, or ``GeneratorError``: its target (name and ELF header),
    kernarg layout, LDS and block size must be the declared ones."""
    described = describe(blob, where)
    verify_arch(described, arch, where)
    verify_generic(blob, arch, where)
    limit = _LDS_LIMIT_BYTES[arch]
    if described["group_segment_fixed_size"] > limit:
        raise GeneratorError(
            f"{where}: needs {described['group_segment_fixed_size']} bytes of LDS; "
            f"{arch} allows {limit} per workgroup"
        )
    expected, _ = _signature()
    if described["signature"] != expected:
        raise GeneratorError(
            f"{where}: kernarg layout is\n  {described['signature']}\nbut {OP} is declared as\n"
            f"  {expected}\nThe kernel's parameter list changed; update _PARAMS, the native "
            "dispatch and its signature together."
        )
    size = expected[-1][2] + expected[-1][1]
    if described["kernarg_segment_size"] != size:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, expected {size}"
        )
    threads = _max_flat_workgroup_size(blob, where, described["symbol"])
    if threads != _BLOCK_THREADS:
        raise GeneratorError(
            f"{where}: built for {threads} threads per workgroup, declared {_BLOCK_THREADS}"
        )
    return {**described, "block_threads": threads}


def _empty_dir(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


def _spec_of(instance) -> Spec:
    """The Spec of a row of this op: the CDNA dBias-reduce spelling of the same row."""
    return to_spec(dataclasses.replace(instance, op="sdpa_bwd_dbias_reduce"))


def _record(instance, spec: Spec, outcome: Outcome, op_dir: Path, arch: str) -> dict:
    """The manifest record of one row: its build status, and for an ``ok`` row the object."""
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
    if outcome.status != "ok":
        record["detail"] = outcome.detail
        return record
    filename = f"{instance.name}.hsaco"
    where = f"{arch}/{instance.op}/{filename}"
    blob = outcome.blob
    described = verify(blob, arch, where)
    (op_dir / filename).write_bytes(blob)
    delivered = sha256((op_dir / filename).read_bytes())
    _, names = _signature()
    record.update(
        {
            "block": [described["block_threads"], 1, 1],
            "grid_rule": _GRID_RULE,
            "args": arg_records(names, described["signature"]),
            "file": filename,
            "symbol": described["symbol"],
            "bytes": len(blob),
            "sha256": delivered,
            "kernarg_segment_size": described["kernarg_segment_size"],
            "lds_bytes": described["group_segment_fixed_size"],
            "vgprs": described["vgpr_count"],
            "vgpr_spills": described.get("vgpr_spill_count", 0),
            "llvm_options": _LLVM_OPTIONS[arch],
            "named_args": described["named_args"],
            "gates": "n/a",
        }
    )
    return record


def generate(arch: str, out_root: Path, only: set[str] | None = None) -> int:
    """Build every row of the op (or the rows whose Spec hash is in @p only) for @p arch.

    Writes ``<out_root>/<op>/<arch>/``; returns the number of ``error`` rows. The
    directory must not hold files already.
    """
    refusal = _refuse(arch)
    if refusal:
        raise GeneratorError(refusal)
    instances = instances_for(OP, arch)
    if only is not None:
        unknown = only - {_spec_of(i).hash() for i in instances}
        if unknown:
            raise GeneratorError(f"{OP} has no row with hash {sorted(unknown)}")
        instances = [i for i in instances if _spec_of(i).hash() in only]
    op_dir = out_root / OP / arch
    if not _empty_dir(op_dir):
        raise GeneratorError(f"{op_dir} is not empty")
    env.assert_flydsl_version()
    env.prepare(arch)
    _generic_targets.install(arch)
    specs = [_spec_of(instance) for instance in instances]
    outcomes = compile_all(
        _compile_one,
        [(spec.key(), arch) for spec in specs],
        arch,
        lambda task: task[0],
    )

    op_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for instance, spec, outcome in zip(instances, specs, outcomes):
        try:
            records.append(_record(instance, spec, outcome, op_dir, arch))
        except GeneratorError as exc:
            failed = Outcome("error", detail=" ".join(str(exc).split())[:500])
            records.append(_record(instance, spec, failed, op_dir, arch))
    # This kernel does not come from AITER, so its pin is no part of how the bytes came to be.
    provenance = env.provenance(arch)
    provenance.pop("aiter_kernels_commit", None)
    write_op_manifest(op_dir, arch, OP, provenance, records)
    source_md = refresh_source_md(op_dir)

    status = {
        s: sum(r["build_status"] == s for r in records)
        for s in ("ok", "dual_object", "error")
    }
    print(f"{OP}: {status}; wrote {op_dir}")
    print(f"refreshed {source_md}")
    return status["error"]


def run_op(description: str, argv: list[str] | None = None) -> int:
    """The command line of the op's generator (``gen_sdpa_bwd_rdna_dbias_reduce.py``)."""
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
    args = parser.parse_args(argv)
    refusal = _refuse(args.arch)
    if refusal:
        print(f"error: {refusal}", file=sys.stderr)
        return 1
    only = set(args.only.split(",")) if args.only else None
    try:
        errors = generate(args.arch, args.out_dir, only)
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1 if errors else 0
