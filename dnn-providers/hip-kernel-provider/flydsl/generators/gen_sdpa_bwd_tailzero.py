# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the SDPA backward tail-zero objects (gfx950 and gfx11-generic): the kernel that zeroes the padded rows of dense gradients.

    python -m generators.gen_sdpa_bwd_tailzero --arch gfx950 --out-dir <scratch root>

One object per arch family and dtype (``f16``, ``bf16``, ``fp32``; six in all), and one V-head object per arch family and 16-bit dtype
(``tailzero_hv``: ``f16``, ``bf16``; four in all; fp32 U2 declines): the kernel with a separate ``Hv`` (dV head count) after ``Hkv``, for
graphs whose V has its own head count. The kernels are ``fmha_bwd_tailzero_gfx950`` and ``fmha_bwd_tailzero_gfx11``
(``fmha_bwd_tailzero_hv_gfx950``, ``fmha_bwd_tailzero_hv_gfx11``) under ``kernels_src/kernels/attention_bwd_{cdna,rdna}/``. The objects are
companions of the dQ and merged rows (``companion_entries``: the dQ and merged builds write them into the dQ arch folder and list them
under ``companions`` of the dQ manifest, role ``tailzero``). They have no pin: this generator is their provenance, so each is built
twice in fresh processes and the two builds must be byte-identical, its kernarg layout must be ``PARAMS`` (``PARAMS_HV``) and, on gfx950, the R-MIR scan
of ``tools/sdpa_bwd_gate.py`` must not be red (the gfx11 gates read gfx950 objects only). The resource row of a gfx950 object is recorded.

The command line writes ``<out-dir>/sdpa_bwd_tailzero/<arch>/`` (objects, ``manifest.json``, ``SOURCE.md``) for inspection and for the
spike that measures them; it is not shipped content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    SDPA_BWD_GFX11_TARGET,
    SDPA_BWD_TAILZERO_DTYPES,
    SDPA_BWD_TAILZERO_THREADS,
    SDPA_BWD_TARGET,
    instances_for,
)
from ._manifest import FLYDSL_DIR, refresh_source_md, sha256, write_op_manifest

OP = "sdpa_bwd_tailzero"
ROLE = "tailzero"
ROLE_HV = "tailzero_hv"
HV_DTYPES = ("f16", "bf16")
ARCHES = (SDPA_BWD_TARGET, SDPA_BWD_GFX11_TARGET)

# The kernel's name in each family's object (a one-kernel object: FlyDSL suffixes it ``_kernel_0``).
SYMBOL = {
    SDPA_BWD_TARGET: "fmha_bwd_tailzero_gfx950_kernel_0",
    SDPA_BWD_GFX11_TARGET: "fmha_bwd_tailzero_gfx11_kernel_0",
}
SYMBOL_HV = {
    SDPA_BWD_TARGET: "fmha_bwd_tailzero_hv_gfx950_kernel_0",
    SDPA_BWD_GFX11_TARGET: "fmha_bwd_tailzero_hv_gfx11_kernel_0",
}

# The kernel's parameter list, in order (``@flyc.kernel`` of both families). The native dispatch marshals by position, so a list that
# changes is a change of this declaration, of the dispatch and of the descriptors together.
PARAMS: tuple[tuple[str, str], ...] = (
    ("DQ", "ptr"),
    ("DK", "ptr"),
    ("DV", "ptr"),
    ("seq_len_q", "ptr"),
    ("seq_len_kv", "ptr"),
    ("B", "i32"),
    ("Hq", "i32"),
    ("Hkv", "i32"),
    ("Sq", "i32"),
    ("Skv", "i32"),
    ("Dqk", "i32"),
    ("Dv", "i32"),
    ("stride_dq_batch", "i64"),
    ("stride_dq_head", "i64"),
    ("stride_dq_seq", "i64"),
    ("stride_dk_batch", "i64"),
    ("stride_dk_head", "i64"),
    ("stride_dk_seq", "i64"),
    ("stride_dv_batch", "i64"),
    ("stride_dv_head", "i64"),
    ("stride_dv_seq", "i64"),
    ("mask", "i32"),
)

# The V-head kernel's list: ``PARAMS`` with ``Hv`` (dV's head count) after ``Hkv``; planes per batch are ``Hq + Hk + Hv``.
_AFTER_HKV = [name for name, _ in PARAMS].index("Hkv") + 1
PARAMS_HV: tuple[tuple[str, str], ...] = (
    *PARAMS[:_AFTER_HKV],
    ("Hv", "i32"),
    *PARAMS[_AFTER_HKV:],
)

# Launch grid, as the launcher's ``.launch(grid=...)`` computes it: blocks over the tail units, B * (Hq + Hk + Hv) planes, 1 (Hv = Hkv for the standard object).
GRID_RULE = "blocks_by_planes"

# Builds of one object that must agree.
_BUILDS = 2


def spec_key(arch: str, dtype: str, hv: bool = False) -> str:
    """The companion's ``spec`` string: role (``tailzero_hv`` when @p hv), dtype, and the arch for gfx11-generic (as the gfx11 reduce
    companions spell theirs)."""
    key = f"{ROLE_HV if hv else ROLE},{dtype}"
    return key if arch == SDPA_BWD_TARGET else f"{key};arch={arch}"


def spec_hash(arch: str, dtype: str, hv: bool = False) -> str:
    """The first 12 hex digits of the sha256 of the key, as a ``Spec`` hash."""
    return hashlib.sha256(spec_key(arch, dtype, hv).encode()).hexdigest()[:12]


def object_name(arch: str, dtype: str, hv: bool = False) -> str:
    return f"{spec_hash(arch, dtype, hv)}.hsaco"


def _launcher(arch: str, dtype: str, hv: bool = False):
    """(module, launcher) of the family's builder for @p dtype. Static imports only: the inputs digest follows import statements."""
    if arch == SDPA_BWD_TARGET:
        from kernels.attention_bwd_cdna import fmha_bwd_tailzero_gfx950 as mod

        if hv:
            return mod, mod.build_fmha_bwd_tailzero_hv_gfx950_module_primary(dtype)
        return mod, mod.build_fmha_bwd_tailzero_gfx950_module_primary(dtype)
    from kernels.attention_bwd_rdna import fmha_bwd_tailzero_gfx11 as mod

    if hv:
        return mod, mod.build_fmha_bwd_tailzero_hv_gfx11_module_primary(dtype)
    return mod, mod.build_fmha_bwd_tailzero_gfx11_module_primary(dtype)


def _run_gate(blob: bytes, greedy: Path, dtype: str, hv: bool = False) -> dict:
    """The gfx950 gates of ``tools/sdpa_bwd_gate.py`` over @p blob and the greedy dump of its compile."""
    obj = greedy.with_name("obj.hsaco")
    obj.write_bytes(blob)
    done = subprocess.run(
        [
            sys.executable,
            str(FLYDSL_DIR / "tools" / "sdpa_bwd_gate.py"),
            "--hsaco",
            str(obj),
            "--greedy-dump",
            str(greedy),
            "--spec",
            spec_key(SDPA_BWD_TARGET, dtype, hv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise GeneratorError(f"gate: {done.stderr.strip()}")
    gate = json.loads(done.stdout)
    if gate["object_sha256"] != sha256(blob):
        raise GeneratorError("gate: scanned an object other than the delivered one")
    return gate


def _compile_one(task: tuple[str, str, bool]) -> cdna.Outcome:
    """Compile one object (and gate it on gfx950). Never raises for a build failure.

    Cache mode, as every backward object is built: no IR dump is requested, and the object is read back from the one artifact this
    compile wrote to a fresh cache directory.
    """
    arch, dtype, hv = task
    try:
        _, launch = _launcher(arch, dtype, hv)
        jf = cdna._single_jit(launch)
        work = Path(tempfile.mkdtemp(prefix=f"sdpa_bwd_tailzero_{dtype}_"))
        gate = None
        try:
            os.environ.pop("FLYDSL_DUMP_IR", None)
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(work / "cache")
            greedy = work / "greedy.txt"
            if arch == SDPA_BWD_TARGET:
                with cdna._stderr_to(greedy):
                    cdna._print_after_greedy()
                    jf(*cdna.synth_compile_args(jf))
            else:
                jf(*cdna.synth_compile_args(jf))
            ((_, artifact),) = jf._mem_cache.items()
            label = f"{spec_key(arch, dtype, hv)} cache artifact"
            blob = extract_bin_attr(artifact._ir_text, label)
            if blob[:4] != b"\x7fELF":
                raise GeneratorError(f"{label}: the extracted object is not an ELF")
            if arch == SDPA_BWD_TARGET:
                gate = _run_gate(blob, greedy, dtype, hv)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    except HsacoExtractionError as exc:
        status = "dual_object" if cdna._DUAL_OBJECT_MARK in str(exc) else "error"
        return cdna.Outcome(status, detail=" ".join(str(exc).split())[:500])
    except Exception as exc:  # noqa: BLE001 - reported per object, never retried
        return cdna.Outcome(
            "error", detail=f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500]
        )
    return cdna.Outcome("ok", blob, gate=gate)


def verify(blob: bytes, arch: str, where: str, hv: bool = False) -> dict:
    """The facts of @p blob, or ``GeneratorError``: target, symbol, kernarg layout and block size must be the declared ones
    (the V-head object's: ``SYMBOL_HV`` and ``PARAMS_HV``)."""
    symbol = (SYMBOL_HV if hv else SYMBOL)[arch]
    params = PARAMS_HV if hv else PARAMS
    described = describe(blob, where)
    verify_arch(described, arch, where)
    verify_generic(blob, arch, where)
    if described["symbol"] != symbol:
        raise GeneratorError(
            f"{where}: kernel is {described['symbol']!r}, expected {symbol!r}"
        )
    expected, _ = cdna._signature(params)
    if described["signature"] != expected:
        raise GeneratorError(
            f"{where}: kernarg layout is\n  {described['signature']}\nbut the tail-zero kernel is declared as\n  {expected}\n"
            "The kernel's parameter list changed; update PARAMS and PARAMS_HV, the native dispatch and its signature together."
        )
    size = expected[-1][2] + expected[-1][1]
    if described["kernarg_segment_size"] != size:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, expected {size}"
        )
    threads = cdna._max_flat_workgroup_size(blob, where, described["symbol"])
    if threads != SDPA_BWD_TAILZERO_THREADS:
        raise GeneratorError(
            f"{where}: built for {threads} threads per workgroup, declared {SDPA_BWD_TAILZERO_THREADS}"
        )
    if described["group_segment_fixed_size"]:
        raise GeneratorError(
            f"{where}: uses {described['group_segment_fixed_size']} bytes of LDS; the kernel has none"
        )
    return {**described, "block_threads": threads}


def build_objects(arch: str, hv: bool = False) -> dict[str, tuple[bytes, dict | None]]:
    """The object of each dtype for @p arch: ``{dtype: (bytes, gate or None)}``; the V-head objects (``HV_DTYPES``) when @p hv.

    Every object is built ``_BUILDS`` times (``FLYDSL_GEN_JOBS`` workers, each build in a process of its own); the builds must be
    byte-identical, the object must be its declared self (``verify``) and a gfx950 object's R-MIR verdict must not be red.
    """
    if arch not in ARCHES:
        raise GeneratorError(f"{OP} is built for {', '.join(ARCHES)}, not {arch!r}")
    if env.rocm_version() == "unknown":
        raise GeneratorError(
            "the ROCm version is unknown: set ROCM_PATH to the ROCm install the build compiles against"
        )
    env.assert_flydsl_version()
    env.prepare(arch)
    dtypes = HV_DTYPES if hv else SDPA_BWD_TAILZERO_DTYPES
    tasks = [(arch, dtype, hv) for dtype in dtypes] * _BUILDS
    outcomes = cdna._compile_all(
        _compile_one, tasks, arch, lambda task: spec_hash(*task)
    )
    built: dict[str, tuple[bytes, dict | None]] = {}
    reasons = []
    for (_, dtype, _), outcome in zip(tasks, outcomes):
        where = f"{arch}/{spec_key(arch, dtype, hv)}"
        if outcome.status != "ok":
            reasons.append(f"{where}: {outcome.status}: {outcome.detail}")
        elif dtype not in built:
            built[dtype] = (outcome.blob, outcome.gate)
        elif built[dtype][0] != outcome.blob:
            reasons.append(
                f"{where}: two builds differ ({sha256(built[dtype][0])[:16]} != {sha256(outcome.blob)[:16]})"
            )
        if outcome.gate and outcome.gate["rmir"]["verdict"] not in ("ok", "known"):
            reasons.append(f"{where}: R-MIR is {outcome.gate['rmir']['verdict']}")
    for dtype, (blob, _) in built.items():
        try:
            verify(blob, arch, f"{arch}/{spec_key(arch, dtype, hv)}", hv)
        except GeneratorError as exc:
            reasons.append(" ".join(str(exc).split())[:500])
    if reasons:
        raise GeneratorError(
            f"{len(reasons)} object(s) stop the run, nothing is shipped:\n  "
            + "\n  ".join(reasons)
        )
    return built


def companion_entries(arch: str, folder: Path) -> list[dict]:
    """Build the objects of @p arch (the three standard ones, then the two V-head ones) into @p folder and return their manifest
    ``companions`` entries.

    An entry is ``hash``, ``spec`` (``spec_key``: its first field is the role), ``file``, ``symbol`` and ``sha256``, as the gfx11 reduce
    companions are; a gfx950 entry also carries ``resources``, the resource row of its object.
    """
    entries = []
    for hv in (False, True):
        symbol = (SYMBOL_HV if hv else SYMBOL)[arch]
        for dtype, (blob, gate) in build_objects(arch, hv).items():
            name = object_name(arch, dtype, hv)
            (folder / name).write_bytes(blob)
            entry = {
                "hash": spec_hash(arch, dtype, hv),
                "spec": spec_key(arch, dtype, hv),
                "file": name,
                "symbol": symbol,
                "sha256": sha256((folder / name).read_bytes()),
            }
            if gate is not None:
                entry["resources"] = gate["resources"]
            entries.append(entry)
    return entries


def generate(arch: str, out_root: Path) -> int:
    """Write ``<out_root>/sdpa_bwd_tailzero/<arch>/``: the objects (the V-head ones beside them, listed by no manifest row),
    ``manifest.json`` (the rows of ``_instances`` with their object) and ``SOURCE.md``. The directory must not hold files already. Returns 0.
    """
    op_dir = out_root / OP / arch
    if op_dir.is_dir() and any(op_dir.iterdir()):
        raise GeneratorError(f"{op_dir} is not empty")
    built = build_objects(arch)
    built_hv = build_objects(arch, hv=True)
    op_dir.mkdir(parents=True, exist_ok=True)
    for dtype, (blob, _) in built_hv.items():
        name = object_name(arch, dtype, hv=True)
        (op_dir / name).write_bytes(blob)
        verify(blob, arch, f"{arch}/{name}", hv=True)
    records = []
    for instance in instances_for(OP, arch):
        blob, gate = built[instance.dtype]
        name = object_name(arch, instance.dtype)
        (op_dir / name).write_bytes(blob)
        described = verify(blob, arch, f"{arch}/{name}")
        _, names = cdna._signature(PARAMS)
        record = instance.to_record()
        record.update(
            {
                "build_status": "ok",
                "spec": spec_key(arch, instance.dtype),
                "spec_hash": spec_hash(arch, instance.dtype),
                "file": name,
                "symbol": SYMBOL[arch],
                "sha256": sha256((op_dir / name).read_bytes()),
                "block": [described["block_threads"], 1, 1],
                "grid_rule": GRID_RULE,
                "args": arg_records(names, described["signature"]),
                "bytes": len(blob),
                "kernarg_segment_size": described["kernarg_segment_size"],
                "lds_bytes": described["group_segment_fixed_size"],
                "vgprs": described["vgpr_count"],
                "vgpr_spills": described.get("vgpr_spill_count", 0),
                "named_args": described["named_args"],
            }
        )
        if gate is not None:
            record["resources"] = gate["resources"]
            record["gates"] = {"rmir": gate["rmir"]["verdict"]}
        records.append(record)
    provenance = env.provenance(arch)
    provenance.pop("aiter_kernels_commit", None)
    write_op_manifest(op_dir, arch, OP, provenance, records)
    refresh_source_md(op_dir)
    print(f"{OP}: {len(records)} ok, {len(built_hv)} {ROLE_HV}; wrote {op_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch", required=True, help="GPU target; one arch per invocation"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="scratch root to write sdpa_bwd_tailzero/<arch>/ under; it must be empty",
    )
    args = parser.parse_args(argv)
    if args.arch not in ARCHES:
        print(
            f"error: {OP} builds {', '.join(ARCHES)}, not {args.arch!r}",
            file=sys.stderr,
        )
        return 1
    try:
        return generate(args.arch, args.out_dir)
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
