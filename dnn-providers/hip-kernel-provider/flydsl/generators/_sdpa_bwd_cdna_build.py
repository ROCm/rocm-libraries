# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build the CDNA SDPA backward code objects from ``Spec`` strings (``_sdpa_bwd_spec``).

    python -m generators._sdpa_bwd_cdna_build --specs specs.csv --arch gfx950 --out-dir /path/to/empty

``specs.csv`` has a ``spec`` column. Each spec is compiled once, in its own process
slot (``_parallel``: ``FLYDSL_GEN_JOBS`` sets the worker count), and the object is
recovered from the IR dump. ``<out-dir>/<hash>.hsaco`` is written per built spec, and
``<out-dir>/results.csv`` (``hash,spec,status,sha256,detail``) records every spec:

* ``ok``          the object was extracted;
* ``dual_object`` the dump holds two different objects and the extractor refused to choose;
* ``error``       anything else (an invalid spec, a build or compile failure).

Rebuilding from identical sources can yield a different object, or a dump holding two
different objects: LLVM's AMDGPU backend is not deterministic for some kernels. Those
outcomes are reported, never retried; one build per spec. Comparing an object against
a pin is the caller's job.

The kernels are the ones under ``kernels_src/kernels/attention_bwd_cdna/``. Each is
traced with no tensors: the launchers take raw pointers and integers, so the arguments
are synthesised from the launcher's own signature. Every build constructs a fresh
launcher (a reused JitFunction is a cache hit) and compiles into a fresh dump directory.

The exit status is 0 when every spec is ``ok`` or ``dual_object``, and 1 when any is
``error`` or the arch is refused.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _flydsl_env as env
from ._codeobject import GeneratorError
from ._extract_hsaco import HsacoExtractionError, hsaco_from_dump
from ._manifest import sha256
from ._parallel import compile_all
from ._sdpa_bwd_spec import (
    KERNELS,
    LADDER,
    Spec,
    rung_below,
)  # noqa: F401  (LADDER is part of this module's surface)

# The arches each op is built for. One entry per op, by `sdpa_bwd_<kernel>`.
_OP_ARCHES = {f"sdpa_bwd_{kernel}": ("gfx950",) for kernel in KERNELS}

# `_extract_hsaco.extract_bin_attr` words a dump with two different objects this way.
_DUAL_OBJECT_MARK = "embedded objects are not identical"

_RESULT_COLUMNS = ("hash", "spec", "status", "sha256", "detail")


def op_of(spec: Spec) -> str:
    return f"sdpa_bwd_{spec.kernel}"


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
        bd.meta, bd.knobs, bq.meta, bq.knobs, spec.role_order or "concat"
    )
    return Built(
        spec,
        launch.jit,
        mod,
        (bd.meta, bq.meta),
        (bd.knobs, bq.knobs),
        _hints_of(mod, spec),
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


def _compile_one(task: tuple[str, str]) -> tuple[str, bytes | None, str]:
    """Compile one spec; return (status, object or None, detail). Never raises for a build failure."""
    spec_s, _ = task
    try:
        spec = Spec.parse(spec_s)
        built = build(spec)
        dump_dir = Path(tempfile.mkdtemp(prefix=f"sdpa_bwd_{spec.hash()}_"))
        try:
            env.set_dump_dir(dump_dir)
            with hints_ctx(built):
                built.jf(*synth_compile_args(built.jf))
            blob, _ = hsaco_from_dump(dump_dir)
        finally:
            shutil.rmtree(dump_dir, ignore_errors=True)
    except HsacoExtractionError as exc:
        status = "dual_object" if _DUAL_OBJECT_MARK in str(exc) else "error"
        return status, None, " ".join(str(exc).split())[:500]
    except Exception as exc:  # noqa: BLE001 - reported per spec, never retried
        return (
            "error",
            None,
            f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500],
        )
    return "ok", blob, ""


def _read_specs(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "spec" not in reader.fieldnames:
            raise GeneratorError(f"{path}: no `spec` column")
        return [row["spec"] for row in reader if row["spec"].strip()]


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
    refused = sorted(
        {
            op_of(s)
            for _, _, s, _ in entries
            if s is not None and args.arch not in _OP_ARCHES[op_of(s)]
        }
    )
    if refused:
        print(
            f"error: {', '.join(refused)} cannot be built for {args.arch!r}; "
            f"allowed: {', '.join(sorted({a for op in refused for a in _OP_ARCHES[op]}))}",
            file=sys.stderr,
        )
        return 1

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        print(f"error: --out-dir {args.out_dir} is not empty", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        env.assert_flydsl_version()
        env.prepare(args.arch)
        todo = [e for e in entries if e[2] is not None]
        outcomes = compile_all(
            _compile_one,
            [(key, digest) for digest, key, _, _ in todo],
            args.arch,
            lambda task: task[1],
        )
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
            if spec is None:
                status, blob, detail = "error", None, problem
            else:
                status, blob, detail = by_hash[digest]
            if blob is not None:
                (args.out_dir / f"{digest}.hsaco").write_bytes(blob)
            failed |= status == "error"
            writer.writerow(
                (digest, key, status, sha256(blob) if blob is not None else "", detail)
            )
    counts = {
        s: sum(1 for o in by_hash.values() if o[0] == s)
        for s in ("ok", "dual_object", "error")
    }
    print(
        f"{counts} (+{sum(1 for e in entries if e[2] is None)} unparsable); results in {args.out_dir / 'results.csv'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
