# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Register / resource + ISA-mix + occupancy probe for a single kernel.

This is the builder-agnostic version of the old grouped-GEMM ``_probe_regs``
one-off: it reports, for one compiled kernel, the VGPR / SGPR / spill / LDS /
scratch footprint, the coarse ISA opcode mix (MFMA, ``ds_read`` / ``ds_write``,
``buffer_load_lds``, ``s_waitcnt`` / ``s_barrier`` / ``sched_barrier``, VALU /
SALU), the ``ds_read/mfma`` ratio, and a gfx950-style occupancy estimate.

It imports **nothing from** ``rocke.instances`` / kernels. It only touches
``rocke.core`` (types + builder, used by the self-contained ``--demo`` kernel),
``rocke.helpers.compile`` (to turn a ``KernelDef`` into a code object) and
``rocke.analysis.isa`` (to disassemble + parse). Feed it a kernel one of three
ways, none of which couple this file to a specific kernel family:

1. A pre-built code object on disk (fully decoupled -- no rocke build at all)::

       python probe_regs.py --co path/to/kernel.co
       python probe_regs.py --hsaco path/to/kernel.hsaco   # hipcc bundle ok

2. A builder resolved dynamically at runtime (the import happens in *your*
   string, not in this module)::

       python probe_regs.py --builder rocke.instances.gfx950.grouped_gemm:build_grouped_gemm \
           --builder-kwargs '{}' --bs 512

3. Programmatically, passing ``KernelDef`` objects you built yourself::

       from probe_regs import probe_regs
       probe_regs([("my_variant", my_kernel_def, 512)])  # (label, KernelDef, block_size)

With no arguments it builds a tiny self-contained kernel via ``IRBuilder`` and
probes that, so the tool is CI-safe and needs no GPU (compile + static
inspection only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Silence the cpp-engine fallback note: this probe only inspects the emitted
# code object, and the Python lowerer is byte-identical to the C++ engine.
os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")


# ---- rocke bootstrap ---------------------------------------------------


def _bootstrap_rocke() -> None:
    """Add the platform ``Python/`` dir to ``sys.path`` if rocke is not yet
    importable. Set ``PYTHONPATH`` to override."""
    try:
        import rocke  # noqa: F401

        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Python"
        if (candidate / "rocke" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
        if (parent / "rocke" / "__init__.py").exists():
            sys.path.insert(0, str(parent))
            return


_bootstrap_rocke()

from rocke.analysis.isa import analyze_hsaco  # noqa: E402
from rocke.helpers.compile import (  # noqa: E402
    compile_kernel,
    compile_kernel_via_hipcc,
)


# ---- arch caps (occupancy model) ---------------------------------------


@dataclass(frozen=True)
class ArchCaps:
    """Per-CU caps for the coarse occupancy estimate.

    CDNA3+ (gfx942/gfx950) has a *unified* vector register file: VGPRs hold
    both the MFMA accumulators and operand/address registers (no separate AGPR
    pool), so occupancy is bounded by ``max(VGPR-per-wave, LDS-per-block)``.
    """

    name: str
    vgpr_per_simd: int = 512
    simd_per_cu: int = 4
    lds_per_cu: int = 160 * 1024
    vgpr_alloc_granularity: int = 8


ARCH_BY_NAME = {
    "gfx950": ArchCaps("gfx950", 512, 4, 160 * 1024, 8),  # MI355X (CDNA4)
    "gfx942": ArchCaps("gfx942", 512, 4, 64 * 1024, 8),  # MI300X (CDNA3)
}


# ---- llvm tool discovery -----------------------------------------------

_LLVM_BIN = "/opt/rocm/llvm/bin"


def _llvm_tool(name: str, env_var: str) -> str:
    """Best available path for an llvm tool (env override -> ROCm -> PATH)."""
    for cand in (os.environ.get(env_var), f"{_LLVM_BIN}/{name}", name):
        if cand and shutil.which(cand):
            return cand
    return name  # let the downstream call raise a clear error


def _bundler() -> str:
    return shutil.which("clang-offload-bundler") or f"{_LLVM_BIN}/clang-offload-bundler"


def _ensure_code_object(blob: bytes, arch: str, out: Path) -> Path:
    """Return a bare AMDGPU ELF path for ``blob``.

    ``compile_kernel`` (comgr) already yields a bare ELF; ``hipcc --genco`` and
    saved ``.hsaco`` files may be a ``__CLANG_OFFLOAD_BUNDLE__`` that must be
    unbundled first.
    """
    if not blob.startswith(b"__CLANG_OFFLOAD_BUNDLE__"):
        out.write_bytes(blob)
        return out
    bpath = out.with_suffix(".bundle")
    bpath.write_bytes(blob)
    subprocess.run(
        [
            _bundler(),
            "--type=o",
            "--unbundle",
            f"--input={bpath}",
            f"--output={out}",
            f"--targets=hipv4-amdgcn-amd-amdhsa--{arch}",
        ],
        check=True,
        capture_output=True,
    )
    return out


# ---- occupancy math ----------------------------------------------------


def estimate_occupancy(
    *, vgpr: int, lds_bytes: int, block_size: int, caps: ArchCaps
) -> dict:
    """Coarse blocks/CU from VGPR + LDS on a unified-VGPR arch."""
    waves_per_block = max(block_size // 64, 1)
    v = max(vgpr, 1)
    vgpr_alloc = (
        math.ceil(v / caps.vgpr_alloc_granularity) * caps.vgpr_alloc_granularity
    )
    waves_per_simd = caps.vgpr_per_simd // vgpr_alloc
    blocks_by_vgpr = (waves_per_simd * caps.simd_per_cu) // waves_per_block
    blocks_by_lds = caps.lds_per_cu // (lds_bytes or 1)
    occupancy = min(blocks_by_vgpr, blocks_by_lds)
    limiter = "LDS" if blocks_by_lds <= blocks_by_vgpr else "VGPR"
    return {
        "waves_per_block": waves_per_block,
        "blocks_by_vgpr": blocks_by_vgpr,
        "blocks_by_lds": blocks_by_lds,
        "occupancy": occupancy,
        "limiter": limiter,
    }


# ---- report one code object --------------------------------------------


def report_code_object(
    label: str, co_path: Path, *, block_size: int, caps: ArchCaps
) -> dict:
    """Analyze one bare code object and print the resource/ISA/occupancy lines."""
    a = analyze_hsaco(
        co_path,
        objdump=_llvm_tool("llvm-objdump", "LLVM_OBJDUMP"),
        readelf=_llvm_tool("llvm-readelf", "LLVM_READELF"),
    )
    r, isa = a.resources, a.isa
    vspill = r.raw.get("amdhsa_vgpr_spill_count", r.raw.get("vgpr_spill_count"))
    vgpr = r.vgpr_count or 0
    lds = r.lds_bytes or 0
    occ = estimate_occupancy(vgpr=vgpr, lds_bytes=lds, block_size=block_size, caps=caps)
    ratio = (isa.ds_read / isa.mfma) if isa.mfma else 0.0

    print(
        f"[probe:{label}] arch={caps.name} block_size={block_size} "
        f"waves/block={occ['waves_per_block']}"
    )
    print(
        f"[probe:{label}] vgpr={vgpr} sgpr={r.sgpr_count} spill={vspill} "
        f"lds={lds}B scratch={r.scratch_bytes}B"
    )
    print(
        f"[probe:{label}] mfma={isa.mfma} ds_read={isa.ds_read} "
        f"ds_write={isa.ds_write} buf_load_lds={isa.buffer_load_lds} "
        f"ds_read/mfma={ratio:.3f}"
    )
    print(
        f"[probe:{label}] s_waitcnt={isa.s_waitcnt} s_barrier={isa.s_barrier} "
        f"sched_barrier={isa.sched_barrier} valu={isa.valu} salu={isa.salu}"
    )
    print(
        f"[probe:{label}] blocks/CU: by_vgpr={occ['blocks_by_vgpr']} "
        f"by_lds={occ['blocks_by_lds']} -> occupancy={occ['occupancy']} "
        f"block(s)/CU (limiter={occ['limiter']})"
    )
    return {"label": label, **r.as_dict(), **isa.as_dict(), **occ}


# ---- programmatic entry: probe KernelDefs ------------------------------


def probe_regs(
    entries: Iterable[tuple],
    *,
    arch: str = "gfx950",
    use_hipcc: bool = False,
) -> list[dict]:
    """Compile and probe each ``(label, KernelDef, block_size)`` entry.

    ``block_size`` (threads/block) drives only the occupancy estimate; pass the
    launch block size (``BS``) your builder reports. This function never imports
    a kernel family -- the caller supplies the already-built ``KernelDef``.
    """
    caps = ARCH_BY_NAME[arch]
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as d:
        for i, (label, kdef, block_size) in enumerate(entries):
            try:
                art = (
                    compile_kernel_via_hipcc(kdef, arch=arch)
                    if use_hipcc
                    else compile_kernel(kdef, arch=arch)
                )
            except Exception as e:  # noqa: BLE001
                print(f"[probe:{label}] BUILD-FAIL: {type(e).__name__}: {e}")
                continue
            co = _ensure_code_object(art.hsaco, arch, Path(d) / f"k{i}.co")
            rows.append(report_code_object(label, co, block_size=block_size, caps=caps))
    return rows


# ---- input adapters ----------------------------------------------------


def _probe_existing(path: Path, *, arch: str, block_size: int) -> int:
    caps = ARCH_BY_NAME[arch]
    with tempfile.TemporaryDirectory() as d:
        co = _ensure_code_object(path.read_bytes(), arch, Path(d) / "k.co")
        report_code_object(path.stem, co, block_size=block_size, caps=caps)
    return 0


def _resolve_builder(spec: str):
    """Import ``module:function`` at runtime (kept out of this file's imports)."""
    if ":" not in spec:
        raise SystemExit(f"--builder must be 'module.path:function', got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    import importlib

    return getattr(importlib.import_module(mod_name), fn_name)


def _probe_via_builder(
    spec: str, kwargs_json: str, *, arch: str, block_size: int, use_hipcc: bool
) -> int:
    build_fn = _resolve_builder(spec)
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    result = build_fn(**kwargs)
    # Builders commonly return a KernelDef, or (KernelDef, BS, tm, tn, ...).
    if isinstance(result, tuple):
        kdef = result[0]
        if block_size is None and len(result) > 1 and isinstance(result[1], int):
            block_size = result[1]
    else:
        kdef = result
    bs = block_size or 256
    label = getattr(kdef, "name", spec.split(":")[-1])
    probe_regs([(label, kdef, bs)], arch=arch, use_hipcc=use_hipcc)
    return 0


def _probe_demo(*, arch: str, block_size: int) -> int:
    """Self-contained smoke kernel: no instance/kernel import, no GPU launch."""
    from rocke.core.ir import I32, IRBuilder, PtrType

    b = IRBuilder("probe_regs_demo")
    out = b.param("out", PtrType(I32, "global"), noalias=True)
    b.store(b.const_i32(42), out, b.const_i32(0))
    b.ret()
    # backend="python" keeps the demo deterministic and independent of a built
    # rocke_engine C++ extension.
    caps = ARCH_BY_NAME[arch]
    with tempfile.TemporaryDirectory() as d:
        art = compile_kernel(b.kernel, arch=arch, backend="python")
        co = _ensure_code_object(art.hsaco, arch, Path(d) / "demo.co")
        report_code_object("demo", co, block_size=block_size, caps=caps)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--co", type=Path, help="probe an existing bare code object")
    src.add_argument(
        "--hsaco", type=Path, help="probe an existing HSACO (hipcc bundle ok)"
    )
    src.add_argument(
        "--builder",
        help="'module.path:function' resolved at runtime to build a KernelDef",
    )
    p.add_argument(
        "--builder-kwargs", default="", help="JSON kwargs passed to --builder"
    )
    p.add_argument(
        "--arch",
        choices=list(ARCH_BY_NAME),
        default="gfx950",
        help="arch caps for the occupancy estimate (default gfx950)",
    )
    p.add_argument(
        "--bs",
        type=int,
        default=None,
        help="launch block size (threads/block) for the occupancy estimate",
    )
    p.add_argument(
        "--hipcc",
        action="store_true",
        help="compile through hipcc --genco instead of the comgr LLVM path",
    )
    args = p.parse_args(argv)

    if args.co:
        return _probe_existing(args.co, arch=args.arch, block_size=args.bs or 256)
    if args.hsaco:
        return _probe_existing(args.hsaco, arch=args.arch, block_size=args.bs or 256)
    if args.builder:
        return _probe_via_builder(
            args.builder,
            args.builder_kwargs,
            arch=args.arch,
            block_size=args.bs,
            use_hipcc=args.hipcc,
        )
    return _probe_demo(arch=args.arch, block_size=args.bs or 256)


if __name__ == "__main__":
    raise SystemExit(main())
