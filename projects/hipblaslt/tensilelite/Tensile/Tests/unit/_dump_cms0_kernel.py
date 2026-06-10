"""Dump a real cms=0 (UseCustomMainLoopSchedule=0) kernel for the BPG#11 TF32
4x4 TN canonical config, and slice out the ML / NGL / NLL regions.

This is NOT the shadow capture — it is a genuine non-CMS `_getKernelSource`
emit, so we can read what the real default-codegen kernel does in each loop
body for the rotating pack buffer ValuA/B_X0_I0+12..15.

Run:
  cd <worktree>/projects/hipblaslt/tensilelite
  pip install -e ./rocisa
  python Tensile/Tests/unit/_dump_cms0_kernel.py

Artifacts in hxcx_artifacts/:
  cms0_kernel.s        — full default-codegen assembly
  cms0_bodies.txt      — ML / NGL / NLL regions with the X0_I0+12..15 + T0_I0
                         ds_read / pack / mfma lines pulled out, line-numbered
"""

import os
import sys
import pathlib


def _assert_tensile_tree_matches_test_tree():
    import Tensile.KernelWriter as _kw
    test_tree = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    kw_tree = os.path.abspath(os.path.join(os.path.dirname(_kw.__file__), ".."))
    if test_tree != kw_tree:
        raise RuntimeError(
            f"Tensile loaded from a different tree. test_tree={test_tree!r}, "
            f"kw_tree={kw_tree!r}. cd {test_tree} first."
        )


_assert_tensile_tree_matches_test_tree()


CANONICAL_KERNEL_CONFIG = {
    'ProblemType': {
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    'MatrixInstruction': [16, 16, 32, 1, 1, 4, 4, 2, 2],
    'DepthU': 32, 'PrefetchGlobalRead': 2, 'PrefetchLocalRead': 1,
    'DirectToLds': 1, 'TransposeLDS': 1, 'LocalReadVectorWidth': 4,
    'GlobalReadVectorWidthA': 4, 'GlobalReadVectorWidthB': 4,
    'UseCustomMainLoopSchedule': 0,   # <-- the whole point: real cms=0 build
    'ExpandPointerSwap': 0,
    'SourceSwap': 1, 'StreamK': 0,
    'UseMFMAF32XEmulation': True, 'UsePLRPack': True,
}


def _isa():
    import shutil
    from Tensile.Common import IsaVersion
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Component import Assembler
    compiler = shutil.which('amdclang++') or shutil.which('clang++')
    assembler_bin = shutil.which('amdclang') or shutil.which('clang')
    assert compiler and assembler_bin
    isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0)], compiler)
    return None, isaInfoMap, Assembler(assembler_bin, 'V5')


def main():
    out_dir = pathlib.Path("hxcx_artifacts").resolve()
    out_dir.mkdir(exist_ok=True)
    _, isaInfoMap, asm = _isa()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig

    cfg = dict(CANONICAL_KERNEL_CONFIG)
    solution = _make_solution(cfg, asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    print("Building cms=0 kernel via _getKernelSource ...")
    src = writer._getKernelSource(solution)

    ks = out_dir / "cms0_kernel.s"
    ks.write_text(src)
    print(f"  -> {ks} ({len(src)} bytes, {src.count(chr(10))} lines)")

    lines = src.splitlines()

    # Identify body regions by label. The default kernel uses recognizable
    # labels: the main unroll loop, the no-load-loop (NGL / OptNLL), and the
    # tail loop. We slice on those.
    markers = []
    for i, ln in enumerate(lines, start=1):
        s = ln.strip()
        if (s.endswith(":") and (
                "LoopBeginL" in s or "LoopEndL" in s or "openLoop" in s or
                "toPGR" in s or "NoLoadLoop" in s or "NLL" in s.replace("OptNLL", "OptNLL") or
                "OptNLL" in s or "Summation" in s or "TailLoop" in s or
                "PrefetchGlobalRead" in s or "Loop" in s)):
            markers.append((i, s))

    out = []
    out.append("# cms=0 (UseCustomMainLoopSchedule=0) kernel body markers")
    out.append("# (label line number : label)")
    out.append("#" + "=" * 70)
    for i, s in markers:
        out.append(f"{i:6d} : {s}")

    out.append("")
    out.append("#" + "=" * 70)
    out.append("# ALL ds_read / pack-cvt / pack-mfma / consume-mfma touching")
    out.append("# ValuA_X0_I0+12..15 and ValuA_T0_I0 (the rotating-buffer halves)")
    out.append("# across the whole kernel, line-numbered. Compare ML vs NGL vs NLL.")
    out.append("#" + "=" * 70)
    pat_terms = [
        "ValuA_X0_I0+12", "ValuA_X0_I0+13", "ValuA_X0_I0+14", "ValuA_X0_I0+15",
    ]
    for i, ln in enumerate(lines, start=1):
        if any(t in ln for t in pat_terms):
            kind = ""
            if "ds_read" in ln:
                kind = "DS_READ "
            elif "v_cvt_pk" in ln:
                kind = "PACK_CVT"
            elif "v_mfma_f32_4x4x4" in ln:
                kind = "PACK_MFMA"
            elif "v_mfma" in ln:
                kind = "CONS_MFMA"
            else:
                kind = "OTHER   "
            out.append(f"{i:6d} | {kind} | {ln.strip()}")

    bodies = out_dir / "cms0_bodies.txt"
    bodies.write_text("\n".join(out) + "\n")
    print(f"  -> {bodies} ({len(out)} lines)")
    print("\nDone. Inspect hxcx_artifacts/cms0_kernel.s and cms0_bodies.txt")


if __name__ == "__main__":
    main()
