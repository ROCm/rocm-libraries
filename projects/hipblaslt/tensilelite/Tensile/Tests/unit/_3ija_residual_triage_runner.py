"""rocm-libraries-3ija — triage compare_graphs / wait-coverage residuals on
real Build #2 across the gfx950 CMS test surface.

Investigation script (NOT a pytest test). Iterates every registered
gfx950 CMS schedule (16-bit + TF32 — 8-bit out of scope per the bead's
5gd rescope reference), constructs a kernel config that matches the
schedule's TileConfig, runs Build #1 (CMS) + Build #2 (non-CMS reference
via `build_non_cms_reference`), and tabulates the residuals from
`compare_graphs(ref=Build#2, subj=Build#1)` and
`validate_edge_wait_coverage(Build#2)`.

Usage (from tensilelite/):
    pytest Tensile/Tests/unit/_3ija_residual_triage_runner.py -s \\
        --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py

The "test" exits cleanly regardless of residuals — it dumps a per-fixture
table to stdout for the investigator. It does NOT assert correctness.
"""
import os
import sys
import traceback
from collections import Counter

import pytest


# Make sibling test utils importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _cms_kernel_info_to_config(info):
    """Translate a CMSKernelInfo into a kernel_config dict consumable by
    `cms_test_utils._make_solution`. Matches the structure of
    `CANONICAL_TF32_4X4_TN_CONFIG`.
    """
    if info.dtype == "TF32":
        problem_type = {
            'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
            'F32XdlMathOp': 'X',
            'TransposeA': info.TransposeA, 'TransposeB': info.TransposeB,
            'UseBeta': True, 'Batched': True,
        }
    elif info.dtype == "16bit":
        problem_type = {
            'OperationType': 'GEMM', 'DataType': 'H', 'DestDataType': 'H',
            'ComputeDataType': 'S',
            'HighPrecisionAccumulate': True,
            'TransposeA': info.TransposeA, 'TransposeB': info.TransposeB,
            'UseBeta': True, 'Batched': True,
        }
    else:
        return None

    mi = list(info.MatrixInstruction)
    miwg = list(info.MIWaveGroup)
    # Build the 9-element MI parameter shape consumed by
    # matrixInstructionToMIParameters: [m, n, k, b, mvw, wt0, wt1, wg0, wg1]
    # where wt0/wt1 are the wave-tile sizes derived from MacroTile/(mi*miwg).
    wt0 = info.MacroTile0 // (mi[0] * miwg[0])
    wt1 = info.MacroTile1 // (mi[1] * miwg[1])
    matrix_instruction_9 = mi + [1, wt0, wt1, miwg[0], miwg[1]]

    config = {
        'ProblemType': problem_type,
        'MatrixInstruction': matrix_instruction_9,
        'DepthU': info.DepthU,
        'PrefetchGlobalRead': info.PrefetchGlobalRead,
        'PrefetchLocalRead': info.PrefetchLocalRead,
        'DirectToLds': info.DirectToLds,
        'TransposeLDS': info.TransposeLDS,
        'LocalReadVectorWidth': info.LocalReadVectorWidth,
        'GlobalReadVectorWidthA': info.GlobalReadVectorWidthA,
        'GlobalReadVectorWidthB': info.GlobalReadVectorWidthB,
        'UseCustomMainLoopSchedule': 1, 'ExpandPointerSwap': 0,
        'SourceSwap': 1, 'StreamK': 0,
    }
    return config


def _classify_failure(f):
    ftype = type(f).__name__
    prod = getattr(f, "producer", None)
    cons = getattr(f, "consumer", None)
    prod_cat = getattr(prod, "category", "") or ""
    cons_cat = getattr(cons, "category", "") or ""
    prod_body = getattr(prod, "body_label", None)
    cons_body = getattr(cons, "body_label", None)
    return (ftype, prod_cat, cons_cat, prod_body, cons_body)


def _exercise_one(info, asm, isaInfoMap):
    """Build #1 + Build #2 + compare_graphs + validate_edge_wait_coverage
    for one CMS fixture. Returns dict with keys:
      - cg_failures: List[ClassificationKey]
      - wc_failures: List[ClassificationKey]
      - error: Optional[str]
    """
    from Tensile.Components.CustomSchedule.approach_a import (
        build_non_cms_reference,
    )
    from Tensile.Components.CMSValidator import (
        build_dataflow_graph, compare_graphs, validate_edge_wait_coverage,
    )
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    from cms_test_utils import _make_solution

    config = _cms_kernel_info_to_config(info)
    if config is None:
        return {"error": f"unsupported dtype {info.dtype}"}

    try:
        cms_solution = _make_solution(config, asm, isaInfoMap)
    except Exception as e:
        return {"error": f"_make_solution failed: {type(e).__name__}: {e}"}

    cms_writer = KernelWriterAssembly(asm, DebugConfig())
    try:
        cms_writer._getKernelSource(cms_solution)
    except Exception:
        # Pre-existing shadow-vs-CMS divergence may raise the assert; the
        # CMS-side capture is populated before the assert fires.
        pass
    cms_cap = cms_writer._last_cms_capture
    if cms_cap is None:
        return {"error": "_last_cms_capture is None — CMS build did not "
                         "populate kernelBody post-loop assembly"}

    try:
        ref_cap = build_non_cms_reference(config, asm, isaInfoMap)
    except Exception as e:
        return {"error": f"build_non_cms_reference failed: "
                         f"{type(e).__name__}: {e}\n"
                         f"{traceback.format_exc(limit=4)}"}

    try:
        ref_graph = build_dataflow_graph(ref_cap)
        subj_graph = build_dataflow_graph(cms_cap)
        cg_failures = compare_graphs(ref_graph, subj_graph)
    except Exception as e:
        return {"error": f"compare_graphs failed: {type(e).__name__}: {e}"}

    try:
        wc_failures = validate_edge_wait_coverage(ref_graph)
    except Exception as e:
        wc_failures = [(f"WC-EXCEPTION:{type(e).__name__}", str(e), "", "", "")]
    else:
        wc_failures = [_classify_failure(f) for f in wc_failures]

    return {
        "cg_failures": [_classify_failure(f) for f in cg_failures],
        "wc_failures": wc_failures,
        "ref_cap": ref_cap,
        "cms_cap": cms_cap,
        "error": None,
    }


def test_3ija_residual_triage(isa_infrastructure):
    """Investigation runner — see module docstring."""
    # Importing the gfx950 schedule package triggers RegisterSchedule
    # decorators which populate _SCHEDULE_METADATA.
    from Tensile.Components.CustomSchedule.gfx950 import __init__ as _init  # noqa
    from Tensile.Components.CustomSchedule.dispatch import _SCHEDULE_METADATA

    _isa, isaInfoMap, asm = isa_infrastructure

    # Filter to TF32 and 16bit (8bit out of scope per 5gd rescope).
    fixtures = [info for info in _SCHEDULE_METADATA
                if info.dtype in ("TF32", "16bit")]
    # Deduplicate fixtures — RegisterSchedule emits one entry per detected
    # layout, so a TN-only schedule produces 1 entry; a TN+NT schedule
    # produces 2. We want to exercise EACH (name, layout) combo exactly
    # once (each may produce different residuals).
    seen = set()
    unique = []
    for info in fixtures:
        key = (info.name, info.TransposeA, info.TransposeB,
               info.LDSTrInst, info.TransposeLDS)
        if key in seen:
            continue
        seen.add(key)
        unique.append(info)
    fixtures = unique
    fixtures.sort(key=lambda i: (i.dtype, i.name, i.TransposeA, i.TransposeB))

    print(f"\n=== 3ija TRIAGE: exercising {len(fixtures)} CMS fixtures ===\n")

    rows = []
    cg_shape_counter = Counter()
    wc_shape_counter = Counter()
    error_counter = Counter()
    for info in fixtures:
        layout = ("T" if info.TransposeA else "N") + (
            "T" if info.TransposeB else "N")
        tag = (f"{info.name} {layout} LDSTr={info.LDSTrInst} "
               f"TLDS={info.TransposeLDS}")
        try:
            result = _exercise_one(info, asm, isaInfoMap)
        except Exception as e:
            print(f"[{tag}] runner-exception: {type(e).__name__}: {e}")
            error_counter[type(e).__name__] += 1
            continue

        if result.get("error"):
            err_summary = result["error"].splitlines()[0]
            print(f"[{tag}] ERROR: {err_summary}")
            error_counter[err_summary[:60]] += 1
            rows.append((tag, "ERROR", result["error"][:200], ""))
            continue

        cg = result["cg_failures"]
        wc = result["wc_failures"]
        cg_summary = Counter((k[0], k[1], k[2], k[3], k[4]) for k in cg)
        wc_summary = Counter((k[0], k[1], k[2], k[3], k[4]) for k in wc)
        for k, n in cg_summary.items():
            cg_shape_counter[k] += n
        for k, n in wc_summary.items():
            wc_shape_counter[k] += n

        cg_str = "; ".join(f"{n}x {k[0]}({k[1]}->{k[2]} {k[3]}->{k[4]})"
                           for k, n in cg_summary.most_common())
        wc_str = "; ".join(f"{n}x {k[0]}({k[1]}->{k[2]} {k[3]}->{k[4]})"
                           for k, n in wc_summary.most_common())
        print(f"[{tag}] cg={len(cg)} wc={len(wc)} | CG: {cg_str or '-'} "
              f"| WC: {wc_str or '-'}")
        rows.append((tag, len(cg), cg_str, len(wc), wc_str))

    print("\n=== AGGREGATE compare_graphs RESIDUAL SHAPES ===")
    for shape, count in cg_shape_counter.most_common():
        ftype, pcat, ccat, pbody, cbody = shape
        print(f"  {count:5d}x {ftype} {pcat}({pbody}) -> "
              f"{ccat}({cbody})")
    print("\n=== AGGREGATE wait-coverage RESIDUAL SHAPES ===")
    for shape, count in wc_shape_counter.most_common():
        ftype, pcat, ccat, pbody, cbody = shape
        print(f"  {count:5d}x {ftype} {pcat}({pbody}) -> "
              f"{ccat}({cbody})")
    print("\n=== ERROR SHAPES ===")
    for err, count in error_counter.most_common():
        print(f"  {count:5d}x {err}")
    print(f"\n=== TOTAL: {len(fixtures)} fixtures, "
          f"{sum(cg_shape_counter.values())} CG residuals, "
          f"{sum(wc_shape_counter.values())} WC residuals ===")
