# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 (CDNA3 / MI300X) prefill-2D unified-attention benchmark.

The gfx942 analogue of ``examples/gfx950/attention/benchmark_prefill2d_live.py``,
torch-reference-free: it sweeps the shipped gfx942 tiled-2D kernel variants over
the canonical shapes in ``shapes.json`` and reports per-shape latency + TFLOPS.

Variants (D128 fp16 only -- the flash 32x32x8 atom is fp16-only):

  * ``wide4`` -- the SHIPPED default (WG=256 / num_warps=4 flash regime).
    This is the PROVIDER's analytic default (``compile_service.py``
    ``_flash_wide=4`` + ``SdpaCandidateSelector.analyticTarget``); the DSL
    spec's own default is the L4 geometry, so this harness builds the wide4
    spec EXPLICITLY (``num_warps=4, block_m_per_warp=32,
    use_mfma_32x32x8=True, use_transposed_qk_32x32=True,
    use_k_single_buffer=False``) to reproduce the shipped peak. Case study
    Batch 5: +19.7% over L4 (153.6 -> 183.8 TF, 63% of PyTorch flash) on the
    GQA S2048 shape; ~191 TF measured here (see ``expected_perf.csv``).
  * ``L4``    -- the WG=64 fallback (transposed-x8 + K single-buffer),
    the DSL-side default / production kill-switch
    (``HIPDNN_GFX942_FLASH_WIDE=0``). ~163 TF measured here.

Non-D128-fp16 shapes (D64, bf16) run their single narrow-path variant
(``narrow``); D64 fp16 GQA S2048 is ~149 TF measured here.

``--check`` compares the measured TFLOPS for each (shape, config) against the
shipped baselines in ``expected_perf.csv`` and FLAGS regressions more than
``--regress-pct`` (default 10%) below baseline (exit code 1).

Run (needs torch + a gfx942 GPU):

    PYTHONPATH=python .venv/bin/python \\
        python/ck_dsl/examples/gfx942/attention/benchmark_prefill2d.py \\
        --scenario perf --variants wide4 L4

    PYTHONPATH=python .venv/bin/python \\
        python/ck_dsl/examples/gfx942/attention/benchmark_prefill2d.py \\
        --scenario perf --check
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[5]  # composablekernel/
sys.path.insert(0, str(ROOT / "python"))

_HERE = Path(__file__).resolve().parent
EXPECTED_PERF_CSV = _HERE / "expected_perf.csv"

# Reuse the shape loader / input maker / launcher from the parity harness so the
# two examples exercise byte-identical build + launch plumbing.
sys.path.insert(0, str(_HERE))
from parity_unified_attention import (  # noqa: E402
    Shape,
    _build_kernel,
    _is_flash_wide_eligible,
    _run_ck_dsl,
    attention_tflops,
    load_shapes,
    select_shapes,
)

# All variants this benchmark knows how to force. ``wide4`` / ``L4`` are the
# D128-fp16 flash-regime configs (driven via HIPDNN_GFX942_FLASH_WIDE);
# ``cfv`` is the legacy gather-fill conflict-free-V diagnostic. The ``cfvst_*``
# names are an explicit composition matrix for the store-path conflict-free-V
# stack; each adds one structural lever on top of wide4. ``ksring`` adds the
# sliced K-ring scaffold on top of the full cfvst stack. Other shapes only have
# the single ``narrow`` variant.
ALL_VARIANTS = (
    "wide4",
    "L4",
    "cfv",
    "cfvst_base",   # store-path cfv, simple V_lds, immediate store
    "cfvst_ckvl",   # + CK-style packed V LDS layout
    "cfvst_split",  # + split V load/store schedule, simple V_lds
    "cfvst",        # + split schedule + CK-style V LDS layout
    "cfvst_wpe2",
    "cfvst_wpe3",
    "cfvst_iglp",
    "cfvst_cpall",
    "cfvst_cpglobal",
    "cfvst_cpnt",
    "cfvst_qdir",
    "cfvst_qdir_cpglobal",
    "cfvst_w8",
    "cfvst_gldlds",
    "cfvst_gldlds_cpglobal",
    "cfvst_qgrid",
    "ksring",       # + sliced K ring scaffold
    "ksldsseq",     # + CK Tile LdsSeq sliced K variant
    "auto",
    "2d",
    "3d",
    "narrow",
)


_EXTRA_ENV_GUARD = (
    "HIPDNN_GFX942_FLASH_MLIM",
    "HIPDNN_GFX942_3D_HOIST",
    "HIPDNN_GFX942_CFV_SCALAR_READ",
    "HIPDNN_GFX942_CFV_STORE_SCALAR_LOAD",
    "HIPDNN_GFX942_CFV_STORE_SCATTER",
    "HIPDNN_GFX942_CFV_STORE_PREZERO",
    "HIPDNN_GFX942_CFV_STORE_SEPOFF",
    "HIPDNN_GFX942_SWIZZLE_VLDS",
)


def _snapshot_env(names):
    return {name: os.environ.get(name) for name in names}


def _restore_env(snapshot):
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _variant_env(variant: str) -> Optional[str]:
    """The ``HIPDNN_GFX942_FLASH_WIDE`` value that selects ``variant`` (or None)."""
    if variant == "wide4":
        return "4"
    if variant == "L4":
        return "0"
    if variant in (
        "cfv",
        "cfvst_base",
        "cfvst_ckvl",
        "cfvst_split",
        "cfvst",
        "cfvst_wpe2",
        "cfvst_wpe3",
        "cfvst_iglp",
        "cfvst_cpall",
        "cfvst_cpglobal",
        "cfvst_cpnt",
        "cfvst_qdir",
        "cfvst_qdir_cpglobal",
        "cfvst_w8",
        "cfvst_gldlds",
        "cfvst_gldlds_cpglobal",
        "cfvst_qgrid",
        "ksring",
        "ksldsseq",
    ):
        return "4"
    return None  # narrow: not flash-regime, env is inert


def _variant_applies(variant: str, s: Shape) -> bool:
    if variant in ("auto", "2d", "3d"):
        return True
    if variant in (
        "wide4",
        "L4",
        "cfv",
        "cfvst_base",
        "cfvst_ckvl",
        "cfvst_split",
        "cfvst",
        "cfvst_wpe2",
        "cfvst_wpe3",
        "cfvst_iglp",
        "cfvst_cpall",
        "cfvst_cpglobal",
        "cfvst_cpnt",
        "cfvst_qdir",
        "cfvst_qdir_cpglobal",
        "cfvst_w8",
        "cfvst_gldlds",
        "cfvst_gldlds_cpglobal",
        "cfvst_qgrid",
        "ksring",
        "ksldsseq",
    ):
        return _is_flash_wide_eligible(s)
    # narrow applies to everything that is NOT the D128-fp16 flash path.
    return not _is_flash_wide_eligible(s)


def _run_common_path(s: Shape, path: str, *, warmup: int, attempts: int):
    """Run production dispatcher path (auto/2d/3d) for any shape."""
    import torch
    from ck_dsl.instances import UnifiedAttentionProblem, run_unified_attention_torch
    from ck_dsl.runtime import synchronize_and_release, time_launches
    from parity_unified_attention import make_inputs

    data = make_inputs(s)
    out = torch.empty_like(data["query"])
    backend = "tiled" if path == "2d" else path
    problem = UnifiedAttentionProblem(
        total_q=data["query"].shape[0],
        num_seqs=s.batch,
        num_query_heads=s.heads,
        num_kv_heads=s.kv_heads,
        head_size=s.head_size,
        block_size=64,
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        dtype=s.dtype,
        sliding_window=0,
        softcap=0.0,
        use_sinks=False,
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        num_sms=120,
    )
    hip_stream = int(torch.cuda.current_stream().cuda_stream)

    def call_once():
        run_unified_attention_torch(
            problem=problem,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=0.0,
            backend=backend,
            stream=hip_stream,
        )

    ms = time_launches(call_once, warmup=warmup, iters=attempts, stream=hip_stream)
    synchronize_and_release(hip_stream)
    return attention_tflops(s, ms), ms * 1e3, path


def _run_variant(s: Shape, variant: str, *, warmup: int, attempts: int):
    """Build + time one (shape, variant). Returns ``(tflops, median_us, config)``."""
    import torch

    if variant in ("auto", "2d", "3d"):
        guarded = _snapshot_env(_EXTRA_ENV_GUARD)
        try:
            return _run_common_path(s, variant, warmup=warmup, attempts=attempts)
        finally:
            _restore_env(guarded)

    guarded = _snapshot_env(_EXTRA_ENV_GUARD)
    env = _variant_env(variant)
    prev = os.environ.get("HIPDNN_GFX942_FLASH_WIDE")
    prev_cfv = os.environ.get("HIPDNN_GFX942_CFV")
    prev_cfvst = os.environ.get("HIPDNN_GFX942_CFV_STORE")
    prev_cfvst_split = os.environ.get("HIPDNN_GFX942_CFV_STORE_SPLIT")
    prev_cfv_ck_vlds = os.environ.get("HIPDNN_GFX942_CFV_CK_VLDS")
    prev_ksring = os.environ.get("HIPDNN_GFX942_K_SLICED_RING")
    prev_ksldsseq = os.environ.get("HIPDNN_GFX942_K_LDSSEQ")
    prev_wpe = os.environ.get("HIPDNN_GFX942_WAVES_PER_EU")
    prev_iglp = os.environ.get("HIPDNN_GFX942_IGLP")
    prev_kvcp = os.environ.get("HIPDNN_GFX942_KV_CACHE_POLICY")
    prev_qdir = os.environ.get("HIPDNN_GFX942_Q_DIRECT")
    prev_gldlds = os.environ.get("HIPDNN_GFX942_GLOBAL_LOAD_LDS_K")
    prev_nw = os.environ.get("HIPDNN_GFX942_NUM_WARPS")
    prev_qgrid = os.environ.get("HIPDNN_GFX942_Q_MAJOR_GRID")
    if env is not None:
        os.environ["HIPDNN_GFX942_FLASH_WIDE"] = env
    if variant == "cfv":
        os.environ["HIPDNN_GFX942_CFV"] = "1"
    else:
        os.environ.pop("HIPDNN_GFX942_CFV", None)
    if variant in (
        "cfvst_base",
        "cfvst_ckvl",
        "cfvst_split",
        "cfvst",
        "cfvst_wpe2",
        "cfvst_wpe3",
        "cfvst_iglp",
        "cfvst_cpall",
        "cfvst_cpglobal",
        "cfvst_cpnt",
        "cfvst_qdir",
        "cfvst_qdir_cpglobal",
        "cfvst_w8",
        "cfvst_gldlds",
        "cfvst_gldlds_cpglobal",
        "cfvst_qgrid",
        "ksring",
        "ksldsseq",
    ):
        os.environ["HIPDNN_GFX942_CFV_STORE"] = "1"
    else:
        os.environ.pop("HIPDNN_GFX942_CFV_STORE", None)
    if variant in (
        "cfvst_split",
        "cfvst",
        "cfvst_wpe2",
        "cfvst_wpe3",
        "cfvst_iglp",
        "cfvst_cpall",
        "cfvst_cpglobal",
        "cfvst_cpnt",
        "cfvst_qdir",
        "cfvst_qdir_cpglobal",
        "cfvst_w8",
        "cfvst_gldlds",
        "cfvst_gldlds_cpglobal",
        "cfvst_qgrid",
        "ksring",
        "ksldsseq",
    ):
        os.environ["HIPDNN_GFX942_CFV_STORE_SPLIT"] = "1"
    elif variant in ("cfvst_base", "cfvst_ckvl"):
        os.environ["HIPDNN_GFX942_CFV_STORE_SPLIT"] = "0"
    else:
        os.environ.pop("HIPDNN_GFX942_CFV_STORE_SPLIT", None)
    if variant in (
        "cfvst_ckvl",
        "cfvst",
        "cfvst_wpe2",
        "cfvst_wpe3",
        "cfvst_iglp",
        "cfvst_cpall",
        "cfvst_cpglobal",
        "cfvst_cpnt",
        "cfvst_qdir",
        "cfvst_qdir_cpglobal",
        "cfvst_w8",
        "cfvst_gldlds",
        "cfvst_gldlds_cpglobal",
        "cfvst_qgrid",
        "ksring",
        "ksldsseq",
    ):
        os.environ["HIPDNN_GFX942_CFV_CK_VLDS"] = "1"
    elif variant in ("cfvst_base", "cfvst_split"):
        os.environ["HIPDNN_GFX942_CFV_CK_VLDS"] = "0"
    else:
        os.environ.pop("HIPDNN_GFX942_CFV_CK_VLDS", None)
    if variant in ("ksring", "ksldsseq"):
        os.environ["HIPDNN_GFX942_K_SLICED_RING"] = "1"
    else:
        os.environ.pop("HIPDNN_GFX942_K_SLICED_RING", None)
    if variant == "ksldsseq":
        os.environ["HIPDNN_GFX942_K_LDSSEQ"] = "1"
    else:
        os.environ.pop("HIPDNN_GFX942_K_LDSSEQ", None)
    if variant == "cfvst_wpe2":
        os.environ["HIPDNN_GFX942_WAVES_PER_EU"] = "2"
    elif variant == "cfvst_wpe3":
        os.environ["HIPDNN_GFX942_WAVES_PER_EU"] = "3"
    else:
        os.environ.pop("HIPDNN_GFX942_WAVES_PER_EU", None)
    if variant == "cfvst_iglp":
        os.environ["HIPDNN_GFX942_IGLP"] = "1"
    else:
        os.environ.pop("HIPDNN_GFX942_IGLP", None)
    if variant == "cfvst_cpall":
        os.environ["HIPDNN_GFX942_KV_CACHE_POLICY"] = "all"
    elif variant in ("cfvst_cpglobal", "cfvst_qdir_cpglobal", "cfvst_gldlds_cpglobal"):
        os.environ["HIPDNN_GFX942_KV_CACHE_POLICY"] = "global"
    elif variant == "cfvst_cpnt":
        os.environ["HIPDNN_GFX942_KV_CACHE_POLICY"] = "nt"
    else:
        os.environ.pop("HIPDNN_GFX942_KV_CACHE_POLICY", None)
    if variant in ("cfvst_qdir", "cfvst_qdir_cpglobal", "cfvst_w8"):
        os.environ["HIPDNN_GFX942_Q_DIRECT"] = "1"
    else:
        os.environ.pop("HIPDNN_GFX942_Q_DIRECT", None)
    if variant in ("cfvst_gldlds", "cfvst_gldlds_cpglobal"):
        os.environ["HIPDNN_GFX942_GLOBAL_LOAD_LDS_K"] = "1"
    else:
        os.environ.pop("HIPDNN_GFX942_GLOBAL_LOAD_LDS_K", None)
    if variant == "cfvst_w8":
        os.environ["HIPDNN_GFX942_NUM_WARPS"] = "8"
    else:
        os.environ.pop("HIPDNN_GFX942_NUM_WARPS", None)
    if variant == "cfvst_qgrid":
        os.environ["HIPDNN_GFX942_Q_MAJOR_GRID"] = "1"
    else:
        os.environ.pop("HIPDNN_GFX942_Q_MAJOR_GRID", None)
    try:
        from parity_unified_attention import make_inputs

        launcher, spec, config = _build_kernel(s)
        data = make_inputs(s)
        out, ms = _run_ck_dsl(s, data, launcher, spec, warmup=warmup, attempts=attempts)
        del out
        torch.cuda.synchronize()
    finally:
        _restore_env(guarded)
        if env is not None:
            if prev is None:
                os.environ.pop("HIPDNN_GFX942_FLASH_WIDE", None)
            else:
                os.environ["HIPDNN_GFX942_FLASH_WIDE"] = prev
        if prev_cfv is None:
            os.environ.pop("HIPDNN_GFX942_CFV", None)
        else:
            os.environ["HIPDNN_GFX942_CFV"] = prev_cfv
        if prev_cfvst is None:
            os.environ.pop("HIPDNN_GFX942_CFV_STORE", None)
        else:
            os.environ["HIPDNN_GFX942_CFV_STORE"] = prev_cfvst
        if prev_cfvst_split is None:
            os.environ.pop("HIPDNN_GFX942_CFV_STORE_SPLIT", None)
        else:
            os.environ["HIPDNN_GFX942_CFV_STORE_SPLIT"] = prev_cfvst_split
        if prev_cfv_ck_vlds is None:
            os.environ.pop("HIPDNN_GFX942_CFV_CK_VLDS", None)
        else:
            os.environ["HIPDNN_GFX942_CFV_CK_VLDS"] = prev_cfv_ck_vlds
        if prev_ksring is None:
            os.environ.pop("HIPDNN_GFX942_K_SLICED_RING", None)
        else:
            os.environ["HIPDNN_GFX942_K_SLICED_RING"] = prev_ksring
        if prev_ksldsseq is None:
            os.environ.pop("HIPDNN_GFX942_K_LDSSEQ", None)
        else:
            os.environ["HIPDNN_GFX942_K_LDSSEQ"] = prev_ksldsseq
        if prev_wpe is None:
            os.environ.pop("HIPDNN_GFX942_WAVES_PER_EU", None)
        else:
            os.environ["HIPDNN_GFX942_WAVES_PER_EU"] = prev_wpe
        if prev_iglp is None:
            os.environ.pop("HIPDNN_GFX942_IGLP", None)
        else:
            os.environ["HIPDNN_GFX942_IGLP"] = prev_iglp
        if prev_kvcp is None:
            os.environ.pop("HIPDNN_GFX942_KV_CACHE_POLICY", None)
        else:
            os.environ["HIPDNN_GFX942_KV_CACHE_POLICY"] = prev_kvcp
        if prev_qdir is None:
            os.environ.pop("HIPDNN_GFX942_Q_DIRECT", None)
        else:
            os.environ["HIPDNN_GFX942_Q_DIRECT"] = prev_qdir
        if prev_gldlds is None:
            os.environ.pop("HIPDNN_GFX942_GLOBAL_LOAD_LDS_K", None)
        else:
            os.environ["HIPDNN_GFX942_GLOBAL_LOAD_LDS_K"] = prev_gldlds
        if prev_nw is None:
            os.environ.pop("HIPDNN_GFX942_NUM_WARPS", None)
        else:
            os.environ["HIPDNN_GFX942_NUM_WARPS"] = prev_nw
        if prev_qgrid is None:
            os.environ.pop("HIPDNN_GFX942_Q_MAJOR_GRID", None)
        else:
            os.environ["HIPDNN_GFX942_Q_MAJOR_GRID"] = prev_qgrid
    return attention_tflops(s, ms), ms * 1e3, config


def _load_expected(path: Path = EXPECTED_PERF_CSV) -> Dict[Tuple[str, str], dict]:
    """Load shipped baselines keyed by (shape, config)."""
    if not path.exists():
        return {}
    table: Dict[Tuple[str, str], dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            table[(row["shape"], row["config"])] = row
    return table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="gfx942 prefill-2D attention benchmark"
    )
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["wide4", "L4", "narrow"],
        help=f"subset of {ALL_VARIANTS}",
    )
    parser.add_argument("--attempts", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare measured TFLOPS to expected_perf.csv and flag regressions",
    )
    parser.add_argument("--regress-pct", type=float, default=10.0)
    parser.add_argument(
        "--write-expected",
        type=Path,
        default=None,
        help="write the measured rows to a CSV (used to seed expected_perf.csv)",
    )
    parser.add_argument("--rocm-ver", default=os.environ.get("ROCM_VERSION", "unknown"))
    parser.add_argument("--date", default="")
    args = parser.parse_args()

    for v in args.variants:
        if v not in ALL_VARIANTS:
            print(f"unknown variant {v!r} (expected {ALL_VARIANTS})", file=sys.stderr)
            return 2

    import torch

    if not torch.cuda.is_available():
        print("CUDA/HIP device unavailable; exiting", file=sys.stderr)
        return 1
    dev_name = torch.cuda.get_device_name(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    print(f"device: {dev_name}  arch: {arch}")

    selectors = args.scenario if args.scenario is not None else ["perf"]
    shapes = select_shapes(load_shapes(), selectors)
    if not shapes:
        print(f"no shapes matched {selectors!r}", file=sys.stderr)
        return 2

    expected = _load_expected() if args.check else {}
    rows: List[dict] = []
    regressions: List[str] = []

    for s in shapes:
        for variant in args.variants:
            if not _variant_applies(variant, s):
                continue
            print(f"\n=== {s.name}  variant={variant} ===")
            try:
                tflops, median_us, config = _run_variant(
                    s, variant, warmup=args.warmup, attempts=args.attempts
                )
            except NotImplementedError as e:
                print(f"  SKIP (unsupported on gfx942): {e}")
                continue
            print(f"  config={config:11s}  {median_us:9.2f} us  {tflops:7.1f} TFLOPS")
            row = {
                "shape": s.name,
                "dtype": s.dtype,
                "config": config,
                "tflops": round(tflops, 1),
                "median_us": round(median_us, 2),
                "arch": arch,
                "rocm_ver": args.rocm_ver,
                "date": args.date,
            }
            rows.append(row)

            if args.check:
                base = expected.get((s.name, config))
                if base is None:
                    print(f"  CHECK: no baseline for ({s.name}, {config}) -- skipped")
                    continue
                base_tf = float(base["tflops"])
                pct = (tflops - base_tf) / base_tf * 100.0 if base_tf > 0 else 0.0
                if pct < -args.regress_pct:
                    msg = (
                        f"{s.name}/{config}: {tflops:.1f} TF vs baseline "
                        f"{base_tf:.1f} TF ({pct:+.1f}% < -{args.regress_pct:.0f}%)"
                    )
                    regressions.append(msg)
                    print(f"  CHECK: REGRESSION {msg}")
                else:
                    print(
                        f"  CHECK: OK {tflops:.1f} TF vs baseline {base_tf:.1f} TF "
                        f"({pct:+.1f}%)"
                    )

    if args.write_expected:
        fieldnames = [
            "shape",
            "dtype",
            "config",
            "tflops",
            "median_us",
            "arch",
            "rocm_ver",
            "date",
        ]
        with args.write_expected.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.write_expected}")

    if args.check:
        if regressions:
            print(f"\nCHECK FAILED: {len(regressions)} regression(s):")
            for m in regressions:
                print(f"  - {m}")
            return 1
        print("\nCHECK PASSED: all measured shapes within tolerance of baseline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
