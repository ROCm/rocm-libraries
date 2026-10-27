# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A production-shaped fake world for one-command runs on CPU.

`build_world(base)` writes, under `base`:

    bench/projects/hipblaslt/build/release   a gfx950 BF16 TN build: the
                                             library's Prediction table, a
                                             CU-specific Equality bank of the
                                             same problem type (some of its
                                             kernels share a table kernel's
                                             signature) and `bench_main` as
                                             its hipblaslt-bench
    datasets/fake_val.yaml                   a stage08 dataset

and returns the environment the run needs. `config(world, ...)` is a small
v3 config with the planner (seed bootstrap and bank), execution contexts
(trained and held out, re-planned), the near-winner re-reads, splits, the
deploy gate and an audit set.

`bench_main` is fake_hipblaslt's bench whose candidates are the merged pool,
the table followed by the bank when the run serves it (fake_hipblaslt
`bank_served`), as hipblaslt-bench resolves the solutions of a
`solution_index_list` (FAKE_BANK_STEMS names the bank's files), whose rows
carry the data types of FAKE_TYPES ("a,b,c,d,compute,scaleA,scaleB"), and
that spends FAKE_SLEEP_PER_KERNEL seconds on every timed kernel.
"""
from __future__ import annotations

import os
import random
import stat
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TESTS_DIR = Path(__file__).resolve().parent
for _p in (TESTS_DIR, TESTS_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_fakes as bf  # noqa: E402
import fake_hipblaslt as fh  # noqa: E402

ARCH = "gfx950"
PREFIX = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk"
)
STEM = f"{PREFIX}_ID75a0_{ARCH}"
BANK = f"{PREFIX}_CU256_ID75a0_{ARCH}"
N_CU, LDS, L2, CHIP = 256, 163840, 4194304, 0x75A0
TYPES = "bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0"
_MIS = ((16, 16, 32, 1), (32, 32, 16, 1))
_HARDWARE = {
    "type": "AMDGPU",
    "value": {
        "type": "And",
        "value": [
            {"type": "Processor", "value": ARCH},
            {"type": "CUCount", "value": N_CU},
            {"type": "PciChipId", "value": CHIP},
        ],
    },
}


def pool_and_bank() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The table's kernels and the bank's: new signatures, and twins of
    table kernels (the same 8 signature fields, another workgroup mapping
    and occupancy), which join their key as further members."""
    pool: List[Dict[str, Any]] = []
    bank: List[Dict[str, Any]] = []
    idx = 0
    for mt in ((64, 64, 64), (128, 128, 64), (256, 128, 64), (128, 256, 64)):
        for mi in _MIS:
            pool.append(bf.solution(idx, idx, mt=mt, mi=mi))
            idx += 1
    for mt, nta, ntb in (((256, 256, 64), 4, 4), ((128, 128, 64), 0, 4)):
        pool.append(bf.solution(idx, idx, mt=mt, mi=_MIS[1], nta=nta, ntb=ntb))
        idx += 1
    for mt in ((64, 128, 64), (128, 64, 64), (256, 256, 64)):
        for mi in _MIS:
            s = bf.solution(idx, idx, mt=mt, mi=mi)
            s["hardwarePredicate"] = _HARDWARE
            bank.append(s)
            idx += 1
    for p in pool[:4]:
        sm = dict(p["sizeMapping"], workGroupMapping=4, CUOccupancy=2)
        s = dict(p, index=idx, libraryLogicIndex=idx, sizeMapping=sm)
        s["hardwarePredicate"] = _HARDWARE
        bank.append(s)
        idx += 1
    return pool, bank


def write_bank(path: Path, solutions: List[Dict[str, Any]]) -> None:
    """A logic file whose kernels are Equality rows only."""
    import msgpack

    table = [
        {"key": [1, s["index"], 1, 1], "index": s["index"], "speed": 0.0}
        for s in solutions
    ]
    lib = {
        "type": "Problem",
        "rows": [
            {
                "predicate": {"type": "EqualityMatching"},
                "library": {"type": "Matching", "table": table},
            }
        ],
    }
    raw = msgpack.packb({"solutions": solutions, "library": lib}, use_bin_type=True)
    path.write_bytes(zlib.compress(raw))


def _merged_library(libdir: Path, stem: str, model_path: Optional[Path] = None):
    """fake_hipblaslt's `_library` over the table plus the bank, for a device
    with the bank's PCI chip id, when the run serves the bank."""
    from lib import bank_pool
    from lib.dat import library_logic_path, read_tensile_logic

    banks = [s for s in os.environ.get("FAKE_BANK_STEMS", "").split(",") if s]
    if not banks or not fh.bank_served(model_path):
        return _ORIGINAL_LIBRARY(libdir, stem, model_path)
    n_cu = int(os.environ.get("FAKE_HW", fh.FAKE_HW).split(",")[0])
    device = bank_pool.BankDevice(ARCH, n_cu, CHIP)
    pool = bank_pool.library_pool(libdir, stem, device, banks)
    by_index: Dict[int, Dict[str, Any]] = {}
    for s in [stem, *banks]:
        data = read_tensile_logic(library_logic_path(libdir, s)) or {}
        for sol in data.get("solutions", []):
            by_index.setdefault(int(sol["index"]), sol)
    return [by_index[k["sol_idx_global"]] for k in pool.kernels], pool.kernels


def _typed_values(g: Any, us: float, **kw: Any) -> List[str]:
    vals = _ORIGINAL_VALUES(g, us, **kw)
    types = os.environ.get("FAKE_TYPES")
    if types:
        vals[17:24] = types.split(",")
    return vals


def _slow_kernel_time_us(*args: Any, **kw: Any) -> float:
    delay = float(os.environ.get("FAKE_SLEEP_PER_KERNEL") or 0)
    if delay > 0:
        time.sleep(delay)
    return _ORIGINAL_TIME(*args, **kw)


_ORIGINAL_LIBRARY = fh._library
_ORIGINAL_VALUES = bf._values
_ORIGINAL_TIME = fh.kernel_time_us


def bench_main(argv: List[str]) -> int:
    fh._library = _merged_library
    bf._values = _typed_values
    fh.kernel_time_us = _slow_kernel_time_us
    return fh.bench_main(argv)


def _executable(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_world(base: Path) -> Dict[str, str]:
    """Write the fake build and dataset under `base`; the environment of
    the run (also its benches')."""
    base = Path(base)
    build = base / "bench" / "projects" / "hipblaslt" / "build" / "release"
    pool, bank = pool_and_bank()
    fh.write_fake_build(build, ARCH, STEM, pool, table=[s["index"] for s in pool])
    libdir = build / "Tensile" / "library" / ARCH
    write_bank(libdir / f"{BANK}.dat.zlib", bank)
    _executable(
        build / "clients" / "hipblaslt-bench",
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "from onecmd_fakes import bench_main\n"
        "sys.exit(bench_main(sys.argv[1:]))\n",
    )
    rng = random.Random(7)
    sizes = (256, 384, 768, 1024, 2048, 3000, 4096, 8192)
    entries = [
        {
            "M": rng.choice(sizes),
            "N": rng.choice(sizes),
            "K": rng.choice(sizes),
            "batch_count": rng.choice((1, 1, 1, 4)),
            "transA": "T",
            "transB": "N",
            "a_type": "bf16_r",
            "b_type": "bf16_r",
            "c_type": "bf16_r",
            "d_type": "bf16_r",
            "compute_type": "c_f32_r",
            "scale_type": "f32_r",
        }
        for _ in range(24)
    ]
    import yaml

    (base / "datasets").mkdir(parents=True, exist_ok=True)
    (base / "datasets" / "fake_val.yaml").write_text(yaml.safe_dump(entries))
    return {
        "FAKE_LIBRARY_DIR": str(libdir),
        "FAKE_LIBRARY_STEM": STEM,
        "FAKE_BANK_STEMS": BANK,
        "FAKE_HW": f"{N_CU},{LDS},{L2}",
        "FAKE_TYPES": TYPES,
    }


def config(base: Path, **overrides: Any) -> Dict[str, Any]:
    """The run config of the world under `base`; `overrides` set dotted
    keys."""
    base = Path(base)
    cfg: Dict[str, Any] = {
        "config_id": "onecmd_fake_bbs_tn",
        "arch": ARCH,
        "hipblaslt": {
            "a_type": "bf16_r",
            "b_type": "bf16_r",
            "c_type": "bf16_r",
            "d_type": "bf16_r",
            "scale_type": "f32_r",
            "compute_type": "c_f32_r",
            "transA": "T",
            "transB": "N",
            "library_stem": STEM,
        },
        "hardware": {
            "n_cu": N_CU,
            "lds_bytes": LDS,
            "l2_bytes": L2,
            "pci_chip_id": CHIP,
        },
        "runtime": {
            "devices": [0, 1],
            "rocm_libraries_root": str(base / "bench"),
            "gpu_lease_wait_s": 0,
            "stage_retry_backoff_s": 0,
        },
        "seed": {
            "target_per_cell": 12,
            "shapes_per_cell": 12,
            "seed": 42,
            "n_strata": 2,
            "exclude": [
                {"max_m": 128},
                {"max_n": 128},
                {"max_k": 32},
                {"min_batch": 2},
                {"min_m": 4097},
                {"min_n": 4097},
            ],
        },
        "bench": {
            "rebench": {"enabled": True},
            "blocks_per_gpu": 2,
            "stall_timeout_s": 120,
            "startup_grace_s": 120,
            "skip_slow_solution_ratio": 0.45,
        },
        "planner": {
            "enabled": True,
            "mode": "planned",
            "seed_per_cell": 3,
            "neighbors_k": 2,
            "bank_library_stems": [BANK],
            "top_k": 6,
            "families_top_f": 2,
            "family_variants": 0,
            "explore_random_k": 2,
            "min_coverage": 1,
        },
        "contexts": {
            "list": [
                {"cu_budget": 0},
                {"cu_budget": 128, "cotenant": True, "fraction": 0.25},
            ],
            "holdout": [{"cu_budget": 192, "cotenant": True, "fraction": 0.2}],
            "replan": True,
        },
        "train": {
            "workers": 2,
            "threads_per_worker": 1,
            "epochs": 2,
            "smart_k": 0,
            "catalog": "v3",
            "whitelist": "none",
            "cascade_k": 4,
            "min_cu_fraction": 0.5,
            "min_cell_gemms": 8,
            "min_base_cell_gemms": 4,
            "served_candidates": True,
            "off_list_sel_eff": "auto",
            "off_list_fallback": 0.83,
            "eval_points": 4,
        },
        "validate": {
            "sel_eff_threshold": 0.999,
            "min_cell_gemms": 4,
            "split_floor": 0.83,
            "split_after_attempts": 0,
        },
        "full": {"total_rounds": 3},
        "deploy": {"weight_dtype": "int8"},
        "stage08": {
            "datasets": [
                {"name": "fake_val", "path": str(base / "datasets" / "fake_val.yaml")}
            ]
        },
        "audit": {
            "dir": str(base / "audit"),
            "shapes_per_cell": 3,
            "shapes_per_context": 1,
            "part_shapes": 24,
            # the fewest the per-cell gate's cell tests take at gate.alpha 0.05
            "bootstrap": 40,
        },
    }
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return cfg
