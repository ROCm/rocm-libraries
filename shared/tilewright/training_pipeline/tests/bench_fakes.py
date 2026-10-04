# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Synthetic hipblaslt-bench output, Tensile logic files, a fake bench
executable and a fake KFD sysfs tree for CPU-only tests of the data path.

The log lines follow clients/common/include/testing_matmul.hpp and
argument_model.hpp: per tested candidate a `[rank]:<header>` line, a value
line, and `--Solution index` / `--Solution name` / `--kernel name` lines.
"""
from __future__ import annotations

import json
import os
import re
import stat
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import msgpack

TESTS_DIR = Path(__file__).resolve().parent

ARG_COLUMNS = [
    "transA",
    "transB",
    "grouped_gemm",
    "batch_count",
    "m",
    "n",
    "k",
    "alpha",
    "lda",
    "stride_a",
    "beta",
    "ldb",
    "stride_b",
    "ldc",
    "stride_c",
    "ldd",
    "stride_d",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "scaleA",
    "scaleB",
    "scaleC",
    "scaleD",
    "amaxD",
    "swizzle_a",
    "swizzle_b",
    "activation_type",
    "bias_vector",
    "bias_type",
    "aux_type",
]
ADAPTIVE_COLUMNS = [
    "batch",
    "samples",
    "hot_iters",
    "mean_us",
    "min_us",
    "cv",
    "rel_iqr",
    "status",
]

BANNER = [
    "hipBLASLt version: 100000",
    "Query device success: there are 1 devices. (Target device ID is 0)",
]

LIBRARY_STEM = (
    "TensileLibrary_F8F8_BF8_HA_MXAE8B32_MXBE8B32_Bias_SAV_UA_Type_F8B_HPA_"
    "Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1250"
)
OTHER_STEM = (
    "TensileLibrary_F8F8_BF8_HA_Bias_SAB_SAV_UA_Type_F8B_HPA_"
    "Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1250"
)


@dataclass(frozen=True)
class Gemm:
    m: int
    n: int
    k: int
    batch: int = 1
    transA: str = "T"
    transB: str = "N"


def kernel_name(mt: Tuple[int, int, int]) -> str:
    return f"Cijk_Alik_Bljk_F8BS_MT{mt[0]}x{mt[1]}x{mt[2]}_MI16x16x1_SN_TEST"


def columns(*, split_k: bool = False, adaptive: bool = False) -> List[str]:
    cols = list(ARG_COLUMNS)
    if split_k:
        cols += ["splitK", "wgm"]
    cols += ["rotating_buffer", "flush", "use_gpu_timer"]
    cols += ["hipblaslt-Gflops", "hipblaslt-GB/s", "us"]
    if adaptive:
        cols += ADAPTIVE_COLUMNS
    return cols


def _values(
    g: Gemm, us: float, *, split_k: bool, adaptive: bool, samples: int, cv: float
) -> List[str]:
    lda = g.k if g.transA == "T" else g.m
    ldb = g.k if g.transB == "N" else g.n
    vals = [
        g.transA,
        g.transB,
        "0",
        str(g.batch),
        str(g.m),
        str(g.n),
        str(g.k),
        "1",
        str(lda),
        str(g.m * g.k),
        "0",
        str(ldb),
        str(g.n * g.k),
        str(g.m),
        str(g.m * g.n),
        str(g.m),
        str(g.m * g.n),
        "f8_r",
        "f8_r",
        "bf16_r",
        "bf16_r",
        "f32_r",
        "3",
        "3",
        "0",
        "0",
        "0",
        "0",
        "0",
        "none",
        "0",
        "f32_r",
        "f32_r",
    ]
    if split_k:
        vals += ["2", "8"]
    gflops = 2.0 * g.m * g.n * g.k * g.batch / (us * 1e3)
    vals += ["16", "1", "1", f"{gflops:g}", f"{gflops / 10:g}", f"{us:g}"]
    if adaptive:
        vals += [
            "4",
            str(samples),
            "400",
            f"{us * 1.01:g}",
            f"{us * 0.99:g}",
            f"{cv:g}",
            f"{cv * 1.5:g}",
            "converged",
        ]
    return vals


def measured_lines(
    rank: int,
    g: Gemm,
    sol_index: int,
    us: float,
    mt: Tuple[int, int, int] = (128, 128, 64),
    *,
    split_k: bool = False,
    adaptive: bool = False,
    samples: int = 50,
    cv: float = 0.01,
    with_identity: bool = True,
) -> List[str]:
    out = [
        f"[{rank}]:" + ",".join(columns(split_k=split_k, adaptive=adaptive)),
        "    "
        + ",".join(
            _values(g, us, split_k=split_k, adaptive=adaptive, samples=samples, cv=cv)
        ),
    ]
    if with_identity:
        name = kernel_name(mt)
        out += [
            f"    --Solution index: {sol_index}",
            f"    --Solution name:  {name}",
            f"    --kernel name:    {name}",
        ]
    return out


def skip(
    rank: int,
    sol_index: Optional[int],
    warm: str,
    best: str = "10.00",
    ratio: str = "0.5",
) -> str:
    tail = "" if sol_index is None else f", solution index = {sol_index}"
    return (
        f"Skip solution: {rank} (best warm-up = {best} us , warm-up = {warm} us, "
        f"skip ratio = {ratio}{tail})"
    )


def preamble(n_supported: int, *, rotating: bool = True, adaptive: bool = False):
    out = []
    if rotating:
        out.append("Rotating buffer 16 MiB. Needed Size: 1 MiB. Needed block count: 16")
    if adaptive:
        out.append("Adaptive timing: warmup 1ms. Sample 1ms. Measure 2ms (max 4ms).")
    out.append(f"Is supported {n_supported} / Total solutions: {n_supported}")
    return out


def winner(rank: int, g: Gemm, sol_index: int, us: float, **kw) -> List[str]:
    return ["Winner: "] + measured_lines(rank, g, sol_index, us, **kw)


# ── Tensile logic files ──────────────────────────────────────────────────────


def solution(
    index: int,
    local: int,
    mt: Tuple[int, int, int] = (128, 128, 64),
    mi: Sequence[int] = (16, 16, 128, 1),
    *,
    nta: int = 0,
    ntb: int = 0,
    occupancy: int = 1,
    grvw: Tuple[int, int] = (16, 16),
    gwvw: int = 4,
) -> Dict[str, object]:
    return {
        "index": index,
        "libraryLogicIndex": local,
        "name": kernel_name(mt),
        "sizeMapping": {
            "macroTile": [mt[0], mt[1], 1],
            "depthU": mt[2],
            "matrixInstruction": list(mi),
            "nonTemporalA": nta,
            "nonTemporalB": ntb,
            "CUOccupancy": occupancy,
            "grvwA": grvw[0],
            "grvwB": grvw[1],
            "gwvwD": gwvw,
        },
    }


def write_logic(path: Path, solutions: Iterable[Dict[str, object]]) -> Path:
    raw = msgpack.packb({"solutions": list(solutions)}, use_bin_type=True)
    path = Path(path)
    path.write_bytes(zlib.compress(raw) if path.name.endswith(".zlib") else raw)
    return path


# ── fake hipblaslt-bench ─────────────────────────────────────────────────────
#
# Plan file (path in env FAKE_BENCH_PLAN), all keys optional:
#   n_candidates     candidates per problem (default 3; 1 with rsn: 1)
#   crash            ["MxNxKxB", ...] crash in the middle of these problems
#   crash_once       same, but only the first time (state in `state_dir`)
#   crash_silent     crash before printing anything for these problems
#   crash_silent_once  same, but only the first time
#   no_solution      print `NO solution found` for these problems
#   hang             print the preamble of these problems, then stop making
#                    progress with the output still open
#   teardown_crash   exit non-zero after every problem was printed
#   us               {"MxNxKxB": base_us} (default 10.0)
#   state_dir        directory for crash_once markers
#   pid_dir          every process writes `<pid_dir>/<pid>`, a JSON object
#                    with its process group id (`pgid`) and `yaml` path


def _field(line: str, key: str, default: Optional[str] = None) -> Optional[str]:
    m = re.search(rf"(?<![\w]){key}:\s*([^,}}]+)", line)
    return m.group(1).strip() if m else default


def fake_bench_main(argv: List[str]) -> int:
    yaml_path = argv[argv.index("--yaml") + 1]
    plan: Dict[str, object] = {}
    if os.environ.get("FAKE_BENCH_PLAN"):
        plan = json.loads(Path(os.environ["FAKE_BENCH_PLAN"]).read_text())
    if plan.get("pid_dir"):
        tmp = Path(str(plan["pid_dir"])) / f".{os.getpid()}"
        tmp.write_text(json.dumps({"pgid": os.getpgid(0), "yaml": yaml_path}))
        tmp.rename(tmp.with_name(str(os.getpid())))
    state_dir = Path(str(plan.get("state_dir", "."))) if plan.get("state_dir") else None

    def first_time(kind: str, key: str) -> bool:
        if key not in plan.get(kind, []) or state_dir is None:
            return False
        marker = state_dir / f"{kind}_{key}"
        if marker.exists():
            return False
        marker.write_text("1")
        return True

    lines = [
        ln
        for ln in Path(yaml_path).read_text().splitlines()
        if ln.strip().startswith("-")
    ]
    for ln in BANNER:
        print(ln, flush=True)
    for line in lines:
        g = Gemm(
            int(_field(line, "M")),
            int(_field(line, "N")),
            int(_field(line, "K")),
            int(_field(line, "batch_count", "1")),
            _field(line, "transA", "T"),
            _field(line, "transB", "N"),
        )
        key = f"{g.m}x{g.n}x{g.k}x{g.batch}"
        rsn = int(_field(line, "requested_solution_num", "-1"))
        n = int(plan.get("n_candidates", 3))
        if rsn > 0:
            n = min(n, rsn)
        if key in plan.get("crash_silent", []) or first_time("crash_silent_once", key):
            sys.stdout.flush()
            os._exit(134)
        print(preamble(n)[0], flush=True)
        if key in plan.get("no_solution", []):
            print("error: NO solution found! at testing_matmul.hpp:1", flush=True)
            continue
        print(preamble(n)[1], flush=True)
        if key in plan.get("hang", []):
            time.sleep(3600)
        crash_now = key in plan.get("crash", []) or first_time("crash_once", key)
        base = float(dict(plan.get("us", {})).get(key, 10.0))
        times = [base * (1.0 + 0.25 * ((r + 1) % 3)) for r in range(n)]
        # A crash happens while a candidate runs, before its record is printed.
        crash_rank = min(1, n - 1) if crash_now else -1
        for r in range(n):
            if r == crash_rank:
                print("HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: fake", flush=True)
                os._exit(134)
            for out in measured_lines(r, g, 100 + r, times[r]):
                print(out, flush=True)
        if n > 1:
            best = min(range(n), key=lambda r: times[r])
            for out in winner(best, g, 100 + best, times[best]):
                print(out, flush=True)
    sys.stdout.flush()
    return 1 if plan.get("teardown_crash") else 0


def write_fake_bench(directory: Path) -> Path:
    """Executable that runs `fake_bench_main` with this interpreter."""
    path = Path(directory) / "fake-hipblaslt-bench"
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "from bench_fakes import fake_bench_main\n"
        "sys.exit(fake_bench_main(sys.argv[1:]))\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def bench_line(g: Gemm, **knobs) -> str:
    kv = {
        "function": "matmul",
        "transA": g.transA,
        "transB": g.transB,
        "a_type": "f8_r",
        "b_type": "f8_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "scale_type": "f32_r",
        "compute_type": "f32_r",
        "scaleA": 3,
        "scaleB": 3,
        "M": g.m,
        "N": g.n,
        "K": g.k,
        "batch_count": g.batch,
        "iters": 10,
        "cold_iters": 10,
        "requested_solution_num": -1,
        "print_kernel_info": 1,
    }
    kv.update(knobs)
    return "- {" + ", ".join(f"{k}: {v}" for k, v in kv.items()) + "}\n"


# ── fake KFD sysfs topology ──────────────────────────────────────────────────


def write_kfd_tree(
    root: Path,
    gpu_banks: Sequence[Sequence[Tuple[int, int]]],
    *,
    cpu_nodes: int = 1,
    gpu_local_mem: Optional[Sequence[int]] = None,
) -> Path:
    """`root/<node>/properties` and `mem_banks/<b>/properties`: `cpu_nodes`
    CPU nodes, then one GPU node per entry of `gpu_banks`, each a list of
    (heap_type, size_in_bytes) banks."""
    root = Path(root)
    node = 0
    for _ in range(cpu_nodes):
        d = root / str(node)
        (d / "mem_banks" / "0").mkdir(parents=True)
        (d / "properties").write_text("cpu_cores_count 8\nsimd_count 0\n")
        (d / "mem_banks" / "0" / "properties").write_text(
            "heap_type 0\nsize_in_bytes 1000000000\n"
        )
        node += 1
    for g, banks in enumerate(gpu_banks):
        d = root / str(node)
        d.mkdir(parents=True)
        local = gpu_local_mem[g] if gpu_local_mem else 0
        (d / "properties").write_text(
            f"cpu_cores_count 0\nsimd_count 64\nlocal_mem_size {local}\n"
        )
        for b, (heap, size) in enumerate(banks):
            bd = d / "mem_banks" / str(b)
            bd.mkdir(parents=True)
            (bd / "properties").write_text(f"heap_type {heap}\nsize_in_bytes {size}\n")
        node += 1
    return root
