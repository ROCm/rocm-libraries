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
    "Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1250-strict"
)
OTHER_STEM = (
    "TensileLibrary_F8F8_BF8_HA_Bias_SAB_SAV_UA_Type_F8B_HPA_"
    "Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1250-strict"
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


# a_type, b_type, c_type, d_type and compute_type of the rows, then scaleA
# and scaleB.
F8_TYPES = ("f8_r", "f8_r", "bf16_r", "bf16_r", "f32_r", "3", "3")
BF16_TYPES = ("bf16_r", "bf16_r", "bf16_r", "bf16_r", "f32_r", "0", "0")


def _values(
    g: Gemm,
    us: float,
    *,
    split_k: bool,
    adaptive: bool,
    samples: int,
    cv: float,
    types: Sequence[str] = F8_TYPES,
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
        *types,
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
    types: Sequence[str] = F8_TYPES,
) -> List[str]:
    values = _values(
        g, us, split_k=split_k, adaptive=adaptive, samples=samples, cv=cv, types=types
    )
    out = [
        f"[{rank}]:" + ",".join(columns(split_k=split_k, adaptive=adaptive)),
        "    " + ",".join(values),
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
    **size_mapping: object,
) -> Dict[str, object]:
    """A logic-file solution whose sizeMapping holds every key TensileLite
    requires; `size_mapping` adds or replaces keys."""
    sm: Dict[str, object] = {
        "waveNum": 4,
        "workGroup": [256, 1, 1],
        "threadTile": [1, 1, 1],
        "macroTile": [mt[0], mt[1], 1],
        "matrixInstruction": list(mi),
        "grvwA": grvw[0],
        "grvwB": grvw[1],
        "gwvwC": gwvw,
        "gwvwD": gwvw,
        "staggerU": 32,
        "staggerUMapping": 0,
        "depthU": mt[2],
        "globalSplitUPGR": 0,
        "globalSplitU": 1,
        "staggerStrideShift": 2,
        "workGroupMapping": 8,
        "sourceKernel": False,
        "globalAccumulation": 0,
        "workspaceSizePerElemC": 0,
        "workspaceSizePerElemBias": 0,
        "workGroupMappingXCC": 1,
        "workGroupMappingXCCGroup": -1,
        "globalSplitUCoalesced": False,
        "globalSplitUWorkGroupMappingRoundRobin": False,
        "CUOccupancy": occupancy,
        "PrefetchGlobalRead": 2,
        "MathClocksUnrolledLoop": 0,
        "synchronizerSizePerWG": 0,
        "nonTemporalA": nta,
        "nonTemporalB": ntb,
        "customMainLoopScheduling": 0,
        "NonTemporalD": 0,
        "WaveSeparateGlobalReadA": 0,
        "WaveSeparateGlobalReadB": 0,
        "UnrollLoopSwapGlobalReadOrder": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "NumLoadsCoalescedA": 1,
        "NumLoadsCoalescedB": 1,
        "WaveGroup": [2, 2],
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "LocalSplitU": 1,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
    }
    sm.update(size_mapping)
    return {
        "index": index,
        "libraryLogicIndex": local,
        "name": kernel_name(mt),
        "sizeMapping": sm,
    }


def write_logic(
    path: Path,
    solutions: Iterable[Dict[str, object]],
    table: Optional[Sequence[int]] = None,
) -> Path:
    """A logic file of `solutions`. With `table`, its library tree has a
    Prediction row listing those solution indices and a grid-based row with
    the other solutions, as TensileCreateLibrary writes them."""
    doc: Dict[str, object] = {"solutions": list(solutions)}
    if table is not None:
        other = [s["index"] for s in doc["solutions"] if s["index"] not in table]
        doc["library"] = {
            "type": "Problem",
            "rows": [
                {
                    "predicate": {"type": "PredictionMatching"},
                    "library": {"type": "Prediction", "table": list(table)},
                },
                {
                    "predicate": {"type": "GridBasedMatching"},
                    "library": {
                        "type": "Matching",
                        "table": [{"key": [1, 1, 1, 1], "index": i} for i in other],
                    },
                },
            ],
        }
    raw = msgpack.packb(doc, use_bin_type=True)
    path = Path(path)
    path.write_bytes(zlib.compress(raw) if path.name.endswith(".zlib") else raw)
    return path


# ── fake hipblaslt-bench ─────────────────────────────────────────────────────
#
# The fake runs the entries of its --yaml the bench runs (`gentest_entries`),
# one problem each. Plan file (path in env FAKE_BENCH_PLAN), all keys
# optional:
#   n_candidates     candidates per problem, solution indices 100, 101, ...
#                    (default 3; 1 with rsn: 1); a `solution_index_list`
#                    gives the candidates instead
#   crash            ["MxNxKxB", ...] crash in the middle of these problems
#   crash_once       same, but only the first time (state in `state_dir`)
#   crash_silent     crash before printing anything for these problems
#   crash_silent_once  same, but only the first time
#   crash_text       crash in the middle of every problem whose line holds this
#                    text
#   crash_silent_text  crash before printing anything for those problems
#   no_solution      print `NO solution found` for these problems
#   silent_text      print nothing for an entry whose line holds this text, as
#                    the bench does for an entry it gives up before its
#                    problem's first line
#   hang             print the preamble of these problems, then stop making
#                    progress with the output still open
#   teardown_crash   exit non-zero after every problem was printed
#   us               {"MxNxKxB": base_us} (default 10.0)
#   slow             [{"shape": "MxNxKxB", "factor": f, "processes": [i, ...]}]:
#                    the i-th process (counted from 0 per entry, in
#                    `state_dir`) with a line of that shape times every
#                    problem f times slower
#   state_dir        directory for crash_once markers
#   pid_dir          every process writes `<pid_dir>/<pid>`, a JSON object
#                    with its process group id (`pgid`), `yaml` path,
#                    `--device` and ROCR_VISIBLE_DEVICES (`rocr`)
#   yaml_dir         every process copies its yaml to `<yaml_dir>/<pid>.yaml`
#   kfd              {"root": fake sysfs (tests/gpu_fakes.py), "gpu_ids":
#                    {"<gpu>": gpu_id}, "hold_s": seconds, "foreign":
#                    {"<gpu>": pid}, "hidden": {"<gpu>": bytes}, "vram":
#                    {"<gpu>": bytes}, "linger_s": seconds}: the process
#                    holds memory on the gpu_id of the physical GPU its
#                    --device names in the fake KFD tree, sleeping `hold_s`
#                    before its first problem; on a GPU of `foreign` the pid
#                    holds memory on that gpu_id from then on; on a GPU of
#                    `hidden` the first process (state in `state_dir`) sets
#                    the card's use to bytes more than the processes hold;
#                    on a GPU of `vram` the process holds those bytes and
#                    the card shows them in use, for `linger_s` more after
#                    the process exits


def _field(line: str, key: str, default: Optional[str] = None) -> Optional[str]:
    m = re.search(rf"(?<![\w]){key}:\s*([^,}}]+)", line)
    return m.group(1).strip() if m else default


def _argument(value: object) -> object:
    """A yaml value as the bench's Arguments struct holds it: booleans and
    numbers as one number type, lists element by element."""
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_argument(v) for v in value)
    return value


def gentest_entries(text: str) -> List[str]:
    """The entry lines of a bench yaml that hipblaslt-bench runs, in file
    order. It converts the yaml with clients/hipblaslt_gentest.py, which
    writes an entry only when its Arguments differ from every earlier
    entry's, so of entries equal in every field (in any order) the first
    runs alone."""
    import yaml

    try:
        from yaml import CSafeLoader as SafeLoader
    except ImportError:  # a PyYAML built without libyaml
        from yaml import SafeLoader

    out: List[str] = []
    seen = set()
    for ln in text.splitlines():
        if not ln.strip().startswith("-"):
            continue
        try:
            (entry,) = yaml.load(ln, Loader=SafeLoader)
            key: object = tuple(
                sorted((str(k), _argument(v)) for k, v in entry.items())
            )
        except Exception:
            key = ln.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out


def _listed(line: str) -> Optional[List[int]]:
    m = re.search(r"(?<![\w])solution_index_list:\s*\[([^\]]*)\]", line)
    if m is None:
        return None
    return [int(t) for t in m.group(1).split(",") if t.strip()]


def _line_key(line: str) -> str:
    return "x".join(
        _field(line, k, "1" if k == "batch_count" else "0")
        for k in ("M", "N", "K", "batch_count")
    )


def _count(path: Path) -> int:
    """How many processes counted themselves at `path` before this one,
    which counts itself."""
    import fcntl

    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        seen = int(f.read() or 0)
        f.seek(0)
        f.truncate()
        f.write(str(seen + 1))
    return seen


def fake_bench_main(argv: List[str]) -> int:
    yaml_path = argv[argv.index("--yaml") + 1]
    plan: Dict[str, object] = {}
    if os.environ.get("FAKE_BENCH_PLAN"):
        plan = json.loads(Path(os.environ["FAKE_BENCH_PLAN"]).read_text())
    if plan.get("pid_dir"):
        tmp = Path(str(plan["pid_dir"])) / f".{os.getpid()}"
        record = {
            "pgid": os.getpgid(0),
            "yaml": yaml_path,
            "device": argv[argv.index("--device") + 1] if "--device" in argv else None,
            "rocr": os.environ.get("ROCR_VISIBLE_DEVICES"),
        }
        tmp.write_text(json.dumps(record))
        tmp.rename(tmp.with_name(str(os.getpid())))
    if plan.get("yaml_dir"):
        copy = Path(str(plan["yaml_dir"])) / f"{os.getpid()}.yaml"
        copy.write_text(Path(yaml_path).read_text())
    state_dir = Path(str(plan.get("state_dir", "."))) if plan.get("state_dir") else None

    def first_time(kind: str, key: str, listed: bool = False) -> bool:
        if state_dir is None or not (listed or key in plan.get(kind, [])):
            return False
        marker = state_dir / f"{kind}_{key}"
        if marker.exists():
            return False
        marker.write_text("1")
        return True

    lines = gentest_entries(Path(yaml_path).read_text())
    slow = 1.0
    for n, rule in enumerate(plan.get("slow") or []):
        if any(_line_key(ln) == rule["shape"] for ln in lines):
            if _count(Path(str(plan["state_dir"])) / f"slow_{n}") in rule["processes"]:
                slow *= float(rule["factor"])
    for ln in BANNER:
        print(ln, flush=True)
    kfd = plan.get("kfd")
    if kfd:
        import gpu_fakes

        device = argv[argv.index("--device") + 1] if "--device" in argv else "0"
        rocr = os.environ.get("ROCR_VISIBLE_DEVICES")
        gpu = rocr.split(",")[int(device)] if rocr else device
        gpu_id = int(kfd["gpu_ids"][gpu])
        held = (kfd.get("vram") or {}).get(gpu)
        gpu_fakes.set_memory(
            Path(kfd["root"]), os.getpid(), {gpu_id: int(held or 4096)}
        )
        if held is not None:
            gpu_fakes.set_card_used(
                Path(kfd["root"]),
                gpu_fakes.GPUS[int(gpu)],
                gpu_fakes.IDLE_CARD_BYTES + int(held),
            )
        intruder = (kfd.get("foreign") or {}).get(gpu)
        if intruder is not None:
            gpu_fakes.set_memory(Path(kfd["root"]), int(intruder), {gpu_id: 8192})
        hidden = (kfd.get("hidden") or {}).get(gpu)
        if hidden is not None and first_time("hidden", gpu, listed=True):
            gpu_fakes.set_card_used(
                Path(kfd["root"]),
                gpu_fakes.GPUS[int(gpu)],
                gpu_fakes.IDLE_CARD_BYTES + 4096 + int(hidden),
            )
        time.sleep(float(kfd.get("hold_s", 0)))
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
        listed = _listed(line)
        if listed is not None:
            n = len(listed)
        ids = listed if listed is not None else [100 + r for r in range(n)]
        crash_silent = plan.get("crash_silent_text")
        if (
            key in plan.get("crash_silent", [])
            or first_time("crash_silent_once", key)
            or (crash_silent is not None and str(crash_silent) in line)
        ):
            sys.stdout.flush()
            os._exit(134)
        if plan.get("silent_text") is not None and str(plan["silent_text"]) in line:
            continue
        print(preamble(n)[0], flush=True)
        if key in plan.get("no_solution", []) or n == 0:
            print("error: NO solution found! at testing_matmul.hpp:1", flush=True)
            continue
        print(preamble(n)[1], flush=True)
        if key in plan.get("hang", []):
            time.sleep(3600)
        crash_now = key in plan.get("crash", []) or first_time("crash_once", key)
        crash_text = plan.get("crash_text")
        crash_now = crash_now or (crash_text is not None and str(crash_text) in line)
        base = float(dict(plan.get("us", {})).get(key, 10.0)) * slow
        times = [base * (1.0 + 0.25 * ((r + 1) % 3)) for r in range(n)]
        # A crash happens while a candidate runs, before its record is printed.
        crash_rank = min(1, n - 1) if crash_now else -1
        for r in range(n):
            if r == crash_rank:
                print("HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: fake", flush=True)
                os._exit(134)
            for out in measured_lines(r, g, ids[r], times[r], types=line_types(line)):
                print(out, flush=True)
        if n > 1:
            best = min(range(n), key=lambda r: times[r])
            for out in winner(best, g, ids[best], times[best], types=line_types(line)):
                print(out, flush=True)
    sys.stdout.flush()
    if kfd:
        import shutil

        shutil.rmtree(Path(kfd["root"]) / "class/kfd/kfd/proc" / str(os.getpid()))
        if held is not None and kfd.get("linger_s"):
            _release_later(Path(kfd["root"]), int(gpu), float(kfd["linger_s"]))
    return 1 if plan.get("teardown_crash") else 0


def _release_later(root: Path, gpu: int, seconds: float) -> None:
    """Set the card's use of `gpu` back to idle after `seconds`, from a
    process that outlives this one and holds none of its output."""
    import subprocess

    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "import gpu_fakes\n"
        f"time.sleep({seconds!r})\n"
        f"gpu_fakes.set_card_used({str(root)!r}, gpu_fakes.GPUS[{gpu}], "
        "gpu_fakes.IDLE_CARD_BYTES)\n"
    )
    subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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


def line_types(line: str) -> Tuple[str, ...]:
    """The F8_TYPES fields of a bench yaml line as hipblaslt-bench prints
    them in its rows (the compute type without its `c_`, a fast compute type
    without `_fast`); a field the line leaves out takes F8_TYPES'."""
    names = ("a_type", "b_type", "c_type", "d_type", "compute_type", "scaleA", "scaleB")
    out = []
    for name, default in zip(names, F8_TYPES):
        value = str(_field(line, name, default))
        if name == "compute_type" and value.startswith("c_"):
            value = value[2:].replace("_fast_", "_")
        out.append(value)
    return tuple(out)


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
