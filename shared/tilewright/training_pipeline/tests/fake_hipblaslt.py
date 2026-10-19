# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A CPU-only stand-in for hipblaslt-bench and a fake hipBLASLt build tree.

`write_fake_build(root, arch, stem, kernels, table)` lays out
`<root>/clients/hipblaslt-bench` (an executable running `bench_main` with
this interpreter), `<root>/Tensile/library/<arch>/<stem>.dat` and
`<root>/tensilelite/libtensilelite-host.so.1.0`, a file holding the strings
by which the orchestrator's preflight tells what the build's TensileLite
serves (`capabilities`, default `CAPABILITIES`).

The bench runs the entries of `--yaml` the real one runs
(`bench_fakes.gentest_entries`), takes the candidate pool from the library
named by FAKE_LIBRARY_STEM in HIPBLASLT_TENSILE_LIBPATH (the solutions of its
Prediction table, in table order, when it has one; with FAKE_BANK_STEMS, a
comma list of bank libraries such as `write_bank_library` writes, followed
by the bank of lib/bank_pool.py for a device of the library directory's
arch, which a tilewright run ranks as TENSILE_TILEWRIGHT_BANK says or, when
it is unset, when the model reads kernel attributes (`bank_served`), and
never with FAKE_BANK_UNSERVED=1, as a TensileLite that does not serve the
bank), and times every candidate
with a deterministic synthetic cost model. Without tilewright the candidates
come in "Origami" order (a fixed analytical guess, so rank 0 is the Origami
pick); with TENSILE_USE_TILEWRIGHT=1 they come in the order the deployed
model ranks them (through the tilewright Python module when it is installed,
otherwise unchanged), from a pool without the kernels of the library stem's
blocklist sidecar next to its tilewright_index, matched by their
`lib.correctness.blocklist_key`, as TensileLite's adapter leaves them out.
A `solution_index_list` replaces the candidates by the
listed solutions in list order, `algo_method: 1` takes them in library order,
and `skip_screen_prepass` screens against the fastest of all candidates
instead of the fastest so far. The output follows testing_matmul.hpp: an optional
`Solution selection time` line per problem (TENSILE_DB=0x10000), `Is
supported`, one result block per timed candidate, `Skip solution` lines for
pruned ones and a `Winner:` block; a problem left without candidates prints
`NO solution found` instead. TILEWRIGHT_DIAG and TILEWRIGHT_PICK_LOG
print the engine's diagnostic lines. Every result row names the data types,
compute type and scale modes of its yaml line as the real bench prints them
(`bench_fakes.line_types`).

Without TENSILE_PREDICTION_LIB=1, as production runs, the rows of the
library files FAKE_ROUTE_STEMS names (a comma list of stems in the library
directory) are searched before the Prediction row, in that order: an
Equality row answers a problem whose (M, N, batch, K) is a key of its table,
a GridBased row every problem, with its kernels in table order. A problem
those rows answer in full does not reach the Prediction row, so the model
ranks nothing for it and prints no pick-log line. With FAKE_ROUTE_MODEL_FIRST=1
the runtime searches the Prediction row of a loaded model before those
GridBased rows, as a TensileLite that looks ahead to a loaded model does;
Equality rows still answer first. FAKE_UNSCORED (comma-separated M values)
makes the model score nothing for problems of those M, as for a problem of
no cell: their candidates keep the analytical order and no pick-log line
prints.

FAKE_HW ("n_cu,lds_bytes,l2_bytes") is the hardware the fake runtime passes
to the engine. As a broken integration would, FAKE_PICK_RANK=k makes the pick
log report the k-th ranked kernel instead of the first, FAKE_PICK_TIE=last
the last of the kernels tied with the first, FAKE_PICK_NO_INDEX=1 drops its
`top1_index` field, and FAKE_ORDER_SWAP=1 returns the first two ranked
kernels swapped. FAKE_DROP_RANK=k leaves the k-th ranked kernel out of the
returned list, as a runtime does with a kernel whose predicates fail, and
FAKE_UNSUPPORTED (comma-separated solution indices) leaves those kernels out
of an explicit `solution_index_list`. FAKE_BENCH_YAML_DIR receives a copy of
every yaml the bench runs. FAKE_SLOW_PLAN names a JSON list of slowdowns,
each {"shape": [m, n, k, b], "alone": bool (optional), "factor": f,
"processes": [i, ...]}: the i-th bench process (counted from 0, in a counter
file beside the plan) with a line of that shape (with "alone", only lines of
that shape; stage03's warm-up line aside) times every kernel of those
lines f times slower. FAKE_FIRST_WARMUP=f makes the screen's warm-up of the
first listed kernel f times its latency, as the real bench's first cold call
after the problem's data initialisation reads. FAKE_CODE_OBJECT_WARMUP=f
makes the screen's warm-up of the first kernel a bench process runs of each
library file (the pool's and FAKE_BANK_STEMS') f times its latency, as the
real bench reads it in the problem that loads the file's code object.
FAKE_SUSTAINED names, as JSON {"default": f, "<solution index>": f, ...},
how much slower a line with a warm-up of at least 100 ms (a sustained
window) times each kernel. FAKE_CRASH_SHAPES (comma-separated MxNxKxB)
makes the bench die on a problem of those shapes after it selected its
kernels, as the real bench dies of an illegal memory access on some
block-scaled problems; with FAKE_CRASH_IN (comma-separated) only a bench
whose --yaml path holds one of those strings does.

`--sm_count_target N` narrows the machine width of the cost model to N CUs
(so winners move with the CU budget); `--hybrid_assignment_policy` is
accepted and ignored. `<root>/clients/hipblaslt-cotenant` stands in for the
cotenant launcher: it logs its `--cus` to `--log`, prints launcher lines and
runs the command after `--`, with HIP_VISIBLE_DEVICES set to its `--device`
as the real launcher does.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TESTS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TESTS_DIR.parent
for _p in (TESTS_DIR, PIPELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_fakes as bf  # noqa: E402


FAKE_HW = "64,65536,4194304"
CAPABILITIES = (
    "MLREC_v2",
    "MLREC_v3",
    "TENSILE_TILEWRIGHT_BANK",
    "CU floor",
)


def kernels_grid(twins: bool = False) -> List[Dict[str, object]]:
    """A small kernel pool: macro tiles x matrix instructions. With `twins`
    the grid is followed by a twin of every kernel, identical but for its
    solution index and workgroup mapping."""
    out = []
    idx = 0
    for mt in (
        (64, 64, 64),
        (128, 128, 64),
        (256, 128, 64),
        (128, 256, 64),
        (256, 256, 64),
    ):
        for mi in ((16, 16, 128, 1), (32, 32, 64, 1)):
            out.append(bf.solution(idx, idx, mt=mt, mi=mi))
            idx += 1
    if twins:
        for s in list(out):
            sm = dict(s["sizeMapping"], workGroupMapping=4)
            out.append(dict(s, index=idx, libraryLogicIndex=idx, sizeMapping=sm))
            idx += 1
    return out


def write_fake_build(
    root: Path,
    arch: str,
    stem: str,
    kernels: Optional[Sequence[Dict[str, object]]] = None,
    table: Optional[Sequence[int]] = None,
    capabilities: Sequence[str] = CAPABILITIES,
) -> Path:
    root = Path(root)
    host = root / "tensilelite" / "libtensilelite-host.so.1.0"
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_bytes(b"\x7fELF\0" + "\0".join(capabilities).encode() + b"\0")
    lib = root / "Tensile" / "library" / arch
    lib.mkdir(parents=True, exist_ok=True)
    bf.write_logic(lib / f"{stem}.dat", list(kernels or kernels_grid()), table)
    clients = root / "clients"
    clients.mkdir(parents=True, exist_ok=True)
    exe = clients / "hipblaslt-bench"
    exe.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "from fake_hipblaslt import bench_main\n"
        "sys.exit(bench_main(sys.argv[1:]))\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    cot = clients / "hipblaslt-cotenant"
    cot.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "from fake_hipblaslt import cotenant_main\n"
        "sys.exit(cotenant_main(sys.argv[1:]))\n"
    )
    cot.chmod(cot.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    kernel = clients / "hipblaslt-cotenant-kernel"
    kernel.write_text("#!/bin/sh\nexit 0\n")
    kernel.chmod(kernel.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (clients / "hipblaslt_common.yaml").write_text(
        "Arguments:\n  - solution_index_list: c_int32*4096\n"
        "  - skip_screen_prepass: c_bool\n"
    )
    (root / "CMakeCache.txt").write_text("# fake build\n")
    return root


def _field(line: str, key: str, default: Optional[str] = None) -> Optional[str]:
    m = re.search(rf"(?<![\w]){key}:\s*([^,}}]+)", line)
    return m.group(1).strip() if m else default


def _solution_list(line: str) -> Optional[List[int]]:
    m = re.search(r"(?<![\w])solution_index_list:\s*\[([^\]]*)\]", line)
    if m is None:
        return None
    return [int(t) for t in m.group(1).split(",") if t.strip()]


def kernel_time_us(
    m: int, n: int, k: int, b: int, sm: Dict[str, object], width: int = 64
) -> float:
    """Synthetic latency: tile-quantized work over a machine `width` CUs wide
    plus a per-tile overhead that favors big tiles on big problems; a
    workgroup mapping other than 8 costs 2% more."""
    mt0, mt1 = sm["macroTile"][0], sm["macroTile"][1]
    mi = sm["matrixInstruction"]
    tiles = math.ceil(m / mt0) * math.ceil(n / mt1) * b
    waves = math.ceil(tiles / max(int(width), 1))
    eff = 1.0 if mi[0] == 32 else 0.92
    work = waves * mt0 * mt1 * max(k, 1) / (2.0e5 * eff)
    wgm = 1.0 if sm.get("workGroupMapping", 8) == 8 else 1.02
    return round((2.0 + work + 0.02 * tiles + 0.0005 * (mt0 + mt1)) * wgm, 4)


def origami_order(m: int, n: int, k: int, b: int, sols: Sequence[Dict]) -> List[int]:
    """A deliberately imperfect analytical ranking: prefers tiles that cover
    the problem with the fewest tiles."""

    def guess(i: int) -> Tuple[float, int]:
        sm = sols[i]["sizeMapping"]
        tiles = math.ceil(m / sm["macroTile"][0]) * math.ceil(n / sm["macroTile"][1])
        return (tiles * b + 0.01 * sm["macroTile"][0], i)

    return sorted(range(len(sols)), key=guess)


def write_bank_library(
    libdir: Path, stem: str, solutions: Sequence[Dict[str, object]]
) -> Path:
    """A logic file whose solutions are the rows of one Equality library,
    as the bank's files are."""
    import msgpack

    table = [{"key": [1, s["index"], 1, 1], "index": s["index"]} for s in solutions]
    doc = {
        "solutions": list(solutions),
        "library": {
            "type": "Problem",
            "rows": [
                {
                    "predicate": {"type": "EqualityMatching"},
                    "library": {"type": "Matching", "table": table},
                }
            ],
        },
    }
    path = Path(libdir) / f"{stem}.dat"
    path.write_bytes(msgpack.packb(doc, use_bin_type=True))
    return path


def bank_served(model_path: Optional[Path]) -> bool:
    """Whether a run ranks the bank, as TensileLite's adapter decides: every
    run without tilewright; a tilewright run as TENSILE_TILEWRIGHT_BANK says
    or, when it is unset, when the model reads kernel attributes; never with
    FAKE_BANK_UNSERVED=1."""
    if os.environ.get("TENSILE_USE_TILEWRIGHT") != "1":
        return True
    if os.environ.get("FAKE_BANK_UNSERVED") == "1":
        return False
    setting = os.environ.get("TENSILE_TILEWRIGHT_BANK")
    if setting is not None:
        return int(setting, 0) != 0
    if model_path is None or not model_path.is_file():
        return False
    from lib import mlrec

    return bool(mlrec.read_model(model_path.read_bytes()).get("attribute_spec"))


def _library(libdir: Path, stem: str, model_path: Optional[Path] = None):
    """(solutions, kernel_dat_info entries) of the library's kernel pool, in
    pool order: the candidates of its Prediction library, then the bank's
    when the run serves it (`bank_served` with the model at `model_path`),
    for a device of the library directory's arch."""
    from lib import bank_pool
    from lib.arch import canonical_arch
    from lib.dat import library_logic_path, load_library_kernels, read_tensile_logic

    banks = [b for b in os.environ.get("FAKE_BANK_STEMS", "").split(",") if b]
    if banks and bank_served(model_path):
        n_cu = int(os.environ.get("FAKE_HW", FAKE_HW).split(",")[0])
        device = bank_pool.BankDevice(canonical_arch(libdir.name), n_cu)
        kernels = bank_pool.library_pool(libdir, stem, device, banks).kernels
    else:
        banks, kernels = [], load_library_kernels(libdir, stem)
    by_index: Dict[int, Dict[str, object]] = {}
    for s in [stem] + banks:
        data = read_tensile_logic(library_logic_path(libdir, s)) or {}
        for sol in data.get("solutions", []):
            by_index.setdefault(int(sol["index"]), sol)
    return [by_index[k["sol_idx_global"]] for k in kernels], kernels


def _kernel_files(libdir: Path, stem: str) -> Dict[int, str]:
    """{solution index: library file} of the pool's file and the
    FAKE_BANK_STEMS files, the first file's where several hold an index."""
    from lib.dat import library_logic_path, read_tensile_logic

    banks = [b for b in os.environ.get("FAKE_BANK_STEMS", "").split(",") if b]
    out: Dict[int, str] = {}
    for s in [stem] + banks:
        try:
            data = read_tensile_logic(library_logic_path(libdir, s)) or {}
        except FileNotFoundError:
            continue
        for sol in data.get("solutions", []):
            out.setdefault(int(sol["index"]), s)
    return out


def _indexed_model(libdir: Path, stem: str) -> Optional[Path]:
    """The model the engine loads for `stem` on a device of unknown chip ID."""
    from lib import tilewright_index as twi

    idx = libdir / "tilewright_index"
    if not idx.is_file():
        return None
    try:
        name = twi.parse(twi.read_text(idx)).model_of(stem)
    except ValueError:
        return None
    return None if name is None else libdir / name


# The problem properties of a Matching library's keys, (M, N, batch, K), as
# TensileLite names them.
ROUTE_KEY_PROPERTIES = [
    {"type": "FreeSizeA", "index": 0},
    {"type": "FreeSizeB", "index": 0},
    {"type": "BatchSize", "index": 0},
    {"type": "BoundSize", "index": 0},
]


def write_route_library(
    libdir: Path,
    stem: str,
    solutions: Sequence[Dict[str, object]],
    kind: str = "GridBasedMatching",
    keys: Optional[Sequence[Sequence[int]]] = None,
) -> Path:
    """A logic file whose solutions are the rows of one Matching library of
    row predicate `kind`, keyed by `keys` ((M, N, batch, K) per solution, for
    an Equality row, `ROUTE_KEY_PROPERTIES`), as the GridBased and Equality
    files of a problem type hold them."""
    import msgpack

    table = [
        {"key": list(keys[i]) if keys else [1, 1, 1, 1], "index": s["index"]}
        for i, s in enumerate(solutions)
    ]
    matching = {"type": "Matching", "properties": ROUTE_KEY_PROPERTIES, "table": table}
    doc = {
        "solutions": list(solutions),
        "library": {
            "type": "Problem",
            "rows": [{"predicate": {"type": kind}, "library": matching}],
        },
    }
    path = Path(libdir) / f"{stem}.dat"
    path.write_bytes(msgpack.packb(doc, use_bin_type=True))
    return path


def _route_rows(libdir: Path) -> List[Tuple[str, List[Tuple[Tuple[int, ...], Dict]]]]:
    """(row predicate, [(key, solution)]) of the rows of the FAKE_ROUTE_STEMS
    files, in their order."""
    from lib.dat import library_logic_path, read_tensile_logic

    out = []
    for stem in [s for s in os.environ.get("FAKE_ROUTE_STEMS", "").split(",") if s]:
        data = read_tensile_logic(library_logic_path(libdir, stem)) or {}
        by_index = {int(sol["index"]): sol for sol in data.get("solutions", [])}
        for row in (data.get("library") or {}).get("rows") or []:
            entries = [
                (tuple(int(v) for v in e.get("key") or ()), by_index[int(e["index"])])
                for e in (row.get("library") or {}).get("table") or []
                if int(e["index"]) in by_index
            ]
            out.append(((row.get("predicate") or {}).get("type"), entries))
    return out


def unhide_devices(monkeypatch) -> None:
    """Drop the HIP / CUDA device lists that hide the GPUs from the tests: a
    context wrapper refuses a device index outside them."""
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)


def cotenant_main(argv: List[str]) -> int:
    sep = argv.index("--")
    opts, cmd = argv[:sep], argv[sep + 1 :]

    def opt(name: str) -> Optional[str]:
        return opts[opts.index(name) + 1] if name in opts else None

    cus = int(opt("--cus"))
    if opt("--log"):
        Path(opt("--log")).write_text(f"cus {cus} device {opt('--device')}\n")
    print(f"launching cotenant on {cus} CUs", flush=True)
    print("running: " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    if opt("--device") is not None:
        env["HIP_VISIBLE_DEVICES"] = opt("--device")
    os.execve(cmd[0], cmd, env)


def _slowdowns(lines: Sequence[str]) -> Dict[int, float]:
    """{line index: factor} of this process under FAKE_SLOW_PLAN; stage03's
    warm-up line (entry `warmup_line`) is none of the process's lines."""
    import fcntl

    plan_path = os.environ.get("FAKE_SLOW_PLAN")
    if not plan_path:
        return {}
    out: Dict[int, float] = {}
    own = [i for i, ln in enumerate(lines) if _field(ln, "name") != "warmup_line"]
    for n, rule in enumerate(json.loads(Path(plan_path).read_text())):

        def hit(line: str) -> bool:
            shape = [
                int(_field(line, key, "1" if key == "batch_count" else "0"))
                for key in ("M", "N", "K", "batch_count")
            ]
            return shape == list(rule["shape"])

        idx = [i for i in own if hit(lines[i])]
        if not idx or (rule.get("alone") and len(idx) < len(own)):
            continue
        counter = Path(f"{plan_path}.count{n}")
        with open(counter, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            seen = int(f.read() or 0)
            f.seek(0)
            f.truncate()
            f.write(str(seen + 1))
        if seen in rule["processes"]:
            for i in idx:
                out[i] = float(rule["factor"])
    return out


def bench_main(argv: List[str]) -> int:
    yaml_path = argv[argv.index("--yaml") + 1]
    if os.environ.get("FAKE_BENCH_YAML_DIR"):
        copy = Path(os.environ["FAKE_BENCH_YAML_DIR"]) / (
            f"{os.getpid()}_{time.time_ns()}.yaml"
        )
        copy.write_text(Path(yaml_path).read_text())
    width = 64
    if "--sm_count_target" in argv:
        width = int(argv[argv.index("--sm_count_target") + 1]) or width
    libdir = Path(
        os.environ.get("HIPBLASLT_TENSILE_LIBPATH") or os.environ["FAKE_LIBRARY_DIR"]
    )
    stem = os.environ["FAKE_LIBRARY_STEM"]
    tw_on = os.environ.get("TENSILE_USE_TILEWRIGHT") == "1"
    diag = tw_on and os.environ.get("TILEWRIGHT_DIAG") == "1"
    pick_log = tw_on and os.environ.get("TILEWRIGHT_PICK_LOG") == "1"
    sel_time = "0x10000" in os.environ.get("TENSILE_DB", "")
    for knob in ("TILEWRIGHT_DIAG", "TILEWRIGHT_PICK_LOG"):
        os.environ.pop(knob, None)
    model_path = _indexed_model(libdir, stem) if tw_on else None
    sols, kernels = _library(libdir, stem, model_path)
    sidecar = libdir / f"{stem}.tilewright.blocklist"
    if tw_on and sidecar.is_file():
        from lib.correctness import blocklist_key, read_sidecar

        blocked = {blocklist_key(n) for n in read_sidecar(sidecar)}
        keep = [
            i
            for i, k in enumerate(kernels)
            if blocklist_key(k["kernel_name"]) not in blocked
        ]
        sols, kernels = [sols[i] for i in keep], [kernels[i] for i in keep]
    model = cs = hw = None
    if tw_on:
        if model_path is not None and model_path.is_file():
            data = model_path.read_bytes()
            if diag:
                from lib import mlrec

                p = mlrec.read_model(data)
                print(
                    f"[TILEWRIGHT_DIAG FILE] path={model_path} arch={p['arch']} "
                    f"qhash={p['feature_catalog_hash']} qdim={p['dims'][0]} "
                    f"idim={p['dims'][1]} xdim={p['dims'][2]} n_cells={len(p['cells'])} "
                    f"n_splits={len(p['splits'])} weights={p['weight_dtype_name']}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                import tilewright

                from lib import evaluate
                from lib.hardware import DeviceHardware

                model = evaluate.load_model_bytes(data, tilewright)
                n_cu, lds, l2 = (
                    int(v) for v in os.environ.get("FAKE_HW", FAKE_HW).split(",")
                )
                device = DeviceHardware(n_cu=n_cu, lds_bytes=lds, l2_bytes=l2)
                cs = evaluate.candidate_set(
                    tilewright, model, evaluate.pool_from_kernels(kernels), device
                )
                hw = evaluate.make_hardware(tilewright, device)
            except ImportError:
                model = None
    lines = bf.gentest_entries(Path(yaml_path).read_text())
    slow = _slowdowns(lines)
    route_rows = (
        _route_rows(libdir) if os.environ.get("TENSILE_PREDICTION_LIB") != "1" else []
    )
    # A TensileLite that looks ahead to a loaded model searches it before
    # the GridBased rows.
    grid_after = model is not None and os.environ.get("FAKE_ROUTE_MODEL_FIRST") == "1"
    first_load = float(os.environ.get("FAKE_CODE_OBJECT_WARMUP", "1"))
    file_of = _kernel_files(libdir, stem) if first_load != 1.0 else {}
    loaded = set()
    for ln in bf.BANNER:
        print(ln, flush=True)
    for pi, line in enumerate(lines):
        g = bf.Gemm(
            int(_field(line, "M")),
            int(_field(line, "N")),
            int(_field(line, "K")),
            int(_field(line, "batch_count", "1")),
            _field(line, "transA", "T"),
            _field(line, "transB", "N"),
        )
        types = bf.line_types(line)
        rsn = int(_field(line, "requested_solution_num", "-1"))
        ssr = float(_field(line, "skip_slow_solution_ratio", "0") or 0)
        listed = _solution_list(line)
        algo_method = int(_field(line, "algo_method", "0"))
        prescreen = _field(line, "skip_screen_prepass", "false") in ("true", "1")
        soaked = {}
        if (
            os.environ.get("FAKE_SUSTAINED")
            and float(_field(line, "warmup_time", "0") or 0) >= 100
        ):
            soaked = json.loads(os.environ["FAKE_SUSTAINED"])
        order = origami_order(g.m, g.n, g.k, g.batch, sols)
        ahead: List[Dict] = []
        after: List[Dict] = []
        if listed is None and algo_method != 1:
            for kind, entries in route_rows:
                if kind == "EqualityMatching":
                    ahead += [
                        sol for key, sol in entries if key == (g.m, g.n, g.batch, g.k)
                    ]
                elif kind == "GridBasedMatching":
                    (after if grid_after else ahead).extend(sol for _, sol in entries)
        asked = rsn < 0 or len(ahead) < rsn
        if listed is not None:
            pos = {int(sol["index"]): i for i, sol in enumerate(sols)}
            unsupported = {
                int(t) for t in os.environ.get("FAKE_UNSUPPORTED", "").split(",") if t
            }
            order = [pos[i] for i in listed if i in pos and i not in unsupported]
            rsn = -1
        elif algo_method == 1:
            order = list(range(len(sols)))
        elif model is not None and asked:
            from lib import evaluate

            row = {
                "m": g.m,
                "n": g.n,
                "k": g.k,
                "batch_count": g.batch,
                "transA": g.transA,
                "transB": g.transB,
                "a_type": _field(line, "a_type", "f8_r"),
                "b_type": _field(line, "b_type", "f8_r"),
                "c_type": _field(line, "c_type", "bf16_r"),
                "d_type": _field(line, "d_type", "bf16_r"),
                "compute_type": _field(line, "compute_type", "c_f32_r"),
            }
            import tilewright

            from lib import features as fs

            pobj = evaluate.make_problem(tilewright, fs.problem_kwargs_from_row(row))
            ranked = cs.rank(pobj, hw, max(rsn, 1))
            scored = [int(r.config_index) for r in ranked if r.scored]
            if str(g.m) in os.environ.get("FAKE_UNSCORED", "").split(","):
                scored = []
            if scored:
                rest = [i for i in order if i not in scored]
                order = scored + rest
                if os.environ.get("FAKE_ORDER_SWAP") == "1" and len(scored) > 1:
                    order[0], order[1] = order[1], order[0]
                drop = os.environ.get("FAKE_DROP_RANK")
                if drop is not None and len(scored) > int(drop):
                    order.pop(int(drop))
                if pick_log:
                    shown = int(os.environ.get("FAKE_PICK_RANK", "0"))
                    pick = scored[min(shown, len(scored) - 1)]
                    if os.environ.get("FAKE_PICK_TIE") == "last":
                        pick = [
                            int(r.config_index)
                            for r in ranked
                            if r.scored and r.score == ranked[0].score
                        ][-1]
                    top = sols[pick]["sizeMapping"]
                    leaf = evaluate.model_cell_label(tilewright, model, pobj)
                    index = (
                        ""
                        if os.environ.get("FAKE_PICK_NO_INDEX") == "1"
                        else f" top1_index={pick}"
                    )
                    print(
                        f"[TILEWRIGHT_PICK] m={g.m} n={g.n} k={g.k} b={g.batch} "
                        f"tA={g.transA} tB={g.transB} leaf={leaf} top1_sig=(mt_m="
                        f"{top['macroTile'][0]},mt_n={top['macroTile'][1]},mt_k="
                        f"{top['depthU']},mi_m={top['matrixInstruction'][0]},mi_n="
                        f"{top['matrixInstruction'][1]},mi_k={top['matrixInstruction'][2]},"
                        f"cha={top['nonTemporalA']},chb={top['nonTemporalB']}) "
                        f"top1_score={ranked[0].score:f}{index} n_configs={len(sols)}",
                        file=sys.stderr,
                        flush=True,
                    )
        if sel_time:
            base = 20.0 + (pi % 7) * 0.5 + (40.0 if pi == 0 else 0.0)
            print(
                f"Solution selection time: {base + (3.0 if tw_on else 0.0):.3f} us",
                flush=True,
            )
        cands = [sols[si] for si in order]
        if listed is None and algo_method != 1:
            cands = ahead + (cands + after if asked else [])
        n = len(cands) if rsn < 0 else min(rsn, len(cands))
        if n == 0:
            print("error: NO solution found! at testing_matmul.hpp:1", flush=True)
            continue
        print(f"Is supported {n} / Total solutions: {n}", flush=True)
        crash_in = os.environ.get("FAKE_CRASH_IN")
        if f"{g.m}x{g.n}x{g.k}x{g.batch}" in os.environ.get(
            "FAKE_CRASH_SHAPES", ""
        ).split(",") and (
            not crash_in or any(s in yaml_path for s in crash_in.split(","))
        ):
            print("error: an illegal memory access was encountered", flush=True)
            return 1
        best = None
        best_us = None
        screen_us = None
        first = float(os.environ.get("FAKE_FIRST_WARMUP", "1"))
        warm_factor = []
        for rank, sol in enumerate(cands[:n]):
            factor = first if rank == 0 else 1.0
            owner = file_of.get(int(sol["index"]))
            if owner is not None and owner not in loaded:
                loaded.add(owner)
                factor *= first_load
            warm_factor.append(factor)
        if prescreen and ssr > 0 and n:
            screen_us = min(
                kernel_time_us(g.m, g.n, g.k, g.batch, sol["sizeMapping"], width)
                * warm_factor[rank]
                for rank, sol in enumerate(cands[:n])
            )
        for rank, sol in enumerate(cands[:n]):
            us = kernel_time_us(g.m, g.n, g.k, g.batch, sol["sizeMapping"], width)
            if soaked:
                us *= soaked.get(str(sol["index"]), soaked.get("default", 1.0))
            us = round(us * slow.get(pi, 1.0), 4)
            warm_us = us * warm_factor[rank]
            if screen_us is not None and warm_us * ssr > screen_us:
                print(
                    f"Skip solution: {rank} (best warm-up = {screen_us:.2f} us , warm-up = "
                    f"{warm_us:.2f} us, skip ratio = {ssr}, solution index = "
                    f"{sol['index']})",
                    flush=True,
                )
                continue
            if (
                screen_us is None
                and best_us is not None
                and ssr > 0
                and us * ssr > best_us
            ):
                print(
                    f"Skip solution: {rank} (best warm-up = {best_us:.2f} us , warm-up = "
                    f"{us:.2f} us, skip ratio = {ssr}, solution index = {sol['index']})",
                    flush=True,
                )
                continue
            sm = sol["sizeMapping"]
            mt = (sm["macroTile"][0], sm["macroTile"][1], sm["depthU"])
            for out in bf.measured_lines(
                rank, g, int(sol["index"]), us, mt=mt, types=types
            ):
                print(out, flush=True)
            if best_us is None or us < best_us:
                best, best_us = (rank, sol), us
        if n > 1 and best is not None:
            rank, sol = best
            for out in bf.winner(rank, g, int(sol["index"]), best_us, types=types):
                print(out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(bench_main(sys.argv[1:]))
