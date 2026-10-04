# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A CPU-only stand-in for hipblaslt-bench and a fake hipBLASLt build tree.

`write_fake_build(root, arch, stem, kernels)` lays out
`<root>/clients/hipblaslt-bench` (an executable running `bench_main` with
this interpreter) and `<root>/Tensile/library/<arch>/<stem>.dat`.

The bench reads `--yaml`, takes the candidate pool from the library named by
FAKE_LIBRARY_STEM in HIPBLASLT_TENSILE_LIBPATH, and times every candidate
with a deterministic synthetic cost model. Without tilewright the candidates
come in "Origami" order (a fixed analytical guess, so rank 0 is the Origami
pick); with TENSILE_USE_TILEWRIGHT=1 they come in the order the deployed
model ranks them (through the tilewright Python module when it is installed,
otherwise unchanged). The output follows testing_matmul.hpp: an optional
`Solution selection time` line per problem (TENSILE_DB=0x10000), `Is
supported`, one result block per timed candidate, `Skip solution` lines for
pruned ones and a `Winner:` block. TILEWRIGHT_DIAG and TILEWRIGHT_PICK_LOG
print the engine's diagnostic lines.

FAKE_HW ("n_cu,lds_bytes,l2_bytes") is the hardware the fake runtime passes
to the engine; FAKE_PICK_RANK=k makes the pick log report the k-th ranked
kernel instead of the first, as a broken integration would.
"""
from __future__ import annotations

import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TESTS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TESTS_DIR.parent
for _p in (TESTS_DIR, PIPELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_fakes as bf  # noqa: E402


FAKE_HW = "64,65536,4194304"


def kernels_grid() -> List[Dict[str, object]]:
    """A small kernel pool: macro tiles x matrix instructions x hints."""
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
    return out


def write_fake_build(
    root: Path,
    arch: str,
    stem: str,
    kernels: Optional[Sequence[Dict[str, object]]] = None,
) -> Path:
    root = Path(root)
    lib = root / "Tensile" / "library" / arch
    lib.mkdir(parents=True, exist_ok=True)
    bf.write_logic(lib / f"{stem}.dat", list(kernels or kernels_grid()))
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
    (root / "CMakeCache.txt").write_text("# fake build\n")
    return root


def _field(line: str, key: str, default: Optional[str] = None) -> Optional[str]:
    m = re.search(rf"(?<![\w]){key}:\s*([^,}}]+)", line)
    return m.group(1).strip() if m else default


def kernel_time_us(m: int, n: int, k: int, b: int, sm: Dict[str, object]) -> float:
    """Synthetic latency: tile-quantized work over a fixed machine width plus
    a per-tile overhead that favors big tiles on big problems."""
    mt0, mt1 = sm["macroTile"][0], sm["macroTile"][1]
    mi = sm["matrixInstruction"]
    tiles = math.ceil(m / mt0) * math.ceil(n / mt1) * b
    waves = math.ceil(tiles / 64)
    eff = 1.0 if mi[0] == 32 else 0.92
    work = waves * mt0 * mt1 * max(k, 1) / (2.0e5 * eff)
    return round(2.0 + work + 0.02 * tiles + 0.0005 * (mt0 + mt1), 4)


def origami_order(m: int, n: int, k: int, b: int, sols: Sequence[Dict]) -> List[int]:
    """A deliberately imperfect analytical ranking: prefers tiles that cover
    the problem with the fewest tiles."""

    def guess(i: int) -> Tuple[float, int]:
        sm = sols[i]["sizeMapping"]
        tiles = math.ceil(m / sm["macroTile"][0]) * math.ceil(n / sm["macroTile"][1])
        return (tiles * b + 0.01 * sm["macroTile"][0], i)

    return sorted(range(len(sols)), key=guess)


def _library(libdir: Path, stem: str):
    from lib.dat import load_library_kernels, read_tensile_logic

    path = libdir / f"{stem}.dat"
    data = read_tensile_logic(path) or {}
    return data.get("solutions", []), load_library_kernels(libdir, stem)


def _indexed_model(libdir: Path, stem: str) -> Optional[Path]:
    idx = libdir / "tilewright_index"
    if not idx.is_file():
        return None
    for line in idx.read_text().splitlines():
        tok = line.split("#", 1)[0].split()
        if len(tok) >= 2 and tok[0] == stem:
            return libdir / tok[1]
    return None


def bench_main(argv: List[str]) -> int:
    yaml_path = argv[argv.index("--yaml") + 1]
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
    sols, kernels = _library(libdir, stem)
    model = cs = hw = None
    if tw_on:
        model_path = _indexed_model(libdir, stem)
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
                pool = evaluate.pool_from_kernels(kernels)
                cs = tilewright.CandidateSet(
                    model,
                    [
                        evaluate.make_config(tilewright, c, i)
                        for i, c in enumerate(pool)
                    ],
                )
                n_cu, lds, l2 = (
                    int(v) for v in os.environ.get("FAKE_HW", FAKE_HW).split(",")
                )
                hw = evaluate.make_hardware(
                    tilewright, DeviceHardware(n_cu=n_cu, lds_bytes=lds, l2_bytes=l2)
                )
            except ImportError:
                model = None
    lines = [
        ln
        for ln in Path(yaml_path).read_text().splitlines()
        if ln.strip().startswith("-")
    ]
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
        rsn = int(_field(line, "requested_solution_num", "-1"))
        ssr = float(_field(line, "skip_slow_solution_ratio", "0") or 0)
        order = origami_order(g.m, g.n, g.k, g.batch, sols)
        if model is not None:
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
            if scored:
                rest = [i for i in order if i not in scored]
                order = scored + rest
                if pick_log:
                    shown = int(os.environ.get("FAKE_PICK_RANK", "0"))
                    top = sols[scored[min(shown, len(scored) - 1)]]["sizeMapping"]
                    leaf = evaluate.model_cell_label(tilewright, model, pobj)
                    print(
                        f"[TILEWRIGHT_PICK] m={g.m} n={g.n} k={g.k} b={g.batch} "
                        f"tA={g.transA} tB={g.transB} leaf={leaf} top1_sig=(mt_m="
                        f"{top['macroTile'][0]},mt_n={top['macroTile'][1]},mt_k="
                        f"{top['depthU']},mi_m={top['matrixInstruction'][0]},mi_n="
                        f"{top['matrixInstruction'][1]},mi_k={top['matrixInstruction'][2]},"
                        f"cha={top['nonTemporalA']},chb={top['nonTemporalB']}) "
                        f"top1_score={ranked[0].score:f} n_configs={len(sols)}",
                        file=sys.stderr,
                        flush=True,
                    )
        if sel_time:
            base = 20.0 + (pi % 7) * 0.5 + (40.0 if pi == 0 else 0.0)
            print(
                f"Solution selection time: {base + (3.0 if tw_on else 0.0):.3f} us",
                flush=True,
            )
        n = len(order) if rsn < 0 else min(rsn, len(order))
        print(f"Is supported {n} / Total solutions: {n}", flush=True)
        best = None
        best_us = None
        for rank, si in enumerate(order[:n]):
            us = kernel_time_us(g.m, g.n, g.k, g.batch, sols[si]["sizeMapping"])
            if best_us is not None and ssr > 0 and us * ssr > best_us:
                print(
                    f"Skip solution: {rank} (best warm-up = {best_us:.2f} us , warm-up = "
                    f"{us:.2f} us, skip ratio = {ssr}, solution index = {sols[si]['index']})",
                    flush=True,
                )
                continue
            sm = sols[si]["sizeMapping"]
            mt = (sm["macroTile"][0], sm["macroTile"][1], sm["depthU"])
            for out in bf.measured_lines(rank, g, int(sols[si]["index"]), us, mt=mt):
                print(out, flush=True)
            if best_us is None or us < best_us:
                best, best_us = (rank, si), us
        if n > 1 and best is not None:
            rank, si = best
            for out in bf.winner(rank, g, int(sols[si]["index"]), best_us):
                print(out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(bench_main(sys.argv[1:]))
