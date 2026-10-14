#!/usr/bin/env python3

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VARIANT_BUILDS = {
    "normal": "build_gfx1250",
    "postra": "build_gfx1250_postra",
    "expert": "build_gfx1250_expert",
    "combo": "build_gfx1250_combo",
    "priority": "build_gfx1250_priority",
}

EXPERT_VARIANTS = {"expert", "combo"}
VARIANT_ORDER = tuple(VARIANT_BUILDS)
TARGET_PATTERN = re.compile(
    r"tile_example_gemm_sweep_(?P<pipeline>basic|tdm)_"
    r"m(?P<tile_m>\d+)_n(?P<tile_n>\d+)_k(?P<tile_k>\d+)_"
    r"wm(?P<warp_m>\d+)_wn(?P<warp_n>\d+)_wk(?P<warp_k>\d+)$"
)
TILE_PATTERN = re.compile(
    r"m(?P<tile_m>\d+)_n(?P<tile_n>\d+)_k(?P<tile_k>\d+)_"
    r"wm(?P<warp_m>\d+)_wn(?P<warp_n>\d+)_wk(?P<warp_k>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep matched CK Tile basic/TDM binaries across scheduling variants."
    )
    parser.add_argument("--problems", required=True, type=Path, help="CSV with m,n,k columns")
    parser.add_argument("--output", required=True, type=Path, help="Result CSV")
    parser.add_argument("--prec", choices=("fp16", "bf16", "fp8"), default="fp16")
    parser.add_argument("--precs", nargs="+", choices=("fp16", "bf16", "fp8"))
    parser.add_argument("--markdown", type=Path, help="Write a Markdown report")
    parser.add_argument("--variants", nargs="+", choices=VARIANT_BUILDS, default=list(VARIANT_BUILDS))
    parser.add_argument("--pipelines", nargs="+", choices=("basic", "tdm"), default=("basic", "tdm"))
    parser.add_argument("--tiles", nargs="*", help="Optional tile suffixes such as m128_n128_k64_wm2_wn4_wk1")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--init", type=int, choices=(0, 2), default=0)
    parser.add_argument("--validate", action="store_true", help="CPU-validate every exact binary/problem")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    return parser.parse_args()


def gpu_is_idle() -> bool:
    result = subprocess.run(
        ["/opt/rocm/bin/amd-smi", "process", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "No running processes detected" in result.stdout


def load_problems(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"m", "n", "k"}.issubset(rows[0]):
        raise ValueError("problem CSV must contain m,n,k columns")
    return [{key: int(row[key]) for key in ("m", "n", "k")} for row in rows]


def discover_binaries(repo: Path, variant: str, pipelines: list[str], tiles: set[str] | None):
    binary_dir = repo / VARIANT_BUILDS[variant] / "bin"
    for binary in sorted(binary_dir.glob("tile_example_gemm_sweep_*")):
        match = TARGET_PATTERN.fullmatch(binary.name)
        if not match or match["pipeline"] not in pipelines:
            continue
        tile = binary.name.split(f"sweep_{match['pipeline']}_", 1)[1]
        if tiles and tile not in tiles:
            continue
        yield binary, match.groupdict(), tile


def parse_result(output: str) -> tuple[float, float]:
    match = re.search(r":\s+([0-9.eE+-]+) ms,\s+([0-9.eE+-]+) TFlops", output)
    if not match:
        raise RuntimeError("unable to parse GEMM timing output")
    return float(match.group(1)), float(match.group(2))


def format_problem(row: dict) -> str:
    return f"{row['m']}x{row['n']}x{row['k']}"


def format_tile(tile: str) -> tuple[str, str]:
    match = TILE_PATTERN.fullmatch(tile)
    if not match:
        raise ValueError(f"unexpected tile name: {tile}")
    block = f"{match['tile_m']}x{match['tile_n']}x{match['tile_k']}"
    warps = f"{match['warp_m']}x{match['warp_n']}x{match['warp_k']}"
    return block, warps


def write_markdown(rows: list[dict], output: Path, csv_output: Path, args: argparse.Namespace):
    best = {}
    for row in rows:
        key = (
            row["prec"],
            row["m"],
            row["n"],
            row["k"],
            row["variant"],
            row["pipeline"],
        )
        if key not in best or row["tflops"] > best[key]["tflops"]:
            best[key] = row

    lines = [
        "# CK Tile Basic V1 and TDM V1 Pipeline Sweep",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Method",
        "",
        "- GPU target: `gfx1250`",
        "- Layout: FP16/BF16/FP8 RCR with row-major output",
        f"- Warmup iterations: {args.warmup}",
        f"- Measured iterations: {args.repeat}",
        "- Expert scheduling variants: `expert`, `combo`",
        "- Validation: forced for every expert and combo result",
        f"- Raw results: [{csv_output.name}]({csv_output.name})",
        "",
        "## Scheduling Variants",
        "",
        "| Variant | Compiler flags |",
        "|---|---|",
        "| normal | default |",
        "| postra | `-mllvm -enable-post-misched=1` |",
        "| expert | `-mllvm -amdgpu-expert-scheduling-mode` |",
        "| combo | `-mllvm -enable-post-misched=1 -mllvm -amdgpu-expert-scheduling-mode` |",
        "| priority | `-mllvm -amdgpu-set-wave-priority` |",
        "",
        "## Best Configuration Per Variant",
        "",
        "| Precision | Problem | Variant | Pipeline | Tile | Latency (ms) | TFLOP/s | Validated |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(best.values(), key=lambda item: (
        item["prec"], item["m"], item["n"], item["k"], item["pipeline"],
        VARIANT_ORDER.index(item["variant"])
    )):
        lines.append(
            f"| {row['prec']} | {format_problem(row)} | {row['variant']} | "
            f"{row['pipeline']} | `{row['tile']}` | {row['latency_ms']:.6f} | "
            f"{row['tflops']:.3f} | {row['validated']} |"
        )

    lines.extend([
        "",
        "## Scheduler Performance vs Normal",
        "",
        "Each variant uses its fastest measured tile for the listed precision, problem, and pipeline. Speedup and TFLOP/s change use the best `normal` result as the baseline; positive values favor the scheduling variant.",
    ])
    precisions = sorted({row["prec"] for row in rows})
    problems = sorted({(row["m"], row["n"], row["k"]) for row in rows})
    pipelines = sorted({row["pipeline"] for row in rows})
    for precision in precisions:
        lines.extend([
            "",
            f"### {precision.upper()}",
            "",
            "| Problem | Pipeline | Schedule | Best Tile | Runtime (ms) | Normal (ms) | Speedup | TFLOP/s | Normal TFLOP/s | TFLOP/s Change |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for m, n, k in problems:
            for pipeline in pipelines:
                baseline = best[(precision, m, n, k, "normal", pipeline)]
                for variant in VARIANT_ORDER:
                    row = best[(precision, m, n, k, variant, pipeline)]
                    speedup = baseline["latency_ms"] / row["latency_ms"]
                    tflops_change = (row["tflops"] / baseline["tflops"] - 1) * 100
                    lines.append(
                        f"| {format_problem(row)} | {pipeline} | {variant} | `{row['tile']}` | "
                        f"{row['latency_ms']:.6f} | {baseline['latency_ms']:.6f} | "
                        f"{speedup:.3f}x | {row['tflops']:.3f} | {baseline['tflops']:.3f} | "
                        f"{tflops_change:+.1f}% |"
                    )

    by_tile = {
        (row["prec"], row["m"], row["n"], row["k"], row["pipeline"], row["tile"], row["variant"]): row
        for row in rows
    }
    tiles = sorted({row["tile"] for row in rows})
    lines.extend([
        "",
        "## Tile Size Performance",
        "",
        "For each exact tile, the fastest schedule is compared with `normal` on that same tile. This isolates scheduling effects while exposing how block dimensions and warp grids affect runtime and throughput.",
    ])
    for precision in precisions:
        lines.extend([
            "",
            f"### {precision.upper()} Tiles",
            "",
            "| Problem | Pipeline | Block Tile | Warp Grid | Normal Runtime (ms) | Normal TFLOP/s | Best Schedule | Best Runtime (ms) | Best TFLOP/s | Speedup |",
            "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ])
        for m, n, k in problems:
            for pipeline in pipelines:
                for tile in tiles:
                    key = (precision, m, n, k, pipeline, tile)
                    baseline = by_tile.get((*key, "normal"))
                    if baseline is None:
                        continue
                    candidates = [by_tile[(*key, variant)] for variant in VARIANT_ORDER]
                    winner = max(candidates, key=lambda row: row["tflops"])
                    block, warps = format_tile(tile)
                    speedup = baseline["latency_ms"] / winner["latency_ms"]
                    lines.append(
                        f"| {format_problem(baseline)} | {pipeline} | {block} | {warps} | "
                        f"{baseline['latency_ms']:.6f} | {baseline['tflops']:.3f} | "
                        f"{winner['variant']} | {winner['latency_ms']:.6f} | "
                        f"{winner['tflops']:.3f} | {speedup:.3f}x |"
                    )

    lines.extend([
        "",
        "## Complete Results",
        "",
        "| Precision | Problem | Variant | Pipeline | Tile | Latency (ms) | TFLOP/s | Validated |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['prec']} | {row['m']}x{row['n']}x{row['k']} | {row['variant']} | "
            f"{row['pipeline']} | `{row['tile']}` | {row['latency_ms']:.6f} | "
            f"{row['tflops']:.3f} | {row['validated']} |"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    problems = load_problems(args.problems)
    tiles = set(args.tiles) if args.tiles else None
    precisions = args.precs if args.precs else [args.prec]
    rows = []

    for precision in precisions:
        for variant in args.variants:
            binaries = list(discover_binaries(repo, variant, args.pipelines, tiles))
            if not binaries:
                raise RuntimeError(f"no sweep binaries found for {variant}")

            for binary, metadata, tile in binaries:
                if precision == "fp8" and int(metadata["tile_k"]) < 64:
                    continue
                for problem in problems:
                    if not args.allow_busy_gpu and not gpu_is_idle():
                        raise RuntimeError("GPU is busy or unavailable; aborting before launch")

                    validate = args.validate or variant in EXPERT_VARIANTS
                    command = [
                        str(binary),
                        f"-prec={precision}",
                        f"-m={problem['m']}",
                        f"-n={problem['n']}",
                        f"-k={problem['k']}",
                        "-a_layout=R",
                        "-b_layout=C",
                        "-c_layout=R",
                        f"-v={1 if validate else 0}",
                        f"-init={args.init}",
                        f"-warmup={args.warmup}",
                        f"-repeat={args.repeat}",
                        "-flush_cache=false",
                    ]
                    print("RUN", " ".join(command), flush=True)
                    result = subprocess.run(command, check=False, capture_output=True, text=True)
                    command_output = result.stdout + result.stderr
                    if result.returncode != 0 or (
                        validate and "verification result is:correct" not in command_output
                    ):
                        print(command_output, file=sys.stderr)
                        raise RuntimeError(f"failed or invalid result for {binary.name} {problem}")

                    latency_ms, tflops = parse_result(command_output)
                    rows.append(
                        {
                            "variant": variant,
                            "pipeline": metadata["pipeline"],
                            "tile": tile,
                            "prec": precision,
                            **problem,
                            "latency_ms": latency_ms,
                            "tflops": tflops,
                            "validated": int(validate),
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if args.markdown:
        write_markdown(rows, args.markdown, args.output, args)
    print(json.dumps({"runs": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())