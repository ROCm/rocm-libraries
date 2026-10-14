#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


VARIANT_BUILDS = {
    "normal": "build_gfx1250",
    "postra": "build_gfx1250_postra",
    "expert": "build_gfx1250_expert",
    "combo": "build_gfx1250_combo",
    "priority": "build_gfx1250_priority",
}

EXPERT_VARIANTS = {"expert", "combo"}
MANIFEST_FIELDS = [
    "variant",
    "pipeline",
    "tile",
    "prec",
    "m",
    "n",
    "k",
    "validated",
    "trace_dir",
    "att_files",
    "stats_files",
    "ui_dirs",
    "bytes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture ATT for the fastest measured tile in each sweep group."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--profiler", type=Path, default=Path("/opt/rocm-404163/bin/rocprofv3"))
    parser.add_argument("--att-library-path", type=Path, default=Path("/opt/rocm-404163/lib"))
    parser.add_argument("--variants", nargs="+", choices=VARIANT_BUILDS)
    parser.add_argument("--pipelines", nargs="+", choices=("basic", "tdm"))
    parser.add_argument("--precs", nargs="+", choices=("fp16", "bf16", "fp8"))
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    return parser.parse_args()


def select_best(args: argparse.Namespace) -> list[dict]:
    with args.results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    selected = {}
    for row in rows:
        if args.variants and row["variant"] not in args.variants:
            continue
        if args.pipelines and row["pipeline"] not in args.pipelines:
            continue
        if args.precs and row["prec"] not in args.precs:
            continue
        if args.sizes and int(row["m"]) not in args.sizes:
            continue
        key = (
            row["prec"],
            row["m"],
            row["n"],
            row["k"],
            row["variant"],
            row["pipeline"],
        )
        if key not in selected or float(row["tflops"]) > float(selected[key]["tflops"]):
            selected[key] = row

    rows = sorted(selected.values(), key=lambda row: (
        row["prec"], int(row["m"]), row["variant"], row["pipeline"]
    ))
    return rows[: args.limit] if args.limit else rows


def gpu_is_idle() -> bool:
    result = subprocess.run(
        ["/opt/rocm/bin/amd-smi", "process", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "No running processes detected" in result.stdout


def write_manifest(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def capture_key(row: dict) -> tuple[str, ...]:
    return tuple(row[field] for field in ("variant", "pipeline", "tile", "prec", "m", "n", "k"))


def capture_is_complete(row: dict, repo: Path) -> bool:
    trace_dir = repo / row["trace_dir"]
    return (
        trace_dir.is_dir()
        and bool(list(trace_dir.rglob("*.att")))
        and bool(list(trace_dir.rglob("stats_ui_output*.csv")))
        and any(path.is_dir() for path in trace_dir.rglob("ui_output_agent_*"))
    )


def append_markdown(rows: list[dict], markdown: Path, manifest: Path):
    marker = "\n## ATT Captures\n"
    content = markdown.read_text(encoding="utf-8")
    if marker in content:
        content = content.split(marker, 1)[0].rstrip() + "\n"

    lines = [
        "",
        "## ATT Captures",
        "",
        "One zero-warmup, one-repeat ATT capture was collected for the fastest measured tile in each precision, problem, scheduling variant, and pipeline group. ATT timings are not used in the benchmark tables.",
        "",
        "Expert and combo captures used CPU validation. Each trace directory contains the raw ATT stream, decoded instruction statistics, UI data, and `run.log`.",
        "",
        f"Manifest: [{manifest.name}]({manifest.name})",
        "",
        "| Precision | Problem | Variant | Pipeline | Tile | Validated | Raw ATT | Trace |",
        "|---|---:|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        trace = Path(row["trace_dir"])
        lines.append(
            f"| {row['prec']} | {row['m']}x{row['n']}x{row['k']} | {row['variant']} | "
            f"{row['pipeline']} | `{row['tile']}` | {row['validated']} | {row['att_files']} | "
            f"[{trace.name}]({trace.as_posix()}) |"
        )
    markdown.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output_root = args.output_dir.resolve()
    selected = select_best(args)
    if not selected:
        raise RuntimeError("no benchmark rows matched the ATT selection")
    if not args.profiler.is_file():
        raise RuntimeError(f"rocprofv3 not found: {args.profiler}")

    output_root.mkdir(parents=True, exist_ok=True)
    captured = load_manifest(args.manifest)
    captured_by_key = {capture_key(row): row for row in captured}
    for index, row in enumerate(selected, start=1):
        name = (
            f"{row['prec']}_m{row['m']}_n{row['n']}_k{row['k']}_{row['variant']}_"
            f"{row['pipeline']}_{row['tile']}"
        )
        trace_dir = output_root / name
        existing = captured_by_key.get(capture_key(row))
        if existing and capture_is_complete(existing, repo):
            print(f"SKIP {index}/{len(selected)} {name}", flush=True)
            continue
        if existing:
            captured.remove(existing)
            captured_by_key.pop(capture_key(row))
        if trace_dir.exists():
            shutil.rmtree(trace_dir)
        if not gpu_is_idle():
            raise RuntimeError("GPU is busy or unavailable; aborting before ATT launch")

        binary = (
            repo
            / VARIANT_BUILDS[row["variant"]]
            / "bin"
            / f"tile_example_gemm_sweep_{row['pipeline']}_{row['tile']}"
        )
        validate = row["variant"] in EXPERT_VARIANTS
        command = [
            str(args.profiler),
            "--att",
            "--att-library-path",
            str(args.att_library_path),
            "--att-gpu-index",
            "0",
            "--att-target-cu",
            "0",
            "--att-shader-engine-mask",
            "0x1",
            "--att-buffer-size",
            "0x20000000",
            "--kernel-include-regex",
            "GemmKernel",
            "--output-directory",
            str(trace_dir),
            "--",
            str(binary),
            f"-prec={row['prec']}",
            f"-m={row['m']}",
            f"-n={row['n']}",
            f"-k={row['k']}",
            "-a_layout=R",
            "-b_layout=C",
            "-c_layout=R",
            f"-v={1 if validate else 0}",
            "-init=0",
            "-warmup=0",
            "-repeat=1",
            "-flush_cache=false",
        ]
        print(f"ATT {index}/{len(selected)} {name}", flush=True)
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        output = result.stdout + result.stderr
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "run.log").write_text(output, encoding="utf-8")
        if result.returncode != 0 or (validate and "verification result is:correct" not in output):
            raise RuntimeError(f"ATT capture failed or was invalid: {name}")

        att_files = list(trace_dir.rglob("*.att"))
        stats_files = list(trace_dir.rglob("stats_ui_output*.csv"))
        ui_dirs = [path for path in trace_dir.rglob("ui_output_agent_*") if path.is_dir()]
        if not att_files or not stats_files or not ui_dirs:
            raise RuntimeError(f"ATT capture is incomplete: {name}")

        relative_trace = trace_dir.relative_to(repo)
        captured_row = {
            "variant": row["variant"],
            "pipeline": row["pipeline"],
            "tile": row["tile"],
            "prec": row["prec"],
            "m": row["m"],
            "n": row["n"],
            "k": row["k"],
            "validated": int(validate),
            "trace_dir": relative_trace.as_posix(),
            "att_files": len(att_files),
            "stats_files": len(stats_files),
            "ui_dirs": len(ui_dirs),
            "bytes": sum(path.stat().st_size for path in trace_dir.rglob("*") if path.is_file()),
        }
        captured.append(captured_row)
        captured_by_key[capture_key(row)] = captured_row
        write_manifest(captured, args.manifest)

    if args.markdown:
        append_markdown(captured, args.markdown, args.manifest)
    print(f"Captured {len(captured)} ATT traces in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())