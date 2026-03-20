#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Integrate Wave assembly kernels into the hipBLASLt rocroller build.

Reads wave kernel assembly from --asm-dir (output of benchmark_mxfp4.py with
--asm-dir), copies .s files into the custom_kernels directory, and updates the
CMakeLists.txt, custom_kernels.cpp, and solution_selection.cpp files so that
hipBLASLt can build and select the new kernels.

Optionally rebuilds hipblaslt-bench and runs benchmark_shapes.py.

Usage:
    python integrate_wave_kernels.py --asm-dir /tmp/wave_asm
    python integrate_wave_kernels.py --asm-dir /tmp/wave_asm --build --benchmark
    python integrate_wave_kernels.py --asm-dir /tmp/wave_asm --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HIPBLASLT_DIR = Path(__file__).resolve().parent
BUILD_DIR = HIPBLASLT_DIR / "build"
ROCROLLER_SRC = (
    HIPBLASLT_DIR / "library" / "src" / "amd_detail" / "rocblaslt" / "src" / "rocroller"
)
CUSTOM_KERNELS_DIR = ROCROLLER_SRC / "custom_kernels"
CMAKE_PATH = CUSTOM_KERNELS_DIR / "CMakeLists.txt"
CPP_PATH = ROCROLLER_SRC / "custom_kernels.cpp"
SELECTION_PATH = ROCROLLER_SRC / "solution_selection.cpp"

WAVE_STATIC_RE = re.compile(
    r"^wave_mxfp4_static_gemm_"
    r"(?P<mt_m>\d+)x(?P<mt_n>\d+)x(?P<mt_k>\d+)_"
    r"(?P<wm>\d+)x(?P<wn>\d+)x(?P<wk>\d+)$"
)

WAVE_DYNAMIC_RE = re.compile(
    r"^wave_mxfp4_dynamic_gemm_" r"(?P<mt_m>\d+)x(?P<mt_n>\d+)x(?P<mt_k>\d+)$"
)

CMAKE_BEGIN = "# --- BEGIN AUTO-GENERATED WAVE KERNELS (do not edit manually) ---"
CMAKE_END = "# --- END AUTO-GENERATED WAVE KERNELS ---"
CPP_BEGIN = "// --- BEGIN AUTO-GENERATED WAVE KERNELS (do not edit manually) ---"
CPP_END = "// --- END AUTO-GENERATED WAVE KERNELS ---"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class WaveKernel:
    name: str
    macrotile: tuple[int, int, int]
    wave_shape: tuple[int, int, int]  # (0,0,0) for dynamic kernels
    block_size: tuple[int, int, int]
    asm_file: str
    dynamic: bool = False
    wave_shapes: list[tuple[int, int, int]] | None = (
        None  # all validated shapes (dynamic only)
    )

    @property
    def mt_m(self):
        return self.macrotile[0]

    @property
    def mt_n(self):
        return self.macrotile[1]

    @property
    def mt_k(self):
        return self.macrotile[2]

    @property
    def wave_m(self):
        return self.wave_shape[0]

    @property
    def wave_n(self):
        return self.wave_shape[1]

    @property
    def wave_k(self):
        return self.wave_shape[2]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_kernels(asm_dir: Path) -> list[WaveKernel]:
    """Read manifest or parse filenames to discover wave kernels in asm_dir."""
    manifest_path = asm_dir / "wave_kernels_manifest.json"
    if manifest_path.exists():
        return _load_manifest(manifest_path)
    return _scan_filenames(asm_dir)


def _load_manifest(manifest_path: Path) -> list[WaveKernel]:
    with open(manifest_path) as f:
        data = json.load(f)
    kernels = []
    for entry in data["kernels"]:
        is_dynamic = entry.get("dynamic", False)
        if is_dynamic:
            wave_shapes = [tuple(s) for s in entry.get("wave_shapes", [])]
            kernels.append(
                WaveKernel(
                    name=entry["name"],
                    macrotile=tuple(entry["macrotile"]),
                    wave_shape=(0, 0, 0),
                    block_size=tuple(entry.get("block_size", [256, 2, 1])),
                    asm_file=entry["asm_file"],
                    dynamic=True,
                    wave_shapes=wave_shapes,
                )
            )
        else:
            kernels.append(
                WaveKernel(
                    name=entry["name"],
                    macrotile=tuple(entry["macrotile"]),
                    wave_shape=tuple(entry["wave_shape"]),
                    block_size=tuple(entry.get("block_size", [256, 2, 1])),
                    asm_file=entry["asm_file"],
                    dynamic=False,
                )
            )
    print(f"Loaded {len(kernels)} kernels from manifest {manifest_path}")
    return kernels


def _scan_filenames(asm_dir: Path) -> list[WaveKernel]:
    kernels = []
    for p in sorted(asm_dir.glob("wave_mxfp4_static_gemm_*.s")):
        stem = p.stem
        m = WAVE_STATIC_RE.match(stem)
        if not m:
            print(f"  Warning: skipping unrecognized file {p.name}", file=sys.stderr)
            continue
        mt = (int(m.group("mt_m")), int(m.group("mt_n")), int(m.group("mt_k")))
        ws = (int(m.group("wm")), int(m.group("wn")), int(m.group("wk")))
        kernels.append(
            WaveKernel(
                name=stem,
                macrotile=mt,
                wave_shape=ws,
                block_size=(256, 2, 1),
                asm_file=p.name,
                dynamic=False,
            )
        )
    for p in sorted(asm_dir.glob("wave_mxfp4_dynamic_gemm_*.s")):
        stem = p.stem
        m = WAVE_DYNAMIC_RE.match(stem)
        if not m:
            print(f"  Warning: skipping unrecognized file {p.name}", file=sys.stderr)
            continue
        mt = (int(m.group("mt_m")), int(m.group("mt_n")), int(m.group("mt_k")))
        kernels.append(
            WaveKernel(
                name=stem,
                macrotile=mt,
                wave_shape=(0, 0, 0),
                block_size=(256, 2, 1),
                asm_file=p.name,
                dynamic=True,
            )
        )
    print(
        f"Discovered {len(kernels)} kernels by scanning filenames in {asm_dir}",
    )
    return kernels


# ---------------------------------------------------------------------------
# Copy assembly files
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_asm_files(asm_dir: Path, kernels: list[WaveKernel], dry_run: bool) -> int:
    """Copy .s files from asm_dir to custom_kernels/. Returns number of files copied."""
    copied = 0
    for k in kernels:
        src = asm_dir / k.asm_file
        dst = CUSTOM_KERNELS_DIR / k.asm_file
        if not src.exists():
            print(f"  Warning: {src} not found, skipping", file=sys.stderr)
            continue
        if dst.exists() and _file_hash(src) == _file_hash(dst):
            continue
        if dry_run:
            print(f"  [dry-run] Would copy {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"  Copied {k.asm_file}")
        copied += 1
    return copied


# ---------------------------------------------------------------------------
# CMakeLists.txt update
# ---------------------------------------------------------------------------


def _scan_wave_asm_in_dir() -> list[str]:
    """Return sorted list of wave_mxfp4_*.s filenames present in custom_kernels/."""
    static = set(p.name for p in CUSTOM_KERNELS_DIR.glob("wave_mxfp4_static_gemm_*.s"))
    dynamic = set(
        p.name for p in CUSTOM_KERNELS_DIR.glob("wave_mxfp4_dynamic_gemm_*.s")
    )
    return sorted(static | dynamic)


def update_cmake(dry_run: bool) -> bool:
    """Replace content between markers in CMakeLists.txt. Returns True if changed."""
    text = CMAKE_PATH.read_text()
    if CMAKE_BEGIN not in text or CMAKE_END not in text:
        print(
            f"  ERROR: markers not found in {CMAKE_PATH}. "
            f"Expected '{CMAKE_BEGIN}' and '{CMAKE_END}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    wave_files = _scan_wave_asm_in_dir()
    entries = "\n".join(f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/{f}" for f in wave_files)
    new_block = f"    {CMAKE_BEGIN}\n{entries}\n    {CMAKE_END}"

    pattern = re.compile(
        re.escape(f"    {CMAKE_BEGIN}") + r".*?" + re.escape(f"    {CMAKE_END}"),
        re.DOTALL,
    )
    new_text = pattern.sub(new_block, text)

    if new_text == text:
        print("  CMakeLists.txt: no changes needed")
        return False

    if dry_run:
        print("  [dry-run] Would update CMakeLists.txt")
    else:
        CMAKE_PATH.write_text(new_text)
        print(f"  Updated {CMAKE_PATH.name} ({len(wave_files)} wave kernels)")
    return True


# ---------------------------------------------------------------------------
# custom_kernels.cpp update
# ---------------------------------------------------------------------------


def _gen_cpp_kernel_block(
    kernels: list[WaveKernel], indent: str, flip_macrotiles: bool
) -> str:
    """Generate the C++ registration block for wave kernels."""
    lines = [f"{indent}{CPP_BEGIN}"]
    lines.append(f"{indent}mxfp4Kernel.swizzleA = true;\n")

    for k in kernels:
        mt_m, mt_n, mt_k = k.macrotile
        tile_0, tile_1 = (mt_n, mt_m) if flip_macrotiles else (mt_m, mt_n)
        bx, by, bz = k.block_size

        if k.dynamic:
            lines.append(
                f"{indent}params.workgroupTile = {{{tile_0}, {tile_1}, {mt_k}}};\n"
                f"{indent}cache.addKernel(\n"
                f"{indent}    mxfp4Kernel,\n"
                f"{indent}    params,\n"
                f'{indent}    createCustomGemmKernel("{k.name}",\n'
                f"{indent}                           mxfp4Kernel,\n"
                f"{indent}                           params.workgroupTile,\n"
                f"{indent}                           {{{bx}, {by}, {bz}}},\n"
                f'{indent}                           getCoPath() / "rr_custom_kernels.co"));\n'
            )
        else:
            wm, wn, wk = k.wave_shape
            # StaticShape uses hipBLASLt convention: {wave_n, wave_m, wave_k}
            lines.append(
                f"{indent}params.workgroupTile = {{{tile_0}, {tile_1}, {mt_k}}};\n"
                f"{indent}cache.addKernel(\n"
                f"{indent}    mxfp4Kernel,\n"
                f"{indent}    params,\n"
                f'{indent}    createCustomGemmKernel("{k.name}",\n'
                f"{indent}                           mxfp4Kernel,\n"
                f"{indent}                           params.workgroupTile,\n"
                f"{indent}                           {{{bx}, {by}, {bz}}},\n"
                f"{indent}                           StaticShape{{{wn}, {wm}, {wk}}},\n"
                f'{indent}                           getCoPath() / "rr_custom_kernels.co"));\n'
            )

    lines.append(f"{indent}{CPP_END}")
    return "\n".join(lines)


def update_cpp(kernels: list[WaveKernel], dry_run: bool, flip_macrotiles: bool = False) -> bool:
    """Replace content between markers in custom_kernels.cpp. Returns True if changed."""
    text = CPP_PATH.read_text()
    if CPP_BEGIN not in text or CPP_END not in text:
        print(
            f"  ERROR: markers not found in {CPP_PATH}. "
            f"Expected '{CPP_BEGIN}' and '{CPP_END}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Detect indentation from the existing BEGIN marker
    for line in text.splitlines():
        if CPP_BEGIN in line:
            indent = line[: len(line) - len(line.lstrip())]
            break
    else:
        indent = "            "

    sorted_kernels = sorted(kernels, key=lambda k: (k.macrotile, k.wave_shape))
    new_block = _gen_cpp_kernel_block(sorted_kernels, indent, flip_macrotiles)

    pattern = re.compile(
        re.escape(f"{indent}{CPP_BEGIN}") + r".*?" + re.escape(f"{indent}{CPP_END}"),
        re.DOTALL,
    )
    new_text = pattern.sub(new_block, text)

    if new_text == text:
        print("  custom_kernels.cpp: no changes needed")
        return False

    if dry_run:
        print("  [dry-run] Would update custom_kernels.cpp")
    else:
        CPP_PATH.write_text(new_text)
        print(
            f"  Updated {CPP_PATH.name} ({len(sorted_kernels)} wave kernel registrations)"
        )
    return True


# ---------------------------------------------------------------------------
# solution_selection.cpp update
# ---------------------------------------------------------------------------


def _parse_tile_array(text: str, array_name: str) -> list[tuple[int, int, int]]:
    """Parse a constexpr std::array<WorkGroupTileSize, N> from source text."""
    pattern = re.compile(
        rf"constexpr\s+std::array<WorkGroupTileSize,\s*\w+>\s+{array_name}\s*=\s*\{{(.*?)\}}\}};",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return []
    body = m.group(1)
    tiles = []
    for tm in re.finditer(r"\{(\d+),\s*(\d+),\s*(\d+)\}", body):
        tiles.append((int(tm.group(1)), int(tm.group(2)), int(tm.group(3))))
    return tiles


def _format_tile_array(
    tiles: list[tuple[int, int, int]], items_per_line: int = 5
) -> str:
    """Format tiles as comma-separated {m, n, k} entries with line wrapping."""
    parts = [f"{{{m}, {n}, {k}}}" for m, n, k in tiles]
    lines = []
    for i in range(0, len(parts), items_per_line):
        chunk = parts[i : i + items_per_line]
        lines.append("     " + ", ".join(chunk))
    return ",\n".join(lines)


def update_solution_selection(
    kernels: list[WaveKernel], dry_run: bool, flip_macrotiles: bool = False
) -> bool:
    """Add missing wave macrotile sizes to possibleSwizzleTileSizes and possibleTileSizes."""
    text = SELECTION_PATH.read_text()

    if flip_macrotiles:
        needed_tiles = sorted(set((k.mt_n, k.mt_m, k.mt_k) for k in kernels))
    else:
        needed_tiles = sorted(set((k.mt_m, k.mt_n, k.mt_k) for k in kernels))
    changed = False

    for array_name, count_name in [
        ("possibleSwizzleTileSizes", "possibleSwizzleTileSizesCount"),
        ("possibleTileSizes", "possibleTileSizesCount"),
    ]:
        existing = _parse_tile_array(text, array_name)
        if not existing:
            print(f"  Warning: could not parse {array_name}, skipping", file=sys.stderr)
            continue

        existing_set = set(existing)
        to_add = [t for t in needed_tiles if t not in existing_set]
        if not to_add:
            continue

        merged = existing + to_add
        new_count = len(merged)

        count_pattern = re.compile(
            rf"(constexpr\s+size_t\s+{count_name}\s*=\s*)\d+(\s*;)"
        )
        text = count_pattern.sub(rf"\g<1>{new_count}\g<2>", text)

        array_pattern = re.compile(
            rf"(constexpr\s+std::array<WorkGroupTileSize,\s*){count_name}(>\s*{array_name}\s*=\s*\{{)\s*\{{.*?\}}\}}\}};",
            re.DOTALL,
        )
        formatted = _format_tile_array(merged)
        inner = "{" + formatted + "}};"
        replacement = rf"\g<1>{count_name}\g<2>\n    {inner}"
        text = array_pattern.sub(replacement, text)
        changed = True
        print(f"  {array_name}: added {len(to_add)} tile size(s) -> {new_count} total")

    if not changed:
        print("  solution_selection.cpp: no changes needed")
        return False

    if dry_run:
        print("  [dry-run] Would update solution_selection.cpp")
    else:
        SELECTION_PATH.write_text(text)
        print(f"  Updated {SELECTION_PATH.name}")
    return True


# ---------------------------------------------------------------------------
# Generate hipBLASLt shapes CSV
# ---------------------------------------------------------------------------


def generate_shapes_csv(
    kernels: list[WaveKernel], output_path: Path, dry_run: bool
) -> None:
    """Write a hipBLASLt-convention shapes CSV (tag, m, n, k) for benchmark_shapes.py.

    hipBLASLt m = wave_n, hipBLASLt n = wave_m (due to the A/B + M/N swap).
    """
    rows = []
    seen = set()
    for k in kernels:
        if k.dynamic and k.wave_shapes:
            for ws in k.wave_shapes:
                wm, wn, wk = ws
                hbl_m, hbl_n, hbl_k = wn, wm, wk
                key = (hbl_m, hbl_n, hbl_k)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"tag": "wave_dynamic", "m": hbl_m, "n": hbl_n, "k": hbl_k})
        elif not k.dynamic:
            hbl_m = k.wave_n
            hbl_n = k.wave_m
            hbl_k = k.wave_k
            key = (hbl_m, hbl_n, hbl_k)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"tag": "wave_static", "m": hbl_m, "n": hbl_n, "k": hbl_k})

    if dry_run:
        print(f"  [dry-run] Would write {len(rows)} shapes to {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "m", "n", "k"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} shapes to {output_path}")


# ---------------------------------------------------------------------------
# Build and benchmark
# ---------------------------------------------------------------------------


def build_hipblaslt(jobs: int = 0) -> None:
    cmd = ["ninja", "hipblaslt-bench"]
    if jobs:
        cmd += [f"-j{jobs}"]
    print(f"Building: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(BUILD_DIR))
    print("Build complete.")


def run_benchmark(
    shapes_csv: Path,
    output: Path,
    iters: int = 3,
    num_gpus: int = 1,
    jobs: int = 0,
) -> None:
    cmd = [
        sys.executable,
        str(HIPBLASLT_DIR / "benchmark_shapes.py"),
        "--no-build",
        "--shapes-file",
        str(shapes_csv),
        "--iters",
        str(iters),
        "-o",
        str(output),
        "--num-gpus",
        str(num_gpus),
    ]
    print(f"Benchmarking: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(HIPBLASLT_DIR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrate Wave assembly kernels into hipBLASLt rocroller build.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example workflow:
  # 1. Compile wave kernels (wave repo)
  python benchmark_mxfp4.py --shapes shapes.csv --asm-dir /tmp/wave_asm -o wave_results.csv

  # 2. Integrate, build, and benchmark (hipblaslt repo)
  python integrate_wave_kernels.py --asm-dir /tmp/wave_asm --build --benchmark
""",
    )
    p.add_argument(
        "--asm-dir",
        type=Path,
        required=True,
        help="Directory containing wave kernel .s files (and optional manifest JSON)",
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="Run ninja hipblaslt-bench after integration",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark_shapes.py after build",
    )
    p.add_argument(
        "--benchmark-iters",
        type=int,
        default=3,
        metavar="N",
        help="Iterations per shape for benchmark (default: 3)",
    )
    p.add_argument(
        "--benchmark-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output CSV for benchmark results (default: <asm-dir>/hipblaslt_results.csv)",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="Ninja parallel jobs (default: auto)",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel benchmarking (default: 1)",
    )
    p.add_argument(
        "--flip-macrotiles",
        action="store_true",
        help="Swap M and N macrotile dimensions to match hipBLASLt column-major order",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing any files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asm_dir = args.asm_dir.resolve()
    dry_run = args.dry_run

    if not asm_dir.is_dir():
        print(f"Error: {asm_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # 1. Discover kernels
    kernels = discover_kernels(asm_dir)
    if not kernels:
        print("No wave kernels found. Nothing to do.")
        return

    print(f"\nFound {len(kernels)} wave kernel(s):")
    for k in kernels:
        if k.dynamic:
            n_shapes = len(k.wave_shapes) if k.wave_shapes else 0
            print(f"  {k.name}  macrotile={k.macrotile}  dynamic ({n_shapes} shapes)")
        else:
            print(f"  {k.name}  macrotile={k.macrotile}  shape={k.wave_shape}")

    # 2. Copy assembly files
    print("\nCopying assembly files...")
    copied = copy_asm_files(asm_dir, kernels, dry_run)
    if copied:
        print(f"  {copied} file(s) copied")
    else:
        print("  All files up-to-date")

    # 3. Update CMakeLists.txt
    print("\nUpdating CMakeLists.txt...")
    update_cmake(dry_run)

    # 4. Update custom_kernels.cpp
    print("\nUpdating custom_kernels.cpp...")
    update_cpp(kernels, dry_run, flip_macrotiles=args.flip_macrotiles)

    # 5. Update solution_selection.cpp
    print("\nUpdating solution_selection.cpp...")
    update_solution_selection(kernels, dry_run, flip_macrotiles=args.flip_macrotiles)

    # 6. Generate shapes CSV
    shapes_csv = asm_dir / "wave_hipblaslt_shapes.csv"
    print("\nGenerating hipBLASLt shapes CSV...")
    generate_shapes_csv(kernels, shapes_csv, dry_run)

    if dry_run:
        print("\n[dry-run] No files were modified.")
        return

    # 7. Build
    if args.build:
        print("\nBuilding hipblaslt-bench...")
        build_hipblaslt(args.jobs)

    # 8. Benchmark
    if args.benchmark:
        if not args.build:
            print(
                "Warning: --benchmark without --build; assuming build is up-to-date",
                file=sys.stderr,
            )
        benchmark_output = args.benchmark_output or (asm_dir / "hipblaslt_results.csv")
        print(f"\nRunning benchmark -> {benchmark_output}")
        run_benchmark(
            shapes_csv,
            benchmark_output,
            iters=args.benchmark_iters,
            num_gpus=args.num_gpus,
            jobs=args.jobs,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
