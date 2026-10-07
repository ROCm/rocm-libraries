#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile one Tile Engine gemm_universal instance for a GPU target without a
GPU or a CMake build tree, keep the ISA and print an isa_report.py summary.

The instance is either an existing single-instance header (``--header``, as
written by ``gemm_universal_instance_builder.py --gen_single`` or by the CMake
build) or is generated here from ``--datatype/--layout/--tile-config/
--trait-combo``. The compile mirrors the device-relevant flags of the CMake
benchmark target (definitions from the top-level CMakeLists.txt for the target
architecture plus the -mllvm options it adds), and by default only the device
pass is run (``--cuda-device-only``, about 30-60 s per instance).

Example (no hardcoded paths; ROCm from --rocm-path or $ROCM_PATH):

  python3 compile_one.py --rocm-path /opt/rocm --arch gfx1250 \\
      --datatype bf16 --layout rcr --tile-config 256x256x64_2x4x1_16x16x32 \\
      --trait-combo comp_tdm_tdm_intrawave_False_False_False_False \\
      --out-dir /tmp/te_isa

Outputs in --out-dir: ``build.log`` (with -Rpass-analysis=kernel-resource-usage
remarks), the ``-save-temps`` files including ``*-<arch>.s``, ``cmd.txt`` and
``isa_report.txt`` / ``isa_report.json``.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_CK_ROOT = _HERE.parents[2]
_VARIANT = "gemm_universal"

DATATYPE_DEFINES = [
    "CK_ENABLE_INT8",
    "CK_ENABLE_FP8",
    "CK_ENABLE_BF8",
    "CK_ENABLE_FP16",
    "CK_ENABLE_FP32",
    "CK_ENABLE_FP64",
    "CK_ENABLE_BF16",
]

# Options the top-level CMakeLists.txt adds for current ROCm compilers.
CMAKE_COMPILE_OPTIONS = [
    "-fbracket-depth=1024",
    "-fno-offload-uniform-block",
    "-mllvm",
    "--lsr-drop-solution=1",
    "-mllvm",
    "-enable-post-misched=0",
    "-mllvm",
    "-amdgpu-early-inline-all=true",
    "-mllvm",
    "-amdgpu-function-calls=false",
]


# Generic and variant targets whose id is not the digits after "gfx"
# (_ck_gpu_target_string_to_id in the top-level CMakeLists.txt).
_SPECIAL_TARGET_IDS = {
    "gfx10-3-generic": "0x103F",
    "gfx11-generic": "0x11FF",
    "gfx12-generic": "0x12FF",
    "gfx1250-strict": "0x1250",
}


def base_arch(arch):
    """Processor name without target features, e.g. gfx90a:xnack- -> gfx90a."""
    return arch.split(":", 1)[0].lower()


def target_id(arch):
    """CK_CMAKE_GPU_TARGET_IDS value, e.g. gfx942 -> 0x0942, gfx90a -> 0x090A.
    Target features are ignored; unknown generic targets are rejected."""
    name = base_arch(arch)
    if name in _SPECIAL_TARGET_IDS:
        return _SPECIAL_TARGET_IDS[name]
    if not re.fullmatch(r"gfx[0-9a-f]{3,4}", name):
        raise ValueError(f"unsupported GPU target '{arch}'")
    return "0x" + name[len("gfx") :].upper().rjust(4, "0")


def arch_defines(arch):
    """Per-architecture definitions, following the SUPPORTED_GPU_TARGETS
    checks in the top-level CMakeLists.txt for a single-target build with
    the default DTYPES (all types, so CK_ENABLE_TF32 on gfx942/gfx95x)."""
    arch = base_arch(arch)
    d = []
    if arch.startswith(("gfx9", "gfx11", "gfx12")):
        d.append("CK_USE_XDL")
    if arch.startswith(("gfx94", "gfx95")):
        d.append("CK_USE_GFX94")
    if arch.startswith(("gfx942", "gfx95")):
        d.append("CK_ENABLE_TF32")
    if arch.startswith("gfx950"):
        d += ["CK_USE_GFX950", "CK_USE_NATIVE_MX_SUPPORT", "CK_GFX950_SUPPORT"]
    if arch.startswith("gfx10"):
        d.append("CK_GFX1030_SUPPORT")
    wmma = arch.startswith(("gfx11", "gfx12"))
    if wmma:
        d.append("CK_USE_WMMA")
    d.append(f"CK_TILE_USE_WMMA={1 if wmma else 0}")
    if arch.startswith("gfx12"):
        d.append("CK_USE_WMMA_FP8")
    if arch.startswith(("gfx12", "gfx950")):
        d += ["CK_USE_OCP_FP8", "CK_TILE_USE_OCP_FP8"]
    if arch.startswith(("gfx90a", "gfx94")):
        d.append("CK_USE_FNUZ_FP8")
    if arch.startswith("gfx1250"):
        d += [
            "CK_USE_GFX1250",
            "CK_USE_NATIVE_MX_SUPPORT",
            "CK_GFX1250_SUPPORT",
            "USE_NEW_UNIFIED_FRAMEWORK=0",
        ]
    if arch.startswith("gfx12"):
        d.append("CK_GFX12_SUPPORT")
    d.append(f"CK_CMAKE_GPU_TARGET_IDS={target_id(arch)}")
    return list(dict.fromkeys(d))


def default_config_json(ck_root, arch):
    gemm_dir = ck_root / "tile_engine" / "ops" / "gemm"
    per_arch = gemm_dir / _VARIANT / "configs" / f"default_config_{arch}.json"
    return (
        per_arch if per_arch.is_file() else gemm_dir / "configs" / "default_config.json"
    )


def generate_header(args, ck_root, gen_dir):
    """Run the instance builder in --gen_single mode; return the header path."""
    builder = (
        ck_root
        / "tile_engine"
        / "ops"
        / "gemm"
        / _VARIANT
        / f"{_VARIANT}_instance_builder.py"
    )
    config = (
        Path(args.config_json)
        if args.config_json
        else default_config_json(ck_root, base_arch(args.arch))
    )
    gen_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(builder),
        "--working_path",
        str(gen_dir),
        "--gpu_target",
        base_arch(args.arch),
        "--datatype",
        args.datatype,
        "--layout",
        args.layout,
        "--config_json",
        str(config),
        "--gen_single",
        "--kernel_name",
        "compile_one",
        "--tile_config",
        args.tile_config,
        "--trait_combo",
        args.trait_combo,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("Generated "):
            return Path(line[len("Generated ") :].strip())
    sys.stderr.write(proc.stdout + proc.stderr)
    sys.exit("compile_one: instance generation failed (see output above)")


def compile_command(args, rocm, ck_root, header, out_dir):
    variant_dir = ck_root / "tile_engine" / "ops" / "gemm" / _VARIANT
    source = variant_dir / f"{_VARIANT}_benchmark_single.cpp"
    cmd = [
        str(rocm / "lib" / "llvm" / "bin" / "clang++"),
        "-x",
        "hip",
        f"--offload-arch={args.arch}",
        f"--rocm-path={rocm}",
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-DCK_TIME_KERNEL=1",
    ]
    cmd += [f"-D{d}" for d in DATATYPE_DEFINES + arch_defines(args.arch)]
    cmd += [f'-DGEMM_UNIVERSAL_SINGLE_INSTANCE_HPP="{header}"']
    cmd += CMAKE_COMPILE_OPTIONS
    cmd += [
        f"-I{ck_root / 'include'}",
        f"-I{ck_root / 'library' / 'include'}",
        f"-I{ck_root / 'tile_engine' / 'include'}",
        f"-I{ck_root / 'tile_engine' / 'ops'}",
        f"-I{variant_dir}",
        f"-I{header.parent}",
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        "-include",
        str(header),
        "--save-temps=obj",
        "-Rpass-analysis=kernel-resource-usage",
    ]
    cmd += args.extra_flag or []
    if args.full:
        cmd += [str(source), "-o", str(out_dir / "bench.exe")]
    else:
        cmd += ["--cuda-device-only", "-c", str(source), "-o", str(out_dir / "dev.o")]
    return cmd


def tile_of(header_or_config):
    """Pull the MxNxK_WMxWNxWK_wmxwnxwk tile string out of an instance name."""
    m = re.search(r"(\d+x\d+x\d+_\d+x\d+x\d+_\d+x\d+x\d+)", str(header_or_config))
    return m.group(1) if m else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True, help="output / scratch directory")
    parser.add_argument(
        "--rocm-path",
        default=os.environ.get("ROCM_PATH", "/opt/rocm"),
        help="ROCm root with lib/llvm/bin/clang++ (default: $ROCM_PATH or /opt/rocm)",
    )
    parser.add_argument(
        "--ck-root",
        default=str(_DEFAULT_CK_ROOT),
        help="composablekernel root (default: the tree this script lives in)",
    )
    parser.add_argument(
        "--arch", default="gfx1250", help="offload arch (default gfx1250)"
    )
    parser.add_argument("--header", help="existing single-instance header")
    parser.add_argument("--datatype", help="fp16/bf16/fp8/bf8 (generate mode)")
    parser.add_argument("--layout", help="rcr/rrr/crr/ccr (generate mode)")
    parser.add_argument("--tile-config", help="e.g. 256x256x64_2x4x1_16x16x32")
    parser.add_argument(
        "--trait-combo", help="e.g. comp_tdm_tdm_intrawave_False_False_False_False"
    )
    parser.add_argument(
        "--config-json", help="builder config (default: per-arch default)"
    )
    parser.add_argument(
        "--full", action="store_true", help="also compile the host side and link"
    )
    parser.add_argument(
        "--extra-flag", action="append", help="extra compiler flag (repeatable)"
    )
    parser.add_argument("--no-report", action="store_true", help="skip isa_report.py")
    parser.add_argument("--dry-run", action="store_true", help="print the command only")
    args = parser.parse_args(argv)

    rocm = Path(args.rocm_path).resolve()
    ck_root = Path(args.ck_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.header:
        header = Path(args.header).resolve()
    else:
        missing = [
            n
            for n in ("datatype", "layout", "tile_config", "trait_combo")
            if not getattr(args, n)
        ]
        if missing:
            parser.error(
                "need --header or all of " + ", ".join("--" + m for m in missing)
            )
        header = generate_header(args, ck_root, out_dir / "gen")

    cmd = compile_command(args, rocm, ck_root, header, out_dir)
    (out_dir / "cmd.txt").write_text(shlex.join(cmd) + "\n")
    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    start = time.time()
    with open(out_dir / "build.log", "w") as log:
        rc = subprocess.run(
            cmd, cwd=out_dir, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    print(
        f"compile_one: rc={rc} in {time.time() - start:.0f}s, log {out_dir / 'build.log'}"
    )
    if rc:
        return rc

    # clang names the device asm after the full target id (with any
    # :feature suffixes), so match the processor name with or without them.
    arch = base_arch(args.arch)
    asm = sorted(set(out_dir.glob(f"*-{arch}.s")) | set(out_dir.glob(f"*-{arch}:*.s")))
    if not asm:
        print(f"compile_one: no *-{arch}.s in {out_dir}", file=sys.stderr)
        return 1
    print(f"compile_one: isa {asm[0]}")
    if args.no_report:
        return 0

    report = [
        sys.executable,
        str(_HERE / "isa_report.py"),
        str(asm[0]),
        "--kernel",
        "GemmKernel",
        "--remarks",
        str(out_dir / "build.log"),
    ]
    tile = args.tile_config or tile_of(header.name)
    if tile:
        report += ["--tile", tile]
    text = subprocess.run(report, capture_output=True, text=True)
    as_json = subprocess.run(report + ["--json"], capture_output=True, text=True)
    (out_dir / "isa_report.txt").write_text(text.stdout + text.stderr)
    (out_dir / "isa_report.json").write_text(as_json.stdout)
    sys.stdout.write(text.stdout + text.stderr)
    return text.returncode


if __name__ == "__main__":
    sys.exit(main())
