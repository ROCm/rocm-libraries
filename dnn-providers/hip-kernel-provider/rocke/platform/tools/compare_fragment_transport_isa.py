#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compile the same fragment consumers from two source trees and compare ISA.

This is a source-checkout codegen gate, not a numeric or timing benchmark.
Both arms run in isolated Python processes and use one explicit compiler,
LLVM flavor, target, and optimization level. Artifact and source hashes make
baseline selection replayable even when a source tree has uncommitted edits.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

_EMITTER = """
import json, sys
from rocke.core.lower_llvm import lower_kernel_to_llvm
case = json.loads(sys.argv[1])
if case['kind'] == 'gfx950':
    from rocke.instances.gfx950.block_scaled_gemm import BlockScaledGemmSpec, build_block_scaled_gemm
    spec = BlockScaledGemmSpec('transport_isa', case.get('m', 32), case.get('n', 32), case['k'], case['a'], case['b'], 'bf16')
    kernel = build_block_scaled_gemm(spec)
elif case['kind'] == 'gfx1250':
    from rocke.instances.gfx1250.block_scaled_gemm import BlockScaledGemmSpec, build_block_scaled_gemm
    spec = BlockScaledGemmSpec('transport_isa', case.get('m', 16), case.get('n', 16), case['k'], dtype_a=case['a'], dtype_b=case['b'],
        matrix_path=case['path'], scale_dtype=case['scale'], block_k=case['block_k'],
        tensor_scale=case.get('tensor_scale', False))
    kernel = build_block_scaled_gemm(spec)
else:
    from rocke.instances.gfx942.tf32_mma_probe import Tf32MmaProbeSpec, build_tf32_mma_probe
    kernel = build_tf32_mma_probe(Tf32MmaProbeSpec(case['m'], case['mode']))
if len(sys.argv) > 3 and sys.argv[3] == 'serialized':
    from rocke.core.ir_serialize import serialize
    print(serialize(kernel), end='')
else:
    print(lower_kernel_to_llvm(kernel, arch=case['target'], llvm_flavor=sys.argv[2]), end='')
"""
_OPCODE = re.compile(
    r"^[ \t]*((?:v_|s_|global_|flat_|buffer_|scratch_|ds_|tbuffer_)[A-Za-z0-9_]+)\b(?!:)",
    re.MULTILINE,
)
_RESOURCE = re.compile(
    r"^\s*\.amdhsa_(next_free_vgpr|next_free_sgpr|accum_offset|private_segment_fixed_size|group_segment_fixed_size)\s+(\d+)",
    re.MULTILINE,
)


def opcode_counts(assembly: str) -> dict[str, int]:
    """Instruction census of compiler assembly, excluding labels/directives/comments."""
    return dict(sorted(Counter(_OPCODE.findall(assembly)).items()))


def resource_metadata(assembly: str) -> dict[str, int]:
    """Declared register/scratch/LDS metadata; absent fields remain absent."""
    return {name: int(value) for name, value in _RESOURCE.findall(assembly)}


def compare_assembly(baseline: str, candidate: str) -> dict:
    """Report static differences; matrix-op disagreement is a semantic codegen failure."""
    before, after = opcode_counts(baseline), opcode_counts(candidate)
    delta = {
        op: after.get(op, 0) - before.get(op, 0)
        for op in sorted(before.keys() | after.keys())
        if after.get(op, 0) != before.get(op, 0)
    }
    matrix_before = {
        op: n for op, n in before.items() if op.startswith(("v_mfma", "v_wmma"))
    }
    matrix_after = {
        op: n for op, n in after.items() if op.startswith(("v_mfma", "v_wmma"))
    }
    if not matrix_before or matrix_before != matrix_after:
        status = "matrix-mismatch"
    elif delta or resource_metadata(baseline) != resource_metadata(candidate):
        status = "review-required"
    else:
        status = "unchanged-census"
    return {
        "status": status,
        "opcode_delta": delta,
        "baseline_opcodes": before,
        "candidate_opcodes": after,
        "baseline_resources": resource_metadata(baseline),
        "candidate_resources": resource_metadata(candidate),
        "caveat": "An unchanged census does not establish identical scheduling, numerics, or speed.",
    }


def cases() -> list[dict]:
    """Representative width, scale, mixed-format, iteration, and TF32 preparation axes."""
    result = []
    formats = ("fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1")
    pairs = [(dtype, dtype) for dtype in formats] + [
        (formats[0], formats[-1]),
        (formats[-1], formats[0]),
    ]
    for a, b in pairs:
        for steps in (1, 4):
            result.append(dict(kind="gfx950", target="gfx950", a=a, b=b, k=64 * steps))
            for path, block_k in (("wmma_scale", 32), ("wmma_scale16", 16)):
                result.append(
                    dict(
                        kind="gfx1250",
                        target="gfx1250",
                        a=a,
                        b=b,
                        k=128 * steps,
                        path=path,
                        block_k=block_k,
                        scale="e8m0",
                    )
                )
    result.append(
        dict(
            kind="gfx1250",
            target="gfx1250",
            a="fp4e2m1",
            b="fp4e2m1",
            k=256,
            path="wmma_scale16",
            block_k=16,
            scale="e4m3",
            tensor_scale=True,
        )
    )
    for a in formats[:2]:
        for b in formats[:2]:
            for steps in (1, 4):
                for scale in ("fp16", "fp32"):
                    result.append(
                        dict(
                            kind="gfx1250",
                            target="gfx1250",
                            a=a,
                            b=b,
                            k=64 * steps,
                            path="wmma",
                            block_k=64,
                            scale=scale,
                        )
                    )
    for m in (16, 32):
        for mode in ("raw", "carrier", "rne", "prepacked", "fp32"):
            result.append(dict(kind="gfx942", target="gfx942", m=m, mode=mode))
    return result


def source_digest(root: Path) -> str:
    """Hash source/metadata content, not branch names or generated Python caches."""
    digest = hashlib.sha256()
    for subtree in (root / "python" / "rocke", root / "cpp"):
        for path in sorted(subtree.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".h", ".json"):
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def compile_arm(
    root: Path, case: dict, flavor: str, compiler: str, output: Path
) -> tuple[str, dict]:
    """Isolated source emission followed by explicit optimized clang assembly."""
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "python"),
        ROCKE_BACKEND="python",
        ROCKE_LLVM_FLAVOR=flavor,
        PYTHONDONTWRITEBYTECODE="1",
    )
    ir = subprocess.run(
        [sys.executable, "-c", _EMITTER, json.dumps(case), flavor],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    ir_path, assembly_path = output / "kernel.ll", output / "kernel.s"
    ir_path.write_text(ir, encoding="utf-8")
    command = [
        compiler,
        "-O3",
        "-x",
        "ir",
        "--target=amdgcn-amd-amdhsa",
        f"-mcpu={case['target']}",
        "-nogpulib",
        "-S",
        str(ir_path),
        "-o",
        str(assembly_path),
    ]
    proc = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=120
    )
    (output / "compiler.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(
            f"compiler failed ({proc.returncode}): {output / 'compiler.log'}"
        )
    assembly = assembly_path.read_text(encoding="utf-8")
    return assembly, {
        "command": command,
        "ir_sha256": hashlib.sha256(ir.encode()).hexdigest(),
        "assembly_sha256": hashlib.sha256(assembly.encode()).hexdigest(),
    }


def compiler_driver_path(path: str) -> str:
    """Make the selected driver absolute without changing multicall-driver argv[0]."""
    return os.path.abspath(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True, help="baseline platform source root"
    )
    parser.add_argument(
        "--candidate", type=Path, required=True, help="candidate platform source root"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clang", default=shutil.which("amdclang++"))
    parser.add_argument(
        "--flavor", default="llvm23", choices=("llvm20", "llvm22", "llvm23", "llvm24")
    )
    parser.add_argument(
        "--target", action="append", choices=("gfx942", "gfx950", "gfx1250")
    )
    args = parser.parse_args()
    if not args.clang:
        parser.error("an AMDGPU-capable clang is required; pass --clang explicitly")
    compiler = compiler_driver_path(args.clang)
    manifest = {
        "compiler": compiler,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True),
        "flavor": args.flavor,
        "baseline_source_sha256": source_digest(args.baseline),
        "candidate_source_sha256": source_digest(args.candidate),
        "cases": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for index, case in enumerate(cases()):
        if args.target and case["target"] not in args.target:
            continue
        directory = args.output / f"case-{index:03d}"
        try:
            before, old_info = compile_arm(
                args.baseline, case, args.flavor, compiler, directory / "baseline"
            )
            after, new_info = compile_arm(
                args.candidate, case, args.flavor, compiler, directory / "candidate"
            )
            report = dict(
                case=case,
                baseline=old_info,
                candidate=new_info,
                **compare_assembly(before, after),
            )
            failed |= report["status"] == "matrix-mismatch"
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            report = dict(case=case, status="failed", error=str(error))
            failed = True
        manifest["cases"].append(report)
        (args.output / "comparison.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"case-{index:03d} {case['target']}: {report['status']}", flush=True)
    if not manifest["cases"]:
        parser.error("no cases selected")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
