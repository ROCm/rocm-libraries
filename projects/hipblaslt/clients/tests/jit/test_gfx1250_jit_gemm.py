# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate and build heuristic JIT GEMM solutions for gfx1250 without a gfx1250 GPU.

Runs Tensile.JitGemm on a heuristic request with the arguments hipBLASLt passes
(source only, code object version 4, msgpack library), then builds every ranked
bundle with comgr through the code-object test: it assembles the main kernel,
compiles the helpers and links both into one code object. Nothing is executed.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

TESTS = ("a_tensile_assemble", "c_tensile_helper_compile", "g_tensile_mixed_link")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code_object_test", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("cxx_compiler")
    parser.add_argument("fresh_output", type=Path)
    args = parser.parse_args()
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    request = args.request.resolve(strict=True)
    generated = output / "generated"
    with (output / "generate.log").open("w") as log:
        status = subprocess.run(
            [
                sys.executable,
                "-m",
                "Tensile.JitGemm",
                str(request),
                str(generated),
                "--architecture",
                "gfx1250",
                "--cxx-compiler",
                args.cxx_compiler,
                "--source-only",
                "--code-object-version",
                "4",
                "--library-format",
                "msgpack",
            ],
            cwd=output,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        ).returncode
    require(status == 0, f"Tensile.JitGemm exited with {status}; see {log.name}")

    requested = json.loads(request.read_text())["requested_solutions"]
    command = [
        str(args.code_object_test),
        "--target",
        "gfx1250",
        "--out",
        str(output / "code-object"),
    ]
    kernels = set()
    for rank in range(requested):
        bundle = generated / f"bundle-{rank}"
        manifest = json.loads((bundle / "manifest.json").read_text())
        resolved = manifest["jit_prediction"]["resolved_parameters"]
        require(
            manifest["architecture"]["resolved"] == "gfx1250"
            and resolved["WavefrontSize"] == 32,
            f"bundle-{rank} is not a gfx1250 wave32 solution",
        )
        (assembly,) = (bundle / "sources").glob("*.s")
        require(
            "v_wmma_f32_16x16x32_f16" in assembly.read_text(),
            f"bundle-{rank} does not use the gfx1250 WMMA instruction",
        )
        kernels.add(manifest["main_kernel"]["name"])
        # The code-object test names each bundle after its parent directory.
        named = output / "bundles" / f"rank-{rank}"
        named.mkdir(parents=True)
        (named / "bundle").symlink_to(bundle)
        command += ["--bundle", str(named / "bundle")]
        command += [arg for test in TESTS for arg in ("--only", f"{test}[rank-{rank}]")]
    require(
        len(kernels) == requested,
        f"The {requested} bundles do not hold distinct kernels",
    )

    with (output / "code-object.log").open("w") as log:
        status = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=600
        ).returncode
    require(status == 0, f"The comgr build exited with {status}; see {log.name}")
    passed = (output / "code-object.log").read_text().count("\nPASS ")
    require(
        passed == len(TESTS) * requested,
        f"Expected {len(TESTS) * requested} comgr builds to pass, got {passed}",
    )
    print(
        f"PASS jit-gemm-gfx1250: {requested} ranked solutions generated and built for gfx1250"
    )


if __name__ == "__main__":
    main()
