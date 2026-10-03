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
# The wave size and an instruction each architecture's FP16 kernels use.
NATIVE = {"gfx942": (64, "v_mfma_f32_"), "gfx1250": (32, "v_wmma_f32_16x16x32_f16")}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def generate_and_build(code_object_test, request, cxx_compiler, output, architecture):
    """Generate the request's ranked bundles and build each with comgr; their manifests.

    JitGemm writes fewer bundles than requested when it rejects candidates.
    """
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
                architecture,
                "--cxx-compiler",
                cxx_compiler,
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

    built = len(list(generated.glob("bundle-*")))
    wavefront, instruction = NATIVE[architecture]
    command = [
        str(code_object_test),
        "--target",
        architecture,
        "--out",
        str(output / "code-object"),
    ]
    kernels, manifests = set(), []
    for rank in range(built):
        bundle = generated / f"bundle-{rank}"
        manifest = json.loads((bundle / "manifest.json").read_text())
        resolved = manifest["jit_prediction"]["resolved_parameters"]
        require(
            manifest["architecture"]["resolved"] == architecture
            and resolved["WavefrontSize"] == wavefront,
            f"bundle-{rank} is not an {architecture} wave{wavefront} solution",
        )
        (assembly,) = (bundle / "sources").glob("*.s")
        require(
            instruction in assembly.read_text(),
            f"bundle-{rank} does not use {instruction}",
        )
        kernels.add(manifest["main_kernel"]["name"])
        manifests.append(manifest)
        # The code-object test names each bundle after its parent directory.
        named = output / "bundles" / f"rank-{rank}"
        named.mkdir(parents=True)
        (named / "bundle").symlink_to(bundle)
        command += ["--bundle", str(named / "bundle")]
        command += [arg for test in TESTS for arg in ("--only", f"{test}[rank-{rank}]")]
    require(
        len(kernels) == built,
        f"The {built} bundles do not hold distinct kernels",
    )

    with (output / "code-object.log").open("w") as log:
        status = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=600
        ).returncode
    require(status == 0, f"The comgr build exited with {status}; see {log.name}")
    passed = (output / "code-object.log").read_text().count("\nPASS ")
    require(
        passed == len(TESTS) * built,
        f"Expected {len(TESTS) * built} comgr builds to pass, got {passed}",
    )
    return manifests


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
    manifests = generate_and_build(
        args.code_object_test, request, args.cxx_compiler, output, "gfx1250")
    requested = json.loads(request.read_text())["requested_solutions"]
    require(len(manifests) == requested, f"{len(manifests)} of {requested} solutions generated")
    print(
        f"PASS jit-gemm-gfx1250: {len(manifests)} ranked solutions generated and built for gfx1250"
    )


if __name__ == "__main__":
    main()
