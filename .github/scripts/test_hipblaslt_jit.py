#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run the source JIT regressions on a real GPU of the expected architecture."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("gfx90a", "gfx942", "gfx950", "gfx1250"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        choices=(
            "streamk-api",
            "amax-api",
            "alpha-zero-api",
            "alternate-backend",
            "process-runner",
            "artifact-loader",
            "splitk-api",
            "bundle-failures",
            "helper-failures",
            "disabled-api",
        ),
        help="Run only the selected regression routes (default: all)",
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    build = args.build.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tensile = source / "projects/hipblaslt/tensilelite"
    fixtures = tensile / "Tensile/Tests/unit/test_data"
    sample = build / "clients/staging/hipblaslt-jit-api-test"
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(("HIPBLASLT_JIT_", "TENSILE_STREAMK_")):
            env.pop(key)
    env["PYTHONPATH"] = os.pathsep.join(
        map(str, (build / "tensilelite/rocisa", build / "tensilelite", tensile))
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TENSILE_DISABLE_HELPER_CACHE"] = "1"
    empty = output / "empty-device-library"
    empty.mkdir()
    env["HIPBLASLT_TENSILE_LIBPATH"] = str(empty)
    compiler = str(Path(env.get("ROCM_PATH", "/opt/rocm")) / "bin/amdclang++")

    # The SDK supplies dependencies; all project code must come from this checkout.
    linkage = subprocess.run(
        ["ldd", str(sample)], env=env, text=True, capture_output=True, check=True
    )
    (output / "sample-ldd.txt").write_text(linkage.stdout)
    libraries = re.findall(
        r"(lib(?:hipblaslt|tensilelite)[^\s]*) => (\S+)", linkage.stdout, re.I
    )
    if not any("hipblaslt" in name for name, _ in libraries):
        raise RuntimeError("No dynamic local hipBLASLt linkage")
    for name, path in libraries:
        if not Path(path).resolve().is_relative_to(build):
            raise RuntimeError(f"Installed project library: {name}: {path}")
    provenance = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, Tensile, rocisa; from rocisa import _rocisa; "
            "print(json.dumps([Tensile.__file__, rocisa.__file__, _rocisa.__file__]))",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    paths = [
        Path(path).resolve() for path in json.loads(provenance.stdout.splitlines()[-1])
    ]
    if not paths[0].is_relative_to(tensile) or any(
        not p.is_relative_to(build) for p in paths[1:]
    ):
        raise RuntimeError(
            f"Python project imports do not come from the checkout/build: {paths}"
        )
    (output / "python-provenance.json").write_text(
        json.dumps(list(map(str, paths)), indent=2)
    )

    staging = build / "clients/staging"
    commands = [
        (
            "process-runner",
            [
                str(staging / "hipblaslt-jit-process-test"),
                "--run-tests",
                str(output / "process"),
            ],
            {},
            60,
        ),
        (
            "artifact-loader",
            [str(staging / "hipblaslt-jit-artifacts-test"), str(output / "artifacts")],
            {},
            60,
        ),
    ]
    if not args.case or "alternate-backend" in args.case:
        code_object = output / "alternate-backend.hsaco"
        compile_command = [
            str(Path(compiler).parent / "hipcc"),
            "--genco",
            "--offload-arch=" + args.architecture,
            str(
                source
                / "projects/hipblaslt/clients/tests/jit/alternate_backend_kernels.hip"
            ),
            "-o",
            str(code_object),
        ]
        with (output / "alternate-backend-build.log").open("w") as log:
            subprocess.run(
                compile_command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        commands.append(
            (
                "alternate-backend",
                [str(staging / "hipblaslt-jit-backend-test"), str(code_object)],
                {},
                120,
            )
        )

    fixture_suffix = "_gfx1250" if args.architecture == "gfx1250" else ""
    # These are the public C hipblasLtMatmul and C++ Gemm routes.
    # They use the same generic JIT selection API as any application.
    for feature, fixture, options in [
        ("streamk", "streamk", ["--k", "4096", "--workspace-fallback", "1"]),
        ("amax", "amax", ["--amax", "1"]),
        ("alpha-zero", "amax", ["--amax", "1", "--alpha-zero", "1"]),
    ]:
        name = f"{feature}-api"
        command = [
            str(sample),
            sys.executable,
            str(tensile),
            env["PYTHONPATH"],
            str(fixtures / f"single_solution_{fixture}{fixture_suffix}.yaml"),
            str(output / name),
            args.architecture,
            compiler,
            *options,
        ]
        streamk = (
            {
                "TENSILE_STREAMK_FIXED_GRID": "16",
                "TENSILE_STREAMK_DYNAMIC_GRID": "0",
                "TENSILE_DB": "64",
            }
            if feature == "streamk"
            else {}
        )
        commands.append((name, command, streamk, 420))

    if not args.case or any(
        case in args.case
        for case in ("splitk-api", "helper-failures", "bundle-failures")
    ):
        commands.append(
            (
                "splitk-api",
                [
                    str(sample),
                    sys.executable,
                    str(tensile),
                    env["PYTHONPATH"],
                    str(fixtures / f"single_solution_splitk{fixture_suffix}.yaml"),
                    str(output / "splitk-api"),
                    args.architecture,
                    compiler,
                    "--k",
                    "512",
                ],
                {},
                420,
            )
        )
        commands.append(
            (
                "helper-failures",
                [
                    sys.executable,
                    str(
                        source
                        / "projects/hipblaslt/clients/tests/jit/test_helper_failures.py"
                    ),
                    str(sample),
                    str(output / "splitk-api/bundle"),
                    str(output / "helper-failures"),
                ],
                {},
                420,
            )
        )

        commands.append(
            (
                "bundle-failures",
                [
                    sys.executable,
                    str(
                        source
                        / "projects/hipblaslt/clients/tests/jit/test_bundle_failures.py"
                    ),
                    str(sample),
                    str(output / "splitk-api/bundle"),
                    str(output / "bundle-failures"),
                    "--architecture",
                    args.architecture,
                ],
                {},
                420,
            )
        )

    results = []
    for name, command, overrides, timeout in commands:
        if (
            args.case
            and name not in args.case
            and not (
                name == "splitk-api"
                and any(
                    case in args.case for case in ("helper-failures", "bundle-failures")
                )
            )
        ):
            continue
        print(f"RUN {name} on native {args.architecture}", flush=True)
        (output / f"{name}-command.json").write_text(json.dumps(command, indent=2))
        with (output / f"{name}-run.log").open("w") as log:
            try:
                with subprocess.Popen(
                    command,
                    env=dict(env, **overrides),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process:
                    try:
                        status = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        status = "timeout"
            except OSError as error:
                status = str(error)
        results.append(dict(case=name, architecture=args.architecture, status=status))
        (output / "summary.json").write_text(json.dumps(results, indent=2))
        print(f"{'PASS' if status == 0 else 'FAIL'} {name}: {status}", flush=True)
    if not args.case or "disabled-api" in args.case:
        # Reuse this build directory; the disabled test links the rebuilt library.
        with (output / "disabled-api-build.log").open("w") as log:
            subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(source / "projects/hipblaslt"),
                    "-B",
                    str(build),
                    "-DHIPBLASLT_ENABLE_JIT=OFF",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
            subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(build),
                    "--parallel",
                    "8",
                    "--target",
                    "hipblaslt-jit-disabled-test",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
            status = subprocess.run(
                [str(staging / "hipblaslt-jit-disabled-test")],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
            ).returncode
        results.append(
            dict(case="disabled-api", architecture=args.architecture, status=status)
        )
        (output / "summary.json").write_text(json.dumps(results, indent=2))
        print(f"{'PASS' if status == 0 else 'FAIL'} disabled-api: {status}", flush=True)
    return int(any(row["status"] != 0 for row in results))


if __name__ == "__main__":
    sys.exit(main())
