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
    parser.add_argument("--architecture", choices=("gfx90a", "gfx942", "gfx950"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=(
        "automatic-bench", "streamk-standalone", "streamk-normal", "amax-standalone", "amax-normal"),
        help="Run only the selected regression routes (default: all)")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    build = args.build.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tensile = source / "projects/hipblaslt/tensilelite"
    fixtures = tensile / "Tensile/Tests/unit/test_data"
    bench = build / "clients/hipblaslt-bench"
    sample = build / "clients/staging/hipblaslt-jit-gemm"
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(("HIPBLASLT_JIT_", "TENSILE_STREAMK_")):
            env.pop(key)
    env["PYTHONPATH"] = os.pathsep.join(map(str, (
        build / "tensilelite/rocisa", build / "tensilelite", tensile)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TENSILE_DISABLE_HELPER_CACHE"] = "1"
    empty = output / "empty-device-library"
    empty.mkdir()
    env["HIPBLASLT_TENSILE_LIBPATH"] = str(empty)
    compiler = str(Path(env.get("ROCM_PATH", "/opt/rocm")) / "bin/amdclang++")

    # The SDK supplies dependencies; all project code must come from this checkout.
    linkage = subprocess.run(["ldd", str(sample)], env=env, text=True,
                             capture_output=True, check=True)
    (output / "sample-ldd.txt").write_text(linkage.stdout)
    libraries = re.findall(r"(lib(?:hipblaslt|tensilelite)[^\s]*) => (\S+)", linkage.stdout, re.I)
    if not any("hipblaslt" in name for name, _ in libraries):
        raise RuntimeError("No dynamic local hipBLASLt linkage")
    for name, path in libraries:
        if not Path(path).resolve().is_relative_to(build):
            raise RuntimeError(f"Installed project library: {name}: {path}")
    provenance = subprocess.run(
        [sys.executable, "-c",
         "import json, Tensile, rocisa; from rocisa import _rocisa; "
         "print(json.dumps([Tensile.__file__, rocisa.__file__, _rocisa.__file__]))"],
        env=env, text=True, capture_output=True, check=True)
    paths = [Path(path).resolve() for path in json.loads(provenance.stdout.splitlines()[-1])]
    if not paths[0].is_relative_to(tensile) or any(not p.is_relative_to(build) for p in paths[1:]):
        raise RuntimeError(f"Python project imports do not come from the checkout/build: {paths}")
    (output / "python-provenance.json").write_text(json.dumps(list(map(str, paths)), indent=2))

    commands = [("automatic-bench", [sys.executable,
        str(source / "projects/hipblaslt/clients/bench/test_jit_gemm.py"),
        "--bench", str(bench), "--build-root", str(build), "--python", sys.executable,
        "--architecture", args.architecture, "--output", str(output / "bench"),
        "--timeout", "420"], {}, 1800)]
    for feature, options in [("streamk", ["--k", "4096"]), ("amax", ["--amax", "1"])]:
        for route in ("standalone", "normal"):
            name = f"{feature}-{route}"
            command = [str(sample), sys.executable, str(tensile), env["PYTHONPATH"],
                       str(fixtures / f"single_solution_{feature}.yaml"),
                       str(output / name), args.architecture, compiler, *options]
            if route == "normal":
                command += ["--normal-api", "both"]
                if feature == "streamk":
                    command += ["--workspace-fallback", "1"]
            else:
                command += ["--min-workspace", "1"]
            streamk = {"TENSILE_STREAMK_FIXED_GRID": "16", "TENSILE_STREAMK_DYNAMIC_GRID": "0",
                       "TENSILE_DB": "64"} if feature == "streamk" else {}
            commands.append((name, command, streamk, 420))

    results = []
    for name, command, overrides, timeout in commands:
        if args.case and name not in args.case:
            continue
        print(f"RUN {name} on native {args.architecture}", flush=True)
        (output / f"{name}-command.json").write_text(json.dumps(command, indent=2))
        with (output / f"{name}-run.log").open("w") as log:
            try:
                with subprocess.Popen(command, env=dict(env, **overrides), stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True) as process:
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
    return int(any(row["status"] != 0 for row in results))


if __name__ == "__main__":
    sys.exit(main())
