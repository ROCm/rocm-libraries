# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check public C/C++ GEMM helper failures before submission and state publication.

Run after a split-K source bundle is available. The API test's
--expect-helper-failure mode checks D/workspace sentinels and that failed
reinitialization leaves the prior extension algorithm runnable. With --replay
the executable is the replay API test, whose second solution replays a copy of
the bundle with damaged helpers. Without it the executable is the TensileLite
API test, and a substitute generator writes that copy.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def mutate(source, bundle, case):
    """Damage the helpers of the bundle copy for case; returns what changed."""
    data = json.loads((bundle / "manifest.json").read_text())
    main = data["sources"][0]
    helpers = [bundle / "sources/Kernels.cpp", bundle / "sources/Kernels.h"]
    assert helpers[0].is_file(), "Expected a helper-inclusive split-K source bundle"
    changed = []
    if case == "missing-helper-source":
        helpers[0].unlink()
        changed.append("sources/Kernels.cpp")
    elif case == "missing-helper-symbols":
        changed = re.findall(r"__global__ void (\w+)\(", helpers[0].read_text())
        for path in helpers:
            text = path.read_text()
            for name in changed:
                text = re.sub(r"\b" + name + r"\b", "X" + name[1:], text)
            path.write_text(text)
        assert changed, "No helper entry points mutated"
    else:
        assert case == "valid"
    assert (bundle / main).read_bytes() == (source / main).read_bytes()
    library = data["library"]["path"]
    assert (bundle / library).read_bytes() == (source / library).read_bytes()
    return dict(case=case, changed=changed, main_unchanged=True, library_unchanged=True)


def write_generator(path, source, case):
    """A substitute generator that writes a mutated copy of source as its bundle."""
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, shutil, sys\n"
        "sys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "import test_helper_failures\n"
        f"source = pathlib.Path({str(source)!r})\n"
        "bundle = pathlib.Path(sys.argv[4]) / 'bundle'\n"
        "shutil.copytree(source, bundle)\n"
        f"mutation = test_helper_failures.mutate(source, bundle, {case!r})\n"
        "(bundle.parent / 'mutation.json').write_text(json.dumps(mutation, indent=2))\n"
    )
    path.chmod(0o700)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument(
        "bundle",
        type=Path,
        help="Existing local split-K bundle containing manifest.json",
    )
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=512)
    args = parser.parse_args()
    source = args.bundle.resolve(strict=True)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["counts"]["helper_generators"] < 1:
        parser.error("Source bundle must include a later split-K conversion helper")
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    empty_library = output / "empty-device-library"
    empty_library.mkdir()
    env = dict(
        os.environ,
        HIPBLASLT_TENSILE_LIBPATH=str(empty_library),
        PYTHONDONTWRITEBYTECODE="1",
    )
    sizes = ["--m", str(args.m), "--n", str(args.n), "--k", str(args.k)]
    if not args.replay:
        valid = output / "valid-generator"
        write_generator(valid, source, "valid")
    results = []
    for case in ("missing-helper-source", "missing-helper-symbols"):
        if args.replay:
            damaged = output / (case + "-second") / "bundle"
            shutil.copytree(source, damaged)
            mutation = mutate(source, damaged, case)
            command = [
                str(args.executable.resolve(strict=True)),
                "--replay",
                str(source),
                "--second-replay",
                str(damaged),
            ]
        else:
            damaged = output / (case + "-generator")
            write_generator(damaged, source, case)
            command = [
                str(args.executable.resolve(strict=True)),
                str(valid),
                str(output),
                "",
                "unused.yaml",
                str(output / case),
                manifest["architecture"]["requested"],
                "unused-compiler",
                "--second-python",
                str(damaged),
            ]
        command += ["--expect-helper-failure", "1", *sizes]
        (output / (case + "-command.json")).write_text(json.dumps(command, indent=2))
        proc = subprocess.run(
            command, env=env, text=True, capture_output=True, timeout=300
        )
        (output / (case + ".log")).write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f'{case} failed; inspect {output / (case + ".log")}')
        if not args.replay:
            mutation = json.loads(
                (output / (case + "-second") / "mutation.json").read_text()
            )
        assert mutation["case"] == case and mutation["changed"]
        assert mutation["main_unchanged"] and mutation["library_unchanged"]
        results.append(dict(case=case, passed=True, mutation=mutation))
        print(
            f"PASS {case}: public C/C++ GEMM preflight and retained prior context",
            flush=True,
        )
    (output / "report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
