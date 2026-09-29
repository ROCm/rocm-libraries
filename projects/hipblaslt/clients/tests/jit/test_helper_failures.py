# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check public C/C++ GEMM helper failures before submission and state publication.

Run with the existing venv after a split-K source bundle is available.
The API test's --expect-helper-failure mode checks D/workspace sentinels and that
failed reinitialization leaves the prior extension algorithm runnable.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def write_generator(path, source, case):
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, re, shutil, sys\n"
        f"source = pathlib.Path({str(source)!r})\n"
        f"case = {case!r}\n"
        "bundle = pathlib.Path(sys.argv[4]) / 'bundle'\n"
        "shutil.copytree(source, bundle)\n"
        "data = json.loads((bundle / 'manifest.json').read_text())\n"
        "main = data['sources'][0]\n"
        "helpers = [bundle / 'sources/Kernels.cpp', bundle / 'sources/Kernels.h']\n"
        "assert helpers[0].is_file(), 'Expected a helper-inclusive split-K source bundle'\n"
        "changed = []\n"
        "if case == 'missing-helper-source':\n"
        "    helpers[0].unlink()\n"
        "    changed.append('sources/Kernels.cpp')\n"
        "elif case == 'missing-helper-symbols':\n"
        "    changed = re.findall(r'__global__ void (\\w+)\\(', helpers[0].read_text())\n"
        "    for path in helpers:\n"
        "        text = path.read_text()\n"
        "        for name in changed:\n"
        "            text = re.sub(r'\\b' + name + r'\\b', 'X' + name[1:], text)\n"
        "        path.write_text(text)\n"
        "    assert changed, 'No helper entry points mutated'\n"
        "else:\n"
        "    assert case == 'valid'\n"
        "assert (bundle / main).read_bytes() == (source / main).read_bytes()\n"
        "assert (bundle / data['library']['path']).read_bytes() == "
        "(source / data['library']['path']).read_bytes()\n"
        "(bundle.parent / 'mutation.json').write_text(json.dumps("
        "dict(case=case, changed=changed, main_unchanged=True, library_unchanged=True), indent=2))\n"
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
    valid = output / "valid-generator"
    write_generator(valid, source, "valid")
    results = []
    for case in ("missing-helper-source", "missing-helper-symbols"):
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
            "--expect-helper-failure",
            "1",
            "--m",
            str(args.m),
            "--n",
            str(args.n),
            "--k",
            str(args.k),
        ]
        (output / (case + "-command.json")).write_text(json.dumps(command, indent=2))
        proc = subprocess.run(
            command, env=env, text=True, capture_output=True, timeout=300
        )
        (output / (case + ".log")).write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f'{case} failed; inspect {output / (case + ".log")}')
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
