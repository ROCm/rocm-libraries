# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reject malformed one-solution JIT source bundles before any GPU launch.

Run after the numeric executable has produced a valid helper-inclusive bundle.
A substitute generator copies and damages that bundle, so every case traverses
the direct TensileLite API, the comgr build and C GEMM execution.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument("--architecture", default="gfx950")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=512)
    args = parser.parse_args()
    original = json.loads((args.bundle / "manifest.json").read_text())
    assert original["schema_version"] == 3 and "sources/Kernels.cpp" in original["sources"]
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    cases = {
        "architecture": "targets gfx942",
        "path-escape": "Artifact symlink escapes bundle",
        "missing-sources": "Missing source directory",
        "missing-symbol": "does not define kernel",
        "corrupt-library": "Invalid compressed solution library",
        "unsupported-problem": "does not support the request",
        "missing-main": "No main kernel assembly",
        "missing-helper-source": "C API matmul",
        "missing-helper-symbol": "C API matmul",
        "mismatched-amax": "does not support the request",
    }
    for case, diagnostic in cases.items():
        wrapper = args.fresh_output / (case + "-generator")
        wrapper.write_text(
            f"#!{sys.executable}\n"
            "import json, msgpack, pathlib, re, shutil, sys, yaml, zlib\n"
            f"source = pathlib.Path({str(args.bundle.resolve())!r})\n"
            f"case = {case!r}\n"
            "bundle = pathlib.Path(sys.argv[4]) / 'bundle'\n"
            "shutil.copytree(source, bundle)\n"
            "data = json.loads((bundle / 'manifest.json').read_text())\n"
            "sources = bundle / 'sources'\n"
            "main = bundle / data['sources'][0]\n"
            "helpers = [sources / 'Kernels.cpp', sources / 'Kernels.h']\n"
            "if case == 'architecture':\n"
            "    text = main.read_text()\n"
            "    assert '\"amdgcn-amd-amdhsa--' in text\n"
            "    main.write_text(re.sub(r'amdgcn-amd-amdhsa--[^\"]*', "
            "'amdgcn-amd-amdhsa--gfx942', text))\n"
            "if case == 'path-escape': (sources / 'escape.h').symlink_to(pathlib.Path(sys.argv[0]))\n"
            "if case == 'missing-sources': shutil.rmtree(sources)\n"
            "if case == 'missing-main': main.unlink()\n"
            "if case == 'missing-helper-source': helpers[0].unlink()\n"
            "if case == 'corrupt-library':\n"
            "    (bundle / data['library']['path']).write_bytes(b'bad library')\n"
            "if case == 'missing-symbol':\n"
            "    name = data['main_kernel']['name']\n"
            "    text = main.read_text()\n"
            "    assert name in text\n"
            "    main.write_text(text.replace(name, 'X' + name[1:]))\n"
            "if case == 'missing-helper-symbol':\n"
            "    names = re.findall(r'__global__ void (\\w+)\\(', helpers[0].read_text())\n"
            "    assert names\n"
            "    for path in helpers:\n"
            "        text = path.read_text()\n"
            "        for name in names:\n"
            "            text = re.sub(r'\\b' + name + r'\\b', 'X' + name[1:], text)\n"
            "        path.write_text(text)\n"
            "if case == 'mismatched-amax':\n"
            "    path = bundle / data['library']['path']\n"
            "    packed = data['library']['format'] == 'msgpack'\n"
            "    library = msgpack.unpackb(zlib.decompress(path.read_bytes()), raw=False) "
            "if packed else yaml.safe_load(path.read_text())\n"
            "    solution = library['solutions'][0]\n"
            "    solution['problemType']['outputAmaxD'] = True\n"
            "    checks = [p for p in solution['problemPredicate']['value'] if p['type'] == 'AmaxDCheck']\n"
            "    assert len(checks) == 1\n"
            "    checks[0]['value'] = True\n"
            "    if packed: path.write_bytes(zlib.compress(msgpack.packb(library)))\n"
            "    else: path.write_text(yaml.safe_dump(library))\n"
        )
        wrapper.chmod(0o700)
        output = args.fresh_output / case
        command = [
            str(args.executable.resolve()),
            str(wrapper.resolve()),
            str(args.fresh_output.resolve()),
            "",
            "unused.yaml",
            str(output.resolve()),
            args.architecture,
            "unused-compiler",
            "--m",
            str(args.m),
            "--n",
            str(args.n),
            "--k",
            str(args.k),
        ]
        if case == "unsupported-problem":
            command += ["--trans-b", "T"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        log = result.stdout + result.stderr
        (args.fresh_output / (case + ".log")).write_text(log)
        assert result.returncode == 1, (case, result.returncode, log)
        assert diagnostic in log, (case, diagnostic, log)
        print(f"PASS {case}: {diagnostic}", flush=True)
    print(
        f"PASS: {len(cases)} malformed solution bundles/problems rejected before launch"
    )


if __name__ == "__main__":
    main()
