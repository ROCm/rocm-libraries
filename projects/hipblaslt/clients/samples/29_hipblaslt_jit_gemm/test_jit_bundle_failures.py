# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reject malformed one-solution JIT bundles before any GPU launch.

Run after the numeric executable has produced a valid helper-inclusive bundle.
A substitute generator copies and damages that bundle, so every case traverses
JitGemm::prepare and, for helper symbol resolution, JitGemm::initialize.
"""

import argparse
import json
from pathlib import Path
import shutil
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
    parser.add_argument("--readobj", default=shutil.which("llvm-readobj"))
    args = parser.parse_args()
    if not args.readobj:
        parser.error("llvm-readobj is required to mutate helper symbols")
    original = json.loads((args.bundle / "manifest.json").read_text())
    assert any(
        p != original["main_kernel"]["code_object"] for p in original["code_objects"]
    )
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    cases = {
        "schema": "manifest schema/counts/index",
        "duplicate-key": "duplicate object key",
        "architecture": "does not match device",
        "path-escape": "Artifact path escapes bundle",
        "missing-module": "Missing artifact",
        "missing-symbol": "Resolve generated kernel symbol",
        "corrupt-library": "Expected a non-lazy library",
        "unsupported-problem": "does not support this problem/device",
        "duplicate-code-object": "Duplicate code object artifact",
        "unlisted-main": "Main code object missing from code_objects",
        "missing-helper-module": "Missing artifact",
        "missing-helper-symbol": "Resolve generated invocation symbol",
        "mismatched-amax": "does not support this problem/device",
    }
    for case, diagnostic in cases.items():
        wrapper = args.fresh_output / (case + "-generator")
        wrapper.write_text(
            f"#!{sys.executable}\n"
            "import json, msgpack, pathlib, shutil, subprocess, sys, yaml, zlib\n"
            f"source = pathlib.Path({str(args.bundle.resolve())!r})\n"
            f"readobj = {args.readobj!r}\n"
            f"case = {case!r}\n"
            "bundle = pathlib.Path(sys.argv[4]) / 'bundle'\n"
            "shutil.copytree(source, bundle)\n"
            "manifest = bundle / 'manifest.json'\n"
            "data = json.loads(manifest.read_text())\n"
            "main = data['main_kernel']['code_object']\n"
            "helpers = [p for p in data['code_objects'] if p != main]\n"
            "if case == 'schema': data['schema_version'] = 1\n"
            "if case == 'architecture': data['architecture']['resolved'] = 'gfx000'\n"
            "if case == 'path-escape': data['main_kernel']['code_object'] = '../outside.co'\n"
            "if case == 'missing-module': (bundle / main).unlink()\n"
            "if case == 'missing-helper-module': (bundle / helpers[0]).unlink()\n"
            "if case == 'duplicate-code-object': data['code_objects'].append(main)\n"
            "if case == 'unlisted-main': data['code_objects'] = helpers\n"
            "if case == 'corrupt-library':\n"
            "    (bundle / data['library']['path']).write_bytes(b'bad library')\n"
            "if case == 'missing-symbol':\n"
            "    path = bundle / main\n"
            "    name = data['main_kernel']['name'].encode()\n"
            "    content = path.read_bytes()\n"
            "    assert name in content\n"
            "    path.write_bytes(content.replace(name, b'X' + name[1:]))\n"
            "if case == 'missing-helper-symbol':\n"
            "    changed = 0\n"
            "    for relative in helpers:\n"
            "        path = bundle / relative\n"
            "        notes = subprocess.check_output([readobj, '--notes', str(path)], text=True)\n"
            "        metadata = yaml.safe_load(notes.split('AMDGPU Metadata: ', 1)[1]"
            ".split('...', 1)[0])\n"
            "        content = path.read_bytes()\n"
            "        for kernel in sorted(metadata['amdhsa.kernels'], "
            "key=lambda k: len(k['.name']), reverse=True):\n"
            "            name = kernel['.name'].encode()\n"
            "            assert name in content\n"
            "            content = content.replace(name, b'X' + name[1:])\n"
            "            changed += 1\n"
            "        path.write_bytes(content)\n"
            "    assert changed > 0\n"
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
            "manifest.write_text(json.dumps(data))\n"
            "if case == 'duplicate-key':\n"
            "    manifest.write_text('{\"schema_version\":2,' + manifest.read_text()[1:])\n"
        )
        wrapper.chmod(0o700)
        output = args.fresh_output / case
        mode = "--expect-prepare-failure"
        if case == "unsupported-problem":
            mode = "--expect-unsupported"
        elif case == "missing-helper-symbol":
            mode = "--expect-initialize-failure"
        subprocess.run(
            [
                str(args.executable.resolve()),
                str(wrapper.resolve()),
                str(args.fresh_output.resolve()),
                "",
                "unused.yaml",
                str(output.resolve()),
                args.architecture,
                "unused-compiler",
                mode,
                diagnostic,
                "--m",
                str(args.m),
                "--n",
                str(args.n),
                "--k",
                str(args.k),
            ],
            check=True,
        )
    print(
        f"PASS: {len(cases)} malformed solution bundles/problems rejected before launch"
    )


if __name__ == "__main__":
    main()
