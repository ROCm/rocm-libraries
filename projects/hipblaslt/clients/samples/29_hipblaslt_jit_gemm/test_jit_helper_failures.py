# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check public C/C++ GEMM helper failures before submission and state publication.

Run with the existing venv after a source-built split-K bundle is available.
The sample's --expect-helper-failure mode checks D/workspace sentinels and that
failed reinitialization leaves the prior extension algorithm runnable.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def write_generator(path, source, case, readobj):
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, shutil, struct, subprocess, sys, yaml\n"
        f"source = pathlib.Path({str(source)!r})\n"
        f"case = {case!r}\n"
        f"readobj = {readobj!r}\n"
        "bundle = pathlib.Path(sys.argv[4]) / 'bundle'\n"
        "shutil.copytree(source, bundle)\n"
        "manifest = bundle / 'manifest.json'\n"
        "data = json.loads(manifest.read_text())\n"
        "main = data['main_kernel']['code_object']\n"
        "helpers = [name for name in data['code_objects'] if name != main]\n"
        "assert helpers, 'Expected a helper-inclusive split-K source bundle'\n"
        "changed = []\n"
        "if case == 'unlisted-helper-modules':\n"
        "    for name in helpers:\n"
        "        (bundle / name).unlink()\n"
        "        changed.append(name)\n"
        "    data['code_objects'] = [main]\n"
        "    data['helpers'] = []\n"
        "    data['counts']['helper_generators'] = 0\n"
        "    data['counts']['support_generators'] = 0\n"
        "elif case == 'missing-helper-symbols':\n"
        "    for relative in helpers:\n"
        "        path = bundle / relative\n"
        "        notes = subprocess.check_output([readobj, '--notes', str(path)], text=True)\n"
        "        metadata = yaml.safe_load(notes.split('AMDGPU Metadata: ', 1)[1]"
        ".split('...', 1)[0])\n"
        "        content = path.read_bytes()\n"
        "        for kernel in sorted(metadata.get('amdhsa.kernels', []),"
        " key=lambda item: len(item['.name']), reverse=True):\n"
        "            name = kernel['.name'].encode()\n"
        "            assert name in content\n"
        "            content = content.replace(name, b'X' + name[1:])\n"
        "            changed.append(kernel['.name'])\n"
        "        path.write_bytes(content)\n"
        "    assert changed, 'No helper entry points mutated'\n"
        "else:\n"
        "    assert case == 'valid'\n"
        "assert (bundle / main).read_bytes() == (source / main).read_bytes()\n"
        "assert (bundle / data['library']['path']).read_bytes() == "
        "(source / data['library']['path']).read_bytes()\n"
        "loader = bundle / 'loader.bin'\n"
        "raw = loader.read_bytes()\n"
        "assert raw[:8] == b'TLJIT001'\n"
        "count = struct.unpack_from('<I', raw, 8)[0]\n"
        "offset = 12\n"
        "for _ in range(count):\n"
        "    length = struct.unpack_from('<I', raw, offset)[0]\n"
        "    offset += 4 + length\n"
        "encoded = raw[:offset] + struct.pack('<I', len(data['code_objects']))\n"
        "for name in data['code_objects']:\n"
        "    value = name.encode('utf-8')\n"
        "    encoded += struct.pack('<I', len(value)) + value\n"
        "loader.write_bytes(encoded)\n"
        "manifest.write_text(json.dumps(data, indent=2) + '\\n')\n"
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
    parser.add_argument("--readobj", default=shutil.which("llvm-readobj"))
    args = parser.parse_args()
    if not args.readobj:
        parser.error("llvm-readobj is required for helper-symbol mutation")
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
    write_generator(valid, source, "valid", args.readobj)
    results = []
    for case in ("unlisted-helper-modules", "missing-helper-symbols"):
        damaged = output / (case + "-generator")
        write_generator(damaged, source, case, args.readobj)
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
