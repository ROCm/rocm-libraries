# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reject malformed one-solution JIT source bundles before any GPU launch.

Each case copies a valid helper-inclusive bundle, damages the copy and runs
the public API test on it, so every case traverses a JIT backend, the comgr
build and C GEMM execution. With --replay the executable is the replay API
test, which reads the damaged copy through the mock backend. Without it the
executable is the TensileLite API test, and a substitute generator writes the
damaged copy as its output.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

# Build failures name the comgr log; the others fail while the backend reads
# the bundle, or at launch.
DIAGNOSTICS = {
    "architecture": ("targets gfx942",),
    "path-escape": ("Artifact symlink escapes bundle",),
    "missing-sources": ("Missing source directory",),
    "missing-symbol": ("does not define kernel",),
    "corrupt-assembly": ("comgr could not assemble", "comgr.log"),
    "broken-helper-source": ("comgr could not compile HIP", "comgr.log"),
    "corrupt-library": ("Invalid compressed solution library",),
    "truncated-library": ("Invalid compressed solution library",),
    "unsupported-problem": ("does not support the request",),
    "missing-main": ("No main kernel assembly",),
    "missing-helper-source": ("C API matmul",),
    "missing-helper-symbol": ("C API matmul",),
    "mismatched-amax": ("does not support the request",),
}
# The mock backend declines a problem its solution does not solve before building.
REPLAY_DIAGNOSTICS = {
    "unsupported-problem": ("No replayed solution solves this problem",),
    "mismatched-amax": ("No replayed solution solves this problem",),
}


def damage(bundle, case, outside):
    """Damage the bundle copy for case; outside is a file outside the bundle."""
    data = json.loads((bundle / "manifest.json").read_text())
    sources = bundle / "sources"
    main = bundle / data["sources"][0]
    helpers = [sources / "Kernels.cpp", sources / "Kernels.h"]
    if case == "architecture":
        text = main.read_text()
        assert '"amdgcn-amd-amdhsa--' in text
        main.write_text(re.sub(r'amdgcn-amd-amdhsa--[^"]*', "amdgcn-amd-amdhsa--gfx942", text))
    if case == "path-escape":
        (sources / "escape.h").symlink_to(Path(outside).resolve())
    if case == "missing-sources":
        shutil.rmtree(sources)
    if case == "missing-main":
        main.unlink()
    if case == "missing-helper-source":
        helpers[0].unlink()
    if case == "corrupt-assembly":
        main.write_text(main.read_text() + "\n  s_not_an_instruction v0\n")
    if case == "broken-helper-source":
        helpers[0].write_text(helpers[0].read_text() + "\nthis is not C++;\n")
    if case == "corrupt-library":
        (bundle / data["library"]["path"]).write_bytes(b"bad library")
    if case == "truncated-library":
        library = bundle / data["library"]["path"]
        assert library.suffix == ".zlib"
        encoded = library.read_bytes()
        library.write_bytes(encoded[: len(encoded) // 2])
    if case == "missing-symbol":
        name = data["main_kernel"]["name"]
        text = main.read_text()
        assert name in text
        main.write_text(text.replace(name, "X" + name[1:]))
    if case == "missing-helper-symbol":
        names = re.findall(r"__global__ void (\w+)\(", helpers[0].read_text())
        assert names
        for path in helpers:
            text = path.read_text()
            for name in names:
                text = re.sub(r"\b" + name + r"\b", "X" + name[1:], text)
            path.write_text(text)
    if case == "mismatched-amax":
        import msgpack
        import yaml
        import zlib

        path = bundle / data["library"]["path"]
        packed = data["library"]["format"] == "msgpack"
        library = (
            msgpack.unpackb(zlib.decompress(path.read_bytes()), raw=False)
            if packed
            else yaml.safe_load(path.read_text())
        )
        solution = library["solutions"][0]
        solution["problemType"]["outputAmaxD"] = True
        checks = [p for p in solution["problemPredicate"]["value"] if p["type"] == "AmaxDCheck"]
        assert len(checks) == 1
        checks[0]["value"] = True
        if packed:
            path.write_bytes(zlib.compress(msgpack.packb(library)))
        else:
            path.write_text(yaml.safe_dump(library))


def generator(path, source, case):
    """A substitute generator that writes a damaged copy of source as its bundle."""
    path.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, shutil, sys\n"
        "sys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "import test_bundle_failures\n"
        "bundle = pathlib.Path(sys.argv[4]) / 'bundle'\n"
        f"shutil.copytree({str(source)!r}, bundle)\n"
        f"test_bundle_failures.damage(bundle, {case!r}, sys.argv[0])\n"
    )
    path.chmod(0o700)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--architecture", default="gfx950")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=512)
    args = parser.parse_args()
    source = args.bundle.resolve(strict=True)
    original = json.loads((source / "manifest.json").read_text())
    assert original["schema_version"] == 3 and "sources/Kernels.cpp" in original["sources"]
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    sizes = ["--m", str(args.m), "--n", str(args.n), "--k", str(args.k)]
    for case, diagnostics in DIAGNOSTICS.items():
        if args.replay:
            diagnostics = REPLAY_DIAGNOSTICS.get(case, diagnostics)
            bundle = output / case / "bundle"
            shutil.copytree(source, bundle)
            damage(bundle, case, __file__)
            command = [str(args.executable.resolve()), "--replay", str(bundle), *sizes]
        else:
            wrapper = output / (case + "-generator")
            generator(wrapper, source, case)
            command = [
                str(args.executable.resolve()),
                str(wrapper),
                str(output),
                "",
                "unused.yaml",
                str(output / case),
                args.architecture,
                "unused-compiler",
                *sizes,
            ]
        if case == "unsupported-problem":
            command += ["--trans-b", "T"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        log = result.stdout + result.stderr
        (output / (case + ".log")).write_text(log)
        assert result.returncode == 1, (case, result.returncode, log)
        for diagnostic in diagnostics:
            assert diagnostic in log, (case, diagnostic, log)
        print(f"PASS {case}: {', '.join(diagnostics)}", flush=True)
    print(
        f"PASS: {len(DIAGNOSTICS)} malformed solution bundles/problems rejected before launch"
    )


if __name__ == "__main__":
    main()
