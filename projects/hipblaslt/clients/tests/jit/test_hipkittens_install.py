#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Install the runtime component, run HipKittens kernels from the install, then
move the install and check that they run again with the same published index."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(test, prefix, library, log):
    lib = next(prefix.glob("lib*/libhipblaslt.so*")).parent
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (str(lib), env.get("LD_LIBRARY_PATH")))
    )
    env["HIPBLASLT_JIT_LIBRARY_PATH"] = str(library)
    result = subprocess.run(
        [str(test), "library"], env=env, text=True, capture_output=True, timeout=600
    )
    log.write(result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{test} library failed in {prefix}")
    headers = Path(re.search(r"^headers: (.*)$", result.stdout, re.M).group(1))
    if not headers.is_relative_to(prefix.resolve()):
        raise RuntimeError(f"The headers {headers} are not in the install {prefix}")
    return int(re.search(r"^INDEX (\d+)$", result.stdout, re.M).group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("test", type=Path, help="hipblaslt-jit-hipkittens-test")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    first, moved = args.output / "a", args.output / "b"
    library = args.output / "jit-library"
    with (args.output / "install.log").open("w") as log:
        subprocess.run(
            ["cmake", "--install", str(args.build), "--component", "runtime",
             "--prefix", str(first)],
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600,
        )
        directories = list(first.glob("lib*/hipblaslt/hipkittens/*"))
        if len(directories) != 1:
            raise RuntimeError(f"Expected one installed header directory, found {directories}")
        manifest = json.loads((directories[0] / "manifest.json").read_text())
        missing = [f["path"] for f in manifest["files"] if not (directories[0] / f["path"]).is_file()]
        if missing:
            raise RuntimeError(f"Installed headers are missing {missing}")
        license = first / "share/doc/hipblaslt/third-party/hipkittens/LICENSE"
        if not license.is_file():
            raise RuntimeError(f"No {license}")
        print(f"PASS installed {len(manifest['files'])} headers, the manifest and the license")

        index = run(args.test, first, library, log)
        print(f"PASS ran index {index} from {first}")
        shutil.move(first, moved)
        again = run(args.test, moved, library, log)
        if again != index:
            raise RuntimeError(f"The moved install published index {again}, not {index}")
        print(f"PASS ran the same index from {moved} after moving the install")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
