#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The released tarball on several Linux bases (default: ubuntu:22.04, ubuntu:24.04, debian:bookworm), each in a
throwaway container (--rm, --network none) with Docker or Podman, on Linux or on Docker Desktop. On each base, with
Python 3 and no other tool, a user who is not root unpacks the tarball and runs stages.py tarball from the unpacked
package: every member a plain file under rocke_monitor/ with no owner of the build machine, exactly the built
folder's files byte for byte, and the privacy check on the unpacked folder.

Usage: tests/container/tarball.py --tarball FILE --dist DIR [--engine docker|podman] [--bases B1,B2,...]
       [--keep-images]
FILE is the released tarball and DIR the built folder it was made from (tools/make_package.py --out writes both).
Each container runs with SMOKE_RUN_OPTS (default: --cpus 4 --memory 4g). Network: pulling the base images, and
building an image on each (Python 3 from the base's package archive). The checks themselves run with no network.
Exit 0 when every check on every base passed.
"""

import os
import sys
import tempfile

import hostrun

STAGE = '''
        mkdir -p "$HOME/install" && tar -xzf /src/rocke_monitor.tar.gz -C "$HOME/install" || exit 1
        python3 --version
        python3 -B "$HOME/install/rocke_monitor/tests/container/stages.py" tarball \\
            --tarball /src/rocke_monitor.tar.gz --dist /src/dist --unpacked "$HOME/install/rocke_monitor"'''
STAGE_TEXT = (
    "the tarball's members and files, and the privacy check on the unpacked package"
)


def bases_of(value: str) -> list:
    """The comma-separated bases, as the shell splits them (a trailing comma adds none)."""
    out = value.split(",")
    if out and out[-1] == "":
        out.pop()
    return out


def main(args: list) -> int:
    hostrun.setup_output()
    here = hostrun.host_path(os.path.dirname(os.path.abspath(__file__)))
    engine = os.environ.get("CONTAINER_ENGINE", "")
    tarball, dist, bases, keep = "", "", "ubuntu:22.04,ubuntu:24.04,debian:bookworm", 0
    run_opts = hostrun.run_opts("--cpus 4 --memory 4g")
    while args:
        a = args[0]
        if a == "--tarball":
            tarball = hostrun.need_value(args, a)
        elif a == "--dist":
            dist = hostrun.need_value(args, a)
        elif a == "--engine":
            engine = hostrun.need_value(args, a)
        elif a == "--bases":
            bases = hostrun.need_value(args, a)
        elif a == "--keep-images":
            keep = 1
            args = args[1:]
            continue
        elif a in ("-h", "--help"):
            sys.stdout.write(hostrun.help_lines(__doc__, 12) + "\n")
            return 0
        else:
            print(f"unknown argument: {a} (see --help)")
            return 2
        args = args[2:]
    if not os.path.isfile(tarball):
        print(f"--tarball FILE: the released tarball (got '{tarball}')")
        return 2
    if not os.path.isdir(dist):
        print(f"--dist DIR: the built folder the tarball was made from (got '{dist}')")
        return 2
    tarball = (
        hostrun.folder(os.path.dirname(tarball) or ".")
        + "/"
        + os.path.basename(tarball)
    )
    dist = hostrun.folder(dist)
    engine = hostrun.pick_engine(engine)
    if not engine:
        return 2
    empty = hostrun.host_path(
        tempfile.mkdtemp()
    )  # the images take nothing from a build context
    res = hostrun.Results()
    images = []
    for base in bases_of(bases):
        tag = base.replace(":", "-").replace("/", "-")
        name = f"rkmon-tarball:{tag}"
        print(f"== {base}: building the image with {engine}")
        if (
            hostrun.quiet(
                [
                    engine,
                    "build",
                    "-q",
                    "--build-arg",
                    f"BASE={base}",
                    "--target",
                    "python",
                    "-t",
                    name,
                    "-f",
                    f"{here}/Dockerfile",
                    empty,
                ]
            )
            != 0
        ):
            res.result("FAIL", f"{base}: building the image with Python 3")
            continue
        images.append(name)
        rc, out = hostrun.capture(
            [
                engine,
                "run",
                "--rm",
                "--network",
                "none",
                *run_opts,
                "-v",
                f"{tarball}:/src/rocke_monitor.tar.gz:ro",
                "-v",
                f"{dist}:/src/dist:ro",
                name,
                "sh",
                "-c",
                STAGE,
            ]
        )
        hostrun.indented(out)
        res.result("PASS" if rc == 0 else "FAIL", f"{base}: {STAGE_TEXT}")
    try:
        os.rmdir(empty)
    except OSError:
        pass
    if keep == 0 and images:
        hostrun.capture([engine, "rmi", *images])
    print("TARBALL TEST: PASS" if res.failed == 0 else "TARBALL TEST: FAIL")
    return res.failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
