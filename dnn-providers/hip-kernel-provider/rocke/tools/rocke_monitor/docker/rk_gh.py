#!/usr/bin/python3 -I
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's `gh`: the GitHub request cap's counter.

The image links it as the first gh on every PATH in the container (container_layout.GH_COUNTER_DIR, before the
others: the image's PATH, which the monitor, the Claude session, its agents and the jobs inherit, and the writer's,
rkmon_lib/writer_run.py BASE_ENV). Each call counts in the cap's record, waiting first while the last hour has no room
(rkmon_lib/gh_cap.py `count`, which says when a call is not counted), and then runs the real gh in its place, with
the same arguments and environment: the first gh on PATH after this file, in a folder named from the root (a PATH
entry that is no absolute folder is passed over, so a gh in the folder a call runs in never runs). A wait is said on
standard error as it begins.

With no gh after it on PATH it says that the image's gh is missing, and what brings it back
(container_layout.GH_MISSING_FIX), and exits 127, as a shell does for a command it cannot find. It runs with the
image's own code (CODE, the folder above this one), standard library only: it loads the cap and the container's layout
by their paths there (rkmon_lib/gh_cap.py, rkmon_lib/container_layout.py).
"""

import os
import runpy
import sys
import types

CODE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # the image's code
NOT_FOUND = 127


def real_gh(environ=None) -> str:
    """The gh this file stands before: the first gh on PATH that is not this file, in an absolute folder; "" for none."""
    me = os.path.realpath(__file__)
    for folder in (
        (os.environ if environ is None else environ).get("PATH", "").split(os.pathsep)
    ):
        gh = os.path.join(folder, "gh")
        if (
            os.path.isabs(folder)
            and os.path.realpath(gh) != me
            and os.path.isfile(gh)
            and os.access(gh, os.X_OK)
        ):
            return gh
    return ""


def main(argv: list[str]) -> int:
    sys.dont_write_bytecode = (
        True  # the image's code is read-only (and -I ignores PYTHONDONTWRITEBYTECODE)
    )
    gh = real_gh()
    if not gh:
        layout = types.SimpleNamespace(
            **runpy.run_path(os.path.join(CODE, "rkmon_lib", "container_layout.py"))
        )
        print(
            "gh: the image's gh is missing (no gh after the GitHub request counter on PATH): "
            f"{layout.GH_MISSING_FIX}",
            file=sys.stderr,
        )
        return NOT_FOUND
    gh_cap = types.SimpleNamespace(
        **runpy.run_path(os.path.join(CODE, "rkmon_lib", "gh_cap.py"))
    )
    gh_cap.count(argv)  # a wait, or a record it cannot use, is said on standard error
    os.execv(gh, [gh, *argv])
    return NOT_FOUND  # execv does not return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
