# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Who is this clang, really? Both tools that compile probe modules have to ask,
# because neither `ROCKE_LLVM_BIN` nor PATH is obliged to point at the LLVM that
# rocke itself resolved, and an answer measured by one vintage and filed under
# another is worse than no answer.
#
# Shared rather than duplicated: the two callers act on a mismatch differently
# -- the generator refuses, because it would otherwise commit a permanently
# mis-attributed artifact; the validity gate reports UNVALIDATED, because it
# writes nothing and "we could not ask" is not "the answer is no" -- but they
# must agree on what the mismatch *is*. Two copies of this regex would drift the
# moment a vendor changed its version banner.
#
# Not a CLI entry point -- hence the leading underscore, as with _hostcaps.py.

from __future__ import annotations

import re
import subprocess


def clang_identity(clang: str) -> str:
    """The first line of `clang --version`, or `(unknown)`.

    Never raises: a tool that cannot introduce itself is a fact to record, not
    a reason to abort a sweep that has not started yet.
    """
    try:
        proc = subprocess.run(
            [clang, "--version"], capture_output=True, text=True, check=False
        )
        return (proc.stdout or "").strip().splitlines()[0][:200]
    except (OSError, IndexError):
        return "(unknown)"


def flavor_of_clang(identity: str) -> str | None:
    """Best-effort LLVM major from a `clang --version` line, as a flavor string.

    `None` means the banner did not carry a version we recognise -- which must
    read as "unknown", never as "mismatch". A vendor is free to reword this
    line, and a caller that treated an unparseable banner as disagreement would
    turn a cosmetic change upstream into a red gate everywhere.
    """
    m = re.search(r"clang version (\d+)", identity)
    return f"llvm{m.group(1)}" if m else None
