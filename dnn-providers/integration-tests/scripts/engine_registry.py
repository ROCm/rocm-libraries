#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python mirror of the engine registry in ``EngineNames.hpp``.

Sidecars and observation records name engines as bare strings, so nothing in
the JSON stops a name from outliving the engine it referred to.  A retired or
misspelt engine then sits in the tree claiming support that no build can ever
check: the enforcer never loads it, so the claim is neither satisfied nor
broken, and the matrix renders a column for a thing that does not exist.

The claim tools cannot link against the C++ registry, and a hand-maintained
``KNOWN_ENGINES`` set in each script is a second source of truth that goes
stale the first time an engine is added.  So this module reads the header
itself and parses the ``HIPDNN_REGISTER_ENGINE`` lines.

The parse is deliberately shallow -- it is a lint input, not a build step.
When the header cannot be found or read, :func:`known_engines` returns None
and callers skip the check rather than inventing warnings from a guess: a
partial checkout must not be reported as a tree full of retired engines.
"""

from __future__ import annotations

import pathlib
import re

# scripts/ -> integration-tests/ -> dnn-providers/ -> repo root
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

ENGINE_NAMES_HEADER = (
    _REPO_ROOT
    / "projects"
    / "hipdnn"
    / "data_sdk"
    / "include"
    / "hipdnn_data_sdk"
    / "utilities"
    / "EngineNames.hpp"
)

# Only a registration at column 0 counts.  The macro's own definition and the
# ``HIPDNN_REGISTER_ENGINE(MyEngine)`` examples in its doc comment are indented
# or comment-prefixed, and matching them would invent engines named "MyEngine".
_REGISTRATION_RE = re.compile(r"^HIPDNN_REGISTER_ENGINE\(([^)]*)\)", re.MULTILINE)

# HIPDNN_REGISTER_ENGINE(Name) and HIPDNN_REGISTER_ENGINE(Name, "OnDiskName")
# name the engine differently: with two arguments the string literal wins, and
# it is the string that reaches a sidecar.
_QUOTED_RE = re.compile(r'^\s*"(.*)"\s*$')


def parse_registrations(source: str) -> set[str]:
    """Extract the engine names registered in ``EngineNames.hpp`` source text."""
    names = set()
    for match in _REGISTRATION_RE.finditer(source):
        arguments = [part.strip() for part in match.group(1).split(",")]
        if not arguments or not arguments[0]:
            continue
        if len(arguments) >= 2:
            quoted = _QUOTED_RE.match(arguments[1])
            # An unquoted second argument is not a form the macro accepts;
            # fall back to the identifier rather than guessing.
            names.add(quoted.group(1) if quoted else arguments[0])
        else:
            names.add(arguments[0])
    return names


def known_engines(
    header: pathlib.Path | None = None,
) -> set[str] | None:
    """Every registered engine name, or None when the header is unreadable."""
    path = ENGINE_NAMES_HEADER if header is None else header
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    names = parse_registrations(source)
    # An empty parse means the macro was renamed or the file is not the header
    # we expect.  Treating that as "no engine is known" would condemn every
    # sidecar in the tree, so report it the same way as a missing file.
    return names or None
