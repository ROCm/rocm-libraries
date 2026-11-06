# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest plugin that records selected test files after collection filters.

The library runner loads this only for its manifest pass.  Writing from
``pytest_collection_finish`` means marker and ``-k`` deselection have already
run, so a file enters the fresh-process lane only when it owns selected work.
"""

from __future__ import annotations

from pathlib import Path

_OPTION = "--rocke-selected-file-manifest"


def pytest_addoption(parser) -> None:
    parser.addoption(_OPTION, type=Path, default=None)


def pytest_collection_finish(session) -> None:
    output = session.config.getoption(_OPTION)
    if output is None:
        return
    files: list[str] = []
    seen: set[str] = set()
    for item in session.items:
        path = str(item.path)
        if path not in seen:
            seen.add(path)
            files.append(path)
    output.write_text("".join(f"{path}\n" for path in files), encoding="utf-8")
