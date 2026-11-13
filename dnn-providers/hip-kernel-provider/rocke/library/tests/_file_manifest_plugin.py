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
_EXPECTED_ROOT = "--rocke-expected-test-root"


def pytest_addoption(parser) -> None:
    parser.addoption(_OPTION, type=Path, default=None)
    parser.addoption(_EXPECTED_ROOT, type=Path, default=None)


def pytest_collection_finish(session) -> None:
    output = session.config.getoption(_OPTION)
    if output is None:
        return
    expected_root = session.config.getoption(_EXPECTED_ROOT)
    if expected_root is not None:
        expected = expected_root.resolve()
        selected = [
            Path(str(argument).split("::", 1)[0]).resolve()
            for argument in session.config.args
        ]
        if selected != [expected]:
            import pytest

            pytest.exit(
                "file-worker mode does not accept positional pytest paths/node IDs; "
                "use --jobs 0 or --only",
                returncode=2,
            )
    files: list[str] = []
    seen: set[str] = set()
    for item in session.items:
        path = str(item.path)
        if path not in seen:
            seen.add(path)
            files.append(path)
    output.write_text("".join(f"{path}\n" for path in files), encoding="utf-8")
