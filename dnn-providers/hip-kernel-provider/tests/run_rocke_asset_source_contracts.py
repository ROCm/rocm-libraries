# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the provider's source-only rocKE asset contracts without silent reduction."""

from __future__ import annotations

from pathlib import Path

import pytest

PROVIDER = Path(__file__).resolve().parents[1]
TESTS = (
    PROVIDER / "rocke/platform/tests/tools/test_sdpa_asset_staging.py",
    PROVIDER / "tests/test_rocke_test_assets.py",
    PROVIDER / "tests/test_rocke_asset_source_contracts.py",
)


class _CompleteSourceContracts:
    def __init__(self, paths: tuple[Path, ...]):
        self.paths = {path.resolve() for path in paths}
        self.skipped = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.skipped:
            self.skipped += 1

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        collected = {Path(item.path).resolve() for item in session.items}
        missing = self.paths - collected
        if missing or self.skipped:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            print(
                "Source asset contract coverage failed: "
                f"{len(missing)} uncollected module(s), {self.skipped} skipped report(s)"
            )


def main() -> int:
    """Execute the complete source contract cohort while build files still exist."""
    return int(
        pytest.main(
            ["-q", "-rs", *(str(path) for path in TESTS)],
            plugins=[_CompleteSourceContracts(TESTS)],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
