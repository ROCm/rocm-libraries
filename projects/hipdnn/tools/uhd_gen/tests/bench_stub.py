# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lets `uhd_gen.generate` find a bench that a test's `_run_json` stand-in replaces."""
import shutil


class _ShutilFindingAnyBench:
    """`shutil` as `uhd_gen.generate` sees it, except that `which` finds any name."""

    def __getattr__(self, name):
        return getattr(shutil, name)

    @staticmethod
    def which(name, *args, **kwargs):
        return name


def find_any_bench(monkeypatch) -> None:
    """Replace `uhd_gen.generate`'s `shutil` module attribute, not `shutil.which` itself.

    Patching `shutil.which` would also make every other lookup succeed, including the
    feature evaluator's, so a test that needs a built hipdnn_uhd_features would run (and
    fail) without one instead of the `evaluator` fixture skipping it.
    """
    monkeypatch.setattr("uhd_gen.generate.shutil", _ShutilFindingAnyBench())
