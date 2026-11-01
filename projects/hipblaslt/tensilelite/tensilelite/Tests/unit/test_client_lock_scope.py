# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Clients sharing hardware serialize; independent emulators can overlap."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import filelock
import pytest


def test_client_lock_scope(tmp_path, monkeypatch, pytestconfig):
    spec = importlib.util.spec_from_file_location(
        "tensile_test_fixtures", Path(__file__).parents[1] / "conftest.py"
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    monkeypatch.delenv("HSA_MODEL_MEMFILE", raising=False)
    factory = SimpleNamespace(getbasetemp=lambda: tmp_path / "worker")

    def paths():
        return [
            fixtures.worker_lock_path.__wrapped__(factory, worker, "0", pytestconfig)
            for worker in ("gw0", "gw1")
        ]

    # The default must still protect a single physical GPU from both workers.
    assert pytestconfig.getoption("--client-lock-scope") == "gpu"
    first, second = paths()
    with filelock.FileLock(first):
        with pytest.raises(filelock.Timeout):
            filelock.FileLock(second).acquire(timeout=0)

    # Explicit local emulation allows both clients to run concurrently.
    monkeypatch.setattr(pytestconfig.option, "client_lock_scope", "worker")
    first, second = paths()
    with filelock.FileLock(first), filelock.FileLock(second, timeout=0):
        assert first != second
