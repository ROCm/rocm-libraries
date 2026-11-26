# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Clients sharing hardware serialize; independent emulators can overlap."""

import importlib
from pathlib import Path

import pytest

with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(Path(__file__).parents[1])
    client_lock = importlib.import_module("client_lock")


def paths(tmp_path, scope, environ):
    return [
        client_lock.client_lock_path(tmp_path, worker, "0", scope, environ)
        for worker in ("gw0", "gw1")
    ]


def test_hardware_default_shares_one_lock_per_gpu(tmp_path):
    first, second = paths(tmp_path, None, {})
    assert first == second == tmp_path / "client_execution_gpu0.lock"


@pytest.mark.parametrize("variable", client_lock.EMULATOR_VARIABLES)
def test_emulator_default_gives_each_worker_its_own_lock(tmp_path, variable):
    first, second = paths(tmp_path, None, {variable: "set"})
    assert first != second


def test_explicit_gpu_scope_wins_over_an_emulator(tmp_path):
    first, second = paths(tmp_path, "gpu", {"HSA_MODEL_MEMFILE": "/dev/shm/model"})
    assert first == second


def test_worker_scope_is_refused_on_hardware(tmp_path):
    with pytest.raises(ValueError, match="race on one GPU"):
        paths(tmp_path, "worker", {})


def test_no_gpu_id_falls_back_to_a_lock_per_worker(tmp_path):
    first = client_lock.client_lock_path(tmp_path, "gw0", None, "gpu", {})
    assert first == tmp_path / "client_execution_gw0.lock"
