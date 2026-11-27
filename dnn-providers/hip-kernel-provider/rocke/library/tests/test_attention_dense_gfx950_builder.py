# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 dense emitter matrix, one sharded standard host item per config.

Build the Python reference once per item. Check its canonical IR and all LLVM
flavors against the library differential golden. The installed builder parity
entry supplies the native emitter and also compares the two builders' raw IR
and each .ll byte for byte, without pybind, a GPU, or a compiler. A standalone
source pytest without that executable checks the Python golden only.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from parity.attention_dense_gfx950_emit import (
    _CONFIGS,
    build_attention_dense,
    spec_for_index,
)
from rocke.assets import platform_instances_dir
from rocke.core.golden_file import rerecord_hint
from rocke.core.ir_serialize import canonicalize, serialize
from rocke.core.lower_llvm import LLVM_FLAVORS, lower_kernel_to_llvm
from rocke.tests.lanes import REQUIRE_CPP_ENGINE_ENV

_FAMILY = "attention_dense_gfx950"


@pytest.fixture(scope="module")
def expected():
    # Same schema, flavor aliases and fingerprints as run_diff.py; library ->
    # platform test tooling is the allowed dependency direction. The installed
    # payload supplies this locator through the library conftest.
    sys.path.insert(0, str(platform_instances_dir() / "differential"))
    try:
        from golden_store import GoldenStore, IR_CANONICAL, LL
    finally:
        sys.path.pop(0)
    path = Path(__file__).parent / "parity" / "golden" / "llvm_gfx_all.json"
    store = GoldenStore.load(path)
    hint = rerecord_hint("differential", [_FAMILY])
    assert not store.stale_fingerprints([_FAMILY]), hint
    hashes = {f: store.section(LL, f)[_FAMILY] for f in LLVM_FLAVORS}
    hashes[IR_CANONICAL] = store.section(IR_CANONICAL, None)[_FAMILY]
    indices = {str(i) for i in range(len(_CONFIGS))}
    for table in hashes.values():
        assert set(table) == indices, hint
    return hashes


@pytest.fixture(scope="module")
def native_emitter():
    path = os.environ.get("ROCKE_GFX950_DENSE_EMITTER")
    if path is None:
        if os.environ.get(REQUIRE_CPP_ENGINE_ENV) == "1":
            pytest.fail(
                "native parity required but ROCKE_GFX950_DENSE_EMITTER is missing"
            )
        return None
    return Path(path).resolve(strict=True)


def test_gfx950_dense_emitter_end_of_configs(native_emitter, monkeypatch):
    # run_diff must see the end of both tables, not a failed build at this index.
    monkeypatch.syspath_prepend(str(platform_instances_dir() / "differential"))
    from run_diff import Emission, is_end_emission, python_emitter_env

    parity = Path(__file__).parent / "parity"
    emitters = [[sys.executable, parity / f"{_FAMILY}_emit.py"]]
    if native_emitter is not None:
        emitters.append([native_emitter])
    for command in emitters:
        end = subprocess.run(
            [*command, str(len(_CONFIGS)), "ir"],
            env=python_emitter_env(parity),
            capture_output=True,
            text=True,
            timeout=30,
        )
        result = Emission(end.returncode, bool(end.stdout), None, end.stderr)
        assert is_end_emission(result), (command, result)


def _native_matches(executable, index, mode, expected_bytes, flavor):
    got = subprocess.run(
        [executable, str(index), mode],
        env={**os.environ, "ROCKE_LLVM_FLAVOR": flavor},
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    if got != expected_bytes:
        pytest.fail(
            f"{_FAMILY}[{index}] {mode}/{flavor}: native "
            f"{hashlib.sha256(got).hexdigest()} != Python "
            f"{hashlib.sha256(expected_bytes).hexdigest()}"
        )


@pytest.mark.parametrize("index", range(len(_CONFIGS)))
def test_gfx950_dense_builder_output(index, expected, native_emitter, monkeypatch):
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    kernel = build_attention_dense(spec_for_index(index), arch="gfx950")
    key = str(index)
    hint = rerecord_hint("differential", [_FAMILY])
    assert (
        hashlib.sha256(canonicalize(kernel).encode()).hexdigest()
        == expected["ir_canonical"][key]
    ), hint
    if native_emitter is not None:
        _native_matches(
            native_emitter, index, "ir", serialize(kernel).encode(), LLVM_FLAVORS[0]
        )
    for flavor in LLVM_FLAVORS:
        ll = lower_kernel_to_llvm(kernel, arch="gfx950", llvm_flavor=flavor).encode()
        assert hashlib.sha256(ll).hexdigest() == expected[flavor][key], hint
        if native_emitter is not None:
            _native_matches(native_emitter, index, "ll", ll, flavor)
