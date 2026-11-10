# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen checks for TriggeredGateEpilogue kernels on gfx942 and gfx950."""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit


def _config(arch):
    return os.path.normpath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
        "common", "gemm", f"gate_epilogue_{arch}.yaml"))


_MARKERS = (
    "s_trap 2",
    "buffer_wbl2 sc0 sc1",
    "// triggered gate: ready += 1",
)
_ATOMIC = re.compile(r"^\s*global_atomic_add_x2 \S+, \S+, s\[sgprTrigGateReadyAddr:sgprTrigGateReadyAddr\+1\] sc1\b", re.M)
_D_STORE = re.compile(r"^\s*buffer_store_\S+ .*sgprSrdD.*$", re.M)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_gate_epilogue_emits_assembly(arch):
    results = emit_kernels_from_config(_config(arch), limit=8, arch=arch)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), [(b, e) for (b, _s, e) in results if e != 0]
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "_TGE_" in base, f"Kernel {base!r} lacks the TGE name token"
        missing = [m for m in _MARKERS if m not in src]
        assert not missing, f"Kernel {base!r}: missing {missing}"
        atomics = _ATOMIC.findall(src)
        assert atomics and len(atomics) == src.count("buffer_wbl2 sc0 sc1"), (
            f"Kernel {base!r}: every gate add must follow its own write-back")
        stores = _D_STORE.findall(src)
        assert stores, f"Kernel {base!r}: no D stores found"
        no_sc1 = [s for s in stores if " sc1" not in s]
        assert not no_sc1, f"Kernel {base!r}: D stores without sc1: {no_sc1[:3]}"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_gate_epilogue_golden(arch, snapshot):
    results = emit_kernels_from_config(_config(arch), limit=8, arch=arch)
    digest = sorted(({"basename": b, "err": e} for (b, _s, e) in results), key=lambda d: d["basename"])
    assert digest == snapshot
