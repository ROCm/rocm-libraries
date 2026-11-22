# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""simdSpecDispatch SGPR budget.

The simdSpecDispatch temporaries must go through allocTmpSgpr (which can grow
the SGPR pool up to MaxSgpr). They used to be checked out of the pool with its
non-growing default and overflowed on kernels that define few SGPRs, e.g.
StreamK=3 + StreamKForceDPOnly (TileProcessingStrategy=DataParallel, no
workspace/fixup SGPRs) with CMS.

Whether a real kernel overflows depends on where the free SGPR holes happen to
be when the main loop is emitted, so the pool-level test below reproduces the
failing layout directly. The end-to-end test checks that a kernel of that
family still emits and assembles.
"""

import copy
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool

from tensilelite.KernelWriterAssembly import KernelWriterAssembly

_DATA = Path(__file__).resolve().parent / "simd_spec_dispatch_data"
# Origami SK5 #80 trimmed (MT256x256x64, CMS, AdaptiveGemm=1, AF0EM=256, ASEM=64,
# StreamKForceDPOnly=1).
_LOGIC_80 = _DATA / "gfx950_sk5_fdpo_cms_mt256x256.yaml"


def _amdclang():
    return shutil.which("amdclang++") or (
        "/opt/rocm/bin/amdclang++" if Path("/opt/rocm/bin/amdclang++").exists() else None)


requires_toolchain = pytest.mark.skipif(_amdclang() is None, reason="amdclang++ not available")


def _emit_variant(tmp_path, name, logic, sol=None):
    """Write a copy of a single-solution logic file with solution overrides and emit its kernel."""
    from codegen_harness import emit_kernels_from_logic

    d = copy.deepcopy(yaml.safe_load(logic.read_text()))
    for s in d["Solutions"]:
        s.update(sol or {})
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    res = emit_kernels_from_logic(str(p), canonical=False)
    assert len(res) == 1
    return res[0]


def _assemble(tmp_path, name, src):
    s = tmp_path / f"{name}.s"
    s.write_text(src)
    r = subprocess.run([_amdclang(), "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
                        "-mcpu=gfx950", "-c", str(s), "-o", str(tmp_path / f"{name}.o")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]


class _Writer:
    """Minimal stand-in exposing only what simdSpecDispatch touches."""

    allocTmpSgpr = KernelWriterAssembly.allocTmpSgpr

    def __init__(self, sgprPool):
        self.sgprPool = sgprPool
        self.states = SimpleNamespace(indexChars="IJK", unrollIdx=0,
                                      regCaps={"MaxSgpr": 102}, overflowedResources=0)
        self.db = {"AssertOnSgprOverflow": True}

    def rapLabel(self, name):
        return name


def _tightPool():
    """An SGPR pool like the kernel writer's (overflow prevented by default)
    whose only free registers are s[4:6]. After the 1-SGPR HW_ID temporary
    takes s4, no 2-aligned 3-SGPR block is left without growing the pool."""
    pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=True, printRP=False)
    pool.checkOutAligned(4, 1, "used0", False)            # s[0:3]
    hole = pool.checkOutAligned(3, 1, "hole", False)      # s[4:6]
    pool.checkOutAligned(1, 1, "used7", False)            # s7
    pool.checkIn(hole)
    return pool


@pytest.mark.parametrize("numCodePath", [2, 4])
def test_dispatch_temporaries_grow_a_tight_pool(numCodePath):
    """Regression: the long-branch temporaries used a plain checkOutAligned and
    raised "overflow prevented by preventOverflow flag" on this layout."""
    writer = _Writer(_tightPool())
    kernel = {"AdaptiveGemmNTAB": 0, "ProblemType": {"IndicesSummation": [2]}}

    module = KernelWriterAssembly.simdSpecDispatch(writer, kernel, numCodePath)

    assert writer.states.overflowedResources == 0
    src = str(module)
    assert re.search(r"s_getreg_b32 s4, hwreg\(HW_REG_HW_ID", src)
    assert src.count("label_LoopBeginK_") >= numCodePath
    # All temporaries are returned: only s[0:3] and s7 stay checked out.
    assert writer.sgprPool.available() == writer.sgprPool.size() - 5


@requires_toolchain
def test_force_dp_only_mt256_cms_sk3_emits_and_assembles(tmp_path):
    """#80 as StreamK=3 + ForceDPOnly (normalized to TileProcessingStrategy=
    DataParallel, WorkAssignment=StaticGrid): the simdSpecDispatch temporaries
    fit the SGPR budget."""
    _, src, err = _emit_variant(tmp_path, "fdpo3", _LOGIC_80, sol={"StreamK": 3})
    assert err == 0
    assert re.search(r"_TPSDP_", src) and re.search(r"_WASG_", src)
    assert "AddressFlags" not in src and "AddressWS" not in src
    m = re.search(r"\.amdhsa_next_free_sgpr (\d+)", src)
    assert m and int(m.group(1)) <= 102
    _assemble(tmp_path, "fdpo3", src)
