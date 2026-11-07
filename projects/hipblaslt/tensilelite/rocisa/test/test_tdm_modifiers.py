################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""tensor_load_to_lds cache-policy modifiers (th / scope / nv) in rocisa and the
rocisa -> StinkyTofu lowering."""

import copy
import pickle

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.container import TENSORModifiers, sgpr
from rocisa.enum import CacheScope, NonVolatile, TemporalHint
from rocisa.instruction import TensorLoadToLds

_ISA = (12, 5, 0)

pytestmark = pytest.mark.skipif(
    not rocisa.isSupportedByStinkyTofu(_ISA),
    reason=f"gfx{''.join(str(v) for v in _ISA)} not registered in StinkyTofu BackendRegistry",
)


@pytest.fixture(scope="module", autouse=True)
def _isa_context():
    import os
    import shutil

    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    rocisa.rocIsa.getInstance().init(_ISA, assembler, False)
    rocisa.rocIsa.getInstance().setKernel(_ISA, 32)


def _load(tensor=None, four_groups=False):
    group2 = sgpr(12, 4) if four_groups else None
    group3 = sgpr(16, 4) if four_groups else None
    return TensorLoadToLds(sgpr(0, 4), sgpr(4, 8), group2, group3, tensor=tensor)


_CASES = [
    (None, False, "tensor_load_to_lds s[0:3], s[4:11]"),
    (TENSORModifiers(), False, "tensor_load_to_lds s[0:3], s[4:11]"),
    (TENSORModifiers(scope=CacheScope.SCOPE_SE, th=TemporalHint.TH_NT_HT, nv=NonVolatile.NV), False,
     "tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_NT_HT scope:SCOPE_SE nv"),
    (TENSORModifiers(th=TemporalHint.TH_NT), False, "tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_NT"),
    (TENSORModifiers(scope=CacheScope.SCOPE_DEV), False, "tensor_load_to_lds s[0:3], s[4:11] scope:SCOPE_DEV"),
    (TENSORModifiers(nv=NonVolatile.NV), False, "tensor_load_to_lds s[0:3], s[4:11] nv"),
    (TENSORModifiers(th=TemporalHint.TH_HT, scope=CacheScope.SCOPE_SYS), True,
     "tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_LOAD_HT scope:SCOPE_SYS"),
]


@pytest.mark.parametrize("tensor, four_groups, expected", _CASES)
def test_rocisa_tostring(tensor, four_groups, expected):
    assert str(_load(tensor, four_groups)) == expected + "\n"


def test_modifier_survives_copy_and_pickle():
    mods = TENSORModifiers(scope=CacheScope.SCOPE_DEV, th=TemporalHint.TH_NT, nv=NonVolatile.NV)
    inst = _load(mods)
    expected = "tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_NT scope:SCOPE_DEV nv\n"
    assert str(copy.deepcopy(inst)) == expected
    back = pickle.loads(pickle.dumps(mods))
    assert (back.scope, back.th, back.nv, back.isStore) == (
        CacheScope.SCOPE_DEV, TemporalHint.TH_NT, NonVolatile.NV, False)
    assert inst.tensor is not None and inst.tensor.th == TemporalHint.TH_NT


@pytest.fixture(scope="module")
def _stinky_asm() -> str:
    mod = Module("tdm_cache_policy")
    for tensor, four_groups, _ in _CASES:
        mod.add(_load(tensor, four_groups))
    mod.setParent()
    sig = SignatureBase(
        kernelName="tdm_cache_policy",
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=0,
        sgprWorkGroup=(1, 1, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=64,
        numSgprPreload=0,
    )
    st = rocisa.toStinkyTofuModule(mod, _ISA, "tdm_cache_policy", signature=sig, options={"OptLevel": 0})
    st.runOptimizationPipeline()
    return st.emitAssembly()


def test_stinkytofu_lowering_keeps_cache_policy(_stinky_asm):
    lines = [l.split("//")[0].strip() for l in _stinky_asm.splitlines()
             if l.strip().startswith("tensor_load_to_lds")]
    assert lines == [expected for _, _, expected in _CASES]
