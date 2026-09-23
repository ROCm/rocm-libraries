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

"""Regressions for StreamK fence/wait rocisa -> StinkyTofu lowering (gfx1250).

Covers the three new gfx1250 instructions used by the StreamK release/acquire
fence sequences:

* ``SWaitXCnt``  -> ``s_wait_xcnt N``      (SOPP_WAIT16, immediate via getSrcParams)
* ``GlobalWb``   -> ``global_wb scope:*``  (SOPP, scope carried via CacheScopeModifiers)
* ``GlobalInv``  -> ``global_inv scope:*`` (SOPP, scope carried via CacheScopeModifiers)

The boilerplate (ISA init fixture, SignatureBase setup, toStinkyTofuModule
invocation) mirrors ``test_mubuf.py`` so that a regression in either the
rocisa-side classes or the stinkytofu-side mapping is caught here.
"""

import re

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.enum import CacheScope
from rocisa.instruction import (
    GlobalInv,
    GlobalWb,
    MacroInstruction,
    SWaitCnt,
    SWaitXCnt,
)

_ISA = (12, 5, 0)

# Skip entire module when the target backend isn't compiled into the registry.
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


def _emit(mod: Module, name: str) -> str:
    """Convert ``mod`` to a stinkytofu module and return the emitted assembly."""
    mod.setParent()  # resolves symbolic register names before conversion

    sig = SignatureBase(
        kernelName=name,
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=0,
        sgprWorkGroup=(1, 1, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=64,
        numSgprPreload=0,
    )

    st = rocisa.toStinkyTofuModule(
        mod, _ISA, name, signature=sig, options={"OptLevel": 0}
    )
    st.runOptimizationPipeline()
    return st.emitAssembly()


# ---------------------------------------------------------------------------
# s_wait_xcnt
# ---------------------------------------------------------------------------

def test_swait_xcnt_zero_emits_to_stinkytofu():
    mod = Module("swait_xcnt_zero")
    mod.add(SWaitXCnt(xcnt=0))

    asm = _emit(mod, "swait_xcnt_zero")

    assert re.search(r"s_wait_xcnt 0", asm), (
        f"expected 's_wait_xcnt 0' in emitted assembly, got:\n{asm}"
    )


def test_swait_xcnt_nonzero_emits_to_stinkytofu():
    mod = Module("swait_xcnt_five")
    mod.add(SWaitXCnt(xcnt=5))

    asm = _emit(mod, "swait_xcnt_five")

    assert re.search(r"s_wait_xcnt 5", asm), (
        f"expected 's_wait_xcnt 5' in emitted assembly, got:\n{asm}"
    )


# ---------------------------------------------------------------------------
# global_wb / global_inv
# ---------------------------------------------------------------------------

def test_global_wb_scope_dev_emits_to_stinkytofu():
    mod = Module("global_wb_scope_dev")
    mod.add(GlobalWb(scope=CacheScope.SCOPE_DEV))

    asm = _emit(mod, "global_wb_scope_dev")

    assert re.search(r"global_wb scope:SCOPE_DEV", asm), (
        f"expected 'global_wb scope:SCOPE_DEV' in emitted assembly, got:\n{asm}"
    )


def test_global_inv_scope_dev_emits_to_stinkytofu():
    mod = Module("global_inv_scope_dev")
    mod.add(GlobalInv(scope=CacheScope.SCOPE_DEV))

    asm = _emit(mod, "global_inv_scope_dev")

    assert re.search(r"global_inv scope:SCOPE_DEV", asm), (
        f"expected 'global_inv scope:SCOPE_DEV' in emitted assembly, got:\n{asm}"
    )


@pytest.mark.parametrize("scope_enum,scope_text", [
    (CacheScope.SCOPE_SE,  "SCOPE_SE"),
    (CacheScope.SCOPE_SYS, "SCOPE_SYS"),
])
def test_global_wb_non_dev_scope_emits_to_stinkytofu(scope_enum, scope_text):
    """Non-DEV scopes flow through the same CacheScopeModifiers path used by SCOPE_DEV.

    Locks the rocisa->stinkytofu scope mapping so a future regression in either
    `convertMUBUFScope` or `addModifiersToInstruction` is caught here.
    """
    mod = Module(f"global_wb_{scope_text.lower()}")
    mod.add(GlobalWb(scope=scope_enum))

    asm = _emit(mod, f"global_wb_{scope_text.lower()}")

    assert re.search(rf"global_wb scope:{scope_text}", asm), (
        f"expected 'global_wb scope:{scope_text}' in emitted assembly, got:\n{asm}"
    )


# ---------------------------------------------------------------------------
# Byte-for-byte equivalence with the legacy MacroInstruction text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("typed,legacy_text", [
    (SWaitXCnt(xcnt=0),                                  "s_wait_xcnt 0"),
    (GlobalWb(scope=CacheScope.SCOPE_DEV),               "global_wb scope:SCOPE_DEV"),
    (GlobalInv(scope=CacheScope.SCOPE_DEV),              "global_inv scope:SCOPE_DEV"),
])
def test_typed_instruction_matches_legacy_macro_str(typed, legacy_text):
    """Lock the rocisa __str__ output of the new typed instructions to the
    exact text the previous `MacroInstruction(name=..., args=[])` calls in
    `Tensile/Components/StreamK.py` produced. If this fails, the StreamK
    consumer change has drifted from the byte-for-byte invariant and the
    generated kernels will differ.
    """
    legacy = MacroInstruction(name=legacy_text, args=[])
    assert str(typed) == str(legacy), (
        f"typed={str(typed)!r} legacy={str(legacy)!r}"
    )


# ---------------------------------------------------------------------------
# StreamK release / acquire fence sequences
# ---------------------------------------------------------------------------

def test_streamk_release_fence_sequence():
    """Release fence: ``global_wb scope:SCOPE_DEV`` with the StreamK comment.

    Mirrors ``StreamKMemoryOrderingDevScopeFences.releaseFence`` minus the
    surrounding ``SWaitCnt``s (which are not part of the new instruction set
    under test).
    """
    mod = Module("streamk_release_fence")
    mod.add(GlobalWb(
        scope=CacheScope.SCOPE_DEV,
        comment="release: writeback partials to L2-coherent point",
    ))

    asm = _emit(mod, "streamk_release_fence")

    assert re.search(r"global_wb scope:SCOPE_DEV", asm), (
        f"expected 'global_wb scope:SCOPE_DEV' in emitted assembly, got:\n{asm}"
    )
    # Allow other content / whitespace between the mnemonic and the comment,
    # but stay on the same line (no DOTALL) to avoid matching unrelated text.
    assert re.search(
        r"global_wb scope:SCOPE_DEV.*release: writeback partials to L2-coherent point",
        asm,
    ), (
        "expected release-fence comment to appear next to the global_wb "
        f"line, got:\n{asm}"
    )


def test_streamk_acquire_fence_sequence():
    """Acquire fence: ``global_inv scope:SCOPE_DEV`` with the StreamK comment."""
    mod = Module("streamk_acquire_fence")
    mod.add(GlobalInv(
        scope=CacheScope.SCOPE_DEV,
        comment="acquire: invalidate partials after flag",
    ))

    asm = _emit(mod, "streamk_acquire_fence")

    assert re.search(r"global_inv scope:SCOPE_DEV", asm), (
        f"expected 'global_inv scope:SCOPE_DEV' in emitted assembly, got:\n{asm}"
    )
    assert re.search(
        r"global_inv scope:SCOPE_DEV.*acquire: invalidate partials after flag",
        asm,
    ), (
        "expected acquire-fence comment to appear next to the global_inv "
        f"line, got:\n{asm}"
    )


def test_streamk_full_release_sequence_preserves_ordering():
    """The full release-fence sequence must keep the ``global_wb`` between the
    pre- and post-fence ``s_wait_*cnt`` instructions after stinkytofu's DAG
    scheduler runs. The ``HasSideEffect`` flag on ``GlobalWbInst`` is what
    prevents the scheduler from reordering across the fence; this test locks
    that contract.
    """
    mod = Module("streamk_full_release_fence")
    mod.add(SWaitCnt(vlcnt=0,
        comment="release: drain in-flight loads before global_wb"))
    mod.add(SWaitCnt(vscnt=0, comment="wait for data store"))
    mod.add(GlobalWb(scope=CacheScope.SCOPE_DEV,
        comment="release: writeback partials to L2-coherent point"))
    mod.add(SWaitCnt(vlcnt=0, vscnt=0,
        comment="release: wait for global_wb"))

    asm = _emit(mod, "streamk_full_release_fence")

    # Verify the global_wb appears AFTER a pre-fence drain wait and BEFORE
    # the post-fence wait; use re.DOTALL because we are matching across lines.
    assert re.search(
        r"s_wait_(loadcnt|storecnt) 0.*global_wb scope:SCOPE_DEV.*s_wait_(loadcnt|storecnt) 0",
        asm,
        re.DOTALL,
    ), (
        "expected global_wb to remain between drain and post-fence waits, "
        f"got:\n{asm}"
    )
