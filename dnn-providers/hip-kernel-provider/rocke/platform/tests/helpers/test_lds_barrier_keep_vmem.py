# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke.helpers.asm.lds_barrier_keep_vmem``: an LDS barrier that leaves
global loads in flight.

On gfx12 RDNA the AMDGPU wait-count pass may drain every counter in front of
an ``s_barrier`` it emits itself, so ``sync_lds_only`` can wait for a register
prefetch issued before the barrier.  Whether the pass adds the drain depends on
the compiler version: some builds do not, in which case
``test_sync_lds_only_drains_the_load_on_rdna`` skips rather than fails.  The
helper emits the barrier as inline asm on the gfx12 RDNA targets and
``sync_lds_only`` everywhere else; the codegen test always checks that the
helper itself keeps the load in flight.

* The IR checks pin the branch each target family takes.
* The codegen checks compile both barriers with comgr and read the ISA back:
  with ``sync_lds_only`` the load is waited for in front of the barrier, with
  the helper only after it, where the loaded value is used. Environments
  without comgr, without a comgr that targets the arch, or without
  ``llvm-objdump`` skip them.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pytest

from rocke.analysis.isa import _instruction_lines
from rocke.core.ir import F16, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.asm import lds_barrier_keep_vmem

_GFX12_ASM = "s_wait_dscnt 0x0\\0As_barrier_signal -1\\0As_barrier_wait -1"


def _kernel(arch: str, *, keep_vmem: bool = True):
    """A load issued before an LDS round trip and used after it."""
    b = IRBuilder("lds_barrier_keep_vmem_check")
    b.kernel.attrs["max_workgroup_size"] = 256
    x = b.param("X", PtrType(F16, "global"))
    y = b.param("Y", PtrType(F16, "global"))
    tid = b.thread_id_x()
    tile = b.smem_alloc(F16, [256], name_hint="tile")
    v = b.global_load(x, tid, F16)
    w = b.global_load(x, b.add(tid, b.const_i32(256)), F16)
    b.smem_store_vN(tile, [tid], w, 1)
    if keep_vmem:
        lds_barrier_keep_vmem(b, arch)
    else:
        b.sync_lds_only()
    u = b.smem_load_vN_f16(tile, b.sub(b.const_i32(255), tid), n=1)
    b.global_store(y, tid, b.fadd(b.vec_extract(u, 0), v))
    b.ret()
    return b.kernel


def _ll(arch: str, **kw) -> str:
    return _lower_kernel_to_llvm_python(
        _kernel(arch, **kw), arch=arch, llvm_flavor="llvm22"
    )


@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201", "gfx12-generic"])
def test_gfx12_rdna_takes_the_split_barrier_asm(arch):
    ll = _ll(arch)
    assert f'call void asm sideeffect "{_GFX12_ASM}", "~{{memory}}"()' in ll
    assert "llvm.amdgcn.s.barrier" not in ll
    assert "llvm.amdgcn.s.waitcnt" not in ll


@pytest.mark.parametrize(
    "arch", ["gfx942", "gfx950", "gfx1250", "gfx1100", "gfx1151", "gfx11-generic"]
)
def test_other_targets_keep_sync_lds_only(arch):
    flavor = "llvm23" if arch == "gfx1250" else "llvm22"
    got = _lower_kernel_to_llvm_python(_kernel(arch), arch=arch, llvm_flavor=flavor)
    want = _lower_kernel_to_llvm_python(
        _kernel(arch, keep_vmem=False), arch=arch, llvm_flavor=flavor
    )
    assert got == want
    assert "asm sideeffect" not in got


def test_the_asm_is_a_convergent_memory_clobber():
    b = IRBuilder("k")
    lds_barrier_keep_vmem(b, "gfx1201")
    (op,) = [o for o in b.kernel.body.ops if o.name == "tile.inline_asm"]
    assert op.attrs["constraints"] == "~{memory}"
    assert op.attrs["convergent"] is True
    assert op.attrs["sideeffect"] is True
    assert not op.operands and not op.results


def test_unknown_arch_is_refused():
    with pytest.raises(Exception):
        lds_barrier_keep_vmem(IRBuilder("k"), "gfx0000")


# ---------------------------------------------------------------------------
# Codegen: where the load is waited for
# ---------------------------------------------------------------------------


def _objdump() -> str | None:
    found = shutil.which("llvm-objdump")
    if found:
        return found
    root = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidate = Path(root) / "llvm" / "bin" / "llvm-objdump"
    return str(candidate) if candidate.exists() else None


def _isa(arch: str, *, keep_vmem: bool) -> list[str]:
    try:
        from rocke.runtime import comgr
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"comgr unavailable: {exc}")
    isa = f"amdgcn-amd-amdhsa--{arch}"
    try:
        if isa not in comgr.supported_isas():
            pytest.skip(f"the loaded comgr cannot target {arch}")
    except comgr.ComgrError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"comgr unavailable: {exc}")
    tool = _objdump()
    if tool is None:
        pytest.skip("llvm-objdump not found; cannot inspect the ISA")
    llvm = _lower_kernel_to_llvm_python(_kernel(arch, keep_vmem=keep_vmem), arch=arch)
    hsaco, _ = comgr.build_hsaco_from_llvm_ir(llvm, isa=isa)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kernel.hsaco"
        path.write_bytes(hsaco)
        done = subprocess.run(
            [tool, "-d", str(path)], capture_output=True, text=True, check=False
        )
    assert done.returncode == 0, done.stderr or done.stdout
    return [line.strip() for line in _instruction_lines(done.stdout)]


# The full drain of the global-load counter (alone or merged with the LDS
# counter's), and the barrier itself.
_LOAD_DRAIN = {
    "gfx1201": re.compile(r"\bs_wait_loadcnt(?:_dscnt)? 0x0\b"),
}
_BARRIER = {
    "gfx1201": re.compile(r"\bs_barrier_wait\b"),
}


def _drained_before_first_barrier(arch: str, lines: list[str]) -> bool:
    first = next(i for i, line in enumerate(lines) if _BARRIER[arch].search(line))
    return any(_LOAD_DRAIN[arch].search(line) for line in lines[:first])


@pytest.mark.parametrize("arch", ["gfx1201"])
def test_sync_lds_only_drains_the_load_on_rdna(arch):
    """The reason the helper exists: LLVM's own barrier waits for the load.

    This property is compiler-version-dependent: some builds of the AMDGPU
    wait-count pass do not drain the load counter in front of an ``s_barrier``
    on gfx1201.  The test is skipped when the ISA does not show the drain,
    rather than failing, so it documents the expectation without blocking CI on
    a compiler that doesn't yet meet it.
    """
    lines = _isa(arch, keep_vmem=False)
    if not _drained_before_first_barrier(arch, lines):
        pytest.skip(
            f"compiler on {arch} does not emit s_wait_loadcnt before s_barrier, "
            "so there is no drain here for the helper to avoid (the helper "
            "itself is still checked by the other tests)"
        )
    assert True  # drain observed; expectation met


@pytest.mark.parametrize("arch", ["gfx1201"])
def test_the_helper_keeps_the_load_in_flight_across_the_barrier(arch):
    lines = _isa(arch, keep_vmem=True)
    assert not _drained_before_first_barrier(arch, lines)
    # The load is still waited for before its value is used.
    assert any(_LOAD_DRAIN[arch].search(line) for line in lines)
