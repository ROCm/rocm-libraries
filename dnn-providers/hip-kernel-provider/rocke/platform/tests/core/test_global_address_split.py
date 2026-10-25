# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The global address split (``ISABackend.splits_global_address``).

On a target whose backend turns the switch on (gfx1151 today), a global access
whose i32 element index is wave-uniform terms plus one non-negative, bounded
per-lane term, off a uniform pointer, lowers to a GEP chain -- uniform terms,
then the summed constants, then the per-lane term -- so the AMDGPU backend can
address it with an SGPR base and a VGPR offset. Every other access, and every
access on every other target, keeps the single ``getelementptr inbounds``.

No GPU: text lowering through both engines (the C++ half skips when the engine
is not importable), plus an optional COMGR compile + ``llvm-objdump`` check that
skips when either tool is missing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType
from rocke.core.isa.backend import backend_for, wired_arches
from rocke.helpers.compile import _lower_llvm_via_backend, compile_kernel

_FLAVOR = "llvm22"
_ELEM_LL = {"f16": "half", "f32": "float"}


def _lower(kernel, arch, backend="python"):
    return _lower_llvm_via_backend(
        kernel, arch=arch, backend=backend, spec=None, llvm_flavor=_FLAVOR
    )


def _gather(index, *, elem=F16, max_wg=None, name="split"):
    """``D[tid] = S[index(b, tid, bid, stride)]``."""
    b = IRBuilder(name)
    src = b.param("S", PtrType(elem, "global"), readonly=True)
    dst = b.param("D", PtrType(elem, "global"))
    stride = b.param("stride", I32)
    if max_wg is not None:
        b.kernel.attrs["max_workgroup_size"] = max_wg
    tid, bid = b.thread_id_x(), b.block_id_x()
    v = b.global_load(src, index(b, tid, bid, stride), elem)
    b.global_store(dst, tid, v)
    b.ret()
    return b.kernel


def _row(b, tid, bid, s):
    return b.add(b.mul(bid, s), tid)


def _is_split(ll: str, elem: str = "half") -> bool:
    """Whether the load from %S goes through a split chain."""
    return bool(
        re.search(rf"%gbase\.\d+ = getelementptr {elem}, ptr addrspace\(1\) %S, ", ll)
    )


def _single_gep(ll: str, elem: str = "half") -> bool:
    return bool(
        re.search(rf"= getelementptr inbounds {elem}, ptr addrspace\(1\) %S, i32 %", ll)
    )


def test_switch_is_on_for_gfx1151_only():
    on = [gfx for gfx in wired_arches() if backend_for(gfx).splits_global_address]
    assert on == ["gfx1151"]


def test_uniform_plus_lane_index_splits_on_gfx1151():
    ll = _lower(_gather(_row), "gfx1151")
    base = re.search(
        r"(%gbase\.\d+) = getelementptr half, ptr addrspace\(1\) %S, i32 (%mul\d+)\n", ll
    )
    assert base, ll
    assert re.search(
        rf"= getelementptr half, ptr addrspace\(1\) {re.escape(base.group(1))}, "
        r"i32 %tid\d+\n",
        ll,
    )
    assert not _single_gep(ll)
    # The store's index is the lane term alone: nothing uniform to split off.
    assert re.search(r"= getelementptr inbounds half, ptr addrspace\(1\) %D, i32 %tid", ll)


@pytest.mark.parametrize(
    "arch", [gfx for gfx in wired_arches() if gfx != "gfx1151"]
)
def test_other_targets_keep_the_single_gep(arch):
    ll = _lower(_gather(_row), arch)
    assert "%gbase." not in ll
    assert _single_gep(ll)


def test_constants_come_right_before_the_lane_term():
    def index(b, tid, bid, s):
        return b.add(b.add(tid, b.const_i32(16)), b.mul(bid, s))

    ll = _lower(_gather(index), "gfx1151")
    chain = re.findall(r"= getelementptr half, ptr addrspace\(1\) (%\S+), (i\d+ \S+)\n", ll)
    assert len(chain) == 3, chain
    (base0, index0), (_, index1), (_, index2) = chain
    assert base0 == "%S" and index0.startswith("i32 %mul")
    assert index1 == "i64 16"
    assert index2.startswith("i32 %tid")


@pytest.mark.parametrize(
    "index",
    [
        # two per-lane terms
        lambda b, tid, bid, s: b.add(_row(b, tid, bid, s), b.lane_id()),
        # a per-lane term that can be negative
        lambda b, tid, bid, s: b.add(b.mul(bid, s), b.sub(tid, b.const_i32(1))),
        # a per-lane term with no bound (runtime stride of unknown sign)
        lambda b, tid, bid, s: b.add(b.mul(bid, s), b.mul(tid, s)),
        # a wrapping add is not taken apart
        lambda b, tid, bid, s: b.add_nuw(b.mul(bid, s), tid),
        # a range marker is looked through, not trusted
        lambda b, tid, bid, s: b.add(
            b.mul(bid, s), b._assume_range(b.mul(tid, s), lo=0, hi=1000)
        ),
        # only constants besides the lane term
        lambda b, tid, bid, s: b.add(tid, b.const_i32(64)),
    ],
    ids=["two_lanes", "negative_lane", "unbounded_lane", "nuw_add", "assume_range", "constant_only"],
)
def test_unproven_indices_keep_the_single_gep(index):
    ll = _lower(_gather(index), "gfx1151")
    assert not _is_split(ll)
    assert _single_gep(ll)


def test_per_lane_pointer_keeps_the_single_gep():
    b = IRBuilder("split_lane_ptr")
    src = b.param("S", PtrType(F16, "global"), readonly=True)
    dst = b.param("D", PtrType(F16, "global"))
    stride = b.param("stride", I32)
    tid, bid = b.thread_id_x(), b.block_id_x()
    lane_ptr = b.global_ptr_add(src, b.mul(tid, b.const_i32(2)))
    b.global_store(dst, tid, b.global_load(lane_ptr, _row(b, tid, bid, stride), F16))
    b.ret()
    ll = _lower(b.kernel, "gfx1151")
    assert "%gbase." not in ll


@pytest.mark.parametrize("elem,split", [(F16, True), (F32, False)])
def test_lane_byte_offset_must_stay_below_4_gib(elem, split):
    # tid < 256, so tid * 2**23 < 2**31: within a VGPR offset for 2-byte
    # elements, past 2**32 bytes for 4-byte ones.
    def index(b, tid, bid, s):
        return b.add(b.mul(bid, s), b.mul(tid, b.const_i32(1 << 23)))

    ll = _lower(_gather(index, elem=elem), "gfx1151")
    assert _is_split(ll, _ELEM_LL[elem.name]) is split


@pytest.mark.parametrize("max_wg,split", [(256, True), (1024, False)])
def test_thread_id_bound_comes_from_max_workgroup_size(max_wg, split):
    def index(b, tid, bid, s):
        return b.add(b.mul(bid, s), b.mul(tid, b.const_i32(1 << 22)))

    ll = _lower(_gather(index, elem=F32, max_wg=max_wg), "gfx1151")
    assert _is_split(ll, "float") is split


def _loop_kernel(*, unroll: bool, carry_lane: bool):
    """Sum S over a k loop; the row index is the loop counter (a uniform term)
    plus a carried offset, which is uniform unless it picks up the lane."""
    b = IRBuilder("split_loop")
    src = b.param("S", PtrType(F32, "global"), readonly=True)
    dst = b.param("D", PtrType(F32, "global"))
    stride = b.param("stride", I32)
    n = b.param("n", I32)
    tid, bid = b.thread_id_x(), b.block_id_x()
    upper = b.const_i32(4) if unroll else n
    init = [("acc", b.const_f32(0.0)), ("off", b.mul(bid, stride))]
    loop = b.scf_for_iter(b.const_i32(0), upper, b.const_i32(1), init, unroll=unroll)
    with loop as (k, (acc, off)):
        idx = b.add(b.add(b.mul(k, stride), off), tid)
        step = tid if carry_lane else stride
        b.scf_yield(b.fadd(acc, b.global_load(src, idx, F32)), b.add(off, step))
    b.global_store(dst, tid, loop.results[0])
    b.ret()
    return b.kernel


@pytest.mark.parametrize("unroll", [False, True])
def test_loop_counter_and_uniform_carried_value_are_uniform(unroll):
    ll = _lower(_loop_kernel(unroll=unroll, carry_lane=False), "gfx1151")
    loads = ll.count("= load float, ptr addrspace(1) ")
    splits = len(re.findall(r"= getelementptr float, ptr addrspace\(1\) %gbase\.\d+, i32 %tid", ll))
    assert loads == (4 if unroll else 1)
    assert splits == loads


@pytest.mark.parametrize("unroll", [False, True])
def test_carried_value_updated_per_lane_is_not_uniform(unroll):
    ll = _lower(_loop_kernel(unroll=unroll, carry_lane=True), "gfx1151")
    assert not _is_split(ll, "float")


def _parity_kernels():
    yield "row", _gather(_row)
    yield "lane_bytes", _gather(
        lambda b, tid, bid, s: b.add(b.mul(bid, s), b.mul(tid, b.const_i32(1 << 23)))
    )
    yield "constant", _gather(
        lambda b, tid, bid, s: b.add(b.add(tid, b.const_i32(16)), b.mul(bid, s))
    )
    yield "two_lanes", _gather(lambda b, tid, bid, s: b.add(_row(b, tid, bid, s), b.lane_id()))
    for unroll in (False, True):
        for carry_lane in (False, True):
            yield f"loop_u{int(unroll)}_c{int(carry_lane)}", _loop_kernel(
                unroll=unroll, carry_lane=carry_lane
            )


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201", "gfx950"])
def test_both_engines_emit_the_same_bytes(arch, monkeypatch):
    from rocke.core.backend import BackendError

    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    for name, kernel in _parity_kernels():
        py = _lower(kernel, arch, "python")
        try:
            cpp = _lower(kernel, arch, "cpp")
        except BackendError as e:
            pytest.skip(f"C++ engine not importable: {str(e)[:200]}")
        assert cpp == py, name


def _objdump() -> str | None:
    for candidate in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def test_split_access_selects_an_sgpr_base():
    """The chain reaches the instruction: the gathered load addresses with an
    SGPR pair and a VGPR offset (``global_load_* vD, vOff, s[..]``)."""
    from rocke.runtime import comgr

    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")
    hsaco = compile_kernel(_gather(_row), arch="gfx1151").hsaco
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "split.hsaco"
        path.write_bytes(hsaco)
        isa = subprocess.run(
            [objdump, "-d", "--mcpu=gfx1151", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    loads = [ln.split("//")[0].strip() for ln in isa.splitlines() if "global_load" in ln]
    assert len(loads) == 1, loads
    assert re.search(r", s\[\d+:\d+\]$", loads[0]), loads
