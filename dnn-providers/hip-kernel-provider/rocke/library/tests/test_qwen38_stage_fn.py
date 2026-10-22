# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``kernels.gfx950.qwen38_mono.device_function`` and ``stage_fn`` /
``stage_q2`` (the hybrid path: rocKE stage bodies linked into the FlyDSL mono
layer kernel). CPU only: the emitted LLVM text.

The GPU proofs run on gfx950 in the q38 container with FlyDSL (the harness's
``hybrid/probe_extern.py``: a FlyDSL kernel calling these functions, LDS
modes, inline modes, resources; ``hybrid/ab_extern.py``: the mono QSA layer
with the rocKE Q2 bit-identical to the FlyDSL Q2, plus a mutant that must
differ).
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from kernels.gfx950.qwen38_mono import stage_fn as SF
from kernels.gfx950.qwen38_mono.device_function import (
    DeviceFunctionKnobs,
    lower_device_function,
    write_device_function,
)
from kernels.gfx950.qwen38_mono.mono_layout import LAYER_ARGS, MonoLayout
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm23"


def _kernel(name="t_fn", lds=True):
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 512
    b.kernel.attrs["waves_per_eu"] = 1
    out = b.param("out", PtrType(I32, "global"), noalias=True, align=16)
    x = b.param("x", I32)
    tid = b.thread_id_x()
    v = b.add(tid, x)
    if lds:
        s = b.smem_alloc(I32, [512])
        b.smem_store_vN(s, [tid], v, 1)
        b.s_barrier_bare(fence="compiler")
        v = b.vec_extract(b.smem_load_vN(s, b.xor(tid, b.const_i32(1)), dtype=I32, n=1), 0)
    b.global_store(out, tid, b.wave_reduce(v, "add"), align=4)
    b.ret()
    return b.kernel


def _body(text):
    """The function's block lines (between ``{`` and ``}``)."""
    lines = text.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("define "))
    j = lines.index("}", i)
    return lines[i + 1 : j]


def test_header_abi_and_attributes():
    fn = lower_device_function(_kernel(), arch="gfx950", symbol="rk_t", llvm_flavor=FLAVOR)
    assert fn.abi == (("out", "i64"), ("x", "i32"))
    assert fn.ffi_types() == ["int64", "int32"]
    assert fn.lds_bytes == 2048
    assert "define void @rk_t(i64 %out.abi, i32 %x) #0 {" in fn.ir
    assert "amdgpu_kernel" not in fn.ir
    assert "%out = inttoptr i64 %out.abi to ptr addrspace(1)" in fn.ir
    attrs = next(ln for ln in fn.ir.split("\n") if ln.startswith("attributes #0"))
    assert '"amdgpu-flat-work-group-size"="64,512"' in attrs
    assert "convergent" in attrs.split()  # barriers inside: the call must stay convergent
    for kernel_only in ("uniform-work-group-size", "amdgpu-waves-per-eu", "noalias"):
        assert kernel_only not in fn.ir
    assert re.search(r"^@smem_pool\.t_fn = internal .*addrspace\(3\)", fn.ir, re.M)


def test_body_is_the_kernel_lowering():
    """Apart from the prologue (``inttoptr``s), the body is the kernel's."""
    kern = _lower_kernel_to_llvm_python(_kernel(), arch="gfx950", llvm_flavor=FLAVOR)
    fn = lower_device_function(_kernel(), arch="gfx950", symbol="rk_t", llvm_flavor=FLAVOR)
    fb = [ln for ln in _body(fn.ir) if "inttoptr i64 %out.abi" not in ln]
    assert fb == _body(kern)


def test_kernel_lowering_untouched():
    """Lowering a device function leaves later kernel lowering byte-identical
    (the f64 sqrt declaration lives in the device lowerer's own table)."""
    before = _lower_kernel_to_llvm_python(_kernel(), arch="gfx950", llvm_flavor=FLAVOR)
    lower_device_function(_kernel(), arch="gfx950", llvm_flavor=FLAVOR)
    assert _lower_kernel_to_llvm_python(_kernel(), arch="gfx950", llvm_flavor=FLAVOR) == before
    assert "sqrt.f64" not in before


def test_lds_arg_mode():
    fn = lower_device_function(
        _kernel(), arch="gfx950", symbol="rk_t", llvm_flavor=FLAVOR,
        knobs=DeviceFunctionKnobs(lds="arg"),
    )  # fmt: skip
    assert fn.abi[-1] == ("lds_base", "i32")
    assert "@smem_pool" not in fn.ir
    assert "%lds.pool = inttoptr i32 %lds_base to ptr addrspace(3)" in fn.ir
    assert fn.lds_bytes == 2048
    # no LDS in the body: the parameter is still there (one ABI), unused
    fn0 = lower_device_function(
        _kernel(lds=False), arch="gfx950", llvm_flavor=FLAVOR, knobs=DeviceFunctionKnobs(lds="arg")
    )
    assert fn0.abi[-1] == ("lds_base", "i32") and fn0.lds_bytes == 0


@pytest.mark.parametrize("inline,attr", [("always", "alwaysinline"), ("never", "noinline")])
def test_inline_knob(inline, attr):
    fn = lower_device_function(
        _kernel(), arch="gfx950", llvm_flavor=FLAVOR, knobs=DeviceFunctionKnobs(inline=inline)
    )
    assert attr in next(ln for ln in fn.ir.split("\n") if ln.startswith("attributes #0"))
    assert DeviceFunctionKnobs(inline=inline).suffix() == f"inl{inline}"


def test_bad_inputs():
    with pytest.raises(ValueError):
        DeviceFunctionKnobs(lds="dynamic")
    with pytest.raises(ValueError):
        lower_device_function(_kernel(), arch="gfx950", symbol="bad name", llvm_flavor=FLAVOR)
    b = IRBuilder("f16p")
    b.param("h", PtrType(F32, "global"))
    b.param("bad", F32)
    b.ret()
    lower_device_function(b.kernel, arch="gfx950", llvm_flavor=FLAVOR)  # f32 is C-ABI


def test_f64_sqrt_only_in_device_functions():
    b = IRBuilder("s64")
    b.param("out", PtrType(F32, "global"))
    r = b.sqrt(b.cvt_f32_to_f64(b.const_f32(2.0)))
    b.store(b.cvt_f64_to_f32(r), b.param("o2", PtrType(F32, "global")))
    b.ret()
    fn = lower_device_function(b.kernel, arch="gfx950", llvm_flavor=FLAVOR)
    assert "call double @llvm.sqrt.f64(double" in fn.ir
    assert "declare double @llvm.sqrt.f64(double)" in fn.ir


def test_write_files(tmp_path):
    fn = lower_device_function(_kernel(), arch="gfx950", symbol="rk_w", llvm_flavor=FLAVOR)
    p = write_device_function(fn, str(tmp_path), bitcode=False)
    assert p["bc"] is None
    assert open(p["ll"]).read().startswith("target datalayout")
    import json

    meta = json.load(open(p["json"]))
    assert meta["symbol"] == "rk_w" and meta["abi"] == [["out", "i64"], ["x", "i32"]]
    assert meta["ir_sha256"] == fn.sha256()


# ------------------------------------------------------------------ stage functions

LAYOUT = MonoLayout(
    grid=256, threads=512, max_rows=4, row_pairs=159360,
    sc={"qp": 227584, "qn": 283392}, tasks={"q2": 26},
    qsa_state=("pos", "pages", "key", "value", "index_key", "index_cmp", "cos", "sin",
               "q_norm", "k_norm", "iq_norm", "ik_norm"),
)  # fmt: skip


def _expected_abi():
    return (
        [("tag", "i32"), ("task", "i32"), ("step", "i64"), ("layer", "i32")]
        + [(n, "i64") for n in LAYER_ARGS]
        + [(f"l{i}", "i32") for i in range(4)]
        + [("lds_base", "i32")]
    )


@pytest.mark.parametrize("variant", SF.DEFAULT_VARIANTS)
def test_q2_variants(variant):
    rows, seq, group = SF.parse_variant(variant)
    spec = SF.StageSpec("q2", rows, seq, group)
    fn = SF.lower_stage(spec, LAYOUT, flavor=FLAVOR)
    assert fn.symbol == f"rocke_q38_q2_{variant}"
    assert list(fn.abi) == _expected_abi()
    assert fn.lds_bytes == 0
    ir = fn.ir
    # the mono's arithmetic: f64 square root and divide, no fast-math flags
    assert "call double @llvm.sqrt.f64" in ir and "fdiv double" in ir
    assert not re.search(r"\b(fast|contract|reassoc|afn|arcp|nnan|ninf|nsz)\b", ir)
    # puts are sc1 16-byte stores; polls the sc1 asm loop
    assert "global_store_dwordx4 $0, $1, $2 sc1" in ir
    assert "global_load_dwordx2" in ir and " sc1" in ir
    assert ir.count("call { i32, i32 } @llvm.amdgcn.permlane32.swap") == 1  # the norm's last step


def test_q2_unsupported_variants():
    for spec in (SF.StageSpec("q2", 4, False, 2), SF.StageSpec("q2", 5), SF.StageSpec("q2", 1, True)):
        with pytest.raises(ValueError, match="no rocKE function"):
            SF.build_stage_kernel(spec, LAYOUT)
    with pytest.raises(KeyError):
        SF.build_stage_kernel(SF.StageSpec("zz"), LAYOUT)


def test_q2_knobs_change_only_what_they_name():
    base = SF.lower_stage(SF.StageSpec("q2", 1), LAYOUT, flavor=FLAVOR).ir
    lds = SF.lower_stage(
        SF.StageSpec("q2", 1, knobs=SF.StageKnobs(xshfl="lds")), LAYOUT, flavor=FLAVOR
    ).ir
    assert "update.dpp" in base and "update.dpp" not in lds
    assert "permlane32.swap" not in lds and "ds.bpermute" in lds
    assert SF.StageKnobs(xshfl="lds").suffix() == "xlds"
    assert SF.StageKnobs().suffix() == ""


def test_q2_mutant_differs():
    a = SF.lower_stage(SF.StageSpec("q2", 1), LAYOUT, flavor=FLAVOR).ir
    b = SF.lower_stage(
        SF.StageSpec("q2", 1), replace(LAYOUT, eps_bits=LAYOUT.eps_bits + 1), flavor=FLAVOR
    ).ir
    assert a != b


def test_variant_names():
    assert SF.variant_name(4, True) == "r4s" and SF.parse_variant("r4s") == (4, True, 0)
    assert SF.parse_variant("r8g2") == (8, False, 2)
    with pytest.raises(ValueError):
        SF.parse_variant("x4")


def test_q2_xshfl_dpp2_is_valu_only():
    base = SF.lower_stage(
        SF.StageSpec("q2", 1, knobs=SF.StageKnobs(xshfl="dpp")), LAYOUT, flavor=FLAVOR
    ).ir
    d2 = SF.lower_stage(SF.StageSpec("q2", 1), LAYOUT, flavor=FLAVOR).ir  # default: dpp2
    assert "ds.swizzle" in base and "ds.swizzle" not in d2 and "ds.bpermute" not in d2
    # xor 4: row_shl:4 (0x104) and row_shr:4 (0x114), old = src, full masks
    assert re.search(r"update\.dpp\.i32\(i32 (%\S+), i32 \1, i32 260, i32 15, i32 15, i1 false\)", d2)
    assert re.search(r"update\.dpp\.i32\(i32 (%\S+), i32 \1, i32 276, i32 15, i32 15, i1 false\)", d2)
    assert d2.count("call { i32, i32 } @llvm.amdgcn.permlane16.swap") == 1
    assert d2.count("call { i32, i32 } @llvm.amdgcn.permlane32.swap") == 1
    assert SF.StageKnobs(xshfl="dpp").suffix() == "xdpp"
    assert SF.StageKnobs(xshfl="dpp2").suffix() == ""


def test_q2_tbl_smem_loads_constant_space():
    base = SF.lower_stage(
        SF.StageSpec("q2", 4, True, knobs=SF.StageKnobs(tbl="vmem")), LAYOUT, flavor=FLAVOR
    ).ir
    sm = SF.lower_stage(SF.StageSpec("q2", 4, True), LAYOUT, flavor=FLAVOR).ir  # default: smem
    assert "addrspace(4)" not in base
    n_tbl = len(re.findall(r"load i64, ptr addrspace\(4\) %\S+, align 8, !invariant.load !\{\}", sm))
    assert n_tbl == 8  # pages key value index_key cos sin q_norm k_norm
    assert len(re.findall(r"load i32, ptr addrspace\(4\) %\S+, align 4, !invariant.load", sm)) == 1
    # the rest of the body is the same op for op: only the address space of those 9 loads differs
    strip = lambda t: re.sub(r"addrspace\([14]\)|, !invariant.load !\{\}", "", t)  # noqa: E731
    assert strip(base) == strip(sm)
    assert SF.StageKnobs(tbl="vmem").suffix() == "tvmem"
    assert SF.StageKnobs(tbl="smem").suffix() == ""
    with pytest.raises(ValueError):
        SF.StageKnobs(tbl="x")
    with pytest.raises(ValueError):
        SF.StageKnobs(xshfl="x")


def test_stage_defaults_always_inline():
    """The mono must inline a stage: the default emits ``alwaysinline`` (with
    ``inline="default"`` LLVM's heuristics turned the r4s q2 of one knob set
    into a real call)."""
    for v in ("r1", "r4s"):
        rows, seq, group = SF.parse_variant(v)
        fn = SF.lower_stage(SF.StageSpec("q2", rows, seq, group), LAYOUT, flavor=FLAVOR)
        attrs = next(ln for ln in fn.ir.split(chr(10)) if ln.startswith("attributes #0"))
        assert "alwaysinline" in attrs.split() and "noinline" not in attrs
        assert fn.knobs.inline == "always" and fn.knobs.lds == "arg"
    k = SF.StageKnobs(fn=DeviceFunctionKnobs(lds="arg", inline="default"))
    assert k.suffix() == "inldefault"
    assert SF.StageKnobs(fn=DeviceFunctionKnobs(lds="global", inline="always")).suffix() == "ldsglobal"
    assert SF.StageKnobs(mailbox_impl="ir", tbl="vmem").suffix() == "mbir_tvmem"


@pytest.mark.parametrize("mutant", ["rope_partner", "rope_row", "cache_row"])
def test_q2_mutants(mutant, monkeypatch):
    from kernels.gfx950.qwen38_mono import stage_q2

    assert mutant in stage_q2.MUTANTS
    for v in ("r1", "r4", "r4s"):
        rows, seq, _ = SF.parse_variant(v)
        spec = SF.StageSpec("q2", rows, seq)
        monkeypatch.setattr(stage_q2, "MUTATE", None)
        a = SF.lower_stage(spec, LAYOUT, flavor=FLAVOR).ir
        monkeypatch.setattr(stage_q2, "MUTATE", mutant)
        b = SF.lower_stage(spec, LAYOUT, flavor=FLAVOR).ir
        # rope_row reads row 0's L: the same function at one row
        assert (a == b) == (mutant == "rope_row" and rows == 1), (mutant, v)


def test_q2_page_blocks_power_of_two():
    for page in (8, 32):  # 2 and 8 blocks per page: still powers of two
        SF.lower_stage(SF.StageSpec("q2", 1), replace(LAYOUT, page=page), flavor=FLAVOR)
    with pytest.raises(AssertionError, match="power of two"):
        SF.lower_stage(SF.StageSpec("q2", 1), replace(LAYOUT, page=24), flavor=FLAVOR)
    with pytest.raises(AssertionError):
        SF.lower_stage(SF.StageSpec("q2", 1), replace(LAYOUT, index_compress=8), flavor=FLAVOR)


def test_int_to_ptr_rejects_bad_types():
    b = IRBuilder("t")
    a = b.param("a", I32)
    from rocke.core.ir import I64

    with pytest.raises(TypeError):
        b.int_to_ptr(a, PtrType(I64, "constant"))  # not an i64
    with pytest.raises(TypeError):
        b.int_to_ptr(b.param("q", I64), PtrType(I64, "lds"))


def test_convergent_knob():
    on = lower_device_function(_kernel(), arch="gfx950", symbol="rk_t", llvm_flavor=FLAVOR)
    off = lower_device_function(
        _kernel(), arch="gfx950", symbol="rk_t", llvm_flavor=FLAVOR,
        knobs=DeviceFunctionKnobs(convergent=False),
    )  # fmt: skip
    attr = lambda f: next(ln for ln in f.ir.split(chr(10)) if ln.startswith("attributes #0"))
    assert "convergent" in attr(on).split() and "convergent" not in attr(off).split()
    assert DeviceFunctionKnobs(convergent=False).suffix() == "nocv"
    assert DeviceFunctionKnobs().suffix() == ""


def test_constant_pointer_load_in_kernel_lowering():
    """``int_to_ptr(..., PtrType(t, "constant"))`` loads lower to invariant
    ``addrspace(4)`` loads in the kernel lowering too (a stage body wrapped as a
    framework kernel, path b); global loads keep their text."""
    from rocke.core.ir import I64

    def k(space):
        b = IRBuilder("t_const_" + space)
        b.kernel.attrs["max_workgroup_size"] = 64
        out = b.param("out", PtrType(I32, "global"))
        tbl = b.int_to_ptr(b.param("tbl", I64), PtrType(I32, space))
        b.global_store(out, b.thread_id_x(), b.global_load_i32(tbl, b.const_i32(3)), align=4)
        b.ret()
        return b.kernel

    g = _lower_kernel_to_llvm_python(k("global"), arch="gfx950", llvm_flavor=FLAVOR)
    c = _lower_kernel_to_llvm_python(k("constant"), arch="gfx950", llvm_flavor=FLAVOR)
    assert re.search(r"load i32, ptr addrspace\(1\) %\S+, align 4\n", g)
    assert "addrspace(4)" not in g
    assert re.search(r"inttoptr i64 %\S+ to ptr addrspace\(4\)", c)
    assert re.search(r"load i32, ptr addrspace\(4\) %\S+, align 4, !invariant.load !\{\}", c)
