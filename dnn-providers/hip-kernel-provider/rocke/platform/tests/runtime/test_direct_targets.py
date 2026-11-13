# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The direct code-object backend compiles for the targets its library reports,
refuses the others and the operations it does not lower as DirectUnsupported,
and compile_kernel then builds the kernel with COMGR."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile

import pytest
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.runtime import code_object
from rocke.runtime.code_object import CodeObjectError, DirectUnsupported
from rocke.runtime.direct_session import DirectSession

GFX950 = "amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-"

_PROBE = """ckdsl.ir v1
kernel @target_probe {
  attrs { max_workgroup_size = i:64 }
  params {
    %Y : ptr<f32,global>
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
BODY    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4 }
    cf.return (  )
  }
}
"""
_SUPPORTED = _PROBE.replace("BODY", "    %e : f32 = arith.bitcast ( %tid )\n")
_PERMLANE = _PROBE.replace(
    "BODY",
    "    %p : i32 = tile.permlane64 ( %tid )\n    %e : f32 = arith.bitcast ( %p )\n",
)


_LOWERED = """ckdsl.ir v1
kernel @lowered_probe {
  attrs { max_workgroup_size = i:64 }
  params {
    %Y : ptr<bf16,global>
    %Z : ptr<i32,global>
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
    %c2 : i32 = arith.constant (  ) { ity = s:"i32", value = i:2 }
    %i : i32 = arith.mul ( %tid, %c2 )
    %v : vec<bf16x2> = memref.global_load_vN ( %Y, %i ) { align = i:4, elem_type = s:"bf16", vec = i:2 }
    %old : vec<bf16x2> = memref.global_atomic_add_pk_bf16 ( %Y, %i, %v ) { elem_type = s:"bf16", ordering = s:"monotonic", vec = i:2 }
    %w : i32 = arith.bitcast ( %old )
    %m : i32 = tile.dpp_row_mirror ( %w ) { ctrl = i:320 }
    %h : i32 = tile.dpp_row_mirror ( %m ) { ctrl = i:321 }
    memref.global_store_typed ( %Z, %tid, %h ) { align = i:4, elem_type = s:"i32" }
    cf.return (  )
  }
}
"""


def _require_direct_library() -> None:
    if code_object._direct_library_path() is None:
        pytest.skip("build the Rust cdylib or set ROCKE_CODEOBJ_DIRECT_LIB")
    if "gfx950" not in code_object.direct_supported_targets():
        pytest.skip("the direct library predates gfx950")


def _e_flags(hsaco: bytes) -> int:
    assert hsaco.startswith(b"\x7fELF")
    return struct.unpack_from("<I", hsaco, 48)[0]


def test_the_library_reports_its_targets() -> None:
    _require_direct_library()
    assert code_object.direct_supported_targets() == ["gfx1151", "gfx950"]
    assert "amdgcn-amd-amdhsa--gfx950" in code_object.supported_isas("direct")
    info = code_object.loaded_compiler_info("direct")
    assert info is not None and info.details["supported_targets"] == [
        "gfx1151",
        "gfx950",
    ]


def test_gfx950_compiles_through_the_target_entry() -> None:
    _require_direct_library()
    hsaco, _timings, _info, diagnostics = code_object.build_direct_rocke_ir_hsaco(
        _SUPPORTED, isa=GFX950
    )
    assert _e_flags(hsaco) == 0x54F
    assert diagnostics.startswith("instructions=")
    gfx1151, _timings, _info, _diagnostics = code_object.build_direct_rocke_ir_hsaco(
        _SUPPORTED
    )
    assert _e_flags(gfx1151) != 0x54F


def test_unsupported_targets_and_operations_are_direct_unsupported() -> None:
    _require_direct_library()
    with pytest.raises(DirectUnsupported, match="gfx942"):
        code_object.build_direct_rocke_ir_hsaco(
            _SUPPORTED, isa="amdgcn-amd-amdhsa--gfx942"
        )
    with pytest.raises(DirectUnsupported, match="ROCKE_DIRECT_UNSUPPORTED"):
        code_object.build_direct_rocke_ir_hsaco(_PERMLANE, isa=GFX950)
    # gfx1151 lowers the same operation.
    code_object.build_direct_rocke_ir_hsaco(_PERMLANE)
    # Broken IR is an error, not an unsupported kernel.
    broken = _SUPPORTED.replace("( %tid )\n", "( %nope )\n")
    with pytest.raises(CodeObjectError) as raised:
        code_object.build_direct_rocke_ir_hsaco(broken, isa=GFX950)
    assert not isinstance(raised.value, DirectUnsupported)


def test_packed_bf16_atomic_and_dpp_row_mirror_lower_on_gfx950() -> None:
    _require_direct_library()
    hsaco, _timings, _info, _diagnostics = code_object.build_direct_rocke_ir_hsaco(
        _LOWERED, isa=GFX950
    )
    assert _e_flags(hsaco) == 0x54F
    objdump = shutil.which("llvm-objdump") or "/opt/rocm/llvm/bin/llvm-objdump"
    if not os.path.exists(objdump):
        return
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
        f.write(hsaco)
        f.flush()
        text = subprocess.run(
            [objdump, "-d", "--mcpu=gfx950", f.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    assert "global_atomic_pk_add_bf16" in text
    assert "row_mirror" in text and "row_half_mirror" in text
    # gfx1151 has no packed bf16 atomic.
    with pytest.raises(DirectUnsupported):
        code_object.build_direct_rocke_ir_hsaco(_LOWERED)


def test_sessions_compile_for_their_target() -> None:
    _require_direct_library()
    plain = code_object.build_direct_rocke_ir_hsaco(_SUPPORTED, isa=GFX950)[0]
    with DirectSession(worker_threads=1, isa=GFX950) as session:
        assert session.compile(_SUPPORTED)[0] == plain
        with pytest.raises(DirectUnsupported):
            session.compile(_PERMLANE)
    with pytest.raises(DirectUnsupported):
        DirectSession(isa="amdgcn-amd-amdhsa--gfx942")


def _scale_kernel():
    b = IRBuilder("direct_target_scale")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    y = b.param("Y", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    b.global_store(y, tid, b.fmul(b.global_load_f32(x, tid), b.const_f32(2.0)), align=4)
    return b.kernel


def _asm_kernel():
    # tile.inline_asm with an instruction of its own syntax (SDWA): COMGR
    # assembles it; the direct compiler's assembler refuses it.
    b = IRBuilder("direct_target_asm")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    y = b.param("Y", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    moved = b.inline_asm(
        "v_mov_b32_sdwa $0, $1 dst_sel:WORD_1", "=v,v", [b.global_load_f32(x, tid)], F32
    )
    b.global_store(y, tid, moved, align=4)
    return b.kernel


def test_compile_kernel_falls_back_to_comgr_for_what_direct_refuses() -> None:
    _require_direct_library()
    if code_object.loaded_compiler_info("comgr") is None:
        pytest.skip("COMGR is not available")
    direct = compile_kernel(
        _scale_kernel(), arch="gfx950", code_object_backend="direct"
    )
    assert (direct.code_object_backend, direct.direct_fallback) == ("direct", "")
    assert _e_flags(direct.hsaco) == 0x54F
    fallback = compile_kernel(
        _asm_kernel(), arch="gfx950", code_object_backend="direct"
    )
    assert fallback.code_object_backend == "comgr"
    assert "ROCKE_DIRECT_UNSUPPORTED" in fallback.direct_fallback
    assert fallback.llvm_text and b"direct_target_asm" in fallback.hsaco
    assert _e_flags(fallback.hsaco) & 0xFF == 0x4F
    # A target the library does not compile for goes to COMGR too.
    other = compile_kernel(_scale_kernel(), arch="gfx942", code_object_backend="direct")
    assert other.code_object_backend == "comgr" and "gfx942" in other.direct_fallback


def _escaping_constant_kernel(hoisted: bool):
    """A constant first made in a loop body and read after the loop (a builder
    that caches constants across scopes); ``hoisted`` makes it before the loop."""
    b = IRBuilder("direct_const_scope")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(I32, "global"), noalias=True, readonly=True, align=16)
    y = b.param("Y", PtrType(I32, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    lo, hi, step = b.const_i32(0), b.const_i32(4), b.const_i32(1)
    c = b.const_i32(1234) if hoisted else None
    with b.scf_for(lo, hi, step, iv_name="k") as k:
        c = b.const_i32(1234) if c is None else c
        b.global_store(y, b.add(tid, b.mul(k, c)), tid, align=4)
    b.global_store(y, tid, b.select(b.cmp_lt(tid, b.const_i32(9)), tid, c), align=4)
    b.ret()
    return b.kernel


def test_a_constant_read_after_its_loop_compiles_as_if_made_before_it() -> None:
    """The LLVM lowering inlines constants, so a constant made in a loop body may be reused after it;
    the direct parser scopes names by region and moves such a constant to the kernel's start.
    """
    _require_direct_library()
    from rocke.core.ir_serialize import serialize

    escaping = serialize(_escaping_constant_kernel(hoisted=False))
    hsaco = code_object.build_direct_rocke_ir_hsaco(escaping, isa=GFX950)[0]
    assert _e_flags(hsaco) == 0x54F
    # same instructions as the constant made first (the hoisted text names it differently, so compare sizes)
    hoisted = serialize(_escaping_constant_kernel(hoisted=True))
    assert len(code_object.build_direct_rocke_ir_hsaco(hoisted, isa=GFX950)[0]) == len(
        hsaco
    )


def test_an_unencodable_dpp_xor_mask_is_an_error_not_a_swizzle() -> None:
    _require_direct_library()
    from rocke.core.ir_serialize import serialize

    def kernel(mask: int):
        b = IRBuilder(f"direct_dpp_mask{mask}")
        b.kernel.attrs["max_workgroup_size"] = 64
        x = b.param("X", PtrType(I32, "global"), noalias=True, readonly=True, align=16)
        y = b.param("Y", PtrType(I32, "global"), noalias=True, writeonly=True, align=16)
        tid = b.thread_id_x()
        b.global_store(
            y,
            tid,
            b.warp_shuffle_xor_dpp(b.global_load(x, tid, I32, align=4), mask),
            align=4,
        )
        b.ret()
        return serialize(b.kernel)

    code_object.build_direct_rocke_ir_hsaco(kernel(15), isa=GFX950)
    with pytest.raises(Exception, match="no register-only DPP encoding"):
        code_object.build_direct_rocke_ir_hsaco(kernel(4), isa=GFX950)


def test_a_kernels_direct_levers_apply_to_its_compile_only(monkeypatch) -> None:
    """``attrs["direct_levers"]`` joins ROCKE_DIRECT_LEVERS for that one compile, as setting it would."""
    import os

    _require_direct_library()
    monkeypatch.delenv("ROCKE_DIRECT_LEVERS", raising=False)
    tagged = _scale_kernel()
    tagged.attrs["direct_levers"] = "litreuse+litshare"
    via_attr = compile_kernel(tagged, arch="gfx950", code_object_backend="direct")
    assert via_attr.code_object_backend == "direct", via_attr.direct_fallback
    assert "ROCKE_DIRECT_LEVERS" not in os.environ
    monkeypatch.setenv("ROCKE_DIRECT_LEVERS", "litreuse,litshare")
    via_env = compile_kernel(
        _scale_kernel(), arch="gfx950", code_object_backend="direct"
    )
    assert via_attr.hsaco == via_env.hsaco
    # The environment's levers stay when the kernel names its own.
    tagged = _scale_kernel()
    tagged.attrs["direct_levers"] = ["minmax"]
    compile_kernel(tagged, arch="gfx950", code_object_backend="direct")
    assert os.environ["ROCKE_DIRECT_LEVERS"] == "litreuse,litshare"


def test_strict_mode_turns_a_refusal_into_an_error(monkeypatch) -> None:
    """``ROCKE_DIRECT_STRICT=1``: a kernel the direct compiler refuses raises instead of falling back."""
    _require_direct_library()
    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    with pytest.raises(DirectUnsupported):
        compile_kernel(_asm_kernel(), arch="gfx950", code_object_backend="direct")


def _scoped_scale_kernel(levers):
    b = IRBuilder("direct_target_scale")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    y = b.param("Y", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    with b.lever_scope(levers):
        tid = b.thread_id_x()
        b.global_store(
            y, tid, b.fmul(b.global_load_f32(x, tid), b.const_f32(2.0)), align=4
        )
    return b.kernel


def test_lever_scopes_mark_their_ops_and_nothing_else() -> None:
    """``IRBuilder.lever_scope`` puts ``dlevers`` on the ops emitted inside only; an empty scope adds nothing."""
    from rocke.core.ir_serialize import serialize

    kernel = _scoped_scale_kernel(["nofence", "-loadorder"])
    scoped = [op for op in kernel.body.ops if "dlevers" in op.attrs]
    assert scoped and all(op.attrs["dlevers"] == "nofence,-loadorder" for op in scoped)
    assert all("dlevers" not in op.attrs for op in kernel.body.ops if op not in scoped)
    assert 'dlevers = s:"nofence,-loadorder"' in serialize(kernel)
    assert serialize(_scoped_scale_kernel(None)) == serialize(_scale_kernel())
    assert serialize(_scoped_scale_kernel("")) == serialize(_scale_kernel())
    # Nested scopes: the inner one wins inside, the outer one is back after it.
    b = IRBuilder("nested")
    with b.lever_scope("a"):
        b.thread_id_x()
        with b.lever_scope("b+c"):
            b.thread_id_x()
        b.thread_id_x()
    assert [op.attrs.get("dlevers") for op in b.kernel.body.ops] == ["a", "b,c", "a"]


def test_lever_scopes_leave_the_llvm_lowering_alone() -> None:
    """COMGR builds ignore the scopes: the same LLVM text with and without."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    plain = lower_kernel_to_llvm(_scale_kernel(), arch="gfx950", llvm_flavor="llvm22")
    scoped = lower_kernel_to_llvm(
        _scoped_scale_kernel("nofence,litreuse"), arch="gfx950", llvm_flavor="llvm22"
    )
    assert plain == scoped


def test_a_scope_over_every_op_is_the_kernels_levers(monkeypatch) -> None:
    """Scoping every op with levers L builds what ``ROCKE_DIRECT_LEVERS=L`` does; ``-L`` scopes undo L."""
    _require_direct_library()
    monkeypatch.delenv("ROCKE_DIRECT_LEVERS", raising=False)
    levers = "litreuse,litshare,minmax,vmemfirst"
    scoped = compile_kernel(
        _scoped_scale_kernel(levers), arch="gfx950", code_object_backend="direct"
    )
    assert scoped.code_object_backend == "direct", scoped.direct_fallback
    monkeypatch.setenv("ROCKE_DIRECT_LEVERS", levers)
    via_env = compile_kernel(
        _scale_kernel(), arch="gfx950", code_object_backend="direct"
    )
    assert scoped.hsaco == via_env.hsaco
    off = ",".join("-" + x for x in levers.split(","))
    undone = compile_kernel(
        _scoped_scale_kernel(off), arch="gfx950", code_object_backend="direct"
    )
    monkeypatch.delenv("ROCKE_DIRECT_LEVERS")
    plain = compile_kernel(_scale_kernel(), arch="gfx950", code_object_backend="direct")
    assert undone.hsaco == plain.hsaco
