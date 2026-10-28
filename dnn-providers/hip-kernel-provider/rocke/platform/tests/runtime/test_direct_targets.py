# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The direct code-object backend compiles for the targets its library reports,
refuses the others and the operations it does not lower as DirectUnsupported,
and compile_kernel then builds the kernel with COMGR."""

from __future__ import annotations

import struct

import pytest
from rocke.core.ir import F32, IRBuilder, PtrType
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
    # tile.inline_asm with an instruction in it: COMGR assembles it; the
    # direct compiler takes only the empty "=v,0" barrier.
    b = IRBuilder("direct_target_asm")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    y = b.param("Y", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    moved = b.inline_asm("v_mov_b32 $0, $1", "=v,v", [b.global_load_f32(x, tid)], F32)
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
