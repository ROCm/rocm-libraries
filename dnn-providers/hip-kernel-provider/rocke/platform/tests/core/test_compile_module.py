# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""`KernelModule` / `compile_module`: several kernels in ONE code object.

The host-only cases mock lowering and COMGR to pin the contract (validation,
per-kernel lowering, link order, shared codegen options). The COMGR case
compiles two real kernels -- different signatures, each with its own LDS pool --
and reads the object's AMDGPU metadata back: one note naming both kernels with
the argument lists their KernelDefs declare.
"""

from __future__ import annotations

import importlib
import struct
from unittest import mock

import pytest

from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Region
from rocke.helpers import KernelModule, ModuleArtifact, compile_module
from rocke.runtime.comgr import ComgrTimings

compile_impl = importlib.import_module("rocke.helpers.compile")
comgr_impl = importlib.import_module("rocke.runtime.comgr")


def _empty(name: str) -> KernelDef:
    return KernelDef(name, [], Region("entry"))


def _reverse_through_lds(b: IRBuilder, n, value_at, store_at) -> None:
    """Each 64-lane block writes ``value_at(i)`` to LDS and, after a barrier,
    stores its mirror lane's value: real cross-lane LDS traffic, so the pool
    survives optimisation and every kernel keeps one."""
    smem = b.smem_alloc(F32, [64], name_hint="stage")
    tid = b.thread_id_x()
    base = b.mul(b.block_id_x(), b.const_i32(64))
    idx = b.add(base, tid)
    safe = b.select(b.cmp_lt(idx, n), idx, b.const_i32(0))
    b.smem_store_vN(smem, [tid], value_at(safe), 1)
    b.sync()
    mirror = b.sub(b.const_i32(63), tid)
    out_idx = b.add(base, mirror)
    with b.scf_if(b.cmp_lt(out_idx, n)):
        staged = b.vec_extract(b.smem_load_vN(smem, mirror, dtype=F32, n=1), 0)
        store_at(out_idx, staged)


def _toy_scale(name: str) -> KernelDef:
    """``Y[i] = X[i] * alpha`` for ``i < N``, exchanged through LDS."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F32, "global"), noalias=True, readonly=True, align=4)
    y = b.param("Y", PtrType(F32, "global"), noalias=True, writeonly=True, align=4)
    n = b.param("N", I32)
    alpha = b.param("alpha", F32)
    _reverse_through_lds(
        b,
        n,
        lambda i: b.fmul(b.global_load(x, i, F32, align=4), alpha),
        lambda i, v: b.global_store(y, i, v, align=4),
    )
    return b.kernel


def _toy_merge(name: str) -> KernelDef:
    """``Out[i] = A[i] + B[i]`` for ``i < N``, exchanged through LDS."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    a = b.param("A", PtrType(F32, "global"), noalias=True, readonly=True, align=4)
    c = b.param("B", PtrType(F32, "global"), noalias=True, readonly=True, align=4)
    out = b.param("Out", PtrType(F32, "global"), noalias=True, writeonly=True, align=4)
    n = b.param("N", I32)
    _reverse_through_lds(
        b,
        n,
        lambda i: b.fadd(
            b.global_load(a, i, F32, align=4), b.global_load(c, i, F32, align=4)
        ),
        lambda i, v: b.global_store(out, i, v, align=4),
    )
    return b.kernel


# --- KernelModule validation ---------------------------------------------------


def test_module_orders_primary_then_companions() -> None:
    p, m, r = _empty("p"), _empty("m"), _empty("r")
    module = KernelModule(primary=p, companions={"merge": m, "reduce": r})
    assert module.kernels == (p, m, r)
    assert list(module.companions) == ["merge", "reduce"]


def test_module_copies_its_companion_mapping() -> None:
    companions = {"merge": _empty("m")}
    module = KernelModule(primary=_empty("p"), companions=companions)
    companions["late"] = _empty("late")
    assert list(module.companions) == ["merge"]


def test_module_needs_a_companion() -> None:
    with pytest.raises(ValueError, match="at least one companion"):
        KernelModule(primary=_empty("p"), companions={})


@pytest.mark.parametrize("role", ["Merge", "1merge", "merge-kernel", "", "merge.x"])
def test_module_rejects_a_role_that_cannot_name_metadata(role: str) -> None:
    with pytest.raises(ValueError, match="lowercase identifier"):
        KernelModule(primary=_empty("p"), companions={role: _empty("m")})


def test_module_rejects_kernels_sharing_a_symbol() -> None:
    with pytest.raises(ValueError, match="share the name"):
        KernelModule(primary=_empty("same"), companions={"merge": _empty("same")})


def test_module_rejects_a_non_kernel() -> None:
    with pytest.raises(TypeError):
        KernelModule(primary=_empty("p"), companions={"merge": "m"})
    with pytest.raises(TypeError):
        KernelModule(primary="p", companions={"merge": _empty("m")})


# --- compile_module, lowering and COMGR mocked ----------------------------------


def test_compile_module_lowers_each_kernel_and_links_in_order() -> None:
    module = KernelModule(primary=_empty("p"), companions={"merge": _empty("m")})
    with (
        mock.patch.object(
            compile_impl,
            "_lower_llvm_via_backend",
            side_effect=lambda k, **kw: f"llvm:{k.name}:{kw['arch']}:{kw['backend']}",
        ) as lower,
        mock.patch.object(
            compile_impl,
            "build_hsaco_from_llvm_modules",
            return_value=(b"hsaco", ComgrTimings(link=0.5)),
        ) as build,
        mock.patch.object(compile_impl, "build_hsaco_from_llvm_ir") as single,
    ):
        artifact = compile_module(module, arch="gfx1101", backend="python")

    assert isinstance(artifact, ModuleArtifact)
    assert [c.args[0].name for c in lower.call_args_list] == ["p", "m"]
    texts = ("llvm:p:gfx11-generic:python", "llvm:m:gfx11-generic:python")
    assert build.call_args.args[0] == texts
    assert build.call_args.kwargs["isa"] == "amdgcn-amd-amdhsa--gfx1101"
    assert build.call_args.kwargs["options"] == ["-O3"]
    single.assert_not_called()
    assert artifact.llvm_texts == texts
    assert artifact.hsaco == b"hsaco"
    assert artifact.kernel_name == "p"
    assert artifact.companion_names == {"merge": "m"}
    assert artifact.timings["comgr_link"] == 500.0


def test_compile_module_refuses_kernels_needing_different_codegen_options() -> None:
    vgpr_form = _empty("m")
    vgpr_form.attrs["mfma_vgpr_form"] = True
    module = KernelModule(primary=_empty("p"), companions={"merge": vgpr_form})
    with (
        mock.patch.object(compile_impl, "_lower_llvm_via_backend") as lower,
        pytest.raises(ValueError, match="one set of COMGR codegen options"),
    ):
        compile_module(module, arch="gfx950")
    lower.assert_not_called()


def test_compile_module_refuses_a_fresh_process_kernel() -> None:
    fresh = _empty("m")
    fresh.attrs[compile_impl.COMGR_FRESH_PROCESS_ATTR] = True
    module = KernelModule(primary=_empty("p"), companions={"merge": fresh})
    with (
        mock.patch.object(compile_impl, "_lower_llvm_via_backend") as lower,
        pytest.raises(ValueError, match="fresh-process compile"),
    ):
        compile_module(module, arch="gfx950")
    lower.assert_not_called()


def test_link_entry_point_needs_two_modules() -> None:
    with pytest.raises(ValueError, match="two or more"):
        comgr_impl.build_hsaco_from_llvm_modules(["only"])


# --- real COMGR: one object, both kernels --------------------------------------


_NT_AMDGPU_METADATA = 32


def _metadata_notes(elf: bytes) -> list:
    """Every NT_AMDGPU_METADATA payload in an ELF64 LE object, msgpack-decoded."""
    msgpack = pytest.importorskip("msgpack")
    (shoff,) = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    notes = []
    for i in range(shnum):
        (sh_type,) = struct.unpack_from("<I", elf, shoff + i * shentsize + 4)
        if sh_type != 7:  # SHT_NOTE
            continue
        offset, size = struct.unpack_from("<QQ", elf, shoff + i * shentsize + 0x18)
        pos = offset
        while pos < offset + size:
            namesz, descsz, ntype = struct.unpack_from("<III", elf, pos)
            pos += 12
            name = elf[pos : pos + namesz].rstrip(b"\0")
            pos += (namesz + 3) & ~3
            if ntype == _NT_AMDGPU_METADATA and name == b"AMDGPU":
                notes.append(msgpack.unpackb(elf[pos : pos + descsz], raw=False))
            pos += (descsz + 3) & ~3
    return notes


def _signature(kernel: dict) -> list:
    return [
        (a[".value_kind"], a[".size"], a[".offset"], a.get(".name"))
        for a in kernel[".args"]
        if not a[".value_kind"].startswith("hidden_")
    ]


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1101"])
def test_compile_module_emits_one_object_exporting_every_kernel(arch: str) -> None:
    try:
        comgr_impl._resolve_lib()
    except comgr_impl.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")

    module = KernelModule(
        primary=_toy_scale("toy_module_scale"),
        companions={"merge": _toy_merge("toy_module_merge")},
    )
    artifact = compile_module(
        module, arch=arch, capture_ir_text=False, backend="python"
    )

    notes = _metadata_notes(artifact.hsaco)
    assert len(notes) == 1, "the HIP runtime reads exactly one metadata note"
    kernels = {k[".name"]: k for k in notes[0]["amdhsa.kernels"]}
    assert set(kernels) == {"toy_module_scale", "toy_module_merge"}
    assert _signature(kernels["toy_module_scale"]) == [
        ("global_buffer", 8, 0, "X"),
        ("global_buffer", 8, 8, "Y"),
        ("by_value", 4, 16, "N"),
        ("by_value", 4, 20, "alpha"),
    ]
    assert _signature(kernels["toy_module_merge"]) == [
        ("global_buffer", 8, 0, "A"),
        ("global_buffer", 8, 8, "B"),
        ("global_buffer", 8, 16, "Out"),
        ("by_value", 4, 24, "N"),
    ]
    # Each kernel keeps its own LDS pool through the link.
    for name in kernels:
        assert kernels[name][".group_segment_fixed_size"] == 256

    # Every kernel lowers to exactly the text it lowers to alone.
    for kernel, text in zip(module.kernels, artifact.llvm_texts):
        assert (
            text
            == compile_impl.lower_kernel_for_comgr(
                kernel, arch=arch, backend="python"
            ).llvm_text
        )
