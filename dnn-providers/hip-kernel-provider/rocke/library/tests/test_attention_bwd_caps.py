# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the backward per-arch capability checks.

Covers the capability record per arch, atom selection, and the DMA-width
legality check (illegal requests raise before any op is emitted). The
native-atomic fact is checked at the IR level for every arch and flavor, and
at the ISA level when ``libamd_comgr`` and ``llvm-objdump`` are available.
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
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common import _attention_bwd_caps as C

ARCHS = C.BWD_ARCHES
FLAVORS = ("llvm20", "llvm22", "llvm23")


# --- capability record --------------------------------------------------------


@pytest.mark.parametrize(
    "arch, wave, path, lds, dma, tr64, agprs",
    [
        ("gfx942", 64, "mfma", 65536, (1,), False, 256),
        ("gfx950", 64, "mfma", 163840, (1, 3, 4), True, 256),
        ("gfx1151", 32, "wmma", 65536, (), False, 0),
        ("gfx1201", 32, "wmma", 65536, (), False, 0),
    ],
)
def test_capability_record(arch, wave, path, lds, dma, tr64, agprs):
    caps = C.bwd_arch_caps(arch)
    assert caps.wave_size == wave
    assert caps.matrix_path == path
    assert caps.lds_capacity_bytes == lds
    assert caps.lds_dma_dwords == dma
    assert caps.max_lds_dma_dwords == max(dma, default=0)
    assert caps.has_tr_read_b64 is tr64
    assert caps.has_tr_read_b128 is False
    assert caps.agprs == agprs
    assert caps.has_agprs is (agprs > 0)
    assert caps.arch_vgprs == 256
    assert caps.native_fp32_atomic_add is True
    assert caps.legal_waves == ((1, 2, 4, 8) if path == "mfma" else (1, 2, 4))


def test_unknown_arch_is_refused():
    with pytest.raises(ValueError, match="does not target"):
        C.bwd_arch_caps("gfx90a")


# --- atoms ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("arch", ARCHS)
def test_atom_catalog(arch, dtype):
    ops = C.bwd_atoms(arch, dtype)
    ks = [op.k for op in ops]
    assert ks == ([16, 32] if arch == "gfx950" else [16])
    for op in ops:
        assert (op.m, op.n) == (16, 16), "no 32x32 atom in the backward"
        assert op.wave_size == C.bwd_arch_caps(arch).wave_size
        assert op.a_dtype == op.b_dtype == dtype
    assert C.wide_k_atom_available(arch, dtype) is (arch == "gfx950")
    assert C.select_bwd_atom(arch, dtype).k == 16
    if arch == "gfx950":
        assert C.select_bwd_atom(arch, dtype, 32).k == 32
    else:
        with pytest.raises(ValueError, match="no native 16x16x32"):
            C.select_bwd_atom(arch, dtype, 32)


def test_atom_dtype_is_checked():
    with pytest.raises(ValueError, match="fp16 or bf16"):
        C.bwd_atoms("gfx942", "fp8")


@pytest.mark.parametrize("arch", ARCHS)
def test_waves_and_lds_checks(arch):
    caps = C.bwd_arch_caps(arch)
    for w in caps.legal_waves:
        C.check_waves(arch, w)
    for w in (0, 3, 16) + ((8,) if caps.matrix_path == "wmma" else ()):
        with pytest.raises(ValueError, match="not legal"):
            C.check_waves(arch, w)
    C.check_lds_bytes(arch, caps.lds_capacity_bytes)
    with pytest.raises(ValueError, match="exceeds"):
        C.check_lds_bytes(arch, caps.lds_capacity_bytes + 1)


# --- DMA width (illegal requests raise before emission) -------------------------


def _dma_kernel(arch, dwords):
    b = IRBuilder(f"bwd_caps_dma_{arch}_{dwords}")
    x = b.param("X", PtrType(F16, "global"), readonly=True)
    s = b.smem_alloc(F16, (64, 64), "s")
    rsrc = b.buffer_rsrc(x, b.const_i32(8192))
    lds = b.smem_addr_of(s)
    voff = b.mul(b.thread_id_x(), b.const_i32(4 * dwords))
    before = len(b.kernel.body.ops)
    return b, rsrc, lds, voff, before


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("dwords", [1, 2, 3, 4])
def test_dma_width_legality(arch, dwords):
    legal = dwords in C.bwd_arch_caps(arch).lds_dma_dwords
    b, rsrc, lds, voff, before = _dma_kernel(arch, dwords)
    zero = b.const_i32(0)
    before = len(b.kernel.body.ops)
    if not legal:
        with pytest.raises(ValueError):
            C.check_lds_dma_width(arch, dwords)
        with pytest.raises(ValueError):
            C.emit_lds_dma(
                b, arch=arch, rsrc=rsrc, lds_ptr=lds, voffset=voff, soffset=zero,
                dwords=dwords,
            )  # fmt: skip
        assert len(b.kernel.body.ops) == before, "an illegal DMA emitted an op"
        return
    C.check_lds_dma_width(arch, dwords)
    C.emit_lds_dma(
        b, arch=arch, rsrc=rsrc, lds_ptr=lds, voffset=voff, soffset=zero,
        dwords=dwords,
    )  # fmt: skip
    ops = b.kernel.body.ops[before:]
    assert [op.name for op in ops] == ["tile.async_buffer_load_lds"]
    b.ret()
    ir = lower(b.kernel, arch=arch, llvm_flavor="llvm22")
    assert f"i32 {4 * dwords}," in ir
    assert "@llvm.amdgcn.raw.ptr.buffer.load.lds(" in ir


def test_dma_messages_name_the_legal_widths():
    with pytest.raises(ValueError, match=r"legal widths: \(1,\)"):
        C.check_lds_dma_width("gfx942", 4)
    with pytest.raises(ValueError, match="no global-to-LDS DMA"):
        C.check_lds_dma_width("gfx1151", 1)


# --- transpose read and AGPR knobs -----------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_transpose_read_width(arch):
    with pytest.raises(ValueError, match="b128"):
        C.check_transpose_read(arch, 128)
    with pytest.raises(ValueError, match="unsupported"):
        C.check_transpose_read(arch, 32)
    if arch == "gfx950":
        C.check_transpose_read(arch, 64)
    else:
        with pytest.raises(ValueError, match="no ds_read_tr16_b64"):
            C.check_transpose_read(arch, 64)


@pytest.mark.parametrize("arch", ARCHS)
def test_agpr_alloc_values(arch):
    C.check_agpr_alloc(arch, None)
    cdna = C.bwd_arch_caps(arch).has_agprs
    for v in ((0, 0), (128, 128), (192, 192), (256, 256)):
        if cdna:
            C.check_agpr_alloc(arch, v)
        else:
            with pytest.raises(ValueError, match="no AGPRs"):
                C.check_agpr_alloc(arch, v)
    if cdna:
        with pytest.raises(ValueError, match="not a legal value"):
            C.check_agpr_alloc(arch, (64, 64))


# --- native fp32 atomic ------------------------------------------------------------


def _atomic_kernel(name):
    b = IRBuilder(name)
    w = b.param("W", PtrType(F32, "global"))
    n = b.param("n", I32)
    tid = b.thread_id_x()
    b.global_atomic_add_f32(w, b.mod(tid, n), b.const_f32(1.0))
    b.ret()
    return b.kernel


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
def test_atomic_lowers_to_agent_scope_monotonic_fadd(arch, flavor):
    ir = lower(_atomic_kernel(f"bwd_caps_atomic_{arch}"), arch=arch, llvm_flavor=flavor)
    lines = [ln for ln in ir.splitlines() if "atomicrmw" in ln]
    assert len(lines) == 1
    assert re.search(r'atomicrmw fadd .*syncscope\("agent"\) monotonic', lines[0])
    assert "cmpxchg" not in ir


def _objdump():
    explicit = os.environ.get("LLVM_OBJDUMP")
    if explicit and Path(explicit).is_file():
        return explicit
    root = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if root:
        for cand in (Path(root) / "llvm" / "bin" / "llvm-objdump",):
            if cand.is_file():
                return str(cand)
    return shutil.which("llvm-objdump")


@pytest.mark.parametrize("arch", ARCHS)
def test_atomic_is_native_in_the_isa(arch):
    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (set LLVM_OBJDUMP)")
    try:
        from rocke.helpers.compile import compile_kernel

        art = compile_kernel(
            _atomic_kernel(f"bwd_caps_atomic_isa_{arch}"),
            arch=arch,
            capture_ir_text=False,
            backend="python",
        )
    except Exception as exc:  # noqa: BLE001 - no comgr on this host
        pytest.skip(f"comgr unavailable: {exc}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.hsaco"
        path.write_bytes(art.hsaco)
        proc = subprocess.run(
            [objdump, "-d", f"--mcpu={arch}", "--triple=amdgcn-amd-amdhsa", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    if proc.returncode != 0 and "not a recognized processor" in proc.stderr:
        pytest.skip(f"{objdump} predates {arch} (set LLVM_OBJDUMP to a newer one)")
    assert proc.returncode == 0, proc.stderr
    isa = proc.stdout
    assert re.search(r"global_atomic_add_f32", isa) is not None
    assert "cmpswap" not in isa
    assert C.bwd_arch_caps(arch).native_fp32_atomic_add


def test_no_exported_helper_looks_like_a_builder():
    assert not [n for n in C.__all__ if n.startswith("build_")]


def test_no_backward_module_defines_a_build_prefixed_non_builder():
    # Only family builders (signature ``(spec, *, arch)``) carry the build_
    # prefix; the backward helper modules and their tests define none.
    lib = Path(__file__).resolve().parents[1]
    files = sorted((lib / "kernels" / "common").glob("_attention_bwd*.py"))
    files += sorted((lib / "tests").glob("test_attention_bwd*.py"))
    files += sorted((lib / "benchmarks" / "common").glob("attention_bwd*.py"))
    assert files
    pat = re.compile(r"^\s*def (build_\w+)\(([^)]*)\)", re.MULTILINE)
    bad = []
    for f in files:
        for name, args in pat.findall(f.read_text(encoding="utf-8")):
            if not re.fullmatch(r"\s*spec\s*(:[^,]*)?,\s*\*,\s*arch\b.*", args, re.S):
                bad.append(f"{f.name}:{name}")
    assert not bad, bad
