# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The BF16x3 expansion survives codegen for gfx950 (XF32 plan, step 8).

The lowering tests pin the emitted LLVM IR, which proves the recipe is written
correctly but not that it is *codegen-able*: a vector ``fptrunc ... to bfloat``
with no legal packed form, or an MFMA intrinsic the backend does not carry on
this target, would both pass a text check and then fail — or abort — in comgr.
So this compiles the IR the whole way to a loadable code object.

The disassembly is then read back for the two facts the design actually rests
on, neither of which is visible in the IR:

* exactly three ``v_mfma_*_bf16`` instructions survive to the ISA -- the
  backend neither folded nor duplicated a term;
* the fp32->bf16 narrowing selects the **packed** ``v_cvt_pk_bf16_f32``, which
  rounds to nearest-even. That is CK's shipping packed path. Had the backend
  fallen back to a scalar truncating sequence, the emulation would silently
  disagree with CK on the low bits of every input.

No GPU is needed: comgr compiles for a named ISA, it does not run anything.
Environments without comgr, or with a comgr that does not list gfx950 among its
targets, skip. Past that check a compile error *fails* -- it is the regression
this gate exists to catch. Only the disassembly checks need ``llvm-objdump``;
without it they skip and the compile check still runs.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from rocke.analysis.isa import _extract_mnemonic, _instruction_lines
from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, TF32, IRBuilder, KernelDef, PtrType, VectorType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_ARCH = "gfx950"
_ISA = f"amdgcn-amd-amdhsa--{_ARCH}"

# Each emulated atom and the ISA mnemonic its physical atom must become.
_EMULATED = {
    "mfma_f32_16x16x32_xf32": "v_mfma_f32_16x16x32_bf16",
    "mfma_f32_32x32x16_xf32": "v_mfma_f32_32x32x16_bf16",
}


def _gate_kernel(op_id: str) -> KernelDef:
    """One emulated ``tile.mma`` between a real load and a real store.

    The operands are loaded and the result stored so that ``-O3`` cannot fold
    the whole thing away -- a kernel whose MMA result is dead compiles to
    nothing and would gate on nothing.
    """
    atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id)
    b = IRBuilder("xf32_compile_gate")
    b.kernel.attrs["max_workgroup_size"] = atom.wave_size
    a_ptr = b.param("a", PtrType(F32, "global"))
    b_ptr = b.param("b", PtrType(F32, "global"))
    out_ptr = b.param("out", PtrType(F32, "global"))
    idx = b.const_i32(0)
    a = b.global_load_vN(a_ptr, idx, F32, atom.a_frag_len)
    bv = b.global_load_vN(b_ptr, idx, F32, atom.b_frag_len)
    # fp32 storage reinterpreted as the logical TF32 operand, as for a native atom.
    a = b.bitcast(a, VectorType(TF32, atom.a_frag_len))
    bv = b.bitcast(bv, VectorType(TF32, atom.b_frag_len))
    c = b.vector_splat(b.const_f32(0.0), atom.c_frag_len)
    d = b.mma(atom, a, bv, c)
    # Element-wise rather than one wide store: the 32x32 accumulator is 16
    # lanes, past what global_store_vN accepts for f32. Every lane is stored,
    # so no part of the accumulator is dead.
    for lane in range(atom.c_frag_len):
        b.store(b.vec_extract(d, lane), out_ptr, b.const_i32(lane))
    b.ret()
    return b.kernel


def _objdump() -> str | None:
    """``llvm-objdump``, which ROCm installs outside the default PATH."""
    found = shutil.which("llvm-objdump")
    if found:
        return found
    root = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidate = Path(root) / "llvm" / "bin" / "llvm-objdump"
    return str(candidate) if candidate.exists() else None


def _mnemonics(hsaco: bytes) -> list[str]:
    """Every instruction mnemonic in the disassembled code object."""
    tool = _objdump()
    if tool is None:
        raise unittest.SkipTest("llvm-objdump not found; cannot inspect the ISA")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kernel.hsaco"
        path.write_bytes(hsaco)
        done = subprocess.run(
            [tool, "-d", str(path)], capture_output=True, text=True, check=False
        )
    if done.returncode != 0:
        raise AssertionError(f"llvm-objdump failed: {done.stderr or done.stdout}")
    # Reuse the repo's own disassembly reader rather than a second regex for
    # the same format -- it already strips address and encoding prefixes.
    return [_extract_mnemonic(line) for line in _instruction_lines(done.stdout)]


def _compiler():
    """comgr's compile entry point, once comgr is known to target gfx950.

    Only a comgr that cannot be imported or loaded, or that does not list
    gfx950, skips. Anything comgr raises after this is the IR failing to build,
    so callers let it propagate as a failure.
    """
    try:
        from rocke.runtime import comgr
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"comgr unavailable: {exc}") from exc
    try:
        isas = comgr.supported_isas()
    except comgr.ComgrError as exc:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"comgr unavailable: {exc}") from exc
    if _ISA not in isas:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"the loaded comgr cannot target {_ARCH}")
    return comgr.build_hsaco_from_llvm_ir


class TestGfx950Codegen(unittest.TestCase):
    def _compile(self, op_id: str) -> bytes:
        """The code object for the gate kernel. A compile error fails the test."""
        build_hsaco_from_llvm_ir = _compiler()
        # No explicit flavor: the lowerer resolves it from the comgr that will
        # actually compile the IR, which is the only pairing that is safe.
        llvm = _lower_kernel_to_llvm_python(_gate_kernel(op_id), arch=_ARCH)
        hsaco, _ = build_hsaco_from_llvm_ir(llvm, isa=_ISA)
        self.assertTrue(hsaco, "comgr produced an empty code object")
        return hsaco

    def test_the_expansion_compiles_to_a_code_object(self):
        """Needs comgr only -- no disassembler."""
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                self._compile(op_id)

    def test_exactly_three_bf16_mfma_instructions_survive(self):
        """Three terms in, three terms out -- nothing folded or duplicated."""
        for op_id, mnemonic in _EMULATED.items():
            with self.subTest(op_id=op_id):
                seen = _mnemonics(self._compile(op_id))
                self.assertEqual(seen.count(mnemonic), 3)
                stray = [m for m in seen if m.startswith("v_mfma") and m != mnemonic]
                self.assertEqual(stray, [], "an unexpected MFMA reached the ISA")

    def test_the_narrowing_selects_the_packed_rne_convert(self):
        """``v_cvt_pk_bf16_f32`` is RNE; the scalar fallback truncates.

        CK ships the packed path, so selecting it is what makes the emulation
        agree with CK rather than merely resemble it.
        """
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                seen = _mnemonics(self._compile(op_id))
                self.assertTrue(
                    any(m.startswith("v_cvt_pk_bf16_f32") for m in seen),
                    "the fp32->bf16 narrowing did not select the packed convert",
                )


if __name__ == "__main__":
    unittest.main()
