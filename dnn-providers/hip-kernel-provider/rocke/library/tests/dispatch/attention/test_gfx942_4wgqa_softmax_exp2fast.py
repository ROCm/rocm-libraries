# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 4-warp GQA softmax: lowers to ``exp2_fast``, and its register footprint.

The 4-warp GQA kernel (``build_gfx942_4warp_gqa``) runs the online softmax as
``alpha = exp2(m_old - m_new)`` and ``P = exp2(S*scale - m_new)``. Both arguments
are ``<= 0`` by the running-max invariant, so the ``exp2`` overflow/underflow
range-reduction guard is unnecessary and the kernel emits ``math.exp2_fast``
(raw ``v_exp_f32``) rather than the guarded ``math.exp2``.

Two layers:

* **Emit test (host-only, always runs):** the softmax lowers to ``exp2_fast`` for
  bf16 and fp16. Fails against the pre-change kernel (guarded ``math.exp2``).
* **Scratch guard (needs comgr + llvm-readelf/objdump; skipped otherwise):**
  ``exp2_fast`` drops the range-reduction guard, which changes the instruction
  schedule and raises register pressure toward the gfx942 256-VGPR cap. Whether
  that turns into a spill is comgr-version-dependent
  (older comgr spills; the 7.13/7.14 comgr allocates the same IR spill-free), and
  the exact scratch size is not stable across toolchains -- so this test does NOT
  pin a specific byte count. It compiles the kernel and asserts
  ``.private_segment_fixed_size`` stays under a generous ceiling, catching a gross
  future blow-up on whatever toolchain CI runs, so the register footprint the swap
  changed is guarded, not just the op name. The spill (where it occurs) is
  perf-neutral -- the path is memory-bound -- so it is left as-is, not pinned down.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import tempfile
import unittest

import kernels.common.attention_unified as au
from kernels.common.attention_unified import _tiled_spec_from_problem
from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa
from rocke.core.ir_print import print_ir
from rocke.runtime.comgr import resolved_lib_path

# Generous absolute ceiling: measured scratch on this cohort spans 0 B (7.13/7.14
# comgr, spill-free) to ~100 B (ROCm 7.2 sweep comgr). This guards against a gross
# regression on whatever toolchain CI runs -- it is not a match to any one build's
# exact count (those are comgr-version-specific). Lower it if CI pins one toolchain.
SCRATCH_CEIL_BYTES = {"bf16": 256, "fp16": 256}


class _PinArch:
    """Pin ``_RESOLVED_ATTENTION_ARCH`` so spec resolution runs off-GPU."""

    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self.arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _problem(dtype: str) -> "au.UnifiedAttentionProblem":
    """The D128 sliding-window 4-warp GQA cohort (block_size=16) the swap ships on."""
    return au.UnifiedAttentionProblem(
        total_q=8192,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=8192,
        max_seqlen_k=8192,
        dtype=dtype,
        sliding_window=4096,
    )


def _emit_4wgqa_ir(dtype: str) -> str:
    """Build the D128 SW 4-warp GQA kernel for ``dtype`` and return its IR text."""
    spec = _tiled_spec_from_problem(_problem(dtype))
    return print_ir(build_gfx942_4warp_gqa(spec, arch="gfx942"))


def _llvm_tool(name: str) -> str | None:
    """Locate an LLVM tool: PATH first, then a ROCm-bundled LLVM.

    A stock ROCm install keeps ``llvm-readelf``/``llvm-objdump`` in
    ``<rocm>/llvm/bin``, which is not on PATH. Same tiers as ``_llvm_as_path``
    in ``platform/tests/test_rocke.py``: ``$ROCM_PATH``/``$ROCM_HOME``, then the
    globbed ``/opt/rocm*`` installs, newest first.
    """
    found = shutil.which(name)
    if found:
        return found
    roots = [os.environ.get(v) for v in ("ROCM_PATH", "ROCM_HOME")]
    roots += sorted(glob.glob("/opt/rocm*"), reverse=True)
    for root in roots:
        if not root:
            continue
        cand = os.path.join(root, "llvm", "bin", name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _comgr_ready() -> bool:
    """The scratch guard needs the comgr library + llvm-readelf/objdump to read notes.

    comgr is dlopened lazily on the first compile, so importing ``rocke`` proves
    nothing; look the library up on disk instead (a pure lookup, no dlopen).
    """
    if _llvm_tool("llvm-readelf") is None or _llvm_tool("llvm-objdump") is None:
        return False
    path = resolved_lib_path()
    return path is not None and os.path.exists(path)


def _scratch_bytes(dtype: str) -> int:
    """Compile the kernel to a gfx942 code object and read
    ``.private_segment_fixed_size`` (scratch bytes) from its HSACO notes."""
    from rocke import compile_kernel
    from rocke.analysis.isa import analyze_hsaco

    spec = _tiled_spec_from_problem(_problem(dtype))
    kernel = build_gfx942_4warp_gqa(spec, arch="gfx942")
    artifact = compile_kernel(kernel, arch="gfx942", capture_ir_text=False)
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
            f.write(artifact.hsaco)
            path = f.name
        res = analyze_hsaco(
            path,
            objdump=_llvm_tool("llvm-objdump"),
            readelf=_llvm_tool("llvm-readelf"),
        ).resources
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
    return int(res.scratch_bytes or 0)


class TestGfx942_4wgqaSoftmaxExp2Fast(unittest.TestCase):
    def test_softmax_uses_exp2_fast_not_guarded_exp2(self):
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                ir = _emit_4wgqa_ir(dt)
                n_fast = len(re.findall(r"math\.exp2_fast\b", ir))
                n_slow = len(re.findall(r"math\.exp2(?!_fast)", ir))
                # alpha + P are the two online-softmax exp2 sites.
                self.assertGreaterEqual(
                    n_fast,
                    2,
                    msg=f"{dt}: expected >=2 math.exp2_fast (alpha + P), got {n_fast}",
                )
                # The exp2 -> exp2_fast swap must be complete (no guarded exp2 left).
                self.assertEqual(
                    n_slow,
                    0,
                    msg=f"{dt}: guarded math.exp2 must be gone from the softmax, "
                    f"got {n_slow}",
                )

    @unittest.skipUnless(
        _comgr_ready(), "needs comgr + llvm-readelf/llvm-objdump (compile lane)"
    )
    def test_softmax_scratch_within_ceiling(self):
        # exp2_fast trades a few VALU per exp for register pressure: the kernel may
        # spill at the 256-VGPR cap (comgr-version-dependent). That is perf-neutral
        # (memory-bound) and intentional -- this guards against a *future* blow-up,
        # so the footprint the swap changed is covered by CI, not only the op name.
        with _PinArch("gfx942"):
            for dt in ("bf16", "fp16"):
                scratch = _scratch_bytes(dt)
                self.assertLessEqual(
                    scratch,
                    SCRATCH_CEIL_BYTES[dt],
                    msg=f"{dt}: scratch {scratch} B > {SCRATCH_CEIL_BYTES[dt]} B "
                    f"ceiling -- exp2_fast register-pressure regression?",
                )


if __name__ == "__main__":
    unittest.main()
