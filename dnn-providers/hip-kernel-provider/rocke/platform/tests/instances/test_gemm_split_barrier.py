# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for ``TraitSpec.split_barrier`` on the gfx1250 TDM ping-pong.

A split barrier fails in two ways that compile cleanly:

* **A hang.** Every ``s_barrier_wait`` must pair with a signal raised by every
  wave. A wait no signal satisfies parks the whole workgroup; on silicon that
  is a hung GPU, not an error. Hence the pairing count and the narrow validity
  gate: the signal is raised from the WMMA phase of the depth-2 loop only.
* **A silent race.** The signal releases the refill of the buffer the wave is
  reading. An LDS read issued after the signal reads a buffer that may already
  be overwritten. Hence: no LDS load between the in-loop signal and the end of
  the iteration, and the wait must come before the iteration's first LDS read
  and before its refill.

These are IR-shape assertions on the lowered kernel; they need no GPU.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

ARCH = "gfx1250"

_SIGNAL = "call void @llvm.amdgcn.s.barrier.signal("
_WAIT = "call void @llvm.amdgcn.s.barrier.wait("
_JOINT = "call void @llvm.amdgcn.s.barrier()"
_LDS_LOAD = re.compile(r"= load .*addrspace\(3\)")
_WMMA = re.compile(r"call .*@llvm\.amdgcn\.wmma")


def _spec(**trait):
    """The gfx1250 fp8 wide-atom TDM depth-2 shape the trait was measured on."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    base = dict(
        pipeline="mem",
        scheduler="intrawave",
        epilogue="cshuffle",
        tdm=True,
        tdm_depth=2,
        lds_k_pad=48,
    )
    base.update(trait)
    return UniversalGemmSpec(
        name="split_barrier_probe",
        tile=TileSpec(
            tile_m=256,
            tile_n=256,
            tile_k=256,
            warp_m=4,
            warp_n=4,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=128,
        ),
        trait=TraitSpec(**base),
        data=DataSpec(
            dtype_a="fp8e4m3",
            dtype_b="fp8e4m3",
            dtype_c="bf16",
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=32,
    )


def _lower(spec) -> str:
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.instances.common.gemm_universal import (
        build_universal_gemm,
        is_valid_spec,
    )

    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        raise AssertionError(f"spec rejected: {why}")
    return lower_kernel_to_llvm(build_universal_gemm(spec, arch=ARCH), arch=ARCH)


def _loop_body(ll: str) -> list:
    """Lines of the K-loop body: from the ``for.body`` label to its latch."""
    lines = ll.splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r"for\.body\.\d+:", l))
    end = next(
        i
        for i, l in enumerate(lines)
        if i > start and re.match(r"for\.latch\.\d+:", l)
    )
    return lines[start:end]


class TestGemmSplitBarrierValidity(unittest.TestCase):
    def _valid(self, spec, arch=ARCH):
        from rocke.instances.common.gemm_universal import is_valid_spec

        return is_valid_spec(spec, arch=arch)

    def test_admitted_on_the_depth2_tdm_wmma_loop(self):
        ok, why = self._valid(_spec(split_barrier=True))
        self.assertTrue(ok, why)

    def test_rejected_where_the_signal_is_never_raised(self):
        """Any other loop shape would wait on a barrier nothing signals."""
        for label, kw in (
            ("depth 1", dict(tdm_depth=1)),
            ("depth 3 ring", dict(tdm_depth=3)),
            ("no tdm", dict(tdm=False, tdm_depth=1)),
            ("wmma_v1", dict(pipeline="wmma_v1")),
            ("split_k", dict(split_k=2)),
        ):
            with self.subTest(label):
                ok, why = self._valid(_spec(split_barrier=True, **kw))
                self.assertFalse(ok, f"{label} admitted")
                self.assertIn("split_barrier", why)


class TestGemmSplitBarrierEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = _lower(_spec())
        cls.on = _lower(_spec(split_barrier=True))

    def test_default_emits_no_split_barrier(self):
        self.assertNotIn(_SIGNAL, self.off)
        self.assertNotIn(_WAIT, self.off)
        self.assertEqual(self.off, _lower(_spec(split_barrier=False)))

    def test_every_wait_pairs_with_a_signal(self):
        """Prologue + in-loop signals pair with in-loop + epilogue waits."""
        self.assertEqual(self.on.count(_SIGNAL), 2)
        self.assertEqual(self.on.count(_WAIT), 2)

    def test_loop_has_no_joint_barrier(self):
        body = _loop_body(self.on)
        self.assertFalse(any(_JOINT in l for l in body))
        self.assertTrue(any(_JOINT in l for l in _loop_body(self.off)))

    def test_wait_precedes_the_first_read_and_the_refill(self):
        body = _loop_body(self.on)
        wait = next(i for i, l in enumerate(body) if _WAIT in l)
        first_read = next(i for i, l in enumerate(body) if _LDS_LOAD.search(l))
        refill = next(i for i, l in enumerate(body) if "tensor.load.to.lds" in l)
        self.assertLess(wait, first_read)
        self.assertLess(wait, refill)

    def test_no_lds_read_after_the_signal_and_wmmas_overlap_it(self):
        """The signal frees the buffer: nothing may read it afterwards."""
        body = _loop_body(self.on)
        sig = next(i for i, l in enumerate(body) if _SIGNAL in l)
        tail = body[sig + 1 :]
        self.assertFalse(
            any(_LDS_LOAD.search(l) for l in tail),
            "LDS read after the split-barrier signal races the refill",
        )
        self.assertTrue(
            any(_WMMA.search(l) for l in tail),
            "no WMMA between signal and wait: the split hides nothing",
        )
        self.assertIn(
            "s.wait.tensorcnt", body[sig - 1], "signal before the fill landed"
        )


if __name__ == "__main__":
    unittest.main()
