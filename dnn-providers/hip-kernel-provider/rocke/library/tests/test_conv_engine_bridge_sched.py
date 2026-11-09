# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``sched`` validation in the public ``lower_*`` entry points of
``kernels.conv.engine_bridge``.

Every conv family whose spec carries ``sched`` (a
:class:`rocke.core.codegen_policy.SchedulerStrategy` value, or ``None``)
refuses an unknown one before either engine runs, through the shared
``scheduler_strategy_reason`` helper: ``backend="cpp"`` raises the same
``ValueError`` as ``backend="python"`` instead of the binding's build error,
and a valid ``sched`` still lowers to the same ``.ll`` on both engines. The
binding's own refusal (``*_spec_to_dict`` straight into the C++ validator) is
covered by each family's parity test and is not bypassed here.

CPU only (no kernel launch). Skips when the ``rocke_library_engine``
extension is not importable.

Run:
  PYTHONPATH=<engine build>/cpp/bindings:rocke/platform/python:rocke/library \\
      <python> -m pytest rocke/library/tests/test_conv_engine_bridge_sched.py -v
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import unittest
from unittest import mock

from kernels.common.conv_dgrad_patch_wmma import DgradPatchWmmaSpec
from kernels.common.conv_dgrad_smallc_wmma import DgradSmallcWmmaSpec
from kernels.common.conv_direct_dgrad_wmma import DirectDgradWmmaSpec
from kernels.common.conv_direct_fwd_gstream_wmma import FwdGstreamWmmaSpec
from kernels.common.conv_direct_fwd_tapouter_wmma import DirectFwdTapOuterSpec
from kernels.common.conv_direct_fwd_wmma import DirectFwdWmmaSpec
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_fwd_small_wmma import FwdSmallcWmmaSpec, FwdSmallkWmmaSpec
from kernels.common.conv_winograd_gemm_wmma import WinogradGemmWmmaSpec
from kernels.conv import engine_bridge as eb

_HAS_ENGINE = importlib.util.find_spec("rocke_library_engine") is not None
_SKIP = "" if _HAS_ENGINE else "rocke_library_engine extension not importable"
_ARCH = "gfx1250"


def _p(C, K, KH=3, S=1, PAD=1, groups=1, H=32, W=48, **kw):
    return DirectConvProblem(
        N=2,
        H=H,
        W=W,
        groups=groups,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=S,
        dtype="bf16",
        **kw,
    )


def _cases():
    """(lower_fn, valid spec) for every family that forwards ``sched``."""
    return (
        (
            eb.lower_conv_direct_fwd_smallc_wmma,
            FwdSmallcWmmaSpec(problem=_p(3, 64)),
        ),
        (
            eb.lower_conv_direct_fwd_smallk_wmma,
            FwdSmallkWmmaSpec(problem=_p(64, 4)),
        ),
        (
            eb.lower_conv_direct_fwd_gstream_wmma,
            FwdGstreamWmmaSpec(problem=_p(16, 16, groups=8, H=28, W=28)),
        ),
        (
            eb.lower_conv_direct_fwd_tapouter_wmma,
            DirectFwdTapOuterSpec(
                problem=_p(64, 96, PAD=12, H=29, W=37, dil_h=12, dil_w=12)
            ),
        ),
        (
            eb.lower_conv_dgrad_patch_wmma,
            DgradPatchWmmaSpec(
                problem=_p(3, 64, KH=4, S=4, PAD=0),
                waves_c=3,
                waves_p=1,
                tiles_c=1,
                r_blk=4,
            ),
        ),
        (
            eb.lower_conv_dgrad_smallc_wmma,
            DgradSmallcWmmaSpec(
                problem=_p(3, 64),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=6,
                out_h=16,
                out_w=16,
            ),
        ),
        (
            eb.lower_conv_direct_dgrad_wmma,
            DirectDgradWmmaSpec(problem=_p(64, 64, H=28, W=28)),
        ),
        (
            eb.lower_conv_direct_fwd_wmma,
            DirectFwdWmmaSpec(problem=_p(64, 64, H=28, W=28)),
        ),
        (
            eb.lower_conv_winograd_gemm_wmma,
            WinogradGemmWmmaSpec(
                waves_m=1, waves_n=2, frags_m=2, frags_n=1, tile_k=32, stages=2
            ),
        ),
    )


# The gfx1250 builders lower tensor_load_to_lds only at llvm23, so pin the
# flavor rather than inherit the host's.
@unittest.skipIf(bool(_SKIP), _SKIP)
@mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm23"})
class TestEngineBridgeSched(unittest.TestCase):
    def _raised(self, fn, spec, backend):
        with self.assertRaises(ValueError) as cm:
            fn(spec, arch=_ARCH, backend=backend)
        return str(cm.exception)

    def test_unknown_sched_same_value_error_on_both_backends(self):
        for fn, spec in _cases():
            bad = dataclasses.replace(spec, sched="bogus")
            with self.subTest(family=fn.__name__):
                why_py = self._raised(fn, bad, "python")
                why_cpp = self._raised(fn, bad, "cpp")
                self.assertEqual(why_cpp, why_py)
                self.assertTrue(
                    why_py.startswith(
                        f"invalid {type(spec).__name__} for {_ARCH}: "
                        "unsupported scheduler_strategy 'bogus'; expected one of: "
                    ),
                    why_py,
                )

    def test_valid_sched_lowers_identically(self):
        """A supported ``sched`` (and ``None``) passes the check unchanged:
        ``backend="both"`` diffs the two engines' ``.ll``."""
        for fn, spec in _cases():
            for sched in (None, "max-ilp"):
                with self.subTest(family=fn.__name__, sched=sched):
                    res = fn(
                        dataclasses.replace(spec, sched=sched),
                        arch=_ARCH,
                        backend="both",
                    )
                    self.assertTrue(res.llvm_text)


if __name__ == "__main__":
    unittest.main()
