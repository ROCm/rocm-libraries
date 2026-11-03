# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared ``TestCase`` base that gives every adaptor test the same starting state.

The adaptor keeps rocisa's process-wide singleton state (selected ISA, caps,
``KernelInfo``, output options, VGPR bookkeeping) in ``base.py``. Instruction
``toString`` reads it, so a test that changes it would otherwise leak into
every test that runs after it in the same process.

``AdaptorTestCase.run`` snapshots that state, resets it to gfx1250 / wave32,
runs the test (``setUp`` included), and restores the snapshot. Overriding
``run`` rather than ``setUp`` means subclasses with their own ``setUp`` get the
reset without having to call ``super().setUp()``. Tests that need another ISA,
another wavefront size, or no ISA at all set it in their own ``setUp``.
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor import base as _base  # noqa: E402
from rocisa_stinkytofu_adaptor.base import OutputOptions  # noqa: E402

DEFAULT_ISA = (12, 5, 0)
DEFAULT_WAVEFRONT_SIZE = 32


def _snapshot():
    return {
        "kernel": _base.getKernel(),
        "current_isa": _base._current_isa,
        "is_init": _base._is_init,
        "assembler_path": _base._assembler_path,
        "data": dict(_base.getData()),
        "vgpr_idx": dict(_base.getVgprIdx()),
        "vgpr_msb": _base.getVgprMsb(),
        "output_no_comment": _base.getOutputOptions().outputNoComment,
    }


def _restore(saved):
    _base.setKernelInfo(saved["kernel"])
    _base._current_isa = saved["current_isa"]
    _base._assembler_path = saved["assembler_path"]
    # getData / getVgprIdx return the live dicts; refill them in place so
    # references captured elsewhere keep seeing the restored contents.
    live_data = _base.getData()
    live_data.clear()
    live_data.update(saved["data"])
    _base._is_init = saved["is_init"]
    live_vgpr = _base.getVgprIdx()
    live_vgpr.clear()
    live_vgpr.update(saved["vgpr_idx"])
    _base.setVgprMsb(saved["vgpr_msb"])
    _base.setOutputOptions(OutputOptions(saved["output_no_comment"]))


def reset_to_default():
    """Select gfx1250 / wave32 with fresh output options and VGPR bookkeeping."""
    _base.setOutputOptions(OutputOptions())
    _base.getVgprIdx().clear()
    _base.setVgprMsb(0)
    _base.init(DEFAULT_ISA, "", False)
    _base.setKernel(DEFAULT_ISA, DEFAULT_WAVEFRONT_SIZE)


class AdaptorTestCase(unittest.TestCase):
    """``unittest.TestCase`` that starts every test from ``reset_to_default()``."""

    def run(self, result=None):
        saved = _snapshot()
        try:
            reset_to_default()
            return super().run(result)
        finally:
            _restore(saved)
