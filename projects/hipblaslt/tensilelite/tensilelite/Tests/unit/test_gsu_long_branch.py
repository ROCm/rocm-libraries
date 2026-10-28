# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# The global write's "branch if GSU == 1" is a short s_cbranch_scc1 unless the
# assembler rejects it as out of simm16 range; that kernel is then emitted
# again with the long branch. Every other kernel is emitted once, unchanged.
################################################################################

import pytest

from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

_KERNEL = {"KernelLanguage": "Assembly", "WavefrontSize": 32}
_OUT_OF_RANGE = ("kernel.s:4:1: error: branch size exceeds simm16\n"
                 "s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1\n^")


def _source(span, long=False):
    branch = ("s_cbranch_scc0 label_NoBranch_1                    // long branch if GSU == 1" if long else
              "s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1")
    return "\n".join(['.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"', "s_cmp_eq_u32 s8, 1  // GSU == 1 ?", branch]
                     + ["v_mov_b32 v0, v1"] * span + ["label_GSU_1:", "s_endpgm", ""])


class _Assembler:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def __call__(self, targetGfx, wavefrontSize, srcPath, destPath):
        with open(srcPath) as f:
            self.calls.append((targetGfx, wavefrontSize, f.read()))
        if self.error:
            raise RuntimeError(self.error)


class _Writer(KernelWriterAssembly):
    """A writer whose kernel source has `span` lines between the GSU branch and its label."""

    def __init__(self, span, assembler):
        self.assembler, self.span, self.forceLongGsuBranch, self.emits = assembler, span, False, []

    def _getKernelSource(self, kernel):
        self.emits.append(self.forceLongGsuBranch)
        return _source(self.span, long=self.forceLongGsuBranch)


@pytest.mark.parametrize("span", [16, 6553])
def test_branch_that_always_fits_is_emitted_once_without_assembling(span):
    assembler = _Assembler(_OUT_OF_RANGE)
    writer = _Writer(span, assembler)
    assert writer.getSourceFileString(dict(_KERNEL)) == (0, _source(span))
    assert writer.emits == [False] and assembler.calls == []


def test_branch_out_of_range_is_emitted_again_with_the_long_branch():
    assembler = _Assembler(_OUT_OF_RANGE)
    writer = _Writer(6554, assembler)
    assert writer.getSourceFileString(dict(_KERNEL)) == (0, _source(6554, long=True))
    assert writer.emits == [False, True] and writer.forceLongGsuBranch is False
    assert assembler.calls == [("gfx1250", 32, _source(6554))]


@pytest.mark.parametrize("error", [
    None,
    "error: invalid integral value 'default' in '-mcode-object-version=default'",
    "kernel.s:9:1: error: branch size exceeds simm16\ns_cbranch_scc0 label_Other                // other branch\n^",
], ids=["assembles", "other-error", "other-branch"])
def test_long_span_keeps_the_short_branch_unless_the_gsu_branch_is_rejected(error):
    assembler = _Assembler(error)
    writer = _Writer(20000, assembler)
    assert writer.getSourceFileString(dict(_KERNEL)) == (0, _source(20000))
    assert writer.emits == [False] and len(assembler.calls) == 1


def test_failed_second_emit_resets_the_flag():
    class _FailsAgain(_Writer):
        def _getKernelSource(self, kernel):
            if self.emits:
                raise RuntimeError("second emit")
            return super()._getKernelSource(kernel)

    writer = _FailsAgain(6554, _Assembler(_OUT_OF_RANGE))
    assert writer.getSourceFileString(dict(_KERNEL)) == (-2, "")
    assert writer.forceLongGsuBranch is False
