# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression test for the true16 (real-true16) activation-clamp defect (ROCM-3994).

hipBLASLt / TensileLite's ``AMaxKernelGenerator`` emits the ``v_max_f16``
activation clamp for half inputs. Under a compiler that enforces real16 mode
(``+real-true16``) the *fake16* form of this instruction::

    v_max_f16 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+0])

is rejected by the assembler ("operands are not valid for this GPU or mode").
The correct *true16* form binds a half-word selector (``.l``) to every register
operand, with the suffix placed *inside* the closing paren of ``abs(...)``::

    v_max_f16 v[vgprOutput].l, v[vgprOutput].l, abs(v[vgprValue+0].l)

This is a pure host-side codegen assertion: it renders the generator's module to
a string and checks the emitted text. No GPU and no assembler are required. The
target arch is forced from a gfx string, so no real device has to be present.

RED on ``origin/develop`` (emits fake16), GREEN on the true16 fix branch (emits
true16). The test is parametrized across the whole gfx11 family enabled by the
``NoSDWA`` arch cap (isaVersion 11/12), not a single arch, so it does not repeat
the original mistake of pinning one target.
"""

import os
import re
import sys

import pytest

# Make the tensilelite root importable (mirrors gpu_test_helpers).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if TENSILE_ROOT not in sys.path:
    sys.path.insert(0, TENSILE_ROOT)

from gpu_test_helpers import init_rocisa  # noqa: E402

from Tensile.Common.Architectures import gfxToIsa  # noqa: E402
from Tensile.Common.DataType import DataType  # noqa: E402

import AMaxGenerator  # noqa: E402

# gfx11 family covered by the NoSDWA arch cap (checkInList(isaVersion[0], {11, 12})).
# All of these are wave32 and select the true16 activation path on the fix branch.
GFX11_TARGETS = [
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
]

# ROCM-3994 known-bug quarantine (W-KNOWN-BUG two-PR flow).
#
# This reproducer lands ahead of the true16 fix (users/ericwan/true16-patch).
# On develop the AMax activation clamp is emitted in the fake16 form, so the
# assertions below fail -> reported as xfailed -> CI stays green. When the fix
# lands, the emitted text becomes true16, the test passes, and strict=True turns
# that XPASS into a hard failure -- forcing the fix PR to delete this marker.
# The test always executes (it is quarantined, not disabled).
#
# Time-box: tracked by ROCM-3994; re-evaluate by the next hipBLASLt release if
# still unfixed. REMOVE this marker in the fix PR.
_ROCM3994_XFAIL = pytest.mark.xfail(
    reason="ROCM-3994: fake16 v_max_f16 emitted for the AMax activation clamp; "
           "real-true16 assemblers reject it. Remove this marker in the true16 fix PR.",
    strict=True,
    raises=AssertionError,
)

# Matches a v_max_f16 whose destination and both sources carry a true16 half-word
# selector (.l/.h). Critically, the abs() source suffix is *inside* the paren.
_TRUE16_RE = re.compile(
    r"v_max_f16\s+"
    r"v\[[^\]]*\](?:\.[lh]),\s+"          # dst.l
    r"v\[[^\]]*\](?:\.[lh]),\s+"          # src0.l
    r"(?:abs\(v\[[^\]]*\](?:\.[lh])\)|v\[[^\]]*\](?:\.[lh]))"  # src1(.l) or abs(src1.l)
)

# The bare fake16 form: a v_max_f16 with no half-word selector on any operand.
_FAKE16_RE = re.compile(
    r"v_max_f16\s+"
    r"v\[[^\]]*\],\s+"                    # dst  (no .l/.h)
    r"v\[[^\]]*\],\s+"                    # src0 (no .l/.h)
    r"(?:abs\(v\[[^\]]*\]\)|v\[[^\]]*\])" # src1 (no .l/.h)
)


def _make_generator(target: str) -> "AMaxGenerator.AMaxKernelGenerator":
    """Build a half-input AMaxKernelGenerator initialized for ``target``.

    Uses the shared init_rocisa helper (gfxToIsa -> ri.init -> ri.setKernel),
    which drives the same rocIsa singleton (``_global_ti``) the generator reads
    its arch caps from. gfx11 is wave32.
    """
    isa = gfxToIsa(target)
    assert isa, f"unknown gfx target: {target}"
    init_rocisa(target=target, wavesize=32)
    half = DataType("H")
    return AMaxGenerator.AMaxKernelGenerator(
        i_type=half,
        o_type=half,
        scale_type=DataType("S"),
        num_workitems=256,
        num_load_count=4,
        num_load_size=4,
        wavefront_size=32,
        arch=target,
        isa=isa,
        is_scale=False,
    )


def _vmax_lines(text: str):
    return [ln.strip() for ln in text.splitlines() if "v_max_f16" in ln]


@_ROCM3994_XFAIL
@pytest.mark.unit
@pytest.mark.gfx11
@pytest.mark.parametrize("target", GFX11_TARGETS)
def test_amax_activation_clamp_is_true16(target):
    """max_per_data must emit the true16 v_max_f16 (with abs source)."""
    gen = _make_generator(target)
    text = str(gen.max_per_data(0))

    lines = _vmax_lines(text)
    assert lines, f"[{target}] expected at least one v_max_f16 in max_per_data:\n{text}"

    for ln in lines:
        # Positive: true16 half-word selectors present on the clamp.
        assert _TRUE16_RE.search(ln), (
            f"[{target}] v_max_f16 is not in true16 form (missing .l selector "
            f"and/or abs suffix inside paren): {ln!r}"
        )
        # Negative: reject the bare fake16 form outright.
        assert not _FAKE16_RE.search(ln), (
            f"[{target}] v_max_f16 emitted in fake16 form (no half-word "
            f"selector) — this is the ROCM-3994 defect: {ln!r}"
        )

    # The abs() source suffix must bind *inside* the paren: abs(v[..].l), never abs(v[..]).l
    assert "abs(" in text, f"[{target}] expected an abs() source operand:\n{text}"
    assert not re.search(r"abs\(v\[[^\]]*\]\)\.[lh]", text), (
        f"[{target}] true16 suffix placed outside abs() paren (invalid form):\n{text}"
    )


@_ROCM3994_XFAIL
@pytest.mark.unit
@pytest.mark.gfx11
@pytest.mark.parametrize("target", GFX11_TARGETS)
def test_amax_merge_sum_is_true16(target):
    """merge_sum must emit the true16 v_max_f16 (plain register sources)."""
    gen = _make_generator(target)
    text = str(gen.merge_sum())

    lines = _vmax_lines(text)
    assert lines, f"[{target}] expected at least one v_max_f16 in merge_sum:\n{text}"

    for ln in lines:
        assert _TRUE16_RE.search(ln), (
            f"[{target}] merge_sum v_max_f16 is not in true16 form: {ln!r}"
        )
        assert not _FAKE16_RE.search(ln), (
            f"[{target}] merge_sum v_max_f16 emitted in fake16 form "
            f"(ROCM-3994 defect): {ln!r}"
        )
