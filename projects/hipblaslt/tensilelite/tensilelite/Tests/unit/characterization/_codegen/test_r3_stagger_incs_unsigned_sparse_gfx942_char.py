# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""Sparse-metadata StaggerU increment must be widened unsigned (CPU-only).

Companion to ``test_r3_stagger_incs_unsigned_gfx1201_char.py``, which covers the
A/B increments. ``calculateStagger`` has a third and fourth widening in its
sparse arm, on ``GlobalReadIncsMetadata``, and they have the same exposure:
the increment is a byte stride that cannot be negative, so widening it with
``s_mul_i64_i32`` sign-extends it once it reaches ``2^31`` and ``incrementMetadataSrd``
then walks the metadata descriptor backwards.

Why this needs its own config and arch. The arm is gated on
``ProblemType.Sparse``, and no shipped solution declares ``Sparse >= 1`` on any
architecture, so no tuned kernel reaches it. The sibling ``sparse_dtvsm.yaml``
does build sparse kernels but sets ``StaggerU: 0``, which compiles the stagger
path out, so it cannot cover this either. ``sparse_stagger.yaml`` is that config
with StaggerU kept on. gfx942 rather than gfx1201 because SMFMA sparse shapes
are a gfx9 feature.

Provenance, for anyone wondering why this sat unfixed. tensilelite added the
sparse metadata stagger in October 2022 using the signed idiom of the day.
Tensile fixed its own A/B widenings in February 2023 but has no metadata arm to
fix, so there was nothing to port. tensilelite's ``removeStagger`` metadata
multiply was fixed in December 2024. This arm is the last one.
"""

import os
import re

import pytest

import codegen_harness as _ch
import config_harness as _cfgh

from test_r3_stagger_incs_unsigned_gfx1201_char import _emit_asm

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    _ARCH,
    "sparse_stagger.yaml",
)

_LIMIT = 2

# The two comments calculateStagger attaches to the metadata widenings. The
# second is shared with the A/B arm, so it is only meaningful together with the
# WrapUMetadata destination.
_METADATA_COMMENT = "stagger byte offset of metadata"
_WRAPU_METADATA_DEST = "sgprWrapUMetadata"

_SIGNED_HI = re.compile(r"^\s*s_mul_hi_i32\b")
_UNSIGNED_HI = re.compile(r"^\s*s_mul_hi_u32\b")


@pytest.fixture(scope="module")
def emitted():
    asm = _emit_asm(_CONFIG, _ARCH, _LIMIT)
    assert asm, "no kernels emitted"
    return asm


def _metadata_widening_lines(src):
    """High-word multiply lines from the sparse arm of calculateStagger."""
    stagger = [
        line
        for line in src.splitlines()
        if _METADATA_COMMENT in line and "s_mul_hi_" in line
    ]
    wrapu = [
        line
        for line in src.splitlines()
        if _WRAPU_METADATA_DEST in line
        and "Number of bytes accessed by the unroll loop" in line
        and "s_mul_hi_" in line
    ]
    return stagger, wrapu


def test_metadata_incs_widened_unsigned(emitted):
    """Neither metadata widening may sign-extend GlobalReadIncsMetadata."""
    checked = 0
    for base, src in emitted:
        if "sgprStaggerUIter" not in src:
            # StaggerU compiled out for this variant; nothing to widen.
            continue
        stagger, wrapu = _metadata_widening_lines(src)
        assert stagger and wrapu, (
            f"{base}: the sparse arm of calculateStagger emitted no metadata "
            "widening. Either the comment text changed or the arm stopped being "
            "reached; re-point this test at the current site rather than deleting it."
        )
        for line in stagger + wrapu:
            assert not _SIGNED_HI.match(line), (
                f"{base}: GlobalReadIncsMetadata widened with a signed high multiply, "
                "which sign-extends it once the metadata byte stride reaches 2^31 "
                f"(ROCM-31230):\n    {line.strip()}"
            )
            assert _UNSIGNED_HI.match(line), (
                f"{base}: unexpected widening instruction:\n    {line.strip()}"
            )
        checked += 1

    assert checked, (
        "no emitted kernel reached the sparse stagger path, so this test proved "
        "nothing. Pick a config whose solutions keep Sparse and StaggerU enabled."
    )
