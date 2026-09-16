################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R14 -- pins the rocisa fp8->f16 convert-instruction contract behind F3.

Components/LocalRead.py's ConvertAfterDS fp8->f16 ``elif
asmCaps["HasCvtFP8toF16"]`` arm calls ``VCvtScalePkFP8toF16(...)`` WITHOUT the
required positional ``scale`` argument. This pins the rocisa contract that makes
that arm a latent TypeError:

  - VCvtScalePkFP8toF16 REQUIRES scale (TypeError if omitted).
  - VCvtFP8toF16 takes NO scale (constructs without it).

So the F3 fix is to SWAP that arm to the non-scaled VCvtFP8toF16 (not, as the
original finding said, to add a scale to the scaled class). The buggy arm is
unreachable on any shipping arch (MFMA is CDNA; byte-sel v_cvt_f16_fp8 is RDNA4),
so this is a correctness-hygiene contract pin, not an emit test. See
work/char-findings.md F3 and .handoff/.../remediation-plan.md.
"""

import pytest

pytestmark = pytest.mark.unit


def test_r14_vcvt_scalepk_fp8_to_f16_requires_scale():
    """VCvtScalePkFP8toF16 raises TypeError when constructed without `scale`."""
    from rocisa.instruction import VCvtScalePkFP8toF16
    from rocisa.container import vgpr, VOP3PModifiers

    with pytest.raises(TypeError):
        VCvtScalePkFP8toF16(vgpr(0), vgpr(1), vop3=VOP3PModifiers())


def test_r14_vcvt_fp8_to_f16_takes_no_scale():
    """VCvtFP8toF16 constructs without a scale operand (the correct F3 target)."""
    from rocisa.instruction import VCvtFP8toF16
    from rocisa.container import vgpr, VOP3PModifiers

    inst = VCvtFP8toF16(vgpr(0), vgpr(1), vop3=VOP3PModifiers())
    assert inst is not None
