################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R14 -- half-precision Activation getModule pins the F7 defects.

Tensile/Activation.py's half/PK activation arms reference names never imported
(SelectBit, UnusedBit, VMaxF16, SDWAModifiers, VOP3PModifiers) and use `coef`
before assignment in the half gelu branch. Calling getModule(DataType('h'), ...)
therefore raises NameError / UnboundLocalError for these activations.

This test PINS the CURRENT (defective) behavior. When the F7 fix lands (add the
five missing rocisa imports and hoist `coef`), this test flips to assert these
five activations build a non-empty Module (covering Activation.py half branches
~507-523, 581-628, 911-924). The exp-family half paths (exp/sigmoid/tanh/silu/
swish) are out of scope (blocked by the separate TransOpWait decision). See
work/char-findings.md F7 and .handoff/.../remediation-plan.md.
"""

import pytest

pytestmark = pytest.mark.unit

# Activations whose half path the F7 import+coef fix unblocks.
_FIXABLE = ["clippedrelu", "leakyrelu", "clamp", "gelu", "geluscaling"]


def _init_gfx942():
    from rocisa import rocIsa
    rocIsa.getInstance().init((9, 4, 2), "", False)


def test_r14_activation_half_pins_defect():
    """Each fixable half activation currently raises NameError/UnboundLocalError."""
    _init_gfx942()
    from Tensile.Activation import ActivationModule, ActivationType
    from Tensile.Common.DataType import DataType

    failures = {}
    for act in _FIXABLE:
        try:
            ActivationModule().getModule(DataType("h"), ActivationType(act), "0", "0")
            failures[act] = "NO-RAISE"
        except (NameError, UnboundLocalError):
            pass
        except Exception as e:  # noqa: BLE001 - pin only the documented defect classes
            failures[act] = f"{type(e).__name__}: {e}"
    assert not failures, (
        "Expected NameError/UnboundLocalError (F7 defect) for all fixable half "
        f"activations; got: {failures}"
    )
