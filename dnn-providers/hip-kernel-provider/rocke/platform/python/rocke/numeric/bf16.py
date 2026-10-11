# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free bf16 host encoding for numeric verify harnesses.

numpy has no native bfloat16 and numpy is rocKE's only hard dependency, so a
bf16 tensor is carried on the host as raw ``uint16`` and converted by hand --
per TESTING.md, "bf16 gets a hand-rolled encoding or an explicit
NotImplementedError, never a silent upcast".

This lives in ``platform`` so the one implementation is shared by every harness
that needs it (``library/builders`` verify drivers and the ``library/tests``
numeric gate alike). A second copy is how a harness ends up agreeing with
itself rather than with the device.
"""

from __future__ import annotations

__all__ = ["bf16_to_f32", "f32_to_bf16"]


def f32_to_bf16(a):
    """fp32 -> bf16 (raw ``uint16``) with round-to-nearest-even.

    Adds the tie-breaking bias ``0x7FFF + lsb`` to the fp32 bit pattern before
    truncating to the high 16 bits, which is exactly RNE on the retained
    mantissa. Plain truncation would bias every value toward zero and show up
    downstream as kernel "error".

    NaN is forced to a quiet NaN rather than carried through the bias. A NaN
    whose payload lives only in the low 16 bits -- ``0x7F800001``, say -- would
    otherwise have the bias carry into the exponent and emerge as ``0x7F80``,
    i.e. +inf. Silently turning a NaN into an infinity on the way to the device
    is exactly how a special-value masking bug gets hidden: the kernel would be
    judged on an input the harness never meant to send. The sign bit is kept so
    the sign of a NaN still survives the round trip.
    """
    import numpy as np

    f = np.ascontiguousarray(a, dtype=np.float32)
    u = f.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    out = ((u + bias) >> np.uint32(16)).astype(np.uint16)
    # Take the high 16 bits as-is and force the quiet bit on. That keeps sign,
    # exponent, and whatever payload already lives in the retained 7 mantissa
    # bits (0x7FD30000 -> 0x7FD3); only a payload confined to the discarded low
    # 16 bits has nothing to keep and lands on the canonical 0x7FC0. No
    # rounding happens here, which is the whole point -- rounding is what turned
    # the NaN into an infinity.
    quiet_nan = ((u >> np.uint32(16)).astype(np.uint16)) | np.uint16(0x0040)
    return np.where(np.isnan(f), quiet_nan, out).astype(np.uint16)


def bf16_to_f32(a):
    """bf16 (raw ``uint16``) -> fp32 by shifting the pattern back into place."""
    import numpy as np

    return (np.ascontiguousarray(a, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )
