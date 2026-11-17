# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""UseScaleAlphaVec solution support and the vector direction encoded in it."""

SCALE_ALPHA_VECTOR_MASK = 0x3
SCALE_ALPHA_DEVICE_SCALAR_BIT = 0x4
VALID_SCALE_ALPHA_VEC_MODES = (0, 1, 2, 3, 5, 6, 7)


def scaleAlphaVectorMode(modes):
    """Return legacy M/N/MN direction bits without the scalar capability bit."""
    return modes & SCALE_ALPHA_VECTOR_MASK


def scaleAlphaSupportsDeviceScalar(modes):
    """Whether the generated kernel must implement internalArg0[11]."""
    return bool(modes & SCALE_ALPHA_DEVICE_SCALAR_BIT)
