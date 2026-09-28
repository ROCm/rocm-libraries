# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The LDS padding value space the gfx1250 solver works in.

``ldsPadError`` and ``ldsBlockError`` are the rules. ``ldsPadValues`` and
``ldsBlockValues`` are the values each rule accepts.

The bounds are the TDM descriptor's, and the solver keeps to them on every read
path it models, TDM or not.

Imports nothing: ``ValidParameters`` cannot reach ``SolutionStructs``, whose
package import pulls in ``Solution``, which imports ``Common`` right back.
"""

# TDM encodes the pad in a descriptor:
#   pad_interval = log2(LdsBlockSizePerPad // 4) - 1, must fit in 3 bits => <=7
#   pad_amount   = LdsPad // 4 - 1,                  must fit in 7 bits => <=127
# So the block is a power of two from 8 to 1024 bytes and the pad is a positive
# multiple of 4 up to 512.
LDS_PAD_BLOCK_BYTES = (8, 16, 32, 64, 128, 256, 512, 1024)
LDS_MAX_PAD_BYTES   = 512

# LdsPad in bytes must be an even number of dwords. Odd-dword padding leaves
# the hardware in a state we do not understand, so it is never a candidate.
LDS_PAD_STEP_BYTES = 8

# The pad has to hold the load's own alignment, so b128 steps by 16; the rest
# take LDS_PAD_STEP_BYTES.
B128_PAD_STEP_BYTES = 16

# (bytes per element, pad step in bytes) per read path.
#   FP4  ds_load_tr4_b64    FP8  ds_load_tr8_b64
#   FP16 ds_load_tr16_b128  FP32 ds_load_b32
LDS_PAD_UNITS = ((0.5, LDS_PAD_STEP_BYTES), (1.0, LDS_PAD_STEP_BYTES),
                 (2.0, B128_PAD_STEP_BYTES), (4.0, LDS_PAD_STEP_BYTES))


# ---------------------------------------------------------------------------
# The rules. Each returns why the value cannot be used, or None.
# ---------------------------------------------------------------------------

def ldsPadError(padBytes, stepBytes=LDS_PAD_STEP_BYTES):
    """Why this pad, in bytes, cannot be used.

    stepBytes is what the pad has to be a multiple of, and it belongs to the
    path: LDS_PAD_STEP_BYTES for the even-dword paths, B128_PAD_STEP_BYTES
    where ds_load_tr16_b128 reads it, and one dword for the sparse metadata,
    which rounds its pad up from a vector width.
    """
    if padBytes % stepBytes:
        return "%dB is not a multiple of %dB" % (padBytes, stepBytes)
    if not 0 < padBytes <= LDS_MAX_PAD_BYTES:
        return "%dB is outside (0, %dB]" % (padBytes, LDS_MAX_PAD_BYTES)
    return None


def ldsBlockError(blockBytes):
    """Why this pad block, in bytes, cannot be used."""
    if blockBytes not in LDS_PAD_BLOCK_BYTES:
        return ("%dB is not a power of two in [%d, %d]"
                % (blockBytes, LDS_PAD_BLOCK_BYTES[0], LDS_PAD_BLOCK_BYTES[-1]))
    return None


# ---------------------------------------------------------------------------
# The values each rule accepts, for the yaml parameter lists. A yaml is read
# before the data type is resolved, so LdsPad is the union over the read
# paths and the rule decides again once the type is known.
# ---------------------------------------------------------------------------

def ldsPadValues():
    """Every LdsPad a yaml may name, counted in the operand's elements.

    The rule works in bytes and LdsPad counts elements, so one byte size lands
    on a different number per data type. A value here only means some path can
    reach it; ldsPadError is what decides for the path in hand.
    """
    return {int(padBytes / bpeDS)
            for bpeDS, stepBytes in LDS_PAD_UNITS
            for padBytes in range(stepBytes, LDS_MAX_PAD_BYTES + 1, stepBytes)
            if ldsPadError(padBytes, stepBytes) is None}


def ldsBlockValues():
    """Every LdsBlockSizePerPad a yaml may name. This one is already in bytes,
    so it is the rule's own set."""
    return {blockBytes for blockBytes in LDS_PAD_BLOCK_BYTES
            if ldsBlockError(blockBytes) is None}
