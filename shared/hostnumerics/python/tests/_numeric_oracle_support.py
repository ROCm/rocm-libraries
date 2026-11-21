# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math

import numpy as np


__all__ = [
    "counter_random",
    "decode_e8m0",
    "indexed_uniform_unit",
    "pack_little_endian",
    "quantize_bfloat16",
    "wrap_int32",
]

_MASK64 = (1 << 64) - 1


def counter_random(seed, domain, index):
    """SplitMix-style counter RNG, with every operation wrapped to uint64."""

    value = (
        int(seed)
        ^ ((int(domain) + 0x9E3779B97F4A7C15) & _MASK64)
        ^ ((int(index) * 0xBF58476D1CE4E5B9) & _MASK64)
    ) & _MASK64
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def indexed_uniform_unit(seed, domain, index):
    mantissa = counter_random(seed, domain, index) >> 11
    return (float(mantissa) + 0.5) / 9007199254740992.0


def pack_little_endian(codes, bits_per_value):
    storage = bytearray((len(codes) * bits_per_value + 7) // 8)
    for index, code in enumerate(codes):
        bit_offset = index * bits_per_value
        for bit in range(bits_per_value):
            if (code >> bit) & 1:
                absolute_bit = bit_offset + bit
                storage[absolute_bit // 8] |= 1 << (absolute_bit % 8)
    return bytes(storage)


def decode_e8m0(raw):
    return math.nan if raw == 0xFF else math.ldexp(1.0, raw - 127)


def quantize_bfloat16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    least_significant_bit = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + least_significant_bit
    quantized = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    return quantized[()] if quantized.ndim == 0 else quantized


def wrap_int32(value):
    unsigned = int(value) & 0xFFFFFFFF
    return unsigned if unsigned < 0x80000000 else unsigned - 0x100000000
