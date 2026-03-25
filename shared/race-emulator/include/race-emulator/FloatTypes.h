// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef RACE_EMULATOR_FLOAT_TYPES_H
#define RACE_EMULATOR_FLOAT_TYPES_H

#include <bit>
#include <cstdint>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace raceemulator {

/// BF16 (bfloat16): 1 sign, 8 exponent (bias 127), 7 mantissa.
/// Identical to the upper 16 bits of IEEE 754 binary32.

inline float bf16ToFloat(uint16_t b) {
  uint32_t val = static_cast<uint32_t>(b) << 16;
  return std::bit_cast<float>(val);
}

inline uint16_t floatToBf16(float f) {
  uint32_t bits = std::bit_cast<uint32_t>(f);
  return static_cast<uint16_t>(bits >> 16);
}

/// F16 (IEEE 754 binary16): 1 sign, 5 exponent (bias 15), 10 mantissa.

inline float f16ToFloat(uint16_t h) {
#if defined(__F16C__)
  // Hardware F16C: single vcvtsh2ss instruction.
  return _cvtsh_ss(h);
#else
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  uint32_t f32;

  if (exp == 0) {
    if (mant == 0) {
      // Zero (positive or negative)
      f32 = sign << 31;
    } else {
      // Denormal: normalize it
      // Shift mantissa left until the leading 1 is in bit 10
      exp = 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        exp++;
      }
      mant &= 0x3FF; // Remove the leading 1
      // f16 exponent 1 maps to f32 exponent (1 - 15 + 127) = 113
      // We subtracted (exp-1) extra, so f32_exp = 113 - (exp - 1) = 114 - exp
      f32 = (sign << 31) | ((114 - exp) << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    // Inf or NaN: f32 exponent = 0xFF
    f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
  } else {
    // Normal: rebias exponent from bias-15 to bias-127
    f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  }

  return std::bit_cast<float>(f32);
#endif
}

inline uint16_t floatToF16(float f) {
#if defined(__F16C__)
  // Hardware F16C: single vcvtss2sh instruction (round-to-nearest-even).
  return _cvtss_sh(f, 0);
#else
  uint32_t bits = std::bit_cast<uint32_t>(f);
  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF);
  uint32_t mant = bits & 0x7FFFFF;

  uint16_t h;

  if (exp == 0xFF) {
    // Inf or NaN
    h = static_cast<uint16_t>((sign << 15) | (0x1F << 10) | (mant >> 13));
  } else if (exp > 142) {
    // Overflow -> Inf (f32 exponent > 15 + 127 = 142)
    h = static_cast<uint16_t>((sign << 15) | (0x1F << 10));
  } else if (exp < 103) {
    // Underflow -> zero (f32 exponent < -14 + 127 - 10 = 103, too small even
    // for f16 denormals)
    h = static_cast<uint16_t>(sign << 15);
  } else if (exp < 113) {
    // Denormal range (f32 exponent < 1 - 15 + 127 = 113)
    // Add the implicit leading 1 to mantissa, then shift right
    mant |= 0x800000;
    int shift = 113 - exp + 13; // 13 to go from 23-bit to 10-bit mantissa
    h = static_cast<uint16_t>((sign << 15) | (mant >> shift));
  } else {
    // Normal
    h = static_cast<uint16_t>((sign << 15) | ((exp - 112) << 10) |
                              (mant >> 13));
  }

  return h;
#endif
}

} // namespace raceemulator

#endif // RACE_EMULATOR_FLOAT_TYPES_H
