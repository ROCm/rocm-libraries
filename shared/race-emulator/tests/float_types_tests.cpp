// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/FloatTypes.h"
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

using namespace raceemulator;

// === BF16 tests ===

TEST(FloatTypes, Bf16RoundTrip) {
  float values[] = {0.0f, 1.0f, -1.0f, 2.0f, -0.5f, 100.0f};
  for (float v : values) {
    uint16_t bf = floatToBf16(v);
    float back = bf16ToFloat(bf);
    EXPECT_EQ(back, v) << "BF16 round-trip failed for " << v;
  }
}

TEST(FloatTypes, Bf16Zero) {
  EXPECT_EQ(bf16ToFloat(0x0000), 0.0f);
  EXPECT_EQ(bf16ToFloat(0x8000), -0.0f);
  EXPECT_EQ(floatToBf16(0.0f), 0x0000);
}

TEST(FloatTypes, Bf16Inf) {
  EXPECT_TRUE(std::isinf(bf16ToFloat(0x7F80))); // +inf
  EXPECT_TRUE(std::isinf(bf16ToFloat(0xFF80))); // -inf
}

TEST(FloatTypes, Bf16Nan) { EXPECT_TRUE(std::isnan(bf16ToFloat(0x7FC0))); }

// === F16 tests ===

TEST(FloatTypes, F16Zero) {
  EXPECT_EQ(f16ToFloat(0x0000), 0.0f);
  EXPECT_EQ(f16ToFloat(0x8000), -0.0f);
  EXPECT_EQ(floatToF16(0.0f), 0x0000);
  EXPECT_EQ(floatToF16(-0.0f), 0x8000);
}

TEST(FloatTypes, F16One) {
  // f16 1.0 = 0 01111 0000000000 = 0x3C00
  EXPECT_EQ(f16ToFloat(0x3C00), 1.0f);
  EXPECT_EQ(floatToF16(1.0f), 0x3C00);
}

TEST(FloatTypes, F16NegOne) {
  // f16 -1.0 = 1 01111 0000000000 = 0xBC00
  EXPECT_EQ(f16ToFloat(0xBC00), -1.0f);
  EXPECT_EQ(floatToF16(-1.0f), 0xBC00);
}

TEST(FloatTypes, F16Inf) {
  // +inf = 0 11111 0000000000 = 0x7C00
  EXPECT_TRUE(std::isinf(f16ToFloat(0x7C00)));
  EXPECT_GT(f16ToFloat(0x7C00), 0.0f);
  // -inf = 1 11111 0000000000 = 0xFC00
  EXPECT_TRUE(std::isinf(f16ToFloat(0xFC00)));
  EXPECT_LT(f16ToFloat(0xFC00), 0.0f);
  EXPECT_EQ(floatToF16(std::numeric_limits<float>::infinity()), 0x7C00);
  EXPECT_EQ(floatToF16(-std::numeric_limits<float>::infinity()), 0xFC00);
}

TEST(FloatTypes, F16Nan) {
  // NaN: exponent all 1s, mantissa non-zero
  float nan = f16ToFloat(0x7E00); // 0 11111 1000000000
  EXPECT_TRUE(std::isnan(nan));

  // Converting NaN f32 -> f16 should produce NaN
  uint16_t nanH = floatToF16(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(std::isnan(f16ToFloat(nanH)));
}

TEST(FloatTypes, F16MaxValue) {
  // Max f16 = 0 11110 1111111111 = 0x7BFF = 65504.0
  EXPECT_EQ(f16ToFloat(0x7BFF), 65504.0f);
  EXPECT_EQ(floatToF16(65504.0f), 0x7BFF);
}

TEST(FloatTypes, F16Denormal) {
  // Smallest positive denormal: 0 00000 0000000001 = 0x0001
  // Value = 2^(-14) * (0/1024 + 1/1024) = 2^(-14) * 2^(-10) = 2^(-24)
  float smallest = f16ToFloat(0x0001);
  EXPECT_GT(smallest, 0.0f);
  EXPECT_FLOAT_EQ(smallest, std::ldexp(1.0f, -24));

  // Largest denormal: 0 00000 1111111111 = 0x03FF
  // Value = 2^(-14) * (1023/1024)
  float largest_denorm = f16ToFloat(0x03FF);
  EXPECT_GT(largest_denorm, 0.0f);
  EXPECT_LT(largest_denorm, std::ldexp(1.0f, -14));
}

TEST(FloatTypes, F16RoundTrip) {
  // Test round-trip for normal values representable in f16
  float values[] = {0.5f,  1.0f,  -1.0f,  2.0f,
                    -2.0f, 0.25f, 100.0f, 0.001953125f};
  for (float v : values) {
    uint16_t h = floatToF16(v);
    float back = f16ToFloat(h);
    EXPECT_EQ(back, v) << "F16 round-trip failed for " << v;
  }
}

TEST(FloatTypes, F16Overflow) {
  // Values beyond f16 max (65504) should become inf
  uint16_t h = floatToF16(100000.0f);
  EXPECT_TRUE(std::isinf(f16ToFloat(h)));
}

TEST(FloatTypes, F16Underflow) {
  // Very small values below f16 denormal range should become zero
  uint16_t h = floatToF16(1e-10f);
  EXPECT_EQ(f16ToFloat(h), 0.0f);
}

// === FP4 (E2M1) tests ===

TEST(FloatTypes, Fp4AllValues) {
  // Exhaustive: all 16 possible nibble values.
  // Positive (nibbles 0-7):
  EXPECT_EQ(fp4ToFloat(0), 0.0f);
  EXPECT_EQ(fp4ToFloat(1), 0.5f);
  EXPECT_EQ(fp4ToFloat(2), 1.0f);
  EXPECT_EQ(fp4ToFloat(3), 1.5f);
  EXPECT_EQ(fp4ToFloat(4), 2.0f);
  EXPECT_EQ(fp4ToFloat(5), 3.0f);
  EXPECT_EQ(fp4ToFloat(6), 4.0f);
  EXPECT_EQ(fp4ToFloat(7), 6.0f);
  // Negative (nibbles 8-15):
  EXPECT_EQ(fp4ToFloat(8), -0.0f);
  EXPECT_EQ(fp4ToFloat(9), -0.5f);
  EXPECT_EQ(fp4ToFloat(10), -1.0f);
  EXPECT_EQ(fp4ToFloat(11), -1.5f);
  EXPECT_EQ(fp4ToFloat(12), -2.0f);
  EXPECT_EQ(fp4ToFloat(13), -3.0f);
  EXPECT_EQ(fp4ToFloat(14), -4.0f);
  EXPECT_EQ(fp4ToFloat(15), -6.0f);
}

TEST(FloatTypes, Fp4SignBit) {
  // Bit 3 is the sign bit: nibble ^ 8 flips sign.
  for (uint8_t i = 0; i < 8; ++i) {
    EXPECT_EQ(fp4ToFloat(i), -fp4ToFloat(i | 8))
        << "Sign mismatch for nibble " << (int)i;
  }
}

TEST(FloatTypes, Fp4MasksHighBits) {
  // Only the low 4 bits should matter.
  EXPECT_EQ(fp4ToFloat(0xF2), fp4ToFloat(2));
  EXPECT_EQ(fp4ToFloat(0xA7), fp4ToFloat(7));
}

// === E8M0 (shared exponent) tests ===

TEST(FloatTypes, E8m0Identity) {
  // exponent=127 → 2^0 = 1.0
  EXPECT_EQ(e8m0ToFloat(127), 1.0f);
}

TEST(FloatTypes, E8m0PowersOfTwo) {
  EXPECT_EQ(e8m0ToFloat(128), 2.0f);
  EXPECT_EQ(e8m0ToFloat(126), 0.5f);
  EXPECT_EQ(e8m0ToFloat(130), 8.0f);
  EXPECT_EQ(e8m0ToFloat(0), std::bit_cast<float>(uint32_t(0))); // 2^(-127)
}

TEST(FloatTypes, E8m0NanReturnsZero) {
  // exponent=255 is NaN in E8M0 spec.
  EXPECT_EQ(e8m0ToFloat(255), 0.0f);
}

TEST(FloatTypes, Fp4WithScale) {
  // FP4 value 3 (=1.5) scaled by E8M0 exponent 128 (=2.0) → 3.0
  EXPECT_EQ(fp4ToFloat(3) * e8m0ToFloat(128), 3.0f);
  // FP4 value 7 (=6.0) scaled by E8M0 exponent 126 (=0.5) → 3.0
  EXPECT_EQ(fp4ToFloat(7) * e8m0ToFloat(126), 3.0f);
}

// === mxfp4BlockToF32 tests ===

TEST(FloatTypes, Mxfp4BlockAllZeros) {
  uint8_t data[16] = {};
  float out[32];
  mxfp4BlockToF32(out, data, 127); // scale=1.0
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(out[i], 0.0f) << "index " << i;
  }
}

TEST(FloatTypes, Mxfp4BlockNibbleOrder) {
  // Byte 0x21: low nibble = 1 (0.5), high nibble = 2 (1.0).
  uint8_t data[16] = {};
  data[0] = 0x21;
  float out[32];
  mxfp4BlockToF32(out, data, 127); // scale=1.0
  EXPECT_EQ(out[0], 0.5f);         // low nibble
  EXPECT_EQ(out[1], 1.0f);         // high nibble
}

TEST(FloatTypes, Mxfp4BlockWithScale) {
  // All nibbles = 2 (1.0), scale = 128 (2.0) → all outputs 2.0.
  uint8_t data[16];
  std::fill(data, data + 16, 0x22); // both nibbles = 2
  float out[32];
  mxfp4BlockToF32(out, data, 128);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(out[i], 2.0f) << "index " << i;
  }
}

TEST(FloatTypes, Mxfp4BlockMixedValues) {
  uint8_t data[16] = {};
  // Byte 0: nibbles 7 (6.0) and 9 (=-0.5)
  data[0] = 0x97; // low=7, high=9
  float out[32];
  mxfp4BlockToF32(out, data, 127); // scale=1.0
  EXPECT_EQ(out[0], 6.0f);
  EXPECT_EQ(out[1], -0.5f);
}

// === mxfp4MatrixToF32 tests ===

TEST(FloatTypes, Mxfp4MatrixAllOnes) {
  // 2 rows × K=32 of all 1.0 (nibble 2), scale=1.0.
  constexpr int K = 32;
  std::vector<uint8_t> data(2 * K / 2, 0x22);
  std::vector<uint8_t> scales(2, 127);
  std::vector<float> out(2 * K);
  mxfp4MatrixToF32<2, K>(out.data(), data.data(), scales.data());
  for (int i = 0; i < 2 * K; ++i) {
    EXPECT_EQ(out[i], 1.0f) << "index " << i;
  }
}

TEST(FloatTypes, Mxfp4MatrixWithScale) {
  // 1 row × K=64 (2 blocks), scales = [2.0, 0.5].
  // All nibbles = 3 (1.5). Block 0: 1.5*2.0=3.0, Block 1: 1.5*0.5=0.75.
  constexpr int K = 64;
  std::vector<uint8_t> data(K / 2, 0x33);
  uint8_t scales[2] = {128, 126}; // 2.0, 0.5
  std::vector<float> out(K);
  mxfp4MatrixToF32<1, K>(out.data(), data.data(), scales);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(out[i], 3.0f) << "block 0, index " << i;
  }
  for (int i = 32; i < 64; ++i) {
    EXPECT_EQ(out[i], 0.75f) << "block 1, index " << i;
  }
}

TEST(FloatTypes, Mxfp4MatrixMultipleRows) {
  // 2 rows × K=32, row 0 all 1.0, row 1 all 2.0 (nibble 4).
  constexpr int K = 32;
  std::vector<uint8_t> data(2 * K / 2);
  std::fill(data.begin(), data.begin() + K / 2, 0x22); // row 0: 1.0
  std::fill(data.begin() + K / 2, data.end(), 0x44);   // row 1: 2.0
  std::vector<uint8_t> scales(2, 127);
  std::vector<float> out(2 * K);
  mxfp4MatrixToF32<2, K>(out.data(), data.data(), scales.data());
  for (int i = 0; i < K; ++i) {
    EXPECT_EQ(out[i], 1.0f) << "row 0, index " << i;
  }
  for (int i = K; i < 2 * K; ++i) {
    EXPECT_EQ(out[i], 2.0f) << "row 1, index " << i;
  }
}
