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
