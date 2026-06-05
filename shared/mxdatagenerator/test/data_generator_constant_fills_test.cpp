// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tests for constant-fill and pathological DataInitMode variants
// (Twos, NegOnes, MaxVals, DenormMins, DenormMaxs, NaNs, Infs, RandInt).
// Each mode has DTYPE-specific expectations, so this is a hand-rolled
// per-DTYPE matrix rather than TYPED_TEST_SUITE.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/bf16.hpp>
#include <mxDataGenerator/f32.hpp>
#include <mxDataGenerator/fp16.hpp>

using namespace DGen;

namespace
{
    // 32 matches the canonical MX block size used everywhere downstream.
    constexpr index_t kBlockScaling = 32;
    constexpr index_t kRows         = 32;
    constexpr index_t kCols         = 4;

    template <typename DTYPE>
    DataGenerator<DTYPE> generateConstant(DataInitMode initMode)
    {
        DataGeneratorOptions opts;
        opts.blockScaling = kBlockScaling;
        opts.initMode     = initMode;
        opts.forceDenorm  = false;
        std::vector<index_t> sizes{kRows, kCols};
        std::vector<index_t> strides{1, kRows};
        DataGenerator<DTYPE> dgen;
        dgen.generate(sizes, strides, opts);
        return dgen;
    }

    template <typename DTYPE>
    void expectAllValues(DataInitMode initMode, float expected)
    {
        auto dgen = generateConstant<DTYPE>(initMode);
        auto ref  = dgen.getReferenceFloat();
        ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
        for(size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_EQ(ref[i], expected) << "element " << i;
        }
    }

    template <typename DTYPE>
    void expectAllNaN(DataInitMode initMode)
    {
        auto dgen = generateConstant<DTYPE>(initMode);
        auto ref  = dgen.getReferenceFloat();
        ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
        for(size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_TRUE(std::isnan(ref[i])) << "element " << i << " = " << ref[i];
        }
    }

    template <typename DTYPE>
    void expectAllInf(DataInitMode initMode)
    {
        auto dgen = generateConstant<DTYPE>(initMode);
        auto ref  = dgen.getReferenceFloat();
        ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
        for(size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_TRUE(std::isinf(ref[i]) && ref[i] > 0)
                << "element " << i << " = " << ref[i];
        }
    }
} // namespace

// Twos: dequantized == 2.0 for every supported DTYPE.
TEST(DataGeneratorConstantFills, TwosF4_E8M0)
{
    expectAllValues<ocp_e2m1_mxfp4>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF4_E4M3)
{
    expectAllValues<ocp_e2m1_mxfp4_e4m3>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF4_E5M3)
{
    expectAllValues<ocp_e2m1_mxfp4_e5m3>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF6_E2M3)
{
    expectAllValues<ocp_e2m3_mxfp6>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF6_E3M2)
{
    expectAllValues<ocp_e3m2_mxfp6>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF8_E4M3)
{
    expectAllValues<ocp_e4m3_mxfp8>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF8_E5M2)
{
    expectAllValues<ocp_e5m2_mxfp8>(Twos{}, 2.0f);
}

// NegOnes: dequantized == -1.0 for every supported DTYPE.
TEST(DataGeneratorConstantFills, NegOnesF4_E8M0)
{
    expectAllValues<ocp_e2m1_mxfp4>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF4_E4M3)
{
    expectAllValues<ocp_e2m1_mxfp4_e4m3>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF6_E2M3)
{
    expectAllValues<ocp_e2m3_mxfp6>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF6_E3M2)
{
    expectAllValues<ocp_e3m2_mxfp6>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF8_E4M3)
{
    expectAllValues<ocp_e4m3_mxfp8>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF8_E5M2)
{
    expectAllValues<ocp_e5m2_mxfp8>(NegOnes{}, -1.0f);
}

// MaxVals: dequantized == data type's max normal number.
TEST(DataGeneratorConstantFills, MaxValsF4)
{
    expectAllValues<ocp_e2m1_mxfp4>(MaxVals{}, 6.0f);
}
TEST(DataGeneratorConstantFills, MaxValsF6_E2M3)
{
    expectAllValues<ocp_e2m3_mxfp6>(MaxVals{}, 7.5f);
}
TEST(DataGeneratorConstantFills, MaxValsF6_E3M2)
{
    expectAllValues<ocp_e3m2_mxfp6>(MaxVals{}, 28.0f);
}
TEST(DataGeneratorConstantFills, MaxValsF8_E4M3)
{
    expectAllValues<ocp_e4m3_mxfp8>(MaxVals{}, 448.0f);
}
TEST(DataGeneratorConstantFills, MaxValsF8_E5M2)
{
    expectAllValues<ocp_e5m2_mxfp8>(MaxVals{}, 57344.0f);
}

// DenormMins: every element is the same nonzero positive subnormal.
// (Per-DTYPE magnitudes differ, so we don't pin the exact float.)
template <typename DTYPE>
void expectDenormMinAllEqualNonzero()
{
    auto dgen = generateConstant<DTYPE>(DenormMins{});
    auto ref  = dgen.getReferenceFloat();
    ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
    ASSERT_NE(ref[0], 0.0f);
    EXPECT_GT(ref[0], 0.0f);
    for(size_t i = 1; i < ref.size(); ++i)
    {
        EXPECT_EQ(ref[i], ref[0]) << "element " << i;
    }
}

TEST(DataGeneratorConstantFills, DenormMinsF4)
{
    expectDenormMinAllEqualNonzero<ocp_e2m1_mxfp4>();
}
TEST(DataGeneratorConstantFills, DenormMinsF6_E2M3)
{
    expectDenormMinAllEqualNonzero<ocp_e2m3_mxfp6>();
}
TEST(DataGeneratorConstantFills, DenormMinsF6_E3M2)
{
    expectDenormMinAllEqualNonzero<ocp_e3m2_mxfp6>();
}
TEST(DataGeneratorConstantFills, DenormMinsF8_E4M3)
{
    expectDenormMinAllEqualNonzero<ocp_e4m3_mxfp8>();
}
TEST(DataGeneratorConstantFills, DenormMinsF8_E5M2)
{
    expectDenormMinAllEqualNonzero<ocp_e5m2_mxfp8>();
}

// DenormMaxs: every element is the largest positive subnormal.
// (F4 only has one subnormal, so DenormMaxs == DenormMins there.)
template <typename DTYPE>
void expectDenormMaxAllEqualNonzero()
{
    auto dgen = generateConstant<DTYPE>(DenormMaxs{});
    auto ref  = dgen.getReferenceFloat();
    ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
    ASSERT_NE(ref[0], 0.0f);
    EXPECT_GT(ref[0], 0.0f);
    for(size_t i = 1; i < ref.size(); ++i)
    {
        EXPECT_EQ(ref[i], ref[0]) << "element " << i;
    }
}

TEST(DataGeneratorConstantFills, DenormMaxsF6_E2M3)
{
    expectDenormMaxAllEqualNonzero<ocp_e2m3_mxfp6>();
}
TEST(DataGeneratorConstantFills, DenormMaxsF6_E3M2)
{
    expectDenormMaxAllEqualNonzero<ocp_e3m2_mxfp6>();
}
TEST(DataGeneratorConstantFills, DenormMaxsF8_E4M3)
{
    expectDenormMaxAllEqualNonzero<ocp_e4m3_mxfp8>();
}
TEST(DataGeneratorConstantFills, DenormMaxsF8_E5M2)
{
    expectDenormMaxAllEqualNonzero<ocp_e5m2_mxfp8>();
}

TEST(DataGeneratorConstantFills, DenormMinLessOrEqualDenormMax_F8_E4M3)
{
    auto dmin = generateConstant<ocp_e4m3_mxfp8>(DenormMins{}).getReferenceFloat();
    auto dmax = generateConstant<ocp_e4m3_mxfp8>(DenormMaxs{}).getReferenceFloat();
    ASSERT_FALSE(dmin.empty());
    ASSERT_FALSE(dmax.empty());
    EXPECT_LE(dmin[0], dmax[0]);
    EXPECT_LT(0.0f, dmin[0]);
}

// NaNs: every dequantized element is NaN. setNaN routes through whichever
// buffer can encode NaN (data byte for F8 E4M3/E5M2 with E8M0 scales;
// scale byte for F4/F6 and any F8 with E4M3/E5M3 scales).
TEST(DataGeneratorConstantFills, NaNsF4_E8M0)
{
    expectAllNaN<ocp_e2m1_mxfp4>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF4_E4M3)
{
    expectAllNaN<ocp_e2m1_mxfp4_e4m3>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF4_E5M3)
{
    expectAllNaN<ocp_e2m1_mxfp4_e5m3>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF6_E2M3)
{
    expectAllNaN<ocp_e2m3_mxfp6>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF6_E3M2)
{
    expectAllNaN<ocp_e3m2_mxfp6>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF8_E4M3)
{
    expectAllNaN<ocp_e4m3_mxfp8>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF8_E5M2)
{
    expectAllNaN<ocp_e5m2_mxfp8>(NaNs{});
}

// Infs: only F8 E5M2 has an Inf representation; every other DTYPE must throw.
TEST(DataGeneratorConstantFills, InfsF8_E5M2)
{
    expectAllInf<ocp_e5m2_mxfp8>(Infs{});
}

template <typename DTYPE>
void expectInfsThrow()
{
    DataGeneratorOptions opts;
    opts.blockScaling = kBlockScaling;
    opts.initMode     = Infs{};
    opts.forceDenorm  = false;
    std::vector<index_t> sizes{kRows, kCols};
    std::vector<index_t> strides{1, kRows};
    DataGenerator<DTYPE> dgen;
    EXPECT_THROW(dgen.generate(sizes, strides, opts), std::runtime_error);
}

TEST(DataGeneratorConstantFills, InfsF4_Throws)
{
    expectInfsThrow<ocp_e2m1_mxfp4>();
}
TEST(DataGeneratorConstantFills, InfsF6_E2M3_Throws)
{
    expectInfsThrow<ocp_e2m3_mxfp6>();
}
TEST(DataGeneratorConstantFills, InfsF6_E3M2_Throws)
{
    expectInfsThrow<ocp_e3m2_mxfp6>();
}
TEST(DataGeneratorConstantFills, InfsF8_E4M3_Throws)
{
    // E4M3 lacks an Inf representation even though it is an 8-bit float.
    expectInfsThrow<ocp_e4m3_mxfp8>();
}

// RandInt: integerness + range bounds + non-degenerate spread for the per-DTYPE
// ranges that hipBLASLt's legacy random_int<T> uses.
template <typename DTYPE>
void expectRandIntInRange(int lo, int hi)
{
    DataGeneratorOptions opts;
    opts.blockScaling = kBlockScaling;
    opts.initMode     = RandInt{lo, hi};
    opts.forceDenorm  = false;
    std::vector<index_t> sizes{kRows, kCols};
    std::vector<index_t> strides{1, kRows};
    DataGenerator<DTYPE> dgen;
    // Use a fixed seed so the spread assertion is deterministic.
    dgen.setSeed(123u);
    dgen.generate(sizes, strides, opts);
    auto ref = dgen.getReferenceFloat();
    ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));

    int distinctCount = 0;
    bool seen[256]    = {};
    for(size_t i = 0; i < ref.size(); ++i)
    {
        float const v = ref[i];
        EXPECT_FALSE(std::isnan(v)) << "element " << i;
        EXPECT_FALSE(std::isinf(v)) << "element " << i;
        EXPECT_EQ(v, std::trunc(v)) << "element " << i << " = " << v;
        EXPECT_GE(v, static_cast<float>(lo)) << "element " << i;
        EXPECT_LE(v, static_cast<float>(hi)) << "element " << i;
        int const slot = static_cast<int>(v) - lo;
        if(slot >= 0 && slot < 256 && !seen[slot])
        {
            seen[slot] = true;
            ++distinctCount;
        }
    }
    if(hi > lo)
    {
        EXPECT_GT(distinctCount, 1) << "RandInt produced a degenerate sample";
    }
}

TEST(DataGeneratorConstantFills, RandIntF4_Range_m4_4)
{
    expectRandIntInRange<ocp_e2m1_mxfp4>(-4, 4);
}
TEST(DataGeneratorConstantFills, RandIntF6_E2M3_Range_m7_7)
{
    expectRandIntInRange<ocp_e2m3_mxfp6>(-7, 7);
}
TEST(DataGeneratorConstantFills, RandIntF6_E3M2_Range_m28_28)
{
    expectRandIntInRange<ocp_e3m2_mxfp6>(-28, 28);
}
TEST(DataGeneratorConstantFills, RandIntF8_E4M3_Range_1_10)
{
    expectRandIntInRange<ocp_e4m3_mxfp8>(1, 10);
}
TEST(DataGeneratorConstantFills, RandIntF8_E5M2_Range_1_10)
{
    expectRandIntInRange<ocp_e5m2_mxfp8>(1, 10);
}
TEST(DataGeneratorConstantFills, RandIntDegenerateRangeIsConstant)
{
    // lo == hi must produce that single value at every element.
    DataGeneratorOptions opts;
    opts.blockScaling = kBlockScaling;
    opts.initMode     = RandInt{3, 3};
    opts.forceDenorm  = false;
    std::vector<index_t> sizes{kRows, kCols};
    std::vector<index_t> strides{1, kRows};
    DataGenerator<ocp_e4m3_mxfp8> dgen;
    dgen.generate(sizes, strides, opts);
    auto ref = dgen.getReferenceFloat();
    for(float v : ref)
        EXPECT_EQ(v, 3.0f);
}
TEST(DataGeneratorConstantFills, RandIntInvertedRangeThrows)
{
    DataGeneratorOptions opts;
    opts.blockScaling = kBlockScaling;
    opts.initMode     = RandInt{5, -5};
    opts.forceDenorm  = false;
    std::vector<index_t> sizes{kRows, kCols};
    std::vector<index_t> strides{1, kRows};
    DataGenerator<ocp_e4m3_mxfp8> dgen;
    EXPECT_THROW(dgen.generate(sizes, strides, opts), std::invalid_argument);
}

// Multi-byte (host) DTYPE coverage: bf16 / fp16 / f32. Regression guard for
// the original `generate_data_constant_byte` helper which truncated the data
// bit-pattern to a single byte (invisible to MX types, but bf16(2.0)=0x4000
// became 0.0 because only the low 0x00 was written).
namespace
{
    template <typename DTYPE>
    void expectAllValuesUnscaled(DataInitMode initMode, float expected)
    {
        auto dgen = generateConstant<DTYPE>(initMode);
        auto ref  = dgen.getReferenceFloat();
        ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
        for(size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_EQ(ref[i], expected) << "element " << i;
        }
    }
} // namespace

TEST(DataGeneratorConstantFills, TwosBf16)
{
    expectAllValuesUnscaled<bf16>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosFp16)
{
    expectAllValuesUnscaled<fp16>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF32)
{
    expectAllValuesUnscaled<f32>(Twos{}, 2.0f);
}

TEST(DataGeneratorConstantFills, NegOnesBf16)
{
    expectAllValuesUnscaled<bf16>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesFp16)
{
    expectAllValuesUnscaled<fp16>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF32)
{
    expectAllValuesUnscaled<f32>(NegOnes{}, -1.0f);
}

TEST(DataGeneratorConstantFills, MaxValsBf16)
{
    expectAllValuesUnscaled<bf16>(MaxVals{}, bf16::dataMaxNormalNumber);
}
TEST(DataGeneratorConstantFills, MaxValsFp16)
{
    expectAllValuesUnscaled<fp16>(MaxVals{}, fp16::dataMaxNormalNumber);
}
TEST(DataGeneratorConstantFills, MaxValsF32)
{
    expectAllValuesUnscaled<f32>(MaxVals{}, f32::dataMaxNormalNumber);
}

// Use std::ldexp to compute the IEEE-correct denorm-min instead of relying
// on each DTYPE's `dataMinSubNormalNumber` literal -- some (fp16) are
// truncated decimal approximations that don't bit-match getReferenceFloat().
TEST(DataGeneratorConstantFills, DenormMinsBf16)
{
    expectAllValuesUnscaled<bf16>(DenormMins{}, std::ldexp(1.0f, -133));
}
TEST(DataGeneratorConstantFills, DenormMinsFp16)
{
    expectAllValuesUnscaled<fp16>(DenormMins{}, std::ldexp(1.0f, -24));
}
TEST(DataGeneratorConstantFills, DenormMinsF32)
{
    expectAllValuesUnscaled<f32>(DenormMins{}, std::ldexp(1.0f, -149));
}

// Pass the per-type smallest-normal-float threshold in by value rather than
// computing it from `DTYPE::dataMinSubNormalNumber * (1ull << 24)`. The
// previous formula relied on a denormal float surviving a runtime
// multiplication, which is wrong under flush-to-zero (bf16's 2^-133 and f32's
// 2^-149 dataMinSubNormalNumber both flush to 0 before the multiply, giving
// a useless `< 0` bound). `std::ldexp(1.0f, ...)` produces the threshold via
// bit-pattern construction so it's FTZ-safe, and the smallest-normal itself
// is a normal float for every type we test here.
template <typename DTYPE>
void expectDenormMaxAllPositiveBelow(float smallestNormal)
{
    auto dgen = generateConstant<DTYPE>(DenormMaxs{});
    auto ref  = dgen.getReferenceFloat();
    ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
    ASSERT_GT(ref[0], 0.0f);
    // largest-denorm < smallest-normal is the type-agnostic ordering invariant.
    EXPECT_LT(ref[0], smallestNormal);
    for(size_t i = 1; i < ref.size(); ++i)
        EXPECT_EQ(ref[i], ref[0]) << "element " << i;
}

TEST(DataGeneratorConstantFills, DenormMaxsBf16)
{
    expectDenormMaxAllPositiveBelow<bf16>(std::ldexp(1.0f, -126));
}
TEST(DataGeneratorConstantFills, DenormMaxsFp16)
{
    expectDenormMaxAllPositiveBelow<fp16>(std::ldexp(1.0f, -14));
}
TEST(DataGeneratorConstantFills, DenormMaxsF32)
{
    expectDenormMaxAllPositiveBelow<f32>(std::ldexp(1.0f, -126));
}

// Cross-check: DenormMaxs > DenormMins for every multi-byte type.
TEST(DataGeneratorConstantFills, DenormMaxGreaterThanMin_Bf16)
{
    auto dmin = generateConstant<bf16>(DenormMins{}).getReferenceFloat();
    auto dmax = generateConstant<bf16>(DenormMaxs{}).getReferenceFloat();
    EXPECT_LT(dmin[0], dmax[0]);
}
TEST(DataGeneratorConstantFills, DenormMaxGreaterThanMin_Fp16)
{
    auto dmin = generateConstant<fp16>(DenormMins{}).getReferenceFloat();
    auto dmax = generateConstant<fp16>(DenormMaxs{}).getReferenceFloat();
    EXPECT_LT(dmin[0], dmax[0]);
}
TEST(DataGeneratorConstantFills, DenormMaxGreaterThanMin_F32)
{
    auto dmin = generateConstant<f32>(DenormMins{}).getReferenceFloat();
    auto dmax = generateConstant<f32>(DenormMaxs{}).getReferenceFloat();
    EXPECT_LT(dmin[0], dmax[0]);
}

TEST(DataGeneratorConstantFills, NaNsBf16)
{
    expectAllNaN<bf16>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsFp16)
{
    expectAllNaN<fp16>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF32)
{
    expectAllNaN<f32>(NaNs{});
}
TEST(DataGeneratorConstantFills, InfsBf16)
{
    expectAllInf<bf16>(Infs{});
}
TEST(DataGeneratorConstantFills, InfsFp16)
{
    expectAllInf<fp16>(Infs{});
}
TEST(DataGeneratorConstantFills, InfsF32)
{
    expectAllInf<f32>(Infs{});
}

TEST(DataGeneratorConstantFills, RandIntBf16_Range_m10_10)
{
    expectRandIntInRange<bf16>(-10, 10);
}
TEST(DataGeneratorConstantFills, RandIntFp16_Range_m10_10)
{
    expectRandIntInRange<fp16>(-10, 10);
}
TEST(DataGeneratorConstantFills, RandIntF32_Range_m100_100)
{
    expectRandIntInRange<f32>(-100, 100);
}
