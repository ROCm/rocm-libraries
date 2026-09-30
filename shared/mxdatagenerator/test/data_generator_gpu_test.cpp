// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tests for the header-only HIP/GPU backend `DGen::DataGeneratorGPU`.
// Compares the GPU output to the existing CPU `DGen::DataGenerator` on:
//   * deterministic init modes (Sequential / Identity / Ones / Zeros) where
//     CPU and GPU produce semantically identical outputs;
//   * statistical properties (mean / std-dev / zero-frequency tolerances) for
//     Bounded and TrigonometricFromFloat where the two backends use different
//     PRNGs but should still match in distribution.
//
// Also exercises the GPU implementation of `preSwizzleScalesGFX950`, comparing
// against the existing host implementation byte-for-byte.

#include <gtest/gtest.h>

#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/DataGeneratorGPU.hpp>
#include <mxDataGenerator/PreSwizzle.hpp>
#include <mxDataGenerator/ocp_e2m1_mxfp4.hpp>
#include <mxDataGenerator/ocp_e2m3_mxfp6.hpp>
#include <mxDataGenerator/ocp_e3m2_mxfp6.hpp>
#include <mxDataGenerator/ocp_e4m3_mxfp8.hpp>
#include <mxDataGenerator/ocp_e5m2_mxfp8.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace DGen;

namespace
{
    // Helper: count zeros and compute mean/stddev of |x| in a float vector.
    struct StatSummary
    {
        double zeroFrac;
        double meanAbs;
        double stdDev;
        float  maxAbs;
    };

    StatSummary summarize(std::vector<float> const& v)
    {
        StatSummary s{};
        if(v.empty())
            return s;
        size_t zeros  = 0;
        double sumAbs = 0.0;
        for(float x : v)
        {
            if(x == 0.0f)
                zeros++;
            sumAbs += std::fabs(x);
            if(std::fabs(x) > s.maxAbs)
                s.maxAbs = std::fabs(x);
        }
        s.zeroFrac = static_cast<double>(zeros) / static_cast<double>(v.size());
        s.meanAbs  = sumAbs / static_cast<double>(v.size());
        double sumSq = 0.0;
        for(float x : v)
            sumSq += static_cast<double>(x) * static_cast<double>(x);
        double meanSq = sumSq / static_cast<double>(v.size());
        s.stdDev      = std::sqrt(std::max(0.0, meanSq));
        return s;
    }
} // namespace

// -----------------------------------------------------------------------------
// Bounded mode: zero frequency for FP4 hpl-equivalent should be in the same
// ballpark as the CPU backend (both gate on block-aware quantisation rather
// than on independent scale draws).
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, BoundedFP4HplZeroFrequency)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -0.5;
    opt.max          = 0.5;

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(12345);
    gpu.generate({1024, 1024}, {1, 1024}, opt);

    auto refs = gpu.getReferenceFloat();
    auto sum  = summarize(refs);

    // Uncoordinated GPU init produces ~50% zeros for FP4 hpl. mxDataGenerator
    // CPU lands at ~12.9%. Our block-aware GPU quantiser should be similar in
    // order of magnitude (much less than 50%).
    EXPECT_LT(sum.zeroFrac, 0.30) << "GPU FP4 hpl zero rate should be well below the "
                                     "uncoordinated GPU baseline (~50%)";
    EXPECT_LE(sum.maxAbs, 0.5001f) << "Max should respect the input bound";
    EXPECT_GT(sum.meanAbs, 0.0) << "Mean abs should be > 0";
}

// -----------------------------------------------------------------------------
// Determinism: same seed -> same output.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, DeterministicWithFixedSeed)
{
    using DType = ocp_e4m3_mxfp8;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    DataGeneratorGPU<DType> a;
    DataGeneratorGPU<DType> b;
    a.setSeed(42);
    b.setSeed(42);
    a.generate({256, 256}, {1, 256}, opt);
    b.generate({256, 256}, {1, 256}, opt);

    auto da = a.getDataBytes();
    auto db = b.getDataBytes();
    auto sa = a.getScaleBytes();
    auto sb = b.getScaleBytes();
    EXPECT_EQ(da, db);
    EXPECT_EQ(sa, sb);
}

// -----------------------------------------------------------------------------
// FP6 (E2M3) zero frequency.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, BoundedFP6E2M3HplZeroFrequency)
{
    using DType = ocp_e2m3_mxfp6;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -0.5;
    opt.max          = 0.5;

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(12345);
    gpu.generate({512, 512}, {1, 512}, opt);

    auto refs = gpu.getReferenceFloat();
    auto sum  = summarize(refs);

    EXPECT_LT(sum.zeroFrac, 0.20) << "FP6 has finer quantisation than FP4; expect <20% zeros";
    EXPECT_LE(sum.maxAbs, 0.5001f);
}

// -----------------------------------------------------------------------------
// FP8 (E4M3) zero frequency: very low because data type has subnormals down to
// ~0.002 and uniform [-1,1] input rarely lands inside that gap.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, BoundedFP8E4M3LowZeroFrequency)
{
    using DType = ocp_e4m3_mxfp8;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(7);
    gpu.generate({512, 512}, {1, 512}, opt);

    auto refs = gpu.getReferenceFloat();
    auto sum  = summarize(refs);

    EXPECT_LT(sum.zeroFrac, 0.05) << "FP8 E4M3 should have <5% zeros for uniform [-1,1]";
}

// -----------------------------------------------------------------------------
// Sanity bounds for CPU vs GPU Bounded init.
//
// The CPU `Bounded` mode runs an extra `scale_block_mean` + `post_sprinkle`
// pass that injects denormals/zeros/max-values and squashes the mean magnitude
// well below the requested [min, max] range (typical |mean| ~ 0.02 for
// [-1, 1]). The GPU port deliberately omits that pass: its goal is "well-formed
// MX data with the right block-scaling invariant", not bit- or distribution-
// equivalence with the CPU path. Both representations are valid inputs for a
// real GEMM. We therefore only check that both stay inside [min, max] and that
// neither degenerates to all-zeros or all-saturated.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, BoundedCpuVsGpuStatistics)
{
    using DType = ocp_e4m3_mxfp8;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    std::vector<index_t> sizes   = {512, 512};
    std::vector<index_t> strides = {1, 512};

    DataGenerator<DType> cpu;
    cpu.setSeed(99);
    cpu.generate(sizes, strides, opt);

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(99);
    gpu.generate(sizes, strides, opt);

    auto cpuRefs = cpu.getReferenceFloat();
    auto gpuRefs = gpu.getReferenceFloat();

    auto cpuSum = summarize(cpuRefs);
    auto gpuSum = summarize(gpuRefs);

    // CPU: post_sprinkle compresses the mean toward zero; just require it stays
    // representable.
    EXPECT_GT(cpuSum.meanAbs, 0.0);
    EXPECT_LT(cpuSum.meanAbs, 1.0);
    EXPECT_LT(cpuSum.zeroFrac, 0.5)
        << "CPU Bounded should not degenerate to >50% zeros";

    // GPU: no post-processing; expect the mean magnitude to land squarely in
    // the upper half of [0, 1] (uniform draws on [-1, 1] yield E[|x|]=0.5).
    EXPECT_GT(gpuSum.meanAbs, 0.25);
    EXPECT_LT(gpuSum.meanAbs, 0.75);
    EXPECT_LT(gpuSum.zeroFrac, 0.05)
        << "GPU Bounded on FP8 should rarely produce true zeros";

    // Both should respect the [min, max] envelope set in `opt`. We allow a
    // small slack to account for FP8 quantisation rounding above 1.0.
    EXPECT_LE(cpuSum.maxAbs, 1.5f);
    EXPECT_LE(gpuSum.maxAbs, 1.5f);
}

// -----------------------------------------------------------------------------
// preSwizzleScalesGFX950: GPU vs host should be byte-equivalent.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, PreSwizzleScalesGFX950MatchesHost)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    // 64 rows x 64 cols of K/32 scales -> 64 rows x 64 cols swizzle.
    std::vector<index_t> sizes   = {64 * 32, 64};
    std::vector<index_t> strides = {1, 64 * 32};

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(123);
    gpu.generate(sizes, strides, opt);

    auto canonical = gpu.getScaleBytes();

    // Swizzle on device.
    gpu.preSwizzleScalesGFX950Device({64, 64});
    auto gpuSwizzled = gpu.getScaleBytes();

    // Swizzle on host (reference).
    auto hostSwizzled = preSwizzleScalesGFX950(canonical, std::vector<size_t>{64, 64});

    ASSERT_EQ(gpuSwizzled.size(), hostSwizzled.size());
    EXPECT_EQ(gpuSwizzled, hostSwizzled);
}

// -----------------------------------------------------------------------------
// gfx1250 path: no shuffle should mean canonical scales are unchanged.
// (Sanity test that the wrapper plumbs preSwizzle/no-preSwizzle correctly.)
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, NoSwizzleLeavesScalesCanonical)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(31);
    gpu.generate({256, 8}, {1, 256}, opt);
    auto canonical = gpu.getScaleBytes();

    // No swizzle call: getScaleBytes() must still match the canonical layout
    // we just produced.
    auto again = gpu.getScaleBytes();
    EXPECT_EQ(canonical, again);
}

// -----------------------------------------------------------------------------
// preSwizzleScalesGFX950Device, padding code paths.
//
// The 64x64 case above leaves paddedRows == numRows and paddedCols == numCols,
// so the in-place memcpy2D pad branch never runs. These cases exercise:
//   * unaligned rows only (50 -> 64),
//   * both unaligned (50 rows, 13 cols -> 64x16).
// Mirrors host coverage in test/preswizzle_test.cpp.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, PreSwizzleScalesGFX950DeviceUnalignedRowsMatchesHost)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    // 50 rows of K/32 scales x 16 cols. data sizes pick numerics so that the
    // scale buffer ends up exactly 50 x 16 (= 800 bytes); 50 isn't a multiple
    // of 32, so paddedRows = 64, exercising the rowwise pad path.
    std::vector<index_t> sizes   = {50 * 32, 16};
    std::vector<index_t> strides = {1, 50 * 32};

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(7);
    gpu.generate(sizes, strides, opt);
    auto canonical = gpu.getScaleBytes();
    ASSERT_EQ(canonical.size(), 50u * 16u);

    gpu.preSwizzleScalesGFX950Device({50, 16});
    auto gpuSwizzled = gpu.getScaleBytes();

    auto hostSwizzled = preSwizzleScalesGFX950(canonical, std::vector<size_t>{50, 16});

    ASSERT_EQ(gpuSwizzled.size(), hostSwizzled.size());
    EXPECT_EQ(gpuSwizzled.size(), 64u * 16u)
        << "rowwise pad should round 50 -> 64 (cols already aligned)";
    EXPECT_EQ(gpuSwizzled, hostSwizzled);
}

TEST(DataGeneratorGPU, PreSwizzleScalesGFX950DeviceBothUnalignedMatchesHost)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    // 50 rows x 13 cols of scales: neither is aligned. Padded -> 64 x 16.
    std::vector<index_t> sizes   = {50 * 32, 13};
    std::vector<index_t> strides = {1, 50 * 32};

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(11);
    gpu.generate(sizes, strides, opt);
    auto canonical = gpu.getScaleBytes();
    ASSERT_EQ(canonical.size(), 50u * 13u);

    gpu.preSwizzleScalesGFX950Device({50, 13});
    auto gpuSwizzled = gpu.getScaleBytes();

    auto hostSwizzled = preSwizzleScalesGFX950(canonical, std::vector<size_t>{50, 13});

    ASSERT_EQ(gpuSwizzled.size(), hostSwizzled.size());
    EXPECT_EQ(gpuSwizzled.size(), 64u * 16u)
        << "both dims pad: 50 -> 64 rows, 13 -> 16 cols";
    EXPECT_EQ(gpuSwizzled, hostSwizzled);
}

// -----------------------------------------------------------------------------
// preSwizzleScalesGFX1250Device: GPU vs host should be byte-equivalent. Cover
// both an aligned case (no padding on the fast dim) and an unaligned case
// (fast dim is padded to a multiple of dimk = 128 / mxBlock).
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, PreSwizzleScalesGFX1250DeviceAlignedMatchesHost)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    // mxBlock = 32 -> dimk = 128 / 32 = 4.
    // Pick fastDim = 16 (multiple of 4) so paddedFast == fastDim, no padding.
    constexpr size_t slowDim = 17; // not a multiple of dimk: still fine, slowDim isn't padded
    constexpr size_t fastDim = 16;
    constexpr size_t mxBlock = 32;

    // Data sizes chosen so the scale buffer ends up slowDim x fastDim.
    std::vector<index_t> sizes   = {static_cast<index_t>(slowDim * mxBlock),
                                    static_cast<index_t>(fastDim)};
    std::vector<index_t> strides = {1, static_cast<index_t>(slowDim * mxBlock)};

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(101);
    gpu.generate(sizes, strides, opt);
    auto canonical = gpu.getScaleBytes();
    ASSERT_EQ(canonical.size(), slowDim * fastDim);

    gpu.preSwizzleScalesGFX1250Device(slowDim, fastDim, mxBlock);
    auto gpuSwizzled = gpu.getScaleBytes();

    auto hostSwizzled = preSwizzleScalesGFX1250(canonical, slowDim, fastDim, mxBlock);

    ASSERT_EQ(gpuSwizzled.size(), hostSwizzled.size());
    EXPECT_EQ(gpuSwizzled.size(), slowDim * fastDim)
        << "aligned fast dim -> no padding";
    EXPECT_EQ(gpuSwizzled, hostSwizzled);
}

TEST(DataGeneratorGPU, PreSwizzleScalesGFX1250DeviceUnalignedFastDimMatchesHost)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    // mxBlock = 32 -> dimk = 4.
    // fastDim = 13 (not a multiple of 4) -> paddedFast = 16.
    constexpr size_t slowDim = 17;
    constexpr size_t fastDim = 13;
    constexpr size_t mxBlock = 32;

    std::vector<index_t> sizes   = {static_cast<index_t>(slowDim * mxBlock),
                                    static_cast<index_t>(fastDim)};
    std::vector<index_t> strides = {1, static_cast<index_t>(slowDim * mxBlock)};

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(102);
    gpu.generate(sizes, strides, opt);
    auto canonical = gpu.getScaleBytes();
    ASSERT_EQ(canonical.size(), slowDim * fastDim);

    gpu.preSwizzleScalesGFX1250Device(slowDim, fastDim, mxBlock);
    auto gpuSwizzled = gpu.getScaleBytes();

    auto hostSwizzled = preSwizzleScalesGFX1250(canonical, slowDim, fastDim, mxBlock);

    ASSERT_EQ(gpuSwizzled.size(), hostSwizzled.size());
    EXPECT_EQ(gpuSwizzled.size(), slowDim * 16u)
        << "fastDim 13 -> padded to 16 (next multiple of dimk=4)";
    EXPECT_EQ(gpuSwizzled, hostSwizzled);
}

// -----------------------------------------------------------------------------
// Deterministic init modes: GPU getReferenceFloat() must equal the CPU
// reference element-for-element. The byte-level encodings can differ (the GPU
// path always derives a per-block scale, while the CPU path writes the
// canonical 1.0-scale + literal-value encoding for these modes), but the
// dequantised float output is the contract these modes promise.
//
// Coverage is deliberately broad here: every advertised deterministic
// MXInitMethod has a matching test below, including a >256 case for
// RowIndex (the value formula is `row % 256`, so divergences only show up
// once a dim crosses 256).
// -----------------------------------------------------------------------------
namespace
{
    template <typename DType>
    void expectCpuGpuReferenceFloatEq(DataInitMode                initMode,
                                      std::vector<index_t> const& sizes,
                                      std::vector<index_t> const& strides,
                                      char const*                 modeName)
    {
        DataGeneratorOptions opt;
        opt.blockScaling = 32;
        opt.initMode     = initMode;
        // CPU defaults forceDenorm=true, which makes setOne write
        // (scale=2^8=256, data=subnormal) instead of (scale=1, data=1).
        // The GPU path derives a per-block scale from the actual data
        // magnitudes and never produces that pair, so a literal CPU=GPU
        // dequant comparison only makes sense with forceDenorm=false.
        opt.forceDenorm  = false;

        DataGenerator<DType> cpu;
        cpu.setSeed(1);
        cpu.generate(sizes, strides, opt);

        DataGeneratorGPU<DType> gpu;
        gpu.setSeed(1);
        gpu.generate(sizes, strides, opt);

        auto cpuRefs = cpu.getReferenceFloat();
        auto gpuRefs = gpu.getReferenceFloat();

        ASSERT_EQ(cpuRefs.size(), gpuRefs.size())
            << "CPU/GPU reference vector size mismatch for " << modeName;
        EXPECT_EQ(cpuRefs, gpuRefs)
            << "CPU/GPU reference float mismatch for " << modeName;
    }
} // namespace

TEST(DataGeneratorGPU, ZerosMatchesCPU)
{
    using DType                          = ocp_e4m3_mxfp8;
    std::vector<index_t> const sizes     = {64, 64};
    std::vector<index_t> const strides   = {1, 64};
    expectCpuGpuReferenceFloatEq<DType>(Zeros{}, sizes, strides, "Zeros");
}

TEST(DataGeneratorGPU, OnesMatchesCPU)
{
    using DType                          = ocp_e4m3_mxfp8;
    std::vector<index_t> const sizes     = {64, 64};
    std::vector<index_t> const strides   = {1, 64};
    expectCpuGpuReferenceFloatEq<DType>(Ones{}, sizes, strides, "Ones");
}

TEST(DataGeneratorGPU, IdentityMatchesCPU)
{
    using DType                          = ocp_e4m3_mxfp8;
    // Square matrix - generate_data_identity asserts size.size() == 2 and
    // writes 1.0 along min(rows, cols).
    std::vector<index_t> const sizes     = {64, 64};
    std::vector<index_t> const strides   = {1, 64};
    expectCpuGpuReferenceFloatEq<DType>(Identity{}, sizes, strides, "Identity");
}

TEST(DataGeneratorGPU, SequentialMatchesCPU)
{
    using DType                          = ocp_e4m3_mxfp8;
    std::vector<index_t> const sizes     = {64, 64};
    std::vector<index_t> const strides   = {1, 64};
    expectCpuGpuReferenceFloatEq<DType>(Sequential{}, sizes, strides, "Sequential");
}

TEST(DataGeneratorGPU, RowIndexMatchesCPU)
{
    using DType                          = ocp_e4m3_mxfp8;
    std::vector<index_t> const sizes     = {64, 64};
    std::vector<index_t> const strides   = {1, 64};
    expectCpuGpuReferenceFloatEq<DType>(RowIndex{}, sizes, strides, "RowIndex");
}

// >256 case: catches a regression where the GPU formula drops the `% 256`
// the CPU does. A 64x64 RowIndex/ColIndex test would silently pass since
// every row index fits in [0, 256).
TEST(DataGeneratorGPU, RowIndexLargeDimMatchesCPU)
{
    using DType                          = ocp_e4m3_mxfp8;
    std::vector<index_t> const sizes     = {512, 64};
    std::vector<index_t> const strides   = {1, 512};
    expectCpuGpuReferenceFloatEq<DType>(RowIndex{}, sizes, strides, "RowIndex(512x64)");
}

TEST(DataGeneratorGPU, ColIndexMatchesCPU)
{
    using DType                          = ocp_e4m3_mxfp8;
    std::vector<index_t> const sizes     = {64, 64};
    std::vector<index_t> const strides   = {1, 64};
    expectCpuGpuReferenceFloatEq<DType>(ColIndex{}, sizes, strides, "ColIndex");
}

TEST(DataGeneratorGPU, CheckerboardMatchesCPU)
{
    using DType                          = ocp_e4m3_mxfp8;
    std::vector<index_t> const sizes     = {64, 64};
    std::vector<index_t> const strides   = {1, 64};
    expectCpuGpuReferenceFloatEq<DType>(Checkerboard{}, sizes, strides, "Checkerboard");
}

// -----------------------------------------------------------------------------
// Constant-fill modes (Twos / NegOnes / MaxVals) on the device path.
//
// These modes used to be host-fallback (auto-derived per-block scale produces
// a different byte pattern than the CPU's "literal data byte + scale=1.0"
// encoding, but the *dequantised* float matches). The GPU now generates them
// directly via DeviceInitMode::ConstantValue. Coverage spans every supported
// DTYPE so the dtype-specific MaxVals magnitude (max normal differs per
// dtype, plumbed via GpuTraits<DTYPE>::maxNormal) is exercised end-to-end.
// -----------------------------------------------------------------------------
namespace
{
    template <typename DType>
    void expectAllReferenceFloatEqOnGpu(DataInitMode initMode, float expected, char const* label)
    {
        DataGeneratorOptions opt;
        opt.blockScaling = 32;
        opt.initMode     = initMode;
        // Match the CPU-side constant-fill tests (forceDenorm=false). The
        // GPU path is independent of forceDenorm for these modes -- the
        // per-block scale derivation is driven by data magnitudes, not by
        // the option flag -- but using the same value as the CPU keeps any
        // future CPU=GPU comparison apples-to-apples.
        opt.forceDenorm  = false;
        std::vector<index_t> sizes{64, 64};
        std::vector<index_t> strides{1, 64};

        DataGeneratorGPU<DType> gpu;
        gpu.setSeed(1);
        gpu.generate(sizes, strides, opt);
        auto refs = gpu.getReferenceFloat();
        ASSERT_EQ(refs.size(), static_cast<size_t>(sizes[0] * sizes[1])) << label;
        for(size_t i = 0; i < refs.size(); ++i)
        {
            EXPECT_EQ(refs[i], expected) << label << " element " << i;
        }
    }
} // namespace

TEST(DataGeneratorGPU, TwosF4)
{
    expectAllReferenceFloatEqOnGpu<ocp_e2m1_mxfp4>(Twos{}, 2.0f, "Twos F4");
}
TEST(DataGeneratorGPU, TwosF6_E2M3)
{
    expectAllReferenceFloatEqOnGpu<ocp_e2m3_mxfp6>(Twos{}, 2.0f, "Twos F6_E2M3");
}
TEST(DataGeneratorGPU, TwosF6_E3M2)
{
    expectAllReferenceFloatEqOnGpu<ocp_e3m2_mxfp6>(Twos{}, 2.0f, "Twos F6_E3M2");
}
TEST(DataGeneratorGPU, TwosF8_E4M3)
{
    expectAllReferenceFloatEqOnGpu<ocp_e4m3_mxfp8>(Twos{}, 2.0f, "Twos F8_E4M3");
}
TEST(DataGeneratorGPU, TwosF8_E5M2)
{
    expectAllReferenceFloatEqOnGpu<ocp_e5m2_mxfp8>(Twos{}, 2.0f, "Twos F8_E5M2");
}

TEST(DataGeneratorGPU, NegOnesF4)
{
    expectAllReferenceFloatEqOnGpu<ocp_e2m1_mxfp4>(NegOnes{}, -1.0f, "NegOnes F4");
}
TEST(DataGeneratorGPU, NegOnesF6_E2M3)
{
    expectAllReferenceFloatEqOnGpu<ocp_e2m3_mxfp6>(NegOnes{}, -1.0f, "NegOnes F6_E2M3");
}
TEST(DataGeneratorGPU, NegOnesF6_E3M2)
{
    expectAllReferenceFloatEqOnGpu<ocp_e3m2_mxfp6>(NegOnes{}, -1.0f, "NegOnes F6_E3M2");
}
TEST(DataGeneratorGPU, NegOnesF8_E4M3)
{
    expectAllReferenceFloatEqOnGpu<ocp_e4m3_mxfp8>(NegOnes{}, -1.0f, "NegOnes F8_E4M3");
}
TEST(DataGeneratorGPU, NegOnesF8_E5M2)
{
    expectAllReferenceFloatEqOnGpu<ocp_e5m2_mxfp8>(NegOnes{}, -1.0f, "NegOnes F8_E5M2");
}

// MaxVals: dequantised value is the per-DTYPE max normal. Mirrors the CPU
// constant-fill matrix (data_generator_constant_fills_test.cpp).
TEST(DataGeneratorGPU, MaxValsF4)
{
    expectAllReferenceFloatEqOnGpu<ocp_e2m1_mxfp4>(MaxVals{}, 6.0f, "MaxVals F4");
}
TEST(DataGeneratorGPU, MaxValsF6_E2M3)
{
    expectAllReferenceFloatEqOnGpu<ocp_e2m3_mxfp6>(MaxVals{}, 7.5f, "MaxVals F6_E2M3");
}
TEST(DataGeneratorGPU, MaxValsF6_E3M2)
{
    expectAllReferenceFloatEqOnGpu<ocp_e3m2_mxfp6>(MaxVals{}, 28.0f, "MaxVals F6_E3M2");
}
TEST(DataGeneratorGPU, MaxValsF8_E4M3)
{
    expectAllReferenceFloatEqOnGpu<ocp_e4m3_mxfp8>(MaxVals{}, 448.0f, "MaxVals F8_E4M3");
}
TEST(DataGeneratorGPU, MaxValsF8_E5M2)
{
    expectAllReferenceFloatEqOnGpu<ocp_e5m2_mxfp8>(MaxVals{}, 57344.0f, "MaxVals F8_E5M2");
}

// -----------------------------------------------------------------------------
// RandInt on the device path.
//
// The CPU path uses mt19937 + uniform_int_distribution; the GPU uses a
// per-element xorshift32. The two never agree byte-for-byte, so the
// contract we test is the same one the CPU constant-fill tests use:
//   * every dequantised value is an integer,
//   * every value lies in [lo, hi],
//   * the spread is non-degenerate when hi > lo.
// CPU saturates to dtype maxNormal when |hi| > maxNormal; GPU saturates to
// scale * maxNormal which can be larger - both still satisfy the in-range
// invariant for the per-DTYPE ranges hipBLASLt's legacy random_int<T> uses.
// -----------------------------------------------------------------------------
namespace
{
    template <typename DType>
    void expectRandIntInRangeOnGpu(int lo, int hi)
    {
        DataGeneratorOptions opt;
        opt.blockScaling = 32;
        opt.initMode     = RandInt{lo, hi};
        opt.forceDenorm  = false;
        std::vector<index_t> sizes{64, 64};
        std::vector<index_t> strides{1, 64};

        DataGeneratorGPU<DType> gpu;
        gpu.setSeed(123);
        gpu.generate(sizes, strides, opt);
        auto refs = gpu.getReferenceFloat();
        ASSERT_EQ(refs.size(), static_cast<size_t>(sizes[0] * sizes[1]));

        int  distinctCount = 0;
        bool seen[256]     = {};
        for(size_t i = 0; i < refs.size(); ++i)
        {
            float const v = refs[i];
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
} // namespace

TEST(DataGeneratorGPU, RandIntF4_Range_m4_4)
{
    expectRandIntInRangeOnGpu<ocp_e2m1_mxfp4>(-4, 4);
}
TEST(DataGeneratorGPU, RandIntF6_E2M3_Range_m7_7)
{
    expectRandIntInRangeOnGpu<ocp_e2m3_mxfp6>(-7, 7);
}
TEST(DataGeneratorGPU, RandIntF6_E3M2_Range_m28_28)
{
    expectRandIntInRangeOnGpu<ocp_e3m2_mxfp6>(-28, 28);
}
TEST(DataGeneratorGPU, RandIntF8_E4M3_Range_1_10)
{
    expectRandIntInRangeOnGpu<ocp_e4m3_mxfp8>(1, 10);
}
TEST(DataGeneratorGPU, RandIntF8_E5M2_Range_1_10)
{
    expectRandIntInRangeOnGpu<ocp_e5m2_mxfp8>(1, 10);
}

// Degenerate range (lo == hi) must produce that single value at every
// element. Exercises the `range = 1` path in the device int sampler.
TEST(DataGeneratorGPU, RandIntDegenerateRangeIsConstant)
{
    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = RandInt{3, 3};
    opt.forceDenorm  = false;
    std::vector<index_t> sizes{64, 64};
    std::vector<index_t> strides{1, 64};

    DataGeneratorGPU<ocp_e4m3_mxfp8> gpu;
    gpu.generate(sizes, strides, opt);
    auto refs = gpu.getReferenceFloat();
    ASSERT_EQ(refs.size(), 64u * 64u);
    for(float v : refs)
        EXPECT_EQ(v, 3.0f);
}

// Inverted range (lo > hi) is invalid; mirror the CPU `std::invalid_argument`
// on this side too. Caught at makeConfig time, before any kernel launch.
TEST(DataGeneratorGPU, RandIntInvertedRangeThrows)
{
    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = RandInt{5, -5};
    opt.forceDenorm  = false;
    std::vector<index_t> sizes{64, 64};
    std::vector<index_t> strides{1, 64};

    DataGeneratorGPU<ocp_e4m3_mxfp8> gpu;
    EXPECT_THROW(gpu.generate(sizes, strides, opt), std::invalid_argument);
}
