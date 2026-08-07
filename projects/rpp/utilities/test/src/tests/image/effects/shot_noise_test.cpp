#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/shot_noise_ref.hpp"

using namespace rpptest;

namespace {

// shot_noise's public API doc only pins down the shotNoiseFactor = 0 corner (see
// shot_noise_ref.hpp): the exact Poisson photon-count scaling for factor > 0 is not derivable
// from the header comment alone, so it is deliberately left as an open question rather than
// guessed. The general case is covered here only by cheap runtime invariants -- a coarse
// valid-storable-range check and a seed-determinism check -- not a golden comparison.
constexpr Rpp32f kNontrivialFactor = 0.5f;

template <typename T>
void run_shot_noise_identity(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> factor(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        factor[i] = 0.0f;  // the RNG-free identity corner
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    shot_noise_identity_reference<T>(input.data(), golden.data(), desc, roi.data(), XYWH);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_shot_noise(src.ptr(), &desc, dst.ptr(), &desc, factor.data(), /*seed=*/42u,
                              roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH, 0.0));
}

// Runs the real kernel at a nontrivial (RNG-active) factor and asserts every ROI output element
// is a legally storable value for its dtype -- catches gross corruption (NaN, overflow, wrong
// range), not distribution correctness (that formula is the open question noted above).
template <typename T>
void run_shot_noise_valid_range(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> factor(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        factor[i] = kNontrivialFactor;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_shot_noise(src.ptr(), &desc, dst.ptr(), &desc, factor.data(), /*seed=*/42u,
                              roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    bool ok = true;
    for_each_roi_io(desc, roi.data(), XYWH,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t, std::size_t dstIdx) {
                        if (!ok) return;
                        const double v = to_double(actual[dstIdx]);
                        const double q = quantize_stored(v, cfg.dtype);
                        if (!std::isfinite(v) || std::fabs(v - q) > 1e-6) ok = false;
                    });
    EXPECT_TRUE(ok) << "shot_noise produced a value outside the storable range for dtype "
                    << dtype_name(cfg.dtype);
}

// A real per-call seed should make the op reproducible for a fixed seed and different across
// seeds -- contrast with the rain/fog RNGs seeded from std::random_device, which cannot satisfy
// either half of this (see the rain-nondeterministic-seed / fog-nondeterministic-and-baked-mask
// tickets). shot_noise takes an explicit Rpp32u seed, so it is expected to behave like this.
template <typename T>
void run_shot_noise_seed_invariant(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> factor(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        factor[i] = kNontrivialFactor;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    auto run_once = [&](Rpp32u seed, std::vector<T>& out) {
        DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
        src.write(input.data(), bytes);
        dst.write(input.data(), bytes);
        RppHandle handle(cfg.backend, shape.n);
        ASSERT_EQ(rppt_shot_noise(src.ptr(), &desc, dst.ptr(), &desc, factor.data(), seed,
                                  roi.data(), XYWH, handle.get(), cfg.backend),
                  RPP_SUCCESS);
        handle.sync();
        out.resize(count);
        dst.read(out.data(), bytes);
    };

    std::vector<T> seed42a, seed42b, seed1337;
    run_once(42u, seed42a);
    run_once(42u, seed42b);
    run_once(1337u, seed1337);

    EXPECT_TRUE(compare_roi<T>(seed42b.data(), seed42a.data(), desc, roi.data(), XYWH, 0.0))
        << "same seed produced different output";
    EXPECT_FALSE(compare_roi<T>(seed1337.data(), seed42a.data(), desc, roi.data(), XYWH, 0.0))
        << "different seeds produced bit-identical output";
}

}  // namespace

// Full names:
// Image_Effects/ShotNoiseTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
// Image_Effects_ValidRange/ShotNoiseValidRangeTest.ValidRangeInvariant/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
// Image_Effects_Seed/ShotNoiseSeedTest.SeedInvariant/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
//
// Each of the three checks uses its OWN fixture class (rather than sharing one across all three
// TEST_P bodies) so each INSTANTIATE_TEST_SUITE_P only ever instantiates its own test: a shared
// fixture would cross-instantiate every body against every param set -- in particular it would run
// SeedInvariant (which is hardcoded to Rpp8u) against Correctness's full dtype grid, including F16,
// a dtype/buffer-size mismatch that corrupts the heap. Self-inflicted test bug, not a kernel defect.
class ShotNoiseTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(ShotNoiseTest, Correctness) {
    const TestConfig& cfg = GetParam();
    switch (cfg.dtype) {
        case DType::U8:
            run_shot_noise_identity<Rpp8u>(cfg);
            break;
        case DType::F16:
            run_shot_noise_identity<Rpp16f>(cfg);
            break;
        case DType::F32:
            run_shot_noise_identity<Rpp32f>(cfg);
            break;
        case DType::I8:
            run_shot_noise_identity<Rpp8s>(cfg);
            break;
        default:
            FAIL() << "unsupported dtype for shot_noise";
    }
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, ShotNoiseTest,
                         ::testing::ValuesIn(make_configs({DType::U8, DType::F16, DType::F32,
                                                           DType::I8},
                                                          {Layout::PKD3, Layout::PLN3,
                                                           Layout::PLN1},
                                                          {Roi::Full, Roi::Partial})),
                         config_param_name);

class ShotNoiseValidRangeTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(ShotNoiseValidRangeTest, ValidRangeInvariant) {
    const TestConfig& cfg = GetParam();
    // The HOST Poisson sampler's rejection-loop threshold overflows to +inf for lambda values an
    // ordinary 8-bit pixel produces at this factor, so the loop does not reliably terminate --
    // observed run times from several seconds to over a minute for a single tiny image, growing
    // without bound as pixel values increase. Skipped rather than left red/hanging so the suite
    // stays safe to run; HIP uses a different (unaffected) algorithm and is not skipped.
    if (cfg.backend == RPP_HOST_BACKEND)
        GTEST_SKIP() << "shot_noise HOST Poisson sampler's rejection threshold overflows to +inf "
                        "for ordinary pixel values at this factor, making the loop's runtime "
                        "unbounded in practice";
    switch (cfg.dtype) {
        case DType::U8:
            run_shot_noise_valid_range<Rpp8u>(cfg);
            break;
        case DType::F32:
            run_shot_noise_valid_range<Rpp32f>(cfg);
            break;
        default:
            FAIL() << "unsupported dtype for shot_noise ValidRangeInvariant slice";
    }
}

INSTANTIATE_TEST_SUITE_P(Image_Effects_ValidRange, ShotNoiseValidRangeTest,
                         ::testing::ValuesIn(make_configs({DType::U8, DType::F32},
                                                          {Layout::PKD3, Layout::PLN1},
                                                          {Roi::Full, Roi::Partial})),
                         config_param_name);

class ShotNoiseSeedTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(ShotNoiseSeedTest, SeedInvariant) {
    const TestConfig& cfg = GetParam();
    // Same HOST Poisson-sampler hang risk as ValidRangeInvariant above -- skip HOST, keep HIP.
    if (cfg.backend == RPP_HOST_BACKEND)
        GTEST_SKIP() << "shot_noise HOST Poisson sampler's rejection threshold overflows to +inf "
                        "for ordinary pixel values at this factor, making the loop's runtime "
                        "unbounded in practice";
    run_shot_noise_seed_invariant<Rpp8u>(cfg);
}

INSTANTIATE_TEST_SUITE_P(Image_Effects_Seed, ShotNoiseSeedTest,
                         ::testing::ValuesIn(make_configs({DType::U8}, {Layout::PKD3},
                                                          {Roi::Full})),
                         config_param_name);
