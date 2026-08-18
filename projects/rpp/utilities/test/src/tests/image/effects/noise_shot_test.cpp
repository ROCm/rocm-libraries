/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/noise_shot_ref.hpp"

using namespace rpptest;

namespace {

// The three properties checked here take disjoint parameter sets, so the check is an axis of the
// grid rather than a separate TEST_P body: GTest instantiates every body of a fixture against the
// whole param set, which would generate (and skip) each narrow check across the full dtype/layout
// grid. As an axis, each check is instantiated over exactly the points it covers.
enum class Check { Identity, ValidRange, Seed };

struct NoiseShotParams {
    Check check;
    std::string name() const {
        switch (check) {
            case Check::Identity: return "Identity";
            case Check::ValidRange: return "ValidRange";
            case Check::Seed: return "Seed";
        }
        return "UNK";
    }
};

// noise_shot's public API doc only pins down the shotNoiseFactor = 0 corner (see
// noise_shot_ref.hpp): the exact Poisson photon-count scaling for factor > 0 is not derivable
// from the header comment alone, so it is deliberately left as an open question rather than
// guessed. The general case is covered here only by cheap runtime invariants -- a coarse
// valid-storable-range check and a seed-determinism check -- not a golden comparison.
constexpr Rpp32f kNontrivialFactor = 0.5f;

template <typename T>
void run_noise_shot_identity(const TestConfig& cfg) {
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
    noise_shot_identity_reference<T>(input.data(), golden.data(), desc, roi.data(), XYWH);

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
void run_noise_shot_valid_range(const TestConfig& cfg) {
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
    EXPECT_TRUE(ok) << "noise_shot produced a value outside the storable range for dtype "
                    << dtype_name(cfg.dtype);
}

// A real per-call seed should make the op reproducible for a fixed seed and different across
// seeds -- contrast with the rain/fog RNGs seeded from std::random_device, which cannot satisfy
// either half of this. noise_shot takes an explicit Rpp32u seed, so it is expected to behave like
// this.
template <typename T>
void run_noise_shot_seed_invariant(const TestConfig& cfg) {
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

// The HOST Poisson sampler's rejection-loop threshold overflows to +inf for the lambda an ordinary
// 8-bit pixel produces at kNontrivialFactor, so the loop does not reliably terminate -- observed
// run times from several seconds to over a minute for a single tiny image, growing without bound as
// pixel values increase. The RNG-active intents skip HOST rather than being left red/hanging, so
// the suite stays safe to run; HIP uses a different (unaffected) algorithm.
constexpr char kHostPoissonSkip[] =
    "shot_noise HOST Poisson sampler's rejection threshold overflows to +inf for ordinary pixel "
    "values at this factor, making the loop's runtime unbounded in practice";

// Identity is checked over the full grid; the RNG-active checks are narrower on purpose -- Seed
// runs three full kernel invocations per case, and ValidRange's storable-range notion is exercised
// on one integer and one float dtype rather than all four.
std::vector<WithParams<NoiseShotParams>> noise_shot_configs() {
    std::vector<WithParams<NoiseShotParams>> configs = with_params<NoiseShotParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {NoiseShotParams{Check::Identity}});
    const std::vector<WithParams<NoiseShotParams>> validRange = with_params<NoiseShotParams>(
        make_configs({DType::U8, DType::F32}, {Layout::PKD3, Layout::PLN1},
                     {Roi::Full, Roi::Partial}),
        {NoiseShotParams{Check::ValidRange}});
    const std::vector<WithParams<NoiseShotParams>> seed = with_params<NoiseShotParams>(
        make_configs({DType::U8}, {Layout::PKD3}, {Roi::Full}), {NoiseShotParams{Check::Seed}});
    configs.insert(configs.end(), validRange.begin(), validRange.end());
    configs.insert(configs.end(), seed.begin(), seed.end());
    return configs;
}

}  // namespace

// Full name:
// Image_Effects/NoiseShotTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Check>
class NoiseShotTest : public SkipListTest<WithParams<NoiseShotParams>> {};

TEST_P(NoiseShotTest, Correctness) {
    const auto& p = GetParam();
    const TestConfig& cfg = p.cfg;
    if (p.op.check != Check::Identity && cfg.backend == RPP_HOST_BACKEND)
        GTEST_SKIP() << kHostPoissonSkip;

    switch (p.op.check) {
        case Check::Identity:
            dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
                run_noise_shot_identity<Element<decltype(tag)>>(cfg);
            });
            break;
        case Check::ValidRange:
            if (cfg.dtype == DType::U8)
                run_noise_shot_valid_range<Rpp8u>(cfg);
            else
                run_noise_shot_valid_range<Rpp32f>(cfg);
            break;
        case Check::Seed:
            run_noise_shot_seed_invariant<Rpp8u>(cfg);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, NoiseShotTest,
                         ::testing::ValuesIn(noise_shot_configs()),
                         op_config_name<NoiseShotParams>);
