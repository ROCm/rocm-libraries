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
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/intensity.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/noise_shot_ref.hpp"

using namespace rpptest;

namespace {

// The checks take disjoint parameter sets, so the check is a grid axis rather than separate
// TEST_P bodies.
enum class Check { Identity, ValidRange, Seed };

struct NoiseShotParams {
    Check check;
    std::string name() const {
        switch (check) {
            case Check::Identity:
                return "Identity";
            case Check::ValidRange:
                return "ValidRange";
            case Check::Seed:
                return "Seed";
        }
        return "UNK";
    }
};

// The API only pins down the shotNoiseFactor = 0 identity; factor > 0 is covered by storable-range
// and seed-determinism invariants rather than a golden.
constexpr Rpp32f kNontrivialFactor = 0.5f;

template <typename T>
void run_noise_shot_identity(const TestConfig& cfg) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32f> factor(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        factor[i] = 0.0f;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    noise_shot_identity_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, roi.data(),
                                     XYWH);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_shot_noise(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, factor.data(),
                              /*seed=*/42u, roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, 0.0));
}

// Asserts every ROI output element is storable for its dtype; does not check the distribution.
template <typename T>
void run_noise_shot_valid_range(const TestConfig& cfg) {
    RpptDesc desc = make_src_descriptor(cfg);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> factor(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        factor[i] = kNontrivialFactor;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
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

// Same seed must give bit-identical output; a different seed must differ.
template <typename T>
void run_noise_shot_seed_invariant(const TestConfig& cfg) {
    RpptDesc desc = make_src_descriptor(cfg);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> factor(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        factor[i] = kNontrivialFactor;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    // Returns the status: an ASSERT_* inside a lambda would only return from the lambda.
    auto run_once = [&](Rpp32u seed, std::vector<T>& out) -> RppStatus {
        DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
        src.write(input.data(), bytes);
        dst.write(input.data(), bytes);
        RppHandle handle(cfg.backend, cfg.size.n);
        const RppStatus status = rppt_shot_noise(src.ptr(), &desc, dst.ptr(), &desc, factor.data(),
                                                 seed, roi.data(), XYWH, handle.get(), cfg.backend);
        if (status != RPP_SUCCESS) return status;
        handle.sync();
        out.resize(count);
        dst.read(out.data(), bytes);
        return status;
    };

    std::vector<T> seed42a, seed42b, seed1337;
    ASSERT_EQ(run_once(42u, seed42a), RPP_SUCCESS);
    ASSERT_EQ(run_once(42u, seed42b), RPP_SUCCESS);
    ASSERT_EQ(run_once(1337u, seed1337), RPP_SUCCESS);

    EXPECT_TRUE(compare_roi<T>(seed42b.data(), seed42a.data(), desc, roi.data(), XYWH, 0.0))
        << "same seed produced different output";
    EXPECT_FALSE(compare_roi<T>(seed1337.data(), seed42a.data(), desc, roi.data(), XYWH, 0.0))
        << "different seeds produced bit-identical output";
}

// The HOST Poisson sampler's rejection threshold overflows to +inf at kNontrivialFactor, so its
// loop may not terminate. The RNG-active checks skip HOST; HIP uses a different algorithm.
constexpr char kHostPoissonSkip[] =
    "shot_noise HOST Poisson sampler's rejection threshold overflows to +inf for ordinary pixel "
    "values at this factor, making the loop's runtime unbounded in practice";

// The RNG-active checks run on a narrower grid: Seed runs three kernel calls per case and
// ValidRange covers one integer and one float dtype.
std::vector<WithParams<NoiseShotParams>> noise_shot_configs() {
    // Same-layout cases plus both directions of the fused output-layout conversion.
    std::vector<WithParams<NoiseShotParams>> configs = with_params<NoiseShotParams>(
        concat_configs({
            make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFull,
                         {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
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
            dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
                cfg.dtype, [&](auto tag) { run_noise_shot_identity<Element<decltype(tag)>>(cfg); });
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

INSTANTIATE_TEST_SUITE_P(Image_Effects, NoiseShotTest, ::testing::ValuesIn(noise_shot_configs()),
                         op_config_name<NoiseShotParams>);
