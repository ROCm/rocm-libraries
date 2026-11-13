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

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/intensity.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/jitter_ref.hpp"

using namespace rpptest;

namespace {

// The checks take disjoint parameter sets, so the check is a grid axis rather than separate
// TEST_P bodies.
enum class Check { Identity, Window, Seed };

// kernelSize is an odd window size; kernelSize=1 is the RNG-free identity corner.
struct JitterParams {
    Rpp32u kernelSize;
    Check check;
    std::string name() const {
        std::string token = "k" + std::to_string(kernelSize);
        switch (check) {
            case Check::Identity:
                return token + "_Identity";
            case Check::Window:
                return token + "_Window";
            case Check::Seed:
                return token + "_Seed";
        }
        return token;
    }
};

template <typename T>
void run_jitter_identity(const TestConfig& cfg, const JitterParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32u> kernelSize(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        kernelSize[i] = op.kernelSize;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    jitter_identity_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, roi.data(), XYWH);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_jitter(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, kernelSize.data(), /*seed=*/42u,
                          roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // kernelSize=1 collapses the offset window to {0,0}, so the op is a bit-exact identity.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, 0.0));
}

// kernelSize > 1 is RNG-driven: check every output pixel equals the source at some ROI-clamped
// offset within [-r,r]^2, with the same offset for all channels.

template <typename T>
void run_jitter_window(const TestConfig& cfg, const JitterParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32u> kernelSize(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        kernelSize[i] = op.kernelSize;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_jitter(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, kernelSize.data(),
                          /*seed=*/123u, roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);
    handle.sync();
    dst.read(actual.data(), bytes);

    const int r = static_cast<int>(op.kernelSize / 2);
    ASSERT_GT(r, 0) << "ReachableWithinWindow requires kernelSize > 1";

    for (Rpp32u n = 0; n < srcDesc.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], XYWH);
        const int xlo = static_cast<int>(b.x0), xhi = static_cast<int>(b.x0 + b.w - 1);
        const int ylo = static_cast<int>(b.y0), yhi = static_cast<int>(b.y0 + b.h - 1);
        for (Rpp32u j = 0; j < b.h; ++j) {
            for (Rpp32u i = 0; i < b.w; ++i) {
                const int sx = static_cast<int>(b.x0 + i);
                const int sy = static_cast<int>(b.y0 + j);
                const std::size_t dstPix = plane_index(dstDesc, plane_base(dstDesc, n, 0), j, i);

                bool found = false;
                for (int dy = -r; dy <= r && !found; ++dy) {
                    for (int dx = -r; dx <= r && !found; ++dx) {
                        const int csy = clamp_coord(sy + dy, ylo, yhi);
                        const int csx = clamp_coord(sx + dx, xlo, xhi);
                        bool allMatch = true;
                        for (Rpp32u c = 0; c < srcDesc.c; ++c) {
                            const std::size_t srcIdx = plane_index(
                                srcDesc, plane_base(srcDesc, n, c), static_cast<std::size_t>(csy),
                                static_cast<std::size_t>(csx));
                            const std::size_t dstIdx = channel_index(dstDesc, dstPix, c);
                            if (to_double(actual[dstIdx]) != to_double(input[srcIdx])) {
                                allMatch = false;
                                break;
                            }
                        }
                        if (allMatch) found = true;
                    }
                }
                if (!found) {
                    ADD_FAILURE() << "n=" << n << " row=" << j << " col=" << i
                                  << ": output pixel is not any clamped candidate within the ["
                                  << -r << "," << r << "] window (kernelSize=" << op.kernelSize
                                  << ")";
                    return;
                }
            }
        }
    }
}

// Same seed must give bit-identical output; a different seed must differ somewhere in the ROI.

template <typename T>
void run_jitter_seed_invariant(const TestConfig& cfg, const JitterParams& op) {
    RpptDesc desc = make_src_descriptor(cfg);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32u> kernelSize(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        kernelSize[i] = op.kernelSize;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    // Returns the status: an ASSERT_* inside a lambda would only return from the lambda.
    auto run = [&](Rpp32u seed, std::vector<T>& out) -> RppStatus {
        DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
        src.write(input.data(), bytes);
        dst.write(input.data(), bytes);
        RppHandle handle(cfg.backend, cfg.size.n);
        const RppStatus status = rppt_jitter(src.ptr(), &desc, dst.ptr(), &desc, kernelSize.data(),
                                             seed, roi.data(), XYWH, handle.get(), cfg.backend);
        if (status != RPP_SUCCESS) return status;
        handle.sync();
        out.resize(count);
        dst.read(out.data(), bytes);
        return status;
    };

    std::vector<T> outA1, outA2, outB;
    ASSERT_EQ(run(42u, outA1), RPP_SUCCESS);
    ASSERT_EQ(run(42u, outA2), RPP_SUCCESS);
    ASSERT_EQ(run(1337u, outB), RPP_SUCCESS);

    EXPECT_TRUE(compare_roi<T>(outA1.data(), outA2.data(), desc, roi.data(), XYWH, 0.0));

    bool anyDiffer = false;
    for_each_roi_io(desc, roi.data(), XYWH,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t, std::size_t dstIdx) {
                        if (outA1[dstIdx] != outB[dstIdx]) anyDiffer = true;
                    });
    EXPECT_TRUE(anyDiffer) << "different seeds produced identical output over the whole ROI";
}

// The RNG-active checks run on a narrower grid: Seed runs three kernel calls per case and Window is
// a per-pixel candidate search.
std::vector<WithParams<JitterParams>> jitter_configs() {
    // Same-layout cases plus both directions of the fused output-layout conversion.
    std::vector<WithParams<JitterParams>> configs = with_params<JitterParams>(
        concat_configs({
            make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFull,
                         {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
        {JitterParams{1, Check::Identity}});
    const std::vector<WithParams<JitterParams>> window = with_params<JitterParams>(
        make_configs({DType::U8, DType::F32}, {Layout::PKD3, Layout::PLN1},
                     {Roi::Full, Roi::Partial}),
        {JitterParams{3, Check::Window}, JitterParams{5, Check::Window}});
    const std::vector<WithParams<JitterParams>> seed = with_params<JitterParams>(
        make_configs({DType::U8}, {Layout::PKD3}, {Roi::Full}), {JitterParams{5, Check::Seed}});
    configs.insert(configs.end(), window.begin(), window.end());
    configs.insert(configs.end(), seed.begin(), seed.end());
    return configs;
}

}  // namespace

// Full name:
// Image_Effects/JitterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_k<N>_<Check>
class JitterTest : public SkipListTest<WithParams<JitterParams>> {};

TEST_P(JitterTest, Correctness) {
    const auto& p = GetParam();
    switch (p.op.check) {
        case Check::Identity:
            dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
                p.cfg.dtype,
                [&](auto tag) { run_jitter_identity<Element<decltype(tag)>>(p.cfg, p.op); });
            break;
        case Check::Window:
            if (p.cfg.dtype == DType::U8)
                run_jitter_window<Rpp8u>(p.cfg, p.op);
            else
                run_jitter_window<Rpp32f>(p.cfg, p.op);
            break;
        case Check::Seed:
            run_jitter_seed_invariant<Rpp8u>(p.cfg, p.op);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, JitterTest, ::testing::ValuesIn(jitter_configs()),
                         op_config_name<JitterParams>);
