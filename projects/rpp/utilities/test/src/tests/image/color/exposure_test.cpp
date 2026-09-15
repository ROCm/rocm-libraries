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
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/exposure_ref.hpp"

using namespace rpptest;

namespace {

// exposureFactor 0.5 => multiplier 2^0.5 ~= 1.414.
struct ExposureParams {
    float exposureFactor;
    std::string name() const {
        return "e" + num_token(exposureFactor);
    }
};

template <typename T>
void run_exposure(const TestConfig& cfg, const ExposureParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32f> exposureFactor(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        exposureFactor[i] = op.exposureFactor;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    exposure_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(),
                          XYWH, op.exposureFactor);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_exposure(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, exposureFactor.data(),
                            roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH,
                               kRoundingTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Color/ExposureTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Exposure>
class ExposureTest : public SkipListTest<WithParams<ExposureParams>> {};

TEST_P(ExposureTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_exposure<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Color, ExposureTest,
                         ::testing::ValuesIn(with_params<ExposureParams>(
                             concat_configs({
                                 make_configs({DType::U8, DType::F16, DType::F32},
                                              presets::kLayoutsFullConv, {Roi::Full, Roi::Partial},
                                              {presets::kTailWidthSize}),
                                 make_configs({DType::U8, DType::F16, DType::F32},
                                              presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {ExposureParams{0.5f}})),
                         op_config_name<ExposureParams>);
