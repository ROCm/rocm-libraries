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

#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/flip_ref.hpp"

using namespace rpptest;

namespace {

// The flags are independent 0/1 masks, so the grid covers all four combinations.
struct FlipParams {
    Rpp32u horizontal, vertical;
    std::string name() const {
        return "h" + std::to_string(horizontal) + "_v" + std::to_string(vertical);
    }
};

template <typename T>
void run_flip(const TestConfig& cfg, const FlipParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    PinnedArray<Rpp32u> horizontal(cfg.backend, cfg.size.n);
    PinnedArray<Rpp32u> vertical(cfg.backend, cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        horizontal[i] = op.horizontal;
        vertical[i] = op.vertical;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    flip_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, roiVec.data(), XYWH,
                      op.horizontal, op.vertical);

    // dst is pre-filled with a distinct pattern so a no-op kernel is caught in the identity case.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_flip(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, horizontal.data(),
                        vertical.data(), roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // Uses the test's own ROI copy: HIP rewrites the op's ROI to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH, 0.0));
}

}  // namespace

// Full name:
// Image_Geometric/FlipTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Flags>
class FlipTest : public SkipListTest<WithParams<FlipParams>> {};

TEST_P(FlipTest, Correctness) {
    const auto& p = GetParam();
    // The HIP vertical flip reads out of bounds when the ROI does not reach the bottom row, and the
    // fault poisons the HIP context for the rest of the process. Remove once the kernel is fixed.
    if (p.cfg.backend == RPP_HIP_BACKEND && p.cfg.roi == Roi::Partial && p.op.vertical)
        GTEST_SKIP() << "HIP flip reads out of bounds on a vertical flip with a partial ROI and "
                        "faults the device";

    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_flip<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, FlipTest,
    ::testing::ValuesIn(with_params<FlipParams>(
        concat_configs({
            make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFullConv,
                         {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
            make_configs({DType::U8, DType::F16, DType::F32, DType::I8}, presets::kLayoutsFull,
                         {Roi::Full, Roi::Partial},
                         {presets::kDefaultSize, presets::kSubVectorSize}),
        }),
        {FlipParams{0, 0}, FlipParams{1, 0}, FlipParams{0, 1}, FlipParams{1, 1}})),
    op_config_name<FlipParams>);
