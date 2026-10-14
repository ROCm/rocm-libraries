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
#include "reference/channel_permute_ref.hpp"

using namespace rpptest;

namespace {

// Output channel i takes source channel order[i]; each value in 0..2 (documented range).
struct ChannelPermuteParams {
    Rpp32u order[3];
    std::string name() const {
        return "perm" + std::to_string(order[0]) + std::to_string(order[1]) +
               std::to_string(order[2]);
    }
};

template <typename T>
void run_channel_permute(const TestConfig& cfg, const ChannelPermuteParams& op) {
    RpptDesc srcDesc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t count = element_count(srcDesc);
    const std::size_t bytes = byte_size(srcDesc, cfg.dtype);

    // permutationTensor holds n contiguous per-image triples. The op takes no ROI; roi only drives
    // the golden over the full frame.
    PinnedArray<Rpp32u> perm(cfg.backend, 3 * cfg.size.n);
    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) {
        perm[i * 3 + 0] = op.order[0];
        perm[i * 3 + 1] = op.order[1];
        perm[i * 3 + 2] = op.order[2];
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    channel_permute_reference<T>(input.data(), srcDesc, golden.data(), dstDesc, perm.data(),
                                 roi.data(), XYWH);

    // dst is pre-filled with a distinct pattern so a no-op kernel would be caught.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    std::vector<T> dstInit(count);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_channel_permute(src.ptr(), &srcDesc, dst.ptr(), &dstDesc, perm.data(),
                                   handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // Pure reordering, so bit-exact. HOST_I8toI8_PKD3 can fail because the HOST I8 packed kernel
    // reads out of bounds for a few elements; do not loosen the tolerance to hide it.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, 0.0));
}

}  // namespace

// Full name:
// Image_DataExchange/ChannelPermuteTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Perm>
class ChannelPermuteTest : public SkipListTest<WithParams<ChannelPermuteParams>> {};

TEST_P(ChannelPermuteTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_channel_permute<Element<decltype(tag)>>(p.cfg, p.op); });
}

// channel_permute is 3-channel only (c = 3) and takes no ROI, so PLN1 and Partial are not
// instantiated. A rotation {2,0,1} moves every channel and is sensitive to the mapping direction.
// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_DataExchange, ChannelPermuteTest,
                         ::testing::ValuesIn(with_params<ChannelPermuteParams>(
                             concat_configs({
                                 make_configs(presets::kDefaultDTypes, presets::kLayouts3ChConv,
                                              {Roi::Full}, {presets::kTailWidthSize}),
                                 make_configs(presets::kDefaultDTypes, presets::kLayouts3Ch,
                                              {Roi::Full},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {ChannelPermuteParams{{2, 0, 1}}})),
                         op_config_name<ChannelPermuteParams>);
