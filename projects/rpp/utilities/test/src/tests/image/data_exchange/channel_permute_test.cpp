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
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/channel_permute_ref.hpp"

using namespace rpptest;

namespace {

// The per-image channel order: output channel i takes source channel order[i]. Each value in 0..2
// (documented range). A rotation exercises every channel moving to a new slot.
struct ChannelPermuteParams {
    Rpp32u order[3];
    std::string name() const {
        return "perm" + std::to_string(order[0]) + std::to_string(order[1]) +
               std::to_string(order[2]);
    }
};

template <typename T>
void run_channel_permute(const TestConfig& cfg, const ChannelPermuteParams& op) {
    const TensorShape shape{cfg.size.n, 3, cfg.size.h, cfg.size.w};  // channel_permute is 3-channel
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);   // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Parameters live in host-accessible (pinned for HIP) memory. permutationTensor holds n
    // contiguous per-image triples. channel_permute takes no ROI; the frame is walked in full.
    PinnedArray<Rpp32u> perm(cfg.backend, 3 * shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        perm[i * 3 + 0] = op.order[0];
        perm[i * 3 + 1] = op.order[1];
        perm[i * 3 + 2] = op.order[2];
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. golden starts as a copy of the input; the reference overwrites the
    // whole frame with the permuted channels.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    channel_permute_reference<T>(input.data(), golden.data(), desc, perm.data(), roi.data(), XYWH);

    // (2) Run RPP on the configured backend. dst is pre-filled with a distinct pattern so a
    // no-op kernel would be caught.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    std::vector<T> dstInit(count);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_channel_permute(src.ptr(), &desc, dst.ptr(), &desc, perm.data(), handle.get(),
                                   cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) Compare over the frame. channel_permute only reorders channels (no arithmetic), so it is
    // bit-exact -- tolerance 0. NOTE: HOST_I8toI8_PKD3 fails here because the HOST I8 packed kernel
    // reads uninitialized/out-of-bounds memory for a few elements (heap/order-dependent, values
    // typically I8-black -128); a real kernel bug, not a reference/tolerance issue. Do not loosen
    // the tolerance to hide it.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH, 0.0));
}

}  // namespace

// Full name:
// Image_DataExchange/ChannelPermuteTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Perm>
class ChannelPermuteTest : public ::testing::TestWithParam<WithParams<ChannelPermuteParams>> {};

TEST_P(ChannelPermuteTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_channel_permute<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

// channel_permute is 3-channel only (c = 3) and takes no ROI, so PLN1 and Partial are not
// instantiated. A rotation {2,0,1} moves every channel and is sensitive to the mapping direction.
INSTANTIATE_TEST_SUITE_P(
    Image_DataExchange, ChannelPermuteTest,
    ::testing::ValuesIn(with_params<ChannelPermuteParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3}, {Roi::Full}),
        {ChannelPermuteParams{{2, 0, 1}}})),
    op_config_name<ChannelPermuteParams>);
