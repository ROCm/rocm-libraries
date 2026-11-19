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
#include "framework/reduction.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/tensor_max_ref.hpp"

using namespace rpptest;

namespace {

// Tin is the source element type; Tout is the op's output element type.
template <typename Tin, typename Tout>
void run_tensor_max(const TestConfig& cfg) {
    RpptDesc desc = make_src_descriptor(cfg);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);
    const std::size_t outLen = reduction_length(desc);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    std::vector<Tin> input(count);
    fill_input<Tin>(input.data(), count, cfg.dtype);
    const std::vector<double> golden =
        tensor_max_reference<Tin>(input.data(), desc, roi.data(), XYWH);

    DeviceTensor src(cfg.backend, bytes);
    src.write(input.data(), bytes);
    PinnedArray<Tout> out(cfg.backend, outLen);
    for (std::size_t i = 0; i < outLen; ++i) out[i] = static_cast<Tout>(0);

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(rppt_tensor_max(src.ptr(), &desc, out.data(), static_cast<Rpp32u>(outLen), roi.data(),
                              XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // pinned output is then directly readable on the host

    // Selects an existing element, so bit-exact.
    EXPECT_TRUE(compare_reduction<Tout>(out.data(), golden, kExact(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Statistical/TensorMaxTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class TensorMaxTest : public SkipListTest<TestConfig> {};

TEST_P(TensorMaxTest, Correctness) {
    const TestConfig cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_tensor_max<Element<decltype(tag)>, Element<decltype(tag)>>(cfg);
    });
}

INSTANTIATE_TEST_SUITE_P(Image_Statistical, TensorMaxTest,
                         ::testing::ValuesIn(make_configs(
                             {DType::U8, DType::F16, DType::F32, DType::I8},
                             {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial},
                             // Smaller images underflow the tensor_max HIP stage-2
                             // reduction index and fault. The odd shape hits a partial
                             // vector at the end of a row.
                             {{2, 160, 160}, {2, 137, 167}})),
                         config_param_name);
