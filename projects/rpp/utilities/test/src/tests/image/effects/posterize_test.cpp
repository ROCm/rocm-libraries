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
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/posterize_ref.hpp"

using namespace rpptest;

namespace {

// levelBits within the documented range 1 <= posterizeLevelBits <= 8. 4 keeps the top nibble; 2
// reduces harder to the top two bits -- both drop low bits so most pixels change.
struct PosterizeParams {
    int levelBits;
    std::string name() const { return "b" + num_token(static_cast<float>(levelBits)); }
};

// The integer bit-mask is exact; only the float paths carry rounding error.
constexpr Tolerance kPosterizeTolerance = tolerance(0.0, 1e-3, 4e-3);

template <typename T>
void run_posterize(const TestConfig& cfg, const PosterizeParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp8u> levelBits(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        levelBits[i] = static_cast<Rpp8u>(op.levelBits);
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    golden = input;
    posterize_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                           op.levelBits);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_posterize(src.ptr(), &desc, dst.ptr(), &desc, levelBits.data(), roi.data(), XYWH,
                             handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back

    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               kPosterizeTolerance(cfg.dtype)));
}

}  // namespace

// Full name: Image_Effects/PosterizeTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<LevelBits>
class PosterizeTest : public SkipListTest<WithParams<PosterizeParams>> {};

TEST_P(PosterizeTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_posterize<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Effects, PosterizeTest,
    ::testing::ValuesIn(with_params<PosterizeParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {PosterizeParams{4}, PosterizeParams{2}})),
    op_config_name<PosterizeParams>);
