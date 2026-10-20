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
#include "framework/tolerance.hpp"
#include "reference/emboss_ref.hpp"

using namespace rpptest;

namespace {

// strength has no documented range. 1.0 is DC-preserving; 2.0 saturates the scaling and exercises
// the I8 intensity-space shift, which cancels at strength 1.
struct EmbossParams {
    float strength;
    Rpp32u kernelSize;
    std::string name() const {
        return "k" + std::to_string(kernelSize) + "_s" + num_token(strength);
    }
};

// Linear convolution rounded to nearest, so only accumulation error. Not loosened to cover the
// kernel reading neighbours from outside a partial ROI.
constexpr Tolerance kEmbossTolerance = tolerance(1.0, 1e-3, 5e-3);

template <typename T>
void run_emboss(const TestConfig& cfg, const EmbossParams& op) {
    // The HIP kernel requires srcDesc.offsetInBytes >= 12 * (kernelSize/2) as read-slack for the
    // window (RPP_ERROR_LOW_OFFSET otherwise), so src carries a leading pad. dst keeps offset 0.
    RpptDesc srcDesc = make_src_descriptor(cfg);
    RpptDesc dstDesc = make_dst_descriptor(cfg);
    const std::size_t offsetBytes = 12u * (op.kernelSize / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t count = element_count(srcDesc);
    const std::size_t imageBytes = count * dtype_size(cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, cfg.size.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) roi[i] = roiVec[i];

    PinnedArray<Rpp32f> strength(cfg.backend, cfg.size.n);
    for (Rpp32u i = 0; i < cfg.size.n; ++i) strength[i] = op.strength;

    // src is [front pad][image][back pad]; the image base sits at offsetElems.
    std::vector<T> src(offsetElems + count + offsetElems), golden(count), actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    T* image = src.data() + offsetElems;
    golden.assign(image, image + count);
    emboss_reference<T>(image, srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                        op.strength, op.kernelSize);

    // REPLICATE is the only border type the op supports.
    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(image, imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    ASSERT_EQ(
        rppt_emboss(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, strength.data(), op.kernelSize,
                    RpptImageBorderType::REPLICATE, roi.data(), XYWH, handle.get(), cfg.backend),
        RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), imageBytes);

    // Uses the test's own ROI copy: HIP rewrites the op's ROI to LTRB in place.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH,
                               kEmbossTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Filter/EmbossTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_k<K>_s<Strength>
class EmbossTest : public SkipListTest<WithParams<EmbossParams>> {};

TEST_P(EmbossTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_emboss<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Filter, EmbossTest,
                         ::testing::ValuesIn(with_params<EmbossParams>(
                             concat_configs({
                                 make_configs(presets::kDefaultDTypes, presets::kLayoutsFullConv,
                                              {Roi::Full, Roi::Partial}, {presets::kTailWidthSize}),
                                 make_configs(presets::kDefaultDTypes, presets::kLayoutsFull,
                                              {Roi::Full, Roi::Partial},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {EmbossParams{1.0f, 3}, EmbossParams{1.0f, 5}, EmbossParams{1.0f, 7},
                              EmbossParams{1.0f, 9}, EmbossParams{2.0f, 3}, EmbossParams{2.0f, 5},
                              EmbossParams{2.0f, 7}, EmbossParams{2.0f, 9}})),
                         op_config_name<EmbossParams>);
