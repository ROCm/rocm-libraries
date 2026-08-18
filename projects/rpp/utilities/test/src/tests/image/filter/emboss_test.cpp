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
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/emboss_ref.hpp"

using namespace rpptest;

namespace {

// kernelSize is documented as 3/5/7/9; the grid covers 3 and 5, matching how box/gaussian are
// gridded. strength has no documented range -- 1.0 is the value the legacy harness uses and leaves
// the filter DC-preserving, 2.0 is the point the scaling saturates at, which also exercises the
// I8 intensity-space lift (at strength 1 the +128 shift cancels and would hide a mistake there).
struct EmbossParams {
    float strength;
    Rpp32u kernelSize;
    std::string name() const {
        return "k" + std::to_string(kernelSize) + "_s" + num_token(strength);
    }
};

// A linear KxK convolution rounded to nearest, so the only legitimate error is float-vs-double
// accumulation -- the same tolerances box_filter uses. Not loosened to cover the shared
// spatial-filter defect where a partial ROI reads neighbours from outside the ROI rectangle
// instead of treating its edge as the border, which stays red.
constexpr Tolerance kEmbossTolerance = tolerance(1.0, 1e-3, 5e-3);

template <typename T>
void run_emboss(const TestConfig& cfg, const EmbossParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    // src carries a leading border pad: like the other KxK filters, the HIP kernel requires
    // srcDesc.offsetInBytes >= 12 * (kernelSize/2) as read-slack for the window and returns
    // RPP_ERROR_LOW_OFFSET (-3) otherwise. dst keeps offset 0 so the golden and comparator index
    // the destination from 0. Applied on both backends (HOST honours offsetInBytes identically).
    RpptDesc srcDesc = make_descriptor(shape, cfg.dtype, cfg.layout);
    RpptDesc dstDesc = srcDesc;  // same dims/strides; only src gets the pad offset
    const std::size_t offsetBytes = 12u * (op.kernelSize / 2);
    const std::size_t offsetElems = offsetBytes / dtype_size(cfg.dtype);
    srcDesc.offsetInBytes = static_cast<Rpp32u>(offsetBytes);

    const std::size_t count = element_count(srcDesc);
    const std::size_t imageBytes = count * dtype_size(cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(srcDesc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) roi[i] = roiVec[i];

    PinnedArray<Rpp32f> strength(cfg.backend, shape.n);
    for (Rpp32u i = 0; i < shape.n; ++i) strength[i] = op.strength;

    // (1) Host golden model. The src buffer is [front pad][image][back pad]; the image base sits
    // at offsetElems. golden starts as a copy of the image so the region the op leaves untouched
    // (outside the ROI) is defined and a no-op kernel is still caught.
    std::vector<T> src(offsetElems + count + offsetElems), golden(count), actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    T* image = src.data() + offsetElems;
    golden.assign(image, image + count);
    emboss_reference<T>(image, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH, op.strength,
                        op.kernelSize);

    // (2) Run RPP on the configured backend. REPLICATE is the only border type the op supports.
    // RPP adds srcDesc.offsetInBytes to the src pointer internally, landing on the image base.
    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(image, imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_emboss(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, strength.data(),
                          op.kernelSize, RpptImageBorderType::REPLICATE, roi.data(), XYWH,
                          handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), imageBytes);

    // (3) Compare the ROI-sized region written at the destination origin. The comparison uses the
    // test's own ROI copy, not the tensor handed to the op: HIP rewrites the caller's XYWH tensor
    // to LTRB in place, so reusing it here would walk the wrong rectangle.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roiVec.data(), XYWH,
                               kEmbossTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Filter/EmbossTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_k<K>_s<Strength>
class EmbossTest : public ::testing::TestWithParam<WithParams<EmbossParams>> {};

TEST_P(EmbossTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_emboss<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Filter, EmbossTest,
    ::testing::ValuesIn(with_params<EmbossParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {EmbossParams{1.0f, 3}, EmbossParams{1.0f, 5}, EmbossParams{2.0f, 3},
         EmbossParams{2.0f, 5}})),
    op_config_name<EmbossParams>);
