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
#include "reference/dilate_ref.hpp"

using namespace rpptest;

namespace {

struct DilateParams {
    Rpp32u kernelSize;
    std::string name() const {
        return "k" + std::to_string(kernelSize);
    }
};

template <typename T>
void run_dilate(const TestConfig& cfg, const DilateParams& op) {
    // The HIP morphology kernel requires srcDesc.offsetInBytes >= 12 * (kernelSize/2) as read-slack
    // for the window, so src carries a leading pad. dst keeps offset 0.
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

    // src is [front pad][image][back pad]; the image base sits at offsetElems.
    std::vector<T> src(offsetElems + count + offsetElems), golden(count), actual(count);
    fill_input<T>(src.data(), src.size(), cfg.dtype);
    T* image = src.data() + offsetElems;
    golden.assign(image, image + count);
    dilate_reference<T>(image, srcDesc, golden.data(), dstDesc, cfg.dtype, roi.data(), XYWH,
                        op.kernelSize);

    // dilate has a separate HOST symbol; the backend-arg symbol only works in a HIP build.
    DeviceTensor srcDev(cfg.backend, src.size() * dtype_size(cfg.dtype));
    DeviceTensor dst(cfg.backend, imageBytes);
    srcDev.write(src.data(), src.size() * dtype_size(cfg.dtype));
    dst.write(image, imageBytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, cfg.size.n);
    RppStatus status;
    if (cfg.backend == RPP_HIP_BACKEND) {
#if RPP_BACKEND_HIP
        status = rppt_dilate(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.kernelSize, roi.data(),
                             XYWH, handle.get(), cfg.backend);
#else
        GTEST_SKIP() << "HIP backend not built";  // unreachable: available_backends() gates it
#endif
    } else {
        status = rppt_dilate_host(srcDev.ptr(), &srcDesc, dst.ptr(), &dstDesc, op.kernelSize,
                                  roi.data(), XYWH, handle.get());
    }
    ASSERT_EQ(status, RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), imageBytes);

    // dilate selects an existing pixel, so it is bit-exact. Known kernel red: HOST float PLN
    // partial-ROI k5 reads neighbours from outside the ROI.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), dstDesc, roi.data(), XYWH, 0.0));
}

}  // namespace

// Full name:
// Image_Morphological/DilateTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_k<KernelSize>
class DilateTest : public SkipListTest<WithParams<DilateParams>> {};

TEST_P(DilateTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_dilate<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Same-layout cases plus both directions of the fused output-layout conversion.
INSTANTIATE_TEST_SUITE_P(Image_Morphological, DilateTest,
                         ::testing::ValuesIn(with_params<DilateParams>(
                             concat_configs({
                                 make_configs({DType::U8, DType::F16, DType::F32},
                                              presets::kLayoutsFullConv, {Roi::Full, Roi::Partial},
                                              {presets::kTailWidthSize}),
                                 make_configs({DType::U8, DType::F16, DType::F32},
                                              presets::kLayoutsFull, {Roi::Full, Roi::Partial},
                                              {presets::kDefaultSize, presets::kSubVectorSize}),
                             }),
                             {DilateParams{3}, DilateParams{5}, DilateParams{7}, DilateParams{9}})),
                         op_config_name<DilateParams>);
