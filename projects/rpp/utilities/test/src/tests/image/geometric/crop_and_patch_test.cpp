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
#include "framework/tensor_setup.hpp"
#include "reference/crop_and_patch_ref.hpp"

using namespace rpptest;

namespace {

// A crop rectangle {cropX,cropY,w,h} taken from srcPtr1, patched at {patchX,patchY} in the output.
// Crop size == patch size (no resize), the documented unambiguous case.
struct CropAndPatchParams {
    Rpp32u cropX, cropY, w, h, patchX, patchY;
    std::string tag;
    std::string name() const { return tag; }
};

template <typename T>
void run_crop_and_patch(const TestConfig& cfg, const CropAndPatchParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Three per-image ROI tensors in host-accessible (pinned for HIP) memory: the destination
    // region is always the full image; crop/patch carry the operator's rectangles.
    PinnedArray<RpptROI> dstRoi(cfg.backend, shape.n), cropRoi(cfg.backend, shape.n),
        patchRoi(cfg.backend, shape.n);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        dstRoi[i].xywhROI.xy.x = 0;
        dstRoi[i].xywhROI.xy.y = 0;
        dstRoi[i].xywhROI.roiWidth = static_cast<int>(desc.w);
        dstRoi[i].xywhROI.roiHeight = static_cast<int>(desc.h);
        cropRoi[i].xywhROI.xy.x = static_cast<int>(op.cropX);
        cropRoi[i].xywhROI.xy.y = static_cast<int>(op.cropY);
        cropRoi[i].xywhROI.roiWidth = static_cast<int>(op.w);
        cropRoi[i].xywhROI.roiHeight = static_cast<int>(op.h);
        patchRoi[i].xywhROI.xy.x = static_cast<int>(op.patchX);
        patchRoi[i].xywhROI.xy.y = static_cast<int>(op.patchY);
        patchRoi[i].xywhROI.roiWidth = static_cast<int>(op.w);
        patchRoi[i].xywhROI.roiHeight = static_cast<int>(op.h);
    }

    // (1) Host golden model. Two distinct operands (salt shifts the second); golden starts as a
    // copy of src2, then the reference overlays the src1 crop at the patch coords.
    std::vector<T> in1(count), in2(count), golden(count), actual(count);
    fill_input<T>(in1.data(), count, cfg.dtype);
    fill_input<T>(in2.data(), count, cfg.dtype, /*salt=*/1);
    golden = in2;
    crop_and_patch_reference<T>(in1.data(), in2.data(), golden.data(), desc, dstRoi.data(),
                                cropRoi.data(), patchRoi.data(), XYWH);

    // (2) Run RPP on the configured backend. dst is pre-filled with src2 to mirror the golden.
    DeviceTensor s1(cfg.backend, bytes), s2(cfg.backend, bytes), dst(cfg.backend, bytes);
    s1.write(in1.data(), bytes);
    s2.write(in2.data(), bytes);
    dst.write(in2.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_crop_and_patch(s1.ptr(), s2.ptr(), &desc, dst.ptr(), &desc, dstRoi.data(),
                                  cropRoi.data(), patchRoi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    // (3) Retrieve the result on the host (no-op copy for HOST, device->host for HIP).
    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // (4) crop_and_patch is a verbatim copy for every dtype, so compare the full destination image
    // bit-exact (tolerance 0).
    PinnedArray<RpptROI> fullRoi(cfg.backend, shape.n);
    const std::vector<RpptROI> fullVec = make_roi(desc, Roi::Full);
    for (Rpp32u i = 0; i < shape.n; ++i) fullRoi[i] = fullVec[i];
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, fullRoi.data(), XYWH, 0.0));
}

}  // namespace

// Full name: Image_Geometric/CropAndPatchTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Case>
class CropAndPatchTest : public ::testing::TestWithParam<WithParams<CropAndPatchParams>> {};

TEST_P(CropAndPatchTest, Correctness) {
    const auto& p = GetParam();
    switch (p.cfg.dtype) {
        case DType::U8:
            run_crop_and_patch<Rpp8u>(p.cfg, p.op);
            break;
        case DType::F16:
            run_crop_and_patch<Rpp16f>(p.cfg, p.op);
            break;
        case DType::F32:
            run_crop_and_patch<Rpp32f>(p.cfg, p.op);
            break;
        case DType::I8:
            run_crop_and_patch<Rpp8s>(p.cfg, p.op);
            break;
        default:
            FAIL() << "unsupported dtype for crop_and_patch";
    }
}

// Only {Roi::Full} on the roi axis: crop_and_patch has no standard source ROI (its regions are the
// crop/patch params) and its destination region is always the full image. Cases (default size
// 2x36x48): inplace patches back onto its own crop location [8,24)x[6,18); moved patches the same
// crop to [24,40)x[18,30) -- both rectangles fit within 48x36.
INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, CropAndPatchTest,
    ::testing::ValuesIn(with_params<CropAndPatchParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full}),
        {CropAndPatchParams{8, 6, 16, 12, 8, 6, "inplace"},
         CropAndPatchParams{8, 6, 16, 12, 24, 18, "moved"}})),
    op_config_name<CropAndPatchParams>);
