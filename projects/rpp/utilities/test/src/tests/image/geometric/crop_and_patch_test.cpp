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
#include "reference/crop_and_patch_ref.hpp"

using namespace rpptest;

namespace {

// A crop {cropX,cropY,w,h} from srcPtr1, patched at {patchX,patchY}. Crop size == patch size.
struct CropAndPatchParams {
    Rpp32u cropX, cropY, w, h, patchX, patchY;
    std::string tag;
    // Extent written into patchRoiTensor, which neither backend reads. 0 means same as the crop.
    Rpp32u patchW = 0, patchH = 0;
    // Destination region; dstW == 0 means the full image.
    Rpp32u dstX = 0, dstY = 0, dstW = 0, dstH = 0;
    std::string name() const {
        return tag;
    }
};

template <typename T>
void run_crop_and_patch(const TestConfig& cfg, const CropAndPatchParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layoutIn));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layoutIn);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // dstRoi sizes the output and offsets the src2 read; crop/patch carry the op's rectangles.
    const Rpp32u dstW = op.dstW ? op.dstW : desc.w, dstH = op.dstW ? op.dstH : desc.h;
    const Rpp32u dstX = op.dstW ? op.dstX : 0u, dstY = op.dstW ? op.dstY : 0u;
    const Rpp32u patchW = op.patchW ? op.patchW : op.w, patchH = op.patchH ? op.patchH : op.h;
    ASSERT_LE(dstX + dstW, desc.w) << "dstRoi overruns image width";
    ASSERT_LE(dstY + dstH, desc.h) << "dstRoi overruns image height";
    PinnedArray<RpptROI> dstRoi(cfg.backend, shape.n), cropRoi(cfg.backend, shape.n),
        patchRoi(cfg.backend, shape.n);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        dstRoi[i].xywhROI.xy.x = static_cast<int>(dstX);
        dstRoi[i].xywhROI.xy.y = static_cast<int>(dstY);
        dstRoi[i].xywhROI.roiWidth = static_cast<int>(dstW);
        dstRoi[i].xywhROI.roiHeight = static_cast<int>(dstH);
        cropRoi[i].xywhROI.xy.x = static_cast<int>(op.cropX);
        cropRoi[i].xywhROI.xy.y = static_cast<int>(op.cropY);
        cropRoi[i].xywhROI.roiWidth = static_cast<int>(op.w);
        cropRoi[i].xywhROI.roiHeight = static_cast<int>(op.h);
        patchRoi[i].xywhROI.xy.x = static_cast<int>(op.patchX);
        patchRoi[i].xywhROI.xy.y = static_cast<int>(op.patchY);
        patchRoi[i].xywhROI.roiWidth = static_cast<int>(patchW);
        patchRoi[i].xywhROI.roiHeight = static_cast<int>(patchH);
    }

    std::vector<T> in1(count), in2(count), golden(count), actual(count);
    fill_input<T>(in1.data(), count, cfg.dtype);
    fill_input<T>(in2.data(), count, cfg.dtype, /*salt=*/1);
    golden = in2;
    crop_and_patch_reference<T>(in1.data(), in2.data(), golden.data(), desc, dstRoi.data(),
                                cropRoi.data(), patchRoi.data(), XYWH);

    // dst is pre-filled with src2 to mirror the golden.
    DeviceTensor s1(cfg.backend, bytes), s2(cfg.backend, bytes), dst(cfg.backend, bytes);
    s1.write(in1.data(), bytes);
    s2.write(in2.data(), bytes);
    dst.write(in2.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_crop_and_patch(s1.ptr(), s2.ptr(), &desc, dst.ptr(), &desc, dstRoi.data(),
                                  cropRoi.data(), patchRoi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // Verbatim copy, so compare the full destination bit-exact.
    PinnedArray<RpptROI> fullRoi(cfg.backend, shape.n);
    const std::vector<RpptROI> fullVec = make_roi(desc, Roi::Full);
    for (Rpp32u i = 0; i < shape.n; ++i) fullRoi[i] = fullVec[i];
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, fullRoi.data(), XYWH, 0.0));
}

}  // namespace

// Full name:
// Image_Geometric/CropAndPatchTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Case>
class CropAndPatchTest : public SkipListTest<WithParams<CropAndPatchParams>> {};

TEST_P(CropAndPatchTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(
        p.cfg.dtype, [&](auto tag) { run_crop_and_patch<Element<decltype(tag)>>(p.cfg, p.op); });
}

// Full ROI only: the op's regions are the dstRoi/crop/patch params, varied directly below.
// "patchextent" checks the patch extent is ignored in favour of the crop's. "dstroi" stays inside
// the destination region to avoid the HOST over-hanging-patch underflow.
INSTANTIATE_TEST_SUITE_P(Image_Geometric, CropAndPatchTest,
                         ::testing::ValuesIn(with_params<CropAndPatchParams>(
                             make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                          {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full},
                                          {presets::kDefaultSize, presets::kTailWidthSize}),
                             {CropAndPatchParams{8, 6, 16, 12, 8, 6, "inplace"},
                              CropAndPatchParams{8, 6, 16, 12, 24, 18, "moved"},
                              CropAndPatchParams{8, 6, 16, 12, 8, 6, "patchextent", 4, 4},
                              CropAndPatchParams{8, 6, 16, 12, 8, 6, "dstroi", 0, 0, 8, 6, 32,
                                                 24}})),
                         op_config_name<CropAndPatchParams>);
