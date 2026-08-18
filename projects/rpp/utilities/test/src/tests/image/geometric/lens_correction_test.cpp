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
#include "reference/lens_correction_ref.hpp"

using namespace rpptest;

namespace {

// The intrinsics the API's own sample uses, calibrated for a 640x480 frame. The distortion
// coefficients are dimensionless (they act on normalized coordinates) so they carry over unchanged
// to any frame size, but fx/fy/cx/cy are in pixels and must be scaled to the tested image or the
// principal point lands far outside it and the whole map falls off-frame.
constexpr double kRefW = 640.0, kRefH = 480.0;
constexpr double kFx = 534.07088364, kCx = 341.53407554;
constexpr double kFy = 534.11914595, kCy = 232.94565259;

enum class LensKind { NoDistortion, Barrel };

// NoDistortion: every coefficient 0, so the map is the exact identity and the output must be a
// verbatim copy of the ROI. This is the case that validates the model independently of the
// undocumented interpolation choice -- at integer source coordinates bilinear and nearest-neighbour
// agree -- and separates a plumbing failure from a distortion-math disagreement.
// Barrel: the sample coefficients from the API/legacy harness (k1 < 0, the barrel case the op's
// description names).
struct LensParams {
    LensKind kind;
    std::string name() const { return kind == LensKind::NoDistortion ? "NoDistortion" : "Barrel"; }
};

// NoDistortion deliberately uses IDENTITY intrinsics (fx = fy = 1, cx = cy = 0) rather than the
// scaled sample ones. With zero coefficients the map reduces to srcX = fx*((outX-cx)/fx)+cx, which
// is only exactly outX when fx and cx are exact: with the scaled intrinsics it lands on outX +/- 1e-7,
// putting the frame border on a float knife-edge where the sign decides whether the low tap is
// in-frame. Identity intrinsics make the map exactly the identity for every pixel, so this set is a
// clean plumbing check and any diff is a real defect. Determinant is 1, so the matrix is valid.
void fill_camera_matrix(Rpp32f* m, Rpp32u w, Rpp32u h, LensKind kind) {
    const bool identity = kind == LensKind::NoDistortion;
    const double sx = static_cast<double>(w) / kRefW, sy = static_cast<double>(h) / kRefH;
    m[0] = identity ? 1.f : static_cast<Rpp32f>(kFx * sx);
    m[1] = 0.f;
    m[2] = identity ? 0.f : static_cast<Rpp32f>(kCx * sx);
    m[3] = 0.f;
    m[4] = identity ? 1.f : static_cast<Rpp32f>(kFy * sy);
    m[5] = identity ? 0.f : static_cast<Rpp32f>(kCy * sy);
    m[6] = 0.f;
    m[7] = 0.f;
    m[8] = 1.f;
}

void fill_distortion(Rpp32f* d, LensKind kind) {
    static const Rpp32f kBarrel[8] = {-0.29297164f, 0.10770696f, 0.00131038f, -0.0000311f,
                                      0.0434798f,   0.f,         0.f,         0.f};
    for (int i = 0; i < 8; ++i) d[i] = (kind == LensKind::Barrel) ? kBarrel[i] : 0.f;
}

// The map is resolved with bilinear sampling, so only genuine fp rounding is allowed. These are the
// suite's shared warp/remap bilinear tolerances; they are NOT loosened to cover the resize-family
// trailing-edge bilinear defect or any partial-ROI placement divergence, which stay red.
constexpr Tolerance kLensCorrectionTolerance = kRoundingTolerance;

template <typename T>
void run_lens_correction(const TestConfig& cfg, const LensParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const Rpp32u N = cfg.size.n, imgH = cfg.size.h, imgW = cfg.size.w;
    const TensorShape shape{N, c, imgH, imgW};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // src == dst
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    // Remap tables: a separate F32, single-channel, UNPADDED tensor (row stride exactly imgW), as
    // the header specifies. They are pure scratch here -- the op derives the map from the
    // intrinsics and writes it into them.
    RpptDesc td{};
    td.numDims = 4;
    td.offsetInBytes = 0;
    td.dataType = F32;
    td.layout = NHWC;
    td.n = N;
    td.c = 1;
    td.h = imgH;
    td.w = imgW;
    td.strides.nStride = static_cast<Rpp32u>(imgH) * imgW;
    td.strides.hStride = imgW;
    td.strides.wStride = 1;
    td.strides.cStride = 1;

    PinnedArray<RpptROI> roi(cfg.backend, N);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < N; ++i) roi[i] = roiVec[i];

    PinnedArray<Rpp32f> cameraMatrix(cfg.backend, static_cast<std::size_t>(N) * 9);
    PinnedArray<Rpp32f> distortion(cfg.backend, static_cast<std::size_t>(N) * 8);
    for (Rpp32u n = 0; n < N; ++n) {
        fill_camera_matrix(cameraMatrix.data() + static_cast<std::size_t>(n) * 9, imgW, imgH, op.kind);
        fill_distortion(distortion.data() + static_cast<std::size_t>(n) * 8, op.kind);
    }

    const std::size_t tblCount = static_cast<std::size_t>(N) * imgH * imgW;
    PinnedArray<Rpp32f> rowT(cfg.backend, tblCount), colT(cfg.backend, tblCount);
    for (std::size_t k = 0; k < tblCount; ++k) rowT[k] = colT[k] = 0.0f;

    // (1) Host golden model. The op writes the ROI-sized region at the destination origin, so
    // golden and the device buffer start from the same distinct pattern and only that region is
    // compared.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/1);
    golden = dstInit;
    lens_correction_reference<T>(input.data(), golden.data(), desc, cfg.dtype, roi.data(), XYWH,
                                 cameraMatrix.data(), distortion.data(), BILINEAR);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, N);
    ASSERT_EQ(rppt_lens_correction(src.ptr(), &desc, dst.ptr(), &desc, rowT.data(), colT.data(),
                                   &td, cameraMatrix.data(), distortion.data(), roi.data(), XYWH,
                                   handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // (3) Compare the ROI-sized region written at the destination origin. The comparison uses the
    // caller's own copy of the ROI, not the tensor handed to the op: HIP rewrites that tensor from
    // XYWH to LTRB in place (roiWidth/roiHeight come back as rb.x/rb.y), which would otherwise make
    // the comparison walk a different region than the golden wrote.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roiVec.data(), XYWH,
                               kLensCorrectionTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Geometric/LensCorrectionTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Kind>
class LensCorrectionTest : public ::testing::TestWithParam<WithParams<LensParams>> {};

TEST_P(LensCorrectionTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_lens_correction<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, LensCorrectionTest,
    ::testing::ValuesIn(with_params<LensParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {LensParams{LensKind::NoDistortion}, LensParams{LensKind::Barrel}})),
    op_config_name<LensParams>);
