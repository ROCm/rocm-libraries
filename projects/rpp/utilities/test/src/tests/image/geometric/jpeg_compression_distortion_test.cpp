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
#include "reference/jpeg_compression_distortion_ref.hpp"

using namespace rpptest;

namespace {

// quality is the JPEG quality factor, documented range 1..99: lower means coarser quantization.
// Three points on the table scaling, chosen so the two failure modes of a mis-scaled quantization
// table separate:
//   q50 : scale = 100, so the scaled table IS the Annex K base table -- integral and well under
//         255, the one quality at which a real-valued quantizer cannot differ from the spec's
//         8-bit one. This is the grid's green half and its regression anchor.
//   q10 : scale = 500, so the spec clamps the high-frequency entries at 255 (unclamped they reach
//         605) -- surfaces the missing baseline clamp.
//   q90 : scale = 20, so the entries are fractional (16 -> 3.2) and the spec rounds them --
//         surfaces the missing integer rounding.
// q10 and q90 are red by design: the kernel quantizes by the raw real-valued product, so those two
// halves of the grid are the standing evidence of that defect. q50 must stay green.
struct JpegParams {
    int quality;
    std::string name() const { return "q" + std::to_string(quality); }
};

// The distortion is a DCT round-trip, so the only legitimate error is fp rounding in the two
// transforms -- these are the suite's standard per-dtype tolerances (~half an intensity level),
// and at q50 the whole pipeline agrees with the kernel inside them. They are deliberately NOT
// loosened to cover q10/q90: the quantize step is a hard threshold, so a wrong quantizer moves
// output pixels by tens of levels, and no honest tolerance hides that.
constexpr Tolerance kJpegTolerance = kRoundingTolerance;

template <typename T>
void run_jpeg_compression_distortion(const TestConfig& cfg, const JpegParams& op) {
    const Rpp32u c = static_cast<Rpp32u>(channels_of(cfg.layout));
    const TensorShape shape{cfg.size.n, c, cfg.size.h, cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32s> quality(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        quality[i] = op.quality;
        roi[i] = roiVec[i];
    }

    // (1) Host golden model. The op writes only the ROI-sized region at the destination origin, so
    // golden and the device buffer start from the same distinct pattern and only that region is
    // compared.
    // Quantization is a hard threshold, so an input whose DCT coefficients land exactly on a
    // rounding boundary makes the comparison ill-posed: which way golden and kernel round is then
    // decided by arithmetic noise rather than by semantics. The shared fill is an arithmetic ramp,
    // which does exactly that -- at the default salt several coefficients are exact half-integers
    // (margin 4e-16), and salts 3/4/5 are degenerate too. salt 1 is the offset with the widest
    // margin, measured over the whole instantiated grid (every dtype, layout, ROI and quality) by
    // probing |F/Q - round(F/Q)| inside the golden: 4.9e-6 quantizer levels, against 2.6e-6 for
    // salt 2. Thin, but empirically clear of the kernel's fp32 noise -- the q50 half of the grid,
    // where the golden and the kernel must agree exactly, is bit-exact on both backends. If the
    // fill pattern, the tensor size or the quality set ever changes, re-measure: a reintroduced
    // tie shows up as a 1-LSB red that reads exactly like a kernel defect.
    std::vector<T> input(count), dstInit(count), golden(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype, /*salt=*/1);
    fill_input<T>(dstInit.data(), count, cfg.dtype, /*salt=*/3);
    golden = dstInit;
    jpeg_compression_distortion_reference<T>(input.data(), golden.data(), desc, cfg.dtype,
                                             roi.data(), XYWH, op.quality);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(dstInit.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_jpeg_compression_distortion(src.ptr(), &desc, dst.ptr(), &desc, quality.data(),
                                               roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();
    dst.read(actual.data(), bytes);

    // (3) Compare the ROI-sized region written at the destination origin.
    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               kJpegTolerance(cfg.dtype)));
}

}  // namespace

// Full name:
// Image_Geometric/JpegCompressionDistortionTest.Correctness/
//   <Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_q<Quality>
class JpegCompressionDistortionTest : public ::testing::TestWithParam<WithParams<JpegParams>> {};

TEST_P(JpegCompressionDistortionTest, Correctness) {
    const auto& p = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(p.cfg.dtype, [&](auto tag) {
        run_jpeg_compression_distortion<Element<decltype(tag)>>(p.cfg, p.op);
    });
}

INSTANTIATE_TEST_SUITE_P(
    Image_Geometric, JpegCompressionDistortionTest,
    ::testing::ValuesIn(with_params<JpegParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {JpegParams{50}, JpegParams{10}, JpegParams{90}})),
    op_config_name<JpegParams>);
