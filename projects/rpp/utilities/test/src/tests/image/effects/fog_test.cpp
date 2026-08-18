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

#include <cmath>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/tensor_setup.hpp"

using namespace rpptest;

namespace {

// fog has no golden model, and unlike the other effects ops it has no parameter corner at which
// one could be written. The output is a blend of the image with a fog layer baked into the library
// (src/include/tensor/fog_mask.hpp, nearest-neighbour resized to the image), and intensityFactor
// is an additive bias on that layer's alpha rather than a weight on the whole contribution: at
// intensityFactor = 0 the baked layer still applies at full strength, so (0, 0) is not an
// identity. Both backends agree on this, and the public API doc does not describe the layer, so
// nothing here is derivable from the spec.
//
// What is checked instead are the properties the API does pin down without the mask: the
// Correctness intent runs the op over the whole supported grid and holds it to storable output,
// and the Negative intent holds it to its documented parameter ranges. They take disjoint
// parameter sets, so they are separate fixtures -- GTest instantiates every body of a fixture
// against that fixture's whole param set, which would otherwise generate (and skip) each check
// across the other's grid.

// Mid-range values inside both documented ranges (0 <= intensityFactor <= 0.5,
// 0 <= greyFactor <= 1), well away from either endpoint so the fog contribution is unambiguous.
constexpr float kIntensity = 0.4f;
constexpr float kGrey = 0.6f;

// Runs the kernel at parameters inside the documented ranges and asserts what holds without
// knowing the fog layer: the call succeeds, the effect is actually applied (the ROI is not
// returned unchanged), and every ROI output element is a legally storable value for its dtype --
// U8 [0,255], I8 [-128,127], F16/F32 [0,1] per the header's depth ranges. Catches a dead kernel,
// NaN, overflow and wrong-intensity-space output; says nothing about the fog layer's shape.
template <typename T>
void run_fog(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> intensity(cfg.backend, shape.n), grey(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        intensity[i] = kIntensity;
        grey[i] = kGrey;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count), actual(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // seeded from the source so "unchanged" means a dead kernel

    RppHandle handle(cfg.backend, shape.n);
    ASSERT_EQ(rppt_fog(src.ptr(), &desc, dst.ptr(), &desc, intensity.data(), grey.data(),
                       roi.data(), XYWH, handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    // The source is read at the ROI offset and the output written packed at the destination
    // origin, so the comparison walks the same mapping the op uses. Uses the test's own ROI copy,
    // not the tensor handed to the op, in case the backend rewrites it from XYWH to LTRB in place.
    bool changed = false, storable = true;
    double firstBad = 0.0;
    for_each_roi_io(desc, roiVec.data(), XYWH,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        const double v = to_double(actual[dstIdx]);
                        if (to_double(input[srcIdx]) != v) changed = true;
                        if (storable &&
                            (!std::isfinite(v) ||
                             std::fabs(v - quantize_stored(v, cfg.dtype)) > 1e-6)) {
                            storable = false;
                            firstBad = v;
                        }
                    });
    EXPECT_TRUE(changed) << "fog left the ROI unchanged at intensityFactor " << kIntensity
                         << " / greyFactor " << kGrey;
    EXPECT_TRUE(storable) << "fog produced " << firstBad << ", outside the storable range for dtype "
                          << dtype_name(cfg.dtype);
}

// One out-of-range parameter pair per case. Both bounds of both ranges are covered; the parameter
// values are the grid axis here because the dtype/layout axes are irrelevant to argument
// validation, which never reaches the kernel.
struct FogNegativeParams {
    float intensity;
    float grey;
    std::string name() const { return "i" + num_token(intensity) + "_g" + num_token(grey); }
};

// The header constrains both parameters (0 <= intensityFactor <= 0.5, 0 <= greyFactor <= 1), so a
// value outside those ranges is not a legal call and must be reported rather than silently
// producing an undefined image. Only the status is asserted -- the header does not name which
// RPP_ERROR* an out-of-range scalar maps to.
template <typename T>
void run_fog_negative(const TestConfig& cfg, const FogNegativeParams& op) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    RpptDesc desc = make_descriptor(shape, cfg.dtype, cfg.layout);
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<Rpp32f> intensity(cfg.backend, shape.n), grey(cfg.backend, shape.n);
    PinnedArray<RpptROI> roi(cfg.backend, shape.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u i = 0; i < shape.n; ++i) {
        intensity[i] = op.intensity;
        grey[i] = op.grey;
        roi[i] = roiVec[i];
    }

    std::vector<T> input(count);
    fill_input<T>(input.data(), count, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, shape.n);
    const RppStatus status = rppt_fog(src.ptr(), &desc, dst.ptr(), &desc, intensity.data(),
                                      grey.data(), roi.data(), XYWH, handle.get(), cfg.backend);
    handle.sync();
    EXPECT_NE(status, RPP_SUCCESS)
        << "fog accepted out-of-range parameters (intensityFactor " << op.intensity
        << ", greyFactor " << op.grey << ")";
}

}  // namespace

// Full name: Image_Effects/FogTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>
class FogTest : public ::testing::TestWithParam<TestConfig> {};

TEST_P(FogTest, Correctness) {
    const TestConfig& cfg = GetParam();
    dispatch_dtype<DType::U8, DType::F16, DType::F32, DType::I8>(cfg.dtype, [&](auto tag) {
        run_fog<Element<decltype(tag)>>(cfg);
    });
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, FogTest,
                         ::testing::ValuesIn(make_configs(
                             {DType::U8, DType::F16, DType::F32, DType::I8},
                             {Layout::PKD3, Layout::PLN3, Layout::PLN1},
                             {Roi::Full, Roi::Partial})),
                         config_param_name);

// Full name:
// Image_Effects/FogNegativeTest.Negative/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_i<I>_g<G>
class FogNegativeTest : public ::testing::TestWithParam<WithParams<FogNegativeParams>> {};

TEST_P(FogNegativeTest, Negative) {
    const auto& p = GetParam();
    run_fog_negative<Rpp8u>(p.cfg, p.op);
}

// Argument validation is dtype- and layout-independent, so it runs on one representative config.
INSTANTIATE_TEST_SUITE_P(
    Image_Effects, FogNegativeTest,
    ::testing::ValuesIn(with_params<FogNegativeParams>(
        make_configs({DType::U8}, {Layout::PKD3}, {Roi::Full}),
        {FogNegativeParams{-0.1f, 0.5f}, FogNegativeParams{0.6f, 0.5f},
         FogNegativeParams{0.25f, -0.1f}, FogNegativeParams{0.25f, 1.1f}})),
    op_config_name<FogNegativeParams>);
