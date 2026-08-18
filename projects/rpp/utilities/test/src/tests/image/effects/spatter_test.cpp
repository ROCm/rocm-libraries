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

#include <algorithm>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare_tensor.hpp"
#include "framework/config_param.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/spatter_ref.hpp"

using namespace rpptest;

namespace {

// spatter samples its texture at an offset drawn from an unseedable RNG on both backends, and the
// texture itself is a baked constant pair in a private, non-installed header. No pointwise golden
// is available at any call shape the suite can build standalone, so the op is covered entirely by
// properties that hold whatever offset was drawn. The two checks take disjoint parameter sets, so
// the check is an axis of the grid rather than a separate TEST_P body.
enum class Check { Identity, ChannelBand };

struct SpatterParams {
    Check check;
    std::string name() const {
        switch (check) {
            case Check::Identity: return "Identity";
            case Check::ChannelBand: return "ChannelBand";
        }
        return "UNK";
    }
};

// Identity needs a colour the source can be filled with exactly; the other two want all three
// components distinct so a channel mix-up cannot pass by coincidence.
constexpr RpptRGB kGreyColor{160, 160, 160};
constexpr RpptRGB kAsymmetricColor{200, 90, 20};

// The blend is one multiply-add per element, so the integer dtypes are bit-exact: a
// deviation is a rounding or clamping defect, not accumulated error. F16 is allowed one
// half-precision ulp near 1.0.
constexpr Tolerance kSpatterTolerance = tolerance(0.0, 1e-6, 1e-3);

// How far outside the src/colour band the store quantization alone can legitimately put a value:
// half an LSB for the integer dtypes, which round to nearest; nothing beyond the compare tolerance
// for the float dtypes, whose stored unit is the intensity itself.
double spatter_band_eps(DType dt) {
    return (dt == DType::U8 || dt == DType::I8) ? 0.5 : kSpatterTolerance(dt);
}

RpptDesc descriptor_for(const TestConfig& cfg) {
    const TensorShape shape{cfg.size.n, static_cast<Rpp32u>(channels_of(cfg.layout)), cfg.size.h,
                            cfg.size.w};
    return make_descriptor(shape, cfg.dtype, cfg.layout);
}

// ---- Identity: src == spatterColor is a fixed point of the blend, at any size ----------------

template <typename T>
void run_spatter_identity(const TestConfig& cfg) {
    RpptDesc desc = descriptor_for(cfg);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, desc.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u n = 0; n < desc.n; ++n) roi[n] = roiVec[n];

    // A constant image at the spatter colour. PLN1 collapses the colour to its mean, which is
    // exact here because all three components are equal.
    std::vector<T> input(count), golden(count), actual(count);
    for_each_image_element(desc, [&](Rpp32u, Rpp32u c, Rpp32u, Rpp32u, std::size_t idx) {
        input[idx] = from_double<T>(spatter_color_stored(kGreyColor, desc.c, c, cfg.dtype));
    });
    golden = input;
    spatter_identity_reference<T>(golden.data(), desc, cfg.dtype, roi.data(), XYWH, kGreyColor);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);  // define outside-ROI dst to mirror the golden

    RppHandle handle(cfg.backend, desc.n);
    ASSERT_EQ(rppt_spatter(src.ptr(), &desc, dst.ptr(), &desc, kGreyColor, roi.data(), XYWH,
                           handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(compare_roi<T>(actual.data(), golden.data(), desc, roi.data(), XYWH,
                               kSpatterTolerance(cfg.dtype)));
}

// ---- ChannelBand: the output is a convex combination of the source and the colour -------------

// mask lies in [0,1] and maskInv is its complement, so every output element must land on the
// segment between its source element and the documented spatter intensity for its channel --
// whatever texel the RNG picked. eps only absorbs the store quantization.
template <typename T>
::testing::AssertionResult check_channel_band(const T* actual, const T* src, const RpptDesc& d,
                                              DType dt, const RpptROI* roi, RpptRGB color,
                                              double eps) {
    ::testing::AssertionResult result = ::testing::AssertionSuccess();
    bool failed = false;
    for_each_roi_io(d, roi, XYWH,
                    [&](Rpp32u n, Rpp32u c, Rpp32u j, Rpp32u i, std::size_t srcIdx,
                        std::size_t dstIdx) {
                        if (failed) return;
                        const double s = to_double(src[srcIdx]);
                        const double v = spatter_color_stored(color, d.c, c, dt);
                        const double a = to_double(actual[dstIdx]);
                        const double lo = std::min(s, v) - eps;
                        const double hi = std::max(s, v) + eps;
                        if (a < lo || a > hi) {
                            failed = true;
                            result = ::testing::AssertionFailure()
                                     << "outside the src/colour band at n=" << n << " c=" << c
                                     << " row=" << j << " col=" << i << ": actual=" << a
                                     << " src=" << s << " spatterValue=" << v << " band=[" << lo
                                     << ", " << hi << "]";
                        }
                    });
    return result;
}

template <typename T>
void run_spatter_channel_band(const TestConfig& cfg) {
    RpptDesc desc = descriptor_for(cfg);  // RPP takes a non-const ptr
    const std::size_t count = element_count(desc);
    const std::size_t bytes = byte_size(desc, cfg.dtype);

    PinnedArray<RpptROI> roi(cfg.backend, desc.n);
    const std::vector<RpptROI> roiVec = make_roi(desc, cfg.roi);
    for (Rpp32u n = 0; n < desc.n; ++n) roi[n] = roiVec[n];

    std::vector<T> input(count), actual(count);
    fill_input_image<T>(input.data(), desc, cfg.dtype);

    DeviceTensor src(cfg.backend, bytes), dst(cfg.backend, bytes);
    src.write(input.data(), bytes);
    dst.write(input.data(), bytes);

    RppHandle handle(cfg.backend, desc.n);
    ASSERT_EQ(rppt_spatter(src.ptr(), &desc, dst.ptr(), &desc, kAsymmetricColor, roi.data(), XYWH,
                           handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytes);

    EXPECT_TRUE(check_channel_band<T>(actual.data(), input.data(), desc, cfg.dtype, roi.data(),
                                      kAsymmetricColor, spatter_band_eps(cfg.dtype)));
}

template <typename Fn>
void dispatch(DType dt, Fn fn) {
    switch (dt) {
        case DType::U8: fn(Rpp8u{}); break;
        case DType::F16: fn(Rpp16f{}); break;
        case DType::F32: fn(Rpp32f{}); break;
        case DType::I8: fn(Rpp8s{}); break;
        default: FAIL() << "unsupported dtype for spatter";
    }
}

std::vector<WithParams<SpatterParams>> spatter_configs() {
    return with_params<SpatterParams>(
        make_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                     {Layout::PKD3, Layout::PLN3, Layout::PLN1}, {Roi::Full, Roi::Partial}),
        {SpatterParams{Check::Identity}, SpatterParams{Check::ChannelBand}});
}

}  // namespace

// Full name:
// Image_Effects/SpatterTest.Correctness/<Backend>_<DType>to<DType>_<Layout>_<Roi>_<Size>_<Check>
class SpatterTest : public SkipListTest<WithParams<SpatterParams>> {};

TEST_P(SpatterTest, Correctness) {
    const auto& p = GetParam();
    const TestConfig& cfg = p.cfg;
    switch (p.op.check) {
        case Check::Identity:
            dispatch(cfg.dtype, [&](auto tag) { run_spatter_identity<decltype(tag)>(cfg); });
            break;
        case Check::ChannelBand:
            dispatch(cfg.dtype, [&](auto tag) { run_spatter_channel_band<decltype(tag)>(cfg); });
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(Image_Effects, SpatterTest, ::testing::ValuesIn(spatter_configs()),
                         op_config_name<SpatterParams>);
