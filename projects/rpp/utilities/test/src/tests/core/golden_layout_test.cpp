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

#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/arithmetic_tensor_ref.hpp"
#include "reference/bitwise_tensor_ref.hpp"
#include "reference/box_filter_ref.hpp"
#include "reference/brightness_ref.hpp"
#include "reference/concat_ref.hpp"
#include "reference/crop_and_patch_ref.hpp"
#include "reference/hue_ref.hpp"
#include "reference/jpeg_compression_distortion_ref.hpp"
#include "reference/log1p_ref.hpp"
#include "reference/log_ref.hpp"
#include "reference/normalize_ref.hpp"
#include "reference/resize_ref.hpp"
#include "reference/rotate_ref.hpp"
#include "reference/slice_ref.hpp"
#include "reference/transpose_ref.hpp"

using namespace rpptest;

// The goldens must address tensors through their descriptor's strides, never by walking the
// buffer flat. Each case computes a golden from identical logical input under dense and padded
// descriptors and requires bit-identical logical results. No RPP op is called.
//
// The image cases target the shared traversal helpers; each op is one representative golden for
// the helper named above its case.

namespace {

// Innermost extent not a multiple of 8, so padding changes the strides.
const NdDims kDims = {2, 3, 5, 20};

int pad_axis(const NdDims& dims) {
    return static_cast<int>(dims.size()) - 1;
}

template <typename T>
std::vector<double> logical_values(const std::vector<T>& buf, const RpptGenericDesc& d) {
    std::vector<double> out;
    out.reserve(generic_logical_count(d));
    for_each_nd_coord(
        d, [&](const NdDims& coord) { out.push_back(to_double(buf[nd_offset(d, coord)])); });
    return out;
}

// Which tensors carry padding. The mixed plans catch a golden that walks the buffer flat, which
// can still agree with itself when src and dst share a layout.
struct PadPlan {
    int src, dst;
};

template <typename Fn>
void expect_layout_agnostic(Fn body) {
    const int p = pad_axis(kDims);
    const std::vector<double> dense = body(PadPlan{-1, -1});
    const struct {
        const char* name;
        PadPlan plan;
    } variants[] = {
        {"all padded", PadPlan{p, p}},
        {"src dense, dst padded", PadPlan{-1, p}},
        {"src padded, dst dense", PadPlan{p, -1}},
    };
    for (const auto& v : variants) {
        const std::vector<double> got = body(v.plan);
        ASSERT_EQ(dense.size(), got.size()) << v.name;
        for (std::size_t i = 0; i < dense.size(); ++i)
            ASSERT_EQ(dense[i], got[i])
                << "golden differs at logical element " << i << " for " << v.name;
    }
}

}  // namespace

TEST(GoldenLayoutTest, ArithmeticTensor) {
    expect_layout_agnostic([](PadPlan pp) {
        RpptGenericDesc s = make_generic_descriptor(kDims, DType::F32, RpptLayout::NCHW, pp.src);
        RpptGenericDesc o = make_generic_descriptor(kDims, DType::F32, RpptLayout::NCHW, pp.dst);
        std::vector<Rpp32f> a(generic_element_count(s)), b(generic_element_count(s)),
            out(generic_element_count(o), 0.f);
        fill_input_nd<Rpp32f>(a.data(), s, DType::F32, 0);
        fill_input_nd<Rpp32f>(b.data(), s, DType::F32, 1);
        arithmetic_tensor_reference<Rpp32f>(a.data(), b.data(), out.data(), o, s, s,
                                            ArithmeticTensorOp::Add);
        return logical_values(out, o);
    });
}

TEST(GoldenLayoutTest, BitwiseTensor) {
    expect_layout_agnostic([](PadPlan pp) {
        RpptGenericDesc s = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pp.src);
        RpptGenericDesc o = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pp.dst);
        std::vector<Rpp8u> a(generic_element_count(s)), b(generic_element_count(s)),
            out(generic_element_count(o), 0);
        fill_input_nd<Rpp8u>(a.data(), s, DType::U8, 0);
        fill_input_nd<Rpp8u>(b.data(), s, DType::U8, 1);
        bitwise_tensor_reference<Rpp8u>(a.data(), b.data(), out.data(), o, s, s,
                                        BitwiseTensorOp::And);
        return logical_values(out, o);
    });
}

TEST(GoldenLayoutTest, Log) {
    expect_layout_agnostic([](PadPlan pp) {
        RpptGenericDesc sd = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pp.src);
        RpptGenericDesc dd = make_generic_descriptor(kDims, DType::F32, RpptLayout::NCHW, pp.dst);
        std::vector<Rpp8u> src(generic_element_count(sd));
        std::vector<Rpp32f> dst(generic_element_count(dd), 0.f);
        fill_input_nd<Rpp8u>(src.data(), sd, DType::U8, 0);
        log_reference<Rpp8u, Rpp32f>(src.data(), dst.data(), sd, dd);
        return logical_values(dst, dd);
    });
}

TEST(GoldenLayoutTest, Log1p) {
    expect_layout_agnostic([](PadPlan pp) {
        RpptGenericDesc sd = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pp.src);
        RpptGenericDesc dd = make_generic_descriptor(kDims, DType::F32, RpptLayout::NCHW, pp.dst);
        std::vector<Rpp8u> src(generic_element_count(sd));
        std::vector<Rpp32f> dst(generic_element_count(dd), 0.f);
        fill_input_nd<Rpp8u>(src.data(), sd, DType::U8, 0);
        log1p_reference<Rpp8u, Rpp32f>(src.data(), dst.data(), sd, dd);
        return logical_values(dst, dd);
    });
}

// The anchor puts the slice off the source origin, so confused strides land on different data.
TEST(GoldenLayoutTest, Slice) {
    expect_layout_agnostic([](PadPlan pp) {
        const Rpp32u nDim = nd_rank(kDims);
        NdDims dstDims = kDims;
        for (Rpp32u a = 0; a < nDim; ++a) dstDims[a + 1] = kDims[a + 1] - 1;
        RpptGenericDesc sd = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pp.src);
        RpptGenericDesc dd = make_generic_descriptor(dstDims, DType::U8, RpptLayout::NCHW, pp.dst);
        std::vector<Rpp8u> src(generic_element_count(sd)), dst(generic_element_count(dd), 0);
        fill_input_nd<Rpp8u>(src.data(), sd, DType::U8, 0);

        std::vector<Rpp32s> anchor(static_cast<std::size_t>(kDims[0]) * nDim, 1);
        std::vector<Rpp32s> shape(static_cast<std::size_t>(kDims[0]) * nDim);
        for (Rpp32u n = 0; n < kDims[0]; ++n)
            for (Rpp32u a = 0; a < nDim; ++a)
                shape[n * nDim + a] = static_cast<Rpp32s>(dstDims[a + 1]);
        const std::vector<Rpp32u> roi = make_nd_roi_tensor(kDims);

        slice_reference<Rpp8u>(src.data(), dst.data(), sd, dd, anchor.data(), shape.data(),
                               roi.data(), 0.0);
        return logical_values(dst, dd);
    });
}

TEST(GoldenLayoutTest, Transpose) {
    expect_layout_agnostic([](PadPlan pp) {
        const std::vector<Rpp32u> perm = {2, 1, 0};  // reverse the per-sample axes
        RpptGenericDesc sd = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pp.src);
        const NdDims outDims = transpose_dst_dims(kDims, perm);
        RpptGenericDesc dd = make_generic_descriptor(outDims, DType::U8, RpptLayout::NCHW, pp.dst);
        std::vector<Rpp8u> src(generic_element_count(sd)), dst(generic_element_count(dd), 0);
        fill_input_nd<Rpp8u>(src.data(), sd, DType::U8, 0);
        transpose_reference<Rpp8u>(src.data(), dst.data(), sd, dd, perm.data());
        return logical_values(dst, dd);
    });
}

TEST(GoldenLayoutTest, Concat) {
    expect_layout_agnostic([](PadPlan pp) {
        NdDims dims2 = kDims;
        dims2.back() = 11;  // operands differ along the concat axis
        NdDims outDims = kDims;
        outDims.back() = kDims.back() + dims2.back();
        RpptGenericDesc s1 = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pp.src);
        RpptGenericDesc s2 = make_generic_descriptor(dims2, DType::U8, RpptLayout::NCHW, pp.src);
        RpptGenericDesc dd = make_generic_descriptor(outDims, DType::U8, RpptLayout::NCHW, pp.dst);
        std::vector<Rpp8u> a(generic_element_count(s1)), b(generic_element_count(s2)),
            out(generic_element_count(dd), 0);
        fill_input_nd<Rpp8u>(a.data(), s1, DType::U8, 0);
        fill_input_nd<Rpp8u>(b.data(), s2, DType::U8, 1);
        concat_reference<Rpp8u>(a.data(), b.data(), out.data(), dd, s1, s2, nd_rank(kDims) - 1);
        return logical_values(out, dd);
    });
}

TEST(GoldenLayoutTest, NormalizeExcludesPaddingFromStatistics) {
    expect_layout_agnostic([](PadPlan pp) {
        RpptGenericDesc sd = make_generic_descriptor(kDims, DType::F32, RpptLayout::NCHW, pp.src);
        RpptGenericDesc dd = make_generic_descriptor(kDims, DType::F32, RpptLayout::NCHW, pp.dst);
        std::vector<Rpp32f> src(generic_element_count(sd)), dst(generic_element_count(dd), 0.f);
        fill_input_nd<Rpp32f>(src.data(), sd, DType::F32, 0);
        std::vector<Rpp32f> mean(256, 0.f), stdDev(256, 0.f);
        normalize_reference<Rpp32f, Rpp32f>(src.data(), dst.data(), sd, dd, DType::F32, 1,
                                            mean.data(), stdDev.data(), 3, 1.0f, 0.0f);
        return logical_values(dst, dd);
    });
}

// The slack poison must not reach the logical elements, and a dense fill must match fill_input().
TEST(GoldenLayoutTest, FillIsLayoutIndependent) {
    RpptGenericDesc dense = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, -1);
    RpptGenericDesc padded =
        make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pad_axis(kDims));
    std::vector<Rpp8u> a(generic_element_count(dense)), b(generic_element_count(padded));
    fill_input_nd<Rpp8u>(a.data(), dense, DType::U8, 0);
    fill_input_nd<Rpp8u>(b.data(), padded, DType::U8, 0);
    EXPECT_EQ(logical_values(a, dense), logical_values(b, padded));

    std::vector<Rpp8u> legacy(generic_element_count(dense));
    fill_input<Rpp8u>(legacy.data(), legacy.size(), DType::U8, 0);
    EXPECT_EQ(a, legacy);
}

// ---- image domain ----------------------------------------------------------
//
// The same property for the RpptDesc goldens, one case per traversal shape.

namespace {

const TensorShape kShape = {2, 3, 9, 20};  // width not a multiple of 8, so padding moves the rows

template <typename T>
std::vector<double> image_logical_values(const std::vector<T>& buf, const RpptDesc& d) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(d.n) * d.c * d.h * d.w);
    for_each_image_element(d, [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t idx) {
        out.push_back(to_double(buf[idx]));
    });
    return out;
}

// Runs `body` against padded and dense descriptors built from the same logical input, and requires
// the logical outputs to be identical. PartialRoi so the ROI offset is non-zero.
template <typename Fn>
void expect_image_layout_agnostic(Layout layout, Fn body) {
    std::vector<double> reference;
    for (bool pad : {true, false}) {
        const RpptDesc d = make_descriptor(kShape, DType::U8, layout, pad);
        const std::vector<RpptROI> roi = make_roi(d, Roi::Partial);
        std::vector<Rpp8u> src1(element_count(d)), src2(element_count(d)), dst(element_count(d), 0);
        fill_input_image<Rpp8u>(src1.data(), d, DType::U8, 0);
        fill_input_image<Rpp8u>(src2.data(), d, DType::U8, 1);
        body(src1, src2, dst, d, roi.data());
        const std::vector<double> got = image_logical_values(dst, d);
        if (pad) {
            reference = got;
            continue;
        }
        ASSERT_EQ(reference.size(), got.size());
        for (std::size_t i = 0; i < got.size(); ++i)
            ASSERT_EQ(reference[i], got[i]) << "golden differs at logical element " << i;
    }
}

}  // namespace

// for_each_roi_io: source read at the ROI offset, output packed at the destination origin.
TEST(GoldenLayoutTest, ImagePointwise) {
    expect_image_layout_agnostic(
        Layout::PKD3, [](const std::vector<Rpp8u>& s1, const std::vector<Rpp8u>&,
                         std::vector<Rpp8u>& o, const RpptDesc& d, const RpptROI* roi) {
            brightness_reference<Rpp8u>(s1.data(), d, o.data(), d, DType::U8, roi, XYWH, 1.4, 12.0);
        });
}

// for_each_roi_pixel: a whole pixel's channels reached through cStride.
TEST(GoldenLayoutTest, ImagePerPixel) {
    expect_image_layout_agnostic(
        Layout::PKD3, [](const std::vector<Rpp8u>& s1, const std::vector<Rpp8u>&,
                         std::vector<Rpp8u>& o, const RpptDesc& d, const RpptROI* roi) {
            hue_reference<Rpp8u>(s1.data(), d, o.data(), d, DType::U8, roi, XYWH, 45.0);
        });
}

// filter_reference: a KxK neighbourhood gathered around each output pixel.
TEST(GoldenLayoutTest, ImageWindowFilter) {
    expect_image_layout_agnostic(
        Layout::PLN3, [](const std::vector<Rpp8u>& s1, const std::vector<Rpp8u>&,
                         std::vector<Rpp8u>& o, const RpptDesc& d, const RpptROI* roi) {
            box_filter_reference<Rpp8u>(s1.data(), d, o.data(), d, DType::U8, roi, XYWH, 3);
        });
}

// geometric_reference: fractional source coordinates resolved through the shared sampler.
TEST(GoldenLayoutTest, ImageGeometric) {
    const std::vector<Rpp32f> angle(kShape.n, 17.0f);
    expect_image_layout_agnostic(Layout::PLN3,
                                 [&](const std::vector<Rpp8u>& s1, const std::vector<Rpp8u>&,
                                     std::vector<Rpp8u>& o, const RpptDesc& d, const RpptROI* roi) {
                                     rotate_reference<Rpp8u>(s1.data(), o.data(), d, DType::U8, roi,
                                                             XYWH, angle.data(), BILINEAR);
                                 });
}

// Two independent descriptors: the source and destination strides must not be interchangeable.
TEST(GoldenLayoutTest, ImageResize) {
    const std::vector<RpptImagePatch> sizes(kShape.n, {kShape.w, kShape.h});
    expect_image_layout_agnostic(Layout::PKD3,
                                 [&](const std::vector<Rpp8u>& s1, const std::vector<Rpp8u>&,
                                     std::vector<Rpp8u>& o, const RpptDesc& d, const RpptROI* roi) {
                                     resize_reference<Rpp8u>(s1.data(), d, o.data(), d, DType::U8,
                                                             roi, XYWH, sizes.data(), BILINEAR);
                                 });
}

// Whole-frame placement (absolute coordinates) rather than the ROI-relative walk.
TEST(GoldenLayoutTest, ImageCropAndPatch) {
    std::vector<RpptROI> crop(kShape.n), patch(kShape.n);
    for (Rpp32u n = 0; n < kShape.n; ++n) {
        crop[n].xywhROI = {{0, 0}, 6, 4};
        patch[n].xywhROI = {{3, 2}, 6, 4};
    }
    expect_image_layout_agnostic(
        Layout::PKD3, [&](const std::vector<Rpp8u>& s1, const std::vector<Rpp8u>& s2,
                          std::vector<Rpp8u>& o, const RpptDesc& d, const RpptROI* roi) {
            crop_and_patch_reference<Rpp8u>(s1.data(), s2.data(), o.data(), d, roi, crop.data(),
                                            patch.data(), XYWH);
        });
}

// A dense per-plane intermediate, resampled (4:2:0 chroma) and scattered back at the ROI: the
// descriptor is read on the way in and out with an unstrided buffer in between.
TEST(GoldenLayoutTest, ImageBlockTransform) {
    expect_image_layout_agnostic(Layout::PLN3,
                                 [&](const std::vector<Rpp8u>& s1, const std::vector<Rpp8u>&,
                                     std::vector<Rpp8u>& o, const RpptDesc& d, const RpptROI* roi) {
                                     jpeg_compression_distortion_reference<Rpp8u>(
                                         s1.data(), o.data(), d, DType::U8, roi, XYWH, 50);
                                 });
}

// The input the cases above compare against must itself be layout-independent, or a golden could
// pass by being fed different data twice.
TEST(GoldenLayoutTest, ImageFillIsLayoutIndependent) {
    for (Layout layout : {Layout::PKD3, Layout::PLN3, Layout::PLN1}) {
        const RpptDesc padded = make_descriptor(kShape, DType::U8, layout, true);
        const RpptDesc dense = make_descriptor(kShape, DType::U8, layout, false);
        std::vector<Rpp8u> a(element_count(padded)), b(element_count(dense));
        fill_input_image<Rpp8u>(a.data(), padded, DType::U8, 0);
        fill_input_image<Rpp8u>(b.data(), dense, DType::U8, 0);
        EXPECT_EQ(image_logical_values(a, padded), image_logical_values(b, dense))
            << layout_name(layout);
    }

    // (n, c, y, x) is memory order only for a dense planar descriptor, so only there does the
    // fill match fill_input() byte for byte.
    const RpptDesc dense = make_descriptor(kShape, DType::U8, Layout::PLN3, false);
    std::vector<Rpp8u> planar(element_count(dense)), legacy(element_count(dense));
    fill_input_image<Rpp8u>(planar.data(), dense, DType::U8, 0);
    fill_input<Rpp8u>(legacy.data(), legacy.size(), DType::U8, 0);
    EXPECT_EQ(planar, legacy);
}
