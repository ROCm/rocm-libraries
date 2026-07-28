#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <vector>

#include "framework/generic_tensor_setup.hpp"
#include "reference/arithmetic_tensor_ref.hpp"
#include "reference/bitwise_tensor_ref.hpp"
#include "reference/concat_ref.hpp"
#include "reference/log_ref.hpp"
#include "reference/normalize_ref.hpp"
#include "reference/transpose_ref.hpp"

using namespace rpptest;

// The ND goldens must address every tensor through its descriptor's strides, never by walking the
// buffer flat, so the same logical case produces the same logical answer whether the descriptor is
// densely packed or carries row padding. That property is what lets a test choose the stride
// convention an op actually documents (slice needs padding, the other ND ops need dense -- see
// .notes/issues/generic-tensor-ops-disagree-on-stride-padding.md) without the golden silently
// following the buffer layout instead of the descriptor.
//
// Each case below computes a golden twice from identical logical input -- once dense, once padded --
// and requires the logical results to be bit-identical. No RPP op is called: this guards the
// references themselves. For a reduction (normalize) it additionally pins that padding slack stays
// out of the statistics.

namespace {

// The innermost extent is deliberately not a multiple of 8, so padding actually changes the strides.
const NdDims kDims = {2, 3, 5, 20};

int pad_axis(const NdDims& dims) { return static_cast<int>(dims.size()) - 1; }

template <typename T>
std::vector<double> logical_values(const std::vector<T>& buf, const RpptGenericDesc& d) {
    std::vector<double> out;
    out.reserve(generic_logical_count(d));
    for_each_nd_coord(
        d, [&](const NdDims& coord) { out.push_back(to_double(buf[nd_offset(d, coord)])); });
    return out;
}

// Which tensors carry padding. Mixed is the case with teeth: a golden that walks a buffer flat
// still lands on the right addresses when input and output share a layout, and only gives itself
// away when they differ -- which RpptGenericDesc allows, since src and dst are separate descriptors.
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
        normalize_reference<Rpp32f, Rpp32f>(src.data(), dst.data(), sd, dd, kDims, 1, mean.data(),
                                            stdDev.data(), 3, 1.0f, 0.0f);
        return logical_values(dst, dd);
    });
}

// The slack poison must not reach the logical elements, and a dense fill must be unchanged by the
// coordinate-addressed path -- otherwise every ND test's input would have shifted silently.
TEST(GoldenLayoutTest, FillIsLayoutIndependent) {
    RpptGenericDesc dense = make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, -1);
    RpptGenericDesc padded =
        make_generic_descriptor(kDims, DType::U8, RpptLayout::NCHW, pad_axis(kDims));
    std::vector<Rpp8u> a(generic_element_count(dense)), b(generic_element_count(padded));
    fill_input_nd<Rpp8u>(a.data(), dense, DType::U8, 0);
    fill_input_nd<Rpp8u>(b.data(), padded, DType::U8, 0);
    EXPECT_EQ(logical_values(a, dense), logical_values(b, padded));

    // A dense fill_input_nd must equal the plain fill_input it replaced.
    std::vector<Rpp8u> legacy(generic_element_count(dense));
    fill_input<Rpp8u>(legacy.data(), legacy.size(), DType::U8, 0);
    EXPECT_EQ(a, legacy);
}
