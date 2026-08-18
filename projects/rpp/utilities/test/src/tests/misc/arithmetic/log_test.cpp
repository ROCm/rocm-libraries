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
#include "framework/config_param.hpp"
#include "framework/generic_tensor_setup.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/log_ref.hpp"

using namespace rpptest;

namespace {

// Tolerances model legitimate floating-point error only: the golden computes std::log in double
// while the kernel evaluates in float (and stores half for F16).
//
// F16 is loose because half's ulp is large at the magnitudes this grid produces: the smallest
// non-zero float input is 1/255, whose log is about -5.5, and a half near 5.5 has an ulp of
// 4/1024 = 3.9e-3. The 8e-3 absolute bound is therefore ~2 ulp of the stored result. Do not
// loosen it further.
double abs_tolerance(DType out) { return out == DType::F16 ? 8e-3 : 1e-5; }
double rel_tolerance(DType out) { return out == DType::F16 ? 2e-3 : 1e-6; }

// Zero-free input fill.
//
// The header says a zero input is replaced via nextafter() to avoid an undefined result, but it
// pins neither the precision nor the direction of that nextafter: nextafterf(0.f, 1.f) = 1.4e-45
// gives log = -103.28, while the double form 4.9e-324 gives -744.44. The expected value is
// ambiguous, so the grid deliberately avoids zero inputs rather than encoding a guess for
// behaviour the API does not specify.
//
// fill_input does produce zeros -- U8 hits 0, I8 hits 0 (the value 128 maps to 0), and the [0,1]
// float fill hits exactly 0.0 -- so every zero is replaced by the smallest magnitude the dtype's
// own fill already uses (1 for U8/I8, 1/255 for F16/F32).
template <typename Tin>
void fill_input_nonzero(Tin* buf, const RpptGenericDesc& d, DType dt) {
    fill_input_nd<Tin>(buf, d, dt);
    const double replacement = (dt == DType::U8 || dt == DType::I8) ? 1.0 : 1.0 / 255.0;
    for_each_nd_coord(d, [&](const NdDims& coord) {
        Tin& v = buf[nd_offset(d, coord)];
        if (to_double(v) == 0.0) v = from_double<Tin>(replacement);
    });
}

template <typename Tin, typename Tout>
void run_log(const NdConfig& cfg) {
    const NdDims dims = nd_extents(cfg.nDim);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor srcDesc(cfg.backend, dims, cfg.dtypeIn);
    GenericDescriptor dstDesc(cfg.backend, dims, cfg.dtypeOut);

    const std::size_t count = generic_element_count(*srcDesc);
    const std::size_t srcBytes = generic_byte_size(*srcDesc, cfg.dtypeIn);
    const std::size_t dstBytes = generic_byte_size(*dstDesc, cfg.dtypeOut);

    // (1) Host golden model. The op writes every output element, so golden needs no pre-seeding.
    std::vector<Tin> input(count);
    std::vector<Tout> golden(count), actual(count);
    fill_input_nonzero<Tin>(input.data(), *srcDesc, cfg.dtypeIn);
    log_reference<Tin, Tout>(input.data(), golden.data(), *srcDesc, *dstDesc);

    // (2) The roiTensor lives in host-accessible (pinned for HIP) memory.
    const std::vector<Rpp32u> roiVec = make_nd_roi_tensor(dims);
    PinnedArray<Rpp32u> roi(cfg.backend, roiVec.size());
    for (std::size_t i = 0; i < roiVec.size(); ++i) roi[i] = roiVec[i];

    // (3) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);

    RppHandle handle(cfg.backend, dims[0]);
    ASSERT_EQ(rppt_log(src.ptr(), srcDesc.get(), dst.ptr(), dstDesc.get(), roi.data(), handle.get(),
                       cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    // (4) Compare the whole output tensor.
    EXPECT_TRUE(compare_nd<Tout>(actual.data(), golden.data(), *dstDesc, abs_tolerance(cfg.dtypeOut),
                                 rel_tolerance(cfg.dtypeOut)));
}

}  // namespace

// Full name: Misc_Arithmetic/LogTest.Correctness/<Backend>_<DTypeConv>_<Rank>_<Shape>
class LogTest : public SkipListTest<NdConfig> {};

TEST_P(LogTest, Correctness) {
    const NdConfig cfg = GetParam();
    if (cfg.dtypeIn == DType::U8 && cfg.dtypeOut == DType::F32)
        run_log<Rpp8u, Rpp32f>(cfg);
    else if (cfg.dtypeIn == DType::I8 && cfg.dtypeOut == DType::F32)
        run_log<Rpp8s, Rpp32f>(cfg);
    else if (cfg.dtypeIn == DType::F16 && cfg.dtypeOut == DType::F16)
        run_log<Rpp16f, Rpp16f>(cfg);
    else if (cfg.dtypeIn == DType::F32 && cfg.dtypeOut == DType::F32)
        run_log<Rpp32f, Rpp32f>(cfg);
    else
        FAIL() << "unsupported dtype conversion for log";
}

// The dtype conversions are exactly the set the header documents ("Supports u8->f32, i8->f32,
// f16->f16 and f32->f32"); log has no op params, so the config is a plain NdConfig.
//
// The zero input is deliberately not exercised: the header replaces it via nextafter() without
// pinning the precision or direction, so the expected value is ambiguous (log of nextafterf(0,1)
// is -103.28, of the double form -744.44). See fill_input_nonzero above.
//
// 6 of these 24 cases are red, deterministically: the I8->F32 path computes log(x + 128) -- RPP's
// image-domain I8->U8 intensity shift -- rather than the log(|x|) the header documents, so the
// whole negative half of the I8 range is wrong. The golden is deliberately left on the documented
// semantics. The other three conversions (U8->F32, F16->F16, F32->F32) are green on both backends
// at every rank.
//
// Note the ND descriptors must be device-addressable on HIP (GenericDescriptor pins them): at
// rank >= 4 the ND kernels read dims/strides on the device. Undocumented and rank-dependent.
INSTANTIATE_TEST_SUITE_P(Misc_Arithmetic, LogTest,
                         ::testing::ValuesIn(make_nd_configs({{DType::U8, DType::F32},
                                                              {DType::I8, DType::F32},
                                                              {DType::F16, DType::F16},
                                                              {DType::F32, DType::F32}},
                                                             {2, 3, 4})),
                         nd_config_param_name);
