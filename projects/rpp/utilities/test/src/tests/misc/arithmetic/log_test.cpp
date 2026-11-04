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
#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/log_ref.hpp"

using namespace rpptest;

namespace {

// The golden computes in double, the kernel in float (stored as half for F16). The F16 bound is
// ~2 ulp of half at the largest |log| this grid produces.
Bound log_tolerance(DType out) {
    return out == DType::F16 ? Bound{8e-3, 2e-3} : Bound{1e-5, 1e-6};
}

// The header replaces a zero input via nextafter() without specifying its precision, so log(0)
// is ambiguous. Zeros are replaced by the smallest magnitude the dtype's fill uses.
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
    const NdDims dims = nd_extents(cfg);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor srcDesc(cfg.backend, dims, cfg.dtypeIn);
    GenericDescriptor dstDesc(cfg.backend, dims, cfg.dtypeOut);

    const std::size_t count = generic_element_count(*srcDesc);
    const std::size_t srcBytes = generic_byte_size(*srcDesc, cfg.dtypeIn);
    const std::size_t dstBytes = generic_byte_size(*dstDesc, cfg.dtypeOut);

    std::vector<Tin> input(count);
    std::vector<Tout> golden(count), actual(count);
    fill_input_nonzero<Tin>(input.data(), *srcDesc, cfg.dtypeIn);
    log_reference<Tin, Tout>(input.data(), golden.data(), *srcDesc, *dstDesc);

    const std::vector<Rpp32u> roiVec = make_nd_roi_tensor(dims);
    PinnedArray<Rpp32u> roi(cfg.backend, roiVec.size());
    for (std::size_t i = 0; i < roiVec.size(); ++i) roi[i] = roiVec[i];

    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);

    RppHandle handle(cfg.backend, dims[0]);
    ASSERT_EQ(rppt_log(src.ptr(), srcDesc.get(), dst.ptr(), dstDesc.get(), roi.data(), handle.get(),
                       cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    EXPECT_TRUE(
        compare_nd<Tout>(actual.data(), golden.data(), *dstDesc, log_tolerance(cfg.dtypeOut)));
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

// The conversions the header documents. Known kernel defect (skip-listed): I8->F32 computes
// log(x + 128) instead of the documented log(|x|).
INSTANTIATE_TEST_SUITE_P(
    Misc_Arithmetic, LogTest,
    ::testing::ValuesIn(make_nd_configs({{DType::U8, DType::F32},
                                         {DType::I8, DType::F32},
                                         {DType::F16, DType::F16},
                                         {DType::F32, DType::F32}},
                                        {2, 3, 4}, {NdShape::VectorAligned, NdShape::Tail})),
    nd_config_param_name);
