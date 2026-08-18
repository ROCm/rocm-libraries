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
#include "reference/log1p_ref.hpp"

using namespace rpptest;

namespace {

// Tolerances reflect legitimate floating-point error only: the golden evaluates log1p in
// double while the kernel works in float. Kept as helpers for consistency with the other ND
// tests, even though log1p documents a single output dtype.
double abs_tolerance(DType) { return 1e-5; }
double rel_tolerance(DType) { return 1e-6; }

template <typename Tin, typename Tout>
void run_log1p(const NdConfig& cfg) {
    const NdDims dims = nd_extents(cfg.nDim);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor srcDesc(cfg.backend, dims, cfg.dtypeIn);
    GenericDescriptor dstDesc(cfg.backend, dims, cfg.dtypeOut);

    const std::size_t count = generic_element_count(*srcDesc);
    const std::size_t srcBytes = generic_byte_size(*srcDesc, cfg.dtypeIn);
    const std::size_t dstBytes = generic_byte_size(*dstDesc, cfg.dtypeOut);

    // (1) Host golden model. The op takes the absolute value before log1p, so the standard
    // fill (which spans zero and both signs) needs no special casing.
    std::vector<Tin> input(count);
    std::vector<Tout> golden(count), actual(count);
    fill_input_nd<Tin>(input.data(), *srcDesc, cfg.dtypeIn);
    log1p_reference<Tin, Tout>(input.data(), golden.data(), *srcDesc, *dstDesc);

    // (2) The roiTensor lives in host-accessible (pinned for HIP) memory.
    const std::vector<Rpp32u> roiVec = make_nd_roi_tensor(dims);
    PinnedArray<Rpp32u> roi(cfg.backend, roiVec.size());
    for (std::size_t i = 0; i < roiVec.size(); ++i) roi[i] = roiVec[i];

    // (3) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);

    RppHandle handle(cfg.backend, dims[0]);
    ASSERT_EQ(rppt_log1p(src.ptr(), srcDesc.get(), dst.ptr(), dstDesc.get(), roi.data(),
                         handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    // (4) Compare the whole output tensor.
    EXPECT_TRUE(compare_nd<Tout>(actual.data(), golden.data(), *dstDesc, abs_tolerance(cfg.dtypeOut),
                                 rel_tolerance(cfg.dtypeOut)));
}

}  // namespace

// Full name: Misc_Arithmetic/Log1pTest.Correctness/<Backend>_<DTypeConv>_<Rank>_<Shape>
class Log1pTest : public SkipListTest<NdConfig> {};

TEST_P(Log1pTest, Correctness) {
    const NdConfig cfg = GetParam();
    if (cfg.dtypeIn == DType::I16 && cfg.dtypeOut == DType::F32)
        run_log1p<Rpp16s, Rpp32f>(cfg);
    else
        FAIL() << "unsupported dtype conversion for log1p";
}

// i16->f32 is the only conversion the header documents ("Supports i16->f32 datatype"), and this
// op is why the DType axis gained I16 at all -- no other op in the suite reaches that dtype.
//
// Note the ND descriptors must be device-addressable on HIP (GenericDescriptor handles this):
// at rank >= 4 the ND kernels read dims/strides on the device. That requirement is undocumented
// and rank-dependent.
// (The DTypeConv vector is spelled out: a single-element {{a, b}} is ambiguous between the
// two make_nd_configs overloads.)
INSTANTIATE_TEST_SUITE_P(Misc_Arithmetic, Log1pTest,
                         ::testing::ValuesIn(make_nd_configs(
                             std::vector<DTypeConv>{{DType::I16, DType::F32}}, {2, 3, 4})),
                         nd_config_param_name);
