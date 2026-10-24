// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compile-only: the shim umbrella under HIPDNN_FRONTEND_SKIP_JSON_LIB, which the
// shim maps onto CUDNN_FRONTEND_SKIP_JSON_LIB, so every v0.x describe() takes
// its non-JSON branch.
#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#error "TestCudnnShimV8SkipHipdnnJson.cpp must be built with HIPDNN_FRONTEND_SKIP_JSON_LIB"
#endif

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <string>

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
#error "the shim must map HIPDNN_FRONTEND_SKIP_JSON_LIB onto CUDNN_FRONTEND_SKIP_JSON_LIB"
#endif

// Redefines `json` as a different kind of symbol if the shim's global
// `using json` leaks past its JSON gate.
// NOLINTNEXTLINE(readability-identifier-naming)
[[maybe_unused]] static int json = 0;

namespace hipdnn_shim_test
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// External linkage and qualified (non-virtual) calls force every describe()
// body to be emitted, not just parsed.
// NOLINTNEXTLINE(misc-use-internal-linkage)
std::string describeEveryV8DescriptorWithoutHipdnnJson(const fe::Tensor_v8& tensor,
                                                       const fe::ConvDesc_v8& conv,
                                                       const fe::PointWiseDesc_v8& pointwise,
                                                       const fe::MatMulDesc_v8& matmul,
                                                       const fe::ReductionDesc_v8& reduction,
                                                       const fe::ResampleDesc_v8& resample,
                                                       const fe::RngDesc_v8& rng,
                                                       const fe::VariantPack_v8& variantPack)
{
    return tensor.Tensor_v8::describe() + conv.ConvDesc_v8::describe()
           + pointwise.PointWiseDesc_v8::describe() + matmul.MatMulDesc_v8::describe()
           + reduction.ReductionDesc_v8::describe() + resample.ResampleDesc_v8::describe()
           + rng.RngDesc_v8::describe() + variantPack.VariantPack_v8::describe();
}

} // namespace hipdnn_shim_test
