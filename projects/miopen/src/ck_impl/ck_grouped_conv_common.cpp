// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_grouped_conv_common.hpp"
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/solver/ck_grouped_conv_error.hpp>
#include <miopen/conv_solution.hpp>

// Shared extern "C" functions used by all three direction implementations.
// These are defined once here rather than in any single direction file.

extern "C" int ckgrpconv_get_api_version() { return CK_GROUPED_CONV_API_VERSION; }

extern "C" ckgrpconv_status_t ckgrpconv_kernel_list_size(const CKKernelListHandle* h,
                                                         size_t* out_size)
{
    return ckgrpconv_try_catch([&]() {
        CKGRPCONV_THROW_IF_NULL(out_size, CKGRPCONV_STATUS_BAD_PARAM, "Null out_size");
        CKGRPCONV_THROW_IF_NULL(h, CKGRPCONV_STATUS_BAD_PARAM, "Null handle");
        *out_size = h->kernels.size();
    });
}

extern "C" ckgrpconv_status_t
ckgrpconv_kernel_list_get(const CKKernelListHandle* h, size_t i, const char** out_str)
{
    return ckgrpconv_try_catch([&]() {
        CKGRPCONV_THROW_IF_NULL(out_str, CKGRPCONV_STATUS_BAD_PARAM, "Null out_str");
        CKGRPCONV_THROW_IF_NULL(h, CKGRPCONV_STATUS_BAD_PARAM, "Null handle");
        CKGRPCONV_THROW_IF_FALSE(
            i < h->kernels.size(), CKGRPCONV_STATUS_INVALID_VALUE, "Index out of range");
        *out_str = h->kernels[i].c_str();
    });
}

extern "C" void ckgrpconv_kernel_list_free(CKKernelListHandle* h) { delete h; }

extern "C" void ckgrpconv_solution_free(miopen::solver::ConvSolution* s) { delete s; }

extern "C" void ckgrpconv_get_last_error_string(const char** error_str)
{
    if(error_str != nullptr)
        *error_str = CKGrpConvLastError::getLastError();
}
