// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_grouped_conv_common.hpp"
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/conv_solution.hpp>

// Shared extern "C" functions used by all three direction implementations.
// These are defined once here rather than in any single direction file.

extern "C" int ck_impl_get_api_version() { return CK_IMPL_API_VERSION; }

extern "C" ck_impl_status_t ck_impl_kernel_list_size(const CKKernelListHandle* h, size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        CK_IMPL_THROW_IF_NULL(h, CK_IMPL_STATUS_BAD_PARAM, "Null handle");
        *out_size = h->kernels.size();
    });
}

extern "C" ck_impl_status_t
ck_impl_kernel_list_get(const CKKernelListHandle* h, size_t i, const char** out_str)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_str, CK_IMPL_STATUS_BAD_PARAM, "Null out_str");
        CK_IMPL_THROW_IF_NULL(h, CK_IMPL_STATUS_BAD_PARAM, "Null handle");
        CK_IMPL_THROW_IF_FALSE(
            i < h->kernels.size(), CK_IMPL_STATUS_INVALID_VALUE, "Index out of range");
        *out_str = h->kernels[i].c_str();
    });
}

extern "C" void ck_impl_kernel_list_free(CKKernelListHandle* h) { delete h; }

extern "C" void ck_impl_solution_free(miopen::solver::ConvSolution* s) { delete s; }

extern "C" void ck_impl_get_last_error_string(const char** error_str)
{
    if(error_str != nullptr)
        *error_str = CkImplLastError::getLastError();
}
