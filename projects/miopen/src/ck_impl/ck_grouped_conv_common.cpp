// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_grouped_conv_common.hpp"
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/conv_solution.hpp>

// Shared extern "C" functions used by all three direction implementations.
// These are defined once here rather than in any single direction file.

extern "C" int ckgrpconv_get_api_version() { return CK_GROUPED_CONV_API_VERSION; }

extern "C" size_t ckgrpconv_kernel_list_size(const CKKernelListHandle* h)
{
    return h ? h->kernels.size() : 0;
}

extern "C" const char* ckgrpconv_kernel_list_get(const CKKernelListHandle* h, size_t i)
{
    return (h && i < h->kernels.size()) ? h->kernels[i].c_str() : nullptr;
}

extern "C" void ckgrpconv_kernel_list_free(CKKernelListHandle* h) { delete h; }

extern "C" void ckgrpconv_solution_free(miopen::solver::ConvSolution* s) { delete s; }
