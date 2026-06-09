/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
//
// Full definition of hipblaslt_gemm_cache_entry. Include this only from .cpp
// files that have already included <hipblaslt/hipblaslt.h>. Public consumers
// should include <miopen/hipblaslt_gemm.hpp> instead, which only forward-
// declares the entry and exposes the cache.
//
#ifndef GUARD_MIOPEN_HIPBLASLT_GEMM_IMPL_HPP_
#define GUARD_MIOPEN_HIPBLASLT_GEMM_IMPL_HPP_

#include <miopen/hipblaslt_gemm.hpp>

#if MIOPEN_USE_HIPBLASLT

#include <hipblaslt/hipblaslt.h>

namespace miopen {

struct hipblaslt_gemm_cache_entry
{
    hipblasLtMatrixLayout_t matA     = nullptr;
    hipblasLtMatrixLayout_t matB     = nullptr;
    hipblasLtMatrixLayout_t matC     = nullptr;
    hipblasLtMatrixLayout_t matD     = nullptr;
    hipblasLtMatmulDesc_t matmul     = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulAlgo_t algo{};

    hipblaslt_gemm_cache_entry()                                             = default;
    hipblaslt_gemm_cache_entry(const hipblaslt_gemm_cache_entry&)            = delete;
    hipblaslt_gemm_cache_entry& operator=(const hipblaslt_gemm_cache_entry&) = delete;
    hipblaslt_gemm_cache_entry(hipblaslt_gemm_cache_entry&&)                 = delete;
    hipblaslt_gemm_cache_entry& operator=(hipblaslt_gemm_cache_entry&&)      = delete;

    // The hipBLASLt destroy calls return a status that is intentionally
    // discarded - matching the prior stack-RAII helper this replaces. They do
    // not throw, so the destructor stays noexcept by default.
    ~hipblaslt_gemm_cache_entry() noexcept
    {
        if(matA != nullptr)
            (void)hipblasLtMatrixLayoutDestroy(matA);
        if(matB != nullptr)
            (void)hipblasLtMatrixLayoutDestroy(matB);
        if(matC != nullptr)
            (void)hipblasLtMatrixLayoutDestroy(matC);
        if(matD != nullptr)
            (void)hipblasLtMatrixLayoutDestroy(matD);
        if(matmul != nullptr)
            (void)hipblasLtMatmulDescDestroy(matmul);
        if(pref != nullptr)
            (void)hipblasLtMatmulPreferenceDestroy(pref);
    }
};

} // namespace miopen

#endif // MIOPEN_USE_HIPBLASLT
#endif // GUARD_MIOPEN_HIPBLASLT_GEMM_IMPL_HPP_
