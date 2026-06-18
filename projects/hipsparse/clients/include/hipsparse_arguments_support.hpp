/*! \file */
/* ************************************************************************
* Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */
#pragma once

#include <vector>

#include <hipsparse.h>

#include "cusparse_routine_support.hpp"

// Compile-time CUDA version integer, or -1 when building against ROCm.
// Passed to CusparseRoutineSupport so it can select the correct YAML entry.
#if defined(CUDART_VERSION)
#define HIPSPARSE_CUDA_VER (CUDART_VERSION)
#else
#define HIPSPARSE_CUDA_VER (-1)
#endif

// Algorithm-support structs.  All version-specific data lives in
// cusparse_support.yaml; these wrappers simply forward to the runtime loader.
// HIPSPARSE_CUDA_VER is -1 on the ROCm backend (triggering the 'rocm' entry
// in the YAML) and CUDART_VERSION on CUDA builds.

struct csr2csc_alg_support
{
    static hipsparseCsr2CscAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseCsr2CscAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("csr2csc", HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("csr2csc",
                                                                            HIPSPARSE_CUDA_VER);
    }
    static std::vector<int> get_supported_algorithms()
    {
        return CusparseRoutineSupport::instance().get_algorithm_supported_values("csr2csc",
                                                                                 HIPSPARSE_CUDA_VER);
    }
};

struct dense2sparse_alg_support
{
    static hipsparseDenseToSparseAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseDenseToSparseAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("dense2sparse",
                                                                     HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("dense2sparse",
                                                                            HIPSPARSE_CUDA_VER);
    }
};

struct sparse2dense_alg_support
{
    static hipsparseSparseToDenseAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseSparseToDenseAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("sparse2dense",
                                                                     HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("sparse2dense",
                                                                            HIPSPARSE_CUDA_VER);
    }
};

struct sddmm_alg_support
{
    static hipsparseSDDMMAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseSDDMMAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("sddmm", HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("sddmm",
                                                                            HIPSPARSE_CUDA_VER);
    }
};

struct spgemm_alg_support
{
    static hipsparseSpGEMMAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseSpGEMMAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("spgemm", HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("spgemm",
                                                                            HIPSPARSE_CUDA_VER);
    }
};

struct spmm_alg_support
{
    static hipsparseSpMMAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseSpMMAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("spmm", HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("spmm",
                                                                            HIPSPARSE_CUDA_VER);
    }
};

struct spmv_alg_support
{
    static hipsparseSpMVAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseSpMVAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("spmv", HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("spmv",
                                                                            HIPSPARSE_CUDA_VER);
    }
};

struct spsm_alg_support
{
    static hipsparseSpSMAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseSpSMAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("spsm", HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("spsm",
                                                                            HIPSPARSE_CUDA_VER);
    }
};

struct spsv_alg_support
{
    static hipsparseSpSVAlg_t get_default_algorithm()
    {
        return static_cast<hipsparseSpSVAlg_t>(
            CusparseRoutineSupport::instance().get_algorithm_default("spsv", HIPSPARSE_CUDA_VER));
    }
    static std::string get_description()
    {
        return CusparseRoutineSupport::instance().get_algorithm_description("spsv",
                                                                            HIPSPARSE_CUDA_VER);
    }
};
