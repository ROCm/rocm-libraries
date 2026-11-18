// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "status.hpp"

#include <hipblaslt/hipblaslt.h>

#include <cstdint>

namespace hipblaslt_py
{

    // Raise HipblasLtError unless a HIP device is visible. hipblasLtCreate calls exit() when
    // there's no device (ROCM-33629), which would end the Python process with no error.
    void require_device();

    class Handle
    {
    public:
        Handle();
        ~Handle()
        {
            close();
        }
        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;

        void close() noexcept;
        bool closed() const
        {
            return h_ == nullptr;
        }
        // Raises HipblasLtError on a closed handle, so a call can't reach hipBLASLt with it.
        hipblasLtHandle_t raw() const;

    private:
        hipblasLtHandle_t h_ = nullptr;
    };

    class MatmulDesc
    {
    public:
        MatmulDesc(hipblasComputeType_t compute_type, hipDataType scale_type);
        ~MatmulDesc();
        MatmulDesc(const MatmulDesc&)                      = delete;
        MatmulDesc&           operator=(const MatmulDesc&) = delete;
        hipblasLtMatmulDesc_t raw() const
        {
            return d_;
        }

    private:
        hipblasLtMatmulDesc_t d_ = nullptr;
    };

    class MatrixLayout
    {
    public:
        MatrixLayout(hipDataType dtype, uint64_t rows, uint64_t cols, int64_t ld);
        ~MatrixLayout();
        MatrixLayout(const MatrixLayout&)                      = delete;
        MatrixLayout&           operator=(const MatrixLayout&) = delete;
        hipblasLtMatrixLayout_t raw() const
        {
            return l_;
        }

    private:
        hipblasLtMatrixLayout_t l_ = nullptr;
    };

    class Preference
    {
    public:
        Preference();
        ~Preference();
        Preference(const Preference&)                            = delete;
        Preference&                 operator=(const Preference&) = delete;
        hipblasLtMatmulPreference_t raw() const
        {
            return p_;
        }

    private:
        hipblasLtMatmulPreference_t p_ = nullptr;
    };

} // namespace hipblaslt_py
