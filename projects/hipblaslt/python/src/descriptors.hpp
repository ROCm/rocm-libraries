// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <nanobind/nanobind.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include "status.hpp"

namespace hipblaslt_py {

class Handle
{
public:
    Handle() { HIPBLASLT_CHECK(hipblasLtCreate(&h_)); }
    ~Handle() { close(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    void close()
    {
        if(h_)
        {
            hipblasLtDestroy(h_);  // best-effort in destructor path
            h_ = nullptr;
        }
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(h_); }
    hipblasLtHandle_t raw() const { return h_; }

private:
    hipblasLtHandle_t h_ = nullptr;
};

class MatrixLayout
{
public:
    MatrixLayout(hipDataType dtype, uint64_t rows, uint64_t cols, int64_t ld)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&l_, dtype, rows, cols, ld));
    }
    ~MatrixLayout() { if(l_) hipblasLtMatrixLayoutDestroy(l_); }
    MatrixLayout(const MatrixLayout&) = delete;
    MatrixLayout& operator=(const MatrixLayout&) = delete;

    void set_attribute(hipblasLtMatrixLayoutAttribute_t attr, int32_t value)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(l_, attr, &value, sizeof(value)));
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(l_); }
    hipblasLtMatrixLayout_t raw() const { return l_; }

private:
    hipblasLtMatrixLayout_t l_ = nullptr;
};

class MatmulDesc
{
public:
    MatmulDesc(hipblasComputeType_t compute, hipDataType scale)
    {
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&d_, compute, scale));
    }
    ~MatmulDesc() { if(d_) hipblasLtMatmulDescDestroy(d_); }
    MatmulDesc(const MatmulDesc&) = delete;
    MatmulDesc& operator=(const MatmulDesc&) = delete;

    void set_attribute_int(hipblasLtMatmulDescAttributes_t attr, int32_t value)
    {
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(d_, attr, &value, sizeof(value)));
    }
    void set_attribute_ptr(hipblasLtMatmulDescAttributes_t attr, std::uintptr_t p)
    {
        void* raw = reinterpret_cast<void*>(p);
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(d_, attr, &raw, sizeof(raw)));
    }
    int32_t get_attribute_int(hipblasLtMatmulDescAttributes_t attr)
    {
        int32_t value = 0;
        size_t written = 0;
        HIPBLASLT_CHECK(hipblasLtMatmulDescGetAttribute(d_, attr, &value, sizeof(value), &written));
        return value;
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(d_); }
    hipblasLtMatmulDesc_t raw() const { return d_; }

private:
    hipblasLtMatmulDesc_t d_ = nullptr;
};

class Preference
{
public:
    Preference() { HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&p_)); }
    ~Preference() { if(p_) hipblasLtMatmulPreferenceDestroy(p_); }
    Preference(const Preference&) = delete;
    Preference& operator=(const Preference&) = delete;

    void set_max_workspace(uint64_t nbytes)
    {
        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
            p_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &nbytes, sizeof(nbytes)));
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(p_); }
    hipblasLtMatmulPreference_t raw() const { return p_; }

private:
    hipblasLtMatmulPreference_t p_ = nullptr;
};

} // namespace hipblaslt_py
