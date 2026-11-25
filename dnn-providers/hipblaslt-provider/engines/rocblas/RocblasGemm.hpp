// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <rocblas/rocblas.h>

#include "common/GemmCall.hpp"

namespace hipblaslt_plugin::rocblas_backend
{

/// The current device's gcnArchName (e.g. "gfx1100:xnack-"), read once per device.
std::string currentGcnArchName();

/// rocBLAS's name for an element type this provider's GEMMs use.
rocblas_datatype rocblasDataType(hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

/**
 * @brief Lets fp32 GEMMs on a handle run as XF32 for the guard's lifetime, then restores the
 *        handle's math mode.
 *
 * Where rocBLAS has no XF32 kernels (anything but gfx942) it leaves the mode unchanged, so the
 * GEMMs run fp32: the fallback TF32 asks for. Setting the mode queries the device, so a plan
 * holds one guard across its calls rather than one per call.
 */
class ScopedXf32
{
public:
    explicit ScopedXf32(rocblas_handle handle);
    ~ScopedXf32();

    ScopedXf32(const ScopedXf32&) = delete;
    ScopedXf32& operator=(const ScopedXf32&) = delete;
    ScopedXf32(ScopedXf32&&) = delete;
    ScopedXf32& operator=(ScopedXf32&&) = delete;

private:
    rocblas_handle _handle;
    rocblas_math_mode _previous = rocblas_default_math;
    bool _set = false;
};

/**
 * @brief Sets @p handle's atomics mode to rocblas_atomics_not_allowed for the guard's lifetime,
 *        then restores the previous mode.
 *
 * rocBLAS allows atomics by default, and a kernel that adds partial results with atomics is not
 * bitwise reproducible; with atomics not allowed rocBLAS gives the same bits on every run
 * (rocblas_set_atomics_mode). The guard costs the speed of the atomic kernels where rocBLAS
 * would have picked one.
 */
class ScopedNoAtomics
{
public:
    explicit ScopedNoAtomics(rocblas_handle handle);
    ~ScopedNoAtomics();

    ScopedNoAtomics(const ScopedNoAtomics&) = delete;
    ScopedNoAtomics& operator=(const ScopedNoAtomics&) = delete;
    ScopedNoAtomics(ScopedNoAtomics&&) = delete;
    ScopedNoAtomics& operator=(ScopedNoAtomics&&) = delete;

private:
    rocblas_handle _handle;
    rocblas_atomics_mode _previous = rocblas_atomics_allowed;
    bool _set = false;
};

/// Whether @p call lets its fp32 operands multiply as XF32 (GemmCall::tf32).
bool permitsXf32(const GemmCall& call);

/**
 * @brief Runs @p call through rocblas_gemm_strided_batched_ex on @p handle's current stream,
 *        accumulating in fp32 and writing the output once.
 *
 * @param a, b, d The device buffers of the call's operands.
 * @param c The addend's (GemmCall::c), or null for a call without one.
 * @throws HipdnnPluginException when rocBLAS reports a failure.
 */
void runGemm(rocblas_handle handle,
             const GemmCall& call,
             const void* a,
             const void* b,
             void* d,
             const void* c = nullptr);

} // namespace hipblaslt_plugin::rocblas_backend
