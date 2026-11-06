// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocblas/rocblas.h>

#include "common/GemmCall.hpp"

namespace hipblaslt_plugin::rocblas_backend
{

/// rocBLAS's name for an element type this provider's GEMMs use.
rocblas_datatype rocblasDataType(hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

/**
 * @brief Runs @p call through rocblas_gemm_strided_batched_ex on @p handle's current stream,
 *        accumulating in fp32 and writing the output once.
 *
 * @param a, b, d The device buffers of the call's operands.
 * @throws HipdnnPluginException when rocBLAS reports a failure.
 */
void runGemm(rocblas_handle handle, const GemmCall& call, const void* a, const void* b, void* d);

} // namespace hipblaslt_plugin::rocblas_backend
