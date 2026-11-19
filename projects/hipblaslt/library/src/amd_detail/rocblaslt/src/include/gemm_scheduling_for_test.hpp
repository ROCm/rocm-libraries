// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hipblaslt/hipblaslt-export.h>

// Internal test hook for the effective contraction settings.
extern "C" HIPBLASLT_EXPORT bool hipblaslt_gemm_scheduling_for_test(
    std::shared_ptr<void> gemmData, bool grouped, size_t index, int32_t* smCount, int32_t* mode);

// Reads a packed uint32 argument from a single GEMM's prepared kernels.
// initialize() must have run with kernel-argument logging enabled (TENSILE_DB=64).
extern "C" HIPBLASLT_EXPORT bool hipblaslt_gemm_argument_for_test(std::shared_ptr<void> gemmData,
                                                                  const char*           name,
                                                                  uint32_t*             value);
