// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{
namespace detail
{

// Shared argument struct — single definition used by both host and device (HipRTC).
#include <GpuRefValidatorArgs.h> // NOLINT(misc-include-cleaner)

std::vector<std::string> buildValidatorDefines(const char* dataType, const char* computeType);

void launchValidatorKernel(hipFunction_t function, int64_t totalElements, ValidatorArgs& args);

} // namespace detail
} // namespace hipdnn_gpu_ref
