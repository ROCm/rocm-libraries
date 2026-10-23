// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

namespace hip_kernel_provider_common
{

namespace config
{
// Const char* const so that tidy recognizes them as global constants
const char* const BFLOAT16 = "bf16";
const char* const HALF = "fp16";
const char* const FLOAT = "fp32";
const char* const FP8BF16 = "fp8bf16"; // fp8 (E4M3) inputs, bf16 output
}
}
