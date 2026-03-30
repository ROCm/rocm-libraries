// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: tensor name exceeds 15-char limit.
// Expected error: "tensor name too long (max 15 chars)"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr TensorName bad("1234567890123456");
