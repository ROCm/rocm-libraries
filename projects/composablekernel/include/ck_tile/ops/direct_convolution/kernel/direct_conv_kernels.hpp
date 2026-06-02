// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Umbrella header aggregating all direct convolution kernel families.
// Each per-kernel-family header defines its variant accessor structs and
// concrete CRTP kernel wrappers, and includes only the impl headers it needs.
// Consumers that depend on a single kernel family should include the matching
// per-family header directly to minimize compile dependencies.

#include "ck_tile/ops/direct_convolution/kernel/direct_conv_4c.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_8c.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_16c.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_32c.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_32c_dense.hpp"
