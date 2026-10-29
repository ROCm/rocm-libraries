// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PackedSubByteTensor.hpp>

namespace hipdnn_data_sdk::utilities
{

/// Device-side packed storage for an FP4 (E2M1) tensor: two 4-bit values per
/// byte (low nibble = even logical index, high nibble = odd), matching the
/// packed `HIP_R_4F_E2M1` device layout. This is the representation a real GPU
/// kernel expects, unlike `Tensor<fp4_e2m1>` which stores one 4-bit code per
/// byte (unpacked) for element-wise CPU access.
///
/// Use this for the GPU-side bundle of an FP4 input; keep `Tensor<fp4_e2m1>` for
/// the CPU-reference bundle. Filled with the same (seed, min, max) the two agree
/// value-for-value AT EVERY COORDINATE, because randomization here mirrors
/// `Tensor<fp4_e2m1>::fillWithRandomValues` exactly: the same RNG sequence drawn in
/// index order with bounds rounded through `fp4_e2m1` first (as `TensorBase` does),
/// each draw packed into the nibble its coordinates reach through the strides. The
/// agreement therefore survives a non-row-major layout.
///
/// Only dense (packed-stride) layouts are supported. Element-wise host access
/// (`operator()`, iteration dereference) is not provided; this type is a buffer
/// holder for the device variant pack.
using PackedFp4Tensor = PackedSubByteTensor<types::fp4_e2m1, 4>;

} // namespace hipdnn_data_sdk::utilities
