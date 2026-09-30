// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PackedSubByteTensor.hpp>

namespace hipdnn_data_sdk::utilities
{

/// Device-side packed storage for an FP6 tensor (`fp6_e2m3` or `fp6_e3m2`): a
/// dense LSB-first 6-bit bitstream, four values per three bytes, element `i` at
/// bits `[6i, 6i+6)`. `Tensor<fp6_*>` instead stores one 6-bit code per *byte*
/// (unpacked) for element-wise CPU access.
///
/// Pair the two: this type for the GPU-side bundle, `Tensor<fp6_*>` for the
/// CPU-reference bundle. Filled with the same (seed, min, max) they agree
/// value-for-value, because randomization mirrors `Tensor<T>::fillWithRandomValues`
/// exactly, each draw packed at the offset its coordinates reach through the
/// strides — so the agreement holds for non-row-major layouts too.
///
/// Only dense (packed-stride) layouts are supported. Element-wise host access is
/// not provided.
///
/// Only `fp6_e2m3` and `fp6_e3m2` are supported; instantiating for any other `T`
/// fails to compile because no `PackedElementTraits<T>` specialization exists for
/// it (declared-but-undefined primary template).
template <typename T>
using PackedFp6Tensor = PackedSubByteTensor<T, 6>;

} // namespace hipdnn_data_sdk::utilities
