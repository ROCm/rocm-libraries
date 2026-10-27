// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

/**
 * @file PackAlignment.hpp
 * @brief The one place a native pack's matcher reads a tensor's declared byte alignment.
 *
 * `TensorAttributes::alignment()` is the byte alignment of the tensor's device POINTER
 * (default 16), and the backend rejects an execute whose pointer address is not a multiple
 * of it (BAD_PARAM). It is a multiple guarantee: a declared 16 guarantees every multiple
 * of 16 divides the address (16, 8, 4, 2, 1), a declared 4 guarantees only 4, 2 and 1, and
 * a declared 24 guarantees 8 but not 16. A kernel whose vector accesses need N-byte
 * alignment is therefore served only by a declared alignment that N divides, which is
 * not the same test as `declared >= N`.
 *
 * Each pack states its per-operand need as a named constant beside its matcher and passes
 * it here; the arithmetic lives once so the packs cannot drift apart on it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::pack_alignment
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Whether @p tensor's declared alignment guarantees an address that is a multiple of
/// @p needBytes. A declared alignment that is not positive guarantees nothing (the backend
/// does not enforce one either), so it never satisfies a need.
inline bool guaranteesAlignment(const data_objects::TensorAttributes& tensor, int64_t needBytes)
{
    const int64_t declared = tensor.alignment();
    return needBytes > 0 && declared > 0 && declared % needBytes == 0;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::pack_alignment
