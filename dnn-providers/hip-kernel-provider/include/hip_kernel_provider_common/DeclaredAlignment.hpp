// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

/**
 * @file DeclaredAlignment.hpp
 * @brief The one place an engine reads a tensor's declared byte alignment.
 *
 * `TensorAttributes::alignment()` is the byte alignment of the tensor's device POINTER
 * (default 16), and the backend rejects an execute whose pointer address is not a multiple
 * of it (BAD_PARAM). It is a multiple guarantee: a declared 16 guarantees every multiple
 * of 16 divides the address (16, 8, 4, 2, 1), a declared 4 guarantees only 4, 2 and 1, and
 * a declared 24 guarantees 8 but not 16. A kernel whose vector accesses need N-byte
 * alignment is therefore served only by a declared alignment that N divides, which is
 * not the same test as `declared >= N`.
 *
 * Each engine states its per-operand need as a named constant beside its matcher and passes
 * it here; the arithmetic lives once so the engines cannot drift apart on it.
 */
namespace hip_kernel_provider_common
{

/**
 * @brief Whether a declared alignment guarantees an address that is a multiple of @p needBytes.
 *
 * A declared alignment that is not positive guarantees nothing (the backend does not enforce
 * one either), so it never satisfies a need; neither does a need that is not positive.
 *
 * @param declaredBytes The alignment the tensor declares, in bytes.
 * @param needBytes The alignment the kernel's accesses need, in bytes.
 * @return True when both are positive and @p needBytes divides @p declaredBytes.
 */
inline bool declaredAlignmentCovers(int64_t declaredBytes, int64_t needBytes)
{
    return declaredBytes > 0 && needBytes > 0 && declaredBytes % needBytes == 0;
}

/**
 * @brief Whether @p tensor's declared alignment guarantees @p needBytes-byte alignment.
 *
 * @param tensor The tensor whose `alignment()` is read.
 * @param needBytes The alignment the kernel's accesses need, in bytes.
 * @return `declaredAlignmentCovers(tensor.alignment(), needBytes)`.
 */
inline bool
    declaredAlignmentCovers(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor,
                            int64_t needBytes)
{
    return declaredAlignmentCovers(tensor.alignment(), needBytes);
}

} // namespace hip_kernel_provider_common
