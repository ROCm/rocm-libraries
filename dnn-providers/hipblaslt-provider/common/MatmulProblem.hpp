// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "common/GemmCall.hpp"

namespace hipblaslt_plugin
{

/**
 * @brief The GEMM a single-node matmul graph asks for, or nullopt when the graph is not one
 *        a strided-batched BLAS GEMM computes.
 *
 * hipDNN's matmul is row-major in its dims: a[..., M, K] x b[..., K, N] = c[..., M, N]. Each
 * matrix may be stored row-major (last stride 1) or column-major (second-last stride 1), with
 * any leading dimension. A row-major c is produced as the column-major GEMM c^T = b^T a^T, a
 * column-major c directly as c = a b.
 *
 * The batch dims of each operand either equal c's or are all 1 (broadcast); the batch dims of
 * size above 1 must collapse into one stride. a and b share an element type; c is the same or
 * FLOAT. Every tensor is real device memory, the node's compute type FLOAT, and every size and
 * stride fits 32 bits.
 */
std::optional<GemmCall>
    gemmForMatmul(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

} // namespace hipblaslt_plugin
