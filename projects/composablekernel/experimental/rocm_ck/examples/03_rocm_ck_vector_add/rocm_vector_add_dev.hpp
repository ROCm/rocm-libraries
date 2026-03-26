// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side kernel for rocm_ck vector add. Uses CK Tile tile primitives
// directly to compute c = alpha * a + beta * b with mixed input/output types.
//
// DEVICE ONLY: compiled via --cuda-device-only in .hip files.
// This header must NOT be included from host-only .cpp files.
//
// Compilation boundary:
//   _kernel.hpp — schema types + consteval factory (both passes)
//   _api.hpp    — host-only helpers (host pass only, #error on device)
//   _dev.hpp (this) — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#error \
    "rocm_vector_add_dev.hpp requires device compilation. Host code should include rocm_vector_add_api.hpp."
#endif

#include "rocm_vector_add_kernel.hpp"

#include <rocm_ck/ck_type_map.hpp>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise.hpp"

namespace rocm_ck {

/// Device function that computes c = alpha * a + beta * b.
///
/// Tensor layout:
///   tensors[0] = a (input),  tensors[1] = b (input),  tensors[2] = c (output)
///   lengths[0] = n,  strides[0] = 1  (contiguous rank-1)
///
/// Scalar layout:
///   scalars[0].f32 = alpha,  scalars[1].f32 = beta
///
/// Call this from an extern "C" __global__ wrapper.
template <VectorAddKernel K>
__device__ void runVectorAdd(Args args)
{
    // Device-side validation — catches invalid manual construction.
    static_assert(K.block_tile > 0, "block_tile must be positive");
    static_assert(K.thread_block_size > 0, "thread_block_size must be positive");
    static_assert(K.block_warps > 0, "block_warps must be positive");

    using X = typename CkTypeMap<K.in_dtype>::type;
    using Y = typename CkTypeMap<K.out_dtype>::type;

    // Use the wider type for ElementWiseShape so kVectorM is valid for both
    // input loads and output stores.
    using WiderType = std::conditional_t<(sizeof(X) >= sizeof(Y)), X, Y>;
    using Shape     = ck_tile::ElementWiseShape<ck_tile::sequence<K.block_warps>,
                                                ck_tile::sequence<K.block_tile>,
                                                ck_tile::sequence<K.warp_tile>,
                                                WiderType>;

    static_assert(sizeof(rocm_ck::index_t) == sizeof(ck_tile::index_t),
                  "rocm_ck::index_t and ck_tile::index_t must match");

    // Unpack generic Args — compiler generates s_load at fixed offsets.
    const TensorArg& t_a = args.tensors[0];
    const TensorArg& t_b = args.tensors[1];
    const TensorArg& t_c = args.tensors[2];

    const auto n       = static_cast<ck_tile::index_t>(t_a.lengths[0]);
    const auto iM      = ck_tile::get_block_id() * Shape::kBlockM;
    const auto lens    = ck_tile::make_tuple(n);
    const auto strides = ck_tile::make_tuple(ck_tile::index_t{1});

    // Tile distribution — same encoding as ElementWiseDefaultPolicy.
    constexpr auto dist = ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<Shape::kRepeatM,
                                             Shape::kWarpPerBlockM,
                                             Shape::kThreadPerWarpM,
                                             Shape::kVectorM>>,
            ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1>>,
            ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<2>>,
            ck_tile::sequence<1, 1>,
            ck_tile::sequence<0, 3>>{});

    const auto merge_transform = ck_tile::make_merge_transform(lens);

    // Helper: create a padded, transformed tile window for a global input pointer.
    auto make_input_window = [&](const X* ptr) {
        const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            ptr, lens, strides, ck_tile::number<Shape::kVectorM>{}, ck_tile::number<1>{});

        const auto transformed = ck_tile::pad_tensor_view(
            ck_tile::transform_tensor_view(view,
                                           ck_tile::make_tuple(merge_transform),
                                           ck_tile::make_tuple(ck_tile::make_index_sequence<1>{}),
                                           ck_tile::make_tuple(ck_tile::sequence<0>{})),
            ck_tile::make_tuple(ck_tile::number<Shape::kBlockM>{}),
            ck_tile::sequence<K.pad>{});

        return ck_tile::make_tile_window(
            transformed, ck_tile::make_tuple(ck_tile::number<Shape::kBlockM>{}), {iM}, dist);
    };

    // Load input tiles.
    auto a_tile = ck_tile::load_tile(make_input_window(static_cast<const X*>(t_a.ptr)));
    auto b_tile = ck_tile::load_tile(make_input_window(static_cast<const X*>(t_b.ptr)));

    // Compute: y = alpha * a + beta * b (in float, then cast to Y).
    auto y_tile = ck_tile::make_static_distributed_tensor<Y>(a_tile.get_tile_distribution());

    const float alpha = args.scalars[0].f32;
    const float beta  = args.scalars[1].f32;

    const auto spans = a_tile.get_distributed_spans();
    ck_tile::sweep_tile_span(spans[ck_tile::number<0>{}], [&](auto idx) {
        const auto tile_idx = ck_tile::make_tuple(idx);
        const float a_val   = ck_tile::type_convert<float>(a_tile(tile_idx));
        const float b_val   = ck_tile::type_convert<float>(b_tile(tile_idx));
        y_tile(tile_idx)    = ck_tile::type_convert<Y>(alpha * a_val + beta * b_val);
    });

    // Store output tile.
    const auto y_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        static_cast<Y*>(const_cast<void*>(t_c.ptr)),
        lens,
        strides,
        ck_tile::number<Shape::kVectorM>{});

    const auto y_transformed = ck_tile::pad_tensor_view(
        ck_tile::transform_tensor_view(y_view,
                                       ck_tile::make_tuple(merge_transform),
                                       ck_tile::make_tuple(ck_tile::make_index_sequence<1>{}),
                                       ck_tile::make_tuple(ck_tile::sequence<0>{})),
        ck_tile::make_tuple(ck_tile::number<Shape::kBlockM>{}),
        ck_tile::sequence<K.pad>{});

    auto y_window =
        ck_tile::make_tile_window(y_transformed,
                                  ck_tile::make_tuple(ck_tile::number<Shape::kBlockM>{}),
                                  {iM},
                                  y_tile.get_tile_distribution());

    ck_tile::store_tile(y_window, y_tile);
}

} // namespace rocm_ck
