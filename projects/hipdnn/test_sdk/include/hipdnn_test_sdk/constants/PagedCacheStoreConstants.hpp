// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard PagedCacheStore constants for testing get/set of valid operations.
// These represent "any valid pagedcachestore" — specific values are not significant.

constexpr int64_t K_PAGED_CACHE_STORE_TENSOR_X_UID = 9500;
constexpr std::array<int64_t, 3> K_PAGED_CACHE_STORE_TENSOR_X_DIMS = {4, 2, 8};
constexpr std::array<int64_t, 3> K_PAGED_CACHE_STORE_TENSOR_X_STRIDES = {16, 8, 1};

constexpr int64_t K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID = 9501;
constexpr std::array<int64_t, 1> K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS = {4};
constexpr std::array<int64_t, 1> K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES = {1};

constexpr int64_t K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID = 9502;
constexpr std::array<int64_t, 4> K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS = {3, 2, 4, 8};
constexpr std::array<int64_t, 4> K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES = {64, 32, 8, 1};

constexpr int64_t K_PAGED_CACHE_STORE_TENSOR_SCALE_UID = 9503;
constexpr std::array<int64_t, 1> K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS = {1};
constexpr std::array<int64_t, 1> K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES = {1};

} // namespace hipdnn_tests::constants
