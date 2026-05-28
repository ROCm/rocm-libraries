// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Common index types shared across CK Tile, rocm_ck, and dispatcher.

#pragma once

#include <cstdint>

namespace ck_common {

using index_t      = std::int32_t;
using long_index_t = std::int64_t;

} // namespace ck_common
