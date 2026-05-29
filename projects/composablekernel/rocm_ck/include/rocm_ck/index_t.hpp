// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Thin wrapper -- canonical definitions live in ck_common/index_t.hpp.
// NOTE: prefer ck_common::X (or the underlying type) in new code; this header
// is a compatibility re-export.

#pragma once

#include <ck_common/index_t.hpp>

namespace rocm_ck {

using ck_common::index_t;
using ck_common::long_index_t;

} // namespace rocm_ck
