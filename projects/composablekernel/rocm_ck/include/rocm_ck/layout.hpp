// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Thin wrapper — canonical definitions live in ck_common/layout.hpp.

#pragma once

#include <ck_common/layout.hpp>

// Keep ROCM_CK_UNREACHABLE available for other rocm_ck headers that include this.
#include "rocm_ck/platform.hpp"

namespace rocm_ck {

using ck_common::isValidLayoutForRank;
using ck_common::Layout;
using ck_common::layoutName;
using ck_common::layoutStrides;
using ck_common::leadingDimStride;

} // namespace rocm_ck
