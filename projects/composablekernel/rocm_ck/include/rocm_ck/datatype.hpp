// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Thin wrapper -- canonical definitions live in ck_common/datatype.hpp.
// NOTE: prefer ck_common::X (or the underlying type) in new code; this header
// is a compatibility re-export.

#pragma once

#include <ck_common/datatype.hpp>

// Keep ROCM_CK_UNREACHABLE available for other rocm_ck headers that include this.
#include "rocm_ck/platform.hpp"

namespace rocm_ck {

using ck_common::DataType;
using ck_common::dataTypeBits;
using ck_common::dataTypeName;

} // namespace rocm_ck
