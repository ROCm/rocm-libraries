// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Thin wrapper -- canonical definition lives in ck_common/gpu_target.hpp.
// NOTE: prefer ck_common::X (or the underlying type) in new code; this header
// is a compatibility re-export.

#pragma once

#include <ck_common/gpu_target.hpp>

namespace rocm_ck {

using ck_common::GpuTarget;

} // namespace rocm_ck
