// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Thin wrapper -- canonical definitions live in ck_common/arch_properties.hpp.
// NOTE: prefer ck_common::X (or the underlying type) in new code; this header
// is a compatibility re-export.

#pragma once

#include <ck_common/arch_properties.hpp>

namespace rocm_ck {

// Re-export types used as parameters by functions below.
using ck_common::DataType;
using ck_common::GpuTarget;

using ck_common::ArchFamily;
using ck_common::isCDNA;
using ck_common::isRDNA;
using ck_common::isValidWaveTile;
using ck_common::properties;
using ck_common::TargetProperties;
using ck_common::TargetSet;
using ck_common::wavefrontSize;

} // namespace rocm_ck
