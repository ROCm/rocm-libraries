// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Backward-compatibility forwarding header.
//
// reduce_common was moved to common_utils/ during the MIOpen layering refactor.
// External consumers (e.g. ROCm/MIFin) still include <miopen/reduce_common.hpp>
// from the installed package, so this shim preserves that entry point.
//
// miopen/config.h is pulled first to supply MIOPEN_USE_RNE_BFLOAT16, which
// common_utils/reduce_common.hpp needs transitively via common_utils/bfloat16.hpp
// (external consumers do not link MIOpen::common_utils, which otherwise propagates
// it as an INTERFACE compile definition). #pragma once avoids colliding with the
// include guard in the real header.
#pragma once

#include <miopen/config.h>
#include <common_utils/reduce_common.hpp>
