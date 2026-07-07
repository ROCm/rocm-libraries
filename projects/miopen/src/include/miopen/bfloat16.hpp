// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Backward-compatibility forwarding header.
//
// bfloat16 was moved to common_utils/ during the MIOpen layering refactor.
// External consumers (e.g. ROCm/MIFin) still include <miopen/bfloat16.hpp> from
// the installed package, so this shim preserves that entry point.
//
// miopen/config.h is pulled first to supply MIOPEN_USE_RNE_BFLOAT16 for consumers
// that do not link MIOpen::common_utils (which otherwise propagates it as an
// INTERFACE compile definition). Uses #pragma once to avoid colliding with the
// BFLOAT16_H_ include guard in common_utils/bfloat16.hpp.
#pragma once

#include <miopen/config.h>
#include <common_utils/bfloat16.hpp>
