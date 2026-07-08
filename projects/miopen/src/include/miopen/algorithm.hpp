// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Backward-compatibility forwarding header.
//
// algorithm was moved to common_utils/ during the MIOpen layering refactor.
// External consumers (e.g. ROCm/MIFin) still include <miopen/algorithm.hpp>
// from the installed package, so this shim preserves that entry point.
// #pragma once avoids colliding with the include guard in the real header.
#pragma once

#include <common_utils/algorithm.hpp>
