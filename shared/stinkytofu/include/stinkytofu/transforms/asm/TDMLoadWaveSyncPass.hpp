// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TDMLoadWaveSyncPass — inserts a workgroup barrier (an s_barrier_signal -1 /
// s_barrier_wait -1 pair) between an urgent and a deferrable group of
// tensor_load_to_lds (different TDM wait groups), so every wave finishes the urgent
// group before any wave issues the deferrable one.

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

STINKYTOFU_EXPORT std::unique_ptr<Pass> createTDMLoadWaveSyncPass();

}  // namespace stinkytofu
