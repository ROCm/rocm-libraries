// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

namespace stinkytofu {

/// Architecture capabilities derived from the ISA tuple. These mirror the archCaps
/// names used by rocisa/TensileLite where a pass needs the same policy in C++.
class STINKYTOFU_EXPORT ArchCaps {
   public:
    static ArchCapsConfig query(GfxArchID archID);

   private:
    ArchCaps() = delete;
};

}  // namespace stinkytofu
