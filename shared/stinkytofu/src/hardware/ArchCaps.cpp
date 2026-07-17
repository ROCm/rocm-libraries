// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/hardware/ArchCaps.hpp"

namespace stinkytofu {

ArchCapsConfig ArchCaps::query(GfxArchID archID) {
    ArchCapsConfig caps;
    caps.RequiresXCntForVolatileVMEM = (archID == GfxArchID::Gfx1250);
    return caps;
}

}  // namespace stinkytofu
