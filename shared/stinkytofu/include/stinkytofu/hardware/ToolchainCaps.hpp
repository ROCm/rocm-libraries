// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

namespace stinkytofu {

/// Lazily probes toolchain-dependent hardware capabilities via comgr.
///
/// Static capabilities (instruction support, modifiers, wait counters) are
/// defined in .def files.  Only a few capabilities depend on the installed
/// toolchain version and must be probed at runtime:
///   - VgprMsbMode: which `s_set_vgpr_msb` encoding the assembler accepts
///
/// Results are cached per GfxArchID for the process lifetime.
class STINKYTOFU_EXPORT ToolchainCaps {
   public:
    static AsmCapsConfig probe(GfxArchID archID);

   private:
    ToolchainCaps() = delete;
};

}  // namespace stinkytofu
