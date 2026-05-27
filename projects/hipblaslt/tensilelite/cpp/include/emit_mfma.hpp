// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "code.hpp"
#include "enum.hpp"
#include <memory>
#include <string>

namespace tl_emit
{
    std::shared_ptr<rocisa::Module> emitMfmaInstruction(int              mxInstTypeInt,
                                                        int              miK,
                                                        bool             sourceSwap,
                                                        bool             miArchVgpr,
                                                        int              vgprAStart,
                                                        int              opASize,
                                                        int              vgprBStart,
                                                        int              opBSize,
                                                        int              vgprCStart,
                                                        int              opCSize,
                                                        bool             cIsAccvgpr,
                                                        int              vgprDStart,
                                                        int              opDSize,
                                                        bool             dIsAccvgpr,
                                                        int              scaleAVgpr,
                                                        int              scaleBVgpr,
                                                        int              scaleAsel,
                                                        int              scaleBsel,
                                                        int              tmpScaleVgpr,
                                                        const std::string& comment);
} // namespace tl_emit
