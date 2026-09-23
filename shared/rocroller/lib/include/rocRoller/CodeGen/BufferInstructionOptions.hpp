// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Utilities/Error.hpp>

/**
 * BufferOptions is a struct that contains the options and optional values for MUBUF instructions.
 * OFFEN: 1 = Supply an offset from VGPR (VADDR), 0 Do not -> offset = 0. 1 bit
 * GLC: global coherent. Controls how reads and writes are handled by L1 cache. 1 bit
 * SLC: System Level Coherent, when set, streaming mode in L2 cache is set. 1 bit
 * LDS:  0 = Return read data to VGPR, 1 = Return read data to LDS. 1 bit
 */

namespace rocRoller
{
    struct BufferInstructionOptions
    {
        bool offen = false;
        bool glc   = false;
        bool slc   = false;
        bool sc1   = false;
        bool lds   = false;
    };

    inline std::string toString(BufferInstructionOptions const& options)
    {
        return concatenate("{",
                           ShowValue(options.offen),
                           ShowValue(options.glc),
                           ShowValue(options.slc),
                           ShowValue(options.sc1),
                           ShowValue(options.lds),
                           "}");
    }
}
