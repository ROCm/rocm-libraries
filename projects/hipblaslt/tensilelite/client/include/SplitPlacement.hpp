// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace TensileLite
{
    namespace Client
    {
        // Checks that --split-placement has one letter each for A, B, C, D, each R or S. Returns an
        // empty string when it does, else the message to report.
        inline std::string splitPlacementFormatError(std::string const& placement)
        {
            if(placement.size() != 4 || placement.find_first_not_of("RS") != std::string::npos)
                return "--split-placement needs 4 letters (A, B, C, D), each R or S, got '"
                       + placement + "'";
            return "";
        }

        // Checks --split-placement against --split-dim for --split-devices > 1. Returns an empty
        // string when the placement is valid, else the message to report.
        //
        // The placement has one letter each for A, B, C, D: R = a copy on every device, S = one
        // copy shared by all devices. Only the operand the split does not cut (B for an M split, A
        // for an N split) has the same shape and strides on every device, so only it can be shared,
        // and so can D, which every device holds in full size. A shared split operand or C would be
        // read from the start of the owner's slice-sized buffer, not from the device's own rows of
        // one full matrix.
        inline std::string splitPlacementError(std::string const& placement, bool splitN)
        {
            auto const formatError = splitPlacementFormatError(placement);
            if(!formatError.empty())
                return formatError;

            char const* const names[3] = {"A", "B", "C"};
            // Index of the operand the split cuts (A for M, B for N) and of the one it does not.
            const size_t      cut   = splitN ? 1 : 0;
            const size_t      uncut = splitN ? 0 : 1;
            std::string const split = splitN ? "N" : "M";
            for(size_t t : {size_t(0), size_t(1), size_t(2)})
            {
                if(placement[t] != 'S' || t == uncut)
                    continue;
                std::string const reason
                    = t == cut ? "the split cuts it" : "each device has its own C slice";
                return std::string("--split-placement: ") + names[t] + " cannot be shared in an "
                       + split + " split, because " + reason + " (only " + names[uncut]
                       + ", the operand the split does not cut, and D can be shared)";
            }
            return "";
        }
    }
}
