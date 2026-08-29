// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Shared by the Stream-K stream-isolation tests. Each of them rests on the
// heuristic answering their shape with a Stream-K solution on the remainder
// path, which is a property of the shipped tuning files rather than of the
// library: when it stops holding they must skip, not pass as no-ops.

#pragma once

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cctype>
#include <string>

namespace streamk_test
{
    // A solution name carries its tuning parameters as tokens, each an
    // abbreviation of the parameter name followed by its value
    // (Naming.py getParameterNameAbbreviation). StreamK abbreviates to "SK", so
    // "_SK0_" is a solution that is not Stream-K and "_SK3_" / "_SK4_" / "_SK5_"
    // are the variants that are. Requiring a digit after the token rules out
    // the neighbouring SKFTR / SKFDPO / SKWS / SKXCCM parameters.
    inline bool isStreamKSolutionName(const std::string& name)
    {
        for(size_t i = name.find("_SK"); i != std::string::npos; i = name.find("_SK", i + 1))
        {
            const char c = i + 3 < name.size() ? name[i + 3] : '\0';
            if(std::isdigit(static_cast<unsigned char>(c)))
                return c != '0';
        }
        return false;
    }

    inline std::string solutionName(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t& algo)
    {
        return hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
    }
} // namespace streamk_test
