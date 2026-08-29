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
    // (Naming.py getParameterNameAbbreviation). Reads the single-digit value
    // "token" carries, or -1 when the name does not carry it. Requiring a digit
    // is what separates "_SK" from the neighbouring SKFTR / SKFDPO / SKWS /
    // SKXCCM tokens, which share its prefix.
    inline int solutionNameValue(const std::string& name, const std::string& token)
    {
        for(size_t i = name.find(token); i != std::string::npos; i = name.find(token, i + 1))
        {
            const size_t v = i + token.size();
            const char   c = v < name.size() ? name[v] : '\0';
            if(std::isdigit(static_cast<unsigned char>(c)))
                return c - '0';
        }
        return -1;
    }

    // What these tests need is not "Stream-K" but "reads the flag region", which
    // is the condition singleCallArgs uses to decide whether to append Flags at
    // all: streamK > 0 && streamKAtomic == 0 && streamKForceDPOnly == 0. A
    // DP-only solution passes for Stream-K and never touches the region, so a
    // test resting on one would run green as a no-op. StreamKForceDPOnly is in
    // the minimum parameter set and so appears in the name; StreamKAtomic is
    // not, and no shipped logic file sets it.
    inline bool readsFlagRegionSolutionName(const std::string& name)
    {
        return solutionNameValue(name, "_SK") > 0 && solutionNameValue(name, "_SKFDPO") <= 0;
    }

    inline std::string solutionName(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t& algo)
    {
        return hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
    }

    // Empty when `name` describes a solution that reads the flag region, and
    // otherwise why it does not. An empty name gets its own reason: it means
    // the name could not be read at all, which is a library that failed to load
    // rather than a heuristic that answered with something else.
    inline std::string flagRegionSkipReason(const std::string& name)
    {
        if(name.empty())
            return "Could not read the solution name; the Tensile library may have failed to load";
        if(!readsFlagRegionSolutionName(name))
            return "The heuristic did not pick a solution that reads the Stream-K flag region: "
                   + name;
        return {};
    }
} // namespace streamk_test
