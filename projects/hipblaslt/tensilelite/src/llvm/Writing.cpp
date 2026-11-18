// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/llvm/Writing.hpp>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/llvm/YAML.hpp>

#include <llvm/Support/raw_ostream.h>

namespace TensileLite
{
    std::string LLVMWriteSolutionData(ContractionSolution const& solution)
    {
        std::string              text;
        llvm::raw_string_ostream stream(text);
        llvm::yaml::Output       yout(stream);

        // The traits take a mutable reference; writing does not modify the solution.
        yout << const_cast<ContractionSolution&>(solution);
        stream.flush();
        return text;
    }
}
