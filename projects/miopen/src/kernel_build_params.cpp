// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ranges>
#include <sstream>

#include <miopen/kernel_build_params.hpp>
#include <miopen/stringutils.hpp>

namespace miopen {

static std::string GenerateDefines(const std::vector<KernelBuildParameter>& options,
                                   const std::string& prefix)
{
    const auto strs =
        options | std::views::transform([&prefix](const KernelBuildParameter& define) {
            std::ostringstream ss;

            ss << '-';
            if(define.type == ParameterTypes::Define)
                ss << prefix;

            ss << define.name;

            if(!define.value.empty())
            {
                switch(define.type)
                {
                case ParameterTypes::Define: ss << '='; break;
                case ParameterTypes::Option: ss << ' '; break;
                }

                ss << define.value;
            }

            return ss.str();
        });

    return JoinStrings(strs, " ");
}

std::string kbp::OpenCL::Generate(const std::vector<KernelBuildParameter>& options)
{
    // Ensure only one space after the -cl-std.
    // >1 space can cause an Apple compiler bug. See clSPARSE issue #141.

    return GenerateDefines(options, "D");
}

std::string kbp::GcnAsm::Generate(const std::vector<KernelBuildParameter>& options)
{
    return GenerateDefines(options, "Wa,-defsym,");
}

std::string kbp::HIP::Generate(const std::vector<KernelBuildParameter>& options)
{
    return GenerateDefines(options, "D");
}

} // namespace miopen
