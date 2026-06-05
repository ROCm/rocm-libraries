// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipCompiledProgram.hpp"
#include "HipUtils.hpp"
#include "IKernelCompiler.hpp"

#include <memory>
#include <string>
#include <vector>

namespace example_provider
{

/// Concrete IKernelCompiler that compiles kernels using HIPRTC.
///
/// Creates a HipCompiledProgram which handles HIPRTC compilation, module loading,
/// and kernel extraction. The current CPU architecture is queried automatically
/// and the corresponding --offload-arch option is prepended to the compile options.
class HipKernelCompiler : public IKernelCompiler
{
public:
    HipKernelCompiler()
    {
        int deviceId = 0;
        HIP_CHECK(hipGetDevice(&deviceId));

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

        // Extract base GPU architecture from gcnArchName
        // e.g., "gfx90a:sramecc+:xnack-" -> "gfx90a"
        std::string archName(props.gcnArchName);
        auto colonPos = archName.find(':');
        if(colonPos != std::string::npos)
        {
            archName = archName.substr(0, colonPos);
        }

        _offloadArchOption = "--offload-arch=" + archName;
    }

    std::unique_ptr<ICompiledProgram>
        compile(const std::string& kernelFileName,
                const std::vector<std::string>& options) const override
    {
        // Prepend the --offload-arch option to the caller-provided options
        std::vector<std::string> allOptions;
        allOptions.reserve(options.size() + 1);
        allOptions.push_back(_offloadArchOption);
        allOptions.insert(allOptions.end(), options.begin(), options.end());

        return std::make_unique<HipCompiledProgram>(kernelFileName, allOptions);
    }

private:
    std::string _offloadArchOption;
};

} // namespace example_provider
