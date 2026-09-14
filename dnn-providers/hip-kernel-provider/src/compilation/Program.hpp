// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <string>
#include <utility>
#include <vector>

namespace hip_kernel_provider::compilation
{

class Program
{
public:
    /// Compiles the kernel the configure-time embedded table holds under
    /// @p kernelFileName, with the embedded include list as its virtual headers.
    Program(std::string kernelFileName, const std::vector<std::string>& options);

    /// Compiles owned source text, for a kernel no embedded table holds -- a hiprtc_file
    /// kernel read out of a drop-in bundle at prepare().
    ///
    /// @param headers The complete virtual-header list as (name, text) pairs. Complete
    ///                rather than merged with the embedded list here, so the caller that
    ///                can name a colliding bundle header is the one that rejects it.
    Program(std::string sourceText,
            std::string programName,
            std::vector<std::pair<std::string, std::string>> headers,
            const std::vector<std::string>& options);

    hipFunction_t getKernel(const std::string& kernelName) const;

    ~Program();

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&) = default;
    Program& operator=(Program&&) = default;

private:
    /// The hipRTC create/compile/extract/load sequence both constructors run, once they
    /// have agreed where the source and headers come from. @p sourceText and every header
    /// text must be null-terminated, which hipRTC requires and both callers satisfy.
    void compileAndLoad(const char* sourceText,
                        const std::vector<const char*>& headerTexts,
                        const std::vector<const char*>& headerNames,
                        const std::vector<std::string>& options);

    std::string _programName;
    hipModule_t _module = nullptr;
    std::vector<char> _binary;
};

} // namespace hip_kernel_provider::compilation
