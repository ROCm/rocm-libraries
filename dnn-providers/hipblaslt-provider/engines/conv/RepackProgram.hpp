// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <unordered_map>

#include <hip/hip_runtime_api.h>

#include "common/ConvGemmFormulation.hpp"
#include "kernels/ConvRepackLaunch.hpp"

namespace hipblaslt_plugin
{

/**
 * @brief The repack kernels (kernels/ConvRepack.cpp), compiled by HIPRTC for the current
 *        device, and the launch of one RepackStep.
 *
 * A minimal copy of the hip-kernel-provider's HIPRTC compile path (src/compilation/Program.cpp
 * there): one program, embedded in the plugin, compiled once per plugin handle with no on-disk
 * cache. It moves to the shared utility library once the team settles where that lives, and to
 * prebuilt code objects once the kernel set settles.
 */
class RepackProgram
{
public:
    /// Compiles the program for the current device; throws when HIPRTC or the load fails.
    RepackProgram();
    ~RepackProgram();

    RepackProgram(const RepackProgram&) = delete;
    RepackProgram& operator=(const RepackProgram&) = delete;
    RepackProgram(RepackProgram&&) = delete;
    RepackProgram& operator=(RepackProgram&&) = delete;

    /// Runs @p step on @p stream, reading @p src and writing @p dst (already advanced by the
    /// step's byte offsets). @p minRunThreads: the disjoint run kernels' narrowing
    /// (ConvRepackLaunch.hpp); tests lower it to reach every run width on small problems.
    void launch(const RepackStep& step,
                const void* src,
                void* dst,
                hipStream_t stream,
                long long minRunThreads = MIN_RUN_THREADS) const;

    /// The kernel entry point that runs @p step.
    static std::string entryName(const RepackStep& step);

private:
    hipModule_t _module = nullptr;
    std::unordered_map<std::string, hipFunction_t> _functions;
};

} // namespace hipblaslt_plugin
