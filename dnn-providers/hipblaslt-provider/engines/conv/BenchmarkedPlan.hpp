// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engines/plans/PlanInterface.hpp"

namespace hipblaslt_plugin
{

/**
 * @brief Several plans for one graph, each computing the same output, of which the first
 *        execute keeps the fastest.
 *
 * The provider's form of the `global.benchmarking` knob (as the kernel ingestor and MIOpen
 * provider sample their candidates): built only when benchmarking is on, it times every
 * candidate on the first execute -- one warm-up, then the best of a few runs, by HIP events on
 * the handle's stream -- and runs only the winner after that. Every candidate writes the whole
 * output, so the timed runs leave a correct result. The workspace is the largest candidate's.
 */
class BenchmarkedPlan : public IPlan
{
public:
    struct Candidate
    {
        std::string name;
        std::unique_ptr<IPlan> plan;
    };

    explicit BenchmarkedPlan(std::vector<Candidate> candidates);

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

    /// The winner's index, or -1 before the first execute.
    int winner() const;

private:
    std::vector<Candidate> _candidates;
    mutable std::mutex _mutex;
    mutable int _winner = -1;
};

} // namespace hipblaslt_plugin
