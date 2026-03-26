// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/softmax/invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/solver.hpp>
#include <miopen/softmax/problem_description.hpp>

namespace miopen {

namespace solver {

namespace softmax {

using SoftmaxSolver = NonTunableSolverBase<ExecutionContext, miopen::softmax::ProblemDescription>;

template <class PerformanceConfig>
using SoftmaxTunableSolver =
    TunableSolverMixin<ExecutionContext, miopen::softmax::ProblemDescription, PerformanceConfig>;

struct PerformanceConfigSoftmax : PerfConfigBase<PerformanceConfigSoftmax>
{
    int local_size;
    bool initialized = false;
    PerformanceConfigSoftmax(int _local_size) : local_size(_local_size) {}
    PerformanceConfigSoftmax() : local_size(start_local_size) {}
    PerformanceConfigSoftmax(bool) : local_size(start_local_size) {}
    void HeuristicInit(const miopen::softmax::ProblemDescription& problem);
    bool SetNextValue(const miopen::softmax::ProblemDescription& problem);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext& context,
                 const miopen::softmax::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.local_size, "local_size");
    }
    bool operator==(const PerformanceConfigSoftmax& other) const;

public:
    static constexpr auto default_local_size = 1024;
    static constexpr auto max_local_size     = 1024;
    static constexpr auto start_local_size   = 1;
};

struct Softmax final : SoftmaxTunableSolver<PerformanceConfigSoftmax>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<Softmax>(); }
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::softmax::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    PerformanceConfigSoftmax
    GetDefaultPerformanceConfig(const ExecutionContext& context,
                                const miopen::softmax::ProblemDescription& problem) const override;
    bool IsValidPerformanceConfig(const ExecutionContext& context,
                                  const miopen::softmax::ProblemDescription& problem,
                                  const PerformanceConfigSoftmax& config) const override;
    PerformanceConfigSoftmax Search(const ExecutionContext& context,
                                    const miopen::softmax::ProblemDescription& problem,
                                    const AnyInvokeParams& invoke_context) const override
    {
        // Forward y and backward dx are both an input and an output, so we need to restore them
        // after tuning to make sure tuning doesn't affect the output
        auto invoke_context_softmax = invoke_context.CastTo<miopen::softmax::InvokeParams>();
        std::vector<unsigned char> data_backup;
        if(problem.IsForward() && invoke_context_softmax.forward_y != nullptr)
        {
            data_backup = std::vector<unsigned char>(invoke_context_softmax.yDesc.GetNumBytes());
            context.GetStream().ReadTo(data_backup.data(),
                                       invoke_context_softmax.forward_y,
                                       invoke_context_softmax.yDesc.GetNumBytes());
        }
        else if(!problem.IsForward() && invoke_context_softmax.dx != nullptr)
        {
            data_backup = std::vector<unsigned char>(invoke_context_softmax.xdxDesc.GetNumBytes());
            context.GetStream().ReadTo(data_backup.data(),
                                       invoke_context_softmax.dx,
                                       invoke_context_softmax.xdxDesc.GetNumBytes());
        }

        auto result = GenericSearch(*this, context, problem, invoke_context);

        if(problem.IsForward() && invoke_context_softmax.forward_y != nullptr)
        {
            context.GetStream().WriteTo(data_backup.data(),
                                        invoke_context_softmax.forward_y,
                                        invoke_context_softmax.yDesc.GetNumBytes());
        }
        else if(!problem.IsForward() && invoke_context_softmax.dx != nullptr)
        {
            context.GetStream().WriteTo(data_backup.data(),
                                        invoke_context_softmax.dx,
                                        invoke_context_softmax.xdxDesc.GetNumBytes());
        }

        return result;
    }
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::softmax::ProblemDescription& problem,
                             const PerformanceConfigSoftmax& config) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext&,
                                 const miopen::softmax::ProblemDescription&) const override
    {
        return 0;
    }

    bool MayNeedWorkspace() const override { return false; }
};

struct AttnSoftmax final : SoftmaxSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<AttnSoftmax>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::softmax::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::softmax::ProblemDescription& problem) const override;

    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::softmax::ProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return false; }
};

} // namespace softmax

} // namespace solver

} // namespace miopen
