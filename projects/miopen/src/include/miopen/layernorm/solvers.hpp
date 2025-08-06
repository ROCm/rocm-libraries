/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

#include "miopen/execution_context.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/performance_config.hpp"
#include <miopen/layernorm/problem_description.hpp>
#include <miopen/solver.hpp>

namespace miopen {

namespace solver {

namespace layernorm {

using NormalizationSolver =
    NonTunableSolverBase<ExecutionContext, miopen::layernorm::ProblemDescription>;

template <class PerformanceConfig>
using NormalizationTunableSolver =
    TunableSolverMixin<ExecutionContext, miopen::layernorm::ProblemDescription, PerformanceConfig>;

struct PerformanceConfigLayernormForward : PerfConfigBase<PerformanceConfigLayernormForward>
{
    int local_size;
    bool initialized = false;
    PerformanceConfigLayernormForward(int _local_size) : local_size(_local_size) {}
    PerformanceConfigLayernormForward() : PerformanceConfigLayernormForward(static_cast<int>(1)) {}
    PerformanceConfigLayernormForward(bool) : PerformanceConfigLayernormForward(static_cast<int>(1))
    {
    }
    void HeuristicInit(const miopen::layernorm::ProblemDescription& problem);
    bool SetNextValue(const miopen::layernorm::ProblemDescription& problem);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext& context,
                 const miopen::layernorm::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.local_size, "local_size");
    }
    bool operator==(const PerformanceConfigLayernormForward& other) const;
};

struct LayernormForward final : NormalizationTunableSolver<PerformanceConfigLayernormForward>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<LayernormForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    PerformanceConfigLayernormForward GetDefaultPerformanceConfig(
        const ExecutionContext& context,
        const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsValidPerformanceConfig(const ExecutionContext& context,
                                  const miopen::layernorm::ProblemDescription& problem,
                                  const PerformanceConfigLayernormForward& config) const override;
    PerformanceConfigLayernormForward Search(const ExecutionContext& context,
                                             const miopen::layernorm::ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_context) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernormForward& config) const override;
};

struct PerformanceConfigLayernormBackward : PerfConfigBase<PerformanceConfigLayernormBackward>
{
    int local_size;
    bool initialized = false;
    PerformanceConfigLayernormBackward(int _local_size) : local_size(_local_size) {}
    PerformanceConfigLayernormBackward() : PerformanceConfigLayernormBackward(static_cast<int>(1))
    {
    }
    PerformanceConfigLayernormBackward(bool)
        : PerformanceConfigLayernormBackward(static_cast<int>(1))
    {
    }
    void HeuristicInit(const miopen::layernorm::ProblemDescription& problem);
    bool SetNextValue(const miopen::layernorm::ProblemDescription& problem);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext& context,
                 const miopen::layernorm::ProblemDescription& problem) const;

    template <typename Self, typename F>
    static void Visit(Self&& s, F f)
    {
        f(s.local_size, "local_size");
    }
    bool operator==(const PerformanceConfigLayernormBackward& other) const;
};

struct LayernormBackward final : NormalizationTunableSolver<PerformanceConfigLayernormBackward>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<LayernormBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    PerformanceConfigLayernormBackward GetDefaultPerformanceConfig(
        const ExecutionContext& context,
        const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsValidPerformanceConfig(const ExecutionContext& context,
                                  const miopen::layernorm::ProblemDescription& problem,
                                  const PerformanceConfigLayernormBackward& config) const override;
    PerformanceConfigLayernormBackward Search(const ExecutionContext& context,
                                              const miopen::layernorm::ProblemDescription& problem,
                                              const AnyInvokeParams& invoke_context) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernormBackward& config) const override;
    std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::layernorm::ProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

struct AddLayernormForward final : NormalizationSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<AddLayernormForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem) const override;
};

struct T5LayernormForward final : NormalizationSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<T5LayernormForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem) const override;
};

struct T5LayernormBackward final : NormalizationSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<T5LayernormBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem) const override;
    std::size_t
    GetWorkspaceSize(const ExecutionContext& context,
                     const miopen::layernorm::ProblemDescription& problem) const override;
    bool MayNeedWorkspace() const override { return true; }
};

} // namespace layernorm

} // namespace solver

} // namespace miopen
