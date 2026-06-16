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
#include "miopen/generic_search.hpp"
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

struct PerformanceConfigLayernorm : PerfConfigBase<PerformanceConfigLayernorm>
{
    int local_size;
    bool vectorized;
    bool separate_stride;
    bool stride_in_local_size;
    bool initialized = false;
    PerformanceConfigLayernorm(int _local_size,
                               bool _vectorized,
                               bool _separate_stride,
                               bool _stride_in_local_size)
        : local_size(_local_size),
          vectorized(_vectorized),
          separate_stride(_separate_stride),
          stride_in_local_size(_stride_in_local_size)
    {
    }
    PerformanceConfigLayernorm()
        : PerformanceConfigLayernorm(
              start_local_size, start_vectorized, start_separate_stride, start_stride_in_local_size)
    {
    }
    PerformanceConfigLayernorm(bool)
        : PerformanceConfigLayernorm(
              start_local_size, start_vectorized, start_separate_stride, start_stride_in_local_size)
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
        f(s.vectorized, "vectorized");
        f(s.separate_stride, "separate_stride");
        f(s.stride_in_local_size, "stride_in_local_size");
    }
    bool operator==(const PerformanceConfigLayernorm& other) const;

public:
    static constexpr auto default_local_size(const miopen::layernorm::ProblemDescription& problem)
    {
        // Heuristics
        switch(problem.GetDirection())
        {
        case miopen::layernorm::Direction::Forward: return problem.stride == 1 ? 1024 : 32;
        case miopen::layernorm::Direction::Backward: return problem.stride == 1 ? 128 : 8;
        }
    }
    static constexpr auto max_local_size          = 1024;
    static constexpr auto max_parallel_local_size = 256;
    static constexpr auto start_local_size        = 1;
    static constexpr auto default_vectorized(const miopen::layernorm::ProblemDescription& problem)
    {
        // Heuristics
        switch(problem.GetDirection())
        {
        case miopen::layernorm::Direction::Forward: return problem.stride == 1;
        case miopen::layernorm::Direction::Backward: return problem.stride != 1;
        }
    }
    static constexpr auto start_vectorized = false;
    static constexpr auto
    default_separate_stride(const miopen::layernorm::ProblemDescription& problem)
    {
        // Heuristics
        return problem.stride != 1;
    }
    static constexpr auto start_separate_stride = false;
    static constexpr auto
    default_stride_in_local_size(const miopen::layernorm::ProblemDescription& problem)
    {
        // Heuristics
        switch(problem.GetDirection())
        {
        case miopen::layernorm::Direction::Forward: return problem.stride != 1;
        case miopen::layernorm::Direction::Backward: return false;
        }
    }
    static constexpr auto start_stride_in_local_size = false;

private:
    bool CheckParallelKernelBounds(const ExecutionContext& context,
                                   const miopen::layernorm::ProblemDescription& problem) const;
};

struct LayernormBase : NormalizationTunableSolver<PerformanceConfigLayernorm>
{
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsDynamic() const override { return true; }
    PerformanceConfigLayernorm GetDefaultPerformanceConfig(
        const ExecutionContext& context,
        const miopen::layernorm::ProblemDescription& problem) const override;
    bool IsValidPerformanceConfig(const ExecutionContext& context,
                                  const miopen::layernorm::ProblemDescription& problem,
                                  const PerformanceConfigLayernorm& config) const override;
};

struct LayernormForward final : LayernormBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<LayernormForward>(); }
    PerformanceConfigLayernorm Search(const ExecutionContext& context,
                                      const miopen::layernorm::ProblemDescription& problem,
                                      const AnyInvokeParams& invoke_context) const override
    {

        return GenericSearch(*this, context, problem, invoke_context);
    }
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernorm& config) const override;
};

struct LayernormBackward final : LayernormBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<LayernormBackward>(); }
    PerformanceConfigLayernorm Search(const ExecutionContext& context,
                                      const miopen::layernorm::ProblemDescription& problem,
                                      const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::layernorm::ProblemDescription& problem,
                             const PerformanceConfigLayernorm& config) const override;
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
