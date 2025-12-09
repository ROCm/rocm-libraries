/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/solver.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/pooling/problem_description.hpp>
#include <miopen/utility/transposing_solver.hpp>
#include <miopen/performance_config.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/generic_search.hpp>

#include <utility>

namespace miopen {

namespace solver {

namespace pooling {

enum class OperationType
{
    Forward,
    Backward
};

using PoolingSolver = NonTunableSolverBase<ExecutionContext, miopen::pooling::ProblemDescription>;
template <class PerformanceConfig>
using PoolingTunableSolver =
    SolverBaseTunable<ExecutionContext, miopen::pooling::ProblemDescription, PerformanceConfig>;

template <OperationType OpType>
struct PerformanceConfigPooling2d : PerfConfigBase<PerformanceConfigPooling2d<OpType>>
{
    static_assert(OpType == OperationType::Forward || OpType == OperationType::Backward,
                  "OperationType must be either Forward or Backward");

    int out_pix_tile0;
    int out_pix_tile1;
    int local_size0;
    int local_size1;
    static constexpr int min_out_pix_tile0 = 1;
    static constexpr int max_out_pix_tile0 = (OpType == OperationType::Forward) ? 1 : 4;
    static constexpr int min_out_pix_tile1 = 1;
    static constexpr int max_out_pix_tile1 = (OpType == OperationType::Forward) ? 16 : 8;
    static constexpr int min_local_size0   = (OpType == OperationType::Forward) ? 8 : 4;
    static constexpr int max_local_size0   = 32;
    static constexpr int min_local_size1   = (OpType == OperationType::Forward) ? 8 : 4;
    static constexpr int max_local_size1   = (OpType == OperationType::Forward) ? 128 : 16;

    PerformanceConfigPooling2d(int out_pix_tile0_,
                               int out_pix_tile1_,
                               int local_size0_,
                               int local_size1_)
        : out_pix_tile0(out_pix_tile0_),
          out_pix_tile1(out_pix_tile1_),
          local_size0(local_size0_),
          local_size1(local_size1_)
    {
    }
    PerformanceConfigPooling2d()
        : PerformanceConfigPooling2d(
              min_out_pix_tile0, min_out_pix_tile1, min_local_size0, min_local_size1)
    {
    }
    PerformanceConfigPooling2d(bool)
        : PerformanceConfigPooling2d(
              min_out_pix_tile0, min_out_pix_tile1, min_local_size0, min_local_size1)
    {
    }

    void HeuristicInit(const miopen::pooling::ProblemDescription&);
    bool SetNextValue(const miopen::pooling::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::pooling::ProblemDescription&) const;
    bool operator==(const PerformanceConfigPooling2d& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.out_pix_tile0, "out_pix_tile0");
        f(self.out_pix_tile1, "out_pix_tile1");
        f(self.local_size0, "local_size0");
        f(self.local_size1, "local_size1");
    }

private:
    void Init(const miopen::pooling::ProblemDescription&);
};

extern template struct PerformanceConfigPooling2d<OperationType::Forward>;
extern template struct PerformanceConfigPooling2d<OperationType::Backward>;

struct PoolingForward2d final
    : PoolingTunableSolver<PerformanceConfigPooling2d<OperationType::Forward>>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingForward2d>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolutionImpl(
        const ExecutionContext& context,
        const miopen::pooling::ProblemDescription& problem,
        const std::optional<PerformanceConfigPooling2d<OperationType::Forward>>& config) const;
    // This method is added to maintain compatibility with TransposedPoolingFwd2d solver
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem) const
    {
        return GetSolutionImpl(context, problem, std::nullopt);
    }
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::pooling::ProblemDescription& problem,
                const PerformanceConfigPooling2d<OperationType::Forward>& config) const override
    {
        return GetSolutionImpl(context, problem, config);
    }
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
    PerformanceConfigPooling2d<OperationType::Forward>
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::pooling::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::pooling::ProblemDescription&,
        const PerformanceConfigPooling2d<OperationType::Forward>&) const override;
    PerformanceConfigPooling2d<OperationType::Forward>
    Search(const ExecutionContext& context,
           const miopen::pooling::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
};

struct PoolingForwardNd final : PoolingSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingForwardNd>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
};

struct PoolingForwardNaive final : PoolingSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingForwardNaive>(); }
    bool IsDynamic() const override { return true; }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
};

template <class Inner>
struct PoolingFwdNCHWTransposingSolver : TransposingSolver<PoolingFwdNCHWTransposingSolver<Inner>,
                                                           PoolingSolver,
                                                           miopen::pooling::ProblemDescription,
                                                           miopen::pooling::FwdInvokeParams,
                                                           Inner>
{
    using Problem      = miopen::pooling::ProblemDescription;
    using InvokeParams = miopen::pooling::FwdInvokeParams;

    inline static auto GetTransposes()
    {
        auto ret = std::array<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, 2>{{
            {
                &Problem::GetXDesc,
                &Problem::GetXDesc,
                &InvokeParams::xDesc,
                {&InvokeParams::x},
                "NCDHW",
                true,
            },
            {
                &Problem::GetYDesc,
                &Problem::GetYDesc,
                &InvokeParams::yDesc,
                {},
                "NCDHW",
                false,
            },
        }};

        // Before C++20 you can't aggregate initialize non-first union element
        ret[1].as_output = &InvokeParams::y;

        return ret;
    }
};

struct TransposedPoolingFwd2d final : PoolingFwdNCHWTransposingSolver<PoolingForward2d>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingFwd2d>();
    }
};

struct TransposedPoolingFwdNd final : PoolingFwdNCHWTransposingSolver<PoolingForwardNd>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingFwdNd>();
    }
};

struct PoolingBackward2d final
    : PoolingTunableSolver<PerformanceConfigPooling2d<OperationType::Backward>>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingBackward2d>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolutionImpl(
        const ExecutionContext& context,
        const miopen::pooling::ProblemDescription& problem,
        const std::optional<PerformanceConfigPooling2d<OperationType::Backward>>& config) const;
    // This method is added to maintain compatibility with TransposedPoolingBwd2d solver
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem) const
    {
        return GetSolutionImpl(context, problem, std::nullopt);
    }
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::pooling::ProblemDescription& problem,
                const PerformanceConfigPooling2d<OperationType::Backward>& config) const override
    {
        return GetSolutionImpl(context, problem, config);
    }
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
    PerformanceConfigPooling2d<OperationType::Backward>
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::pooling::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::pooling::ProblemDescription&,
        const PerformanceConfigPooling2d<OperationType::Backward>&) const override;
    PerformanceConfigPooling2d<OperationType::Backward>
    Search(const ExecutionContext& context,
           const miopen::pooling::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
};

struct PoolingBackwardNd final : PoolingSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingBackwardNd>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
};

template <class Inner>
struct PoolingBwdNCHWTransposingSolver : TransposingSolver<PoolingBwdNCHWTransposingSolver<Inner>,
                                                           PoolingSolver,
                                                           miopen::pooling::ProblemDescription,
                                                           miopen::pooling::BwdInvokeParams,
                                                           Inner>
{
    using Problem      = miopen::pooling::ProblemDescription;
    using InvokeParams = miopen::pooling::BwdInvokeParams;

    inline static auto GetTransposes()
    {
        auto ret = std::array<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, 2>{{
            {
                &Problem::GetXDesc,
                &Problem::GetXDesc,
                &InvokeParams::dxDesc,
                {},
                "NCDHW",
                false,
            },
            {
                &Problem::GetYDesc,
                &Problem::GetYDesc,
                &InvokeParams::dyDesc,
                {&InvokeParams::dy},
                "NCDHW",
                true,
            },
        }};

        // Before C++20 you can't aggregate initialize non-first union element
        ret[0].as_output = &InvokeParams::dx;

        return ret;
    }
};

struct TransposedPoolingBwd2d final : PoolingBwdNCHWTransposingSolver<PoolingBackward2d>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingBwd2d>();
    }
};

struct TransposedPoolingBwdNd final : PoolingBwdNCHWTransposingSolver<PoolingBackwardNd>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingBwdNd>();
    }
};

} // namespace pooling

} // namespace solver

} // namespace miopen
