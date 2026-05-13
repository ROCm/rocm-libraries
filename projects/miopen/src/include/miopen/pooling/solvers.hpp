// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
    virtual bool IsValidValue(const miopen::pooling::ProblemDescription&) const
    {
        throw std::runtime_error(
            "IsValidValue of PerformanceConfigPooling2d<OpType> is called, but it is not "
            "implemented.");
    }
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

struct PerformanceConfigPooling2dForward final
    : public PerformanceConfigPooling2d<OperationType::Forward>
{
    using PerformanceConfigPooling2d<OperationType::Forward>::PerformanceConfigPooling2d;

    bool IsValidValue(const miopen::pooling::ProblemDescription&) const override;
};

struct PerformanceConfigPooling2dBackward final
    : public PerformanceConfigPooling2d<OperationType::Backward>
{
    using PerformanceConfigPooling2d<OperationType::Backward>::PerformanceConfigPooling2d;

    bool IsValidValue(const miopen::pooling::ProblemDescription&) const override;
};

template <OperationType OpType>
struct PerformanceConfigPoolingNd : PerfConfigBase<PerformanceConfigPoolingNd<OpType>>
{
    static_assert(OpType == OperationType::Forward || OpType == OperationType::Backward,
                  "OperationType must be either Forward or Backward");

    int pix_w_per_work;
    int pix_h_per_work;
    int pix_d_per_work;
    int local_size;
    static constexpr int min_pix_per_work = 1;
    static constexpr int max_pix_per_work = 4;

    PerformanceConfigPoolingNd(int pix_w_per_work_,
                               int pix_h_per_work_,
                               int pix_d_per_work_,
                               int local_size_)
        : pix_w_per_work(pix_w_per_work_),
          pix_h_per_work(pix_h_per_work_),
          pix_d_per_work(pix_d_per_work_),
          local_size(local_size_)
    {
    }

    PerformanceConfigPoolingNd() : PerformanceConfigPoolingNd(1, 1, 1, 1) {}

    PerformanceConfigPoolingNd(bool) : PerformanceConfigPoolingNd(1, 1, 1, 1) {}

    void HeuristicInit(const miopen::pooling::ProblemDescription&);
    virtual bool SetNextValue(const miopen::pooling::ProblemDescription&)
    {
        throw std::runtime_error(
            "SetNextValue of PerformanceConfigPoolingNd<OpType> is called, but it is not "
            "implemented.");
    }
    virtual bool IsValidValue() const
    {
        throw std::runtime_error(
            "IsValidValue of PerformanceConfigPoolingNd<OpType> is called, but it is not "
            "implemented.");
    }
    bool IsValid(const ExecutionContext&, const miopen::pooling::ProblemDescription&) const;
    bool operator==(const PerformanceConfigPoolingNd& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.pix_w_per_work, "pix_w_per_work");
        f(self.pix_h_per_work, "pix_h_per_work");
        f(self.pix_d_per_work, "pix_d_per_work");
        f(self.local_size, "local_size");
    }

private:
    void Init(const miopen::pooling::ProblemDescription&);
};

extern template struct PerformanceConfigPoolingNd<OperationType::Forward>;
extern template struct PerformanceConfigPoolingNd<OperationType::Backward>;

struct PoolingForward2d final : PoolingTunableSolver<PerformanceConfigPooling2dForward>
{
    using PerformanceConfigType = PerformanceConfigPooling2dForward;

    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingForward2d>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem,
                             const PerformanceConfigPooling2dForward& config) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
    PerformanceConfigPooling2dForward
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::pooling::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::pooling::ProblemDescription&,
                                  const PerformanceConfigPooling2dForward&) const override;
    PerformanceConfigPooling2dForward Search(const ExecutionContext& context,
                                             const miopen::pooling::ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
};

struct PerformanceConfigPoolingNdForward final : PerformanceConfigPoolingNd<OperationType::Forward>
{
    using PerformanceConfigPoolingNd::PerformanceConfigPoolingNd;

    bool SetNextValue(const miopen::pooling::ProblemDescription&) override;
    bool IsValidValue() const override;
};

struct PoolingForwardNd final : PoolingTunableSolver<PerformanceConfigPoolingNdForward>
{
    using PerformanceConfigType = PerformanceConfigPoolingNdForward;

    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingForwardNd>(); }
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem,
                             const PerformanceConfigPoolingNdForward& config) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
    PerformanceConfigPoolingNdForward
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::pooling::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::pooling::ProblemDescription&,
                                  const PerformanceConfigPoolingNdForward&) const override;
    PerformanceConfigPoolingNdForward Search(const ExecutionContext& context,
                                             const miopen::pooling::ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
};

struct PerformanceConfigPoolingForwardNaive : PerfConfigBase<PerformanceConfigPoolingForwardNaive>
{
    int local_size0;
    int local_size1;
    int local_size2;

    PerformanceConfigPoolingForwardNaive(int local_size0_, int local_size1_, int local_size2_)
        : local_size0(local_size0_), local_size1(local_size1_), local_size2(local_size2_)
    {
    }

    PerformanceConfigPoolingForwardNaive() : PerformanceConfigPoolingForwardNaive(1, 1, 1) {}

    PerformanceConfigPoolingForwardNaive(bool) : PerformanceConfigPoolingForwardNaive(1, 1, 1) {}

    void HeuristicInit(const miopen::pooling::ProblemDescription&);
    bool SetNextValue(const miopen::pooling::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::pooling::ProblemDescription&) const;
    bool operator==(const PerformanceConfigPoolingForwardNaive& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.local_size0, "local_size0");
        f(self.local_size1, "local_size1");
        f(self.local_size2, "local_size2");
    }

private:
    void Init(const miopen::pooling::ProblemDescription&);
};

struct PoolingForwardNaive final : PoolingTunableSolver<PerformanceConfigPoolingForwardNaive>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingForwardNaive>(); }
    bool IsDynamic() const override { return true; }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem,
                             const PerformanceConfigPoolingForwardNaive& config) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
    PerformanceConfigPoolingForwardNaive
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::pooling::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::pooling::ProblemDescription&,
                                  const PerformanceConfigPoolingForwardNaive&) const override;
    PerformanceConfigPoolingForwardNaive
    Search(const ExecutionContext& context,
           const miopen::pooling::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
};

template <class Inner>
struct PoolingFwdNCHWTransposingSolver
    : TransposingSolver<PoolingFwdNCHWTransposingSolver<Inner>,
                        PoolingTunableSolver<typename Inner::PerformanceConfigType>,
                        miopen::pooling::ProblemDescription,
                        miopen::pooling::FwdInvokeParams,
                        Inner>
{
    using Problem      = miopen::pooling::ProblemDescription;
    using InvokeParams = miopen::pooling::FwdInvokeParams;

    inline static auto GetTransposes(const Problem& /*problem*/)
    {
        return std::array<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, 2>{{
            {
                &Problem::GetXDesc,
                &InvokeParams::xDesc,
                &InvokeParams::x, // x is input
                nullptr,
                "NCDHW",
                true,
            },
            {
                &Problem::GetYDesc,
                &InvokeParams::yDesc,
                nullptr,
                &InvokeParams::y, // y is output
                "NCDHW",
                false,
            },
        }};
    }
};

struct TransposedPoolingFwd2d final : PoolingFwdNCHWTransposingSolver<PoolingForward2d>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingFwd2d>();
    }
    PoolingForward2d::PerformanceConfigType
    GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                const miopen::pooling::ProblemDescription& problem) const override
    {
        return PoolingForward2d{}.GetDefaultPerformanceConfig(ctx, problem);
    }
    bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::pooling::ProblemDescription& problem,
                             const PoolingForward2d::PerformanceConfigType& config) const override
    {
        return PoolingForward2d{}.IsValidPerformanceConfig(ctx, problem, config);
    }
    PoolingForward2d::PerformanceConfigType
    Search(const ExecutionContext& context,
           const miopen::pooling::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override
    {
        return PoolingForward2d{}.Search(context, problem, invoke_context);
    }
};

struct TransposedPoolingFwdNd final : PoolingFwdNCHWTransposingSolver<PoolingForwardNd>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingFwdNd>();
    }
    PoolingForwardNd::PerformanceConfigType
    GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                const miopen::pooling::ProblemDescription& problem) const override
    {
        return PoolingForwardNd{}.GetDefaultPerformanceConfig(ctx, problem);
    }
    bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::pooling::ProblemDescription& problem,
                             const PoolingForwardNd::PerformanceConfigType& config) const override
    {
        return PoolingForwardNd{}.IsValidPerformanceConfig(ctx, problem, config);
    }
    PoolingForwardNd::PerformanceConfigType
    Search(const ExecutionContext& context,
           const miopen::pooling::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override
    {
        return PoolingForwardNd{}.Search(context, problem, invoke_context);
    }
};

struct PoolingBackward2d final : PoolingTunableSolver<PerformanceConfigPooling2dBackward>
{
    using PerformanceConfigType = PerformanceConfigPooling2dBackward;

    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingBackward2d>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem,
                             const PerformanceConfigPooling2dBackward& config) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
    PerformanceConfigPooling2dBackward
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::pooling::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::pooling::ProblemDescription&,
                                  const PerformanceConfigPooling2dBackward&) const override;
    PerformanceConfigPooling2dBackward Search(const ExecutionContext& context,
                                              const miopen::pooling::ProblemDescription& problem,
                                              const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
};

struct PerformanceConfigPoolingNdBackward final
    : PerformanceConfigPoolingNd<OperationType::Backward>
{
    using PerformanceConfigPoolingNd::PerformanceConfigPoolingNd;

    bool SetNextValue(const miopen::pooling::ProblemDescription&) override;
    bool IsValidValue() const override;
};

struct PoolingBackwardNd final : PoolingTunableSolver<PerformanceConfigPoolingNdBackward>
{
    using PerformanceConfigType = PerformanceConfigPoolingNdBackward;

    const std::string& SolverDbId() const override { return GetSolverDbId<PoolingBackwardNd>(); }
    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pooling::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::pooling::ProblemDescription& problem,
                             const PerformanceConfigPoolingNdBackward& config) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const override;
    PerformanceConfigPoolingNdBackward
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::pooling::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::pooling::ProblemDescription&,
                                  const PerformanceConfigPoolingNdBackward&) const override;
    PerformanceConfigPoolingNdBackward Search(const ExecutionContext& context,
                                              const miopen::pooling::ProblemDescription& problem,
                                              const AnyInvokeParams& invoke_context) const override
    {
        return GenericSearch(*this, context, problem, invoke_context);
    }
};

template <class Inner>
struct PoolingBwdNCHWTransposingSolver
    : TransposingSolver<PoolingBwdNCHWTransposingSolver<Inner>,
                        PoolingTunableSolver<typename Inner::PerformanceConfigType>,
                        miopen::pooling::ProblemDescription,
                        miopen::pooling::BwdInvokeParams,
                        Inner>
{
    using Problem      = miopen::pooling::ProblemDescription;
    using InvokeParams = miopen::pooling::BwdInvokeParams;

    inline static auto GetTransposes(const Problem& /*problem*/)
    {
        return std::array<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, 2>{{
            {
                &Problem::GetXDesc,
                &InvokeParams::dxDesc,
                nullptr,
                &InvokeParams::dx, // dx is output
                "NCDHW",
                false,
            },
            {
                &Problem::GetYDesc,
                &InvokeParams::dyDesc,
                &InvokeParams::dy, // dy is input
                nullptr,
                "NCDHW",
                true,
            },
        }};
    }
};

struct TransposedPoolingBwd2d final : PoolingBwdNCHWTransposingSolver<PoolingBackward2d>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingBwd2d>();
    }
    PoolingBackward2d::PerformanceConfigType
    GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                const miopen::pooling::ProblemDescription& problem) const override
    {
        return PoolingBackward2d{}.GetDefaultPerformanceConfig(ctx, problem);
    }
    bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::pooling::ProblemDescription& problem,
                             const PoolingBackward2d::PerformanceConfigType& config) const override
    {
        return PoolingBackward2d{}.IsValidPerformanceConfig(ctx, problem, config);
    }
    PoolingBackward2d::PerformanceConfigType
    Search(const ExecutionContext& context,
           const miopen::pooling::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override
    {
        return PoolingBackward2d{}.Search(context, problem, invoke_context);
    }
};

struct TransposedPoolingBwdNd final : PoolingBwdNCHWTransposingSolver<PoolingBackwardNd>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<TransposedPoolingBwdNd>();
    }
    PoolingBackwardNd::PerformanceConfigType
    GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                const miopen::pooling::ProblemDescription& problem) const override
    {
        return PoolingBackwardNd{}.GetDefaultPerformanceConfig(ctx, problem);
    }
    bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::pooling::ProblemDescription& problem,
                             const PoolingBackwardNd::PerformanceConfigType& config) const override
    {
        return PoolingBackwardNd{}.IsValidPerformanceConfig(ctx, problem, config);
    }
    PoolingBackwardNd::PerformanceConfigType
    Search(const ExecutionContext& context,
           const miopen::pooling::ProblemDescription& problem,
           const AnyInvokeParams& invoke_context) const override
    {
        return PoolingBackwardNd{}.Search(context, problem, invoke_context);
    }
};

} // namespace pooling

} // namespace solver

} // namespace miopen
