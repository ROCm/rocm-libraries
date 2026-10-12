// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/// Kernel identity and selection policy shared by the grouped CK convolution solvers. The CK
/// plugin ABI (CkImplLibLoader) takes a CK type string and an explicit use_tf32 flag; this layer
/// owns how MIOpen names, orders and validates kernels on top of it.

namespace miopen {
namespace solver {

/// CK's fp32 and tf32 instances of a kernel share one GetTypeString(), so the type string alone
/// does not say which instance runs. MIOpen prefixes the ids of tf32 instances with kTf32Tag: a
/// kernel id names exactly one instance and keeps its compute type through perf-db round trips.
namespace ck_kernel_id {

inline constexpr std::string_view kTf32Tag = "tf32:";

inline bool IsTf32(std::string_view id) { return id.starts_with(kTf32Tag); }

/// The CK type string (with any "+split_k" suffix) the id names.
inline std::string_view TypeString(std::string_view id)
{
    return IsTf32(id) ? id.substr(kTf32Tag.size()) : id;
}

} // namespace ck_kernel_id

/// The kernels CK has for the problem: its tf32 instances when tf32, else its own-type ones.
inline std::vector<std::string> CkKernelIds(const CkImplLibLoader& loader,
                                            CKSolverType solver,
                                            const ::miopen::conv::ProblemDescription& problem,
                                            miopenDataType_t dtype,
                                            bool tf32)
{
    auto ids = loader.FillValidKernels(solver, problem, dtype, tf32);
    if(tf32)
    {
        for(auto& id : ids)
            id.insert(0, ck_kernel_id::kTf32Tag);
    }
    return ids;
}

/// Candidates for default and heuristic selection: the tf32 kernels when the problem allows TF32
/// and CK has any, otherwise the full-precision kernels.
inline std::vector<std::string>
CkPreferredKernelIds(const CkImplLibLoader& loader,
                     CKSolverType solver,
                     const ::miopen::conv::ProblemDescription& problem,
                     miopenDataType_t dtype)
{
    if(dtype == miopenFloat && problem.UseTF32())
    {
        auto tf32 = CkKernelIds(loader, solver, problem, dtype, true);
        if(!tf32.empty())
            return tf32;
    }
    return CkKernelIds(loader, solver, problem, dtype, false);
}

/// Tuning candidates: the tf32 kernels (when the problem allows TF32) followed by the
/// full-precision kernels, so a search can pick either. TF32 is a precision floor that full
/// precision satisfies.
inline std::vector<std::string> CkSearchKernelIds(const CkImplLibLoader& loader,
                                                  CKSolverType solver,
                                                  const ::miopen::conv::ProblemDescription& problem,
                                                  miopenDataType_t dtype)
{
    auto ids  = (dtype == miopenFloat && problem.UseTF32())
                    ? CkKernelIds(loader, solver, problem, dtype, true)
                    : std::vector<std::string>{};
    auto full = CkKernelIds(loader, solver, problem, dtype, false);
    ids.insert(ids.end(), full.begin(), full.end());
    return ids;
}

/// Whether CK can run the problem with any kernel: a tf32 one when the problem allows TF32, or a
/// full-precision one.
inline bool IsCkApplicable(const CkImplLibLoader& loader,
                           CKSolverType solver,
                           const ::miopen::conv::ProblemDescription& problem,
                           miopenDataType_t dtype)
{
    const bool tf32 = dtype == miopenFloat && problem.UseTF32();
    return (tf32 && loader.IsApplicable(solver, problem, dtype, true)) ||
           loader.IsApplicable(solver, problem, dtype, false);
}

/// Workspace for whichever kernel runs: the larger of the full-precision and (when the problem
/// allows TF32) tf32 requirements.
inline std::size_t CkWorkspaceSize(const CkImplLibLoader& loader,
                                   CKSolverType solver,
                                   const ::miopen::conv::ProblemDescription& problem,
                                   miopenDataType_t dtype)
{
    auto ws = loader.GetWorkspaceSize(solver, problem, dtype, false);
    if(dtype == miopenFloat && problem.UseTF32())
        ws = std::max(ws, loader.GetWorkspaceSize(solver, problem, dtype, true));
    return ws;
}

/// Whether the kernel supports the problem. A tf32 kernel is below the fp32 precision floor, so
/// only a problem that allows TF32 may run it.
inline bool IsCkKernelSupported(const CkImplLibLoader& loader,
                                CKSolverType solver,
                                const ::miopen::conv::ProblemDescription& problem,
                                std::string_view id,
                                miopenDataType_t dtype)
{
    const bool tf32 = ck_kernel_id::IsTf32(id);
    if(tf32 && !(dtype == miopenFloat && problem.UseTF32()))
        return false;
    return loader.IsArgsSupported(
        solver, problem, std::string{ck_kernel_id::TypeString(id)}, dtype, tf32);
}

inline ConvSolution GetCkKernelSolution(const CkImplLibLoader& loader,
                                        CKSolverType solver,
                                        const ExecutionContext& ctx,
                                        const ::miopen::conv::ProblemDescription& problem,
                                        std::string_view id)
{
    return loader.GetSolution(
        solver, ctx, problem, std::string{ck_kernel_id::TypeString(id)}, ck_kernel_id::IsTf32(id));
}

/// For RunAIHeuristics: a creator of validators that check a (kernel id, split_k) pair with CK.
inline auto MakeCkKernelValidatorCreator(const CkImplLibLoader& loader,
                                         CKSolverType solver,
                                         miopenDataType_t dtype)
{
    return [&loader, solver, dtype](const ::miopen::conv::ProblemDescription& p) {
        return [&loader, solver, dtype, &p](const std::string& id, int split_k) {
            return IsCkKernelSupported(
                loader, solver, p, id + "+" + std::to_string(split_k), dtype);
        };
    };
}

/// Index of the first kernel id naming the CK type string ck_type, of either compute type.
inline std::optional<std::size_t> FindCkKernel(const std::vector<std::string>& ids,
                                               std::string_view ck_type)
{
    const auto it = std::find_if(ids.begin(), ids.end(), [&](const std::string& id) {
        return ck_kernel_id::TypeString(id) == ck_type;
    });
    if(it == ids.end())
        return std::nullopt;
    return static_cast<std::size_t>(it - ids.begin());
}

/// The grouped CK solvers choose the compute precision through their tuned kernel id, so a
/// full-precision perf-db record would hide their tf32 instances -- unless CK has none for the
/// problem.
inline bool CkPerfConfigSelectsPrecision(const ExecutionContext& ctx,
                                         const ::miopen::conv::ProblemDescription& problem,
                                         CKSolverType solver)
{
    const auto dtype = problem.GetInDataType();
    if(dtype != miopenFloat || !problem.UseTF32())
        return false;
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    return !CkKernelIds(loader, solver, problem, dtype, true).empty();
}

} // namespace solver
} // namespace miopen
