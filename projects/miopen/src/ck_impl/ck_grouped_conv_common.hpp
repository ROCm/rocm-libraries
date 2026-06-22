// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <miopen/solver/problem_description_interpreter.hpp>

/// Definition of the opaque handle used by the extern "C" interface.
/// Shared across all direction implementation files (fwd, bwd, wrw) and
/// defined in exactly one place to avoid ODR issues when linked into the
/// same shared library.
struct CKKernelListHandle
{
    std::vector<std::string> kernels;
};

/// Convolution dimensions extracted from a ProblemDescription.
/// Pure integer struct with no CK dependency, so it can live in this
/// shared header included by all direction impl files.
///
/// Members are int64 so downstream stride builders (e.g. Hi*Wi*G*C in the
/// NCHW layout path) do not silently overflow on tensors whose contiguous
/// stride exceeds INT_MAX (ROCM-23997).
struct CKConvDims
{
    int64_t G, N, K, C, C1, K1, Hi, Wi, Ho, Wo, Y, X;
};

/// Extract convolution dimensions from a ProblemDescription.
inline CKConvDims ExtractConvDims(const miopen::conv::ProblemDescription& problem)
{
    using miopen::solver::ProblemInterpreter;
    CKConvDims d;
    d.G  = ProblemInterpreter::GetGroupCountG(problem);
    d.N  = ProblemInterpreter::GetBatchN(problem);
    d.K1 = ProblemInterpreter::GetOutputChannelK(problem);
    d.C1 = ProblemInterpreter::GetInputChannelC(problem);
    d.C  = d.C1 / d.G;
    d.K  = d.K1 / d.G;
    d.Hi = ProblemInterpreter::GetInputHeightHi(problem);
    d.Wi = ProblemInterpreter::GetInputWidthWi(problem);
    d.Ho = ProblemInterpreter::GetOutputHeightHo(problem);
    d.Wo = ProblemInterpreter::GetOutputWidthWo(problem);
    d.Y  = ProblemInterpreter::GetFilterHeightY(problem);
    d.X  = ProblemInterpreter::GetFilterWidthX(problem);
    return d;
}
