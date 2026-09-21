// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The spatial batch-norm tuning heuristic must only ever propose vector sizes
// that the kernel's vector machinery (mapped_vector_type<T, N> and the miopen
// vector math / cast<> helpers) actually implements. Originally the heuristic
// could select a vector size of 8 while only sizes 1/2/4 were implemented, which
// instantiated mapped_vector_type<float, 8> (no specialization) and aborted the
// HIPRTC compile.
//
// This build supports vector sizes 1/2/4/8, so kMaxVectorSize equals
// kMaxSupportedVectorSize (currently 8). The tests
// assert the heuristic never proposes a size outside the implemented set. They
// are pure-CPU: they only run the solver's performance-config generation (which
// is handle-free), so they catch a heuristic/kernel mismatch deterministically
// without a GPU or kernel compilation.

#include <gtest/gtest.h>

#include <miopen/batchnorm/solvers.hpp>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batchnorm/common_spatial.hpp>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>
#include <miopen/execution_context.hpp>

#include <string>
#include <vector>

using namespace miopen;
using namespace miopen::batchnorm;
using namespace miopen::solver::batchnorm;

namespace {

// Largest vector size the batch-norm kernels implement (mapped_vector_type<T, N>
// and the vector math / cast<> helpers). Sourced from common_spatial.hpp so that
// adding a new vector width only requires one update (there, not here too).
constexpr size_t kMaxVectorSize = miopen::solver::batchnorm::kMaxSupportedVectorSize;

bool IsSupportedVectorSize(size_t v) { return v == 1 || v == 2 || v == 4 || v == kMaxVectorSize; }

struct Shape
{
    size_t n, c, h, w;
    std::string name;
};

// Shapes chosen to exercise the NHWC vector-size heuristic, including the ones
// that select vectorsize 8 -- both the non-power-of-2 fallback (ticket shape,
// c=24) and the power-of-2 branches in GetHeuristicsConfigTuningNHWC.
std::vector<Shape> GetShapes()
{
    return {
        {42, 24, 240, 320, "c24_nonpow2"},
        {16, 256, 64, 64, "c256_hw4096"},
        {8, 512, 32, 32, "c512_hw1024"},
        {8, 1024, 16, 16, "c1024_hw256"},
        {8, 2048, 17, 17, "c2048_hw289"},
        {64, 256, 56, 56, "c256_hw3136"},
        {16, 8, 128, 256, "c8_smallvec"}, // stays at a small vector size
    };
}

ActivationDescriptor PassthruActiv()
{
    return ActivationDescriptor(miopenActivationPASTHRU, 1.0, 1.0, 1.0);
}

TensorDescriptor MakeDataDesc(miopenDataType_t t, miopenTensorLayout_t layout, const Shape& s)
{
    return TensorDescriptor(t, layout, {s.n, s.c, s.h, s.w});
}

// scale/bias/mean/var are per-channel fp32
TensorDescriptor MakeStatsDesc(const Shape& s)
{
    return TensorDescriptor(miopenFloat, {std::size_t{1}, s.c, std::size_t{1}, std::size_t{1}});
}

ProblemDescription MakeFwdProblem(miopenDataType_t dt, miopenTensorLayout_t layout, const Shape& s)
{
    const auto x     = MakeDataDesc(dt, layout, s);
    const auto y     = MakeDataDesc(dt, layout, s);
    const auto stats = MakeStatsDesc(s);
    const auto act   = PassthruActiv();
    return ProblemDescription(miopenBNSpatial,
                              x,
                              y,
                              stats,
                              stats,
                              stats,
                              stats,
                              /*expAvgFactor*/ 1.0,
                              /*epsilon*/ 1e-5,
                              /*resultsave*/ true,
                              /*resultrunning*/ true,
                              /*min_workgroups*/ 1,
                              act);
}

ProblemDescription MakeBwdProblem(miopenDataType_t dt, miopenTensorLayout_t layout, const Shape& s)
{
    const auto x     = MakeDataDesc(dt, layout, s);
    const auto dy    = MakeDataDesc(dt, layout, s);
    const auto dx    = MakeDataDesc(dt, layout, s);
    const auto stats = MakeStatsDesc(s);
    const auto act   = PassthruActiv();
    return ProblemDescription(miopenBNSpatial,
                              x,
                              dy,
                              dx,
                              stats,
                              stats,
                              stats,
                              stats,
                              /*epsilon*/ 1e-5,
                              /*useSaved*/ false,
                              /*min_workgroups*/ 1,
                              act);
}

size_t VectorSizeOf(const std::string& kernel_id)
{
    int variant       = -1;
    size_t vectorsize = 1, xls = 1, yls = 1, zls = 1, nelem = 1;
    GetVariantFromKernelId(kernel_id, variant, vectorsize, xls, yls, zls, nelem);
    return vectorsize;
}

} // namespace

class CPU_BatchNormSpatialVectorSize_NONE : public ::testing::TestWithParam<Shape>
{
};

// The NHWC tuning heuristic must never propose a vector size the kernel does not
// implement.
TEST_P(CPU_BatchNormSpatialVectorSize_NONE, HeuristicNeverExceedsMax)
{
    const auto& s = GetParam();

    const ProblemDescription problems[] = {
        MakeFwdProblem(miopenBFloat16, miopenTensorNHWC, s),
        MakeBwdProblem(miopenBFloat16, miopenTensorNHWC, s),
    };
    // The heuristic depends only on the tensor dimensions and direction, so the
    // data type is irrelevant here.
    for(const auto& problem : problems)
    {
        size_t vectorsize = 1, xlocalsize = 64;
        GetHeuristicsConfigTuningNHWC(problem, vectorsize, xlocalsize);
        EXPECT_LE(vectorsize, kMaxVectorSize)
            << s.name << ": heuristic selected unsupported vector size " << vectorsize;
        EXPECT_TRUE(IsSupportedVectorSize(vectorsize))
            << s.name << ": vector size " << vectorsize << " is not an implemented size";
    }
}

// End-to-end: every kernel the solver would try during tuning (both fwd and bwd,
// both layouts, all supported data types) must have an implemented vector size.
TEST_P(CPU_BatchNormSpatialVectorSize_NONE, GeneratedConfigsNeverExceedMax)
{
    const auto& s = GetParam();
    ExecutionContext ctx;

    for(auto dt : {miopenFloat, miopenHalf, miopenBFloat16})
    {
        for(auto layout : {miopenTensorNCHW, miopenTensorNHWC})
        {
            {
                const auto problem = MakeFwdProblem(dt, layout, s);
                BnFwdTrainingSpatial solver;
                if(!solver.IsApplicable(ctx, problem))
                    continue;
                const auto cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
                ASSERT_FALSE(cfg.valid_kernels.empty()) << s.name;
                for(const auto& kid : cfg.valid_kernels)
                {
                    EXPECT_TRUE(IsSupportedVectorSize(VectorSizeOf(kid)))
                        << "fwd " << s.name << " layout=" << layout << " kernel_id=" << kid;
                }
            }
            {
                const auto problem = MakeBwdProblem(dt, layout, s);
                BnBwdTrainingSpatial solver;
                if(!solver.IsApplicable(ctx, problem))
                    continue;
                const auto cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
                ASSERT_FALSE(cfg.valid_kernels.empty()) << s.name;
                for(const auto& kid : cfg.valid_kernels)
                {
                    EXPECT_TRUE(IsSupportedVectorSize(VectorSizeOf(kid)))
                        << "bwd " << s.name << " layout=" << layout << " kernel_id=" << kid;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_BatchNormSpatialVectorSize_NONE,
                         ::testing::ValuesIn(GetShapes()),
                         [](const ::testing::TestParamInfo<Shape>& info) {
                             return info.param.name;
                         });
