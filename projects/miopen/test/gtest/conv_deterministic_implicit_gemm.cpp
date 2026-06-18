// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Bit-exact determinism tests for implicit GEMM convolution solvers.
// Runs each solver 3 times with identical input and verifies bit-exact output match.

#include <cstring>
#include <gtest/gtest.h>
#include <miopen/bfloat16.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include "../random.hpp"
#include "get_handle.hpp"
#include "../driver/tensor_driver.hpp"
#include "conv_common.hpp"
#include "gtest_common.hpp"

namespace {

using Direction = miopen::conv::Direction;

struct DeterministicTestConfig
{
    size_t N;
    size_t C;
    size_t K;
    size_t H;
    size_t W;
    size_t y;
    size_t x;
    size_t pad_h;
    size_t pad_w;
    size_t stride_h;
    size_t stride_w;
    size_t dilation_h;
    size_t dilation_w;

    friend std::ostream& operator<<(std::ostream& os, const DeterministicTestConfig& tc)
    {
        os << "N:" << tc.N << " C:" << tc.C << " K:" << tc.K;
        os << " H:" << tc.H << " W:" << tc.W;
        os << " y:" << tc.y << " x:" << tc.x;
        os << " pad:{" << tc.pad_h << "," << tc.pad_w << "}";
        os << " stride:{" << tc.stride_h << "," << tc.stride_w << "}";
        os << " dilation:{" << tc.dilation_h << "," << tc.dilation_w << "}";
        return os;
    }
};

template <typename T, Direction CONV_DIR, typename SolverType>
class GPU_ConvDeterministicImplicitGemm : public ::testing::TestWithParam<DeterministicTestConfig>
{
protected:
    static constexpr int NUM_ITERATIONS = 3;

    // Fix PRNG seed so that input tensor data is identical across runs and machines,
    // ensuring reproducibility of any determinism failure.
    void SetUp() override
    {
        prng::reset_seed();
        // Enable Xdlops solvers (disabled by default on some GPUs)
        putenv(const_cast<char*>("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1"));
    }

    void RunTest()
    {
        const auto& config = GetParam();
        auto& handle       = get_handle();

        // Create tensors (NCHW layout, non-grouped)
        std::vector<size_t> input_dims  = {config.N, config.C, config.H, config.W};
        std::vector<size_t> weight_dims = {config.K, config.C, config.y, config.x};

        tensor<T> input{miopenTensorNCHW, input_dims};
        tensor<T> weights{miopenTensorNCHW, weight_dims};

        // Create convolution descriptor with deterministic mode enabled
        auto conv_desc = miopen::ConvolutionDescriptor{
            2, // num spatial dims
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(config.pad_h), static_cast<int>(config.pad_w)},
            {static_cast<int>(config.stride_h), static_cast<int>(config.stride_w)},
            {static_cast<int>(config.dilation_h), static_cast<int>(config.dilation_w)},
            {0, 0}, // adj
            1,      // group_count
            1.0};   // lowp_quant

        // Enable deterministic mode
        conv_desc.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
        ASSERT_TRUE(conv_desc.attribute.deterministic.Get() == 1);

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        tensor<T> output{miopenTensorNCHW, output_desc.GetLengths()};

        // Initialize tensors
        if constexpr(CONV_DIR == Direction::Forward)
        {
            input.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
            });
            weights.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
            });
            std::fill(output.begin(), output.end(), T(0));
        }
        else if constexpr(CONV_DIR == Direction::BackwardData)
        {
            output.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
            });
            weights.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
            });
            std::fill(input.begin(), input.end(), T(0));
        }
        else // BackwardWeights
        {
            input.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.1), static_cast<T>(0.1));
            });
            output.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.01), static_cast<T>(0.1));
            });
            std::fill(weights.begin(), weights.end(), T{0});
        }

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(output.data);

        // Create solver and problem
        SolverType solv{};
        auto ctx = miopen::ExecutionContext{&handle};

        miopen::conv::ProblemDescription problem = [&]() {
            if constexpr(CONV_DIR == Direction::Forward)
                return miopen::conv::ProblemDescription{
                    input.desc, weights.desc, output.desc, conv_desc, CONV_DIR};
            else
                return miopen::conv::ProblemDescription{
                    output.desc, weights.desc, input.desc, conv_desc, CONV_DIR};
        }();

        problem.SetupFloats(ctx);

        const bool applicable = solv.IsApplicable(ctx, problem);
        if(!applicable)
        {
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable on this GPU/config";
        }

        Workspace wspace{};
        if(solv.MayNeedWorkspace())
        {
            wspace.resize(solv.GetWorkspaceSize(ctx, problem));
        }

        auto perf_config = solv.GetDefaultPerformanceConfig(ctx, problem);
        auto sol         = solv.GetSolution(ctx, problem, perf_config);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);

        // Store results from each iteration
        std::vector<std::vector<T>> results;
        results.reserve(NUM_ITERATIONS);

        for(int i = 0; i < NUM_ITERATIONS; ++i)
        {
            // Reset result buffer before each run
            if constexpr(CONV_DIR == Direction::Forward)
            {
                std::fill(output.begin(), output.end(), T(0));
                out_dev = handle.Write(output.data);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                std::fill(input.begin(), input.end(), T(0));
                in_dev = handle.Write(input.data);
            }
            else
            {
                std::fill(weights.begin(), weights.end(), T(0));
                wei_dev = handle.Write(weights.data);
            }

            // Execute kernel
            if constexpr(CONV_DIR == Direction::Forward)
            {
                auto invoke_params =
                    miopen::conv::DataInvokeParams{miopen::ConvFwdTensors{input.desc,
                                                                          in_dev.get(),
                                                                          weights.desc,
                                                                          wei_dev.get(),
                                                                          output.desc,
                                                                          out_dev.get()},
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   false};
                (invoker)(handle, invoke_params);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                auto invoke_params =
                    miopen::conv::DataInvokeParams{miopen::ConvDataTensors{output.desc,
                                                                           out_dev.get(),
                                                                           weights.desc,
                                                                           wei_dev.get(),
                                                                           input.desc,
                                                                           in_dev.get()},
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   false};
                (invoker)(handle, invoke_params);
            }
            else // BackwardWeights
            {
                auto invoke_params =
                    miopen::conv::WrWInvokeParams{miopen::ConvWrwTensors{output.desc,
                                                                         out_dev.get(),
                                                                         input.desc,
                                                                         in_dev.get(),
                                                                         weights.desc,
                                                                         wei_dev.get()},
                                                  wspace.ptr(),
                                                  wspace.size(),
                                                  false};
                (invoker)(handle, invoke_params);
            }

            handle.Finish();

            // Read result
            if constexpr(CONV_DIR == Direction::Forward)
            {
                handle.ReadToVec(out_dev, output.data);
                results.push_back(output.data);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                handle.ReadToVec(in_dev, input.data);
                results.push_back(input.data);
            }
            else
            {
                handle.ReadToVec(wei_dev, weights.data);
                results.push_back(weights.data);
            }
        }

        // Verify bit-exact determinism: compare all iterations against first run
        const auto& reference = results[0];
        for(int i = 1; i < NUM_ITERATIONS; ++i)
        {
            const auto& current = results[i];
            ASSERT_EQ(reference.size(), current.size());

            bool match            = true;
            size_t first_mismatch = 0;
            for(size_t j = 0; j < reference.size(); ++j)
            {
                if(std::memcmp(&reference[j], &current[j], sizeof(T)) != 0)
                {
                    match          = false;
                    first_mismatch = j;
                    break;
                }
            }

            ASSERT_TRUE(match) << "Bit-exact mismatch at iteration " << i << ", element "
                               << first_mismatch << ": reference = " << reference[first_mismatch]
                               << ", current = " << current[first_mismatch];
        }
    }
};

// Tests that IsApplicable returns false in deterministic mode for shapes that
// would use non-deterministic AtomicAdd.
template <typename T, Direction CONV_DIR, typename SolverType>
class GPU_ConvDeterministicRejection : public ::testing::TestWithParam<DeterministicTestConfig>
{
protected:
    void SetUp() override
    {
        // Enable Xdlops solvers (disabled by default on some GPUs)
        putenv(const_cast<char*>("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1"));
    }

    void RunTest()
    {
        const auto& config = GetParam();
        auto& handle       = get_handle();

        std::vector<size_t> input_dims  = {config.N, config.C, config.H, config.W};
        std::vector<size_t> weight_dims = {config.K, config.C, config.y, config.x};

        tensor<T> input{miopenTensorNCHW, input_dims};
        tensor<T> weights{miopenTensorNCHW, weight_dims};

        auto conv_desc = miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(config.pad_h), static_cast<int>(config.pad_w)},
            {static_cast<int>(config.stride_h), static_cast<int>(config.stride_w)},
            {static_cast<int>(config.dilation_h), static_cast<int>(config.dilation_w)},
            {0, 0},
            1,
            1.0};

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        tensor<T> output{miopenTensorNCHW, output_desc.GetLengths()};

        SolverType solv{};
        auto ctx = miopen::ExecutionContext{&handle};

        // Without deterministic mode: solver should be applicable (or skip if GPU mismatch)
        miopen::conv::ProblemDescription problem_ndet = [&]() {
            if constexpr(CONV_DIR == Direction::Forward)
                return miopen::conv::ProblemDescription{
                    input.desc, weights.desc, output.desc, conv_desc, CONV_DIR};
            else
                return miopen::conv::ProblemDescription{
                    output.desc, weights.desc, input.desc, conv_desc, CONV_DIR};
        }();
        problem_ndet.SetupFloats(ctx);
        const bool applicable_ndet = solv.IsApplicable(ctx, problem_ndet);
        if(!applicable_ndet)
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable on this GPU/config";

        // With deterministic mode: solver must reject this shape
        conv_desc.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
        ASSERT_TRUE(conv_desc.attribute.deterministic.Get() == 1);

        miopen::conv::ProblemDescription problem_det = [&]() {
            if constexpr(CONV_DIR == Direction::Forward)
                return miopen::conv::ProblemDescription{
                    input.desc, weights.desc, output.desc, conv_desc, CONV_DIR};
            else
                return miopen::conv::ProblemDescription{
                    output.desc, weights.desc, input.desc, conv_desc, CONV_DIR};
        }();
        problem_det.SetupFloats(ctx);

        const bool applicable_det = solv.IsApplicable(ctx, problem_det);
        EXPECT_FALSE(applicable_det)
            << solv.SolverDbId() << " must reject non-deterministic shape in deterministic mode";
    }
};

} // namespace

// ============================================================================
// Type aliases for all 7 solvers (BFP16)
// GPU_ConvDeterministic_* naming matches TheRock CI positive filter */GPU_Conv*
// ============================================================================

// Forward solvers
using GPU_ConvDeterministic_FwdV4R1_BFP16 =
    GPU_ConvDeterministicImplicitGemm<bfloat16,
                                      Direction::Forward,
                                      miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd>;

using GPU_ConvDeterministic_FwdV4R4Xdlops_BFP16 =
    GPU_ConvDeterministicImplicitGemm<bfloat16,
                                      Direction::Forward,
                                      miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops>;

using GPU_ConvDeterministic_FwdV4R5Xdlops_BFP16 =
    GPU_ConvDeterministicImplicitGemm<bfloat16,
                                      Direction::Forward,
                                      miopen::solver::conv::ConvHipImplicitGemmForwardV4R5Xdlops>;

// Backward Data solvers
using GPU_ConvDeterministic_BwdV1R1_BFP16 =
    GPU_ConvDeterministicImplicitGemm<bfloat16,
                                      Direction::BackwardData,
                                      miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1>;

using GPU_ConvDeterministic_BwdV4R1_BFP16 =
    GPU_ConvDeterministicImplicitGemm<bfloat16,
                                      Direction::BackwardData,
                                      miopen::solver::conv::ConvHipImplicitGemmBwdDataV4R1>;

// Weight-update solvers
using GPU_ConvDeterministic_WrwV4R4_BFP16 =
    GPU_ConvDeterministicImplicitGemm<bfloat16,
                                      Direction::BackwardWeights,
                                      miopen::solver::conv::ConvHipImplicitGemmV4R4WrW>;

using GPU_ConvDeterministic_WrwV4R4Xdlops_BFP16 =
    GPU_ConvDeterministicImplicitGemm<bfloat16,
                                      Direction::BackwardWeights,
                                      miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops>;

// Rejection tests: verify IsApplicable returns false for non-deterministic shapes
using GPU_ConvDeterministicRejection_BwdV1R1_BFP16 =
    GPU_ConvDeterministicRejection<bfloat16,
                                   Direction::BackwardData,
                                   miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1>;

using GPU_ConvDeterministicRejection_WrwV4R4Xdlops_BFP16 =
    GPU_ConvDeterministicRejection<bfloat16,
                                   Direction::BackwardWeights,
                                   miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops>;

// ============================================================================
// Test definitions
// ============================================================================

// Forward
TEST_P(GPU_ConvDeterministic_FwdV4R1_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_ConvDeterministic_FwdV4R4Xdlops_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_ConvDeterministic_FwdV4R5Xdlops_BFP16, DeterministicTest) { this->RunTest(); };

// Backward Data
TEST_P(GPU_ConvDeterministic_BwdV1R1_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_ConvDeterministic_BwdV4R1_BFP16, DeterministicTest) { this->RunTest(); };

// Weight-update
TEST_P(GPU_ConvDeterministic_WrwV4R4_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_ConvDeterministic_WrwV4R4Xdlops_BFP16, DeterministicTest) { this->RunTest(); };

// Rejection tests: IsApplicable must return false for non-deterministic shapes
TEST_P(GPU_ConvDeterministicRejection_BwdV1R1_BFP16, RejectionTest) { this->RunTest(); };
TEST_P(GPU_ConvDeterministicRejection_WrwV4R4Xdlops_BFP16, RejectionTest) { this->RunTest(); };

// ============================================================================
// Test configs — small shapes for fast CI execution.
// Tests GTEST_SKIP if solver is not applicable on the current GPU.
// Config: {N, C, K, H, W, y, x, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w}
// ============================================================================

// Forward solvers
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministic_FwdV4R1_BFP16,
                         testing::Values(DeterministicTestConfig{
                             64, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministic_FwdV4R4Xdlops_BFP16,
                         testing::Values(DeterministicTestConfig{
                             128, 48, 192, 13, 13, 1, 1, 0, 0, 1, 1, 1, 1}));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministic_FwdV4R5Xdlops_BFP16,
                         testing::Values(DeterministicTestConfig{
                             64, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}));

// Backward Data solvers
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministic_BwdV1R1_BFP16,
                         testing::Values(DeterministicTestConfig{
                             64, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministic_BwdV4R1_BFP16,
                         testing::Values(DeterministicTestConfig{
                             64, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}));

// Weight-update solvers
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministic_WrwV4R4_BFP16,
                         testing::Values(DeterministicTestConfig{
                             1, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}));

// WrwV4R4Xdlops: N=2, C=192, K=16, H=16, W=16 → gemm_k_block=2 (boundary test)
// N=2 allows split into 2 blocks, smaller H/W forces split-K
// This tests the boundary condition: gemm_k_block=2 should be ACCEPTED in deterministic mode
// (deterministic allows k_block <= 2, rejects k_block > 2)
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministic_WrwV4R4Xdlops_BFP16,
                         testing::Values(DeterministicTestConfig{
                             2, 192, 16, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}));

// Rejection test configs
// BwdV1R1: stride=1 < threshold=dilation*(y-1)+1=3 → AtomicAdd path → must be rejected
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministicRejection_BwdV1R1_BFP16,
                         testing::Values(DeterministicTestConfig{
                             //  N   C   K   H   W  y  x  ph pw sh sw dh dw
                             64,
                             64,
                             64,
                             14,
                             14,
                             3,
                             3,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1}));

// WrwV4R4Xdlops: N=8, H=W=8 → gemm_k_block=4 > 2 → AtomicAdd → must be rejected
// (N=4,H=28,W=28 produced gemm_k_block=1 due to divisibility, so changed to this config)
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvDeterministicRejection_WrwV4R4Xdlops_BFP16,
                         testing::Values(DeterministicTestConfig{
                             //  N   C   K   H   W  y  x  ph pw sh sw dh dw
                             8,
                             192,
                             16,
                             8,
                             8,
                             1,
                             1,
                             0,
                             0,
                             1,
                             1,
                             1,
                             1}));
