// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Coverage of the fused-A2A benchmark path. The single-rank cases (world == 1)
// have a rank send to itself, so one device exercises argument parsing, the
// descriptor, solution selection, the launch and the receive check.
//
// The _smoke suffix on each suite is the ctest category token.

#include "a2a_bench.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <vector>

namespace
{
    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    Arguments single_rank_arguments()
    {
        Arguments arg;
        arg.init();
        arg.M[0]         = 4096;
        arg.N[0]         = 256;
        arg.K[0]         = 1024;
        arg.a2a_extent = 2048;
        arg.a2a_world  = 1;
        return arg;
    }
} // namespace

TEST(FusedA2ABench_smoke, single_rank_round_trip)
{
    if(!gpuAvailable())
        GTEST_SKIP() << "no GPU";

    hipblaslt_bench::LauncherEnv env;
    env.rank       = 0;
    env.world      = 1;
    env.local_rank = 0;

    hipblaslt_bench::TcpRendezvous rendezvous(env, 10);
    const Arguments                arg = single_rank_arguments();

    hipblaslt_bench::RankResources res;
    res.rendezvous = &rendezvous;
    ASSERT_TRUE(hipblaslt_bench::setup_rank(env, arg, res));

    hipblasLtMatmulHeuristicResult_t heur{};
    int algoCount = 0;
    ASSERT_EQ(hipblaslt_bench::select_algo(res, heur, algoCount), HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no fused GEMM+A2A solution in the loaded library";

    uint32_t                       launchCount = 0;
    hipblasStatus_t                lastStatus  = HIPBLAS_STATUS_SUCCESS;
    std::vector<hipblasLtBfloat16> gold, landed;
    auto launch = hipblaslt_bench::make_launch(res, heur, launchCount, lastStatus);

    launch(0);
    ASSERT_EQ(hipStreamSynchronize(res.stream), hipSuccess);
    ASSERT_EQ(lastStatus, HIPBLAS_STATUS_SUCCESS);

    EXPECT_TRUE(hipblaslt_bench::check_recv(env, arg, res, gold, landed));
}

// Four launches over two channels walk 0, 1, 0, 1, so each channel is reused once.
TEST(FusedA2ABench_smoke, reused_channels_stay_correct)
{
    if(!gpuAvailable())
        GTEST_SKIP() << "no GPU";

    hipblaslt_bench::LauncherEnv env;
    env.world = 1;

    hipblaslt_bench::TcpRendezvous rendezvous(env, 10);
    const Arguments                arg = single_rank_arguments();

    hipblaslt_bench::RankResources res;
    res.rendezvous = &rendezvous;
    ASSERT_TRUE(hipblaslt_bench::setup_rank(env, arg, res));

    hipblasLtMatmulHeuristicResult_t heur{};
    int algoCount = 0;
    ASSERT_EQ(hipblaslt_bench::select_algo(res, heur, algoCount), HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no fused GEMM+A2A solution in the loaded library";

    uint32_t                       launchCount = 0;
    hipblasStatus_t                lastStatus  = HIPBLAS_STATUS_SUCCESS;
    std::vector<hipblasLtBfloat16> gold, landed;
    auto launch = hipblaslt_bench::make_launch(res, heur, launchCount, lastStatus);

    const size_t recvBytes = size_t(arg.a2a_world) * arg.N[0] * hipblaslt_bench::shard_of(arg)
                             * sizeof(hipblasLtBfloat16);

    for(int i = 0; i < 4; ++i)
    {
        ASSERT_EQ(hipMemset(res.dRecv, 0, recvBytes), hipSuccess);
        launch(i);
        ASSERT_EQ(hipStreamSynchronize(res.stream), hipSuccess);
        EXPECT_TRUE(hipblaslt_bench::check_recv(env, arg, res, gold, landed)) << "launch " << i;
    }

    EXPECT_EQ(lastStatus, HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(launchCount, 4u);
}

TEST(FusedA2ABench_smoke, mismatched_local_rank_is_rejected)
{
    hipblaslt_bench::LauncherEnv env;
    env.rank       = 0;
    env.world      = 2;
    env.local_rank = 1;

    Arguments arg = single_rank_arguments();
    arg.a2a_world = 2;

    EXPECT_FALSE(hipblaslt_bench::peers_reachable(env, arg));
}
