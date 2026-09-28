// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Multi-process coverage of the fused-A2A path. Each rank is a separate
// process, so the peers reach each other through hipIpcOpenMemHandle, which the
// ranks of a single process never do.
//
// The suffix on each suite name is the ctest category token.

#include "a2a_bench.hpp"
#include "testing_multi_gpu.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <signal.h>
#include <sys/prctl.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kWorld          = 2;
    constexpr uint32_t kLaunches       = 4;
    constexpr int      kRankTimeoutSec = 180;

    constexpr uint32_t kRankChildDefaultLaunches = 1;

    uint32_t rank_child_launches()
    {
        if(const char* v = std::getenv("A2A_LAUNCHES"))
        {
            const int parsed = std::atoi(v);
            if(parsed > 0)
                return uint32_t(parsed);
        }
        return kRankChildDefaultLaunches;
    }

    Arguments rank_child_arguments(const hipblaslt_bench::LauncherEnv& env)
    {
        Arguments arg;
        arg.init();
        arg.M[0]       = 4096;
        arg.N[0]       = 256;
        arg.K[0]       = 1024;
        arg.a2a_extent = 2048;
        arg.a2a_world  = uint8_t(env.world);
        return arg;
    }
} // namespace

namespace hipblaslt_bench
{
    // One rank of the tests below, run as the --a2a-rank-child role.
    int run_rank_child()
    try
    {
        // SIGKILL once the spawning test process is gone.
        prctl(PR_SET_PDEATHSIG, SIGKILL);

        const LauncherEnv env = read_launcher_env();

        const Arguments arg = rank_child_arguments(env);

        TcpRendezvous rendezvous(env, kRendezvousTimeoutSec);

        if(!join_group(env, rendezvous, HIPBLASLT_DEVICE_COMM_MAX_WORLD))
            return kRankChildSkipped;

        const CollectiveAgreement agreement = make_agreement(rendezvous, env.world);

        RankResources res;
        res.rendezvous = &rendezvous;
        if(!setup_rank(env, arg, res))
            return kRankChildFailed;

        hipblasLtMatmulHeuristicResult_t heur{};
        int                              algoCount  = 0;
        const hipblasStatus_t            algoStatus = select_algo(res, heur, algoCount);
        if(!agreement.agree(algoStatus == HIPBLAS_STATUS_SUCCESS, std::logical_and<>{}))
        {
            if(algoStatus != HIPBLAS_STATUS_SUCCESS)
                hipblaslt_cerr << "error: hipblasLtMatmulAlgoGetHeuristic -> " << int(algoStatus)
                               << "\n";
            return kRankChildFailed;
        }

        if(!agreement.agree(algoCount > 0, std::logical_and<>{}))
        {
            hipblaslt_cout << "skipped: no fused GEMM+A2A solution in the loaded library\n";
            return kRankChildSkipped;
        }

        uint32_t                       launchCount = 0;
        hipblasStatus_t                lastStatus  = HIPBLAS_STATUS_SUCCESS;
        std::vector<hipblasLtBfloat16> gold, landed;
        auto                           launch = make_launch(res, heur, launchCount, lastStatus);

        const size_t recvBytes
            = size_t(arg.a2a_world) * arg.N[0] * shard_of(arg) * sizeof(hipblasLtBfloat16);

        // A failed launch still runs the rest of the iteration.
        bool           verified = true;
        const uint32_t launches = rank_child_launches();
        for(uint32_t i = 0; i < launches && verified; ++i)
        {
            const bool cleared = hipMemset(res.dRecv, 0, recvBytes) == hipSuccess;

            launch(int64_t(i));

            const bool synced          = hipStreamSynchronize(res.stream) == hipSuccess;
            const bool landedCorrectly = check_recv(env, arg, res, gold, landed);
            const bool ok
                = cleared && synced && landedCorrectly && lastStatus == HIPBLAS_STATUS_SUCCESS;
            if(!ok)
                hipblaslt_cerr << "error: rank " << env.rank << " failed launch " << i << "\n";

            verified = agreement.agree(ok, std::logical_and<>{});
        }

        return verified ? kRankChildPassed : kRankChildFailed;
    }
    catch(const std::exception& e)
    {
        hipblaslt_cerr << "error: " << e.what() << "\n";
        return kRankChildFailed;
    }
} // namespace hipblaslt_bench

// Four launches over two channels walk 0, 1, 0, 1, so each channel is reused
// once while every rank is a peer of the other.
TEST(FusedA2AMultiProcess_multi_gpu, ReusedChannelsStayCorrect)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < int(kWorld))
        GTEST_SKIP() << "needs " << kWorld << " devices, found " << deviceCount;

    uint16_t port = 0;
    ASSERT_TRUE(hipblaslt_bench::free_port(port)) << "could not reserve a loopback port";

    const std::vector<std::pair<std::string, std::string>> extraEnv
        = {{"A2A_LAUNCHES", std::to_string(kLaunches)}};

    std::vector<pid_t> pids(kWorld, -1);
    for(uint32_t rank = 0; rank < kWorld; ++rank)
    {
        const int spawned = hipblaslt_bench::spawn_rank(
            "--a2a-rank-child", rank, kWorld, port, extraEnv, pids[rank]);
        if(spawned != 0)
        {
            pids[rank] = -1;
            hipblaslt_bench::kill_ranks(pids);
            FAIL() << "posix_spawn for rank " << rank << " -> errno " << spawned;
        }
    }

    std::vector<int> codes;
    if(!hipblaslt_bench::wait_for_ranks(pids, codes, kRankTimeoutSec))
    {
        hipblaslt_bench::kill_ranks(pids);
        FAIL() << "ranks did not finish within " << kRankTimeoutSec << "s";
    }

    for(uint32_t rank = 0; rank < kWorld; ++rank)
        if(codes[rank] == hipblaslt_bench::kRankChildSkipped)
            GTEST_SKIP() << "rank " << rank << " reported the run as unsupported";

    for(uint32_t rank = 0; rank < kWorld; ++rank)
        EXPECT_EQ(codes[rank], hipblaslt_bench::kRankChildPassed) << "rank " << rank;
}
