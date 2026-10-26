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
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    constexpr uint32_t kLaunches       = 4;
    constexpr int      kRankTimeoutSec = 180;

    constexpr uint32_t kRankChildDefaultLaunches = 1;

    // Largest MacroTile0 the fused path admits; the other is 128.
    constexpr int64_t kFusedA2AMaxMacroTile0 = 256;

    // Every world moves the same per-rank shard and keeps the same local tail, so
    // only M grows with the world. Both are whole tiles for either admitted
    // MacroTile0, which makes the GemmA2AShape predicate hold for every world. At 2
    // ranks this is the 4096 x 256 x 1024 problem with a 2048 extent.
    constexpr int64_t kFusedA2AShardFeatures = 1024;
    constexpr int64_t kFusedA2ALocalFeatures = 2048;
    constexpr int64_t kFusedA2ATokens        = 256;
    constexpr int64_t kFusedA2AContractBound = 1024;

    struct FusedA2AShape
    {
        int64_t features;
        int64_t extent;
    };

    FusedA2AShape fused_a2a_shape_for_world(uint32_t world)
    {
        const int64_t extent = kFusedA2AShardFeatures * int64_t(world);
        return {extent + kFusedA2ALocalFeatures, extent};
    }

    using hipblaslt_bench::Collective;
    using hipblaslt_bench::collective_name;
    using hipblaslt_bench::parse_collective;

    // A2AGemm: every rank gathers the same K block per peer, so only K grows
    // with the world. At 2 ranks this is the 1024 x 512 x 1024 problem.
    constexpr int64_t kA2AGemmFeatures   = 1024;
    constexpr int64_t kA2AGemmTokens     = 512;
    constexpr int64_t kA2AGemmBlockBound = 512;

    // Largest DepthU * A2ASubRounds the A2AGemm path is tuned with.
    constexpr int64_t kA2AGemmMaxBoundStep = 128;

    int64_t a2a_gemm_bound_for_world(uint32_t world)
    {
        return kA2AGemmBlockBound * int64_t(world);
    }

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

    Arguments rank_child_arguments(const hipblaslt_bench::LauncherEnv& env, Collective collective)
    {
        Arguments arg;
        arg.init();
        if(collective == Collective::A2AGemm)
        {
            arg.M[0]       = kA2AGemmFeatures;
            arg.N[0]       = kA2AGemmTokens;
            arg.K[0]       = a2a_gemm_bound_for_world(env.world);
            arg.a2a_extent = arg.K[0];
            arg.a2a_world  = uint8_t(env.world);
            return arg;
        }

        const FusedA2AShape shape = fused_a2a_shape_for_world(env.world);
        arg.M[0]       = shape.features;
        arg.N[0]       = kFusedA2ATokens;
        arg.K[0]       = kFusedA2AContractBound;
        arg.a2a_extent = shape.extent;
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

        const char*      collectiveName = std::getenv("A2A_COLLECTIVE");
        const Collective collective     = parse_collective(collectiveName);
        if(collective == Collective::Unknown)
        {
            hipblaslt_cerr << "error: unknown collective " << collectiveName << "\n";
            return kRankChildFailed;
        }

        const Arguments arg = rank_child_arguments(env, collective);

        TcpRendezvous rendezvous(env, kRendezvousTimeoutSec);

        if(!join_group(env, rendezvous, HIPBLASLT_DEVICE_COMM_MAX_WORLD))
            return kRankChildSkipped;

        const CollectiveAgreement agreement = make_agreement(rendezvous, env.world);

        RankResources res;
        res.rendezvous = &rendezvous;
        res.collective = collective;
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
        const uint32_t                 launches = rank_child_launches();

        if(collective == Collective::A2AGemm)
        {
            // The agreement after each rank rewrites its B publishes it to the peers.
            bool verified = true;
            for(uint32_t i = 0; i < launches && verified; ++i)
            {
                const bool written = a2a_gemm_write_input(env, arg, res, int64_t(i));
                if(!agreement.agree(written, std::logical_and<>{}))
                    return kRankChildFailed;

                launch(int64_t(i));

                const bool synced  = hipStreamSynchronize(res.stream) == hipSuccess;
                const bool correct = synced && a2a_gemm_check(env, arg, res, int64_t(i));
                const bool ok      = correct && lastStatus == HIPBLAS_STATUS_SUCCESS;
                if(!ok)
                    hipblaslt_cerr << "error: rank " << env.rank << " failed launch " << i << "\n";
                verified = agreement.agree(ok, std::logical_and<>{});
            }
            return verified ? kRankChildPassed : kRankChildFailed;
        }

        const size_t recvBytes
            = size_t(arg.a2a_world) * arg.N[0] * shard_of(arg) * sizeof(hipblasLtBfloat16);

        // A failed launch still runs the rest of the iteration.
        bool verified = true;
        bool cleared  = true;
        for(uint32_t i = 0; i < launches && verified; ++i)
        {
            const uint32_t slice = i % kRecvSlices;
            const bool     sliced
                = hipblasLtFusedEpilogueSetAttribute(res.fused,
                                                     HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS,
                                                     res.recvPtrs[slice],
                                                     arg.a2a_world * sizeof(res.recvPtrs[slice][0]))
                  == HIPBLAS_STATUS_SUCCESS;

            launch(int64_t(i));

            const bool synced          = hipStreamSynchronize(res.stream) == hipSuccess;
            const bool landedCorrectly = check_recv(env, arg, res, i, gold, landed);
            const bool ok              = cleared && sliced && synced && landedCorrectly
                            && lastStatus == HIPBLAS_STATUS_SUCCESS;
            if(!ok)
                hipblaslt_cerr << "error: rank " << env.rank << " failed launch " << i << "\n";

            verified = agreement.agree(ok, std::logical_and<>{});

            // Clears this launch's recv slice ahead of its next use.
            cleared = hipMemsetAsync(res.dRecv[slice], 0, recvBytes, res.stream) == hipSuccess;
        }

        return verified ? kRankChildPassed : kRankChildFailed;
    }
    catch(const std::exception& e)
    {
        hipblaslt_cerr << "error: " << e.what() << "\n";
        return kRankChildFailed;
    }
} // namespace hipblaslt_bench

// Worlds every fused collective is swept over, one GPU per rank. A host with
// fewer devices skips a world rather than failing it.
constexpr uint32_t kFusedWorlds[] = {2, 3, 4, 5, 6, 7, 8};

// Collectives the multi-process sweep runs.
constexpr Collective kFusedCollectives[] = {Collective::GemmA2A, Collective::A2AGemm};

TEST(FusedA2ACollective_smoke, NamesRoundTrip)
{
    EXPECT_EQ(parse_collective(nullptr), Collective::GemmA2A);
    EXPECT_EQ(parse_collective(""), Collective::GemmA2A);
    EXPECT_EQ(parse_collective(collective_name(Collective::GemmA2A)), Collective::GemmA2A);
    EXPECT_EQ(parse_collective(collective_name(Collective::A2AGemm)), Collective::A2AGemm);
    EXPECT_EQ(parse_collective("bias"), Collective::Unknown);
}

TEST(FusedA2AWorldShape_smoke, EveryWorldFrom2To8FitsTheA2AGemmShape)
{
    for(uint32_t world : kFusedWorlds)
    {
        const int64_t k        = a2a_gemm_bound_for_world(world);
        const int64_t rowBytes = (k / int64_t(world)) * int64_t(sizeof(hipblasLtBfloat16));
        EXPECT_EQ(k % (kA2AGemmMaxBoundStep * int64_t(world)), 0) << "world " << world;
        EXPECT_LE(rowBytes / 16, int64_t{1} << 14) << "world " << world;
        EXPECT_EQ((kA2AGemmTokens * rowBytes) % 128, 0) << "world " << world;
    }

    EXPECT_EQ(a2a_gemm_bound_for_world(2), 1024);
}

// One rule has to give a legal fused-A2A problem at every world, with the same
// shard per rank and the same local tail.
TEST(FusedA2AWorldShape_smoke, EveryWorldFrom2To8IsTileDivisible)
{
    for(uint32_t world : kFusedWorlds)
    {
        const FusedA2AShape shape = fused_a2a_shape_for_world(world);
        const int64_t       tile  = kFusedA2AMaxMacroTile0;
        EXPECT_EQ(shape.extent % (tile * int64_t(world)), 0) << "world " << world;
        EXPECT_EQ(shape.features % tile, 0) << "world " << world;
        EXPECT_EQ(shape.extent / int64_t(world), kFusedA2AShardFeatures) << "world " << world;
        EXPECT_EQ(shape.features - shape.extent, kFusedA2ALocalFeatures) << "world " << world;
    }

    const FusedA2AShape two = fused_a2a_shape_for_world(2);
    EXPECT_EQ(two.features, 4096);
    EXPECT_EQ(two.extent, 2048);
}

class FusedA2AMultiProcess_multi_gpu
    : public ::testing::TestWithParam<std::tuple<Collective, uint32_t>>
{
};

// Four launches over two channels walk 0, 1, 0, 1, so each channel is reused
// once while every rank is a peer of the other.
TEST_P(FusedA2AMultiProcess_multi_gpu, ReusedChannelsStayCorrect)
{
    const Collective collective = std::get<0>(GetParam());
    const uint32_t   world      = std::get<1>(GetParam());

    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < int(world))
        GTEST_SKIP() << "needs " << world << " devices, found " << deviceCount;

    uint16_t port = 0;
    ASSERT_TRUE(hipblaslt_bench::free_port(port)) << "could not reserve a loopback port";

    const std::vector<std::pair<std::string, std::string>> extraEnv
        = {{"A2A_LAUNCHES", std::to_string(kLaunches)},
           {"A2A_COLLECTIVE", collective_name(collective)}};

    std::vector<pid_t> pids(world, -1);
    for(uint32_t rank = 0; rank < world; ++rank)
    {
        const int spawned = hipblaslt_bench::spawn_rank(
            "--a2a-rank-child", rank, world, port, extraEnv, pids[rank]);
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

    for(uint32_t rank = 0; rank < world; ++rank)
        if(codes[rank] == hipblaslt_bench::kRankChildSkipped)
            GTEST_SKIP() << "rank " << rank << " reported the run as unsupported";

    for(uint32_t rank = 0; rank < world; ++rank)
        EXPECT_EQ(codes[rank], hipblaslt_bench::kRankChildPassed) << "rank " << rank;
}

INSTANTIATE_TEST_SUITE_P(
    World,
    FusedA2AMultiProcess_multi_gpu,
    ::testing::Combine(::testing::ValuesIn(kFusedCollectives), ::testing::ValuesIn(kFusedWorlds)),
    [](const ::testing::TestParamInfo<std::tuple<Collective, uint32_t>>& info) {
        const Collective c   = std::get<0>(info.param);
        const char*      tag = c == Collective::GemmA2A   ? "GemmA2A"
                               : c == Collective::A2AGemm ? "A2AGemm"
                                                          : "Unknown";
        return std::string(tag) + "_" + std::to_string(std::get<1>(info.param)) + "gpu";
    });
