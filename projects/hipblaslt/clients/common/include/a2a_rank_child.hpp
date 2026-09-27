// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// One rank of a multi-process fused-A2A test, started by the gtest parent with
// the RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT variables that
// read_launcher_env() reads. The process exits with one of the codes below.

#include "a2a_bench.hpp"

#include <signal.h>
#include <sys/prctl.h>

#include <cstdlib>
#include <vector>

namespace hipblaslt_bench
{
    constexpr int kRankChildPassed  = 0;
    constexpr int kRankChildFailed  = 1;
    constexpr int kRankChildSkipped = 2;

    constexpr uint32_t kRankChildDefaultLaunches = 1;

    inline uint32_t rank_child_launches()
    {
        if(const char* v = std::getenv("A2A_LAUNCHES"))
        {
            const int parsed = std::atoi(v);
            if(parsed > 0)
                return uint32_t(parsed);
        }
        return kRankChildDefaultLaunches;
    }

    inline Arguments rank_child_arguments(const LauncherEnv& env)
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

    enum class GroupVerdict
    {
        Agreed,
        Disagreed,
        Unreachable,
    };

    inline GroupVerdict group_verdict(TcpRendezvous& rendezvous, uint32_t world, bool local)
    {
        const uint8_t        mine = local ? 1 : 0;
        std::vector<uint8_t> all(world);
        if(rendezvous.allgather(&mine, all.data(), sizeof(mine)) != HIPBLAS_STATUS_SUCCESS)
            return GroupVerdict::Unreachable;

        for(uint32_t j = 0; j < world; ++j)
            if(all[j] == 0)
                return GroupVerdict::Disagreed;
        return GroupVerdict::Agreed;
    }

    inline int run_rank_child()
    {
        // SIGKILL once the spawning test process is gone.
        prctl(PR_SET_PDEATHSIG, SIGKILL);

        const LauncherEnv env = read_launcher_env();
        if(env.world > HIPBLASLT_DEVICE_COMM_MAX_WORLD)
        {
            hipblaslt_cout << "skipped: WORLD_SIZE " << env.world << " exceeds "
                           << HIPBLASLT_DEVICE_COMM_MAX_WORLD << "\n";
            return kRankChildSkipped;
        }

        const Arguments arg = rank_child_arguments(env);

        TcpRendezvous rendezvous(env, kRendezvousTimeoutSec);

        if(!rendezvous.same_host_group())
        {
            hipblaslt_cerr << "error: the ranks did not form a single-host group\n";
            return kRankChildFailed;
        }

        const GroupVerdict reachable
            = group_verdict(rendezvous, env.world, peers_reachable(env, arg));
        if(reachable == GroupVerdict::Unreachable)
        {
            hipblaslt_cerr << "error: the peer-access allgather failed\n";
            return kRankChildFailed;
        }
        if(reachable == GroupVerdict::Disagreed)
        {
            hipblaslt_cout << "skipped: peer access unavailable on at least one rank\n";
            return kRankChildSkipped;
        }

        RankResources res;
        res.rendezvous = &rendezvous;
        if(!setup_rank(env, arg, res))
            return kRankChildFailed;

        hipblasLtMatmulHeuristicResult_t heur{};
        int                              algoCount  = 0;
        const hipblasStatus_t            algoStatus = select_algo(res, heur, algoCount);
        if(group_verdict(rendezvous, env.world, algoStatus == HIPBLAS_STATUS_SUCCESS)
           != GroupVerdict::Agreed)
        {
            if(algoStatus != HIPBLAS_STATUS_SUCCESS)
                hipblaslt_cerr << "error: hipblasLtMatmulAlgoGetHeuristic -> " << int(algoStatus)
                               << "\n";
            return kRankChildFailed;
        }

        const GroupVerdict solved = group_verdict(rendezvous, env.world, algoCount > 0);
        if(solved == GroupVerdict::Unreachable)
        {
            hipblaslt_cerr << "error: the solution-selection allgather failed\n";
            return kRankChildFailed;
        }
        if(solved == GroupVerdict::Disagreed)
        {
            hipblaslt_cout << "skipped: no fused GEMM+A2A solution in the loaded library\n";
            return kRankChildSkipped;
        }

        uint32_t                       launchCount = 0;
        hipblasStatus_t                lastStatus  = HIPBLAS_STATUS_SUCCESS;
        std::vector<hipblasLtBfloat16> gold, landed;
        auto launch = make_launch(res, heur, launchCount, lastStatus);

        const size_t recvBytes
            = size_t(arg.a2a_world) * arg.N[0] * shard_of(arg) * sizeof(hipblasLtBfloat16);

        // A failed launch still runs the rest of the iteration.
        bool           verified = true;
        const uint32_t launches = rank_child_launches();
        for(uint32_t i = 0; i < launches && verified; ++i)
        {
            const bool cleared = hipMemset(res.dRecv, 0, recvBytes) == hipSuccess;

            launch(int64_t(i));

            const bool synced = hipStreamSynchronize(res.stream) == hipSuccess;
            const bool landedCorrectly = check_recv(env, arg, res, gold, landed);
            const bool ok = cleared && synced && landedCorrectly
                            && lastStatus == HIPBLAS_STATUS_SUCCESS;
            if(!ok)
                hipblaslt_cerr << "error: rank " << env.rank << " failed launch " << i << "\n";

            verified = group_verdict(rendezvous, env.world, ok) == GroupVerdict::Agreed;
        }

        return verified ? kRankChildPassed : kRankChildFailed;
    }
} // namespace hipblaslt_bench
