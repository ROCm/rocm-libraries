// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Benchmark logic for the fused GEMM + all-to-all epilogue. One rank per
// process, started by an external launcher; identity and rendezvous come from
// the environment variables torchrun sets.

#include "benchmark_timing.hpp"
#include "collective_rendezvous.hpp"
#include "hipblaslt_arguments.hpp"
#include "hipblaslt_ostream.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <SdmaQueue.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace hipblaslt_bench
{
    constexpr int    kRendezvousTimeoutSec = 60;
    constexpr size_t kWorkspaceSize        = 128ull * 1024 * 1024;

    constexpr uint32_t kCommChannels = 2;

    struct RankResources
    {
        hipblasLtHandle_t                  handle    = nullptr;
        hipblasLtFusedEpilogueDescriptor_t fused     = nullptr;
        hipblasLtMatmulDesc_t              mm        = nullptr;
        hipblasLtMatrixLayout_t            lay[4]    = {};
        hipblasLtMatmulPreference_t        pref      = nullptr;
        hipStream_t                        stream    = nullptr;
        void*                              dA        = nullptr;
        void*                              dB        = nullptr;
        void*                              dC        = nullptr;
        void*                              dD        = nullptr;
        void*                              dRecv     = nullptr;
        void*                              workspace = nullptr;
        void*                recvPtrs[HIPBLASLT_DEVICE_COMM_MAX_WORLD] = {};
        hipblasLtSdmaQueue_t queues[HIPBLASLT_DEVICE_COMM_MAX_WORLD]   = {};
        std::vector<std::unique_ptr<TensileLite::Client::SdmaQueue>> ownedQueues;
        hipblaslt_bench::TcpRendezvous* rendezvous = nullptr;

        RankResources()                                = default;
        RankResources(const RankResources&)            = delete;
        RankResources& operator=(const RankResources&) = delete;

        ~RankResources()
        {
            if(stream != nullptr)
                static_cast<void>(hipStreamSynchronize(stream));

            // Only a peer's buffer was mapped in; this rank's own is a plain allocation.
            for(void* p : recvPtrs)
                if(p != nullptr && p != dRecv)
                    static_cast<void>(hipIpcCloseMemHandle(p));

            for(hipblasLtMatrixLayout_t l : lay)
                if(l != nullptr)
                    static_cast<void>(hipblasLtMatrixLayoutDestroy(l));
            if(pref != nullptr)
                static_cast<void>(hipblasLtMatmulPreferenceDestroy(pref));
            if(mm != nullptr)
                static_cast<void>(hipblasLtMatmulDescDestroy(mm));
            if(fused != nullptr)
                static_cast<void>(hipblasLtFusedEpilogueDestroy(fused));

            static_cast<void>(hipFree(dA));
            static_cast<void>(hipFree(dB));
            static_cast<void>(hipFree(dC));
            static_cast<void>(hipFree(dD));
            static_cast<void>(hipFree(dRecv));
            static_cast<void>(hipFree(workspace));

            ownedQueues.clear();
            if(handle != nullptr)
                static_cast<void>(hipblasLtDestroy(handle));
            if(stream != nullptr)
                static_cast<void>(hipStreamDestroy(stream));
        }
    };

    inline void print_usage(const char* program)
    {
        hipblaslt_cout
            << "Usage: " << program << " <options>\n"
               "\t-h, --help\t\tShow this help message\n"
               "\t-m, --m\t\t\tFeature extent (free0), default 18432\n"
               "\t-n, --n\t\t\tToken extent (free1), default 2048\n"
               "\t-k, --k\t\t\tBound extent, default 8192\n"
               "\t--a2a_extent\t\tFeatures taking the A2A path, default 10240\n"
               "\t--a2a_world\t\tChecked against WORLD_SIZE; env wins\n"
               "\t--timing\t\t1 to measure, default 0\n"
               "\t--iters\t\t\tEnqueues per sample, default 10\n"
               "\t--adaptive\t\t1 to self-size the sample count; --iters is then unused\n"
               "\t--verify\t\t1 to check every iteration against the closed form\n"
               "Rank identity comes from RANK / WORLD_SIZE / LOCAL_RANK /\n"
               "MASTER_ADDR / MASTER_PORT. With none set the run is single-rank.\n";
    }

    inline bool match(const char* arg, const char* shortName, const char* longName)
    {
        return (shortName && std::strcmp(arg, shortName) == 0)
               || std::strcmp(arg, longName) == 0;
    }

    inline bool parse_a2a_args(int argc, char** argv, Arguments& arg, std::string& error)
    {
        for(int i = 1; i < argc; ++i)
        {
            const char* opt = argv[i];
            if(match(opt, "-h", "--help"))
            {
                error = "help";
                return false;
            }
            if(i + 1 >= argc)
            {
                error = std::string("missing value for ") + opt;
                return false;
            }
            const char* value = argv[++i];

            if(match(opt, "-m", "--m"))
                arg.M[0] = std::strtoll(value, nullptr, 10);
            else if(match(opt, "-n", "--n"))
                arg.N[0] = std::strtoll(value, nullptr, 10);
            else if(match(opt, "-k", "--k"))
                arg.K[0] = std::strtoll(value, nullptr, 10);
            else if(match(opt, nullptr, "--a2a_extent"))
                arg.a2a_extent = std::strtoll(value, nullptr, 10);
            else if(match(opt, nullptr, "--a2a_world"))
                arg.a2a_world = uint8_t(std::strtoul(value, nullptr, 10));
            else if(match(opt, nullptr, "--timing"))
                arg.timing = int8_t(std::strtol(value, nullptr, 10));
            else if(match(opt, nullptr, "--iters"))
                arg.iters = int32_t(std::strtol(value, nullptr, 10));
            else if(match(opt, nullptr, "--adaptive"))
                arg.adaptive = std::strtol(value, nullptr, 10) != 0;
            else if(match(opt, nullptr, "--verify"))
                arg.unit_check = int8_t(std::strtol(value, nullptr, 10));
            else
            {
                error = std::string("unknown option ") + opt;
                return false;
            }
        }
        return true;
    }

#define CHECK_HIP_RC(expr)                                                        \
    do                                                                            \
    {                                                                             \
        const hipError_t _e = (expr);                                             \
        if(_e != hipSuccess)                                                      \
        {                                                                         \
            hipblaslt_cerr << "error: " << #expr << " -> " << hipGetErrorString(_e) \
                           << "\n";                                                \
            return false;                                                         \
        }                                                                         \
    } while(0)

#define CHECK_LT_RC(expr)                                                         \
    do                                                                            \
    {                                                                             \
        const hipblasStatus_t _s = (expr);                                        \
        if(_s != HIPBLAS_STATUS_SUCCESS)                                          \
        {                                                                         \
            hipblaslt_cerr << "error: " << #expr << " -> " << int(_s) << "\n";      \
            return false;                                                         \
        }                                                                         \
    } while(0)

    inline int64_t shard_of(const Arguments& arg)
    {
        return arg.a2a_extent / arg.a2a_world;
    }

    // Mirrors the operands fill_operands writes; the two change together.
    inline float expectedD(uint32_t rank, int64_t feature, int64_t token)
    {
        return float((rank + 1) * ((feature % 7) + 1) * ((token % 5) + 1));
    }

    inline bool fill_operands(const hipblaslt_bench::LauncherEnv& env,
                              const Arguments&                    arg,
                              RankResources&                      res)
    {
        std::vector<hipblasLtBfloat16> hostA(size_t(arg.K[0]) * arg.M[0], hipblasLtBfloat16(0.0f));
        std::vector<hipblasLtBfloat16> hostB(size_t(arg.K[0]) * arg.N[0], hipblasLtBfloat16(0.0f));

        for(int64_t f = 0; f < arg.M[0]; ++f)
            hostA[size_t(f) * arg.K[0]]
                = hipblasLtBfloat16(float((env.rank + 1) * ((f % 7) + 1)));
        for(int64_t t = 0; t < arg.N[0]; ++t)
            hostB[size_t(t) * arg.K[0]] = hipblasLtBfloat16(float((t % 5) + 1));

        CHECK_HIP_RC(hipMemcpy(res.dA,
                               hostA.data(),
                               hostA.size() * sizeof(hipblasLtBfloat16),
                               hipMemcpyHostToDevice));
        CHECK_HIP_RC(hipMemcpy(res.dB,
                               hostB.data(),
                               hostB.size() * sizeof(hipblasLtBfloat16),
                               hipMemcpyHostToDevice));
        return true;
    }

    // Rank s hands peer p the feature shard [p*shard, (p+1)*shard) of all its
    // tokens, landing at recv_p[(s*tokens + t)*shard + fw].
    inline bool check_recv(const hipblaslt_bench::LauncherEnv& env,
                           const Arguments&                    arg,
                           RankResources&                      res,
                           std::vector<hipblasLtBfloat16>&     host)
    {
        const int64_t shard = shard_of(arg);
        const size_t  count = size_t(arg.a2a_world) * arg.N[0] * shard;
        host.resize(count);
        CHECK_HIP_RC(hipMemcpy(host.data(),
                               res.dRecv,
                               count * sizeof(hipblasLtBfloat16),
                               hipMemcpyDeviceToHost));

        size_t mismatches = 0;
        for(uint32_t s = 0; s < arg.a2a_world; ++s)
            for(int64_t t = 0; t < arg.N[0]; ++t)
                for(int64_t fw = 0; fw < shard; ++fw)
                {
                    const size_t  at      = (size_t(s) * arg.N[0] + size_t(t)) * shard + fw;
                    const int64_t feature = int64_t(env.rank) * shard + fw;
                    if(float(host[at]) != expectedD(s, feature, t))
                        ++mismatches;
                }

        if(mismatches != 0)
            hipblaslt_cerr << "error: rank " << env.rank << " recv mismatches=" << mismatches
                           << "\n";
        return mismatches == 0;
    }

    inline bool peers_reachable(const hipblaslt_bench::LauncherEnv& env, const Arguments& arg)
    {
        if(env.world == 1)
            return true;

        int visible = 0;
        if(hipGetDeviceCount(&visible) != hipSuccess || visible < int(env.world))
        {
            hipblaslt_cerr << "error: " << visible << " device(s) visible, need " << env.world
                           << "\n";
            return false;
        }

        if(hipSetDevice(env.local_rank) != hipSuccess)
        {
            hipblaslt_cerr << "error: hipSetDevice(" << env.local_rank << ") failed\n";
            return false;
        }

        for(uint32_t j = 0; j < arg.a2a_world; ++j)
        {
            if(int(j) == env.local_rank)
                continue;

            int canAccess = 0;
            if(hipDeviceCanAccessPeer(&canAccess, env.local_rank, int(j)) != hipSuccess
               || canAccess == 0)
            {
                hipblaslt_cerr << "error: device " << env.local_rank << " cannot peer with " << j
                               << "\n";
                return false;
            }

            const hipError_t e = hipDeviceEnablePeerAccess(int(j), 0);
            if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
            {
                hipblaslt_cerr << "error: hipDeviceEnablePeerAccess(" << env.local_rank << " -> "
                               << j << ") -> " << hipGetErrorString(e) << "\n";
                return false;
            }
        }
        return true;
    }

    // Reports only the local outcome; setup_rank reduces it across ranks before
    // deciding whether to call hipblasLtSetDeviceComm.
    inline bool setup_rank_resources(const hipblaslt_bench::LauncherEnv& env,
                                     const Arguments&                    arg,
                                     RankResources&                      res)
    {
        CHECK_HIP_RC(hipSetDevice(env.local_rank));
        CHECK_HIP_RC(hipStreamCreate(&res.stream));

        const size_t bytesA    = size_t(arg.K[0]) * arg.M[0] * sizeof(hipblasLtBfloat16);
        const size_t bytesB    = size_t(arg.K[0]) * arg.N[0] * sizeof(hipblasLtBfloat16);
        const size_t bytesD    = size_t(arg.M[0]) * arg.N[0] * sizeof(hipblasLtBfloat16);
        const size_t bytesRecv = size_t(arg.a2a_world) * arg.N[0] * shard_of(arg)
                                 * sizeof(hipblasLtBfloat16);

        CHECK_HIP_RC(hipMalloc(&res.dA, bytesA));
        CHECK_HIP_RC(hipMalloc(&res.dB, bytesB));
        CHECK_HIP_RC(hipMalloc(&res.dC, bytesD));
        CHECK_HIP_RC(hipMalloc(&res.dD, bytesD));
        CHECK_HIP_RC(hipMalloc(&res.dRecv, bytesRecv));
        CHECK_HIP_RC(hipMalloc(&res.workspace, kWorkspaceSize));
        CHECK_HIP_RC(hipMemset(res.dRecv, 0, bytesRecv));

        // Every queue loops back to this rank's own device.
        try
        {
            const uint32_t srcNode = TensileLite::Client::sdmaNodeIdForDevice(env.local_rank);
            for(uint32_t j = 0; j < arg.a2a_world; ++j)
            {
                res.ownedQueues.push_back(std::make_unique<TensileLite::Client::SdmaQueue>(
                    srcNode, TensileLite::Client::sdmaSelectEngine(srcNode, srcNode)));
                const HsaQueueResource& q = res.ownedQueues.back()->queueResource();
                res.queues[j] = {res.ownedQueues.back()->ringBase(),
                                 (void*)q.Queue_read_ptr_aql,
                                 (void*)q.Queue_write_ptr_aql,
                                 (void*)q.Queue_DoorBell_aql};
            }
        }
        catch(const std::exception& e)
        {
            hipblaslt_cerr << "error: cannot create an SDMA queue (" << e.what() << ")\n";
            return false;
        }

        CHECK_LT_RC(hipblasLtCreate(&res.handle));
        return true;
    }

    inline bool exchange_recv_pointers(const hipblaslt_bench::LauncherEnv& env,
                                       const Arguments&                    arg,
                                       hipblaslt_bench::TcpRendezvous&     rendezvous,
                                       RankResources&                      res)
    {
        if(env.world == 1)
        {
            res.recvPtrs[0] = res.dRecv;
            return true;
        }

        struct HandleContribution
        {
            uint8_t           ok;
            hipIpcMemHandle_t handle;
        };
        HandleContribution mine{};
        mine.ok = hipIpcGetMemHandle(&mine.handle, res.dRecv) == hipSuccess ? 1 : 0;

        std::vector<HandleContribution> all(env.world);
        if(rendezvous.allgather(&mine, all.data(), sizeof(mine)) != HIPBLAS_STATUS_SUCCESS)
        {
            hipblaslt_cerr << "error: recv-pointer handle allgather failed\n";
            return false;
        }
        bool gotAllHandles = true;
        for(uint32_t j = 0; j < env.world; ++j)
            gotAllHandles = gotAllHandles && all[j].ok != 0;
        if(!gotAllHandles)
        {
            hipblaslt_cerr << "error: hipIpcGetMemHandle failed on at least one rank\n";
            return false;
        }

        uint8_t openedAll = 1;
        for(uint32_t j = 0; j < arg.a2a_world; ++j)
        {
            if(j == env.rank)
                res.recvPtrs[j] = res.dRecv;
            else if(hipIpcOpenMemHandle(
                        &res.recvPtrs[j], all[j].handle, hipIpcMemLazyEnablePeerAccess)
                    != hipSuccess)
                openedAll = 0;
        }

        std::vector<uint8_t> allOpened(env.world);
        if(rendezvous.allgather(&openedAll, allOpened.data(), sizeof(openedAll))
           != HIPBLAS_STATUS_SUCCESS)
        {
            hipblaslt_cerr << "error: recv-pointer open-result allgather failed\n";
            return false;
        }
        bool groupOpened = true;
        for(uint32_t j = 0; j < env.world; ++j)
            groupOpened = groupOpened && allOpened[j] != 0;
        if(!groupOpened)
            hipblaslt_cerr << "error: hipIpcOpenMemHandle failed on at least one rank\n";
        return groupOpened;
    }

    inline bool setup_rank(const hipblaslt_bench::LauncherEnv& env,
                           const Arguments&                    arg,
                           RankResources&                      res)
    {
        const uint8_t ready = setup_rank_resources(env, arg, res) ? 1 : 0;

        std::vector<uint8_t> allReady(env.world);
        if(res.rendezvous->allgather(&ready, allReady.data(), sizeof(ready))
           != HIPBLAS_STATUS_SUCCESS)
        {
            hipblaslt_cerr << "error: rank-readiness allgather failed\n";
            return false;
        }
        bool groupReady = true;
        for(uint32_t j = 0; j < env.world; ++j)
            groupReady = groupReady && allReady[j] != 0;
        if(!groupReady)
        {
            hipblaslt_cerr << "error: rank-local setup failed on at least one rank\n";
            return false;
        }

        CHECK_LT_RC(hipblasLtSetDeviceComm(res.handle,
                                           env.rank,
                                           env.world,
                                           kCommChannels,
                                           hipblaslt_bench::rendezvous_allgather_trampoline,
                                           res.rendezvous));

        if(!exchange_recv_pointers(env, arg, *res.rendezvous, res))
            return false;

        CHECK_LT_RC(hipblasLtFusedEpilogueCreate(&res.fused));
        CHECK_LT_RC(hipblasLtFusedEpilogueAdd(res.fused,
                                              HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX));
        CHECK_LT_RC(
            hipblasLtFusedEpilogueSetAttribute(res.fused,
                                               HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                                               res.queues,
                                               arg.a2a_world * sizeof(res.queues[0])));
        CHECK_LT_RC(
            hipblasLtFusedEpilogueSetAttribute(res.fused,
                                               HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS,
                                               res.recvPtrs,
                                               arg.a2a_world * sizeof(res.recvPtrs[0])));
        CHECK_LT_RC(hipblasLtFusedEpilogueSetAttribute(
            res.fused,
            HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT,
            &arg.a2a_extent,
            sizeof(arg.a2a_extent)));
        const hipblasLtA2ACompletionMode_t mode = HIPBLASLT_A2A_COMPLETION_IN_KERNEL_FULL;
        CHECK_LT_RC(hipblasLtFusedEpilogueSetAttribute(
            res.fused,
            HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_COMPLETION_MODE,
            &mode,
            sizeof(mode)));

        CHECK_LT_RC(hipblasLtMatrixLayoutCreate(
            &res.lay[0], HIP_R_16BF, arg.K[0], arg.M[0], arg.K[0]));
        CHECK_LT_RC(hipblasLtMatrixLayoutCreate(
            &res.lay[1], HIP_R_16BF, arg.K[0], arg.N[0], arg.K[0]));
        CHECK_LT_RC(hipblasLtMatrixLayoutCreate(
            &res.lay[2], HIP_R_16BF, arg.M[0], arg.N[0], arg.M[0]));
        CHECK_LT_RC(hipblasLtMatrixLayoutCreate(
            &res.lay[3], HIP_R_16BF, arg.M[0], arg.N[0], arg.M[0]));

        CHECK_LT_RC(hipblasLtMatmulDescCreate(&res.mm, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        CHECK_LT_RC(hipblasLtMatmulDescSetAttribute(
            res.mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CHECK_LT_RC(hipblasLtMatmulDescSetAttribute(
            res.mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        CHECK_LT_RC(hipblasLtMatmulDescSetAttribute(
            res.mm, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &res.fused, sizeof(res.fused)));

        CHECK_LT_RC(hipblasLtMatmulPreferenceCreate(&res.pref));
        CHECK_LT_RC(hipblasLtMatmulPreferenceSetAttribute(
            res.pref,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceSize,
            sizeof(kWorkspaceSize)));

        return fill_operands(env, arg, res);
    }

    inline bool select_algo(const Arguments&                  arg,
                            RankResources&                    res,
                            hipblasLtMatmulHeuristicResult_t& heur)
    {
        int algoCount = 0;
        CHECK_LT_RC(hipblasLtMatmulAlgoGetHeuristic(res.handle,
                                                    res.mm,
                                                    res.lay[0],
                                                    res.lay[1],
                                                    res.lay[2],
                                                    res.lay[3],
                                                    res.pref,
                                                    1,
                                                    &heur,
                                                    &algoCount));
        return algoCount > 0;
    }

    // Successive launches alternate the communicator's flag regions.
    inline auto make_launch(const LauncherEnv&                      env,
                            const Arguments&                        arg,
                            RankResources&                          res,
                            const hipblasLtMatmulHeuristicResult_t& heur,
                            uint32_t&                               launchCount,
                            hipblasStatus_t&                        lastStatus,
                            std::vector<hipblasLtBfloat16>&         hostRecv)
    {
        // lastStatus is sticky: once a launch fails it must stay failed.
        return [&env, &arg, &res, &heur, &launchCount, &lastStatus, &hostRecv](int64_t) {
            const float    alpha = 1.0f, beta = 0.0f;
            const uint32_t channel = launchCount++ % kCommChannels;

            const hipblasStatus_t attrStatus = hipblasLtFusedEpilogueSetAttribute(
                res.fused, HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL, &channel, sizeof(channel));
            if(attrStatus != HIPBLAS_STATUS_SUCCESS)
            {
                lastStatus = attrStatus;
                return;
            }

            const hipblasStatus_t status = hipblasLtMatmul(res.handle,
                                                           res.mm,
                                                           &alpha,
                                                           res.dA,
                                                           res.lay[0],
                                                           res.dB,
                                                           res.lay[1],
                                                           &beta,
                                                           res.dC,
                                                           res.lay[2],
                                                           res.dD,
                                                           res.lay[3],
                                                           &heur.algo,
                                                           res.workspace,
                                                           kWorkspaceSize,
                                                           res.stream);
            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                lastStatus = status;
                return;
            }

            if(arg.unit_check
               && !(hipStreamSynchronize(res.stream) == hipSuccess
                    && check_recv(env, arg, res, hostRecv)))
                lastStatus = HIPBLAS_STATUS_INTERNAL_ERROR;
        };
    }

    // Each call is one allgather. A failed one sets `failed` and answers "stop".
    inline hipblaslt_bench::CollectiveAgreement
        make_agreement(hipblaslt_bench::TcpRendezvous& rendezvous, uint32_t world, bool& failed)
    {
        hipblaslt_bench::CollectiveAgreement agreement;
        if(world == 1)
            return agreement;

        agreement.value = [&rendezvous, world, &failed](double mine, hipblaslt_bench::AgreeOp op) {
            std::vector<double> all(world);
            if(rendezvous.allgather(&mine, all.data(), sizeof(mine))
               != HIPBLAS_STATUS_SUCCESS)
            {
                failed = true;
                return mine;
            }
            double acc = all[0];
            for(uint32_t j = 1; j < world; ++j)
                acc = (op == hipblaslt_bench::AgreeOp::Max) ? std::max(acc, all[j])
                                                            : std::min(acc, all[j]);
            return acc;
        };
        agreement.flag = [&rendezvous, world, &failed](bool mine, hipblaslt_bench::AgreeOp op) {
            const uint8_t        send = mine ? 1 : 0;
            std::vector<uint8_t> all(world);
            if(rendezvous.allgather(&send, all.data(), sizeof(send))
               != HIPBLAS_STATUS_SUCCESS)
            {
                failed = true;
                return true;
            }
            bool acc = all[0] != 0;
            for(uint32_t j = 1; j < world; ++j)
                acc = (op == hipblaslt_bench::AgreeOp::All) ? (acc && all[j] != 0)
                                                            : (acc || all[j] != 0);
            return acc;
        };
        return agreement;
    }
} // namespace hipblaslt_bench
