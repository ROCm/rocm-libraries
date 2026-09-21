// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Benchmark logic for the fused GEMM + all-to-all epilogue. One rank per
// process, started by an external launcher; identity and rendezvous come from
// the environment variables torchrun sets.

#include "allclose.hpp"
#include "benchmark_timing.hpp"
#include "collective_rendezvous.hpp"
#include "hipblaslt_arguments.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_ostream.hpp"
#include "norm.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <SdmaQueue.hpp>

#include <cstdint>
#include <exception>
#include <memory>
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
        TcpRendezvous*                                               rendezvous = nullptr;

        RankResources()                                = default;
        RankResources(const RankResources&)            = delete;
        RankResources& operator=(const RankResources&) = delete;

        ~RankResources()
        {
            if(stream != nullptr)
                static_cast<void>(hipStreamSynchronize(stream));

            // recvPtrs[rank] aliases dRecv; the rest are IPC maps.
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

    inline bool fill_operands(const LauncherEnv& env, const Arguments& arg, RankResources& res)
    {
        std::vector<hipblasLtBfloat16> hostA(size_t(arg.K[0]) * arg.M[0]);
        std::vector<hipblasLtBfloat16> hostB(size_t(arg.K[0]) * arg.N[0]);

        g_hipblaslt_seed = hipblaslt_rng_t(env.rank + 1);
        hipblaslt_seedrand();
        hipblaslt_init_hpl(hostA.data(), size_t(arg.K[0]), size_t(arg.M[0]), size_t(arg.K[0]));
        hipblaslt_init_hpl(hostB.data(), size_t(arg.K[0]), size_t(arg.N[0]), size_t(arg.K[0]));

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
    // tokens, landing at recv_p[(s*tokens + t)*shard + fw]. Both buffers are
    // resized to one such block, laid out [token, feature].
    inline bool check_recv(const LauncherEnv&              env,
                           const Arguments&                arg,
                           RankResources&                  res,
                           std::vector<hipblasLtBfloat16>& gold,
                           std::vector<hipblasLtBfloat16>& landed)
    {
        const int64_t shard = shard_of(arg);
        const size_t  block = size_t(arg.N[0]) * shard;
        gold.resize(block);
        landed.resize(block);

        // Barrier before reading a peer's receive buffer.
        const uint8_t        here = 1;
        std::vector<uint8_t> group(env.world);
        if(res.rendezvous->allgather(&here, group.data(), sizeof(here)) != HIPBLAS_STATUS_SUCCESS)
        {
            hipblaslt_cerr << "error: allgather before the recv check failed\n";
            return false;
        }

        double normError = 0.0;
        size_t nonzero   = 0;
        bool   allClose  = true;
        for(uint32_t p = 0; p < arg.a2a_world; ++p)
        {
            CHECK_HIP_RC(hipMemcpy2D(gold.data(),
                                     size_t(shard) * sizeof(hipblasLtBfloat16),
                                     static_cast<const hipblasLtBfloat16*>(res.dD) + p * shard,
                                     size_t(arg.M[0]) * sizeof(hipblasLtBfloat16),
                                     size_t(shard) * sizeof(hipblasLtBfloat16),
                                     size_t(arg.N[0]),
                                     hipMemcpyDeviceToHost));
            CHECK_HIP_RC(hipMemcpy(landed.data(),
                                   static_cast<const hipblasLtBfloat16*>(res.recvPtrs[p])
                                       + size_t(env.rank) * block,
                                   block * sizeof(hipblasLtBfloat16),
                                   hipMemcpyDeviceToHost));

            for(size_t i = 0; i < block; ++i)
                if(float(gold[i]) != 0.0f)
                    ++nonzero;

            // allclose_check_general takes 1 as the sentinel for "no tolerance
            // passed yet" and reports back the loosest pair it needed.
            double atol = 1.0, rtol = 1.0;
            allClose = allclose_check_general(
                           'F', shard, arg.N[0], shard, gold.data(), landed.data(), atol, rtol)
                       && allClose;

            const double e = std::abs(
                norm_check_general('F', shard, arg.N[0], shard, gold.data(), landed.data()));
            if(e > normError)
                normError = e;
        }

        if(nonzero == 0)
        {
            hipblaslt_cerr << "error: rank " << env.rank << " exported an all-zero D\n";
            return false;
        }
        if(normError != 0.0 || !allClose)
            hipblaslt_cerr << "error: rank " << env.rank << " recv norm error=" << normError
                           << " allclose=" << (allClose ? "yes" : "no") << "\n";
        return normError == 0.0 && allClose;
    }

    inline bool peers_reachable(const LauncherEnv& env, const Arguments& arg)
    {
        if(env.world == 1)
            return true;

        if(env.local_rank != int(env.rank))
        {
            hipblaslt_cerr << "error: LOCAL_RANK " << env.local_rank << " must equal RANK "
                           << env.rank << " when every rank shares a host\n";
            return false;
        }

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

    // Reports only the local outcome.
    inline bool
        setup_rank_resources(const LauncherEnv& env, const Arguments& arg, RankResources& res)
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

        try
        {
            const uint32_t srcNode = TensileLite::Client::sdmaNodeIdForDevice(env.local_rank);
            for(uint32_t j = 0; j < arg.a2a_world; ++j)
            {
                const uint32_t dstNode = TensileLite::Client::sdmaNodeIdForDevice(j);
                res.ownedQueues.push_back(std::make_unique<TensileLite::Client::SdmaQueue>(
                    srcNode, TensileLite::Client::sdmaSelectEngine(srcNode, dstNode)));
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

    inline bool exchange_recv_pointers(const LauncherEnv& env,
                                       const Arguments&   arg,
                                       TcpRendezvous&     rendezvous,
                                       RankResources&     res)
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

    inline bool setup_rank(const LauncherEnv& env, const Arguments& arg, RankResources& res)
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
                                           rendezvous_allgather_trampoline,
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

    inline hipblasStatus_t
        select_algo(RankResources& res, hipblasLtMatmulHeuristicResult_t& heur, int& algoCount)
    {
        algoCount = 0;
        return hipblasLtMatmulAlgoGetHeuristic(res.handle,
                                               res.mm,
                                               res.lay[0],
                                               res.lay[1],
                                               res.lay[2],
                                               res.lay[3],
                                               res.pref,
                                               1,
                                               &heur,
                                               &algoCount);
    }

    // Successive launches alternate the communicator's flag regions.
    inline auto make_launch(RankResources&                          res,
                            const hipblasLtMatmulHeuristicResult_t& heur,
                            uint32_t&                               launchCount,
                            hipblasStatus_t&                        lastStatus)
    {
        return [&res, &heur, &launchCount, &lastStatus](int64_t) {
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
                lastStatus = status;
        };
    }

    inline CollectiveAgreement make_agreement(TcpRendezvous& rendezvous, uint32_t world)
    {
        CollectiveAgreement agreement;
        agreement.world     = world;
        agreement.allgather = [&rendezvous](const void* send, void* recv, size_t bytes) {
            return rendezvous.allgather(send, recv, bytes) == HIPBLAS_STATUS_SUCCESS;
        };
        return agreement;
    }
} // namespace hipblaslt_bench
