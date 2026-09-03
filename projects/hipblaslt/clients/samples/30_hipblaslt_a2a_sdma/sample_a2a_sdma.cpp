// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fused GEMM+A2A over the public API with real SDMA queues: four ranks in one
// process, a thread and a device each, checking every rank's recv buffer
// against every peer's D.
//
// Axis naming follows the fused-A2A contract rather than GEMM habit: M / free0
// is the FEATURE axis, contiguous in a column-major D, and N / free1 is the
// TOKEN axis. Only the first kExtent features take the A2A path; the remaining
// kFeatures - kExtent write local D.
//
// Rank s hands peer p the feature shard [p*kShard, (p+1)*kShard) of all its
// tokens, and it lands at recv_p[(s*kTokens + t)*kShard + fw].
//
// Needs a device library holding a FusedGemmA2A solution for this shape --
// tensilelite/Tensile/Tests/common/gemm/gfx950/fused_a2a_logic_disabled.yaml
// builds one. Run with:
//
//   HIP_VISIBLE_DEVICES=<four peer-capable cards> \
//   HIPBLASLT_TENSILE_LIBPATH=<devlib>/library/gfx950 ./sample_a2a_sdma
//
// Exits 0 and prints a reason when the machine cannot host the run or the loaded
// library carries no fused GEMM+A2A solution. A library that fails to load is an
// error rather than a skip.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <SdmaQueue.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    constexpr int64_t  kFeatures = 2048; // M / free0
    constexpr int64_t  kTokens   = 1024; // N / free1
    constexpr int64_t  kK        = 1024;
    constexpr int64_t  kExtent   = 1024;
    constexpr uint32_t kWorld    = 4;
    constexpr uint32_t kChannels = 1;
    constexpr uint32_t kChannel  = 0;

    constexpr int64_t kShard = kExtent / kWorld;

    constexpr size_t kWorkspaceSize = 32ull * 1024 * 1024;
    constexpr int    kCommTimeoutSec = 30;

    constexpr int kStatusOk     = 0;
    constexpr int kStatusFailed = 1;

    uint16_t toBf16(float f)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        return uint16_t(bits >> 16);
    }

    float fromBf16(uint16_t h)
    {
        const uint32_t bits = uint32_t(h) << 16;
        float          f    = 0.0f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    float expectedD(uint32_t rank, int64_t feature, int64_t token)
    {
        return float((rank + 1) * ((feature % 7) + 1) * ((token % 5) + 1));
    }

    // arrive() returns false once any thread has called abort(), which also
    // releases whoever is already waiting.
    class Barrier
    {
    public:
        explicit Barrier(uint32_t parties)
            : parties_(parties)
        {
        }

        bool arrive()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const uint64_t               gen = generation_;
            if(++arrived_ == parties_)
            {
                arrived_ = 0;
                ++generation_;
                cv_.notify_all();
            }
            else
            {
                cv_.wait(lock, [&] { return generation_ != gen || aborted_; });
            }
            return !aborted_;
        }

        void abort()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                aborted_ = true;
            }
            cv_.notify_all();
        }

    private:
        std::mutex              mutex_;
        std::condition_variable cv_;
        const uint32_t          parties_;
        uint32_t                arrived_    = 0;
        uint64_t                generation_ = 0;
        bool                    aborted_    = false;
    };

    struct CommContext
    {
        std::mutex              mutex;
        std::condition_variable cv;
        std::vector<char>       records;
        uint32_t                arrived = 0;
        bool                    ready   = false;
    };

    // Passed as the per-rank userData: the callback places sendbuf by this rank
    // rather than by reading the record's contents.
    struct AllgatherSlot
    {
        CommContext* shared = nullptr;
        uint32_t     rank   = 0;
    };

    hipblasStatus_t threadAllgather(void*       userData,
                                    const void* sendbuf,
                                    void*       recvbuf,
                                    size_t      bytesPerRank)
    {
        auto*        slot = static_cast<AllgatherSlot*>(userData);
        CommContext& ctx  = *slot->shared;

        std::unique_lock<std::mutex> lock(ctx.mutex);
        if(ctx.records.empty())
            ctx.records.resize(bytesPerRank * kWorld);
        if(ctx.records.size() != bytesPerRank * kWorld)
            return HIPBLAS_STATUS_INVALID_VALUE;

        std::memcpy(ctx.records.data() + bytesPerRank * slot->rank, sendbuf, bytesPerRank);
        if(++ctx.arrived == kWorld)
        {
            ctx.ready = true;
            ctx.cv.notify_all();
        }
        else if(!ctx.cv.wait_for(lock,
                                 std::chrono::seconds(kCommTimeoutSec),
                                 [&ctx] { return ctx.ready; }))
        {
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        std::memcpy(recvbuf, ctx.records.data(), ctx.records.size());
        return HIPBLAS_STATUS_SUCCESS;
    }

    struct Rank
    {
        hipblasLtHandle_t handle = nullptr;

        void* dA         = nullptr;
        void* dB         = nullptr;
        void* dC         = nullptr;
        void* dD         = nullptr;
        void* dRecv      = nullptr;
        void* dWorkspace = nullptr;

        // Entry j addresses rank j: the queue this rank pushes over, and the
        // buffer it pushes into.
        hipblasLtSdmaQueue_t queues[kWorld]   = {};
        void*                recvPtrs[kWorld] = {};

        std::vector<uint16_t> hD;
        std::vector<uint16_t> hRecv;

        bool ok      = false;
        bool skipped = false;
    };

    // Releases the descriptors a rank builds, on every path out of runRank.
    struct MatmulHandles
    {
        hipblasLtFusedEpilogueDescriptor_t fused  = nullptr;
        hipblasLtMatrixLayout_t            lay[4] = {};
        hipblasLtMatmulDesc_t              mm     = nullptr;
        hipblasLtMatmulPreference_t        pref   = nullptr;

        ~MatmulHandles()
        {
            if(pref != nullptr)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(mm != nullptr)
                hipblasLtMatmulDescDestroy(mm);
            for(auto layout : lay)
                if(layout != nullptr)
                    hipblasLtMatrixLayoutDestroy(layout);
            if(fused != nullptr)
                hipblasLtFusedEpilogueDestroy(fused);
        }
    };

    // Releases the ranks still waiting when this one leaves early. A rank that
    // reaches the end disarms it; without this a single failure hangs the rest.
    struct BarrierGuard
    {
        Barrier& barrier;
        bool     armed = true;

        ~BarrierGuard()
        {
            if(armed)
                barrier.abort();
        }
    };
}

#define CHECK_HIP(expr)                                                                 \
    do                                                                                  \
    {                                                                                   \
        const hipError_t _e = (expr);                                                   \
        if(_e != hipSuccess)                                                            \
        {                                                                               \
            std::printf("rank %u failed: %s -> %s\n", rank, #expr, hipGetErrorString(_e)); \
            return false;                                                               \
        }                                                                               \
    } while(0)

#define CHECK_LT(expr)                                                        \
    do                                                                        \
    {                                                                         \
        const hipblasStatus_t _s = (expr);                                    \
        if(_s != HIPBLAS_STATUS_SUCCESS)                                      \
        {                                                                     \
            std::printf("rank %u failed: %s -> status %d\n", rank, #expr, int(_s)); \
            return false;                                                     \
        }                                                                     \
    } while(0)

namespace
{
    bool runRank(uint32_t rank, Rank& self, Rank* all, AllgatherSlot& slot, Barrier& barrier)
    {
        BarrierGuard  barrierGuard{barrier};
        MatmulHandles handles;

        CHECK_HIP(hipSetDevice(int(rank)));

        const size_t bytesA    = size_t(kK) * kFeatures * sizeof(uint16_t);
        const size_t bytesB    = size_t(kK) * kTokens * sizeof(uint16_t);
        const size_t bytesCD   = size_t(kFeatures) * kTokens * sizeof(uint16_t);
        const size_t bytesRecv = size_t(kWorld) * kTokens * kShard * sizeof(uint16_t);

        CHECK_HIP(hipMalloc(&self.dA, bytesA));
        CHECK_HIP(hipMalloc(&self.dB, bytesB));
        CHECK_HIP(hipMalloc(&self.dC, bytesCD));
        CHECK_HIP(hipMalloc(&self.dD, bytesCD));
        CHECK_HIP(hipMalloc(&self.dRecv, bytesRecv));
        CHECK_HIP(hipMalloc(&self.dWorkspace, kWorkspaceSize));

        // A is K x kFeatures and B is K x kTokens, both column major and read
        // transposed / not, so D[f][t] = A[f][0] * B[t][0] = expectedD(rank, f, t).
        std::vector<uint16_t> hA(size_t(kK) * kFeatures, toBf16(0.0f));
        std::vector<uint16_t> hB(size_t(kK) * kTokens, toBf16(0.0f));
        for(int64_t f = 0; f < kFeatures; ++f)
            hA[size_t(f) * kK] = toBf16(float((rank + 1) * ((f % 7) + 1)));
        for(int64_t t = 0; t < kTokens; ++t)
            hB[size_t(t) * kK] = toBf16(float((t % 5) + 1));

        CHECK_HIP(hipMemcpy(self.dA, hA.data(), bytesA, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(self.dB, hB.data(), bytesB, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(self.dC, 0, bytesCD));
        CHECK_HIP(hipMemset(self.dD, 0, bytesCD));
        CHECK_HIP(hipMemset(self.dRecv, 0, bytesRecv));

        CHECK_LT(hipblasLtCreate(&self.handle));
        CHECK_LT(hipblasLtSetDeviceComm(
            self.handle, rank, kWorld, kChannels, threadAllgather, &slot));

        if(!barrier.arrive())
            return false;

        for(uint32_t j = 0; j < kWorld; ++j)
            self.recvPtrs[j] = all[j].dRecv;

        auto& fused = handles.fused;
        CHECK_LT(hipblasLtFusedEpilogueCreate(&fused));
        CHECK_LT(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX));

        // Both attributes take the per-rank array itself, sized in whole entries.
        CHECK_LT(hipblasLtFusedEpilogueSetAttribute(fused,
                                                    HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                                                    self.queues,
                                                    kWorld * sizeof(self.queues[0])));
        CHECK_LT(hipblasLtFusedEpilogueSetAttribute(fused,
                                                    HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS,
                                                    self.recvPtrs,
                                                    kWorld * sizeof(self.recvPtrs[0])));
        const int64_t extent = kExtent;
        CHECK_LT(hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &extent, sizeof(extent)));
        const hipblasLtA2ACompletionMode_t mode = HIPBLASLT_A2A_COMPLETION_IN_KERNEL;
        CHECK_LT(hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_COMPLETION_MODE, &mode, sizeof(mode)));
        const uint32_t channel = kChannel;
        CHECK_LT(hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL, &channel, sizeof(channel)));

        auto &layA = handles.lay[0], &layB = handles.lay[1];
        auto &layC = handles.lay[2], &layD = handles.lay[3];
        CHECK_LT(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, kK, kFeatures, kK));
        CHECK_LT(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, kK, kTokens, kK));
        CHECK_LT(hipblasLtMatrixLayoutCreate(&layC, HIP_R_16BF, kFeatures, kTokens, kFeatures));
        CHECK_LT(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, kFeatures, kTokens, kFeatures));

        auto& mm = handles.mm;
        CHECK_LT(hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        CHECK_LT(
            hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CHECK_LT(
            hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        CHECK_LT(hipblasLtMatmulDescSetAttribute(
            mm, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused)));

        auto& pref = handles.pref;
        CHECK_LT(hipblasLtMatmulPreferenceCreate(&pref));
        CHECK_LT(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                       HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &kWorkspaceSize,
                                                       sizeof(kWorkspaceSize)));

        hipblasLtMatmulHeuristicResult_t heur[1];
        int                              algoCount = 0;
        CHECK_LT(hipblasLtMatmulAlgoGetHeuristic(
            self.handle, mm, layA, layB, layC, layD, pref, 1, heur, &algoCount));
        if(algoCount == 0)
        {
            self.skipped = true;
            return false;
        }

        // IN_KERNEL completion: every rank has to be in flight at once.
        if(!barrier.arrive())
            return false;

        const float alpha = 1.0f, beta = 0.0f;
        CHECK_LT(hipblasLtMatmul(self.handle,
                                 mm,
                                 &alpha,
                                 self.dA,
                                 layA,
                                 self.dB,
                                 layB,
                                 &beta,
                                 self.dC,
                                 layC,
                                 self.dD,
                                 layD,
                                 &heur[0].algo,
                                 self.dWorkspace,
                                 kWorkspaceSize,
                                 nullptr));
        CHECK_HIP(hipDeviceSynchronize());

        self.hD.resize(size_t(kFeatures) * kTokens);
        self.hRecv.resize(size_t(kWorld) * kTokens * kShard);
        CHECK_HIP(hipMemcpy(self.hD.data(), self.dD, bytesCD, hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(self.hRecv.data(), self.dRecv, bytesRecv, hipMemcpyDeviceToHost));

        barrierGuard.armed = false;
        return true;
    }
}

#undef CHECK_HIP
#undef CHECK_LT

namespace
{
    bool devicesUsable()
    {
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < int(kWorld))
        {
            std::printf("skipped: needs %u devices, found %d\n", kWorld, deviceCount);
            return false;
        }
        for(uint32_t d = 0; d < kWorld; ++d)
        {
            hipDeviceProp_t props{};
            if(hipGetDeviceProperties(&props, int(d)) != hipSuccess
               || std::strncmp(props.gcnArchName, "gfx950", 6) != 0)
            {
                std::printf("skipped: fused GEMM+A2A is wired for gfx950 only\n");
                return false;
            }
        }
        for(uint32_t a = 0; a < kWorld; ++a)
        {
            for(uint32_t b = 0; b < kWorld; ++b)
            {
                if(a == b)
                    continue;
                int canAccess = 0;
                if(hipDeviceCanAccessPeer(&canAccess, int(a), int(b)) != hipSuccess
                   || canAccess == 0)
                {
                    std::printf("skipped: device %u cannot reach device %u\n", a, b);
                    return false;
                }
            }
        }
        return true;
    }

    bool enablePeerAccess()
    {
        for(uint32_t a = 0; a < kWorld; ++a)
        {
            if(hipSetDevice(int(a)) != hipSuccess)
                return false;
            for(uint32_t b = 0; b < kWorld; ++b)
            {
                if(a == b)
                    continue;
                const hipError_t e = hipDeviceEnablePeerAccess(int(b), 0);
                if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                {
                    std::printf("failed: hipDeviceEnablePeerAccess(%u -> %u) -> %s\n",
                                a,
                                b,
                                hipGetErrorString(e));
                    return false;
                }
            }
        }
        return true;
    }

    bool checkLocalD(const Rank* all)
    {
        for(uint32_t s = 0; s < kWorld; ++s)
            for(int64_t t = 0; t < kTokens; ++t)
                for(int64_t f = 0; f < kFeatures; ++f)
                {
                    const float got  = fromBf16(all[s].hD[size_t(t) * kFeatures + f]);
                    const float want = expectedD(s, f, t);
                    if(got != want)
                    {
                        std::printf("FAILED: rank %u D(feature %lld, token %lld) = %.1f, "
                                    "expected %.1f\n",
                                    s,
                                    (long long)f,
                                    (long long)t,
                                    got,
                                    want);
                        return false;
                    }
                }
        return true;
    }

    bool checkRecv(const Rank* all)
    {
        for(uint32_t r = 0; r < kWorld; ++r)
            for(uint32_t s = 0; s < kWorld; ++s)
                for(int64_t t = 0; t < kTokens; ++t)
                    for(int64_t fw = 0; fw < kShard; ++fw)
                    {
                        const size_t   at = (size_t(s) * kTokens + size_t(t)) * kShard + fw;
                        const int64_t  f  = int64_t(r) * kShard + fw;
                        const uint16_t got  = all[r].hRecv[at];
                        const uint16_t want = all[s].hD[size_t(t) * kFeatures + f];
                        if(got != want)
                        {
                            std::printf("FAILED: rank %u recv from source %u at token %lld "
                                        "feature %lld = %.1f, expected %.1f\n",
                                        r,
                                        s,
                                        (long long)t,
                                        (long long)f,
                                        fromBf16(got),
                                        fromBf16(want));
                            return false;
                        }
                    }
        return true;
    }
}

int main()
{
    if(!devicesUsable())
        return kStatusOk;
    if(!enablePeerAccess())
        return kStatusFailed;

    Rank ranks[kWorld];

    std::vector<std::unique_ptr<TensileLite::Client::SdmaQueue>> ownedQueues;
    try
    {
        ownedQueues.reserve(size_t(kWorld) * kWorld);
        for(uint32_t r = 0; r < kWorld; ++r)
        {
            const uint32_t srcNode = TensileLite::Client::sdmaNodeIdForDevice(int(r));
            for(uint32_t j = 0; j < kWorld; ++j)
            {
                const uint32_t dstNode = TensileLite::Client::sdmaNodeIdForDevice(int(j));
                ownedQueues.push_back(std::make_unique<TensileLite::Client::SdmaQueue>(
                    srcNode, TensileLite::Client::sdmaSelectEngine(srcNode, dstNode)));
                const HsaQueueResource& q = ownedQueues.back()->queueResource();
                ranks[r].queues[j]        = {ownedQueues.back()->ringBase(),
                                             (void*)q.Queue_read_ptr_aql,
                                             (void*)q.Queue_write_ptr_aql,
                                             (void*)q.Queue_DoorBell_aql};
            }
        }
    }
    catch(const std::exception& e)
    {
        std::printf("skipped: cannot create an SDMA queue (%s)\n", e.what());
        return kStatusOk;
    }

    CommContext                comm;
    AllgatherSlot              slots[kWorld];
    Barrier                    barrier(kWorld);
    std::vector<std::thread>   threads;
    for(uint32_t r = 0; r < kWorld; ++r)
    {
        slots[r] = {&comm, r};
        threads.emplace_back([&, r] { ranks[r].ok = runRank(r, ranks[r], ranks, slots[r], barrier); });
    }
    for(auto& thread : threads)
        thread.join();

    bool skipped = false;
    for(uint32_t r = 0; r < kWorld; ++r)
        skipped = skipped || ranks[r].skipped;

    int status = kStatusOk;
    if(skipped)
        std::printf("skipped: loaded device library carries no fused GEMM+A2A solution\n");
    else
    {
        for(uint32_t r = 0; r < kWorld; ++r)
            if(!ranks[r].ok)
                status = kStatusFailed;
        if(status == kStatusOk && (!checkLocalD(ranks) || !checkRecv(ranks)))
            status = kStatusFailed;
        if(status == kStatusOk)
            std::printf("A2A SDMA verified over %u ranks: %lld features x %lld tokens, "
                        "extent %lld, shard %lld\n",
                        kWorld,
                        (long long)kFeatures,
                        (long long)kTokens,
                        (long long)kExtent,
                        (long long)kShard);
    }

    for(uint32_t r = 0; r < kWorld; ++r)
    {
        // Each rank's buffers live on its own device.
        static_cast<void>(hipSetDevice(int(r)));
        static_cast<void>(hipFree(ranks[r].dA));
        static_cast<void>(hipFree(ranks[r].dB));
        static_cast<void>(hipFree(ranks[r].dC));
        static_cast<void>(hipFree(ranks[r].dD));
        static_cast<void>(hipFree(ranks[r].dRecv));
        static_cast<void>(hipFree(ranks[r].dWorkspace));
        if(ranks[r].handle != nullptr)
            hipblasLtDestroy(ranks[r].handle);
    }
    return status;
}
