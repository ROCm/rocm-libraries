// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fused GEMM+A2A over the public API with real SDMA queues, run as a mixed
// deployment: four ranks over two processes, two ranks each. One communicator
// therefore carries a self peer, a same-process peer and two cross-process
// peers, and the recv pointer for each kind is resolved a different way -- the
// buffer itself, a raw pointer valid process-wide, and an IPC mapping.
//
// Only the recv buffers are the caller's to exchange; the flag buffer's IPC
// round trip belongs to hipblasLtSetDeviceComm.
//
// Axis naming follows the fused-A2A contract rather than GEMM habit: M / free0
// is the FEATURE axis, contiguous in a column-major D, and N / free1 is the
// TOKEN axis. Only the first kExtent features take the A2A path.
//
// Rank s hands peer p the feature shard [p*kShard, (p+1)*kShard) of all its
// tokens, landing at recv_p[(s*kTokens + t)*kShard + fw]. Each process checks
// its own ranks against the closed form both sides generate from; no D crosses
// the socket.
//
// Once checked, every rank re-fills A and B from the alphabet the tensilelite
// client draws from and repeats the launch behind a per-iteration barrier,
// reporting latency as the max across ranks over kTimingIterations.
//
// Needs a device library holding a FusedGemmA2A solution. Run with:
//
//   HIP_VISIBLE_DEVICES=<four peer-capable cards> \
//   HIPBLASLT_TENSILE_LIBPATH=<devlib>/library/gfx950 \
//   ./sample_a2a_sdma_multiprocess
//
// Exits 0 and prints a reason when the machine cannot host the run or the loaded
// library carries no fused GEMM+A2A solution. A library that fails to load is an
// error rather than a skip.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <SdmaQueue.hpp>

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    constexpr int64_t  kFeatures        = 18432; // M / free0
    constexpr int64_t  kTokens          = 2048; // N / free1
    constexpr int64_t  kK               = 8192;
    constexpr int64_t  kExtent          = 10240;
    constexpr uint32_t kWorld           = 4;
    constexpr uint32_t kRanksPerProcess = 2;
    constexpr uint32_t kChannels        = 2;
    constexpr uint32_t kChannel         = 1;

    constexpr int64_t kShard = kExtent / kWorld;

    constexpr size_t kWorkspaceSize   = 32ull * 1024 * 1024;
    constexpr int    kSocketTimeoutSec = 30;

    constexpr int kTimingIterations = 50;
    constexpr int kTimingWarmup     = 10;

    // Ordered worst-last: combining two verdicts is a max.
    constexpr char kGo   = 0;
    constexpr char kSkip = 1;
    constexpr char kFail = 2;

    constexpr int kStatusOk      = 0;
    constexpr int kStatusFailed  = 1;
    constexpr int kStatusSkipped = 2;

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

    // Matches the (x%7)-3 alphabet the tensilelite client draws from.
    float nextSample(uint32_t& state, float scale)
    {
        state = state * 1664525u + 1013904223u;
        return float(int((state >> 16) % 7) - 3) * scale;
    }

    bool writeAll(int fd, const void* data, size_t bytes)
    {
        const char* p = static_cast<const char*>(data);
        while(bytes > 0)
        {
            const ssize_t n = ::write(fd, p, bytes);
            if(n <= 0)
                return false;
            p += n;
            bytes -= size_t(n);
        }
        return true;
    }

    bool readAll(int fd, void* data, size_t bytes)
    {
        char* p = static_cast<char*>(data);
        while(bytes > 0)
        {
            const ssize_t n = ::read(fd, p, bytes);
            if(n <= 0)
                return false;
            p += n;
            bytes -= size_t(n);
        }
        return true;
    }

    // The last of kRanksPerProcess local threads to arrive runs swap, which is
    // where the socket traffic happens; the others wait for it.
    class Rendezvous
    {
    public:
        bool arrive(const std::function<bool()>& swap)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const uint64_t               epoch = epoch_;
            if(++arrived_ == kRanksPerProcess)
            {
                failed_  = !swap();
                arrived_ = 0;
                ++epoch_;
                cv_.notify_all();
            }
            else if(!cv_.wait_for(lock, std::chrono::seconds(kSocketTimeoutSec), [this, epoch] {
                        return epoch_ != epoch;
                    }))
            {
                return false;
            }
            return !failed_;
        }

    private:
        std::mutex              mutex_;
        std::condition_variable cv_;
        uint32_t                arrived_ = 0;
        uint64_t                epoch_   = 0;
        bool                    failed_  = false;
    };

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

        std::vector<double> latencies;
    };

    struct Events
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;

        ~Events()
        {
            if(start != nullptr)
                static_cast<void>(hipEventDestroy(start));
            if(stop != nullptr)
                static_cast<void>(hipEventDestroy(stop));
        }
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

    struct ProcessContext
    {
        int      sock     = -1;
        uint32_t rankBase = 0;

        Rank ranks[kRanksPerProcess];

        // This process's own recv handles, then the peer process's.
        hipIpcMemHandle_t mine[kRanksPerProcess]   = {};
        hipIpcMemHandle_t theirs[kRanksPerProcess] = {};

        std::mutex        recordMutex;
        std::vector<char> records;
        Rendezvous        allgather;

        Rendezvous verdictSwap;
        std::mutex verdictMutex;
        char       localVerdict = kGo;
        char       verdict      = kGo;

        Rendezvous          timingBarrier;
        std::vector<double> perfUs;
    };

    uint32_t peerRankBase(uint32_t rankBase)
    {
        return rankBase == 0 ? kRanksPerProcess : 0;
    }

    struct AllgatherSlot
    {
        ProcessContext* ctx  = nullptr;
        uint32_t        rank = 0;
    };

    // The peer process contributes the other half of the record array; this
    // process fills its own half before arriving.
    hipblasStatus_t mixedAllgather(void*       userData,
                                   const void* sendbuf,
                                   void*       recvbuf,
                                   size_t      bytesPerRank)
    {
        auto*           slot = static_cast<AllgatherSlot*>(userData);
        ProcessContext& ctx  = *slot->ctx;

        {
            std::lock_guard<std::mutex> lock(ctx.recordMutex);
            if(ctx.records.empty())
                ctx.records.resize(bytesPerRank * kWorld);
            if(ctx.records.size() != bytesPerRank * kWorld)
                return HIPBLAS_STATUS_INVALID_VALUE;
            std::memcpy(ctx.records.data() + bytesPerRank * slot->rank, sendbuf, bytesPerRank);
        }

        const bool swapped = ctx.allgather.arrive([&ctx, bytesPerRank] {
            const size_t chunk = bytesPerRank * kRanksPerProcess;
            return writeAll(ctx.sock, ctx.records.data() + bytesPerRank * ctx.rankBase, chunk)
                   && readAll(ctx.sock,
                              ctx.records.data() + bytesPerRank * peerRankBase(ctx.rankBase),
                              chunk);
        });
        if(!swapped)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        std::memcpy(recvbuf, ctx.records.data(), ctx.records.size());
        return HIPBLAS_STATUS_SUCCESS;
    }

    void reportVerdict(ProcessContext& ctx, char verdict)
    {
        std::lock_guard<std::mutex> lock(ctx.verdictMutex);
        if(verdict > ctx.localVerdict)
            ctx.localVerdict = verdict;
    }

    // Doubles as the pre-matmul barrier: IN_KERNEL completion needs every rank
    // in flight at once.
    char agreeVerdict(ProcessContext& ctx)
    {
        const bool swapped = ctx.verdictSwap.arrive([&ctx] {
            char mine = kGo;
            {
                std::lock_guard<std::mutex> lock(ctx.verdictMutex);
                mine = ctx.localVerdict;
            }
            char theirs = kFail;
            if(!writeAll(ctx.sock, &mine, 1) || !readAll(ctx.sock, &theirs, 1))
                return false;
            ctx.verdict = theirs > mine ? theirs : mine;
            return true;
        });
        return swapped ? ctx.verdict : kFail;
    }
}

#define CHECK_HIP(expr)                                                                    \
    do                                                                                     \
    {                                                                                      \
        const hipError_t _e = (expr);                                                      \
        if(_e != hipSuccess)                                                               \
        {                                                                                  \
            std::printf("rank %u failed: %s -> %s\n", rank, #expr, hipGetErrorString(_e)); \
            return kFail;                                                                  \
        }                                                                                  \
    } while(0)

#define CHECK_LT(expr)                                                              \
    do                                                                              \
    {                                                                               \
        const hipblasStatus_t _s = (expr);                                          \
        if(_s != HIPBLAS_STATUS_SUCCESS)                                            \
        {                                                                           \
            std::printf("rank %u failed: %s -> status %d\n", rank, #expr, int(_s)); \
            return kFail;                                                           \
        }                                                                           \
    } while(0)

namespace
{
    char fillDense(ProcessContext& ctx, uint32_t rank)
    {
        Rank& self = ctx.ranks[rank - ctx.rankBase];

        const size_t elemsA = size_t(kK) * kFeatures;
        const size_t elemsB = size_t(kK) * kTokens;

        std::vector<uint16_t> host(elemsA);
        uint32_t              state = rank + 1;
        for(auto& v : host)
            v = toBf16(nextSample(state, 0.5f));
        CHECK_HIP(
            hipMemcpy(self.dA, host.data(), elemsA * sizeof(uint16_t), hipMemcpyHostToDevice));

        host.resize(elemsB);
        for(auto& v : host)
            v = toBf16(nextSample(state, 0.25f));
        CHECK_HIP(
            hipMemcpy(self.dB, host.data(), elemsB * sizeof(uint16_t), hipMemcpyHostToDevice));
        return kGo;
    }

    char runRank(ProcessContext& ctx, uint32_t rank, AllgatherSlot& slot)
    {
        Rank&         self = ctx.ranks[rank - ctx.rankBase];
        MatmulHandles handles;

        CHECK_HIP(hipSetDevice(int(rank)));

        CHECK_LT(hipblasLtCreate(&self.handle));
        CHECK_LT(hipblasLtSetDeviceComm(
            self.handle, rank, kWorld, kChannels, mixedAllgather, &slot));

        for(uint32_t j = 0; j < kWorld; ++j)
        {
            if(j >= ctx.rankBase && j < ctx.rankBase + kRanksPerProcess)
                self.recvPtrs[j] = ctx.ranks[j - ctx.rankBase].dRecv;
            else
                CHECK_HIP(hipIpcOpenMemHandle(&self.recvPtrs[j],
                                              ctx.theirs[j - peerRankBase(ctx.rankBase)],
                                              hipIpcMemLazyEnablePeerAccess));
        }

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
            reportVerdict(ctx, kSkip);

        const char verdict = agreeVerdict(ctx);
        if(verdict != kGo)
            return verdict;

        const float alpha = 1.0f, beta = 0.0f;
        auto        launch = [&] {
            return hipblasLtMatmul(self.handle,
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
                                   nullptr);
        };
        CHECK_LT(launch());
        CHECK_HIP(hipDeviceSynchronize());

        const size_t bytesCD   = size_t(kFeatures) * kTokens * sizeof(uint16_t);
        const size_t bytesRecv = size_t(kWorld) * kTokens * kShard * sizeof(uint16_t);
        self.hD.resize(size_t(kFeatures) * kTokens);
        self.hRecv.resize(size_t(kWorld) * kTokens * kShard);
        CHECK_HIP(hipMemcpy(self.hD.data(), self.dD, bytesCD, hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(self.hRecv.data(), self.dRecv, bytesRecv, hipMemcpyDeviceToHost));

        if(fillDense(ctx, rank) != kGo)
            return kFail;

        Events events;
        CHECK_HIP(hipEventCreate(&events.start));
        CHECK_HIP(hipEventCreate(&events.stop));
        self.latencies.reserve(kTimingIterations);

        for(int it = 0; it < kTimingWarmup + kTimingIterations; ++it)
        {
            if(!ctx.timingBarrier.arrive([&ctx] {
                   char token = 0;
                   return writeAll(ctx.sock, &token, 1) && readAll(ctx.sock, &token, 1);
               }))
            {
                std::printf("rank %u failed: timing barrier\n", rank);
                return kFail;
            }

            CHECK_HIP(hipEventRecord(events.start, nullptr));
            CHECK_LT(launch());
            CHECK_HIP(hipEventRecord(events.stop, nullptr));
            CHECK_HIP(hipDeviceSynchronize());

            if(it >= kTimingWarmup)
            {
                float ms = 0.0f;
                CHECK_HIP(hipEventElapsedTime(&ms, events.start, events.stop));
                self.latencies.push_back(double(ms) * 1000.0);
            }
        }
        return kGo;
    }

    char allocateRank(ProcessContext& ctx, uint32_t rank)
    {
        Rank& self = ctx.ranks[rank - ctx.rankBase];
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
        // transposed / not: D[f][t] = A[f][0] * B[t][0] = expectedD(rank, f, t).
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

        CHECK_HIP(hipIpcGetMemHandle(&ctx.mine[rank - ctx.rankBase], self.dRecv));
        return kGo;
    }
}

#undef CHECK_HIP
#undef CHECK_LT

namespace
{
    bool topologyAvailable(bool report)
    {
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < int(kWorld))
        {
            if(report)
                std::printf("skipped: needs %u devices, found %d\n", kWorld, deviceCount);
            return false;
        }
        for(uint32_t d = 0; d < kWorld; ++d)
        {
            hipDeviceProp_t props{};
            if(hipGetDeviceProperties(&props, int(d)) != hipSuccess
               || std::strncmp(props.gcnArchName, "gfx950", 6) != 0)
            {
                if(report)
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
                    if(report)
                        std::printf("skipped: device %u cannot reach device %u\n", a, b);
                    return false;
                }
            }
        }
        return true;
    }

    bool enablePeerAccess(const ProcessContext& ctx)
    {
        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            const uint32_t rank = ctx.rankBase + slot;
            if(hipSetDevice(int(rank)) != hipSuccess)
                return false;
            for(uint32_t j = 0; j < kWorld; ++j)
            {
                if(j == rank)
                    continue;
                const hipError_t e = hipDeviceEnablePeerAccess(int(j), 0);
                if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                {
                    std::printf("failed: hipDeviceEnablePeerAccess(%u -> %u) -> %s\n",
                                rank,
                                j,
                                hipGetErrorString(e));
                    return false;
                }
            }
        }
        return true;
    }

    bool createQueues(ProcessContext&                                               ctx,
                      std::vector<std::unique_ptr<TensileLite::Client::SdmaQueue>>& owned)
    {
        try
        {
            owned.reserve(size_t(kRanksPerProcess) * kWorld);
            for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
            {
                const uint32_t rank    = ctx.rankBase + slot;
                const uint32_t srcNode = TensileLite::Client::sdmaNodeIdForDevice(int(rank));
                for(uint32_t j = 0; j < kWorld; ++j)
                {
                    const uint32_t dstNode = TensileLite::Client::sdmaNodeIdForDevice(int(j));
                    owned.push_back(std::make_unique<TensileLite::Client::SdmaQueue>(
                        srcNode, TensileLite::Client::sdmaSelectEngine(srcNode, dstNode)));
                    const HsaQueueResource& q  = owned.back()->queueResource();
                    ctx.ranks[slot].queues[j]  = {owned.back()->ringBase(),
                                                  (void*)q.Queue_read_ptr_aql,
                                                  (void*)q.Queue_write_ptr_aql,
                                                  (void*)q.Queue_DoorBell_aql};
                }
            }
        }
        catch(const std::exception& e)
        {
            std::printf("skipped: cannot create an SDMA queue (%s)\n", e.what());
            return false;
        }
        return true;
    }

    bool checkRank(const ProcessContext& ctx, uint32_t rank)
    {
        const Rank& self = ctx.ranks[rank - ctx.rankBase];

        for(int64_t t = 0; t < kTokens; ++t)
            for(int64_t f = 0; f < kFeatures; ++f)
            {
                const float got  = fromBf16(self.hD[size_t(t) * kFeatures + f]);
                const float want = expectedD(rank, f, t);
                if(got != want)
                {
                    std::printf("FAILED: rank %u D(feature %lld, token %lld) = %.1f, "
                                "expected %.1f\n",
                                rank,
                                (long long)f,
                                (long long)t,
                                got,
                                want);
                    return false;
                }
            }

        for(uint32_t s = 0; s < kWorld; ++s)
            for(int64_t t = 0; t < kTokens; ++t)
                for(int64_t fw = 0; fw < kShard; ++fw)
                {
                    const size_t  at = (size_t(s) * kTokens + size_t(t)) * kShard + fw;
                    const int64_t f  = int64_t(rank) * kShard + fw;
                    const float   got  = fromBf16(self.hRecv[at]);
                    const float   want = expectedD(s, f, t);
                    if(got != want)
                    {
                        std::printf("FAILED: rank %u recv from source %u at token %lld "
                                    "feature %lld = %.1f, expected %.1f\n",
                                    rank,
                                    s,
                                    (long long)t,
                                    (long long)f,
                                    got,
                                    want);
                        return false;
                    }
                }
        return true;
    }

    void reportPerf(std::vector<double> us)
    {
        std::sort(us.begin(), us.end());
        const double lo     = us.front();
        const double median = us[us.size() / 2];
        const double p99    = us[(us.size() * 99) / 100];

        const double gflop  = 2.0 * double(kFeatures) * kTokens * kK / 1e9;
        const double egress = double(kExtent) * kTokens * sizeof(uint16_t) * double(kWorld - 1)
                              / double(kWorld) / 1e6;

        std::printf("perf (whole call, max across ranks, %d iterations after %d warmup; rates at "
                    "min latency):\n",
                    kTimingIterations,
                    kTimingWarmup);
        std::printf(
            "  latency      min %.1f us    median %.1f us    p99 %.1f us\n", lo, median, p99);
        std::printf("  GEMM         %.0f TFLOP/s per rank (%.1f GFLOP)\n", gflop / lo * 1e3, gflop);
        std::printf("  A2A egress   %.1f GB/s per rank (%.1f MB)\n", egress / lo * 1e3, egress);
    }

    int runProcess(ProcessContext& ctx)
    {
        const bool report = ctx.rankBase == 0;
        if(!topologyAvailable(report))
            return kStatusSkipped;
        if(!enablePeerAccess(ctx))
            return kStatusFailed;

        std::vector<std::unique_ptr<TensileLite::Client::SdmaQueue>> owned;
        if(!createQueues(ctx, owned))
            return kStatusSkipped;

        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
            if(allocateRank(ctx, ctx.rankBase + slot) != kGo)
                return kStatusFailed;

        const size_t handleChunk = sizeof(hipIpcMemHandle_t) * kRanksPerProcess;
        if(!writeAll(ctx.sock, ctx.mine, handleChunk)
           || !readAll(ctx.sock, ctx.theirs, handleChunk))
        {
            std::printf("failed: recv handle exchange\n");
            return kStatusFailed;
        }

        AllgatherSlot            slots[kRanksPerProcess];
        char                     verdicts[kRanksPerProcess] = {kFail, kFail};
        std::vector<std::thread> threads;
        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            slots[slot] = {&ctx, ctx.rankBase + slot};
            threads.emplace_back([&ctx, &slots, &verdicts, slot] {
                verdicts[slot] = runRank(ctx, ctx.rankBase + slot, slots[slot]);
            });
        }
        for(auto& thread : threads)
            thread.join();

        int status = kStatusOk;
        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            if(verdicts[slot] == kFail)
                status = kStatusFailed;
            else if(verdicts[slot] == kSkip && status == kStatusOk)
                status = kStatusSkipped;
        }
        if(status == kStatusSkipped && report)
            std::printf("skipped: loaded device library carries no fused GEMM+A2A solution\n");

        if(status == kStatusOk)
            for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
                if(!checkRank(ctx, ctx.rankBase + slot))
                    status = kStatusFailed;

        std::vector<double> local(kTimingIterations, 0.0), peer(kTimingIterations, 0.0);
        if(status == kStatusOk)
            for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
                for(int i = 0; i < kTimingIterations; ++i)
                    local[i] = std::max(local[i], ctx.ranks[slot].latencies[i]);

        const size_t bytes = local.size() * sizeof(double);
        if(!writeAll(ctx.sock, local.data(), bytes) || !readAll(ctx.sock, peer.data(), bytes))
        {
            std::printf("failed: latency exchange\n");
            status = kStatusFailed;
        }
        else if(status == kStatusOk)
        {
            for(int i = 0; i < kTimingIterations; ++i)
                local[i] = std::max(local[i], peer[i]);
            ctx.perfUs = std::move(local);
        }

        for(uint32_t slot = 0; slot < kRanksPerProcess; ++slot)
        {
            Rank& rank = ctx.ranks[slot];
            static_cast<void>(hipSetDevice(int(ctx.rankBase + slot)));

            // Only the peer process's buffers were mapped in; ours are plain
            // allocations that hipFree below owns.
            for(uint32_t j = 0; j < kWorld; ++j)
                if(rank.recvPtrs[j] != nullptr
                   && !(j >= ctx.rankBase && j < ctx.rankBase + kRanksPerProcess))
                    static_cast<void>(hipIpcCloseMemHandle(rank.recvPtrs[j]));

            static_cast<void>(hipFree(rank.dA));
            static_cast<void>(hipFree(rank.dB));
            static_cast<void>(hipFree(rank.dC));
            static_cast<void>(hipFree(rank.dD));
            static_cast<void>(hipFree(rank.dRecv));
            static_cast<void>(hipFree(rank.dWorkspace));
            if(rank.handle != nullptr)
                hipblasLtDestroy(rank.handle);
        }
        return status;
    }
}

int main()
{
    int sockets[2] = {-1, -1};
    if(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0)
    {
        std::printf("socketpair failed\n");
        return kStatusFailed;
    }

    timeval timeout{};
    timeout.tv_sec = kSocketTimeoutSec;
    for(int fd : sockets)
    {
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    }

    // Nothing above has touched HIP or KFD: both are set up after the fork, once
    // per process.
    const pid_t child = ::fork();
    if(child < 0)
    {
        std::printf("fork failed\n");
        return kStatusFailed;
    }

    ProcessContext ctx;
    if(child == 0)
    {
        ::close(sockets[0]);
        ctx.sock         = sockets[1];
        ctx.rankBase     = kRanksPerProcess;
        const int status = runProcess(ctx);
        ::close(ctx.sock);
        std::fflush(stdout);
        ::_exit(status);
    }

    ::close(sockets[1]);
    ctx.sock                = sockets[0];
    ctx.rankBase            = 0;
    const int parentStatus  = runProcess(ctx);
    ::close(ctx.sock);

    int childRaw = 0;
    ::waitpid(child, &childRaw, 0);
    const int childStatus = WIFEXITED(childRaw) ? WEXITSTATUS(childRaw) : kStatusFailed;

    if(parentStatus == kStatusFailed || childStatus == kStatusFailed)
        return kStatusFailed;
    if(parentStatus == kStatusSkipped || childStatus == kStatusSkipped)
        return kStatusOk;

    std::printf("A2A SDMA verified over %u ranks in %u processes: %lld features x %lld tokens, "
                "extent %lld, shard %lld\n",
                kWorld,
                kWorld / kRanksPerProcess,
                (long long)kFeatures,
                (long long)kTokens,
                (long long)kExtent,
                (long long)kShard);
    if(!ctx.perfUs.empty())
        reportPerf(ctx.perfUs);
    return kStatusOk;
}
