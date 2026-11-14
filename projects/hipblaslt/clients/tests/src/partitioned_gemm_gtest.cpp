// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for hipblaslt_ext::PartitionedGemm.
//
//  pre_checkin_partitioned_gemm_ext  run on any single GPU. The "partitions" are extra handles
//                                    on device 0, which is enough for the argument checks and
//                                    for a one-partition run.
//  partitioned_gemm_ext_multi_gpu    need two or more devices (the partitions of a DPX card, or
//                                    two GPUs); they skip otherwise.

#include "utility.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

using namespace hipblaslt_ext;

namespace
{
    using half_t = _Float16;

    // A device buffer on one device.
    class DeviceBuf
    {
    public:
        DeviceBuf(int device, size_t bytes, const void* init = nullptr)
            : m_device(device)
        {
            EXPECT_EQ(hipSetDevice(device), hipSuccess);
            EXPECT_EQ(hipMalloc(&m_ptr, bytes), hipSuccess);
            if(init != nullptr)
                EXPECT_EQ(hipMemcpy(m_ptr, init, bytes, hipMemcpyHostToDevice), hipSuccess);
            else
                EXPECT_EQ(hipMemset(m_ptr, 0, bytes), hipSuccess);
        }
        ~DeviceBuf()
        {
            if(m_ptr != nullptr)
            {
                static_cast<void>(hipSetDevice(m_device));
                static_cast<void>(hipFree(m_ptr));
            }
        }
        DeviceBuf(const DeviceBuf&)            = delete;
        DeviceBuf& operator=(const DeviceBuf&) = delete;
        void*      get() const
        {
            return m_ptr;
        }

    private:
        int   m_device;
        void* m_ptr = nullptr;
    };

    std::vector<half_t> randomHalf(size_t count, unsigned seed)
    {
        uint32_t            state = seed * 2654435761u + 1;
        std::vector<half_t> v(count);
        for(auto& x : v)
        {
            state = state * 1664525u + 1013904223u;
            x     = half_t(float(state >> 9) / float(1u << 23) - 0.5f);
        }
        return v;
    }

    // One handle, stream and workspace per partition. `devices[i]` is the device of partition i.
    struct Partitions
    {
        explicit Partitions(std::vector<int> const& devices)
            : devices(devices)
        {
            for(int device : devices)
            {
                EXPECT_EQ(hipSetDevice(device), hipSuccess);
                handles.push_back(std::make_unique<hipblaslt_local_handle>());
                hipStream_t stream = nullptr;
                EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
                streams.push_back(stream);
                workspaceBufs.push_back(std::make_unique<DeviceBuf>(device, kWorkspace));
            }
        }
        ~Partitions()
        {
            for(size_t i = 0; i < streams.size(); i++)
            {
                static_cast<void>(hipSetDevice(devices[i]));
                static_cast<void>(hipStreamSynchronize(streams[i]));
                static_cast<void>(hipStreamDestroy(streams[i]));
            }
        }

        std::vector<hipblasLtHandle_t> rawHandles() const
        {
            std::vector<hipblasLtHandle_t> out;
            for(auto const& h : handles)
                out.push_back(static_cast<const hipblasLtHandle_t&>(*h));
            return out;
        }
        std::vector<void*> workspaces() const
        {
            std::vector<void*> out;
            for(auto const& w : workspaceBufs)
                out.push_back(w->get());
            return out;
        }
        void sync() const
        {
            for(size_t i = 0; i < streams.size(); i++)
            {
                ASSERT_EQ(hipSetDevice(devices[i]), hipSuccess);
                ASSERT_EQ(hipStreamSynchronize(streams[i]), hipSuccess);
            }
        }

        static constexpr size_t                              kWorkspace = 32 << 20;
        std::vector<int>                                     devices;
        std::vector<std::unique_ptr<hipblaslt_local_handle>> handles;
        std::vector<hipStream_t>                             streams;
        std::vector<std::unique_ptr<DeviceBuf>>              workspaceBufs;
    };

    // The full problem used by every case: fp16 in/out, fp32 compute, TN, column-major.
    constexpr int64_t kM = 256, kN = 192, kK = 128;
    constexpr int64_t kLda = kK, kLdb = kK, kLdc = kM, kLdd = kM; // op(A)=T: K x M, op(B)=N: K x N

    struct Operands
    {
        std::vector<half_t> a = randomHalf(kLda * kM, 1);
        std::vector<half_t> b = randomHalf(kLdb * kN, 2);
        std::vector<half_t> c = std::vector<half_t>(kLdc * kN, half_t(0));
    };

    // D = A * B on device `device`, with the plain Gemm. The reference for every comparison.
    std::vector<half_t> referenceGemm(Operands const& op, hipblasLtHandle_t handle, int device)
    {
        std::vector<half_t> out(kLdd * kN);
        EXPECT_EQ(hipSetDevice(device), hipSuccess);
        DeviceBuf a(device, op.a.size() * 2, op.a.data()), b(device, op.b.size() * 2, op.b.data()),
            c(device, op.c.size() * 2, op.c.data()), d(device, out.size() * 2),
            ws(device, Partitions::kWorkspace);
        float           alpha = 1.f, beta = 0.f;
        Gemm            gemm(handle,
                  HIPBLAS_OP_T,
                  HIPBLAS_OP_N,
                  HIP_R_16F,
                  HIP_R_16F,
                  HIP_R_16F,
                  HIP_R_16F,
                  HIPBLAS_COMPUTE_32F);
        GemmEpilogue    epilogue;
        GemmInputs      inputs;
        GemmProblemType type(HIPBLAS_OP_T,
                             HIPBLAS_OP_N,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIPBLAS_COMPUTE_32F);
        inputs.setA(a.get());
        inputs.setB(b.get());
        inputs.setC(c.get());
        inputs.setD(d.get());
        inputs.setAlpha(&alpha);
        inputs.setBeta(&beta);
        EXPECT_EQ(gemm.setProblem(
                      kM, kN, kK, 1, kLda, kLdb, kLdc, kLdd, 0, 0, 0, 0, epilogue, inputs, type),
                  HIPBLAS_STATUS_SUCCESS);
        GemmPreference pref;
        pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
        gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
        EXPECT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
        if(heuristic.empty())
            return {};
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
        EXPECT_EQ(gemm.initialize(heuristic[0].algo, ws.get(), true, stream),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(gemm.run(stream), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        EXPECT_EQ(hipMemcpy(out.data(), d.get(), out.size() * 2, hipMemcpyDeviceToHost),
                  hipSuccess);
        return out;
    }

    PartitionedGemm makeGemm(Partitions const& parts)
    {
        return PartitionedGemm(parts.rawHandles(),
                               HIPBLAS_OP_T,
                               HIPBLAS_OP_N,
                               HIP_R_16F,
                               HIP_R_16F,
                               HIP_R_16F,
                               HIP_R_16F,
                               HIPBLAS_COMPUTE_32F);
    }

    // Arguments with one whole copy of every matrix per partition (Replicated).
    struct ReplicatedBuffers
    {
        ReplicatedBuffers(Partitions const& parts, Operands const& op)
        {
            for(int device : parts.devices)
            {
                bufs.push_back(std::make_unique<DeviceBuf>(device, op.a.size() * 2, op.a.data()));
                a.push_back(bufs.back()->get());
                bufs.push_back(std::make_unique<DeviceBuf>(device, op.b.size() * 2, op.b.data()));
                b.push_back(bufs.back()->get());
                bufs.push_back(std::make_unique<DeviceBuf>(device, op.c.size() * 2, op.c.data()));
                c.push_back(bufs.back()->get());
                bufs.push_back(std::make_unique<DeviceBuf>(device, kLdd * kN * 2));
                d.push_back(bufs.back()->get());
            }
        }
        PartitionedGemmInputs inputs()
        {
            PartitionedGemmInputs in;
            in.a.assign(a.begin(), a.end());
            in.b.assign(b.begin(), b.end());
            in.c.assign(c.begin(), c.end());
            in.d     = d;
            in.alpha = &alpha;
            in.beta  = &beta;
            return in;
        }
        std::vector<std::unique_ptr<DeviceBuf>> bufs;
        std::vector<void*>                      a, b, c, d;
        float                                   alpha = 1.f, beta = 0.f;
    };

    hipblasStatus_t setReplicated(PartitionedGemm&             gemm,
                                  SplitDim                     dim,
                                  PartitionedGemmInputs const& in,
                                  int64_t                      m = kM,
                                  int64_t                      n = kN)
    {
        return gemm.setProblem(m,
                               n,
                               kK,
                               1,
                               kLda,
                               kLdb,
                               kLdc,
                               kLdd,
                               0,
                               0,
                               0,
                               0,
                               dim,
                               Placement::Replicated,
                               Placement::Replicated,
                               Placement::Replicated,
                               Placement::Replicated,
                               in);
    }

    // Copies every partition's slice of D back and compares it with the reference.
    void expectSlicesMatch(PartitionedGemm&           gemm,
                           Partitions const&          parts,
                           ReplicatedBuffers const&   buffers,
                           std::vector<half_t> const& ref,
                           SplitDim                   dim)
    {
        for(size_t i = 0; i < parts.devices.size(); i++)
        {
            int64_t offset = 0, size = 0;
            ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
            std::vector<half_t> got(kLdd * kN);
            ASSERT_EQ(hipSetDevice(parts.devices[i]), hipSuccess);
            ASSERT_EQ(hipMemcpy(got.data(), buffers.d[i], got.size() * 2, hipMemcpyDeviceToHost),
                      hipSuccess);
            size_t bad = 0;
            for(int64_t col = 0; col < kN; col++)
                for(int64_t row = 0; row < kM; row++)
                {
                    const int64_t split = dim == SplitDim::M ? row : col;
                    if(split < offset || split >= offset + size)
                        continue;
                    const float want = float(ref[row + col * kLdd]);
                    const float have = float(got[row + col * kLdd]);
                    if(std::fabs(want - have) > 2e-3f * std::max(1.f, std::fabs(want)))
                        bad++;
                }
            EXPECT_EQ(bad, 0u) << "partition " << i << " slice [" << offset << ", " << offset + size
                               << ")";
        }
    }

    // The partitions of a single-GPU run are extra handles on device 0.
    Partitions onDeviceZero(size_t count)
    {
        return Partitions(std::vector<int>(count, 0));
    }
} // namespace

// ---------------------------------------------------------------------------------------------
// Any single GPU
// ---------------------------------------------------------------------------------------------

TEST(pre_checkin_partitioned_gemm_ext, PartitionCountAndSliceRule)
{
    auto parts = onDeviceZero(3);
    auto gemm  = makeGemm(parts);
    EXPECT_EQ(gemm.getPartitionCount(), 3u);

    // The first N-1 partitions take size/N, the last takes the rest.
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    ASSERT_EQ(setReplicated(gemm, SplitDim::M, buffers.inputs(), /*m=*/10, kN),
              HIPBLAS_STATUS_SUCCESS);
    const int64_t wantOffset[3] = {0, 3, 6}, wantSize[3] = {3, 3, 4};
    for(size_t i = 0; i < 3; i++)
    {
        int64_t offset = -1, size = -1;
        ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(offset, wantOffset[i]);
        EXPECT_EQ(size, wantSize[i]);
    }
    int64_t offset, size;
    EXPECT_EQ(gemm.getSlice(3, offset, size), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_ext, GetSliceNeedsAProblem)
{
    auto    parts = onDeviceZero(2);
    auto    gemm  = makeGemm(parts);
    int64_t offset, size;
    EXPECT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_ext, RejectsWrongPointerCounts)
{
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    // A Replicated matrix needs N pointers.
    auto in = buffers.inputs();
    in.a.pop_back();
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);

    // A Shared matrix needs exactly one.
    in = buffers.inputs();
    EXPECT_EQ(gemm.setProblem(kM,
                              kN,
                              kK,
                              1,
                              kLda,
                              kLdb,
                              kLdc,
                              kLdd,
                              0,
                              0,
                              0,
                              0,
                              SplitDim::N,
                              Placement::Shared,
                              Placement::Replicated,
                              Placement::Replicated,
                              Placement::Replicated,
                              in),
              HIPBLAS_STATUS_INVALID_VALUE);

    // A null pointer is rejected.
    in      = buffers.inputs();
    in.d[1] = nullptr;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_ext, RejectsBadSizesAndScalars)
{
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    EXPECT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs(), kM, 0),
              HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(setReplicated(gemm, SplitDim::M, buffers.inputs(), -1, kN),
              HIPBLAS_STATUS_INVALID_VALUE);
    // The split dimension must hold at least one index per partition.
    EXPECT_EQ(setReplicated(gemm, SplitDim::M, buffers.inputs(), 1, kN),
              HIPBLAS_STATUS_INVALID_VALUE);

    auto in  = buffers.inputs();
    in.alpha = nullptr;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
    in      = buffers.inputs();
    in.beta = nullptr;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_ext, UserArgsAreNotSupportedYet)
{
    auto parts = onDeviceZero(2);
    auto gemm  = makeGemm(parts);

    // Any non-null device user-args pointer is refused. Null pointers fall through to a
    // standard run, which then fails because no problem is set.
    int                dummy = 0;
    std::vector<void*> userArgs{&dummy, &dummy};
    EXPECT_EQ(gemm.run(userArgs, parts.streams), HIPBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(gemm.getDefaultValueForDeviceUserArguments(userArgs), HIPBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(gemm.run(std::vector<void*>(2, nullptr), parts.streams),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_ext, RejectsStreamsOnTheSameDevice)
{
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    ASSERT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    if(heuristic.empty())
        GTEST_SKIP() << "no algorithm for the test problem on this device";

    // Both partitions' streams are on device 0.
    EXPECT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), true, parts.streams),
              HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_INVALID_VALUE);

    // Wrong number of streams or workspaces.
    EXPECT_EQ(gemm.run({parts.streams[0]}), HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(gemm.initialize(heuristic[0].algo, {parts.workspaces()[0]}, true, parts.streams),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// One partition is a plain GEMM: it must match Gemm exactly.
TEST(pre_checkin_partitioned_gemm_ext, OnePartitionMatchesGemm)
{
    auto              parts = onDeviceZero(1);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    ASSERT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    if(heuristic.empty())
        GTEST_SKIP() << "no algorithm for the test problem on this device";

    ASSERT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), true, parts.streams),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_SUCCESS);
    parts.sync();

    auto ref = referenceGemm(op, parts.rawHandles()[0], 0);
    ASSERT_FALSE(ref.empty());
    expectSlicesMatch(gemm, parts, buffers, ref, SplitDim::N);
}

// ---------------------------------------------------------------------------------------------
// Two or more devices
// ---------------------------------------------------------------------------------------------

namespace
{
    struct Combo
    {
        const char* name;
        SplitDim    dim;
        Placement   a, b, d;
    };

    class partitioned_gemm_ext_multi_gpu : public ::testing::TestWithParam<Combo>
    {
    };
} // namespace

TEST_P(partitioned_gemm_ext_multi_gpu, MatchesGemmOnOneDevice)
{
    int deviceCount = 0;
    ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
    if(deviceCount < 2)
        GTEST_SKIP() << "needs 2 devices, found " << deviceCount;
    const Combo cs = GetParam();

    Partitions parts({0, 1});
    auto       gemm = makeGemm(parts);
    Operands   op;

    // Matrix `shared` lives on device 0 and partition 1 reads it across devices.
    std::vector<std::unique_ptr<DeviceBuf>> bufs;
    auto                                    sharedCopy = [&](std::vector<half_t> const& host) {
        bufs.push_back(std::make_unique<DeviceBuf>(0, host.size() * 2, host.data()));
        return bufs.back()->get();
    };
    for(int device = 1; device < 2; device++)
    {
        ASSERT_EQ(hipSetDevice(device), hipSuccess);
        hipError_t e = hipDeviceEnablePeerAccess(0, 0);
        if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
            GTEST_SKIP() << "no peer access from device " << device << " to device 0";
        static_cast<void>(hipGetLastError());
    }

    ReplicatedBuffers     buffers(parts, op);
    PartitionedGemmInputs in = buffers.inputs();
    // A and B Shared: one pointer, on device 0.
    if(cs.a == Placement::Shared)
        in.a = {sharedCopy(op.a)};
    if(cs.b == Placement::Shared)
        in.b = {sharedCopy(op.b)};
    // A or B Sliced: each partition holds only its slice (contiguous columns of the K x X matrix).
    const SplitDim dim    = cs.dim;
    const int64_t  total  = dim == SplitDim::M ? kM : kN;
    int64_t        off[2] = {0, total / 2}, sz[2] = {total / 2, total - total / 2};
    auto           sliceOf = [&](std::vector<half_t> const& host, int i) {
        std::vector<half_t> s(host.begin() + off[i] * kK, host.begin() + (off[i] + sz[i]) * kK);
        bufs.push_back(std::make_unique<DeviceBuf>(i, s.size() * 2, s.data()));
        return bufs.back()->get();
    };
    if(cs.a == Placement::Sliced)
        in.a = {sliceOf(op.a, 0), sliceOf(op.a, 1)};
    if(cs.b == Placement::Sliced)
        in.b = {sliceOf(op.b, 0), sliceOf(op.b, 1)};
    // D Sliced: each partition's own buffer holds just its slice, with a tight leading dimension.
    int64_t lddUse = kLdd;
    if(cs.d == Placement::Sliced)
    {
        lddUse = dim == SplitDim::M ? std::max(sz[0], sz[1]) : kM;
        for(int i = 0; i < 2; i++)
        {
            bufs.push_back(std::make_unique<DeviceBuf>(
                i, size_t(lddUse) * (dim == SplitDim::M ? kN : sz[i]) * 2));
            in.d[i] = bufs.back()->get();
        }
    }

    ASSERT_EQ(gemm.setProblem(kM,
                              kN,
                              kK,
                              1,
                              kLda,
                              kLdb,
                              kLdc,
                              lddUse,
                              0,
                              0,
                              0,
                              0,
                              dim,
                              cs.a,
                              cs.b,
                              Placement::Replicated,
                              cs.d,
                              in),
              HIPBLAS_STATUS_SUCCESS);

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    if(heuristic.empty())
        GTEST_SKIP() << "no algorithm accepted by every partition";
    ASSERT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), true, parts.streams),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_SUCCESS);
    parts.sync();

    auto ref = referenceGemm(op, parts.rawHandles()[0], 0);
    ASSERT_FALSE(ref.empty());

    for(int i = 0; i < 2; i++)
    {
        int64_t offset = 0, size = 0;
        ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
        const int64_t       rows = cs.d == Placement::Sliced ? lddUse : kLdd;
        std::vector<half_t> got(size_t(rows)
                                * (cs.d == Placement::Sliced && dim == SplitDim::N ? size : kN));
        ASSERT_EQ(hipSetDevice(i), hipSuccess);
        ASSERT_EQ(hipMemcpy(got.data(), in.d[i], got.size() * 2, hipMemcpyDeviceToHost),
                  hipSuccess);
        size_t bad = 0;
        for(int64_t col = 0; col < (dim == SplitDim::N ? size : kN); col++)
            for(int64_t row = 0; row < (dim == SplitDim::M ? size : kM); row++)
            {
                const int64_t r    = dim == SplitDim::M ? offset + row : row;
                const int64_t c    = dim == SplitDim::N ? offset + col : col;
                const float   want = float(ref[r + c * kLdd]);
                const float   have = cs.d == Placement::Sliced ? float(got[row + col * rows])
                                                               : float(got[r + c * rows]);
                if(std::fabs(want - have) > 2e-3f * std::max(1.f, std::fabs(want)))
                    bad++;
            }
        EXPECT_EQ(bad, 0u) << "partition " << i << " slice [" << offset << ", " << offset + size
                           << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(
    partitioned_gemm_ext_multi_gpu,
    partitioned_gemm_ext_multi_gpu,
    ::testing::Values(
        Combo{"M_Replicated",
              SplitDim::M,
              Placement::Replicated,
              Placement::Replicated,
              Placement::Replicated},
        Combo{"N_Replicated",
              SplitDim::N,
              Placement::Replicated,
              Placement::Replicated,
              Placement::Replicated},
        Combo{
            "M_SharedAB", SplitDim::M, Placement::Shared, Placement::Shared, Placement::Replicated},
        Combo{
            "N_SharedAB", SplitDim::N, Placement::Shared, Placement::Shared, Placement::Replicated},
        Combo{"M_SlicedAandD",
              SplitDim::M,
              Placement::Sliced,
              Placement::Replicated,
              Placement::Sliced},
        Combo{"N_SlicedBandD",
              SplitDim::N,
              Placement::Replicated,
              Placement::Sliced,
              Placement::Sliced}),
    [](::testing::TestParamInfo<Combo> const& info) { return std::string(info.param.name); });
