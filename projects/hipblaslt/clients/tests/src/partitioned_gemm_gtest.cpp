// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for hipblaslt_ext::PartitionedGemm.
//
//  pre_checkin_partitioned_gemm_ext  run on any single GPU. The "partitions" are extra handles
//                                    on device 0, which is enough for the argument checks and
//                                    for a one-partition run.
//  partitioned_gemm_ext_*_multi_gpu  run the two partitions of a GPU in DPX mode when there is one
//                                    (two HIP devices with the same PCI domain, bus and device),
//                                    and both partitions on one device otherwise (SPX, or two
//                                    unrelated cards); the ones that only mean something across
//                                    two partitions skip without a DPX GPU.

#include "partition_devices.hpp"
#include "utility.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
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

    // The pointers a test hands to PartitionedGemm::setProblem, one entry per partition. Kept as
    // lists so a test can swap or repeat entries; toGemmInputs turns them into one GemmInputs
    // per partition.
    struct Pointers
    {
        std::vector<const void*> a, b, c;
        std::vector<void*>       d;
        bool                     storePeerD = false;
        const void*              alpha      = nullptr;
        const void*              beta       = nullptr;
    };

    std::vector<GemmInputs> toGemmInputs(Pointers const& in)
    {
        // The lists may have the wrong length on purpose; the library reports it.
        const size_t            n = in.d.size();
        std::vector<GemmInputs> v(n);
        for(size_t i = 0; i < n; i++)
        {
            if(i < in.a.size())
                v[i].setA(in.a[i]);
            if(i < in.b.size())
                v[i].setB(in.b[i]);
            if(i < in.c.size())
                v[i].setC(in.c[i]);
            v[i].setD(in.d[i]);
            v[i].setAlpha(in.alpha);
            v[i].setBeta(in.beta);
        }
        return v;
    }

    // Sets an environment variable for the life of the object.
    class ScopedEnv
    {
    public:
        ScopedEnv(const char* name, const char* value)
            : m_name(name)
        {
            setenv(name, value, 1);
        }
        ~ScopedEnv()
        {
            unsetenv(m_name);
        }

    private:
        const char* m_name;
    };

    // Makes two different devices unreachable for PartitionedGemm::setProblem, so the paths without
    // peer access can be tested on a machine that has it.
    constexpr const char* kForceNoPeer = "HIPBLASLT_PARTITIONED_FORCE_NO_PEER";

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
        Pointers inputs()
        {
            Pointers in;
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

    hipblasStatus_t setReplicated(
        PartitionedGemm& gemm, SplitDim dim, Pointers const& in, int64_t m = kM, int64_t n = kN)
    {
        return gemm.setProblem(
            m, n, kK, 1, kLda, kLdb, kLdc, kLdd, 0, 0, 0, 0, dim, toGemmInputs(in), in.storePeerD);
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

TEST(pre_checkin_partitioned_gemm_ext, MSlicesStartOn256ByteBoundaries)
{
    // fp16: 128 elements are 256 bytes. The first slices round up and the last is smaller.
    struct Case
    {
        size_t  parts;
        int64_t m;
        int64_t wantOffset[3], wantSize[3];
    };
    const Case cases[]
        = {{2, 1000, {0, 512, 0}, {512, 488, 0}}, {3, 1000, {0, 384, 768}, {384, 384, 232}}};
    for(auto const& c : cases)
    {
        auto              parts = onDeviceZero(c.parts);
        auto              gemm  = makeGemm(parts);
        Operands          op;
        ReplicatedBuffers buffers(parts, op);
        ASSERT_EQ(setReplicated(gemm, SplitDim::M, buffers.inputs(), c.m, kN),
                  HIPBLAS_STATUS_SUCCESS);
        for(size_t i = 0; i < c.parts; i++)
        {
            int64_t offset = -1, size = -1;
            ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(offset, c.wantOffset[i]) << c.parts << " partitions, slice " << i;
            EXPECT_EQ(size, c.wantSize[i]) << c.parts << " partitions, slice " << i;
            EXPECT_EQ(offset * 2 % 256, 0);
        }
    }
}

TEST(pre_checkin_partitioned_gemm_ext, GetSliceNeedsAProblem)
{
    auto    parts = onDeviceZero(2);
    auto    gemm  = makeGemm(parts);
    int64_t offset, size;
    EXPECT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(pre_checkin_partitioned_gemm_ext, StorePeerDNeedsTwoDifferentDs)
{
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    // Two partitions with their own D: accepted. Whether a kernel exists is up to the heuristic.
    auto in       = buffers.inputs();
    in.storePeerD = true;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_SUCCESS);

    // The D list in the other order: the peer of each partition is found by device.
    std::swap(in.d[0], in.d[1]);
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_SUCCESS);

    // Overlapping Ds: one partition would overwrite the other's slice.
    in            = buffers.inputs();
    in.storePeerD = true;
    in.d[1]       = static_cast<char*>(in.d[0]) + 64;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);

    in            = buffers.inputs();
    in.storePeerD = true;
    // The same D twice: there is no peer to write to.
    in.d[1] = in.d[0];
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
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

    // A single pointer is too few: every matrix needs one per partition.
    in = buffers.inputs();
    in.a.resize(1);
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);

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

TEST(pre_checkin_partitioned_gemm_ext, TwoPartitionsOnOneDeviceMatchGemm)
{
    // SPX mode: both partitions are on the same device, so their streams share it. Speed does
    // not matter there, the result must still be right.
    for(SplitDim dim : {SplitDim::N, SplitDim::M})
    {
        auto              parts = onDeviceZero(2);
        auto              gemm  = makeGemm(parts);
        Operands          op;
        ReplicatedBuffers buffers(parts, op);
        ASSERT_EQ(setReplicated(gemm, dim, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);

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
        expectSlicesMatch(gemm, parts, buffers, ref, dim);
    }
}

TEST(pre_checkin_partitioned_gemm_ext, RejectsWrongStreamAndWorkspaceCounts)
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

    EXPECT_EQ(gemm.run({parts.streams[0]}), HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(gemm.initialize(heuristic[0].algo, {parts.workspaces()[0]}, true, parts.streams),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(partitioned_gemm_ext_streams_multi_gpu, RejectsAStreamOnAnotherDeviceThanItsHandle)
{
    const auto pair = hipblaslt_partitions::dpxPair();
    if(!pair)
        GTEST_SKIP() << "needs a GPU in DPX mode (two HIP devices that are its two partitions)";

    Partitions        parts({pair->first, pair->second});
    auto              gemm = makeGemm(parts);
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

    // Stream i belongs to handle i's device; swapped, a kernel of one device would be launched
    // on the other device's stream.
    std::vector<hipStream_t> swapped = {parts.streams[1], parts.streams[0]};
    EXPECT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), true, swapped),
              HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(gemm.run(swapped), HIPBLAS_STATUS_INVALID_VALUE);
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

// All partitions on one device (SPX) never need peer access, whatever the machine offers.
TEST(pre_checkin_partitioned_gemm_ext, NoPeerAccessIsNeededOnOneDevice)
{
    ScopedEnv         noPeer(kForceNoPeer, "1");
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    auto in       = buffers.inputs();
    in.storePeerD = true;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_SUCCESS);
}

// ---------------------------------------------------------------------------------------------
// Two or more devices
// ---------------------------------------------------------------------------------------------

namespace
{
    // How a test lays out one matrix: one copy on device 0 that every partition reads (the pointer
    // list repeats it), or a copy per partition.
    enum class Placement
    {
        Shared,
        Replicated,
    };

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
    // The two partitions of a DPX GPU, or both partitions on one device (SPX).
    const auto  devs = hipblaslt_partitions::partitionDevices(2);
    const Combo cs   = GetParam();

    Partitions parts(devs);
    auto       gemm = makeGemm(parts);
    Operands   op;

    // Matrix `shared` lives on the first partition's device and partition 1 reads it from there.
    std::vector<std::unique_ptr<DeviceBuf>> bufs;
    auto                                    sharedCopy = [&](std::vector<half_t> const& host) {
        bufs.push_back(std::make_unique<DeviceBuf>(devs[0], host.size() * 2, host.data()));
        return bufs.back()->get();
    };
    if(devs[1] != devs[0])
    {
        ASSERT_EQ(hipSetDevice(devs[1]), hipSuccess);
        hipError_t e = hipDeviceEnablePeerAccess(devs[0], 0);
        if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
            GTEST_SKIP() << "no peer access from device " << devs[1] << " to device " << devs[0];
        static_cast<void>(hipGetLastError());
    }

    ReplicatedBuffers     buffers(parts, op);
    Pointers              in = buffers.inputs();
    // A and B Shared: one pointer, on device 0.
    if(cs.a == Placement::Shared)
    {
        const void* a = sharedCopy(op.a);
        in.a          = {a, a};
    }
    if(cs.b == Placement::Shared)
    {
        const void* b = sharedCopy(op.b);
        in.b          = {b, b};
    }
    if(cs.d == Placement::Shared)
    {
        bufs.push_back(std::make_unique<DeviceBuf>(devs[0], size_t(kLdd) * kN * 2));
        void* d = bufs.back()->get();
        in.d    = {d, d};
    }
    const SplitDim dim = cs.dim;
    int64_t        lddUse = kLdd;

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
                              toGemmInputs(in),
                              in.storePeerD),
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

    auto ref = referenceGemm(op, parts.rawHandles()[0], devs[0]);
    ASSERT_FALSE(ref.empty());

    for(int i = 0; i < 2; i++)
    {
        int64_t offset = 0, size = 0;
        ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
        const int64_t       rows = kLdd;
        std::vector<half_t> got(size_t(rows) * kN);
        ASSERT_EQ(hipSetDevice(devs[i]), hipSuccess);
        ASSERT_EQ(hipMemcpy(got.data(), in.d[i], got.size() * 2, hipMemcpyDeviceToHost),
                  hipSuccess);
        size_t bad = 0;
        for(int64_t col = 0; col < (dim == SplitDim::N ? size : kN); col++)
            for(int64_t row = 0; row < (dim == SplitDim::M ? size : kM); row++)
            {
                const int64_t r    = dim == SplitDim::M ? offset + row : row;
                const int64_t c    = dim == SplitDim::N ? offset + col : col;
                const float   want = float(ref[r + c * kLdd]);
                const float   have = float(got[r + c * rows]);
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
        Combo{"M_SharedABD", SplitDim::M, Placement::Shared, Placement::Shared, Placement::Shared},
        Combo{"N_SharedABD", SplitDim::N, Placement::Shared, Placement::Shared, Placement::Shared}),
    [](::testing::TestParamInfo<Combo> const& info) { return std::string(info.param.name); });

namespace
{
    // One VMM range whose two halves are on two devices (half h on owners[h]), readable and
    // writable from both. Skips the test when VMM is not available or the halves are not a
    // multiple of the allocation granularity.
    struct SplitVmm
    {
        void*                           va         = nullptr;
        size_t                          half       = 0;
        hipMemGenericAllocationHandle_t handles[2] = {};
        int                             mapped     = 0;

        // Returns false (and a reason) when VMM cannot be used here.
        bool create(size_t halfBytes, int const owners[2], std::string& why)
        {
            half                      = halfBytes;
            hipMemAllocationProp prop = {};
            prop.type                 = hipMemAllocationTypePinned;
            prop.location.type        = hipMemLocationTypeDevice;
            for(int h = 0; h < 2; h++)
            {
                int vmm = 0;
                if(hipDeviceGetAttribute(
                       &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, owners[h])
                       != hipSuccess
                   || !vmm)
                {
                    why = "no virtual memory management on device " + std::to_string(owners[h]);
                    return false;
                }
                prop.location.id   = owners[h];
                size_t granularity = 0;
                if(hipMemGetAllocationGranularity(
                       &granularity, &prop, hipMemAllocationGranularityRecommended)
                       != hipSuccess
                   || granularity == 0 || half % granularity != 0)
                {
                    why = "half size is not a multiple of the allocation granularity";
                    return false;
                }
            }
            EXPECT_EQ(hipMemAddressReserve(&va, 2 * half, 0, nullptr, 0), hipSuccess);
            for(int h = 0; h < 2; h++)
            {
                // The owner must be the current device, or the pages land on the current one.
                EXPECT_EQ(hipSetDevice(owners[h]), hipSuccess);
                prop.location.id = owners[h];
                EXPECT_EQ(hipMemCreate(&handles[h], half, &prop, 0), hipSuccess);
                EXPECT_EQ(hipMemMap(static_cast<char*>(va) + h * half, half, 0, handles[h], 0),
                          hipSuccess);
                mapped++;
            }
            hipMemAccessDesc access[2] = {};
            for(int d = 0; d < 2; d++)
            {
                access[d].location.type = hipMemLocationTypeDevice;
                access[d].location.id   = d;
                access[d].flags         = hipMemAccessFlagsProtReadWrite;
            }
            EXPECT_EQ(hipMemSetAccess(va, 2 * half, access, 2), hipSuccess);
            return true;
        }

        ~SplitVmm()
        {
            for(int h = 0; h < mapped; h++)
            {
                static_cast<void>(hipMemUnmap(static_cast<char*>(va) + h * half, half));
                static_cast<void>(hipMemRelease(handles[h]));
            }
            if(va != nullptr)
                static_cast<void>(hipMemAddressFree(va, 2 * half));
        }
    };

    // GetParam(): the halves of the shared B are on the devices in reverse order.
    class partitioned_gemm_ext_placement_multi_gpu : public ::testing::TestWithParam<bool>
    {
    };
}

// B is shared and split along N, and each half of it is on one partition's device. Whichever
// order the halves are in, each partition must compute the slice on its own device.
TEST_P(partitioned_gemm_ext_placement_multi_gpu, SlicesFollowThePages)
{
    const auto pair = hipblaslt_partitions::dpxPair();
    if(!pair)
        GTEST_SKIP() << "needs a GPU in DPX mode (two HIP devices that are its two partitions)";
    const std::vector<int> dpxDevices = {pair->first, pair->second};
    const bool swapped = GetParam();

    // op(A)=T: A is K x M; op(B)=N: B is K x N, so a half of B (N/2 columns) is 2 MiB.
    constexpr int64_t m = 256, n = 2048, k = 1024;
    const int         owners[2] = {dpxDevices[swapped ? 1 : 0], dpxDevices[swapped ? 0 : 1]};
    SplitVmm          b;
    std::string       why;
    if(!b.create(size_t(k) * (n / 2) * sizeof(half_t), owners, why))
        GTEST_SKIP() << why;

    // B column j holds (j % 8) / 8 and A holds ones, so D(i, j) = k * (j % 8) / 8 exactly.
    std::vector<half_t> hb(size_t(k) * n);
    for(int64_t j = 0; j < n; j++)
        for(int64_t r = 0; r < k; r++)
            hb[j * k + r] = half_t(float(j % 8) / 8.f);
    ASSERT_EQ(hipMemcpy(b.va, hb.data(), hb.size() * sizeof(half_t), hipMemcpyHostToDevice),
              hipSuccess);
    const std::vector<half_t> ha(size_t(k) * m, half_t(1));

    Partitions parts(dpxDevices);
    auto       gemm = makeGemm(parts);
    // A and D: one copy per partition, on its device.
    std::vector<std::unique_ptr<DeviceBuf>> a, d;
    for(int dev : dpxDevices)
    {
        a.push_back(std::make_unique<DeviceBuf>(dev, ha.size() * sizeof(half_t), ha.data()));
        d.push_back(std::make_unique<DeviceBuf>(dev, size_t(m) * n * sizeof(half_t)));
    }
    const float           alpha = 1.f, beta = 0.f;
    Pointers              in;
    in.a     = {a[0]->get(), a[1]->get()};
    in.b     = {b.va, b.va};
    in.c     = {d[0]->get(), d[1]->get()};
    in.d     = {d[0]->get(), d[1]->get()};
    in.alpha = &alpha;
    in.beta  = &beta;
    ASSERT_EQ(gemm.setProblem(
                  m, n, k, 1, k, k, m, m, 0, 0, 0, 0, SplitDim::N, toGemmInputs(in), in.storePeerD),
              HIPBLAS_STATUS_SUCCESS);

    // Partition i computes the half of B that is on device i.
    for(size_t i = 0; i < 2; i++)
    {
        int64_t offset = -1, size = 0;
        ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
        const int64_t expected = (owners[0] == dpxDevices[i] ? 0 : 1) * (n / 2);
        EXPECT_EQ(offset, expected) << "partition " << i;
        EXPECT_EQ(size, n / 2);
    }

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    if(heuristic.empty())
        GTEST_SKIP() << "no algorithm accepted by every partition";
    ASSERT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), false, parts.streams),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_SUCCESS);
    parts.sync();

    // Each partition's D holds its slice, at the slice's columns.
    for(size_t i = 0; i < 2; i++)
    {
        int64_t offset = 0, size = 0;
        ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
        std::vector<half_t> got(size_t(m) * n);
        ASSERT_EQ(hipSetDevice(dpxDevices[i]), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(got.data(), d[i]->get(), got.size() * sizeof(half_t), hipMemcpyDeviceToHost),
            hipSuccess);
        size_t bad = 0;
        for(int64_t col = offset; col < offset + size; col++)
            for(int64_t row = 0; row < m; row++)
                bad += float(got[col * m + row]) != float(k * (col % 8)) / 8.f;
        EXPECT_EQ(bad, 0u) << "partition " << i << " columns [" << offset << ", " << offset + size
                           << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(partitioned_gemm_ext_placement_multi_gpu,
                         partitioned_gemm_ext_placement_multi_gpu,
                         ::testing::Values(false, true),
                         [](::testing::TestParamInfo<bool> const& info) {
                             return std::string(info.param ? "HalvesSwapped" : "HalvesInOrder");
                         });

// Copies are matched to the partitions by device. A list in the other order for every matrix is
// swapped back; lists that disagree are an error.
TEST(partitioned_gemm_ext_copies_multi_gpu, SwappedCopiesAreMatchedByDevice)
{
    const auto pair = hipblaslt_partitions::dpxPair();
    if(!pair)
        GTEST_SKIP() << "needs a GPU in DPX mode (two HIP devices that are its two partitions)";
    const std::vector<int> dpxDevices = {pair->first, pair->second};
    Partitions             parts(dpxDevices);
    auto              gemm = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    // Every list in reverse order: partition i must still use the copy on device i.
    auto in = buffers.inputs();
    std::swap(in.a[0], in.a[1]);
    std::swap(in.b[0], in.b[1]);
    std::swap(in.c[0], in.c[1]);
    std::swap(in.d[0], in.d[1]);
    ASSERT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_SUCCESS);

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    if(heuristic.empty())
        GTEST_SKIP() << "no algorithm accepted by every partition";
    ASSERT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), false, parts.streams),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_SUCCESS);
    parts.sync();

    // The D on device i (buffers.d[i]) holds the slice of partition i.
    auto ref = referenceGemm(op, parts.rawHandles()[0], 0);
    ASSERT_FALSE(ref.empty());
    expectSlicesMatch(gemm, parts, buffers, ref, SplitDim::N);
}

TEST(partitioned_gemm_ext_copies_multi_gpu, CopiesInDifferentOrdersAreRejected)
{
    const auto pair = hipblaslt_partitions::dpxPair();
    if(!pair)
        GTEST_SKIP() << "needs a GPU in DPX mode (two HIP devices that are its two partitions)";
    const std::vector<int> dpxDevices = {pair->first, pair->second};
    Partitions             parts(dpxDevices);
    auto              gemm = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    // A in reverse order, B in order: no assignment keeps both local.
    auto in = buffers.inputs();
    std::swap(in.a[0], in.a[1]);
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);

    // Both copies of B on device 0: partition 1 has no copy of its own.
    in      = buffers.inputs();
    in.b[1] = in.b[0];
    in.b[0] = buffers.c[0]; // another buffer on device 0, so the list is not "shared"
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);

    // The plain order still works.
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);
}

// Batch split: whole matrices, the first partitions take batch_count / N batches, the last the rest.
TEST(pre_checkin_partitioned_gemm_ext, BatchSplitSlices)
{
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    auto              in = buffers.inputs();

    const int64_t sa = kLda * kM, sb = kLdb * kN, sc = kLdc * kN, sd = kLdd * kN;
    ASSERT_EQ(gemm.setProblem(kM,
                              kN,
                              kK,
                              5,
                              kLda,
                              kLdb,
                              kLdc,
                              kLdd,
                              sa,
                              sb,
                              sc,
                              sd,
                              SplitDim::Batch,
                              toGemmInputs(in),
                              in.storePeerD),
              HIPBLAS_STATUS_SUCCESS);
    int64_t offset = -1, size = -1;
    ASSERT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(offset, 0);
    EXPECT_EQ(size, 2);
    ASSERT_EQ(gemm.getSlice(1, offset, size), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(offset, 2);
    EXPECT_EQ(size, 3);

    // One batch cannot be split between two partitions.
    EXPECT_EQ(gemm.setProblem(kM,
                              kN,
                              kK,
                              1,
                              kLda,
                              kLdb,
                              kLdc,
                              kLdd,
                              sa,
                              sb,
                              sc,
                              sd,
                              SplitDim::Batch,
                              toGemmInputs(in),
                              in.storePeerD),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// Two partitions, 5 batches split 2 + 3: each partition computes its batches into its own D.
TEST(partitioned_gemm_ext_batch_multi_gpu, EachPartitionComputesItsBatches)
{
    // The two partitions of a DPX GPU, or both partitions on one device (SPX).
    const auto devs = hipblaslt_partitions::partitionDevices(2);

    // op(A)=T: A is K x M; op(B)=N: B is K x N. A holds ones and column j of batch b of B holds
    // ((j + 3b) % 8) / 8, so D(i, j) of batch b is k * ((j + 3b) % 8) / 8 exactly.
    constexpr int64_t         m = 64, n = 48, k = 32, batch = 5;
    constexpr int64_t         sa = k * m, sb = k * n, sd = m * n;
    const std::vector<half_t> ha(size_t(sa) * batch, half_t(1));
    std::vector<half_t>       hb(size_t(sb) * batch);
    for(int64_t b = 0; b < batch; b++)
        for(int64_t j = 0; j < n; j++)
            for(int64_t r = 0; r < k; r++)
                hb[b * sb + j * k + r] = half_t(float((j + 3 * b) % 8) / 8.f);

    Partitions                              parts(devs);
    auto                                    gemm = makeGemm(parts);
    std::vector<std::unique_ptr<DeviceBuf>> a, b, d;
    for(int dev : devs)
    {
        a.push_back(std::make_unique<DeviceBuf>(dev, ha.size() * 2, ha.data()));
        b.push_back(std::make_unique<DeviceBuf>(dev, hb.size() * 2, hb.data()));
        d.push_back(std::make_unique<DeviceBuf>(dev, size_t(sd) * batch * 2));
    }
    const float           alpha = 1.f, beta = 0.f;
    Pointers              in;
    in.a     = {a[0]->get(), a[1]->get()};
    in.b     = {b[0]->get(), b[1]->get()};
    in.c     = {d[0]->get(), d[1]->get()};
    in.d     = {d[0]->get(), d[1]->get()};
    in.alpha = &alpha;
    in.beta  = &beta;
    ASSERT_EQ(gemm.setProblem(m,
                              n,
                              k,
                              batch,
                              k,
                              k,
                              m,
                              m,
                              sa,
                              sb,
                              sd,
                              sd,
                              SplitDim::Batch,
                              toGemmInputs(in),
                              in.storePeerD),
              HIPBLAS_STATUS_SUCCESS);

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
    gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    ASSERT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
    if(heuristic.empty())
        GTEST_SKIP() << "no algorithm accepted by every partition";
    ASSERT_EQ(gemm.initialize(heuristic[0].algo, parts.workspaces(), false, parts.streams),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(gemm.run(parts.streams), HIPBLAS_STATUS_SUCCESS);
    parts.sync();

    int64_t covered = 0;
    for(size_t i = 0; i < 2; i++)
    {
        int64_t offset = 0, size = 0;
        ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
        covered += size;
        std::vector<half_t> got(size_t(sd) * batch);
        ASSERT_EQ(hipSetDevice(devs[i]), hipSuccess);
        ASSERT_EQ(hipMemcpy(got.data(), d[i]->get(), got.size() * 2, hipMemcpyDeviceToHost),
                  hipSuccess);
        size_t bad = 0;
        for(int64_t bb = offset; bb < offset + size; bb++)
            for(int64_t j = 0; j < n; j++)
                for(int64_t r = 0; r < m; r++)
                    bad += float(got[bb * sd + j * m + r]) != float(k * ((j + 3 * bb) % 8)) / 8.f;
        EXPECT_EQ(bad, 0u) << "partition " << i << " batches [" << offset << ", " << offset + size
                           << ")";
    }
    EXPECT_EQ(covered, batch);
}

// A batch split gives different batches of D to different partitions, so D needs a batch stride.
TEST(pre_checkin_partitioned_gemm_ext, BatchSplitNeedsABatchStrideOfD)
{
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    auto              in = buffers.inputs();

    EXPECT_EQ(gemm.setProblem(kM,
                              kN,
                              kK,
                              4,
                              kLda,
                              kLdb,
                              kLdc,
                              kLdd,
                              kLda * kM,
                              kLdb * kN,
                              kLdc * kN,
                              0,
                              SplitDim::Batch,
                              toGemmInputs(in),
                              in.storePeerD),
              HIPBLAS_STATUS_INVALID_VALUE);
    // The same sizes with a stride of D are fine, and a stride of 0 is fine for one batch.
    EXPECT_EQ(gemm.setProblem(kM,
                              kN,
                              kK,
                              4,
                              kLda,
                              kLdb,
                              kLdc,
                              kLdd,
                              kLda * kM,
                              kLdb * kN,
                              kLdc * kN,
                              kLdd * kN,
                              SplitDim::Batch,
                              toGemmInputs(in),
                              in.storePeerD),
              HIPBLAS_STATUS_SUCCESS);
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
                              toGemmInputs(in),
                              in.storePeerD),
              HIPBLAS_STATUS_SUCCESS);
}

// A setProblem that fails must not leave the previous problem usable.
TEST(pre_checkin_partitioned_gemm_ext, FailedSetProblemClearsTheProblem)
{
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);
    auto              in = buffers.inputs();

    ASSERT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_SUCCESS);
    int64_t offset = 0, size = 0;
    ASSERT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_SUCCESS);

    in.a.pop_back(); // one pointer too few
    ASSERT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_INVALID_VALUE);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    GemmPreference                                pref;
    EXPECT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_INVALID_VALUE);
}

// A moved-from PartitionedGemm and one built with a null handle are unusable, not crashes.
TEST(pre_checkin_partitioned_gemm_ext, MovedFromAndNullHandlesAreRejected)
{
    auto              parts = onDeviceZero(2);
    auto              gemm  = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    PartitionedGemm moved(std::move(gemm));
    EXPECT_EQ(gemm.getPartitionCount(), 0u);
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_INVALID_VALUE);
    int64_t offset = 0, size = 0;
    EXPECT_EQ(gemm.getSlice(0, offset, size), HIPBLAS_STATUS_INVALID_VALUE);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    GemmPreference                                pref;
    EXPECT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_INVALID_VALUE);
    gemm.setMaxWorkspaceBytes(1);
    EXPECT_EQ(setReplicated(moved, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);

    auto handles = parts.rawHandles();
    handles[1]   = nullptr;
    PartitionedGemm withNull(handles,
                             HIPBLAS_OP_T,
                             HIPBLAS_OP_N,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIPBLAS_COMPUTE_32F);
    EXPECT_EQ(withNull.getPartitionCount(), 0u);
    EXPECT_EQ(setReplicated(withNull, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_INVALID_VALUE);
}

// A partition cannot use memory on another device that it has no peer access to; the kernel would
// fault. setProblem refuses it with an error. Copies on the partition's own device, which need no
// access, stay fine.
TEST(partitioned_gemm_ext_streams_multi_gpu, MemoryOnAnotherDeviceNeedsPeerAccess)
{
    const auto pair = hipblaslt_partitions::dpxPair();
    if(!pair)
        GTEST_SKIP() << "needs a GPU in DPX mode (two HIP devices that are its two partitions)";
    Partitions        parts({pair->first, pair->second});
    auto              gemm = makeGemm(parts);
    Operands          op;
    ReplicatedBuffers buffers(parts, op);

    // One copy of A on the first device, read by both partitions.
    DeviceBuf sharedA(pair->first, op.a.size() * 2, op.a.data());
    auto      shared = buffers.inputs();
    shared.a         = {sharedA.get(), sharedA.get()};
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, shared), HIPBLAS_STATUS_SUCCESS);
    {
        ScopedEnv noPeer(kForceNoPeer, "1");
        EXPECT_EQ(setReplicated(gemm, SplitDim::N, shared), HIPBLAS_STATUS_INVALID_VALUE);

        // A copy on each partition's device needs no access.
        EXPECT_EQ(setReplicated(gemm, SplitDim::N, buffers.inputs()), HIPBLAS_STATUS_SUCCESS);

        // storePeerD writes the other partition's D: it needs access both ways.
        auto in       = buffers.inputs();
        in.storePeerD = true;
        EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_INVALID_VALUE);
    }
    auto in       = buffers.inputs();
    in.storePeerD = true;
    EXPECT_EQ(setReplicated(gemm, SplitDim::N, in), HIPBLAS_STATUS_SUCCESS);
}
