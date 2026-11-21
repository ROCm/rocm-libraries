// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for hipblaslt_experimental::PartitionedGemm. The tests of its documented promises are
// contract tests, in experimental/contract_partitioned_gemm_gtest.cpp.
//
//  pre_checkin_partitioned_gemm_ext  run on any single GPU. The "partitions" are extra handles
//                                    on device 0, which is enough for the argument checks and
//                                    for a one-partition run.
//  partitioned_gemm_ext_*_multi_gpu  run the two partitions of a GPU in DPX mode when there is one
//                                    (two HIP devices with the same PCI domain, bus and device),
//                                    and both partitions on one device otherwise (SPX, or two
//                                    unrelated cards); the ones that only mean something across
//                                    two partitions skip without a DPX GPU.

#define ROCM_ENABLE_EXPERIMENTAL_APIS 1
#include "experimental/contract_partitioned_gemm_helpers.hpp"

using namespace partitioned_gemm_test;

// ---------------------------------------------------------------------------------------------
// Two or more devices
// ---------------------------------------------------------------------------------------------

TEST(partitioned_gemm_ext_streams_multi_gpu, RejectsAStreamOnAnotherDeviceThanItsHandle)
{
    SKIP_UNLESS_GFX1250();
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
    SKIP_UNLESS_GFX1250();
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
    SKIP_UNLESS_GFX1250();
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
    SKIP_UNLESS_GFX1250();
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
    SKIP_UNLESS_GFX1250();
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

// Two partitions, 5 batches split 2 + 3: each partition computes its batches into its own D.
TEST(partitioned_gemm_ext_batch_multi_gpu, EachPartitionComputesItsBatches)
{
    SKIP_UNLESS_GFX1250();
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
    SKIP_UNLESS_GFX1250();
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
    SKIP_UNLESS_GFX1250();
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
    SKIP_UNLESS_GFX1250();
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
