// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// API-surface tests for the fused GEMM + all-to-all epilogue.
//
// This covers what the host can decide on its own: the builder's single-stage
// family rule, each attribute's accepted range, the completeness check that runs
// when the descriptor is attached to a matmul descriptor, communicator
// registration, and the shape and layout requirements checked before a solution
// is selected. No architecture carries a fused all-to-all kernel yet, so a
// well-formed request ends in HIPBLAS_STATUS_NOT_SUPPORTED; the tests assert that
// distinction, since an unusable request must be an error while missing
// capability must never present as a rejected shape.
//
// The suite names carry the "pre_checkin" token on purpose: the ctest presets in
// clients/tests/test_categories.yaml select by loose substring on a category
// token, and a plain gtest suite has none, so it would be invisible to them.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // A single-process allgather: with one rank it is exactly the memcpy the
    // design doc names as the single-process case.
    hipblasStatus_t memcpyAllgather(void*       userData,
                                    const void* sendbuf,
                                    void*       recvbuf,
                                    size_t      bytesPerRank)
    {
        (void)userData;
        memcpy(recvbuf, sendbuf, bytesPerRank);
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t failingAllgather(void*       userData,
                                     const void* sendbuf,
                                     void*       recvbuf,
                                     size_t      bytesPerRank)
    {
        (void)userData;
        (void)sendbuf;
        (void)recvbuf;
        (void)bytesPerRank;
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    // Publishes this rank's payload and leaves the second slot zeroed, which no
    // real contribution can ever look like. Stands in for any way the ranks can
    // fail to agree about the communicator.
    hipblasStatus_t silentPeerAllgather(void*       userData,
                                        const void* sendbuf,
                                        void*       recvbuf,
                                        size_t      bytesPerRank)
    {
        (void)userData;
        memcpy(recvbuf, sendbuf, bytesPerRank);
        memset(static_cast<char*>(recvbuf) + bytesPerRank, 0, bytesPerRank);
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Copies this rank's payload into both slots, so the second slot claims to be
    // rank 0 where rank 1 belongs.
    hipblasStatus_t duplicatingAllgather(void*       userData,
                                         const void* sendbuf,
                                         void*       recvbuf,
                                         size_t      bytesPerRank)
    {
        (void)userData;
        memcpy(recvbuf, sendbuf, bytesPerRank);
        memcpy(static_cast<char*>(recvbuf) + bytesPerRank, sendbuf, bytesPerRank);
        return HIPBLAS_STATUS_SUCCESS;
    }

    // A real allgather for ranks that are threads of one process: each publishes
    // into its own slot and waits for the rest before reading the whole buffer
    // back. Registration cannot return until every peer has contributed, so this
    // rendezvous is the only way more than one rank registers in one process. The
    // wait is bounded so a rank that never arrives fails the test instead of
    // hanging it.
    struct Rendezvous
    {
        std::mutex              mutex;
        std::condition_variable arrival;
        std::vector<char>       slots;
        uint32_t                arrived = 0;
        uint32_t                world   = 0;
    };

    struct RendezvousRank
    {
        Rendezvous* shared;
        uint32_t    rank;
    };

    hipblasStatus_t
        rendezvousAllgather(void* userData, const void* sendbuf, void* recvbuf, size_t bytesPerRank)
    {
        RendezvousRank& self   = *static_cast<RendezvousRank*>(userData);
        Rendezvous&     shared = *self.shared;

        std::unique_lock<std::mutex> lock(shared.mutex);
        if(shared.slots.empty())
            shared.slots.resize(bytesPerRank * shared.world);
        memcpy(shared.slots.data() + bytesPerRank * self.rank, sendbuf, bytesPerRank);

        if(++shared.arrived == shared.world)
        {
            shared.arrival.notify_all();
        }
        else if(!shared.arrival.wait_for(
                    lock, std::chrono::seconds(30), [&] { return shared.arrived == shared.world; }))
        {
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        memcpy(recvbuf, shared.slots.data(), bytesPerRank * shared.world);
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Four plausible-looking addresses. The library stores a queue's fields
    // without interpreting them, so a test only needs them to be non-null.
    hipblasLtSdmaQueue_t fakeQueue(uintptr_t base)
    {
        hipblasLtSdmaQueue_t q{};
        q.queueBuf = reinterpret_cast<void*>(base);
        q.rptr     = reinterpret_cast<void*>(base + 0x100);
        q.wptr     = reinterpret_cast<void*>(base + 0x200);
        q.doorbell = reinterpret_cast<void*>(base + 0x300);
        return q;
    }

    /************************************************************************
     * Builder: lifecycle, the single-stage family rule, and attributes.
     * None of this needs a device.
     ***********************************************************************/

    class FusedA2ABuilder_pre_checkin : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
            ASSERT_NE(fused, nullptr);
        }

        void TearDown() override
        {
            if(fused)
                EXPECT_EQ(hipblasLtFusedEpilogueDestroy(fused), HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t addA2A()
        {
            return hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX);
        }

        hipblasStatus_t setExtent(int64_t am)
        {
            return hipblasLtFusedEpilogueSetAttribute(
                fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &am, sizeof(am));
        }

        hipblasStatus_t setRecvPtrs(const std::vector<void*>& ptrs)
        {
            return hipblasLtFusedEpilogueSetAttribute(fused,
                                                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS,
                                                      ptrs.data(),
                                                      ptrs.size() * sizeof(void*));
        }

        hipblasStatus_t setQueues(const std::vector<hipblasLtSdmaQueue_t>& queues)
        {
            return hipblasLtFusedEpilogueSetAttribute(
                fused,
                HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                queues.data(),
                queues.size() * sizeof(hipblasLtSdmaQueue_t));
        }

        hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    };

    TEST(FusedA2ALifecycle_pre_checkin, CreateRejectsNullOut)
    {
        EXPECT_EQ(hipblasLtFusedEpilogueCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST(FusedA2ALifecycle_pre_checkin, DestroyAcceptsNull)
    {
        EXPECT_EQ(hipblasLtFusedEpilogueDestroy(nullptr), HIPBLAS_STATUS_SUCCESS);
    }

    TEST(FusedA2ALifecycle_pre_checkin, AddRejectsNullDescriptor)
    {
        EXPECT_EQ(hipblasLtFusedEpilogueAdd(nullptr, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, AddAcceptsAllToAll)
    {
        EXPECT_EQ(addA2A(), HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, AddRejectsUnrecognizedStage)
    {
        EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, static_cast<hipblasLtFuseableEpilogue_t>(4242)),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    // The collective family holds one stage, so a second all-to-all is rejected
    // at the call expressing the mistake rather than at launch.
    TEST_F(FusedA2ABuilder_pre_checkin, AddRejectsDuplicateAllToAll)
    {
        ASSERT_EQ(addA2A(), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(addA2A(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, SetAttributeRejectsNullArguments)
    {
        const int64_t am = 512;
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      nullptr, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &am, sizeof(am)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, nullptr, sizeof(am)),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, SetAttributeRejectsUnknownAttribute)
    {
        const int64_t value = 1;
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      static_cast<hipblasLtFusedEpilogueAttribute_t>(9999),
                      &value,
                      sizeof(value)),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, ExtentMustBePositiveAndCorrectlySized)
    {
        EXPECT_EQ(setExtent(512), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(setExtent(0), HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(setExtent(-8), HIPBLAS_STATUS_INVALID_VALUE);

        const int32_t narrow = 512;
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &narrow, sizeof(narrow)),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, CompletionModeAcceptsInKernelOnly)
    {
        hipblasLtA2ACompletionMode_t mode = HIPBLASLT_A2A_COMPLETION_IN_KERNEL;
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_COMPLETION_MODE,
                      &mode,
                      sizeof(mode)),
                  HIPBLAS_STATUS_SUCCESS);

        // Value 1 is reserved for the deferred mode, which ships with the
        // primitive a caller would wait on and not before it.
        mode = static_cast<hipblasLtA2ACompletionMode_t>(1);
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_COMPLETION_MODE,
                      &mode,
                      sizeof(mode)),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, PerRankArraysAreSizedInWholeEntries)
    {
        EXPECT_EQ(setRecvPtrs({reinterpret_cast<void*>(0x1000)}), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(setQueues({fakeQueue(0x2000)}), HIPBLAS_STATUS_SUCCESS);

        void* one = reinterpret_cast<void*>(0x1000);
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, &one, 0),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                     HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS,
                                                     &one,
                                                     sizeof(void*) + 1),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, PerRankArraysAreBoundedByTheMaximumWorld)
    {
        const std::vector<void*> tooMany(HIPBLASLT_DEVICE_COMM_MAX_WORLD + 1,
                                         reinterpret_cast<void*>(0x1000));
        EXPECT_EQ(setRecvPtrs(tooMany), HIPBLAS_STATUS_INVALID_VALUE);

        const std::vector<hipblasLtSdmaQueue_t> tooManyQueues(HIPBLASLT_DEVICE_COMM_MAX_WORLD + 1,
                                                              fakeQueue(0x2000));
        EXPECT_EQ(setQueues(tooManyQueues), HIPBLAS_STATUS_INVALID_VALUE);
    }

    /************************************************************************
     * Attach: the descriptor stops being a work in progress here, so this is
     * where a stage's required parameters must all be present.
     ***********************************************************************/

    class FusedA2AAttach_pre_checkin : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX),
                      HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(fused)
                EXPECT_EQ(hipblasLtFusedEpilogueDestroy(fused), HIPBLAS_STATUS_SUCCESS);
            if(matmulDesc)
                EXPECT_EQ(hipblasLtMatmulDescDestroy(matmulDesc), HIPBLAS_STATUS_SUCCESS);
        }

        // Fills in everything a one-rank all-to-all needs.
        void completeForOneRank()
        {
            const int64_t am = 512;
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &am, sizeof(am)),
                      HIPBLAS_STATUS_SUCCESS);
            void* recv = reinterpret_cast<void*>(0x1000);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused,
                          HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS,
                          &recv,
                          sizeof(recv)),
                      HIPBLAS_STATUS_SUCCESS);
            hipblasLtSdmaQueue_t queue = fakeQueue(0x2000);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused,
                          HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                          &queue,
                          sizeof(queue)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t attach()
        {
            return hipblasLtMatmulDescSetAttribute(
                matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused));
        }

        hipblasLtMatmulDesc_t              matmulDesc = nullptr;
        hipblasLtFusedEpilogueDescriptor_t fused      = nullptr;
    };

    TEST_F(FusedA2AAttach_pre_checkin, CompleteDescriptorAttaches)
    {
        completeForOneRank();
        EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2AAttach_pre_checkin, AttachRejectsMissingExtent)
    {
        void* recv = reinterpret_cast<void*>(0x1000);
        ASSERT_EQ(
            hipblasLtFusedEpilogueSetAttribute(
                fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, &recv, sizeof(recv)),
            HIPBLAS_STATUS_SUCCESS);
        hipblasLtSdmaQueue_t queue = fakeQueue(0x2000);
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                      &queue,
                      sizeof(queue)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2AAttach_pre_checkin, AttachRejectsMissingRecvPointers)
    {
        const int64_t am = 512;
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &am, sizeof(am)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtSdmaQueue_t queue = fakeQueue(0x2000);
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                      &queue,
                      sizeof(queue)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2AAttach_pre_checkin, AttachRejectsMissingQueues)
    {
        const int64_t am = 512;
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &am, sizeof(am)),
                  HIPBLAS_STATUS_SUCCESS);
        void* recv = reinterpret_cast<void*>(0x1000);
        ASSERT_EQ(
            hipblasLtFusedEpilogueSetAttribute(
                fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, &recv, sizeof(recv)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2AAttach_pre_checkin, AttachRejectsNullPeerEntry)
    {
        completeForOneRank();
        void* recv[2] = {reinterpret_cast<void*>(0x1000), nullptr};
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, recv, sizeof(recv)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtSdmaQueue_t queues[2] = {fakeQueue(0x2000), fakeQueue(0x3000)};
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                      queues,
                      sizeof(queues)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2AAttach_pre_checkin, AttachRejectsIncompleteQueueEntry)
    {
        completeForOneRank();
        hipblasLtSdmaQueue_t queue = fakeQueue(0x2000);
        queue.doorbell             = nullptr;
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                      &queue,
                      sizeof(queue)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2AAttach_pre_checkin, AttachedDescriptorReadsBack)
    {
        completeForOneRank();
        ASSERT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);

        hipblasLtFusedEpilogueDescriptor_t readBack = nullptr;
        size_t                             written  = 0;
        EXPECT_EQ(hipblasLtMatmulDescGetAttribute(matmulDesc,
                                                  HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                                  &readBack,
                                                  sizeof(readBack),
                                                  &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(written, sizeof(readBack));
        EXPECT_EQ(readBack, fused);
    }

    /************************************************************************
     * Communicator registration.
     ***********************************************************************/

    class FusedA2AComm_pre_checkin : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
        }

        hipblasLtHandle_t handle = nullptr;
    };

    TEST_F(FusedA2AComm_pre_checkin, RejectsInvalidRankAndWorld)
    {
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 1, 1, 1, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 0, 1, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetDeviceComm(handle,
                                         0,
                                         HIPBLASLT_DEVICE_COMM_MAX_WORLD + 1,
                                         1,
                                         memcpyAllgather,
                                         nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 0, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 1, nullptr, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST(FusedA2ACommHandle_pre_checkin, RejectsNullHandle)
    {
        EXPECT_EQ(hipblasLtSetDeviceComm(nullptr, 0, 1, 1, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_NOT_INITIALIZED);
    }

    TEST_F(FusedA2AComm_pre_checkin, RegistersOnceAndOnlyOnce)
    {
        ASSERT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 1, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        // A second call fails whether or not its arguments match, which is what
        // makes world immutable.
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 1, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 4, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2AComm_pre_checkin, PropagatesAllgatherFailure)
    {
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 1, failingAllgather, nullptr),
                  HIPBLAS_STATUS_INTERNAL_ERROR);
        // The failed attempt left nothing registered.
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 1, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2AComm_pre_checkin, RejectsRanksThatDisagree)
    {
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 2, 1, silentPeerAllgather, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2AComm_pre_checkin, RejectsMisorderedAllgather)
    {
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 2, 1, duplicatingAllgather, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2AComm_pre_checkin, MultipleChannelsRegister)
    {
        EXPECT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 4, memcpyAllgather, nullptr),
                  HIPBLAS_STATUS_SUCCESS);
    }

    // The one path a single process can drive all the way through: two ranks, two
    // handles, each finding the other's flag region among the payloads. Where
    // there are two devices the ranks take one each, so registration has to make
    // the peer's region reachable rather than merely record its address. Both
    // handles are then destroyed, which is where the three kinds of entry have to
    // be told apart: a rank frees its own region and leaves its peer's alone.
    TEST(FusedA2ACommPeers_pre_checkin, TwoRanksInOneProcessRegister)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        int deviceCount = 0;
        ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);

        constexpr uint32_t kWorld = 2;

        Rendezvous shared;
        shared.world = kWorld;

        RendezvousRank  args[kWorld] = {{&shared, 0}, {&shared, 1}};
        hipblasStatus_t status[kWorld]
            = {HIPBLAS_STATUS_NOT_INITIALIZED, HIPBLAS_STATUS_NOT_INITIALIZED};
        std::thread ranks[kWorld];

        for(uint32_t r = 0; r < kWorld; ++r)
        {
            ranks[r] = std::thread([&, r] {
                if(hipSetDevice((int)r % deviceCount) != hipSuccess)
                    return;
                hipblasLtHandle_t handle = nullptr;
                if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
                    return;
                status[r]
                    = hipblasLtSetDeviceComm(handle, r, kWorld, 1, rendezvousAllgather, &args[r]);
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
            });
        }
        for(std::thread& rank : ranks)
            rank.join();

        for(uint32_t r = 0; r < kWorld; ++r)
            EXPECT_EQ(status[r], HIPBLAS_STATUS_SUCCESS) << "rank " << r;
    }

    /************************************************************************
     * Dispatch: the shape and layout requirements, checked where both the
     * communicator and D's layout are in hand.
     ***********************************************************************/

    class FusedA2ADispatch_pre_checkin : public ::testing::Test
    {
    protected:
        static constexpr int64_t kM   = 1024;
        static constexpr int64_t kN   = 512;
        static constexpr int64_t kK   = 256;
        static constexpr int64_t kLdd = kM;

        void SetUp() override
        {
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX),
                      HIPBLAS_STATUS_SUCCESS);

            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, kM, kK, kM),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_16BF, kK, kN, kK),
                      HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            for(hipblasLtMatrixLayout_t layout : {Adesc, Bdesc, Ddesc})
                if(layout)
                    EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layout), HIPBLAS_STATUS_SUCCESS);
            if(fused)
                EXPECT_EQ(hipblasLtFusedEpilogueDestroy(fused), HIPBLAS_STATUS_SUCCESS);
            if(pref)
                EXPECT_EQ(hipblasLtMatmulPreferenceDestroy(pref), HIPBLAS_STATUS_SUCCESS);
            if(matmulDesc)
                EXPECT_EQ(hipblasLtMatmulDescDestroy(matmulDesc), HIPBLAS_STATUS_SUCCESS);
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
        }

        void registerOneRank()
        {
            ASSERT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, 1, memcpyAllgather, nullptr),
                      HIPBLAS_STATUS_SUCCESS);
        }

        // A descriptor a one-rank group can actually run, so each test can move
        // exactly one thing out of range.
        void completeAndAttach(int64_t am = 512)
        {
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &am, sizeof(am)),
                      HIPBLAS_STATUS_SUCCESS);
            void* recv = reinterpret_cast<void*>(0x1000);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused,
                          HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS,
                          &recv,
                          sizeof(recv)),
                      HIPBLAS_STATUS_SUCCESS);
            hipblasLtSdmaQueue_t queue = fakeQueue(0x2000);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused,
                          HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                          &queue,
                          sizeof(queue)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(matmulDesc,
                                                      HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                                      &fused,
                                                      sizeof(fused)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        void makeD(hipDataType type = HIP_R_16BF, int64_t m = kM, int64_t ld = kLdd)
        {
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Ddesc, type, m, kN, ld),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t heuristic()
        {
            hipblasLtMatmulHeuristicResult_t result[1] = {};
            int                              returned  = 0;
            return hipblasLtMatmulAlgoGetHeuristic(handle,
                                                   matmulDesc,
                                                   Adesc,
                                                   Bdesc,
                                                   Ddesc,
                                                   Ddesc,
                                                   pref,
                                                   1,
                                                   result,
                                                   &returned);
        }

        hipblasLtHandle_t                  handle     = nullptr;
        hipblasLtMatmulDesc_t              matmulDesc = nullptr;
        hipblasLtMatmulPreference_t        pref       = nullptr;
        hipblasLtFusedEpilogueDescriptor_t fused      = nullptr;
        hipblasLtMatrixLayout_t            Adesc      = nullptr;
        hipblasLtMatrixLayout_t            Bdesc      = nullptr;
        hipblasLtMatrixLayout_t            Ddesc      = nullptr;
    };

    // The distinction the whole error table turns on: a request that is merely
    // unserved by this release reports missing capability, not a bad shape.
    TEST_F(FusedA2ADispatch_pre_checkin, WellFormedRequestReportsMissingCapability)
    {
        registerOneRank();
        completeAndAttach();
        makeD();
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_NOT_SUPPORTED);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsMissingCommunicator)
    {
        completeAndAttach();
        makeD();
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    // The same guard runs at launch, so a caller that skipped the heuristic still
    // learns about the missing communicator rather than dispatching a plain GEMM.
    TEST_F(FusedA2ADispatch_pre_checkin, MatmulRejectsMissingCommunicator)
    {
        completeAndAttach();
        makeD();

        const float alpha = 1.f;
        const float beta  = 0.f;
        EXPECT_EQ(hipblasLtMatmul(handle,
                                  matmulDesc,
                                  &alpha,
                                  nullptr,
                                  Adesc,
                                  nullptr,
                                  Bdesc,
                                  &beta,
                                  nullptr,
                                  Ddesc,
                                  nullptr,
                                  Ddesc,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsChannelOutsideTheCommunicator)
    {
        registerOneRank();
        completeAndAttach();
        makeD();

        const uint32_t channel = 1; // the communicator was registered with one
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL, &channel, sizeof(channel)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsPerRankArraysThatDoNotMatchWorld)
    {
        registerOneRank();
        completeAndAttach();
        makeD();

        void* recv[2] = {reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x1100)};
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, recv, sizeof(recv)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtSdmaQueue_t queues[2] = {fakeQueue(0x2000), fakeQueue(0x3000)};
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                      queues,
                      sizeof(queues)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    // Each array is compared against world on its own, so one of them being right
    // does not excuse the other.
    TEST_F(FusedA2ADispatch_pre_checkin, RejectsOnePerRankArrayThatDoesNotMatchWorld)
    {
        registerOneRank();
        completeAndAttach();
        makeD();

        void* recv[2] = {reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x1100)};
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, recv, sizeof(recv)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsBatchedGemm)
    {
        registerOneRank();
        completeAndAttach();
        makeD();

        const int32_t batchCount = 2;
        ASSERT_EQ(hipblasLtMatrixLayoutSetAttribute(Ddesc,
                                                    HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                    &batchCount,
                                                    sizeof(batchCount)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsExtentExceedingD)
    {
        registerOneRank();
        completeAndAttach(kM * 2);
        makeD();
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsStridedFeatureAxis)
    {
        registerOneRank();
        completeAndAttach();
        makeD();

        const int32_t order = HIPBLASLT_ORDER_ROW;
        ASSERT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      Ddesc, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    // A D type other than BF16 is missing capability, not a bad shape: the
    // element size is a constant in the copy descriptor's field arithmetic, so
    // another type is wrong addresses rather than a slower path.
    TEST_F(FusedA2ADispatch_pre_checkin, ReportsMissingCapabilityForNonBf16D)
    {
        registerOneRank();
        completeAndAttach();
        makeD(HIP_R_32F);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_NOT_SUPPORTED);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsShardExtentThatIsNotA16ByteMultiple)
    {
        registerOneRank();
        completeAndAttach(4);
        makeD();
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsLddThatIsNotA16ByteMultiple)
    {
        registerOneRank();
        completeAndAttach();
        makeD(HIP_R_16BF, kM, kLdd + 1);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    // The copy descriptor's extent field is 14 bits wide, counted in its 16-byte
    // addressing granularity.
    TEST_F(FusedA2ADispatch_pre_checkin, RejectsShardExtentBeyondTheCopyDescriptor)
    {
        constexpr int64_t kWideM = 8 * 16384;
        registerOneRank();
        completeAndAttach(kWideM);
        makeD(HIP_R_16BF, kWideM, kWideM);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }
}
