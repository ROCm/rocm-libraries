// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// API-surface tests for the fused GEMM + all-to-all epilogue: the builder's
// single-stage family rule, attribute ranges, the completeness check run at
// descriptor attach, communicator registration, and the shape and layout
// requirements checked before solution selection.
//
// Suite names must keep their "pre_checkin" or "multi_gpu" token. The ctest
// presets in clients/tests/test_categories.yaml select by loose substring on
// that token, so a suite without one is invisible to them.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // With one rank an allgather is exactly a memcpy.
    hipblasStatus_t
        memcpyAllgather(void* /*userData*/, const void* sendbuf, void* recvbuf, size_t bytesPerRank)
    {
        memcpy(recvbuf, sendbuf, bytesPerRank);
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t failingAllgather(void* /*userData*/,
                                     const void* /*sendbuf*/,
                                     void* /*recvbuf*/,
                                     size_t /*bytesPerRank*/)
    {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    // Leaves the second slot zeroed, which no real contribution can look like.
    hipblasStatus_t silentPeerAllgather(void* /*userData*/,
                                        const void* sendbuf,
                                        void*       recvbuf,
                                        size_t      bytesPerRank)
    {
        memcpy(recvbuf, sendbuf, bytesPerRank);
        memset(static_cast<char*>(recvbuf) + bytesPerRank, 0, bytesPerRank);
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Both slots claim to be rank 0, so rank 1's contribution is missing.
    hipblasStatus_t duplicatingAllgather(void* /*userData*/,
                                         const void* sendbuf,
                                         void*       recvbuf,
                                         size_t      bytesPerRank)
    {
        memcpy(recvbuf, sendbuf, bytesPerRank);
        memcpy(static_cast<char*>(recvbuf) + bytesPerRank, sendbuf, bytesPerRank);
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Registration cannot return until every peer has contributed, so ranks that
    // are threads of one process need this rendezvous to register at all. The
    // wait is bounded so a rank that never arrives fails the test, not hangs it.
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

        hipblasStatus_t addPrologue()
        {
            return hipblasLtFusedPrologueAdd(fused, HIPBLASLT_FUSEABLE_PROLOGUE_A2A_PREFIX);
        }

        hipblasStatus_t setPeerBPtrs(const std::vector<void*>& ptrs)
        {
            return hipblasLtFusedEpilogueSetAttribute(
                fused,
                HIPBLASLT_FUSED_PROLOGUE_A2A_PREFIX_PEER_B_PTRS,
                ptrs.data(),
                ptrs.size() * sizeof(void*));
        }

        hipblasStatus_t setGatherBuffer(uintptr_t address)
        {
            void* buffer = reinterpret_cast<void*>(address);
            return hipblasLtFusedEpilogueSetAttribute(
                fused, HIPBLASLT_FUSED_PROLOGUE_A2A_PREFIX_GATHER_BUFFER, &buffer, sizeof(buffer));
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
        hipblasLtA2ACompletionMode_t mode = HIPBLASLT_A2A_COMPLETION_IN_KERNEL_FULL;
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

    TEST(FusedA2ALifecycle_pre_checkin, PrologueAddRejectsNullDescriptor)
    {
        EXPECT_EQ(hipblasLtFusedPrologueAdd(nullptr, HIPBLASLT_FUSEABLE_PROLOGUE_A2A_PREFIX),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, PrologueAddAcceptsAllToAll)
    {
        EXPECT_EQ(addPrologue(), HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, PrologueAddRejectsUnrecognizedStage)
    {
        EXPECT_EQ(hipblasLtFusedPrologueAdd(fused, static_cast<hipblasLtFuseablePrologue_t>(4242)),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, PrologueAddRejectsDuplicateAllToAll)
    {
        ASSERT_EQ(addPrologue(), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(addPrologue(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, PrologueDoesNotJoinAnEpilogue)
    {
        ASSERT_EQ(addA2A(), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(addPrologue(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, EpilogueDoesNotJoinAPrologue)
    {
        ASSERT_EQ(addPrologue(), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(addA2A(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, PeerBPtrsAreSizedAndBoundedLikeTheOtherPerRankArrays)
    {
        EXPECT_EQ(setPeerBPtrs({reinterpret_cast<void*>(0x1000)}), HIPBLAS_STATUS_SUCCESS);

        void* one = reinterpret_cast<void*>(0x1000);
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_PROLOGUE_A2A_PREFIX_PEER_B_PTRS, &one, 0),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(
            hipblasLtFusedEpilogueSetAttribute(
                fused, HIPBLASLT_FUSED_PROLOGUE_A2A_PREFIX_PEER_B_PTRS, &one, sizeof(void*) + 1),
            HIPBLAS_STATUS_INVALID_VALUE);

        const std::vector<void*> tooMany(HIPBLASLT_DEVICE_COMM_MAX_WORLD + 1,
                                         reinterpret_cast<void*>(0x1000));
        EXPECT_EQ(setPeerBPtrs(tooMany), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ABuilder_pre_checkin, GatherBufferMustBe128ByteAligned)
    {
        EXPECT_EQ(setGatherBuffer(0x1000), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(setGatherBuffer(0x1080), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(setGatherBuffer(0x1040), HIPBLAS_STATUS_INVALID_VALUE);

        void* buffer = reinterpret_cast<void*>(0x1000);
        EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_PROLOGUE_A2A_PREFIX_GATHER_BUFFER, &buffer, 4),
                  HIPBLAS_STATUS_INVALID_VALUE);
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

    TEST(FusedA2APrologueAttach_pre_checkin, AttachedPrologueReadsBackAndDetaches)
    {
        hipblasLtMatmulDesc_t              matmulDesc = nullptr;
        hipblasLtFusedEpilogueDescriptor_t prologue   = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueCreate(&prologue), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedPrologueAdd(prologue, HIPBLASLT_FUSEABLE_PROLOGUE_A2A_PREFIX),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(
            hipblasLtMatmulDescSetAttribute(
                matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, &prologue, sizeof(prologue)),
            HIPBLAS_STATUS_SUCCESS);

        hipblasLtFusedEpilogueDescriptor_t readBack = nullptr;
        size_t                             written  = 0;
        EXPECT_EQ(hipblasLtMatmulDescGetAttribute(matmulDesc,
                                                  HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE,
                                                  &readBack,
                                                  sizeof(readBack),
                                                  &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(written, sizeof(readBack));
        EXPECT_EQ(readBack, prologue);

        hipblasLtFusedEpilogueDescriptor_t none = nullptr;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, &none, sizeof(none)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatmulDescGetAttribute(matmulDesc,
                                                  HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE,
                                                  &readBack,
                                                  sizeof(readBack),
                                                  &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(readBack, nullptr);

        EXPECT_EQ(hipblasLtFusedEpilogueDestroy(prologue), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatmulDescDestroy(matmulDesc), HIPBLAS_STATUS_SUCCESS);
    }

    TEST(FusedA2APrologueAttach_pre_checkin, PrologueAttachRejectsAShortBuffer)
    {
        hipblasLtMatmulDesc_t matmulDesc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtFusedEpilogueDescriptor_t prologue = nullptr;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, &prologue, 4),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtMatmulDescDestroy(matmulDesc), HIPBLAS_STATUS_SUCCESS);
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

    // Two ranks rendezvous, each finding the other's flag region among the
    // payloads; destroying both then frees a rank's own region and leaves its
    // peer's. One shared device keeps this on the branch that needs no peer
    // access - the cross-device one is the multi_gpu test below, split out
    // because `rank % deviceCount` let a single-GPU host pass it unexecuted.
    TEST(FusedA2ACommPeers_pre_checkin, TwoRanksOneDeviceRegister)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

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
                if(hipSetDevice(0) != hipSuccess)
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

    // The cross-device branch: rank r takes device r, so a peer's flag region
    // must be reachable from another device - registration has to enable peer
    // access, not just record the address. Nothing public reads those addresses
    // back, so each rank disables access first and then asserts on the side
    // effect: a request reporting already-enabled can only be registration's
    // work, where pre-fix it reported plain success.
    TEST(FusedA2ACommPeers_multi_gpu, TwoRanksTwoDevicesEnablePeerAccess)
    {
        int deviceCount = 0;
        ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
        if(deviceCount < 2)
            GTEST_SKIP() << "cross-device registration needs 2 devices, found " << deviceCount;

        constexpr uint32_t kWorld = 2;

        Rendezvous shared;
        shared.world = kWorld;

        RendezvousRank  args[kWorld] = {{&shared, 0}, {&shared, 1}};
        hipblasStatus_t status[kWorld]
            = {HIPBLAS_STATUS_NOT_INITIALIZED, HIPBLAS_STATUS_NOT_INITIALIZED};
        // Seeded with a code neither call returns, so an entry no thread reached
        // cannot read as a pass.
        hipError_t  reEnable[kWorld] = {hipErrorUnknown, hipErrorUnknown};
        std::thread ranks[kWorld];

        for(uint32_t r = 0; r < kWorld; ++r)
        {
            ranks[r] = std::thread([&, r] {
                const int peerDevice = (int)(kWorld - 1 - r);
                if(hipSetDevice((int)r) != hipSuccess)
                    return;
                // Whether this returns success or hipErrorPeerAccessNotEnabled,
                // access from this device to the peer's is disabled afterwards,
                // which is all the assertion below needs.
                (void)hipDeviceDisablePeerAccess(peerDevice);

                hipblasLtHandle_t handle = nullptr;
                if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
                    return;
                status[r]
                    = hipblasLtSetDeviceComm(handle, r, kWorld, 1, rendezvousAllgather, &args[r]);
                // Asked while this rank's device is still current and before the
                // handle is destroyed: the direction under test is from device r
                // to the peer's, and that is the one registration enabled.
                if(status[r] == HIPBLAS_STATUS_SUCCESS)
                    reEnable[r] = hipDeviceEnablePeerAccess(peerDevice, 0);
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
            });
        }
        for(std::thread& rank : ranks)
            rank.join();

        // A host whose two devices cannot reach each other is a topology this
        // test has nothing to say about, and NOT_SUPPORTED is registration
        // answering correctly rather than failing.
        for(uint32_t r = 0; r < kWorld; ++r)
            if(status[r] == HIPBLAS_STATUS_NOT_SUPPORTED)
                GTEST_SKIP() << "devices 0 and 1 cannot reach each other";

        for(uint32_t r = 0; r < kWorld; ++r)
        {
            EXPECT_EQ(status[r], HIPBLAS_STATUS_SUCCESS) << "rank " << r;
            EXPECT_EQ(reEnable[r], hipErrorPeerAccessAlreadyEnabled)
                << "rank " << r << ": registration recorded the peer's region without making it "
                << "reachable from this device";
        }
    }

    struct DeviceBuffer
    {
        explicit DeviceBuffer(size_t bytes)
        {
            if(bytes == 0 || hipMalloc(&ptr, bytes) != hipSuccess)
                ptr = nullptr;
            else if(hipMemset(ptr, 0, bytes) != hipSuccess)
                ADD_FAILURE() << "hipMemset failed";
        }

        ~DeviceBuffer()
        {
            if(ptr)
                (void)hipFree(ptr);
        }

        DeviceBuffer(const DeviceBuffer&)            = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        void* ptr = nullptr;
    };

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

        void transposeA()
        {
            const hipblasOperation_t op = HIPBLAS_OP_T;
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutDestroy(Adesc), HIPBLAS_STATUS_SUCCESS);
            Adesc = nullptr;
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, kK, kM, kK),
                      HIPBLAS_STATUS_SUCCESS);
        }

        void detach()
        {
            hipblasLtFusedEpilogueDescriptor_t none = nullptr;
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &none, sizeof(none)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t heuristic(hipblasLtMatmulHeuristicResult_t* result   = nullptr,
                                  int*                              returned = nullptr)
        {
            hipblasLtMatmulHeuristicResult_t scratch[1] = {};
            int                              count      = 0;
            return hipblasLtMatmulAlgoGetHeuristic(handle,
                                                   matmulDesc,
                                                   Adesc,
                                                   Bdesc,
                                                   Ddesc,
                                                   Ddesc,
                                                   pref,
                                                   1,
                                                   result ? result : scratch,
                                                   returned ? returned : &count);
        }

        hipblasLtHandle_t                  handle     = nullptr;
        hipblasLtMatmulDesc_t              matmulDesc = nullptr;
        hipblasLtMatmulPreference_t        pref       = nullptr;
        hipblasLtFusedEpilogueDescriptor_t fused      = nullptr;
        hipblasLtMatrixLayout_t            Adesc      = nullptr;
        hipblasLtMatrixLayout_t            Bdesc      = nullptr;
        hipblasLtMatrixLayout_t            Ddesc      = nullptr;
    };

    TEST_F(FusedA2ADispatch_pre_checkin, DetachRestoresThePlainHeuristic)
    {
        transposeA();
        makeD();

        hipblasLtMatmulHeuristicResult_t plain    = {};
        int                              returned = 0;
        ASSERT_EQ(heuristic(&plain, &returned), HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "no plain BF16 TN solution in the loaded library";

        registerOneRank();
        completeAndAttach();
        hipblasLtMatmulHeuristicResult_t fusedResult = {};
        returned                                     = 0;
        ASSERT_EQ(heuristic(&fusedResult, &returned), HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "no fused BF16 TN solution in the loaded library";
        ASSERT_NE(memcmp(fusedResult.algo.data, plain.algo.data, sizeof(plain.algo.data)), 0);
        detach();

        hipblasLtMatmulHeuristicResult_t afterDetach = {};
        returned                                     = 0;
        ASSERT_EQ(heuristic(&afterDetach, &returned), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(returned, 1);
        EXPECT_EQ(memcmp(afterDetach.algo.data, plain.algo.data, sizeof(plain.algo.data)), 0);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, ForcedRocRollerKeepsTheFusedSolution)
    {
        transposeA();
        registerOneRank();
        completeAndAttach();
        makeD();

        hipblasLtMatmulHeuristicResult_t fusedResult = {};
        int                              returned    = 0;
        ASSERT_EQ(heuristic(&fusedResult, &returned), HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "no fused BF16 TN solution in the loaded library";

        ASSERT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
        handle                  = nullptr;
        const char*       env   = "HIPBLASLT_USE_ROCROLLER";
        const char*       saved = std::getenv(env);
        const std::string savedValue(saved ? saved : "");
        setenv(env, "1", 1);
        const hipblasStatus_t created = hipblasLtCreate(&handle);
        if(saved)
            setenv(env, savedValue.c_str(), 1);
        else
            unsetenv(env);
        ASSERT_EQ(created, HIPBLAS_STATUS_SUCCESS);
        registerOneRank();

        hipblasLtMatmulHeuristicResult_t forcedResult = {};
        returned                                      = 0;
        ASSERT_EQ(heuristic(&forcedResult, &returned), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(returned, 1);
        EXPECT_EQ(
            memcmp(forcedResult.algo.data, fusedResult.algo.data, sizeof(fusedResult.algo.data)),
            0);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, GetAllAlgosOmitsTheFusedSolution)
    {
        transposeA();
        registerOneRank();
        completeAndAttach();
        makeD();

        hipblasLtMatmulHeuristicResult_t fusedResult = {};
        int                              returned    = 0;
        ASSERT_EQ(heuristic(&fusedResult, &returned), HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "no fused BF16 TN solution in the loaded library";
        const int fusedIndex = hipblaslt_ext::getIndexFromAlgo(fusedResult.algo);

        std::vector<hipblasLtMatmulHeuristicResult_t> all;
        ASSERT_EQ(hipblaslt_ext::getAllAlgos(handle,
                                             hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                             HIPBLAS_OP_T,
                                             HIPBLAS_OP_N,
                                             HIP_R_16BF,
                                             HIP_R_16BF,
                                             HIP_R_16BF,
                                             HIP_R_16BF,
                                             HIPBLAS_COMPUTE_32F,
                                             all),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_FALSE(all.empty());
        for(hipblasLtMatmulHeuristicResult_t& result : all)
            EXPECT_NE(hipblaslt_ext::getIndexFromAlgo(result.algo), fusedIndex);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, ExtGroupedGemmRejectsAFusedDescriptor)
    {
        transposeA();
        registerOneRank();
        completeAndAttach();
        makeD();

        hipblasLtMatmulDesc_t plainDesc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&plainDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t op = HIPBLAS_OP_T;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      plainDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)),
                  HIPBLAS_STATUS_SUCCESS);

        DeviceBuffer a(kK * kM * 2), b(kK * kN * 2), d(kLdd * kN * 2);
        ASSERT_TRUE(a.ptr && b.ptr && d.ptr);
        float alpha = 1.f;
        float beta  = 0.f;

        std::vector<hipblasLtMatmulDesc_t>   descs = {plainDesc, matmulDesc};
        std::vector<void*>                   alphas(2, &alpha), betas(2, &beta);
        std::vector<void*>                   as(2, a.ptr), bs(2, b.ptr), ds(2, d.ptr);
        std::vector<hipblasLtMatrixLayout_t> matAs(2, Adesc), matBs(2, Bdesc), matDs(2, Ddesc);

        hipblaslt_ext::GroupedGemm grouped(handle,
                                           HIPBLAS_OP_T,
                                           HIPBLAS_OP_N,
                                           HIP_R_16BF,
                                           HIP_R_16BF,
                                           HIP_R_16BF,
                                           HIP_R_16BF,
                                           HIPBLAS_COMPUTE_32F);
        EXPECT_EQ(
            grouped.setProblem(descs, alphas, as, matAs, bs, matBs, betas, ds, matDs, ds, matDs),
            HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_EQ(hipblasLtMatmulDescDestroy(plainDesc), HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, IsAlgoSupportedRejectsChannelOutsideTheCommunicator)
    {
        transposeA();
        registerOneRank();
        completeAndAttach();
        makeD();

        hipblasLtMatmulHeuristicResult_t fusedResult = {};
        int                              returned    = 0;
        ASSERT_EQ(heuristic(&fusedResult, &returned), HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "no fused BF16 TN solution in the loaded library";

        const float alpha     = 1.f;
        const float beta      = 0.f;
        size_t      workspace = 0;
        ASSERT_EQ(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                       matmulDesc,
                                                       &alpha,
                                                       Adesc,
                                                       Bdesc,
                                                       &beta,
                                                       Ddesc,
                                                       Ddesc,
                                                       fusedResult.algo,
                                                       workspace),
                  HIPBLAS_STATUS_SUCCESS);

        const uint32_t channel = 1;
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL, &channel, sizeof(channel)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                       matmulDesc,
                                                       &alpha,
                                                       Adesc,
                                                       Bdesc,
                                                       &beta,
                                                       Ddesc,
                                                       Ddesc,
                                                       fusedResult.algo,
                                                       workspace),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, IsAlgoSupportedRejectsPerRankArraysNotOfWorldLength)
    {
        transposeA();
        registerOneRank();
        completeAndAttach();
        makeD();

        hipblasLtMatmulHeuristicResult_t fusedResult = {};
        int                              returned    = 0;
        ASSERT_EQ(heuristic(&fusedResult, &returned), HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "no fused BF16 TN solution in the loaded library";

        void* recv[2] = {reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x1100)};
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, recv, sizeof(recv)),
                  HIPBLAS_STATUS_SUCCESS);

        const float alpha     = 1.f;
        const float beta      = 0.f;
        size_t      workspace = 0;
        EXPECT_EQ(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                       matmulDesc,
                                                       &alpha,
                                                       Adesc,
                                                       Bdesc,
                                                       &beta,
                                                       Ddesc,
                                                       Ddesc,
                                                       fusedResult.algo,
                                                       workspace),
                  HIPBLAS_STATUS_INVALID_VALUE);

        int supported = 0;
        EXPECT_NE(hipblaslt_ext::isSolutionSupported(&fusedResult,
                                                     handle,
                                                     matmulDesc,
                                                     &alpha,
                                                     Adesc,
                                                     Bdesc,
                                                     &beta,
                                                     Ddesc,
                                                     Ddesc,
                                                     &workspace,
                                                     &supported),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(supported, 0);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsMissingExtent)
    {
        registerOneRank();
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
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(matmulDesc,
                                                  HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                                  &fused,
                                                  sizeof(fused)),
                  HIPBLAS_STATUS_SUCCESS);
        makeD();

        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    // One entry, matching world, so the length check is not what rejects these two.
    TEST_F(FusedA2ADispatch_pre_checkin, RejectsNullPeerEntry)
    {
        registerOneRank();
        completeAndAttach();
        makeD();

        void* recv = nullptr;
        ASSERT_EQ(
            hipblasLtFusedEpilogueSetAttribute(
                fused, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, &recv, sizeof(recv)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2ADispatch_pre_checkin, RejectsIncompleteQueueEntry)
    {
        registerOneRank();
        completeAndAttach();
        makeD();

        hipblasLtSdmaQueue_t queue = fakeQueue(0x2000);
        queue.doorbell             = nullptr;
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused,
                      HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                      &queue,
                      sizeof(queue)),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
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

    /************************************************************************
     * Prologue dispatch: B is gathered across ranks before the main loop.
     ***********************************************************************/

    class FusedA2APrologueDispatch_pre_checkin : public ::testing::Test
    {
    protected:
        static constexpr int64_t kM = 256;
        static constexpr int64_t kN = 256;
        static constexpr int64_t kK = 512;

        void SetUp() override
        {
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            const hipblasOperation_t opA = HIPBLAS_OP_T;
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueCreate(&prologue), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedPrologueAdd(prologue, HIPBLASLT_FUSEABLE_PROLOGUE_A2A_PREFIX),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, kK, kM, kK),
                      HIPBLAS_STATUS_SUCCESS);
            makeB();
            makeD();
        }

        void TearDown() override
        {
            for(hipblasLtMatrixLayout_t layout : {Adesc, Bdesc, Ddesc})
                if(layout)
                    EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layout), HIPBLAS_STATUS_SUCCESS);
            for(hipblasLtFusedEpilogueDescriptor_t desc : {prologue, epilogue})
                if(desc)
                    EXPECT_EQ(hipblasLtFusedEpilogueDestroy(desc), HIPBLAS_STATUS_SUCCESS);
            if(pref)
                EXPECT_EQ(hipblasLtMatmulPreferenceDestroy(pref), HIPBLAS_STATUS_SUCCESS);
            if(matmulDesc)
                EXPECT_EQ(hipblasLtMatmulDescDestroy(matmulDesc), HIPBLAS_STATUS_SUCCESS);
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
        }

        void makeB(int64_t k = kK, int64_t n = kN, int64_t ldb = kK)
        {
            if(Bdesc)
                ASSERT_EQ(hipblasLtMatrixLayoutDestroy(Bdesc), HIPBLAS_STATUS_SUCCESS);
            Bdesc = nullptr;
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_16BF, k, n, ldb),
                      HIPBLAS_STATUS_SUCCESS);
        }

        void makeD(hipDataType type = HIP_R_16BF, int64_t m = kM, int64_t n = kN)
        {
            if(Ddesc)
                ASSERT_EQ(hipblasLtMatrixLayoutDestroy(Ddesc), HIPBLAS_STATUS_SUCCESS);
            Ddesc = nullptr;
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Ddesc, type, m, n, m), HIPBLAS_STATUS_SUCCESS);
        }

        void registerOneRank(uint32_t channels = 1)
        {
            ASSERT_EQ(hipblasLtSetDeviceComm(handle, 0, 1, channels, memcpyAllgather, nullptr),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t setPeerBPtrs(const std::vector<void*>& ptrs)
        {
            return hipblasLtFusedEpilogueSetAttribute(
                prologue,
                HIPBLASLT_FUSED_PROLOGUE_A2A_PREFIX_PEER_B_PTRS,
                ptrs.data(),
                ptrs.size() * sizeof(void*));
        }

        hipblasStatus_t setGatherBuffer(void* buffer)
        {
            return hipblasLtFusedEpilogueSetAttribute(
                prologue,
                HIPBLASLT_FUSED_PROLOGUE_A2A_PREFIX_GATHER_BUFFER,
                &buffer,
                sizeof(buffer));
        }

        hipblasStatus_t setQueues(const std::vector<hipblasLtSdmaQueue_t>& queues)
        {
            return hipblasLtFusedEpilogueSetAttribute(
                prologue,
                HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                queues.data(),
                queues.size() * sizeof(hipblasLtSdmaQueue_t));
        }

        hipblasStatus_t setExtent(int64_t extent)
        {
            return hipblasLtFusedEpilogueSetAttribute(
                prologue, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &extent, sizeof(extent));
        }

        hipblasStatus_t attach(hipblasLtMatmulDescAttributes_t    attr,
                               hipblasLtFusedEpilogueDescriptor_t desc)
        {
            return hipblasLtMatmulDescSetAttribute(matmulDesc, attr, &desc, sizeof(desc));
        }

        // A descriptor a one-rank group can run.
        void completeAndAttach(void* b      = reinterpret_cast<void*>(0x100000),
                               void* gather = reinterpret_cast<void*>(0x200000))
        {
            ASSERT_EQ(setPeerBPtrs({b}), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(setGatherBuffer(gather), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(setQueues({fakeQueue(0x2000)}), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, prologue),
                      HIPBLAS_STATUS_SUCCESS);
        }

        // A complete epilogue all-to-all for a one-rank group.
        void makeEpilogue()
        {
            ASSERT_EQ(hipblasLtFusedEpilogueCreate(&epilogue), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueAdd(epilogue, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX),
                      HIPBLAS_STATUS_SUCCESS);
            const int64_t am = kM;
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          epilogue, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT, &am, sizeof(am)),
                      HIPBLAS_STATUS_SUCCESS);
            void* recv = reinterpret_cast<void*>(0x1000);
            ASSERT_EQ(
                hipblasLtFusedEpilogueSetAttribute(
                    epilogue, HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS, &recv, sizeof(recv)),
                HIPBLAS_STATUS_SUCCESS);
            hipblasLtSdmaQueue_t queue = fakeQueue(0x2000);
            ASSERT_EQ(
                hipblasLtFusedEpilogueSetAttribute(epilogue,
                                                   HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES,
                                                   &queue,
                                                   sizeof(queue)),
                HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t heuristic(int* returned = nullptr)
        {
            hipblasLtMatmulHeuristicResult_t result[1] = {};
            int                              count     = 0;
            return hipblasLtMatmulAlgoGetHeuristic(handle,
                                                   matmulDesc,
                                                   Adesc,
                                                   Bdesc,
                                                   Ddesc,
                                                   Ddesc,
                                                   pref,
                                                   1,
                                                   result,
                                                   returned ? returned : &count);
        }

        bool firstAlgo(hipblasLtMatmulAlgo_t& algo)
        {
            hipblasLtMatmulHeuristicResult_t result   = {};
            int                              returned = 0;
            if(hipblasLtMatmulAlgoGetHeuristic(
                   handle, matmulDesc, Adesc, Bdesc, Ddesc, Ddesc, pref, 1, &result, &returned)
                   != HIPBLAS_STATUS_SUCCESS
               || returned == 0)
                return false;
            algo = result.algo;
            return true;
        }

        hipblasStatus_t isAlgoSupported(hipblasLtMatmulAlgo_t& algo)
        {
            const float alpha     = 1.f;
            const float beta      = 0.f;
            size_t      workspace = 0;
            return hipblaslt_ext::matmulIsAlgoSupported(
                handle, matmulDesc, &alpha, Adesc, Bdesc, &beta, Ddesc, Ddesc, algo, workspace);
        }

        hipblasStatus_t matmul(const void* a, const void* b, void* d)
        {
            const float alpha = 1.f;
            const float beta  = 0.f;
            return hipblasLtMatmul(handle,
                                   matmulDesc,
                                   &alpha,
                                   a,
                                   Adesc,
                                   b,
                                   Bdesc,
                                   &beta,
                                   d,
                                   Ddesc,
                                   d,
                                   Ddesc,
                                   nullptr,
                                   nullptr,
                                   0,
                                   nullptr);
        }

        hipblasLtHandle_t                  handle     = nullptr;
        hipblasLtMatmulDesc_t              matmulDesc = nullptr;
        hipblasLtMatmulPreference_t        pref       = nullptr;
        hipblasLtFusedEpilogueDescriptor_t prologue   = nullptr;
        hipblasLtFusedEpilogueDescriptor_t epilogue   = nullptr;
        hipblasLtMatrixLayout_t            Adesc      = nullptr;
        hipblasLtMatrixLayout_t            Bdesc      = nullptr;
        hipblasLtMatrixLayout_t            Ddesc      = nullptr;
    };

    TEST_F(FusedA2APrologueDispatch_pre_checkin, CompleteDescriptorPassesTheHeuristicGate)
    {
        registerOneRank();
        completeAndAttach();
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, ExtentEqualToKIsAccepted)
    {
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(setExtent(kK), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, ExtentOtherThanKIsRejected)
    {
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(setExtent(kK / 2), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, MissingQueuesReturnNoAlgo)
    {
        registerOneRank();
        ASSERT_EQ(setPeerBPtrs({reinterpret_cast<void*>(0x100000)}), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(setGatherBuffer(reinterpret_cast<void*>(0x200000)), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, prologue), HIPBLAS_STATUS_SUCCESS);

        int returned = -1;
        EXPECT_EQ(heuristic(&returned), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(returned, 0);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, OwnQueueEntryMayBeEmpty)
    {
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(setQueues({hipblasLtSdmaQueue_t{}}), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsQueuesThatDoNotMatchWorld)
    {
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(setQueues({fakeQueue(0x2000), fakeQueue(0x3000)}), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, IsAlgoSupportedRejectsQueuesThatDoNotMatchWorld)
    {
        registerOneRank();
        completeAndAttach();
        hipblasLtMatmulAlgo_t algo;
        if(!firstAlgo(algo))
            GTEST_SKIP() << "no A2AGemm solution in the loaded library";
        ASSERT_EQ(isAlgoSupported(algo), HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(setQueues({fakeQueue(0x2000), fakeQueue(0x3000)}), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(isAlgoSupported(algo), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, IsAlgoSupportedRejectsMissingQueues)
    {
        registerOneRank();
        completeAndAttach();
        hipblasLtMatmulAlgo_t algo;
        if(!firstAlgo(algo))
            GTEST_SKIP() << "no A2AGemm solution in the loaded library";

        hipblasLtFusedEpilogueDescriptor_t noQueues = nullptr;
        ASSERT_EQ(hipblasLtFusedEpilogueCreate(&noQueues), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedPrologueAdd(noQueues, HIPBLASLT_FUSEABLE_PROLOGUE_A2A_PREFIX),
                  HIPBLAS_STATUS_SUCCESS);
        std::swap(prologue, noQueues);
        ASSERT_EQ(setPeerBPtrs({reinterpret_cast<void*>(0x100000)}), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(setGatherBuffer(reinterpret_cast<void*>(0x200000)), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, prologue), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtFusedEpilogueDestroy(noQueues), HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(isAlgoSupported(algo), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsMissingPeerBPtrs)
    {
        registerOneRank();
        ASSERT_EQ(setGatherBuffer(reinterpret_cast<void*>(0x200000)), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(setQueues({fakeQueue(0x2000)}), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, prologue), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsPeerBPtrsThatDoNotMatchWorld)
    {
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(
            setPeerBPtrs({reinterpret_cast<void*>(0x100000), reinterpret_cast<void*>(0x110000)}),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsNullPeerBEntry)
    {
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(setPeerBPtrs({nullptr}), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsMissingGatherBuffer)
    {
        registerOneRank();
        ASSERT_EQ(setPeerBPtrs({reinterpret_cast<void*>(0x100000)}), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(setQueues({fakeQueue(0x2000)}), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, prologue), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsMissingCommunicator)
    {
        completeAndAttach();
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsChannelOutsideTheCommunicator)
    {
        registerOneRank();
        completeAndAttach();
        const uint32_t channel = 1;
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      prologue, HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL, &channel, sizeof(channel)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsBothSidesAttached)
    {
        registerOneRank();
        completeAndAttach();
        makeEpilogue();
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, epilogue), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsAPrologueOnTheEpilogueAttribute)
    {
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, nullptr), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, prologue), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsAnEpilogueOnThePrologueAttribute)
    {
        registerOneRank();
        makeEpilogue();
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, epilogue), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsTransposedB)
    {
        registerOneRank();
        completeAndAttach();
        const hipblasOperation_t opB = HIPBLAS_OP_T;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                  HIPBLAS_STATUS_SUCCESS);
        makeB(kN, kK, kK);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsAnEmptyBound)
    {
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(hipblasLtMatrixLayoutDestroy(Adesc), HIPBLAS_STATUS_SUCCESS);
        Adesc = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, 0, kM, 1),
                  HIPBLAS_STATUS_SUCCESS);
        makeB(0, kN, 0);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsLdbOtherThanK)
    {
        registerOneRank();
        completeAndAttach();
        makeB(kK, kN, kK + 64);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsBatchedGemm)
    {
        registerOneRank();
        completeAndAttach();
        const int32_t batchCount = 2;
        for(hipblasLtMatrixLayout_t layout : {Adesc, Bdesc, Ddesc})
            ASSERT_EQ(
                hipblasLtMatrixLayoutSetAttribute(
                    layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)),
                HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, ReportsMissingCapabilityForNonBf16B)
    {
        registerOneRank();
        completeAndAttach();
        for(hipblasLtMatrixLayout_t* layout : {&Adesc, &Bdesc})
        {
            ASSERT_EQ(hipblasLtMatrixLayoutDestroy(*layout), HIPBLAS_STATUS_SUCCESS);
            *layout = nullptr;
        }
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16F, kK, kM, kK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_16F, kK, kN, kK),
                  HIPBLAS_STATUS_SUCCESS);
        makeD(HIP_R_16F);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_NOT_SUPPORTED);
    }

    // The gathered row is one copy of K / world elements, whose width field is
    // 14 bits in 16-byte units.
    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsRowBeyondTheCopyDescriptor)
    {
        constexpr int64_t kWideK = 8 * 16384 + 8;
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(hipblasLtMatrixLayoutDestroy(Adesc), HIPBLAS_STATUS_SUCCESS);
        Adesc = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, kWideK, kM, kWideK),
                  HIPBLAS_STATUS_SUCCESS);
        makeB(kWideK, kN, kWideK);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, RejectsABlockThatIsNotA128ByteMultiple)
    {
        constexpr int64_t kNarrowK = 8;
        constexpr int64_t kNarrowN = 4;
        registerOneRank();
        completeAndAttach();
        ASSERT_EQ(hipblasLtMatrixLayoutDestroy(Adesc), HIPBLAS_STATUS_SUCCESS);
        Adesc = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, kNarrowK, kM, kNarrowK),
                  HIPBLAS_STATUS_SUCCESS);
        makeB(kNarrowK, kNarrowN, kNarrowK);
        makeD(HIP_R_16BF, kM, kNarrowN);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, EpilogueShapeRulesDoNotApply)
    {
        registerOneRank();
        completeAndAttach();
        makeD(HIP_R_32F, 64, kN);
        ASSERT_EQ(hipblasLtMatrixLayoutDestroy(Adesc), HIPBLAS_STATUS_SUCCESS);
        Adesc = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, kK, 64, kK),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(), HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, MatmulRejectsMissingQueues)
    {
        registerOneRank();
        DeviceBuffer a(kK * kM * 2), b(kK * kN * 2), d(kM * kN * 2), gather(kK * kN * 2);
        ASSERT_TRUE(a.ptr && b.ptr && d.ptr && gather.ptr);
        ASSERT_EQ(setPeerBPtrs({b.ptr}), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(setGatherBuffer(gather.ptr), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, prologue), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(matmul(a.ptr, b.ptr, d.ptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, MatmulRejectsBOtherThanItsPeerEntry)
    {
        registerOneRank();
        DeviceBuffer a(kK * kM * 2), b(kK * kN * 2), other(kK * kN * 2), d(kM * kN * 2),
            gather(kK * kN * 2);
        ASSERT_TRUE(a.ptr && b.ptr && other.ptr && d.ptr && gather.ptr);
        completeAndAttach(other.ptr, gather.ptr);
        EXPECT_EQ(matmul(a.ptr, b.ptr, d.ptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, MatmulRejectsAGatherBufferOverlappingD)
    {
        registerOneRank();
        DeviceBuffer a(kK * kM * 2), b(kK * kN * 2), d(kK * kN * 2 + kM * kN * 2);
        ASSERT_TRUE(a.ptr && b.ptr && d.ptr);
        completeAndAttach(b.ptr, d.ptr);
        EXPECT_EQ(matmul(a.ptr, b.ptr, d.ptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, MatmulRejectsAGatherBufferOverlappingB)
    {
        registerOneRank();
        DeviceBuffer a(kK * kM * 2), b(2 * kK * kN * 2), d(kM * kN * 2);
        ASSERT_TRUE(a.ptr && b.ptr && d.ptr);
        completeAndAttach(b.ptr, static_cast<char*>(b.ptr) + 128);
        EXPECT_EQ(matmul(a.ptr, b.ptr, d.ptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, MatmulRejectsAGatherBufferOverlappingA)
    {
        registerOneRank();
        DeviceBuffer a(kK * kM * 2), b(kK * kN * 2), d(kM * kN * 2);
        ASSERT_TRUE(a.ptr && b.ptr && d.ptr);
        completeAndAttach(b.ptr, a.ptr);
        EXPECT_EQ(matmul(a.ptr, b.ptr, d.ptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, ExtGemmRejectsAPrologueDescriptor)
    {
        registerOneRank();
        completeAndAttach();

        hipblaslt_ext::Gemm gemm(handle,
                                 HIPBLAS_OP_T,
                                 HIPBLAS_OP_N,
                                 HIP_R_16BF,
                                 HIP_R_16BF,
                                 HIP_R_16BF,
                                 HIP_R_16BF,
                                 HIPBLAS_COMPUTE_32F);
        DeviceBuffer        a(kK * kM * 2), b(kK * kN * 2), d(kM * kN * 2);
        ASSERT_TRUE(a.ptr && b.ptr && d.ptr);
        const float alpha = 1.f;
        const float beta  = 0.f;
        EXPECT_EQ(
            gemm.setProblem(
                matmulDesc, &alpha, a.ptr, Adesc, b.ptr, Bdesc, &beta, d.ptr, Ddesc, d.ptr, Ddesc),
            HIPBLAS_STATUS_INVALID_VALUE);
    }

    // BF16 values drawn from {-1, 0, 1}.
    std::vector<uint16_t> smallIntegerBf16(size_t count, uint32_t seed)
    {
        std::vector<uint16_t> values(count);
        uint32_t              state = seed;
        for(uint16_t& v : values)
        {
            state         = state * 1664525u + 1013904223u;
            const float f = float(int((state >> 16) % 3) - 1);
            uint32_t    bits;
            memcpy(&bits, &f, sizeof(bits));
            v = uint16_t(bits >> 16);
        }
        return values;
    }

    TEST_F(FusedA2APrologueDispatch_pre_checkin, OneRankMatchesAPlainMatmul)
    {
        constexpr int64_t kRunM = 512;
        constexpr int64_t kRunN = 512;
        constexpr int64_t kRunK = 1024;
        ASSERT_EQ(hipblasLtMatrixLayoutDestroy(Adesc), HIPBLAS_STATUS_SUCCESS);
        Adesc = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_16BF, kRunK, kRunM, kRunK),
                  HIPBLAS_STATUS_SUCCESS);
        makeB(kRunK, kRunN, kRunK);
        makeD(HIP_R_16BF, kRunM, kRunN);

        const size_t aBytes = kRunK * kRunM * 2, bBytes = kRunK * kRunN * 2;
        const size_t dBytes = kRunM * kRunN * 2;
        DeviceBuffer a(aBytes), b(bBytes), gather(bBytes), plainD(dBytes), fusedD(dBytes);
        ASSERT_TRUE(a.ptr && b.ptr && gather.ptr && plainD.ptr && fusedD.ptr);
        std::vector<uint16_t> hostA = smallIntegerBf16(kRunK * kRunM, 1);
        std::vector<uint16_t> hostB = smallIntegerBf16(kRunK * kRunN, 2);
        ASSERT_EQ(hipMemcpy(a.ptr, hostA.data(), aBytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(b.ptr, hostB.data(), bBytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(gather.ptr, 0xff, bBytes), hipSuccess);

        hipblasLtMatmulHeuristicResult_t plain    = {};
        int                              returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, matmulDesc, Adesc, Bdesc, Ddesc, Ddesc, pref, 1, &plain, &returned),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(returned, 1);

        registerOneRank();
        ASSERT_EQ(setPeerBPtrs({b.ptr}), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(setGatherBuffer(gather.ptr), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(setQueues({hipblasLtSdmaQueue_t{}}), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, prologue), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulHeuristicResult_t fusedResult = {};
        returned                                     = 0;
        ASSERT_EQ(
            hipblasLtMatmulAlgoGetHeuristic(
                handle, matmulDesc, Adesc, Bdesc, Ddesc, Ddesc, pref, 1, &fusedResult, &returned),
            HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "no A2AGemm BF16 TN solution in the loaded library";
        ASSERT_EQ(fusedResult.workspaceSize, 0u);

        const float alpha = 1.f;
        const float beta  = 0.f;
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  matmulDesc,
                                  &alpha,
                                  a.ptr,
                                  Adesc,
                                  b.ptr,
                                  Bdesc,
                                  &beta,
                                  fusedD.ptr,
                                  Ddesc,
                                  fusedD.ptr,
                                  Ddesc,
                                  &fusedResult.algo,
                                  nullptr,
                                  0,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(attach(HIPBLASLT_MATMUL_DESC_FUSED_PROLOGUE, nullptr), HIPBLAS_STATUS_SUCCESS);
        DeviceBuffer ws(plain.workspaceSize);
        ASSERT_TRUE(plain.workspaceSize == 0 || ws.ptr);
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  matmulDesc,
                                  &alpha,
                                  a.ptr,
                                  Adesc,
                                  b.ptr,
                                  Bdesc,
                                  &beta,
                                  plainD.ptr,
                                  Ddesc,
                                  plainD.ptr,
                                  Ddesc,
                                  &plain.algo,
                                  ws.ptr,
                                  plain.workspaceSize,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<uint16_t> hostPlain(kRunM * kRunN), hostFused(kRunM * kRunN);
        ASSERT_EQ(hipMemcpy(hostPlain.data(), plainD.ptr, dBytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(hostFused.data(), fusedD.ptr, dBytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        size_t mismatches = 0;
        for(size_t i = 0; i < hostPlain.size(); ++i)
            mismatches += hostPlain[i] != hostFused[i];
        EXPECT_EQ(mismatches, 0u);
    }
}
