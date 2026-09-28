// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/ContractionSolution.hpp>
#include <gtest/gtest.h>

// Use a relative include to avoid the library/client utility.hpp collision.
#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_synchronizer.hpp"

#include <array>
#include <vector>

namespace
{
    // Only allocation/lookup is replaced. The binding, width validation and
    // solution classification are the same functions called by tensile_host.
    // Distinct addresses make a stale pointer or a wrapped index observable;
    // no device kernels or particular shipped solutions are needed.
    struct FlagStorage
    {
        static constexpr size_t                 c_syncSkSlotsPerStream = 16;
        std::array<int, c_syncSkSlotsPerStream> sk{}, gsu{};
        size_t                                  claims    = 0;
        bool                                    available = true;
        rocblaslt_status                        status    = rocblaslt_status_success;

        rocblaslt_status streamKFlagsForStream(hipStream_t, size_t i, void** out)
        {
            ++claims;
            *out = available ? &sk.at(i) : nullptr;
            return status;
        }
        void* gsuFlagsForProblem(size_t i)
        {
            return i < gsu.size() ? &gsu[i] : nullptr;
        }
    };

    void setFlagReader(TensileLite::ContractionSolution& solution)
    {
        solution.sizeMapping.streamK       = 1;
        solution.sizeMapping.streamKAtomic = 0;
        solution.problemType.outputAmaxD   = false;
    }

    TEST(SynchronizerBinding, ReinitializeRestoresReductionRegion)
    {
        FlagStorage                      storage;
        TensileLite::ContractionInputs   inputs;
        TensileLite::ContractionSolution solution;
        setFlagReader(solution);
        ASSERT_EQ(rocblaslt::bindSynchronizers(storage, solution, nullptr, &inputs),
                  rocblaslt_status_success);
        ASSERT_EQ(inputs.Synchronizer, &storage.sk[0]);

        // The object API reuses inputs while choosing a non-Stream-K solution.
        solution.sizeMapping.streamK = 0;
        ASSERT_EQ(rocblaslt::bindSynchronizers(storage, solution, nullptr, &inputs),
                  rocblaslt_status_success);
        EXPECT_EQ(inputs.Synchronizer, &storage.gsu[0]);
        EXPECT_EQ(storage.claims, 1u);
    }

    TEST(SynchronizerBinding, AtomicAndAmaxSolutionsRestoreReductionRegion)
    {
        FlagStorage                    storage;
        TensileLite::ContractionInputs inputs;
        for(bool amax : {false, true})
        {
            TensileLite::ContractionSolution solution;
            setFlagReader(solution);
            solution.sizeMapping.streamKAtomic = amax ? 0 : 1;
            solution.problemType.outputAmaxD   = amax;
            inputs.Synchronizer                = &storage.sk[0];
            ASSERT_EQ(rocblaslt::bindSynchronizers(storage, solution, nullptr, &inputs),
                      rocblaslt_status_success);
            EXPECT_EQ(inputs.Synchronizer, &storage.gsu[0]);
        }
        EXPECT_EQ(storage.claims, 0u);
    }

    TEST(SynchronizerBinding, GroupBoundaryAndProblemOffsets)
    {
        FlagStorage                                 storage;
        std::vector<TensileLite::ContractionInputs> inputs(16);
        TensileLite::ContractionSolution            solution;
        setFlagReader(solution);
        ASSERT_EQ(
            rocblaslt::bindSynchronizers(storage, solution, nullptr, inputs.data(), inputs.size()),
            rocblaslt_status_success);
        for(size_t i = 0; i < inputs.size(); ++i)
            EXPECT_EQ(inputs[i].Synchronizer, &storage.sk[i]);

        solution.sizeMapping.streamK = 0;
        ASSERT_EQ(
            rocblaslt::bindSynchronizers(storage, solution, nullptr, inputs.data(), inputs.size()),
            rocblaslt_status_success);
        for(size_t i = 0; i < inputs.size(); ++i)
            EXPECT_EQ(inputs[i].Synchronizer, &storage.gsu[i]);

        // Exactly one beyond capacity must fail before claiming or rebinding.
        inputs.emplace_back();
        inputs.back().Synchronizer = &storage.gsu[0];
        setFlagReader(solution);
        ASSERT_EQ(
            rocblaslt::bindSynchronizers(storage, solution, nullptr, inputs.data(), inputs.size()),
            rocblaslt_status_invalid_value);
        EXPECT_EQ(storage.claims, 16u);
        for(size_t i = 0; i < 16; ++i)
            EXPECT_EQ(inputs[i].Synchronizer, &storage.gsu[i]);
        EXPECT_EQ(inputs.back().Synchronizer, &storage.gsu[0]);

        // A solution that does not read flags remains valid beyond the limit.
        solution.sizeMapping.streamK = 0;
        ASSERT_EQ(
            rocblaslt::bindSynchronizers(storage, solution, nullptr, inputs.data(), inputs.size()),
            rocblaslt_status_success);
        EXPECT_EQ(inputs.back().Synchronizer, nullptr);
    }

    TEST(SynchronizerBinding, MissingStreamKBufferRestoresReductionRegion)
    {
        FlagStorage storage;
        storage.available = false;
        TensileLite::ContractionSolution solution;
        setFlagReader(solution);
        TensileLite::ContractionInputs inputs;
        inputs.Synchronizer = &storage.sk[0];
        ASSERT_EQ(rocblaslt::bindSynchronizers(storage, solution, nullptr, &inputs),
                  rocblaslt_status_success);
        EXPECT_EQ(inputs.Synchronizer, &storage.gsu[0]);
    }

    TEST(SynchronizerBinding, FailedClaimIsReturned)
    {
        FlagStorage storage;
        storage.status = rocblaslt_status_internal_error;
        TensileLite::ContractionSolution solution;
        setFlagReader(solution);
        TensileLite::ContractionInputs inputs;
        inputs.Synchronizer = &storage.gsu[0];
        EXPECT_EQ(rocblaslt::bindSynchronizers(storage, solution, nullptr, &inputs),
                  rocblaslt_status_internal_error);
        EXPECT_EQ(inputs.Synchronizer, &storage.gsu[0]);
    }

    TEST(SynchronizerBinding, DirectBindingOverwritesPreviousRegion)
    {
        int                              sk, gsu;
        TensileLite::ContractionInputs   inputs;
        TensileLite::ContractionSolution solution;
        setFlagReader(solution);
        inputs.Synchronizer = rocblaslt::synchronizerForSolution(solution, &sk, &gsu);
        ASSERT_EQ(inputs.Synchronizer, &sk);
        solution.sizeMapping.streamK = 0;
        inputs.Synchronizer          = rocblaslt::synchronizerForSolution(solution, &sk, &gsu);
        EXPECT_EQ(inputs.Synchronizer, &gsu);
        setFlagReader(solution);
        inputs.Synchronizer = rocblaslt::synchronizerForSolution(solution, nullptr, &gsu);
        EXPECT_EQ(inputs.Synchronizer, &gsu);
    }
}
