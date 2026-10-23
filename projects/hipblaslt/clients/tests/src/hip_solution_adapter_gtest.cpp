// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <gtest/gtest.h>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace
{
    constexpr char primary[] = "/library/primary.co";
    constexpr char helper[]  = "/library/Kernels.so-000-gfx942.hsaco";
    constexpr char minus[]   = "/library/Kernels.so-000-gfx942-xnack-.hsaco";
    constexpr char plus[]    = "/library/Kernels.so-000-gfx942-xnack+.hsaco";

    struct HipLoads
    {
        std::deque<std::pair<std::string, hipError_t>> expected;
        std::map<hipModule_t, std::unique_ptr<int>>    live;
        hipError_t                                     lastError = hipSuccess;
        unsigned                                       unloads   = 0;
    };

    // Each test owns this state and destroys the adapter before releasing it.
    HipLoads* hipLoads = nullptr;

    hipError_t moduleLoad(hipModule_t* module, const char* path);
    hipError_t moduleUnload(hipModule_t module);
    hipError_t getLastError();

    class HipSolutionAdapterTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            hipLoads = &loads;
            adapter  = std::make_unique<TensileLite::hip::SolutionAdapter>(
                TensileLite::hip::SolutionAdapter::ModuleApi{
                    moduleLoad, moduleUnload, getLastError});
        }

        void TearDown() override
        {
            adapter.reset();
            EXPECT_TRUE(loads.live.empty()) << "adapter leaked module handles";
            EXPECT_TRUE(loads.expected.empty()) << "expected module loads did not run";
            hipLoads = nullptr;
        }

        void expectLoad(const char* path, hipError_t error = hipSuccess)
        {
            loads.expected.emplace_back(path, error);
        }

        void expectHelperFailure(hipError_t error)
        {
            expectLoad(helper, error);
            expectLoad(minus, error);
            expectLoad(plus, error);
        }

        HipLoads                                           loads;
        std::unique_ptr<TensileLite::hip::SolutionAdapter> adapter;
    };

    class HipSolutionAdapterRecoveryTest : public HipSolutionAdapterTest,
                                           public ::testing::WithParamInterface<hipError_t>
    {
    };

    hipError_t moduleLoad(hipModule_t* module, const char* path)
    {
        // Unexpected recursive recovery fails the test and returns a non-retryable
        // error, rather than hanging the test process or exhausting its stack.
        if(hipLoads->expected.empty())
        {
            ADD_FAILURE() << "unexpected module load: " << path;
            return hipErrorInvalidValue;
        }
        auto [expectedPath, error] = hipLoads->expected.front();
        hipLoads->expected.pop_front();
        EXPECT_EQ(expectedPath, path);
        if(error == hipSuccess)
        {
            auto storage = std::make_unique<int>(0);
            *module      = reinterpret_cast<hipModule_t>(storage.get());
            hipLoads->live.emplace(*module, std::move(storage));
        }
        else
            hipLoads->lastError = error;
        return error;
    }

    hipError_t moduleUnload(hipModule_t module)
    {
        EXPECT_EQ(hipLoads->live.erase(module), 1u) << "double or unowned module unload";
        ++hipLoads->unloads;
        return hipSuccess;
    }

    hipError_t getLastError()
    {
        return std::exchange(hipLoads->lastError, hipSuccess);
    }

    TEST_F(HipSolutionAdapterTest,
           smoke_ContextDoesNotLoadHelpersAndInitializationDoesNotDuplicateThem)
    {
        adapter->setLazyLoadingContext("gfx942:sramecc+:xnack-", "/library");
        EXPECT_TRUE(loads.live.empty());
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942:sramecc+:xnack-", "/library"), hipSuccess);
        EXPECT_EQ(adapter->initializeLazyLoading("gfx942", "/library/"), hipSuccess);
        EXPECT_EQ(loads.live.size(), 2u);
    }

    TEST_F(HipSolutionAdapterTest, smoke_XnackFallbackRegistersHelperWithoutDuplicateLoads)
    {
        expectLoad(helper, hipErrorFileNotFound);
        expectLoad(minus);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942:sramecc+:xnack-", "/library"), hipSuccess);
        EXPECT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        EXPECT_EQ(loads.live.size(), 1u);
    }

    TEST_F(HipSolutionAdapterTest, smoke_OutOfMemoryDoesNotEnterRecovery)
    {
        adapter->setLazyLoadingContext("gfx942", "/library");
        expectLoad(primary, hipErrorOutOfMemory);
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), hipErrorOutOfMemory);
        EXPECT_EQ(loads.unloads, 0u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_MissingContextPreservesPrimaryError)
    {
        expectLoad(primary, GetParam());
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), GetParam());
        EXPECT_EQ(loads.unloads, 0u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest,
           smoke_EarlyContextAllowsRecoveryBeforeHelperInitialization)
    {
        adapter->setLazyLoadingContext("gfx942:sramecc+:xnack-", "/library");
        expectLoad(primary, GetParam());
        expectLoad(helper);
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        EXPECT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        EXPECT_EQ(loads.live.size(), 2u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_RecoveryUnloadsOldModulesAndRestoresHelpers)
    {
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        expectLoad(primary, GetParam());
        expectLoad(helper);
        expectLoad(primary);
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        EXPECT_EQ(loads.unloads, 1u);
        EXPECT_EQ(loads.live.size(), 2u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_MissingHelpersPreservePrimaryError)
    {
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        expectLoad(primary, GetParam());
        expectHelperFailure(hipErrorFileNotFound);
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), GetParam());
        EXPECT_EQ(loads.unloads, 1u);
        EXPECT_TRUE(loads.live.empty());
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_RecoverableHelperErrorsDoNotRecurse)
    {
        adapter->setLazyLoadingContext("gfx942", "/library");
        expectLoad(primary, hipErrorLaunchFailure);
        expectHelperFailure(GetParam());
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), hipErrorLaunchFailure);
        EXPECT_TRUE(loads.live.empty());
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_DirectHelperInitializationDoesNotRecurse)
    {
        expectHelperFailure(GetParam());
        EXPECT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), GetParam());
        EXPECT_EQ(loads.unloads, 0u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_FailedPrimaryRetryIsReturnedWithoutAnotherRecovery)
    {
        adapter->setLazyLoadingContext("gfx942", "/library");
        expectLoad(primary, hipErrorLaunchFailure);
        expectLoad(helper);
        expectLoad(primary, GetParam());
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), GetParam());
        EXPECT_EQ(loads.live.size(), 1u);
    }

    TEST_P(HipSolutionAdapterRecoveryTest, smoke_RecoveryReclaimsRotationCopies)
    {
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        expectLoad(primary);
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFileExtraCopies(primary, 2), hipSuccess);
        adapter->selectRotationCopy(2);
        ASSERT_EQ(adapter->numRotationModules(), 3);
        expectLoad(primary, GetParam());
        expectLoad(helper);
        expectLoad(primary);
        EXPECT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        EXPECT_EQ(adapter->numRotationModules(), 1);
        EXPECT_EQ(loads.unloads, 4u);
        EXPECT_EQ(loads.live.size(), 2u);
    }

    INSTANTIATE_TEST_SUITE_P(RecoverableErrors,
                             HipSolutionAdapterRecoveryTest,
                             ::testing::Values(hipErrorLaunchFailure, hipErrorNoBinaryForGpu));

} // namespace
