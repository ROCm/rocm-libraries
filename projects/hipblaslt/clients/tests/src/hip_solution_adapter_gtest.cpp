// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <gtest/gtest.h>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>

#ifdef TENSILELITE_ENABLE_LOCAL_KERNELS
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>
#endif

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

#ifdef TENSILELITE_ENABLE_LOCAL_KERNELS
    using TensileLite::hip::SolutionAdapter;

    constexpr char local[] = "/payload/local.co";

    // A local code object that does not load (built for another architecture,
    // say) is caller input, not memory pressure: the shipped modules stay.
    TEST_P(HipSolutionAdapterRecoveryTest, smoke_LocalCodeObjectFailureDoesNotEnterRecovery)
    {
        expectLoad(helper);
        ASSERT_EQ(adapter->initializeLazyLoading("gfx942", "/library"), hipSuccess);
        expectLoad(primary);
        ASSERT_EQ(adapter->loadCodeObjectFile(primary), hipSuccess);
        expectLoad(local, GetParam());
        EXPECT_EQ(adapter->loadLocalCodeObject(local), GetParam());
        EXPECT_EQ(loads.unloads, 0u);
        EXPECT_EQ(loads.live.size(), 2u);
        EXPECT_EQ(adapter->localCodeObjectCount(), 0u);
        EXPECT_TRUE(adapter->FindCodeObject("primary.co")) << "the shipped module is still loaded";
    }

    TEST_F(HipSolutionAdapterTest, smoke_LocalCodeObjectLoadsOncePerPath)
    {
        expectLoad(local);
        ASSERT_EQ(adapter->loadLocalCodeObject(local), hipSuccess);
        EXPECT_EQ(adapter->loadLocalCodeObject(local), hipSuccess);
        EXPECT_TRUE(adapter->FindCodeObject(local));
        EXPECT_EQ(adapter->localCodeObjectCount(), 1u);
    }

    // What a device that did not register a payload does the first time it
    // runs one of its kernels.
    TEST_F(HipSolutionAdapterTest, smoke_FindCodeObjectLoadsLocalCodeObjectOnFirstUse)
    {
        expectLoad(local);
        EXPECT_FALSE(adapter->FindCodeObject(local));
        EXPECT_TRUE(adapter->FindCodeObject(local));
        EXPECT_EQ(adapter->localCodeObjectCount(), 1u);
    }

    TEST_F(HipSolutionAdapterTest, smoke_FailedLocalCodeObjectIsRetried)
    {
        expectLoad(local, hipErrorNoBinaryForGpu);
        EXPECT_FALSE(adapter->FindCodeObject(local));
        expectLoad(local);
        EXPECT_FALSE(adapter->FindCodeObject(local));
        EXPECT_EQ(adapter->localCodeObjectCount(), 1u);
    }

    TEST_F(HipSolutionAdapterTest, smoke_OnlyAbsolutePathsAreLocalCodeObjects)
    {
        EXPECT_TRUE(SolutionAdapter::isLocalCodeObject(local));
        EXPECT_FALSE(SolutionAdapter::isLocalCodeObject("primary.co"));
        EXPECT_FALSE(SolutionAdapter::isLocalCodeObject("payload/local.co"));
        EXPECT_FALSE(SolutionAdapter::isLocalCodeObject(""));
    }

    // hipModuleLoad can crash inside the runtime on such a file.
    TEST_F(HipSolutionAdapterTest, smoke_TruncatedLocalCodeObjectIsNotLoaded)
    {
        auto path = std::filesystem::temp_directory_path()
                    / ("hipblaslt-truncated-" + std::to_string(getpid()) + ".co");

        // An ELF64 header whose section table lies past the end of the file.
        std::vector<unsigned char> elf(64, 0);
        elf[0]    = 0x7f;
        elf[1]    = 'E';
        elf[2]    = 'L';
        elf[3]    = 'F';
        elf[4]    = 2;
        elf[5]    = 1;
        elf[0x29] = 0x10; // e_shoff = 4096
        elf[0x3A] = 64; // e_shentsize
        elf[0x3C] = 4; // e_shnum

        // A compressed offload bundle, the shipped format, that records a
        // larger size than the file has.
        std::vector<unsigned char> ccob(64, 0);
        ccob[0] = 'C';
        ccob[1] = 'C';
        ccob[2] = 'O';
        ccob[3] = 'B';
        ccob[4] = 3; // version
        ccob[6] = 1; // method
        ccob[9] = 0x10; // total size = 4096

        for(auto const* bytes : {&elf, &ccob})
        {
            std::ofstream(path, std::ios::binary)
                .write(reinterpret_cast<char const*>(bytes->data()), bytes->size());
            EXPECT_EQ(adapter->loadLocalCodeObject(path.string()), hipErrorInvalidImage);
            EXPECT_EQ(adapter->localCodeObjectCount(), 0u);
        }
        std::filesystem::remove(path);
    }

    std::atomic<int> localLoads{0};
    std::atomic<int> localUnloads{0};

    hipError_t slowLocalLoad(hipModule_t* module, const char*)
    {
        int n = ++localLoads;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        *module = reinterpret_cast<hipModule_t>(static_cast<uintptr_t>(n));
        return hipSuccess;
    }

    hipError_t throwingLocalLoad(hipModule_t*, const char*)
    {
        throw std::length_error("basic_string::_M_create");
    }

    hipError_t countLocalUnload(hipModule_t)
    {
        ++localUnloads;
        return hipSuccess;
    }

    hipError_t noLastError()
    {
        return hipSuccess;
    }

    TEST(HipSolutionAdapterLocalCodeObject, smoke_ConcurrentFirstUseLoadsOnce)
    {
        localLoads   = 0;
        localUnloads = 0;
        {
            SolutionAdapter adapter(
                SolutionAdapter::ModuleApi{slowLocalLoad, countLocalUnload, noLastError});
            std::vector<std::thread> threads;
            for(int i = 0; i < 8; i++)
                threads.emplace_back([&]() { adapter.FindCodeObject(local); });
            for(auto& thread : threads)
                thread.join();
            EXPECT_EQ(localLoads.load(), 1);
            EXPECT_EQ(adapter.localCodeObjectCount(), 1u);
        }
        EXPECT_EQ(localUnloads.load(), 1) << "the adapter unloads the local module";
    }

    TEST(HipSolutionAdapterLocalCodeObject, smoke_LoadThatThrowsIsAnError)
    {
        SolutionAdapter adapter(
            SolutionAdapter::ModuleApi{throwingLocalLoad, countLocalUnload, noLastError});
        EXPECT_EQ(adapter.loadLocalCodeObject(local), hipErrorInvalidImage);
        EXPECT_EQ(adapter.localCodeObjectCount(), 0u);
    }
#endif

    INSTANTIATE_TEST_SUITE_P(RecoverableErrors,
                             HipSolutionAdapterRecoveryTest,
                             ::testing::Values(hipErrorLaunchFailure, hipErrorNoBinaryForGpu));

} // namespace
