// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <gtest/gtest.h>

#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

extern "C" void hipblaslt_debug_reload();

#ifdef _WIN32
static int setenv(const char* name, const char* value, int overwrite)
{
    return _putenv_s(name, value);
}
static int unsetenv(const char* name)
{
    return _putenv_s(name, "");
}
#endif

#ifdef TENSILELITE_ENABLE_LOCAL_KERNELS
#include "local_kernel_test_common.hpp"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
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

    constexpr char otherLocal[] = "/payload/other.co";

    std::atomic<bool> holdLocalLoad{false};
    std::atomic<bool> localLoadStarted{false};

    // Loads of `local` wait until holdLocalLoad is cleared.
    hipError_t heldLocalLoad(hipModule_t* module, const char* path)
    {
        if(std::string(path) == local)
        {
            localLoadStarted = true;
            while(holdLocalLoad.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        *module = reinterpret_cast<hipModule_t>(static_cast<uintptr_t>(++localLoads));
        return hipSuccess;
    }

    TEST(HipSolutionAdapterLocalCodeObject, smoke_LoadsOfDifferentPathsDoNotWait)
    {
        localLoads       = 0;
        localUnloads     = 0;
        holdLocalLoad    = true;
        localLoadStarted = false;
        {
            SolutionAdapter adapter(
                SolutionAdapter::ModuleApi{heldLocalLoad, countLocalUnload, noLastError});
            std::thread slow([&]() { EXPECT_EQ(adapter.loadLocalCodeObject(local), hipSuccess); });
            while(!localLoadStarted.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

            auto other = std::async(std::launch::async,
                                    [&]() { return adapter.loadLocalCodeObject(otherLocal); });
            EXPECT_EQ(other.wait_for(std::chrono::seconds(5)), std::future_status::ready)
                << "the load of another path waited for the slow one";

            holdLocalLoad = false;
            slow.join();
            EXPECT_EQ(other.get(), hipSuccess);
            EXPECT_EQ(adapter.localCodeObjectCount(), 2u);
        }
        EXPECT_EQ(localUnloads.load(), 2);
    }

    // The check that refuses truncated files accepts every code object this
    // build ships, whatever its layout.
    TEST_F(HipSolutionAdapterTest, smoke_ShippedCodeObjectsAreNotRefused)
    {
        std::vector<std::string> files;
        for(auto const& root : local_kernel_test::libraryRoots())
        {
            std::error_code ec;
            for(auto const& entry : std::filesystem::recursive_directory_iterator(root, ec))
            {
                auto extension = entry.path().extension();
                if(entry.is_regular_file() && (extension == ".co" || extension == ".hsaco"))
                    files.push_back(std::filesystem::absolute(entry.path()).string());
            }
            if(!files.empty())
                break;
        }
        ASSERT_FALSE(files.empty()) << "no code objects in this build's library";

        std::map<std::string, int> layouts;
        for(auto const& file : files)
        {
            char magic[8] = {};
            std::ifstream(file, std::ios::binary).read(magic, sizeof(magic));
            std::string head(magic, sizeof(magic));
            layouts[head.rfind("CCOB", 0) == 0       ? "compressed offload bundle"
                    : head.rfind("__CLANG_", 0) == 0 ? "offload bundle"
                    : head.compare(1, 3, "ELF") == 0 ? "ELF"
                                                     : "other"]++;

            expectLoad(file.c_str());
            EXPECT_EQ(adapter->loadLocalCodeObject(file), hipSuccess) << file;
        }
        for(auto const& [layout, count] : layouts)
            std::printf("%d %s file(s)\n", count, layout.c_str());
    }
#endif

    INSTANTIATE_TEST_SUITE_P(RecoverableErrors,
                             HipSolutionAdapterRecoveryTest,
                             ::testing::Values(hipErrorLaunchFailure, hipErrorNoBinaryForGpu));

    // ----------------------------------------------------------------------
    // Whose failure an unrecordable launch event is.
    //
    // The events launchKernel takes are recorded around a production dispatch, and
    // online tuning times the caller's own hipblasLtMatmul with them. By default an
    // event's record error is the launch's error; EventRecording::BestEffort says
    // the caller would rather lose the measurement than fail the GEMM.
    //
    // Building an unrecordable event takes two devices, so the suite carries the
    // multi_gpu token and runs in the multi_gpu tier. See
    // clients/tests/test_categories.yaml.
    // ----------------------------------------------------------------------

    class HipSolutionAdapterEventTest_multi_gpu : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(hipGetDeviceCount(&m_devices) != hipSuccess || m_devices < 2)
                GTEST_SKIP() << "needs two HIP devices to build an unrecordable event";

            // The launch has to reach the event records without a module or a kernel:
            // TENSILE_DB2 bit 0 skips the kernel launch. The adapter caches it at
            // construction, hence the local adapter in each case, and it lives in
            // libtensilelite-host, so the reload goes through the library hook rather than
            // this binary's Debug::Instance().
            const char* const existing = getenv("TENSILE_DB2");
            m_savedDb2 = existing ? std::optional<std::string>(existing) : std::nullopt;
            setenv("TENSILE_DB2", "0x1", 1);
            hipblaslt_debug_reload();

            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            ASSERT_EQ(hipEventCreateWithFlags(&m_onDeviceZero, hipEventDefault), hipSuccess);

            // Recording an event on another device's stream is rejected with
            // hipErrorInvalidResourceHandle. A destroyed handle would fault instead.
            ASSERT_EQ(hipSetDevice(1), hipSuccess);
            ASSERT_EQ(hipStreamCreate(&m_streamOnDeviceOne), hipSuccess);
            ASSERT_NE(hipEventRecord(m_onDeviceZero, m_streamOnDeviceOne), hipSuccess)
                << "this device pair records across devices, so nothing here can fail";
            static_cast<void>(hipGetLastError());
        }

        void TearDown() override
        {
            if(m_streamOnDeviceOne)
                static_cast<void>(hipStreamDestroy(m_streamOnDeviceOne));
            if(m_onDeviceZero)
            {
                static_cast<void>(hipSetDevice(0));
                static_cast<void>(hipEventDestroy(m_onDeviceZero));
            }

            if(m_savedDb2)
                setenv("TENSILE_DB2", m_savedDb2->c_str(), 1);
            else
                unsetenv("TENSILE_DB2");
            hipblaslt_debug_reload();

            static_cast<void>(hipSetDevice(0));
        }

        // One kernel with no code object, so the launch is all event handling.
        std::vector<TensileLite::KernelInvocation> kernels() const
        {
            return std::vector<TensileLite::KernelInvocation>(1);
        }

        int                        m_devices           = 0;
        hipEvent_t                 m_onDeviceZero      = nullptr;
        hipStream_t                m_streamOnDeviceOne = nullptr;
        std::optional<std::string> m_savedDb2;
    };

    TEST_F(HipSolutionAdapterEventTest_multi_gpu, AnUnrecordableEventFailsTheLaunchByDefault)
    {
        TensileLite::hip::SolutionAdapter adapter;

        EXPECT_EQ(adapter.launchKernels(kernels(), m_streamOnDeviceOne, m_onDeviceZero, nullptr),
                  hipErrorInvalidResourceHandle);
        EXPECT_EQ(adapter.launchKernels(kernels(), m_streamOnDeviceOne, nullptr, m_onDeviceZero),
                  hipErrorInvalidResourceHandle);
        static_cast<void>(hipGetLastError());
    }

    TEST_F(HipSolutionAdapterEventTest_multi_gpu, BestEffortEventsCostTheMeasurementNotTheLaunch)
    {
        TensileLite::hip::SolutionAdapter adapter;

        EXPECT_EQ(adapter.launchKernels(kernels(),
                                        m_streamOnDeviceOne,
                                        m_onDeviceZero,
                                        m_onDeviceZero,
                                        /*isKernelLoaded=*/false,
                                        TensileLite::hip::EventRecording::BestEffort),
                  hipSuccess)
            << "a measurement event failure was returned as the launch's status";

        // And it is not left behind for the caller to trip over later.
        EXPECT_EQ(hipGetLastError(), hipSuccess)
            << "the swallowed event error was left in HIP's last-error state";

        // But an error the caller already had, and has not read, is the caller's.
        ASSERT_NE(hipSetDevice(m_devices), hipSuccess) << "an out-of-range device was accepted";
        ASSERT_NE(hipPeekAtLastError(), hipSuccess) << "the caller's error did not stick";

        EXPECT_EQ(adapter.launchKernels(kernels(),
                                        m_streamOnDeviceOne,
                                        m_onDeviceZero,
                                        m_onDeviceZero,
                                        /*isKernelLoaded=*/false,
                                        TensileLite::hip::EventRecording::BestEffort),
                  hipSuccess);

        EXPECT_NE(hipGetLastError(), hipSuccess)
            << "a swallowed event error cleared an error the caller had not read";
    }

    // A swallowed start record leaves the stop unrecorded too. A reused pair whose
    // start holds an older timestamp would otherwise read a plausible positive
    // duration against a fresh stop. Checked for one kernel, and for several where
    // the start goes with the first kernel and the stop with the last.
    TEST_F(HipSolutionAdapterEventTest_multi_gpu, ASwallowedStartRecordLeavesTheStopUnrecorded)
    {
        TensileLite::hip::SolutionAdapter adapter;

        for(size_t count : {size_t(1), size_t(3)})
        {
            // On device one, current since SetUp, so it records on the stream
            // the launch uses: only the start can fail.
            hipEvent_t stop = nullptr;
            ASSERT_EQ(hipEventCreateWithFlags(&stop, hipEventDefault), hipSuccess);

            EXPECT_EQ(adapter.launchKernels(std::vector<TensileLite::KernelInvocation>(count),
                                            m_streamOnDeviceOne,
                                            m_onDeviceZero,
                                            stop,
                                            /*isKernelLoaded=*/false,
                                            TensileLite::hip::EventRecording::BestEffort),
                      hipSuccess)
                << count << " kernels";
            EXPECT_EQ(hipStreamSynchronize(m_streamOnDeviceOne), hipSuccess);

            // An event never recorded has no timestamp to read.
            float milliseconds = 0.0f;
            EXPECT_NE(hipEventElapsedTime(&milliseconds, stop, stop), hipSuccess)
                << "with " << count << " kernels the stop was recorded although the start was not";
            static_cast<void>(hipGetLastError());

            static_cast<void>(hipEventDestroy(stop));
        }
    }

} // namespace
