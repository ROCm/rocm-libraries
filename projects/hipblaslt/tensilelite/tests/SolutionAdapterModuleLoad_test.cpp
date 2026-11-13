// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// How hip::SolutionAdapter loads code objects from files. On Windows a path of
// MAX_PATH characters or more cannot be opened by hipModuleLoad, so the
// adapter reads such a file itself and loads it with hipModuleLoadData. These
// tests reach that path on any platform by giving the adapter a resolvePath
// that sends it to a different file, the way longPathSafe does on Windows.
//
// The HIP calls are fakes, so no GPU is needed. Like the HIP runtime, the fake
// hipModuleLoadData keeps the caller's pointer rather than a copy, and checks
// the image again when the module is unloaded.

#include <Tensile/hip/HipSolutionAdapter.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

// The allocation-failure tests replace the global operator new, which the
// sanitizers also replace, so they are left out of sanitizer builds.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define TENSILELITE_SANITIZED_ALLOCATOR 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) \
    || __has_feature(memory_sanitizer)
#define TENSILELITE_SANITIZED_ALLOCATOR 1
#endif
#endif

namespace
{
    // When 0 or more, the number of allocations to let through before one
    // fails with std::bad_alloc. -1 lets every allocation through. The
    // replacement applies to the whole test program, and is a pass-through
    // until a test arms it.
    std::atomic<long> allocationsBeforeFailure{-1};
}

#ifndef TENSILELITE_SANITIZED_ALLOCATOR
void* operator new(std::size_t size)
{
    if(allocationsBeforeFailure.load(std::memory_order_relaxed) >= 0
       && allocationsBeforeFailure.fetch_sub(1, std::memory_order_relaxed) == 0)
        throw std::bad_alloc();
    if(void* p = std::malloc(size == 0 ? 1 : size))
        return p;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size)
{
    return ::operator new(size);
}

void operator delete(void* p) noexcept
{
    std::free(p);
}

void operator delete[](void* p) noexcept
{
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept
{
    std::free(p);
}

void operator delete[](void* p, std::size_t) noexcept
{
    std::free(p);
}
#endif

namespace fs = std::filesystem;

using TensileLite::hip::SolutionAdapter;

namespace
{
    const char codeObjectBytes[] = "a stand-in for a code object";

    // Opens `path` + ".real", so a file the adapter is asked for by its
    // original name can only be reached through the redirected path.
    fs::path redirect(std::string const& path)
    {
        return fs::path(path + ".real");
    }

    struct FakeModules
    {
        std::map<hipModule_t, std::unique_ptr<int>> live;
        std::map<hipModule_t, const char*>          images;
        unsigned                                    pathLoads            = 0;
        unsigned                                    dataLoads            = 0;
        unsigned                                    modulesFromData      = 0;
        unsigned                                    imagesIntactAtUnload = 0;
        hipError_t                                  failNextDataLoad     = hipSuccess;
        hipError_t                                  lastError            = hipSuccess;
    };

    FakeModules* fake = nullptr;

    // The HIP calls are C functions that never throw, so the fakes standing in
    // for them pause any armed allocation failure while they run.
    struct NoInjectedFailure
    {
        long saved = allocationsBeforeFailure.exchange(-1);
        ~NoInjectedFailure()
        {
            allocationsBeforeFailure = saved;
        }
    };

    bool holdsCodeObject(const char* image)
    {
        return std::memcmp(image, codeObjectBytes, sizeof(codeObjectBytes)) == 0;
    }

    hipModule_t newModule()
    {
        auto storage = std::make_unique<int>(0);
        auto module  = reinterpret_cast<hipModule_t>(storage.get());
        fake->live.emplace(module, std::move(storage));
        return module;
    }

    // Opens the path it is given, as the runtime does.
    hipError_t fakeLoad(hipModule_t* module, const char* path)
    {
        NoInjectedFailure noFailure;
        ++fake->pathLoads;
        std::ifstream file(path, std::ios::binary);
        if(!file)
        {
            fake->lastError = hipErrorFileNotFound;
            return hipErrorFileNotFound;
        }
        *module = newModule();
        return hipSuccess;
    }

    hipError_t fakeLoadData(hipModule_t* module, const void* image)
    {
        NoInjectedFailure noFailure;
        ++fake->dataLoads;
        if(fake->failNextDataLoad != hipSuccess)
        {
            fake->lastError = std::exchange(fake->failNextDataLoad, hipSuccess);
            return fake->lastError;
        }
        auto const* bytes = static_cast<const char*>(image);
        EXPECT_TRUE(holdsCodeObject(bytes)) << "the image loaded is not the file's contents";
        *module               = newModule();
        fake->images[*module] = bytes;
        ++fake->modulesFromData;
        return hipSuccess;
    }

    hipError_t fakeUnload(hipModule_t module)
    {
        NoInjectedFailure noFailure;
        auto              image = fake->images.find(module);
        if(image != fake->images.end())
        {
            if(holdsCodeObject(image->second))
                ++fake->imagesIntactAtUnload;
            fake->images.erase(image);
        }
        EXPECT_EQ(fake->live.erase(module), 1u) << "double or unowned module unload";
        return hipSuccess;
    }

    hipError_t fakeGetLastError()
    {
        NoInjectedFailure noFailure;
        return std::exchange(fake->lastError, hipSuccess);
    }

    class SolutionAdapterModuleLoad : public ::testing::Test
    {
    protected:
        fs::path    dir;
        FakeModules modules;
        // Same-sized allocations made while modules are loaded, so a freed
        // image would most likely be reused and overwritten before the unload
        // check reads it. AddressSanitizer reports the read either way.
        std::vector<std::vector<char>> scribbles;

        void SetUp() override
        {
            dir = fs::temp_directory_path()
                  / ("tensilelite-module-load-" + std::to_string(getpid()) + "-"
                     + ::testing::UnitTest::GetInstance()->current_test_info()->name());
            fs::create_directories(dir);
            fake = &modules;
        }

        void TearDown() override
        {
            EXPECT_TRUE(modules.live.empty()) << "the adapter leaked module handles";
            fake = nullptr;
            std::error_code ec;
            fs::remove_all(dir, ec);
        }

        std::string writeCodeObject(std::string const& name)
        {
            std::ofstream out(dir / name, std::ios::binary);
            out.write(codeObjectBytes, sizeof(codeObjectBytes));
            EXPECT_TRUE(out.good()) << "cannot write " << (dir / name).string();
            return (dir / name).string();
        }

        std::string path(std::string const& name) const
        {
            return (dir / name).string();
        }

        void scribble()
        {
            for(int i = 0; i < 64; ++i)
                scribbles.emplace_back(sizeof(codeObjectBytes), '#');
        }

        static SolutionAdapter::ModuleApi api(bool redirected)
        {
            SolutionAdapter::ModuleApi api{fakeLoad, fakeUnload, fakeGetLastError, fakeLoadData};
            if(redirected)
                api.resolvePath = redirect;
            return api;
        }

        // Runs `load` on a fresh adapter with its first allocation failing,
        // then its second, and so on, until a run gets through without a
        // failure. `prepare` runs first with every allocation let through.
        // After each run, every module the fake HIP loaded from memory must
        // have been unloaded, with its image still intact at that point.
        template <typename Prepare, typename Load>
        void sweepAllocationFailures(Prepare prepare, Load load)
        {
#ifdef TENSILELITE_SANITIZED_ALLOCATOR
            GTEST_SKIP() << "the sanitizer replaces operator new";
#endif
            for(long n = 0; n < 10000; ++n)
            {
                FakeModules attempt;
                fake        = &attempt;
                bool failed = false;
                {
                    SolutionAdapter adapter(api(true));
                    prepare(adapter);
                    allocationsBeforeFailure = n;
                    try
                    {
                        load(adapter);
                    }
                    catch(std::bad_alloc const&)
                    {
                    }
                    failed = allocationsBeforeFailure.exchange(-1) == -1;
                }
                fake = &modules;

                EXPECT_TRUE(attempt.live.empty())
                    << "with allocation " << n << " failing, a module was never unloaded";
                EXPECT_EQ(attempt.imagesIntactAtUnload, attempt.modulesFromData)
                    << "with allocation " << n
                    << " failing, an image was released before its module was unloaded";
                if(!failed || HasFailure())
                    return;
            }
            FAIL() << "the load never ran without reaching an injected failure";
        }
    };
} // namespace

TEST_F(SolutionAdapterModuleLoad, UnchangedPathGoesToModuleLoad)
{
    auto const file = writeCodeObject("plain.co");
    {
        SolutionAdapter adapter(api(false));
        EXPECT_EQ(adapter.loadCodeObjectFile(file), hipSuccess);
    }
    EXPECT_EQ(modules.pathLoads, 1u);
    EXPECT_EQ(modules.dataLoads, 0u);
}

TEST_F(SolutionAdapterModuleLoad, RedirectedPathLoadsFromMemoryAndKeepsTheImage)
{
    writeCodeObject("library.co.real");
    {
        SolutionAdapter adapter(api(true));
        EXPECT_EQ(adapter.loadCodeObjectFile(path("library.co")), hipSuccess);
        scribble();
    }
    EXPECT_EQ(modules.pathLoads, 0u) << "the runtime was handed the original path";
    EXPECT_EQ(modules.dataLoads, 1u);
    EXPECT_EQ(modules.imagesIntactAtUnload, 1u)
        << "the image passed to hipModuleLoadData was released before its module";
}

TEST_F(SolutionAdapterModuleLoad, RedirectedPathReportsAMissingOrEmptyFile)
{
    std::ofstream(dir / "empty.co.real").close();
    SolutionAdapter adapter(api(true));

    EXPECT_EQ(adapter.loadCodeObjectFile(path("missing.co")), hipErrorFileNotFound);
    EXPECT_EQ(adapter.loadCodeObjectFile(path("empty.co")), hipErrorInvalidImage);
    EXPECT_EQ(modules.dataLoads, 0u);
    EXPECT_EQ(modules.pathLoads, 0u);
}

TEST_F(SolutionAdapterModuleLoad, RotationCopiesKeepTheirImages)
{
    writeCodeObject("library.co.real");
    {
        SolutionAdapter adapter(api(true));
        ASSERT_EQ(adapter.loadCodeObjectFile(path("library.co")), hipSuccess);
        ASSERT_EQ(adapter.loadCodeObjectFileExtraCopies(path("library.co"), 2), hipSuccess);
        scribble();
    }
    EXPECT_EQ(modules.dataLoads, 3u);
    EXPECT_EQ(modules.imagesIntactAtUnload, 3u)
        << "an image was released before its module was unloaded";
}

// A load that fails with hipErrorLaunchFailure makes the adapter unload every
// module, reload the helper kernels, and retry. Each image must last until its
// own module is unloaded, there and when the adapter is destroyed.
TEST_F(SolutionAdapterModuleLoad, RecoveryReleasesImagesOnlyAfterUnload)
{
    writeCodeObject("first.co.real");
    writeCodeObject("second.co.real");
    writeCodeObject("Kernels.so-000-gfx942.hsaco.real");
    {
        SolutionAdapter adapter(api(true));
        adapter.setLazyLoadingContext("gfx942", dir.string());
        ASSERT_EQ(adapter.loadCodeObjectFile(path("first.co")), hipSuccess);
        scribble();

        modules.failNextDataLoad = hipErrorLaunchFailure;
        EXPECT_EQ(adapter.loadCodeObjectFile(path("second.co")), hipSuccess);
        EXPECT_EQ(modules.imagesIntactAtUnload, 1u)
            << "recovery released an image before unloading its module";
        EXPECT_EQ(modules.live.size(), 2u) << "expected the helper and the retried module";
        scribble();
    }
    // first.co, then the failed attempt, the helper, and the retry.
    EXPECT_EQ(modules.dataLoads, 4u);
    EXPECT_EQ(modules.imagesIntactAtUnload, 3u);
}

// An allocation can fail after hipModuleLoadData has succeeded, while the
// adapter records the module and its image. The module must still be
// unloaded, and the image kept until it is. Repeating with up to 20 modules
// already loaded makes the image map's insertion grow its hash table.
constexpr int kMaxPreloaded = 20;

TEST_F(SolutionAdapterModuleLoad, AllocationFailureWhileRecordingALoad)
{
    writeCodeObject("library.co.real");
    for(int preloaded = 0; preloaded <= kMaxPreloaded && !HasFailure(); ++preloaded)
    {
        SCOPED_TRACE("modules already loaded: " + std::to_string(preloaded));
        sweepAllocationFailures(
            [&](SolutionAdapter& adapter) {
                for(int i = 0; i < preloaded; ++i)
                    ASSERT_EQ(adapter.loadCodeObjectFile(path("library.co")), hipSuccess);
            },
            [&](SolutionAdapter& adapter) { adapter.loadCodeObjectFile(path("library.co")); });
    }
}

TEST_F(SolutionAdapterModuleLoad, AllocationFailureWhileRecordingARotationCopy)
{
    writeCodeObject("library.co.real");
    for(int preloaded = 0; preloaded <= kMaxPreloaded && !HasFailure(); ++preloaded)
    {
        SCOPED_TRACE("modules already loaded: " + std::to_string(preloaded));
        sweepAllocationFailures(
            [&](SolutionAdapter& adapter) {
                for(int i = 0; i < preloaded; ++i)
                    ASSERT_EQ(adapter.loadCodeObjectFile(path("library.co")), hipSuccess);
            },
            [&](SolutionAdapter& adapter) {
                adapter.loadCodeObjectFileExtraCopies(path("library.co"), 1);
            });
    }
}

#ifdef TENSILELITE_ENABLE_LOCAL_KERNELS
TEST_F(SolutionAdapterModuleLoad, AllocationFailureWhileRecordingALocalCodeObject)
{
    writeCodeObject("local.co.real");
    for(int i = 0; i < kMaxPreloaded; ++i)
        writeCodeObject("preloaded" + std::to_string(i) + ".co.real");
    for(int preloaded = 0; preloaded <= kMaxPreloaded && !HasFailure(); ++preloaded)
    {
        SCOPED_TRACE("modules already loaded: " + std::to_string(preloaded));
        sweepAllocationFailures(
            [&](SolutionAdapter& adapter) {
                for(int i = 0; i < preloaded; ++i)
                    ASSERT_EQ(
                        adapter.loadLocalCodeObject(path("preloaded" + std::to_string(i) + ".co")),
                        hipSuccess);
            },
            [&](SolutionAdapter& adapter) { adapter.loadLocalCodeObject(path("local.co")); });
    }
}

TEST_F(SolutionAdapterModuleLoad, LocalCodeObjectKeepsItsImage)
{
    writeCodeObject("local.co.real");
    {
        SolutionAdapter adapter(api(true));
        EXPECT_EQ(adapter.loadLocalCodeObject(path("local.co")), hipSuccess);
        scribble();
    }
    EXPECT_EQ(modules.dataLoads, 1u);
    EXPECT_EQ(modules.imagesIntactAtUnload, 1u)
        << "the image passed to hipModuleLoadData was released before its module";
}
#endif
