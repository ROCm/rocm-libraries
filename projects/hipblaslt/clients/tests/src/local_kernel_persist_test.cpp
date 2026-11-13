// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end tests of the local kernel library named by
// HIPBLASLT_LOCAL_KERNEL_LIBRARY. Each worker is a separate process, as a
// process that saves kernels and a later one that loads them would be.

#include <gtest/gtest.h>

#include "local_kernel_test_common.hpp"

#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <map>
#include <set>
#include <sstream>
#include <thread>

namespace
{
    using namespace local_kernel_test;
    namespace fs = std::filesystem;

    constexpr char TEST_M_ENV[]          = "HIPBLASLT_LOCAL_KERNEL_TEST_M";
    constexpr char TEST_SELECTABLE_ENV[] = "HIPBLASLT_LOCAL_KERNEL_TEST_SELECTABLE";

    constexpr char SAVE_WORKER[]           = "LocalKernelPersist.DISABLED_SaveWorker";
    constexpr char LOAD_WORKER[]           = "LocalKernelPersist.DISABLED_LoadWorker";
    constexpr char NOTHING_LOADED_WORKER[] = "LocalKernelPersist.DISABLED_NothingLoadedWorker";

    struct TempDir
    {
        fs::path path;

        TempDir()
        {
            std::string pattern = (fs::temp_directory_path() / "hipblaslt-ukr-XXXXXX").string();
            if(char* made = ::mkdtemp(pattern.data()))
                path = made;
        }
        ~TempDir()
        {
            std::error_code ec;
            if(!path.empty())
                fs::remove_all(path, ec);
        }
    };

    std::string env(char const* name, std::string const& value)
    {
        return std::string(name) + "=" + value;
    }

    std::string libraryEnv(fs::path const& library)
    {
        return env(LIBRARY_ENV, library.string());
    }

    int64_t envInt(char const* name, int64_t fallback)
    {
        char const* value = std::getenv(name);
        return value ? std::strtoll(value, nullptr, 10) : fallback;
    }

    // The current device's scope, e.g. gfx942_CU304, gfx950_CU256_ID75a3 or
    // gfx1250v0_CU<n>. TensileLite reads a PCI chip ID on gfx950 only, and
    // gfx1250 A0 is a target of its own, as for the shipped library.
    std::string deviceScope()
    {
        int             device = 0, chip = 0;
        hipDeviceProp_t prop{};
        std::string     target = deviceArch();
        if(target == "gfx1250" && hipGetDevice(&device) == hipSuccess
           && hipGetDeviceProperties(&prop, device) == hipSuccess && prop.asicRevision == 0)
            target += "v0";
        std::string scope = target + "_CU" + std::to_string(deviceCuCount());
        if(deviceArch() == "gfx950" && hipGetDevice(&device) == hipSuccess
           && hipDeviceGetAttribute(&chip, hipDeviceAttributePciChipId, device) == hipSuccess)
        {
            char id[16];
            std::snprintf(id, sizeof(id), "_ID%04x", unsigned(chip));
            scope += id;
        }
        return scope;
    }

    // Waits up to a minute for another process to create `path`.
    bool waitForFile(fs::path const& path)
    {
        for(int i = 0; i < 6000; ++i)
        {
            if(fs::exists(path))
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }

    fs::path scopePath(fs::path const& library)
    {
        return library / deviceScope();
    }

    // This build's journal for a scope: the only one a test's library holds.
    fs::path journalPath(fs::path const& library, std::string const& scope = deviceScope())
    {
        std::error_code ec;
        for(auto const& entry : fs::directory_iterator(library / scope, ec))
            if(entry.path().filename().string().rfind("journal-", 0) == 0)
                return entry.path();
        return library / scope / "journal-not-found";
    }

    // The journal's lines, including a final one without a newline.
    std::vector<std::string> journalLines(fs::path const&    library,
                                          std::string const& scope = deviceScope())
    {
        std::ifstream            in(journalPath(library, scope));
        std::vector<std::string> lines;
        for(std::string line; std::getline(in, line);)
            lines.push_back(line);
        return lines;
    }

    size_t countRecords(std::vector<std::string> const& lines, char kind)
    {
        return std::count_if(lines.begin(), lines.end(), [kind](std::string const& line) {
            return line.size() > 2 && line[0] == kind && line[1] == ' ';
        });
    }

    size_t countFields(std::string const& line)
    {
        std::istringstream in(line);
        size_t             count = 0;
        for(std::string field; in >> field;)
            ++count;
        return count;
    }

    std::string readFile(fs::path const& path)
    {
        std::ifstream in(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }

    // The store's name for an object with these contents, for tests that save
    // objects themselves.
    std::string objectName(std::string const& bytes)
    {
        uint64_t a = 0xcbf29ce484222325ull, b = 0x9e3779b97f4a7c15ull;
        for(unsigned char c : bytes)
        {
            a = (a ^ c) * 0x100000001b3ull;
            b = (b ^ c) * 0xff51afd7ed558ccdull;
        }
        char text[33];
        std::snprintf(text,
                      sizeof(text),
                      "%016llx%016llx",
                      static_cast<unsigned long long>(a),
                      static_cast<unsigned long long>(b));
        return text;
    }

    // A journal record with the store's checksum, for tests that append records.
    std::string journalRecord(std::string const& body)
    {
        uint32_t sum = 0x811c9dc5u;
        for(unsigned char c : body)
            sum = (sum ^ c) * 0x01000193u;
        char text[9];
        std::snprintf(text, sizeof(text), "%08x", sum);
        return body + " " + text;
    }

    size_t countObjects(fs::path const& library, std::string const& extension)
    {
        size_t          count = 0;
        std::error_code ec;
        for(auto const& entry : fs::directory_iterator(library / "objects", ec))
            if(entry.path().extension() == extension)
                ++count;
        return count;
    }

    // Copies a candidate payload to `scratch`, registers the copy and deletes
    // it, so a later process can only be using the library's copy.
    Registered registerFromCopies(hipblasLtHandle_t               handle,
                                  Problem const&                  p,
                                  std::vector<std::string> const& candidates,
                                  fs::path const&                 scratch)
    {
        std::error_code ec;
        fs::create_directories(scratch, ec);

        for(auto const& stem : candidates)
        {
            const bool     compressed = fs::exists(stem + ".dat.zlib");
            const fs::path dat        = scratch / "payload.dat";
            const fs::path shard      = compressed ? fs::path(dat.string() + ".zlib") : dat;
            const fs::path co         = scratch / "payload.co";
            fs::copy_file(compressed ? stem + ".dat.zlib" : stem + ".dat",
                          shard,
                          fs::copy_options::overwrite_existing);
            fs::copy_file(stem + ".co", co, fs::copy_options::overwrite_existing);

            std::vector<int> indices(4096);
            int              numIndices = 0;
            const auto       status     = hipblasLtLocalKernelRegister(handle,
                                                             dat.c_str(),
                                                             co.c_str(),
                                                             indices.data(),
                                                             static_cast<int>(indices.size()),
                                                             &numIndices);
            fs::remove(shard, ec);
            fs::remove(co, ec);
            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                ADD_FAILURE() << "registration failed for " << stem;
                continue;
            }
            indices.resize(std::min<size_t>(numIndices, indices.size()));

            Registered r;
            r.stem  = stem;
            r.index = findUsableKernel(handle, p, indices, r.result, r.workspace);
            if(r.index >= 0)
                return r;
        }
        return {};
    }

    void counts(hipblasLtHandle_t handle, size_t* registered, size_t* selectable)
    {
        ASSERT_EQ(hipblasLtLocalKernelGetCounts(handle, registered, selectable),
                  HIPBLAS_STATUS_SUCCESS);
    }

    size_t registeredCount(hipblasLtHandle_t handle)
    {
        size_t registered = 0;
        counts(handle, &registered, nullptr);
        return registered;
    }

    // The assigned indices, or empty if registration failed.
    std::vector<int>
        registerPayload(hipblasLtHandle_t handle, std::string const& dat, std::string const& co)
    {
        std::vector<int> indices(4096);
        int              numIndices = 0;
        if(hipblasLtLocalKernelRegister(handle,
                                        dat.c_str(),
                                        co.c_str(),
                                        indices.data(),
                                        static_cast<int>(indices.size()),
                                        &numIndices)
           != HIPBLAS_STATUS_SUCCESS)
            return {};
        indices.resize(std::min<size_t>(numIndices, indices.size()));
        return indices;
    }

    // Registers payloads until two of their kernels serve `p`, and returns
    // those two, or fewer if the build's library has no more.
    std::vector<int> twoUsableKernels(hipblasLtHandle_t               handle,
                                      Problem const&                  p,
                                      std::vector<std::string> const& candidates)
    {
        std::vector<int> kernels;
        for(auto const& stem : candidates)
        {
            for(int index : registerPayload(handle, stem + ".dat", stem + ".co"))
            {
                hipblasLtMatmulHeuristicResult_t result;
                size_t                           workspace = 0;
                if(kernels.size() < 2
                   && findUsableKernel(handle, p, {index}, result, workspace) >= 0)
                    kernels.push_back(index);
            }
            if(kernels.size() == 2)
                break;
        }
        return kernels;
    }

    // The kernel the journal's last mapping at M x N x K names, as "<datHash>
    // <coHash> <payloadIndex>", or empty if there is none.
    std::string lastMapping(fs::path const& library)
    {
        std::string last;
        for(auto const& line : journalLines(library))
        {
            std::istringstream in(line);
            std::string        kind, dat, co, index;
            int64_t            m = 0, n = 0, batch = 0, k = 0;
            if(in >> kind >> dat >> co >> index >> m >> n >> batch >> k && kind == "E" && m == M
               && n == N && batch == 1 && k == K)
                last = dat + " " + co + " " + index;
        }
        return last;
    }

    // Maps M x N x K to each kernel in turn, and returns the kernel each
    // journal mapping names, keyed as lastMapping returns it.
    std::map<std::string, int> mappingNames(hipblasLtHandle_t       handle,
                                            fs::path const&         library,
                                            std::vector<int> const& kernels)
    {
        std::map<std::string, int> kernelOf;
        for(int kernel : kernels)
        {
            if(hipblasLtLocalKernelSetExactMatch(handle, kernel, M, N, 1, K)
               != HIPBLAS_STATUS_SUCCESS)
                return {};
            kernelOf[lastMapping(library)] = kernel;
        }
        return kernelOf;
    }

    // Scratch files shared by the two-code-object workers.
    fs::path twoCodeObjectsDir()
    {
        char const* library = std::getenv(LIBRARY_ENV);
        return library ? fs::path(library).parent_path() / "two-code-objects" : fs::path();
    }
} // namespace

// Registers a usable kernel from a copy of a payload, deletes the copy, and
// maps M (default 16) x N x K to the kernel.
TEST(LocalKernelPersist, DISABLED_SaveWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const int64_t m = envInt(TEST_M_ENV, M);

    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p, m), HIPBLAS_STATUS_SUCCESS);

    const fs::path scratch
        = fs::path(library).parent_path() / ("scratch-" + std::to_string(::getpid()));
    Registered r = registerFromCopies(handle, p, candidates, scratch);
    ASSERT_GE(r.index, 0);

    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, m, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, p), r.index);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

// The first heuristic query for M (default 16) x N x K returns a local kernel,
// without any local kernel call before it, and the kernel computes correctly.
TEST(LocalKernelPersist, DISABLED_LoadWorker)
{
    const int64_t m          = envInt(TEST_M_ENV, M);
    const size_t  selectable = static_cast<size_t>(envInt(TEST_SELECTABLE_ENV, 1));

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p, m), HIPBLAS_STATUS_SUCCESS);

    auto best = topResults(handle, p, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_GE(algoIndex(best[0].algo), LOCAL_INDEX_BASE)
        << "the saved mapping applies to the first query";

    size_t registered = 0, mapped = 0;
    counts(handle, &registered, &mapped);
    EXPECT_GT(registered, 0u);
    EXPECT_EQ(mapped, selectable);

    runAndCheck(handle, p, best[0].algo, best[0].workspaceSize);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, DISABLED_NothingLoadedWorker)
{
    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    const int top = topIndex(handle, p);
    EXPECT_GE(top, 0);
    EXPECT_LT(top, LOCAL_INDEX_BASE);

    size_t registered = 123, selectable = 123;
    counts(handle, &registered, &selectable);
    EXPECT_EQ(registered, 0u);
    EXPECT_EQ(selectable, 0u);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, SavedKernelsLoadInNewProcess)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);

    EXPECT_GE(countObjects(library, ".co"), 1u);
    auto lines = journalLines(library);
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(lines[0].rfind("V 1 ", 0), 0u) << "the journal starts with its version";
    const std::string version = "journal-" + std::to_string(HIPBLASLT_VERSION_MAJOR) + "."
                                + std::to_string(HIPBLASLT_VERSION_MINOR) + "."
                                + std::to_string(HIPBLASLT_VERSION_PATCH);
    EXPECT_EQ(journalPath(library).filename().string().rfind(version, 0), 0u)
        << "the journal is named for the hipBLASLt version, with or without a git stamp";
    EXPECT_GE(countRecords(lines, 'R'), 1u);
    EXPECT_EQ(countRecords(lines, 'E'), 1u);
    for(auto const& entry : fs::directory_iterator(library / "objects"))
        EXPECT_EQ(entry.path().string().find(".tmp."), std::string::npos)
            << "no temporary object left behind: " << entry.path();

    EXPECT_EQ(runWorker(LOAD_WORKER, {libraryEnv(library)}), 0)
        << "a new process loads the saved kernel; the payload it was registered from is gone";
    EXPECT_EQ(runWorker(NOTHING_LOADED_WORKER), 0) << "without the variable, nothing is loaded";
}

// Mapping a shape again to the kernel it already selects, as a tuner run again
// does, records nothing.
TEST(LocalKernelPersist, RepeatedMappingIsRecordedOnce)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    const auto first = journalLines(library);
    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    EXPECT_EQ(journalLines(library), first);
    EXPECT_EQ(countRecords(first, 'E'), 1u);

    EXPECT_EQ(runWorker(LOAD_WORKER, {libraryEnv(library)}), 0);
}

// Registering a payload already registered, or already saved, returns the
// indices it has and records nothing new. Both spellings of a compressed
// shard name the same payload.
TEST(LocalKernelPersist, DISABLED_IdempotentWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";
    const std::string stem = candidates[0];
    const std::string co   = stem + ".co";
    const std::string bare = stem + ".dat";
    const std::string file = fs::exists(bare + ".zlib") ? bare + ".zlib" : bare;

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    const size_t before = registeredCount(handle);

    const auto first = registerPayload(handle, bare, co);
    ASSERT_FALSE(first.empty());
    const size_t afterFirst = registeredCount(handle);
    EXPECT_TRUE(afterFirst == before || afterFirst == before + first.size())
        << "either newly registered or already loaded from the library";

    EXPECT_EQ(registerPayload(handle, file, co), first) << file;
    EXPECT_EQ(registeredCount(handle), afterFirst);

    size_t added = 123;
    ASSERT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(added, 0u) << "this process's own records are not registered again";
    EXPECT_EQ(registeredCount(handle), afterFirst);
    EXPECT_EQ(registerPayload(handle, bare, co), first) << "indices are unchanged by a refresh";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, RegisterIsIdempotent)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker("LocalKernelPersist.DISABLED_IdempotentWorker", {libraryEnv(library)}), 0);
    ASSERT_EQ(runWorker("LocalKernelPersist.DISABLED_IdempotentWorker", {libraryEnv(library)}), 0)
        << "a second process finds the payload already loaded";

    auto lines = journalLines(library);
    EXPECT_EQ(countRecords(lines, 'R'), 1u);
    EXPECT_EQ(countRecords(lines, 'E'), 0u);
}

TEST(LocalKernelPersist, RegisterIsIdempotentWithoutLibrary)
{
    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_IdempotentWorker"), 0);
}

// A running process sees what another saved only after a refresh.
TEST(LocalKernelPersist, DISABLED_RefreshWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    ASSERT_LT(topIndex(handle, p), LOCAL_INDEX_BASE);

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    EXPECT_LT(topIndex(handle, p), LOCAL_INDEX_BASE) << "not visible before a refresh";

    size_t added = 0;
    ASSERT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_GT(added, 0u);
    const size_t registered = registeredCount(handle);
    EXPECT_EQ(registered, added);
    const int mapped = topIndex(handle, p);
    EXPECT_GE(mapped, LOCAL_INDEX_BASE);

    ASSERT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(added, 0u) << "a second refresh finds nothing new";
    EXPECT_EQ(registeredCount(handle), registered);
    EXPECT_EQ(topIndex(handle, p), mapped);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, RefreshLoadsWhatOthersSaved)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    EXPECT_EQ(
        runWorker("LocalKernelPersist.DISABLED_RefreshWorker", {libraryEnv(tmp.path / "lib")}), 0);
}

// A mapping is applied together with what other processes saved before it,
// so this process selects what a new one would.
TEST(LocalKernelPersist, DISABLED_ReadBackWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem small, large;
    ASSERT_EQ(buildProblem(small, 16), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(buildProblem(large, 64), HIPBLAS_STATUS_SUCCESS);

    Registered r = registerUsable(handle, small, candidates);
    ASSERT_GE(r.index, 0);

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library), env(TEST_M_ENV, "64")}), 0);
    EXPECT_LT(topIndex(handle, large), LOCAL_INDEX_BASE) << "not read yet";

    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, 16, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, small), r.index);
    EXPECT_GE(topIndex(handle, large), LOCAL_INDEX_BASE)
        << "the other process's mapping, saved first, is applied too";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, MappingAppliesWhatOthersSavedFirst)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    EXPECT_EQ(
        runWorker("LocalKernelPersist.DISABLED_ReadBackWorker", {libraryEnv(tmp.path / "lib")}), 0);
}

// Two threads map M x N x K to different kernels at once, round after round.
// After each round this process selects the kernel of the journal's last
// mapping at that shape, which is the one every later process selects.
TEST(LocalKernelPersist, DISABLED_RacingMappingsWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t first, second;
    ASSERT_EQ(hipblasLtCreate(&first), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtCreate(&second), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    const auto kernels = twoUsableKernels(first, p, candidates);
    ASSERT_EQ(kernels.size(), 2u) << "no two kernels in this build's library serve the problem";
    const auto kernelOf = mappingNames(first, library, kernels);
    ASSERT_EQ(kernelOf.size(), 2u);

    for(int round = 0; round < 100; ++round)
    {
        std::atomic<int> ready{0};
        hipblasStatus_t  status[2];
        auto             map = [&](int which) {
            ++ready;
            while(ready.load() < 2)
                std::this_thread::yield();
            status[which] = hipblasLtLocalKernelSetExactMatch(
                which ? second : first, kernels[which], M, N, 1, K);
        };
        std::thread a(map, 0), b(map, 1);
        a.join();
        b.join();
        ASSERT_EQ(status[0], HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(status[1], HIPBLAS_STATUS_SUCCESS);

        auto const last = kernelOf.find(lastMapping(library));
        ASSERT_NE(last, kernelOf.end());
        ASSERT_EQ(topIndex(first, p), last->second) << "round " << round;
    }

    EXPECT_EQ(hipblasLtDestroy(second), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtDestroy(first), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, RacingMappingsSelectTheLastSaved)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_RacingMappingsWorker",
                        {libraryEnv(tmp.path / "lib")}),
              0);
}

// Refresh fails, loading nothing, when the journal cannot be read.
TEST(LocalKernelPersist, DISABLED_UnreadableRefreshWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    ASSERT_FALSE(registerPayload(handle, candidates[0] + ".dat", candidates[0] + ".co").empty());

    const fs::path journal = journalPath(library);
    const auto     lines   = journalLines(library);
    ASSERT_GE(lines.size(), 2u);

    {
        std::ofstream out(journal, std::ios::trunc);
        out << "V 1 another-build\n";
        for(size_t i = 1; i < lines.size(); ++i)
            out << lines[i] << '\n';
    }
    size_t added = 123;
    EXPECT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_INTERNAL_ERROR)
        << "written by another build";
    EXPECT_EQ(added, 0u);

    ASSERT_TRUE(fs::remove(journal));
    ASSERT_TRUE(fs::create_directory(journal));
    EXPECT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_INTERNAL_ERROR)
        << "not a readable file";

    ASSERT_TRUE(fs::remove(journal));
    EXPECT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS)
        << "a missing journal reads as empty";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, RefreshReportsUnreadableJournal)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_UnreadableRefreshWorker",
                        {libraryEnv(tmp.path / "lib")}),
              0);
}

// Two processes saving at once both land, and the journal stays well formed.
TEST(LocalKernelPersist, ConcurrentWriters)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    const pid_t small = spawnWorker(SAVE_WORKER, {libraryEnv(library), env(TEST_M_ENV, "16")});
    const pid_t large = spawnWorker(SAVE_WORKER, {libraryEnv(library), env(TEST_M_ENV, "64")});
    ASSERT_EQ(waitWorker(small), 0);
    ASSERT_EQ(waitWorker(large), 0);

    auto lines = journalLines(library);
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(lines[0].rfind("V 1 ", 0), 0u) << lines[0];
    std::set<std::string> payloads;
    for(size_t i = 1; i < lines.size(); ++i)
    {
        auto const& line = lines[i];
        if(line.rfind("R ", 0) == 0)
        {
            EXPECT_EQ(countFields(line), 5u) << line;
            EXPECT_TRUE(payloads.insert(line).second) << "payload recorded twice: " << line;
        }
        else
        {
            EXPECT_EQ(line.rfind("E ", 0), 0u) << line;
            EXPECT_EQ(countFields(line), 9u) << line;
        }
    }
    EXPECT_EQ(countRecords(lines, 'E'), 2u);

    for(char const* m : {"16", "64"})
        EXPECT_EQ(
            runWorker(LOAD_WORKER,
                      {libraryEnv(library), env(TEST_M_ENV, m), env(TEST_SELECTABLE_ENV, "2")}),
            0)
            << "M=" << m;
}

// A group- or world-writable directory is not used: registration still works
// for the process, and nothing is written.
TEST(LocalKernelPersist, DISABLED_InMemoryWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0);
    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, p), r.index);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, IgnoresWritableDirectory)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";
    ASSERT_TRUE(fs::create_directory(library));
    ASSERT_EQ(::chmod(library.c_str(), 0777), 0);

    ASSERT_EQ(runWorker("LocalKernelPersist.DISABLED_InMemoryWorker", {libraryEnv(library)}), 0);
    EXPECT_TRUE(fs::is_empty(library)) << "nothing is written to a writable directory";
    EXPECT_EQ(runWorker(NOTHING_LOADED_WORKER, {libraryEnv(library)}), 0);
}

// A directory owned by another user is not used, though its permissions would
// pass. As root, the test makes one; otherwise it uses "/", which root owns.
TEST(LocalKernelPersist, IgnoresDirectoryOwnedByAnotherUser)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());

    fs::path library = "/";
    if(::geteuid() == 0)
    {
        constexpr uid_t nobody = 65534;
        library                = tmp.path / "lib";
        ASSERT_TRUE(fs::create_directories(library / "objects"));
        for(auto const& dir : {library, library / "objects"})
        {
            ASSERT_EQ(::chown(dir.c_str(), nobody, nobody), 0) << dir;
            ASSERT_EQ(::chmod(dir.c_str(), 0755), 0) << dir;
        }
    }

    const fs::path errors = tmp.path / "stderr";
    ASSERT_EQ(runWorker("LocalKernelPersist.DISABLED_InMemoryWorker",
                        {libraryEnv(library)},
                        errors.string()),
              0);
    EXPECT_NE(readFile(errors).find(library.string() + " is owned by another user"),
              std::string::npos)
        << readFile(errors);
    if(::geteuid() == 0)
        EXPECT_TRUE(fs::is_empty(library / "objects") && !fs::exists(scopePath(library)))
            << "nothing is written to a directory owned by another user";
}

// Lines that do not parse or whose checksum does not match are skipped, and a
// partial last line left by a crashed writer is not read as a record, even
// where what remains would parse, and does not swallow the next record.
TEST(LocalKernelPersist, ToleratesDamagedJournal)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    auto lines = journalLines(library);
    ASSERT_FALSE(lines.empty());
    const std::string mapping = lines.back();
    ASSERT_EQ(mapping.rfind("E ", 0), 0u) << mapping;

    // The mapping with K cut short and no checksum, as a crash could leave
    // it, and with K changed under its checksum.
    const std::string body    = mapping.substr(0, mapping.rfind(' '));
    const std::string torn    = body.substr(0, body.size() - 2);
    const std::string changed = torn + "99" + mapping.substr(mapping.rfind(' '));
    {
        std::ofstream out(journalPath(library), std::ios::app);
        out << "Z unknown record\nR nothex nothex dat 00000000\n" << changed << "\n" << torn;
    }

    ASSERT_EQ(runWorker(LOAD_WORKER, {libraryEnv(library)}), 0)
        << "only the original mapping is selectable";
    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library), env(TEST_M_ENV, "64")}), 0);

    lines = journalLines(library);
    EXPECT_NE(std::find(lines.begin(), lines.end(), torn), lines.end())
        << "the partial line was terminated, not extended";
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(lines.back().rfind("E ", 0), 0u);
    EXPECT_EQ(countFields(lines.back()), 9u);

    EXPECT_EQ(
        runWorker(LOAD_WORKER,
                  {libraryEnv(library), env(TEST_M_ENV, "64"), env(TEST_SELECTABLE_ENV, "2")}),
        0)
        << "the terminated partial line is still not a record";
}

// A record whose objects are gone is skipped with a warning.
TEST(LocalKernelPersist, SkipsMissingObjects)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    for(auto const& entry : fs::directory_iterator(library / "objects"))
        if(entry.path().extension() == ".co")
            fs::remove(entry.path());

    EXPECT_EQ(runWorker(NOTHING_LOADED_WORKER, {libraryEnv(library)}), 0);
}

// A saved code object that no longer matches its name, cut short or changed
// in place, is skipped with a warning, and so are its mappings.
TEST(LocalKernelPersist, SkipsDamagedObjects)
{
    for(const bool truncate : {true, false})
    {
        TempDir tmp;
        ASSERT_FALSE(tmp.path.empty());
        const fs::path library = tmp.path / "lib";

        ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
        for(auto const& entry : fs::directory_iterator(library / "objects"))
        {
            if(entry.path().extension() != ".co")
                continue;
            if(truncate)
            {
                fs::resize_file(entry.path(), fs::file_size(entry.path()) / 2);
                continue;
            }
            std::fstream file(entry.path(), std::ios::in | std::ios::out | std::ios::binary);
            char         last = 0;
            file.seekg(-1, std::ios::end);
            file.get(last);
            file.seekp(-1, std::ios::end);
            file.put(static_cast<char>(~last));
            ASSERT_TRUE(file.good()) << entry.path();
        }

        EXPECT_EQ(runWorker(NOTHING_LOADED_WORKER, {libraryEnv(library)}), 0)
            << (truncate ? "cut short" : "changed in place");
    }
}

// Registers one shard with two code objects that differ only in trailing
// bytes, and maps a kernel of the second pairing. Leaves the shard's stem and
// the second code object in twoCodeObjectsDir().
TEST(LocalKernelPersist, DISABLED_TwoCodeObjectsSaveWorker)
{
    const fs::path dir = twoCodeObjectsDir();
    ASSERT_FALSE(dir.empty());
    fs::create_directories(dir);

    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    std::string      stem;
    std::vector<int> first;
    size_t           position = 0;
    for(auto const& candidate : candidates)
    {
        first = registerPayload(handle, candidate + ".dat", candidate + ".co");
        ASSERT_FALSE(first.empty()) << candidate;

        hipblasLtMatmulHeuristicResult_t result;
        size_t                           workspace = 0;
        const int usable = findUsableKernel(handle, p, first, result, workspace);
        if(usable >= 0)
        {
            stem     = candidate;
            position = std::find(first.begin(), first.end(), usable) - first.begin();
            break;
        }
    }
    ASSERT_FALSE(stem.empty()) << "no payload holds a kernel for the problem";

    const fs::path variant = dir / "variant.co";
    fs::copy_file(stem + ".co", variant, fs::copy_options::overwrite_existing);
    {
        std::ofstream out(variant, std::ios::binary | std::ios::app);
        out << std::string(64, '\0');
    }
    std::ofstream(dir / "stem.txt") << stem;

    const auto second = registerPayload(handle, stem + ".dat", variant.string());
    ASSERT_EQ(second.size(), first.size()) << "the variant code object is a second payload";
    EXPECT_NE(second, first);

    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, second[position], M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, p), second[position]);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

// After a restart both pairings are loaded, and the mapping is restored
// against the second.
TEST(LocalKernelPersist, DISABLED_TwoCodeObjectsLoadWorker)
{
    const fs::path dir = twoCodeObjectsDir();
    std::string    stem;
    std::ifstream(dir / "stem.txt") >> stem;
    ASSERT_FALSE(stem.empty());

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    const int    mapped = topIndex(handle, p);
    const size_t loaded = registeredCount(handle);
    EXPECT_GE(mapped, LOCAL_INDEX_BASE);

    const auto first  = registerPayload(handle, stem + ".dat", stem + ".co");
    const auto second = registerPayload(handle, stem + ".dat", (dir / "variant.co").string());
    ASSERT_FALSE(first.empty());
    ASSERT_EQ(second.size(), first.size());
    EXPECT_NE(second, first);
    EXPECT_EQ(registeredCount(handle), loaded) << "both pairings were loaded at startup";
    EXPECT_EQ(std::count(first.begin(), first.end(), mapped), 0);
    EXPECT_EQ(std::count(second.begin(), second.end(), mapped), 1)
        << "the mapping names the second code object";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, OneShardWithTwoCodeObjects)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(
        runWorker("LocalKernelPersist.DISABLED_TwoCodeObjectsSaveWorker", {libraryEnv(library)}),
        0);

    auto                                  lines = journalLines(library);
    std::vector<std::vector<std::string>> payloads, exacts;
    for(auto const& line : lines)
    {
        std::istringstream       in(line);
        std::vector<std::string> fields;
        for(std::string field; in >> field;)
            fields.push_back(field);
        if(line.rfind("R ", 0) == 0)
            payloads.push_back(fields);
        else if(line.rfind("E ", 0) == 0)
            exacts.push_back(fields);
    }
    ASSERT_EQ(exacts.size(), 1u);
    std::vector<std::vector<std::string>> sameShard;
    for(auto const& fields : payloads)
        if(fields[1] == exacts[0][1])
            sameShard.push_back(fields);
    ASSERT_EQ(sameShard.size(), 2u) << "the mapped shard is recorded with two code objects";
    EXPECT_NE(sameShard[0][2], sameShard[1][2]);
    EXPECT_EQ(exacts[0][2], sameShard[1][2]) << "the mapping names the second code object";

    EXPECT_EQ(
        runWorker("LocalKernelPersist.DISABLED_TwoCodeObjectsLoadWorker", {libraryEnv(library)}),
        0);
}

// A mapping that cannot be saved is not published.
TEST(LocalKernelPersist, DISABLED_FailedSaveWorker)
{
    if(::geteuid() == 0)
        GTEST_SKIP() << "root ignores file permissions";

    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    const int shipped = topIndex(handle, p);

    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0);

    const fs::path journal = journalPath(library);
    ASSERT_EQ(::chmod(journal.c_str(), 0444), 0);
    EXPECT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_INTERNAL_ERROR);
    ASSERT_EQ(::chmod(journal.c_str(), 0644), 0);

    size_t selectable = 123;
    counts(handle, nullptr, &selectable);
    EXPECT_EQ(selectable, 0u);
    EXPECT_EQ(topIndex(handle, p), shipped) << "selection is unchanged";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, FailedSaveIsNotPublished)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker("LocalKernelPersist.DISABLED_FailedSaveWorker", {libraryEnv(library)}), 0);
    auto lines = journalLines(library);
    EXPECT_GE(countRecords(lines, 'R'), 1u);
    EXPECT_EQ(countRecords(lines, 'E'), 0u);
}

// A mapping whose write fails partway is truncated away, so nothing of a
// mapping reported as not saved stays for the next append to complete.
TEST(LocalKernelPersist, DISABLED_FailedWriteWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0);

    const fs::path    journal = journalPath(library);
    const std::string before  = readFile(journal);
    ASSERT_FALSE(before.empty());

    // The record's first byte is written, and the rest fails.
    std::signal(SIGXFSZ, SIG_IGN);
    rlimit saved{};
    ASSERT_EQ(::getrlimit(RLIMIT_FSIZE, &saved), 0);
    rlimit limited   = saved;
    limited.rlim_cur = before.size() + 1;
    ASSERT_EQ(::setrlimit(RLIMIT_FSIZE, &limited), 0);
    const auto status = hipblasLtLocalKernelSetExactMatch(handle, r.index, M, N, 1, K);
    ASSERT_EQ(::setrlimit(RLIMIT_FSIZE, &saved), 0);
    EXPECT_EQ(status, HIPBLAS_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(readFile(journal), before);

    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, FailedWriteLeavesNothing)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker("LocalKernelPersist.DISABLED_FailedWriteWorker", {libraryEnv(library)}), 0);
    auto lines = journalLines(library);
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(countRecords(lines, 'E'), 1u);
    EXPECT_EQ(countFields(lines.back()), 9u) << lines.back();
    EXPECT_EQ(runWorker(LOAD_WORKER, {libraryEnv(library)}), 0);
}

// A journal deleted and written again, here to the same length, is read from
// its start rather than from where the old one ended.
TEST(LocalKernelPersist, DISABLED_ReplacedJournalWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem small, large;
    ASSERT_EQ(buildProblem(small, 16), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(buildProblem(large, 64), HIPBLAS_STATUS_SUCCESS);
    ASSERT_GE(topIndex(handle, small), LOCAL_INDEX_BASE) << "the first journal was loaded";

    const fs::path    journal = journalPath(library);
    const std::string old     = readFile(journal);
    ASSERT_TRUE(fs::remove(journal));
    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library), env(TEST_M_ENV, "64")}), 0);
    ASSERT_EQ(readFile(journal).size(), old.size());
    EXPECT_LT(topIndex(handle, large), LOCAL_INDEX_BASE) << "not read yet";

    size_t added = 123;
    ASSERT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(added, 0u) << "its payload was already registered";
    EXPECT_GE(topIndex(handle, large), LOCAL_INDEX_BASE);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, RefreshRereadsReplacedJournal)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_ReplacedJournalWorker", {libraryEnv(library)}),
              0);
}

// Payloads that cannot be registered are not recorded. One rejected for its
// shard saves nothing. One rejected for its code object keeps the objects it
// saved, which another registration may already be using: here one shard,
// shared by the three bad code objects.
TEST(LocalKernelPersist, RejectedPayloadsAreNotRecorded)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker("LocalKernelRegister.DISABLED_BadPayloadsWorker", {libraryEnv(library)}),
              0);
    size_t temporary = 0, shards = 0, codeObjects = 0;
    for(auto const& entry : fs::directory_iterator(library / "objects"))
    {
        if(entry.path().filename().string().rfind(".tmp.", 0) == 0)
            ++temporary;
        else if(entry.path().extension() == ".co")
            ++codeObjects;
        else
            ++shards;
    }
    EXPECT_EQ(temporary, 0u);
    EXPECT_EQ(shards, 1u) << "the shards that do not parse were not saved";
    EXPECT_EQ(codeObjects, 3u);
    EXPECT_EQ(countRecords(journalLines(library), 'R'), 0u);
}

// A payload another process has saved since this one started is not copied
// again: the saved objects are used as they are.
TEST(LocalKernelPersist, DISABLED_SavedElsewhereWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    // Initializes the device, which reads the still empty library.
    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(registeredCount(handle), 0u);
    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);

    // Each object's inode, and the directory's modification time, which a
    // temporary copy created and removed would change.
    const fs::path objects = fs::path(library) / "objects";
    auto           state   = [&] {
        std::map<std::string, ino_t> found;
        for(auto const& entry : fs::directory_iterator(objects))
        {
            struct stat st
            {
            };
            if(::stat(entry.path().c_str(), &st) == 0)
                found[entry.path().filename().string()] = st.st_ino;
        }
        struct stat dir
        {
        };
        EXPECT_EQ(::stat(objects.c_str(), &dir), 0);
        return std::make_tuple(found, dir.st_mtim.tv_sec, dir.st_mtim.tv_nsec);
    };
    const auto before = state();
    ASSERT_FALSE(registerPayload(handle, candidates[0] + ".dat", candidates[0] + ".co").empty());
    EXPECT_EQ(state(), before);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, SavedObjectsAreNotCopiedAgain)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_SavedElsewhereWorker",
                        {libraryEnv(tmp.path / "lib")}),
              0);
}

// Saving succeeds beside a temporary file named as this process's first one
// was before, and temporary files a crashed writer left an hour ago go.
TEST(LocalKernelPersist, DISABLED_TemporaryNameWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    std::ofstream(fs::path(library) / "objects" / (".tmp." + std::to_string(::getpid()) + ".0"));
    EXPECT_FALSE(registerPayload(handle, candidates[0] + ".dat", candidates[0] + ".co").empty());
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, TemporaryFilesDoNotCollide)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";
    ASSERT_TRUE(fs::create_directories(library / "objects"));
    for(auto const& dir : {library, library / "objects"})
        ASSERT_EQ(::chmod(dir.c_str(), 0755), 0);

    const fs::path stale = library / "objects" / ".tmp.stale";
    const fs::path fresh = library / "objects" / ".tmp.fresh";
    std::ofstream(stale.string());
    std::ofstream(fresh.string());
    const timespec twoHoursAgo[2] = {{::time(nullptr) - 7200, 0}, {::time(nullptr) - 7200, 0}};
    ASSERT_EQ(::utimensat(AT_FDCWD, stale.c_str(), twoHoursAgo, 0), 0);

    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_TemporaryNameWorker", {libraryEnv(library)}),
              0);
    EXPECT_FALSE(fs::exists(stale));
    EXPECT_TRUE(fs::exists(fresh)) << "another writer's copy in progress is kept";
    EXPECT_GE(countRecords(journalLines(library), 'R'), 1u);
}

#ifdef HIPBLASLT_ENABLE_LOCAL_KERNEL_TEST
// A payload registered through device 0 and then through device 1, of another
// scope, keeps its indices, and is recorded in both journals. Mappings made
// through device 1, also of a kernel registered only through device 0, are
// saved in device 1's journal.
TEST(LocalKernelPersist, DISABLED_TwoScopesWorker)
{
    char const* scope = std::getenv("HIPBLASLT_LOCAL_KERNEL_TEST_SCOPE");
    ASSERT_NE(scope, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t first, second;
    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    ASSERT_EQ(hipblasLtCreate(&second), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipblasLtCreate(&first), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    Registered r = registerUsable(first, p, candidates);
    ASSERT_GE(r.index, 0);
    const auto indices = registerPayload(first, r.stem + ".dat", r.stem + ".co");
    EXPECT_EQ(registerPayload(second, r.stem + ".dat", r.stem + ".co"), indices);
    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(second, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);

    for(auto const& stem : candidates)
    {
        if(stem == r.stem)
            continue;
        const auto other = registerPayload(first, stem + ".dat", stem + ".co");
        ASSERT_FALSE(other.empty()) << stem;
        EXPECT_EQ(hipblasLtLocalKernelSetExactMatch(second, other[0], 2 * M, N, 1, K),
                  HIPBLAS_STATUS_SUCCESS);
        break;
    }

    EXPECT_EQ(hipblasLtDestroy(first), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtDestroy(second), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, SavesForTheHandlesDevice)
{
    if(!haveTwoMatchingDevices())
        GTEST_SKIP() << "needs two devices of the same architecture";
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path    library = tmp.path / "lib";
    const std::string scope   = deviceArch() + "_TEST";
    const std::string override("HIPBLASLT_LOCAL_KERNEL_TEST_SCOPE");

    ASSERT_EQ(runWorker("LocalKernelPersist.DISABLED_TwoScopesWorker",
                        {libraryEnv(library), env(override.c_str(), "1=" + scope)}),
              0);
    const auto first  = journalLines(library);
    const auto second = journalLines(library, scope);
    EXPECT_GE(countRecords(first, 'R'), 1u);
    EXPECT_EQ(countRecords(first, 'E'), 0u) << "nothing was mapped through device 0";
    const size_t mapped = candidatePayloads().size() > 1 ? 2 : 1;
    EXPECT_EQ(countRecords(second, 'R'), mapped);
    EXPECT_EQ(countRecords(second, 'E'), mapped);

    EXPECT_EQ(runWorker(LOAD_WORKER,
                        {libraryEnv(library),
                         env(override.c_str(), "0=" + scope),
                         env(TEST_SELECTABLE_ENV, std::to_string(mapped))}),
              0)
        << "a later process on a device of that scope loads its mappings";
}

// Device 0 is given another compute unit count than the device hipBLASLt
// describes its architecture by, which is itself, so the calls that would
// make or load mappings its queries never find fail.
TEST(LocalKernelPersist, DISABLED_OtherDeviceClassWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    const int saved = topIndex(handle, p);
    ASSERT_GE(saved, LOCAL_INDEX_BASE) << "the saved mapping was loaded at startup";

    int numIndices = -1;
    EXPECT_EQ(hipblasLtLocalKernelRegister(handle,
                                           (candidates[0] + ".dat").c_str(),
                                           (candidates[0] + ".co").c_str(),
                                           nullptr,
                                           0,
                                           &numIndices),
              HIPBLAS_STATUS_ARCH_MISMATCH);
    EXPECT_EQ(numIndices, 0);
    EXPECT_EQ(hipblasLtLocalKernelSetExactMatch(handle, saved, 2 * M, N, 1, K),
              HIPBLAS_STATUS_ARCH_MISMATCH);
    size_t added = 123;
    EXPECT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_ARCH_MISMATCH);
    EXPECT_EQ(added, 0u);

    size_t selectable = 0;
    counts(handle, nullptr, &selectable);
    EXPECT_EQ(selectable, 1u) << "nothing was mapped";
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

// A thread maps M x N x K to one kernel and, with
// HIPBLASLT_LOCAL_KERNEL_TEST_PAUSE_MS, waits between saving the mapping and
// reading it back. Meanwhile another process maps the shape to a second
// kernel, here by appending its record, and a Refresh runs. This process
// then selects the second kernel, as every later process does.
TEST(LocalKernelPersist, DISABLED_RefreshDuringMappingWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    const auto kernels = twoUsableKernels(handle, p, candidates);
    ASSERT_EQ(kernels.size(), 2u) << "no two kernels in this build's library serve the problem";
    const auto kernelOf = mappingNames(handle, library, kernels);
    ASSERT_EQ(kernelOf.size(), 2u);
    std::string second;
    for(auto const& [name, kernel] : kernelOf)
        if(kernel == kernels[1])
            second = name;
    auto lastKernel = [&] {
        auto found = kernelOf.find(lastMapping(library));
        return found == kernelOf.end() ? -1 : found->second;
    };

    hipblasStatus_t mapped = HIPBLAS_STATUS_NOT_INITIALIZED;
    std::thread     mapping(
        [&] { mapped = hipblasLtLocalKernelSetExactMatch(handle, kernels[0], M, N, 1, K); });
    for(int i = 0; i < 6000 && lastKernel() != kernels[0]; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_EQ(lastKernel(), kernels[0]) << "the first mapping was saved";
    std::ofstream(journalPath(library).string(), std::ios::app)
        << journalRecord("E " + second + " " + std::to_string(M) + " " + std::to_string(N) + " 1 "
                         + std::to_string(K))
        << '\n';
    size_t added = 123;
    EXPECT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    mapping.join();
    ASSERT_EQ(mapped, HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(lastKernel(), kernels[1]);
    EXPECT_EQ(topIndex(handle, p), kernels[1]) << "the journal's last mapping is selected";
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

// A mapping whose read back fails, or reads a journal that replaced the one
// it was saved in, is published as it is, over a later mapping this process
// had already applied. The next Refresh puts the journal's last mapping back.
TEST(LocalKernelPersist, DISABLED_FailedReadBackWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    const auto kernels = twoUsableKernels(handle, p, candidates);
    ASSERT_EQ(kernels.size(), 2u) << "no two kernels in this build's library serve the problem";
    const auto kernelOf = mappingNames(handle, library, kernels);
    ASSERT_EQ(kernelOf.size(), 2u);
    std::string second;
    for(auto const& [name, kernel] : kernelOf)
        if(kernel == kernels[1])
            second = name;
    const std::string secondMapping
        = journalRecord("E " + second + " " + std::to_string(M) + " " + std::to_string(N) + " 1 "
                        + std::to_string(K));
    auto lastKernel = [&] {
        auto found = kernelOf.find(lastMapping(library));
        return found == kernelOf.end() ? -1 : found->second;
    };

    // Maps the first kernel; `meanwhile` runs once it is saved and before it
    // is read back.
    auto mapFirstWhile = [&](auto const& meanwhile) {
        hipblasStatus_t mapped = HIPBLAS_STATUS_NOT_INITIALIZED;
        std::thread     mapping(
            [&] { mapped = hipblasLtLocalKernelSetExactMatch(handle, kernels[0], M, N, 1, K); });
        for(int i = 0; i < 6000 && lastKernel() != kernels[0]; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        EXPECT_EQ(lastKernel(), kernels[0]) << "the first mapping was saved";
        meanwhile();
        mapping.join();
        EXPECT_EQ(mapped, HIPBLAS_STATUS_SUCCESS);
    };
    size_t added = 0;

    // Another process maps the shape to the second kernel and a Refresh
    // applies both mappings; then the read back cannot lock the journal.
    int held = -1;
    mapFirstWhile([&] {
        std::ofstream(journalPath(library).string(), std::ios::app) << secondMapping << '\n';
        EXPECT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(topIndex(handle, p), kernels[1]);
        held = ::open(journalPath(library).c_str(), O_RDONLY | O_CLOEXEC);
        EXPECT_EQ(::flock(held, LOCK_EX), 0);
    });
    EXPECT_EQ(topIndex(handle, p), kernels[0]) << "published without its read back";
    ::close(held);
    ASSERT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(lastKernel(), kernels[1]);
    EXPECT_EQ(topIndex(handle, p), kernels[1]) << "the next Refresh read the mapping again";

    // The journal is replaced by one that maps the shape to the second kernel.
    mapFirstWhile([&] {
        const auto  lines   = journalLines(library);
        std::string version = lines[0];
        version.back()      = version.back() == '0' ? '1' : '0';
        std::string payload;
        for(auto const& line : lines)
            if(line.rfind("R " + second.substr(0, second.rfind(' ')) + " ", 0) == 0)
                payload = line;
        ASSERT_FALSE(payload.empty());
        const fs::path journal = journalPath(library);
        const fs::path next    = journal.string() + ".next";
        std::ofstream(next.string(), std::ios::binary) << version << '\n'
                                                       << payload << '\n'
                                                       << secondMapping << '\n';
        ASSERT_EQ(::chmod(next.c_str(), 0644), 0);
        fs::rename(next, journal);
    });
    EXPECT_EQ(topIndex(handle, p), kernels[0]) << "published after reading the new journal";
    ASSERT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(lastKernel(), kernels[1]);
    EXPECT_EQ(topIndex(handle, p), kernels[1]) << "the next Refresh read the new journal again";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, FailedReadBackIsRepairedByRefresh)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_FailedReadBackWorker",
                        {libraryEnv(tmp.path / "lib"),
                         env("HIPBLASLT_LOCAL_KERNEL_TEST_PAUSE_MS", "500")}),
              0);
}

TEST(LocalKernelPersist, RefreshDuringMappingKeepsJournalOrder)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_RefreshDuringMappingWorker",
                        {libraryEnv(tmp.path / "lib"),
                         env("HIPBLASLT_LOCAL_KERNEL_TEST_PAUSE_MS", "500")}),
              0);
}

TEST(LocalKernelPersist, OtherDeviceClassIsRefused)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";
    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    const auto   saved   = journalLines(library);
    const size_t objects = countObjects(library, ".co");

    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_OtherDeviceClassWorker",
                        {libraryEnv(library),
                         env("HIPBLASLT_LOCAL_KERNEL_TEST_CU",
                             "0=" + std::to_string(deviceCuCount() + 1))}),
              0);
    EXPECT_EQ(journalLines(library), saved) << "nothing was saved";
    EXPECT_EQ(countObjects(library, ".co"), objects);
}
#endif

// A journal locked by a process that is stopped makes readers and writers
// fail after a bounded wait instead of waiting with it.
TEST(LocalKernelPersist, DISABLED_LockedJournalWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    auto   start = std::chrono::steady_clock::now();
    size_t added = 123;
    EXPECT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_INTERNAL_ERROR)
        << "a reader gives up";
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(30));

    start          = std::chrono::steady_clock::now();
    int numIndices = -1;
    EXPECT_EQ(hipblasLtLocalKernelRegister(handle,
                                           (candidates[0] + ".dat").c_str(),
                                           (candidates[0] + ".co").c_str(),
                                           nullptr,
                                           0,
                                           &numIndices),
              HIPBLAS_STATUS_INTERNAL_ERROR)
        << "a writer gives up";
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(30));
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, LockedJournalTimesOut)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";
    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);

    const int fd = ::open(journalPath(library).c_str(), O_RDONLY | O_CLOEXEC);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::flock(fd, LOCK_EX), 0);
    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_LockedJournalWorker", {libraryEnv(library)}),
              0);
    ::close(fd);
}

// Maps a kernel once the test has locked the journal, so the save waits for
// the lock while the test replaces the journal.
TEST(LocalKernelPersist, DISABLED_AppendDuringReplaceWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const fs::path                 signals    = fs::path(library).parent_path();
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0);

    std::ofstream((signals / "ready").string());
    ASSERT_TRUE(waitForFile(signals / "locked"));
    EXPECT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

// A save granted the lock on a journal that was replaced while it waited
// lands in the journal that replaced it.
TEST(LocalKernelPersist, AppendWaitingOnAReplacedJournal)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    const pid_t worker = spawnWorker("LocalKernelPersist.DISABLED_AppendDuringReplaceWorker",
                                     {libraryEnv(library)});
    ASSERT_GT(worker, 0);
    ASSERT_TRUE(waitForFile(tmp.path / "ready"));

    const fs::path journal = journalPath(library);
    const int      held    = ::open(journal.c_str(), O_RDONLY | O_CLOEXEC);
    ASSERT_GE(held, 0);
    ASSERT_EQ(::flock(held, LOCK_EX), 0);
    std::ofstream((tmp.path / "locked").string());
    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::string content = readFile(journal);
    ASSERT_TRUE(fs::remove(journal));
    std::ofstream(journal.string(), std::ios::binary) << content;
    ASSERT_EQ(::chmod(journal.c_str(), 0644), 0);
    ::close(held);

    ASSERT_EQ(waitWorker(worker), 0);
    EXPECT_EQ(countRecords(journalLines(library), 'E'), 1u);
    EXPECT_EQ(runWorker(LOAD_WORKER, {libraryEnv(library)}), 0);
}

// Registers payloads the library already holds but whose code objects this
// process found damaged, which repairs them, and maps the shape the journal
// already maps to the same kernel.
TEST(LocalKernelPersist, DISABLED_RepairedPayloadWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    ASSERT_LT(topIndex(handle, p), LOCAL_INDEX_BASE) << "the damaged payload was skipped";

    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0);
    size_t added = 123;
    ASSERT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, p), r.index)
        << "the saved mapping applies once its payload is registered";

    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, p), r.index) << "a mapping reported as made is applied";
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, MappingAlreadySavedIsApplied)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    const auto saved = journalLines(library);
    for(auto const& entry : fs::directory_iterator(library / "objects"))
    {
        if(entry.path().extension() != ".co")
            continue;
        std::fstream file(entry.path(), std::ios::in | std::ios::out | std::ios::binary);
        char         last = 0;
        file.seekg(-1, std::ios::end);
        file.get(last);
        file.seekp(-1, std::ios::end);
        file.put(static_cast<char>(~last));
        ASSERT_TRUE(file.good()) << entry.path();
    }

    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_RepairedPayloadWorker", {libraryEnv(library)}),
              0);
    EXPECT_EQ(journalLines(library), saved) << "nothing new was saved";
}

// A payload skipped because its code object was missing is loaded by a later
// Refresh once the file is back.
TEST(LocalKernelPersist, DISABLED_RestoredObjectWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    ASSERT_LT(topIndex(handle, p), LOCAL_INDEX_BASE) << "the payload was skipped";

    // Puts back code objects the test moved away, and takes away the group
    // write bit the test gave others; objects it left alone are not touched.
    for(auto const& entry : fs::directory_iterator(fs::path(library) / "objects"))
    {
        if(entry.path().extension() == ".away")
            fs::rename(entry.path(), fs::path(entry.path()).replace_extension());
        else if((fs::status(entry.path()).permissions() & fs::perms::group_write)
                != fs::perms::none)
            ASSERT_EQ(::chmod(entry.path().c_str(), 0644), 0) << entry.path();
    }

    size_t added = 0;
    ASSERT_EQ(hipblasLtLocalKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_GT(added, 0u);
    EXPECT_GE(topIndex(handle, p), LOCAL_INDEX_BASE);
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, RefreshRetriesSkippedPayload)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    for(auto const& entry : fs::directory_iterator(library / "objects"))
        if(entry.path().extension() == ".co")
            fs::rename(entry.path(), entry.path().string() + ".away");

    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_RestoredObjectWorker", {libraryEnv(library)}),
              0);
}

// The same for a payload skipped because its code object was group-writable,
// once its mode is repaired, which changes neither its contents nor its size.
TEST(LocalKernelPersist, RefreshRetriesPayloadAfterModeRepair)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    for(auto const& entry : fs::directory_iterator(library / "objects"))
        if(entry.path().extension() == ".co")
            ASSERT_EQ(::chmod(entry.path().c_str(), 0664), 0);

    EXPECT_EQ(runWorker("LocalKernelPersist.DISABLED_RestoredObjectWorker", {libraryEnv(library)}),
              0);
}

// A saved payload that verifies but cannot be registered, here a shard that is
// not one, is warned about once: reading the journal again from an earlier
// record, as the Refresh after a skipped payload's repair does, passes it
// quietly.
TEST(LocalKernelPersist, RejectedSavedPayloadIsPassedQuietly)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";
    const fs::path objects = library / "objects";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    std::vector<fs::path> saved;
    for(auto const& entry : fs::directory_iterator(objects))
        if(entry.path().extension() == ".co")
            saved.push_back(entry.path());
    ASSERT_FALSE(saved.empty());

    // The rejected payload: a shard that does not parse, with a code object
    // of its own.
    const std::string shard      = "not a shard";
    const std::string codeObject = readFile(saved[0]) + std::string(16, '\0');
    const std::string shardName = objectName(shard), codeObjectName = objectName(codeObject);
    std::ofstream((objects / (shardName + ".dat")).string(), std::ios::binary) << shard;
    std::ofstream((objects / (codeObjectName + ".co")).string(), std::ios::binary) << codeObject;
    for(auto const& name : {shardName + ".dat", codeObjectName + ".co"})
        ASSERT_EQ(::chmod((objects / name).c_str(), 0644), 0);
    std::ofstream(journalPath(library).string(), std::ios::app)
        << journalRecord("R " + shardName + " " + codeObjectName + " dat") << '\n';

    // The skipped payloads whose repair makes the next Refresh read again
    // from before the rejected one.
    for(auto const& path : saved)
        fs::rename(path, path.string() + ".away");

    const fs::path errors = tmp.path / "stderr";
    ASSERT_EQ(runWorker("LocalKernelPersist.DISABLED_RestoredObjectWorker",
                        {libraryEnv(library)},
                        errors.string()),
              0);
    const std::string log     = readFile(errors);
    const std::string warning = "skipping saved local kernel payload " + shardName;
    size_t            count   = 0;
    for(size_t at = log.find(warning); at != std::string::npos; at = log.find(warning, at + 1))
        ++count;
    EXPECT_EQ(count, 1u) << log;
}

// A payload this process registered is saved again when the library is
// cleaned: by Register, with its objects, and by SetExactMatch with the
// mapping when only the journal is replaced.
TEST(LocalKernelPersist, DISABLED_RecreatedJournalWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0);

    ASSERT_TRUE(fs::remove(journalPath(library)));
    std::vector<fs::path> objects;
    for(auto const& entry : fs::directory_iterator(fs::path(library) / "objects"))
        objects.push_back(entry.path());
    for(auto const& path : objects)
        ASSERT_TRUE(fs::remove(path)) << path;
    EXPECT_FALSE(registerPayload(handle, r.stem + ".dat", r.stem + ".co").empty());
    EXPECT_EQ(countRecords(journalLines(library), 'R'), 1u) << "Register recorded it again";
    EXPECT_EQ(countObjects(library, ".co"), 1u) << "with its objects";

    ASSERT_TRUE(fs::remove(journalPath(library)));
    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    auto lines = journalLines(library);
    EXPECT_EQ(countRecords(lines, 'R'), 1u) << "SetExactMatch recorded its payload";
    EXPECT_EQ(countRecords(lines, 'E'), 1u);
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, RecordsPayloadInAReplacedJournal)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(
        runWorker("LocalKernelPersist.DISABLED_RecreatedJournalWorker", {libraryEnv(library)}), 0);
    EXPECT_EQ(runWorker(LOAD_WORKER, {libraryEnv(library)}), 0)
        << "a later process loads the mapping and its objects from the cleaned library";
}

// Device 0 cannot read the journal when it is initialized, because the test
// holds its lock. Device 1, of the same scope, is initialized after the test
// releases it, and reads it.
TEST(LocalKernelPersist, DISABLED_LockedAtStartupWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);
    const fs::path signals = fs::path(library).parent_path();

    hipblasLtHandle_t first, second;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipblasLtCreate(&first), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(registeredCount(first), 0u) << "the journal was locked";

    std::ofstream((signals / "ready").string());
    ASSERT_TRUE(waitForFile(signals / "released"));
    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    ASSERT_EQ(hipblasLtCreate(&second), HIPBLAS_STATUS_SUCCESS);
    EXPECT_GT(registeredCount(second), 0u);

    EXPECT_EQ(hipblasLtDestroy(second), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    EXPECT_EQ(hipblasLtDestroy(first), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelPersist, FailedStartupReadIsRetried)
{
    if(!haveTwoMatchingDevices())
        GTEST_SKIP() << "needs two devices of the same architecture";
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";
    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);

    const int held = ::open(journalPath(library).c_str(), O_RDONLY | O_CLOEXEC);
    ASSERT_GE(held, 0);
    ASSERT_EQ(::flock(held, LOCK_EX), 0);
    const pid_t worker
        = spawnWorker("LocalKernelPersist.DISABLED_LockedAtStartupWorker", {libraryEnv(library)});
    ASSERT_GT(worker, 0);
    EXPECT_TRUE(waitForFile(tmp.path / "ready"));
    ::close(held);
    std::ofstream((tmp.path / "released").string());
    EXPECT_EQ(waitWorker(worker), 0);
}
