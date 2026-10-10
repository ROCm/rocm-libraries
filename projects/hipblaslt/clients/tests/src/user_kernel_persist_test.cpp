// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end tests of the user kernel library named by
// HIPBLASLT_USER_KERNEL_LIBRARY. Each worker is a separate process, as a
// process that saves kernels and a later one that loads them would be.

#include <gtest/gtest.h>

#include "user_kernel_test_common.hpp"

#include <sys/stat.h>

#include <set>
#include <sstream>

namespace
{
    using namespace user_kernel_test;
    namespace fs = std::filesystem;

    constexpr char TEST_M_ENV[]          = "HIPBLASLT_USER_KERNEL_TEST_M";
    constexpr char TEST_SELECTABLE_ENV[] = "HIPBLASLT_USER_KERNEL_TEST_SELECTABLE";

    constexpr char SAVE_WORKER[]           = "UserKernelPersist.DISABLED_SaveWorker";
    constexpr char LOAD_WORKER[]           = "UserKernelPersist.DISABLED_LoadWorker";
    constexpr char NOTHING_LOADED_WORKER[] = "UserKernelPersist.DISABLED_NothingLoadedWorker";

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

    std::string deviceArch()
    {
        int             device = 0;
        hipDeviceProp_t prop;
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&prop, device) != hipSuccess)
            return {};
        std::string name(prop.gcnArchName);
        return name.substr(0, name.find(':'));
    }

    fs::path journalPath(fs::path const& library)
    {
        return library / deviceArch() / "journal";
    }

    // The journal's lines, including a final one without a newline.
    std::vector<std::string> journalLines(fs::path const& library)
    {
        std::ifstream            in(journalPath(library));
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
            const auto       status     = hipblasLtUserKernelRegister(handle,
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
        ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, registered, selectable),
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
        if(hipblasLtUserKernelRegister(handle,
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

    // Scratch files shared by the two-code-object workers.
    fs::path twoCodeObjectsDir()
    {
        char const* library = std::getenv(LIBRARY_ENV);
        return library ? fs::path(library).parent_path() / "two-code-objects" : fs::path();
    }
} // namespace

// Registers a usable kernel from a copy of a payload, deletes the copy, and
// maps M (default 16) x N x K to the kernel.
TEST(UserKernelPersist, DISABLED_SaveWorker)
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

    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(handle, r.index, m, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, p), r.index);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

// The first heuristic query for M (default 16) x N x K returns a user kernel,
// without any user kernel call before it, and the kernel computes correctly.
TEST(UserKernelPersist, DISABLED_LoadWorker)
{
    const int64_t m          = envInt(TEST_M_ENV, M);
    const size_t  selectable = static_cast<size_t>(envInt(TEST_SELECTABLE_ENV, 1));

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p, m), HIPBLAS_STATUS_SUCCESS);

    auto best = topResults(handle, p, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_GE(algoIndex(best[0].algo), USER_INDEX_BASE)
        << "the saved mapping applies to the first query";

    size_t registered = 0, mapped = 0;
    counts(handle, &registered, &mapped);
    EXPECT_GT(registered, 0u);
    EXPECT_EQ(mapped, selectable);

    runAndCheck(handle, p, best[0].algo, best[0].workspaceSize);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelPersist, DISABLED_NothingLoadedWorker)
{
    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    const int top = topIndex(handle, p);
    EXPECT_GE(top, 0);
    EXPECT_LT(top, USER_INDEX_BASE);

    size_t registered = 123, selectable = 123;
    counts(handle, &registered, &selectable);
    EXPECT_EQ(registered, 0u);
    EXPECT_EQ(selectable, 0u);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelPersist, SavedKernelsLoadInNewProcess)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);

    EXPECT_GE(countObjects(library, ".co"), 1u);
    auto lines = journalLines(library);
    EXPECT_GE(countRecords(lines, 'R'), 1u);
    EXPECT_EQ(countRecords(lines, 'E'), 1u);
    for(auto const& entry : fs::directory_iterator(library / "objects"))
        EXPECT_EQ(entry.path().string().find(".tmp."), std::string::npos)
            << "no temporary object left behind: " << entry.path();

    EXPECT_EQ(runWorker(LOAD_WORKER, {libraryEnv(library)}), 0)
        << "a new process loads the saved kernel; the payload it was registered from is gone";
    EXPECT_EQ(runWorker(NOTHING_LOADED_WORKER), 0) << "without the variable, nothing is loaded";
}

// Registering a payload already registered, or already saved, returns the
// indices it has and records nothing new. Both spellings of a compressed
// shard name the same payload.
TEST(UserKernelPersist, DISABLED_IdempotentWorker)
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
    ASSERT_EQ(hipblasLtUserKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(added, 0u) << "this process's own records are not registered again";
    EXPECT_EQ(registeredCount(handle), afterFirst);
    EXPECT_EQ(registerPayload(handle, bare, co), first) << "indices are unchanged by a refresh";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelPersist, RegisterIsIdempotent)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker("UserKernelPersist.DISABLED_IdempotentWorker", {libraryEnv(library)}), 0);
    ASSERT_EQ(runWorker("UserKernelPersist.DISABLED_IdempotentWorker", {libraryEnv(library)}), 0)
        << "a second process finds the payload already loaded";

    auto lines = journalLines(library);
    EXPECT_EQ(countRecords(lines, 'R'), 1u);
    EXPECT_EQ(countRecords(lines, 'E'), 0u);
}

TEST(UserKernelPersist, RegisterIsIdempotentWithoutLibrary)
{
    EXPECT_EQ(runWorker("UserKernelPersist.DISABLED_IdempotentWorker"), 0);
}

// A running process sees what another saved only after a refresh.
TEST(UserKernelPersist, DISABLED_RefreshWorker)
{
    char const* library = std::getenv(LIBRARY_ENV);
    ASSERT_NE(library, nullptr);

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    ASSERT_LT(topIndex(handle, p), USER_INDEX_BASE);

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    EXPECT_LT(topIndex(handle, p), USER_INDEX_BASE) << "not visible before a refresh";

    size_t added = 0;
    ASSERT_EQ(hipblasLtUserKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_GT(added, 0u);
    const size_t registered = registeredCount(handle);
    EXPECT_EQ(registered, added);
    const int mapped = topIndex(handle, p);
    EXPECT_GE(mapped, USER_INDEX_BASE);

    ASSERT_EQ(hipblasLtUserKernelRefresh(handle, &added), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(added, 0u) << "a second refresh finds nothing new";
    EXPECT_EQ(registeredCount(handle), registered);
    EXPECT_EQ(topIndex(handle, p), mapped);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelPersist, RefreshLoadsWhatOthersSaved)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    EXPECT_EQ(runWorker("UserKernelPersist.DISABLED_RefreshWorker", {libraryEnv(tmp.path / "lib")}),
              0);
}

// Two processes saving at once both land, and the journal stays well formed.
TEST(UserKernelPersist, ConcurrentWriters)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    const pid_t small = spawnWorker(SAVE_WORKER, {libraryEnv(library), env(TEST_M_ENV, "16")});
    const pid_t large = spawnWorker(SAVE_WORKER, {libraryEnv(library), env(TEST_M_ENV, "64")});
    ASSERT_EQ(waitWorker(small), 0);
    ASSERT_EQ(waitWorker(large), 0);

    auto                  lines = journalLines(library);
    std::set<std::string> payloads;
    for(auto const& line : lines)
    {
        if(line.rfind("R ", 0) == 0)
        {
            EXPECT_EQ(countFields(line), 4u) << line;
            EXPECT_TRUE(payloads.insert(line).second) << "payload recorded twice: " << line;
        }
        else
        {
            EXPECT_EQ(line.rfind("E ", 0), 0u) << line;
            EXPECT_EQ(countFields(line), 8u) << line;
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
TEST(UserKernelPersist, DISABLED_InMemoryWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0);
    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, p), r.index);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelPersist, IgnoresWritableDirectory)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";
    ASSERT_TRUE(fs::create_directory(library));
    ASSERT_EQ(::chmod(library.c_str(), 0777), 0);

    ASSERT_EQ(runWorker("UserKernelPersist.DISABLED_InMemoryWorker", {libraryEnv(library)}), 0);
    EXPECT_TRUE(fs::is_empty(library)) << "nothing is written to a writable directory";
    EXPECT_EQ(runWorker(NOTHING_LOADED_WORKER, {libraryEnv(library)}), 0);
}

// Lines that do not parse are skipped, and a partial last line left by a
// crashed writer does not swallow the next record.
TEST(UserKernelPersist, ToleratesDamagedJournal)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library)}), 0);
    {
        std::ofstream out(journalPath(library), std::ios::app);
        out << "Z unknown record\nR nothex nothex dat\nE 12";
    }

    ASSERT_EQ(runWorker(LOAD_WORKER, {libraryEnv(library)}), 0);
    ASSERT_EQ(runWorker(SAVE_WORKER, {libraryEnv(library), env(TEST_M_ENV, "64")}), 0);

    auto lines = journalLines(library);
    EXPECT_NE(std::find(lines.begin(), lines.end(), "E 12"), lines.end())
        << "the partial line was terminated, not extended";
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(lines.back().rfind("E ", 0), 0u);
    EXPECT_EQ(countFields(lines.back()), 8u);

    EXPECT_EQ(
        runWorker(LOAD_WORKER,
                  {libraryEnv(library), env(TEST_M_ENV, "64"), env(TEST_SELECTABLE_ENV, "2")}),
        0);
}

// A record whose objects are gone is skipped with a warning.
TEST(UserKernelPersist, SkipsMissingObjects)
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

// Registers one shard with two code objects that differ only in trailing
// bytes, and maps a kernel of the second pairing. Leaves the shard's stem and
// the second code object in twoCodeObjectsDir().
TEST(UserKernelPersist, DISABLED_TwoCodeObjectsSaveWorker)
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

    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(handle, second[position], M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(topIndex(handle, p), second[position]);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

// After a restart both pairings are loaded, and the mapping is restored
// against the second.
TEST(UserKernelPersist, DISABLED_TwoCodeObjectsLoadWorker)
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
    EXPECT_GE(mapped, USER_INDEX_BASE);

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

TEST(UserKernelPersist, OneShardWithTwoCodeObjects)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(
        runWorker("UserKernelPersist.DISABLED_TwoCodeObjectsSaveWorker", {libraryEnv(library)}), 0);

    auto                                  lines = journalLines(library);
    std::vector<std::vector<std::string>> payloads, exacts;
    for(auto const& line : lines)
    {
        std::istringstream       in(line);
        std::vector<std::string> fields;
        for(std::string field; in >> field;)
            fields.push_back(field);
        (line.rfind("R ", 0) == 0 ? payloads : exacts).push_back(fields);
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
        runWorker("UserKernelPersist.DISABLED_TwoCodeObjectsLoadWorker", {libraryEnv(library)}), 0);
}

// A mapping that cannot be saved is not published.
TEST(UserKernelPersist, DISABLED_FailedSaveWorker)
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
    EXPECT_EQ(hipblasLtUserKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_INTERNAL_ERROR);
    ASSERT_EQ(::chmod(journal.c_str(), 0644), 0);

    size_t selectable = 123;
    counts(handle, nullptr, &selectable);
    EXPECT_EQ(selectable, 0u);
    EXPECT_EQ(topIndex(handle, p), shipped) << "selection is unchanged";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelPersist, FailedSaveIsNotPublished)
{
    TempDir tmp;
    ASSERT_FALSE(tmp.path.empty());
    const fs::path library = tmp.path / "lib";

    ASSERT_EQ(runWorker("UserKernelPersist.DISABLED_FailedSaveWorker", {libraryEnv(library)}), 0);
    auto lines = journalLines(library);
    EXPECT_GE(countRecords(lines, 'R'), 1u);
    EXPECT_EQ(countRecords(lines, 'E'), 0u);
}
