// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>

#if defined(__linux__)
#include <link.h>
#endif

namespace
{

using hipdnn_data_sdk::utilities::detail::pathForDiagnostic;
using hipdnn_frontend::detail::BackendLibraryResolution;
using hipdnn_frontend::detail::BackendResolutionInputs;
using hipdnn_frontend::detail::resolveBackendLibrary;
using hipdnn_test_sdk::utilities::ScopedDirectory;

/// A temporary-directory name no other invocation of this suite can pick. The stamp
/// separates concurrent processes that share TMPDIR -- two build configurations
/// running this binary at once, for instance -- and the counter separates successive
/// cases within one process. ScopedDirectory then creates the name exclusively and
/// throws if it is taken, so an invocation only ever owns, and only ever removes, a
/// directory it created itself. Nothing is deleted before acquisition: a name that
/// already exists belongs to somebody else.
std::filesystem::path uniqueResolutionRoot(const std::string& label)
{
    static const std::string s_session
        = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    static unsigned s_counter = 0;

    return std::filesystem::temp_directory_path()
           / ("hipdnn_resolution_" + label + "_" + s_session + "_" + std::to_string(s_counter++));
}

const std::string& backendFileName()
{
    static const std::string s_name = hipdnn_data_sdk::utilities::getLibraryName("hipdnn_backend");
    return s_name;
}

void closeResolved(const BackendLibraryResolution& resolution)
{
    if(resolution.handle != nullptr)
    {
        hipdnn_data_sdk::utilities::closeLibrary(resolution.handle);
    }
}

#if defined(__linux__)

struct SmallestLoadedLibrary
{
    std::filesystem::path path;
    std::uintmax_t size = 0;
};

int recordSmallestLoadedLibrary(struct dl_phdr_info* info,
                                [[maybe_unused]] size_t phdrSize,
                                void* data)
{
    if(info->dlpi_name == nullptr || info->dlpi_name[0] == '\0')
    {
        return 0;
    }

    std::error_code failed;
    if(!std::filesystem::is_regular_file(info->dlpi_name, failed) || failed)
    {
        return 0;
    }
    const std::uintmax_t size = std::filesystem::file_size(info->dlpi_name, failed);
    if(failed)
    {
        return 0;
    }

    auto& smallest = *static_cast<SmallestLoadedLibrary*>(data);
    if(smallest.size == 0 || size < smallest.size)
    {
        smallest = {info->dlpi_name, size};
    }
    return 0;
}

/// Copy the smallest loaded library as a stand-in; resolution needs no backend symbols.
const std::filesystem::path& loadableLibrarySource()
{
    static const std::filesystem::path s_source = [] {
        SmallestLoadedLibrary smallest;
        dl_iterate_phdr(&recordSmallestLoadedLibrary, &smallest);
        return smallest.path;
    }();
    return s_source;
}

#endif // defined(__linux__)

/// Isolate filesystem candidates in a temporary tree; the loader's search remains external.
class TestBackendLibraryResolution : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* const info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NO_THROW(_root
                        = std::make_unique<ScopedDirectory>(uniqueResolutionRoot(info->name())));
    }

    void TearDown() override
    {
        // Removes exactly the directory this invocation created, and nothing else.
        _root.reset();
    }

    std::filesystem::path directory(const std::string& name)
    {
        const std::filesystem::path created = root() / name;
        std::error_code failed;
        std::filesystem::create_directories(created, failed);
        EXPECT_FALSE(failed) << failed.message();
        return created;
    }

    std::filesystem::path directoryWithUnloadableBackend(const std::string& name)
    {
        const std::filesystem::path created = directory(name);
        std::ofstream corrupt(created / backendFileName(), std::ios::binary);
        corrupt << "this is not a shared object";
        EXPECT_TRUE(corrupt.good());
        return created;
    }

#if defined(__linux__)
    std::filesystem::path directoryWithLoadableBackend(const std::string& name)
    {
        const std::filesystem::path created = directory(name);
        EXPECT_FALSE(loadableLibrarySource().empty()) << "no loaded shared object to copy";

        std::error_code failed;
        std::filesystem::copy_file(loadableLibrarySource(),
                                   created / backendFileName(),
                                   std::filesystem::copy_options::overwrite_existing,
                                   failed);
        EXPECT_FALSE(failed) << failed.message();
        return created;
    }
#endif

    const std::filesystem::path& root() const
    {
        return _root->path();
    }

private:
    std::unique_ptr<ScopedDirectory> _root;
};

} // namespace

// A rejected override may still resolve through the loader, so check the rejection warning.

TEST_F(TestBackendLibraryResolution, EmptyOverrideIsRejectedWithAWarning)
{
    BackendResolutionInputs inputs;
    inputs.overrideDirectory = std::filesystem::path();
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";

    testing::internal::CaptureStderr();
    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);
    const std::string warning = testing::internal::GetCapturedStderr();

    EXPECT_NE(warning.find("ignoring HIPDNN_BACKEND_LIBRARY_PATH"), std::string::npos) << warning;
    EXPECT_NE(warning.find("absolute directory"), std::string::npos) << warning;
    closeResolved(resolution);
}

TEST_F(TestBackendLibraryResolution, RelativeOverrideIsRejectedWithAWarning)
{
    const std::filesystem::path relative = std::filesystem::path("relative") / "lib";

    BackendResolutionInputs inputs;
    inputs.overrideDirectory = relative;
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";

    testing::internal::CaptureStderr();
    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);
    const std::string warning = testing::internal::GetCapturedStderr();

    EXPECT_NE(warning.find("ignoring HIPDNN_BACKEND_LIBRARY_PATH"), std::string::npos) << warning;
    EXPECT_NE(warning.find(relative.string()), std::string::npos) << warning;
    EXPECT_EQ(resolution.diagnostics.find(relative.string()), std::string::npos)
        << "a rejected override was still attempted: " << resolution.diagnostics;
    closeResolved(resolution);
}

// Secure execution must not visit computed directories, even if the loader later succeeds.
TEST_F(TestBackendLibraryResolution, SecureExecutionSkipsTheDerivedTiers)
{
    BackendResolutionInputs inputs;
    inputs.secureExecution = true;
    inputs.selfDirectory = directory("self");
    inputs.hipAnchorDirectory = directory("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_FALSE(resolution.path.has_parent_path())
        << "resolved through a computed directory: " << resolution.path;
    EXPECT_EQ(resolution.diagnostics.find(root().string()), std::string::npos)
        << "a computed directory was attempted: " << resolution.diagnostics;
    closeResolved(resolution);
}

#if defined(__linux__)

// Secure execution trusts the programmatic override, not the invoker's environment.
TEST_F(TestBackendLibraryResolution, SecureExecutionStillHonoursTheProgrammaticOverride)
{
    BackendResolutionInputs inputs;
    inputs.secureExecution = true;
    inputs.overrideDirectory = directoryWithLoadableBackend("override");
    inputs.overrideSource = "setBackendLibraryPath()";
    inputs.selfDirectory = directoryWithLoadableBackend("self");
    inputs.hipAnchorDirectory = directoryWithLoadableBackend("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, *inputs.overrideDirectory / backendFileName());
    EXPECT_EQ(resolution.diagnostics.find(inputs.selfDirectory.string()), std::string::npos)
        << "a derived tier was attempted: " << resolution.diagnostics;
    closeResolved(resolution);
}

#endif // defined(__linux__)

namespace
{

/// The whole one-shot lifecycle, exercised where nothing has resolved yet.
/// The override directories are left empty on purpose, so this resolution cannot
/// cache a stand-in backend. Everything this process owns is destroyed in the scope
/// above the explicit exit, and any failed expectation becomes a nonzero status.
[[noreturn]] void runSetterLifecycle()
{
    int status = 0;
    {
        const ScopedDirectory early(uniqueResolutionRoot("early"));
        const ScopedDirectory late(uniqueResolutionRoot("late"));

        const auto require = [&status](bool held, const char* what) {
            if(!held)
            {
                status = 1;
                std::fprintf(stderr, "setter lifecycle: %s\n", what);
            }
        };

        require(hipdnn_frontend::setBackendLibraryPath(early.path()),
                "the setter was refused before resolution had run");

        hipdnn_frontend::detail::resolveBackendLibraryPath();
        const auto& resolution = hipdnn_frontend::detail::backendLibraryResolution();
        std::fprintf(stderr, "candidates tried:%s\n", resolution.diagnostics.c_str());
        require(resolution.diagnostics.find(pathForDiagnostic(early.path() / backendFileName()))
                    != std::string::npos,
                "the stored override was never offered as a candidate");

        require(!hipdnn_frontend::setBackendLibraryPath(late.path()),
                "the setter was accepted after resolution had run");
    }
    std::exit(status);
}

} // namespace

// Resolution is a per-process one-shot, so a second iteration in the same process would
// find it already started and report correct behaviour as a regression. Each iteration
// runs the lifecycle in a fresh process instead: threadsafe death-test style re-execs
// this binary, and the child's exit status carries its verdict back.
// The suite name ends in Death rather than DeathTest because hipDNN's test-name
// validator reserves "Test" for the leading keyword. Ordering protection comes from
// the threadsafe style set below, which re-execs instead of forking this process.
TEST(TestBackendLibraryResolutionDeath, SetterIsRefusedOnceResolutionHasRun)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // EXPECT_EXIT expands to a switch over AssumeRole() carrying no default label. The
    // diagnostic is attributed to this expansion site rather than to the GoogleTest
    // header, so -isystem does not suppress it and -Werror makes it fatal.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-default"
    EXPECT_EXIT(runSetterLifecycle(), testing::ExitedWithCode(0), "candidates tried:");
#pragma clang diagnostic pop
}

#ifdef _WIN32

// A Windows path can hold UTF-16 with no UTF-8 spelling. Formatting such a candidate
// for the diagnostic must not escape the candidate loop: that would abandon every
// later candidate and cache the failure for the life of the process.
TEST_F(TestBackendLibraryResolution, MalformedNativeCandidateDoesNotAbortResolution)
{
    // An unpaired high surrogate: a legal Windows filename, not convertible to UTF-8.
    const std::filesystem::path malformed = root() / (L"malformed_" + std::wstring(1, L'\xD800'));
    ASSERT_TRUE(malformed.is_absolute());

    BackendResolutionInputs inputs;
    inputs.overrideDirectory = malformed;
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";
    inputs.selfDirectory = directoryWithUnloadableBackend("self");

    BackendLibraryResolution resolution;
    ASSERT_NO_THROW(resolution = resolveBackendLibrary(inputs));

    EXPECT_NE(
        resolution.diagnostics.find(pathForDiagnostic(inputs.selfDirectory / backendFileName())),
        std::string::npos)
        << "the candidate after the unconvertible one was never reached: "
        << resolution.diagnostics;
    closeResolved(resolution);
}

// A path that does convert must still reach the diagnostic as its own UTF-8 bytes,
// not as an active-code-page approximation of them.
TEST_F(TestBackendLibraryResolution, NonAsciiCandidateIsReportedAsUtf8)
{
    const std::filesystem::path missing = root() / std::wstring(L"\u6D4B\u8BD5_\u0416_\u03A9");

    BackendResolutionInputs inputs;
    inputs.overrideDirectory = missing;
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    // Converted with the Win32 API directly, independently of the helper under test.
    const std::wstring candidate = (missing / backendFileName()).wstring();
    const int size = WideCharToMultiByte(CP_UTF8,
                                         0,
                                         candidate.c_str(),
                                         static_cast<int>(candidate.size()),
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr);
    ASSERT_GT(size, 0);
    std::string utf8(static_cast<size_t>(size), '\0');
    ASSERT_GT(WideCharToMultiByte(CP_UTF8,
                                  0,
                                  candidate.c_str(),
                                  static_cast<int>(candidate.size()),
                                  utf8.data(),
                                  size,
                                  nullptr,
                                  nullptr),
              0);

    EXPECT_NE(resolution.diagnostics.find(utf8), std::string::npos) << resolution.diagnostics;
    closeResolved(resolution);
}

#endif // _WIN32

#if defined(__linux__)

TEST_F(TestBackendLibraryResolution, ExecutableAddressUsesTheExecutableDirectory)
{
    namespace utilities = hipdnn_data_sdk::utilities;
    EXPECT_EQ(utilities::getLoadedLibraryDirectoryForAddress(
                  reinterpret_cast<const void*>(&closeResolved)),
              utilities::getCurrentExecutableDirectory());
}

TEST_F(TestBackendLibraryResolution, OverrideOutranksEveryOtherTier)
{
    BackendResolutionInputs inputs;
    inputs.overrideDirectory = directoryWithLoadableBackend("override");
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";
    inputs.selfDirectory = directoryWithLoadableBackend("self");
    inputs.hipAnchorDirectory = directoryWithLoadableBackend("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, *inputs.overrideDirectory / backendFileName());
    closeResolved(resolution);
}

TEST_F(TestBackendLibraryResolution, SelfDirectoryOutranksHipAnchor)
{
    BackendResolutionInputs inputs;
    inputs.selfDirectory = directoryWithLoadableBackend("tree/bin");
    inputs.hipAnchorDirectory = directoryWithLoadableBackend("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, inputs.selfDirectory / backendFileName());
    closeResolved(resolution);
}

TEST_F(TestBackendLibraryResolution, SelfParentLibOutranksHipAnchor)
{
    BackendResolutionInputs inputs;
    inputs.selfDirectory = directory("tree/bin");
    const std::filesystem::path siblingLib = directoryWithLoadableBackend("tree/lib");
    inputs.hipAnchorDirectory = directoryWithLoadableBackend("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, siblingLib / backendFileName());
    closeResolved(resolution);
}

TEST_F(TestBackendLibraryResolution, SelfParentLib64OutranksHipAnchor)
{
    BackendResolutionInputs inputs;
    inputs.selfDirectory = directory("tree/bin");
    const std::filesystem::path siblingLib64 = directoryWithLoadableBackend("tree/lib64");
    inputs.hipAnchorDirectory = directoryWithLoadableBackend("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, siblingLib64 / backendFileName());
    closeResolved(resolution);
}

TEST_F(TestBackendLibraryResolution, SelfParentLibOutranksLib64)
{
    BackendResolutionInputs inputs;
    inputs.selfDirectory = directory("tree/bin");
    const std::filesystem::path siblingLib = directoryWithLoadableBackend("tree/lib");
    directoryWithLoadableBackend("tree/lib64");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, siblingLib / backendFileName());
    closeResolved(resolution);
}

TEST_F(TestBackendLibraryResolution, HipAnchorUsedWhenNoSelfRelativeCandidateExists)
{
    BackendResolutionInputs inputs;
    inputs.selfDirectory = directory("tree/bin");
    inputs.hipAnchorDirectory = directoryWithLoadableBackend("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, inputs.hipAnchorDirectory / backendFileName());
    closeResolved(resolution);
}

TEST_F(TestBackendLibraryResolution, AbsentCandidateIsSkippedAndReported)
{
    BackendResolutionInputs inputs;
    inputs.overrideDirectory = directory("override");
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";
    inputs.selfDirectory = directoryWithLoadableBackend("self");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, inputs.selfDirectory / backendFileName());
    EXPECT_NE(resolution.diagnostics.find((*inputs.overrideDirectory / backendFileName()).string()
                                          + ": not present"),
              std::string::npos)
        << resolution.diagnostics;
    closeResolved(resolution);
}

// A corrupt override must not prevent fallback to a usable backend.
TEST_F(TestBackendLibraryResolution, UnloadableCandidateFallsThroughToTheNextTier)
{
    BackendResolutionInputs inputs;
    inputs.overrideDirectory = directoryWithUnloadableBackend("override");
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";
    inputs.selfDirectory = directoryWithLoadableBackend("self");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, inputs.selfDirectory / backendFileName());
    EXPECT_NE(resolution.diagnostics.find((*inputs.overrideDirectory / backendFileName()).string()),
              std::string::npos)
        << resolution.diagnostics;
    closeResolved(resolution);
}

TEST_F(TestBackendLibraryResolution, DuplicateDirectoriesFallThroughToHipAnchor)
{
    BackendResolutionInputs inputs;
    inputs.overrideDirectory = directoryWithUnloadableBackend("shared");
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";
    inputs.selfDirectory = *inputs.overrideDirectory;
    inputs.hipAnchorDirectory = directoryWithLoadableBackend("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    EXPECT_EQ(resolution.path, inputs.hipAnchorDirectory / backendFileName());
    closeResolved(resolution);
}

#endif // defined(__linux__)

TEST_F(TestBackendLibraryResolution, UnloadableCandidatesEverywhereNeverBecomeTheAnswer)
{
    BackendResolutionInputs inputs;
    inputs.overrideDirectory = directoryWithUnloadableBackend("override");
    inputs.overrideSource = "HIPDNN_BACKEND_LIBRARY_PATH";
    inputs.selfDirectory = directoryWithUnloadableBackend("self");
    inputs.hipAnchorDirectory = directoryWithUnloadableBackend("hip");

    const BackendLibraryResolution resolution = resolveBackendLibrary(inputs);

    // The loader may still succeed, but no unloadable candidate may be adopted.
    EXPECT_FALSE(resolution.path.has_parent_path()) << resolution.path;
    closeResolved(resolution);
}
