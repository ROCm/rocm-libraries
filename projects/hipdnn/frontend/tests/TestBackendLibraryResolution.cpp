// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#if defined(__linux__)
#include <link.h>
#endif

namespace
{

using hipdnn_frontend::detail::BackendLibraryResolution;
using hipdnn_frontend::detail::BackendResolutionInputs;
using hipdnn_frontend::detail::resolveBackendLibrary;

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
        std::error_code failed;
        const auto* const info = ::testing::UnitTest::GetInstance()->current_test_info();
        _root = std::filesystem::temp_directory_path(failed)
                / (std::string("hipdnn_resolution_") + info->name());
        ASSERT_FALSE(failed) << failed.message();

        std::filesystem::remove_all(_root, failed);
        ASSERT_TRUE(std::filesystem::create_directories(_root, failed)) << failed.message();
    }

    void TearDown() override
    {
        std::error_code failed;
        std::filesystem::remove_all(_root, failed);
    }

    std::filesystem::path directory(const std::string& name)
    {
        const std::filesystem::path created = _root / name;
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
        return _root;
    }

private:
    std::filesystem::path _root;
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

// Use an empty override so this one-shot resolution cannot cache a stand-in backend.
TEST_F(TestBackendLibraryResolution, SetterIsRefusedOnceResolutionHasRun)
{
    EXPECT_TRUE(hipdnn_frontend::setBackendLibraryPath(directory("early")));

    hipdnn_frontend::detail::resolveBackendLibraryPath();

    EXPECT_FALSE(hipdnn_frontend::setBackendLibraryPath(directory("late")));
}

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
