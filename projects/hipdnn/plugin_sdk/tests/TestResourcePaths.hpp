// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

/**
 * @file TestResourcePaths.hpp
 * @brief Files the SDK tests load, found from the running test binary.
 *
 * The build and install trees put these at the same offset from the binary, so an
 * installed run reads its own copies rather than the tree that built it, which an
 * artifact-only test runner does not have.
 */
namespace hipdnn_plugin_sdk::test
{

/// The test plugin directory: `HIPDNN_TEST_PLUGIN_RELDIR` from the binary's directory.
inline std::filesystem::path testPluginDir()
{
    return hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / HIPDNN_TEST_PLUGIN_RELDIR;
}

/// The scorer library built from heuristics/uhd/test_scorer_lib.cpp.
inline std::filesystem::path testScorerLibrary()
{
    return testPluginDir() / hipdnn_data_sdk::utilities::getLibraryName("hipdnn_test_scorer_lib");
}

/// The committed `uhd_gen` output from ingestor/uhd/fixtures that the runtime reads.
inline std::filesystem::path uhdGeneratedFixtureDir()
{
    return testPluginDir() / HIPDNN_UHD_GENERATED_FIXTURE_SUBDIR;
}

} // namespace hipdnn_plugin_sdk::test
