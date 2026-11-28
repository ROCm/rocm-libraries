// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// An existing, empty directory this test process owns, removed when the process exits.
///
/// Anchor a KPACK descriptor here (originDirectory and treeRoot) to name an archive that
/// does not exist: its `library` resolves inside the tree, so the containment and link
/// checks pass and the checks after them run, and loading then fails on the missing file.
/// Any path below it is also guaranteed absent.
///
/// A fixed path such as "/nonexistent" cannot stand in for this. Some hosts have it (it is
/// the home directory of service accounts), and where it exists but is unreadable, path
/// resolution fails before the check under test is reached. A shared directory such as
/// temp_directory_path() is readable, but nothing keeps the archive's name out of it.
inline const std::filesystem::path& emptyDescriptorTree()
{
    static const hipdnn_test_sdk::utilities::ScopedDirectory s_tree
        = hipdnn_test_sdk::utilities::claimScratchDirectory("emptydescriptortree");
    return s_tree.path();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing
