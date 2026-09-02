/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hip/hip_runtime_api.h>

#include "TestDescriptorRoot.hpp"

namespace hip_kernel_provider::testing
{

/// The kpack coordinates one built descriptor declares, and where they were read from.
///
/// `library` is kept in its authored, relative form because a KernelDefinition carries it
/// that way and resolves it against originDirectory. `archive` is the same value already
/// resolved, for callers that open the file directly rather than through a definition.
struct PackedKernelSource
{
    std::string library;
    std::string tocKey;
    /// The descriptor's OWN directory, which `library` is relative to. Not the arch root:
    /// the packer preserves each descriptor's authored subpath, so the two differ for
    /// every nested descriptor.
    std::filesystem::path originDirectory;
    std::filesystem::path archive;
};

/// The bare arch of device 0 and the directory this build packed for it. `directory` is
/// left empty when nothing was packed for that arch -- environmental, not a broken build.
///
/// hipGetDeviceProperties reports feature flags on some configurations ("gfx1152:xnack-")
/// while the packager uses the bare name, so everything past here uses the stripped form.
///
/// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void findPackedArchDirectory(hipDeviceProp_t& properties,
                                    std::string& arch,
                                    std::filesystem::path& directory)
{
    ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);

    const std::string reported = properties.gcnArchName;
    arch = reported.substr(0, reported.find(':'));

    const std::filesystem::path candidate = unitKpackRoot() / arch;
    directory = std::filesystem::is_directory(candidate) ? candidate : std::filesystem::path{};
}

/// Reads `kernel_source` out of a built descriptor. A .kdp.json nests it under its first
/// inline kernel descriptor; a .ukd.json carries it at the top level. Parsed directly
/// rather than through DescriptorLoader, whose contract the integration tier covers.
///
/// Found by RECURSIVE search rather than a join on the arch root: the packer preserves
/// each descriptor's authored subpath, so a descriptor sits wherever its source root put
/// it. Searching by filename keeps callers indifferent to that depth, which is the point
/// -- a flat join is what made this suite blind to nesting.
///
/// Asserts rather than skips -- the per-arch directory exists by the time this is called,
/// so anything missing inside it is a broken build. Call through ASSERT_NO_FATAL_FAILURE.
inline void readPackedKernelSource(const std::filesystem::path& directory,
                                   const std::string& descriptorFile,
                                   PackedKernelSource& out)
{
    std::filesystem::path descriptor;
    std::error_code walkError;
    for(const auto& entry : std::filesystem::recursive_directory_iterator(directory, walkError))
    {
        if(entry.is_regular_file() && entry.path().filename() == descriptorFile)
        {
            descriptor = entry.path();
            break;
        }
    }
    ASSERT_FALSE(descriptor.empty()) << "the packed descriptor is missing anywhere under "
                                     << directory << ": " << descriptorFile;

    std::ifstream in(descriptor);
    ASSERT_TRUE(in.good()) << "could not open " << descriptor;

    nlohmann::json document;
    ASSERT_NO_THROW(document = nlohmann::json::parse(in)) << descriptor;

    const nlohmann::json& kernel
        = document.contains("kernelDescriptors") ? document["kernelDescriptors"][0] : document;
    ASSERT_TRUE(kernel.contains("kernel_source")) << descriptor;

    const nlohmann::json& source = kernel["kernel_source"];
    ASSERT_TRUE(source.contains("toc_key")) << descriptor;
    ASSERT_TRUE(source.contains("library")) << descriptor;

    out.tocKey = source["toc_key"].get<std::string>();
    out.library = source["library"].get<std::string>();
    out.originDirectory = descriptor.parent_path();
    // `library` is relative to the directory holding the descriptor that declared it --
    // the same anchoring KernelDefinition::originDirectory describes. That directory is
    // the descriptor's OWN parent, not the arch root, so a nested descriptor resolves
    // through the `..` segments the packer wrote.
    out.archive = out.originDirectory / out.library;
    ASSERT_TRUE(std::filesystem::exists(out.archive))
        << descriptor << " names an archive that is not on disk: " << out.archive;
}

} // namespace hip_kernel_provider::testing
