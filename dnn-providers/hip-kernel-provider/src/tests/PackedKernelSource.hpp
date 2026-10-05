/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

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
    std::string sha256;
};

/// The bare arch of device 0 and the directory this build packed for it under @p root.
/// `directory` is left empty when nothing was packed for that arch -- environmental, not a
/// broken build.
///
/// hipGetDeviceProperties reports feature flags on some configurations ("gfx1152:xnack-")
/// while the packager uses the bare name, so everything past here uses the stripped form.
///
/// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
inline void findPackedArchDirectoryUnder(const std::filesystem::path& root,
                                         hipDeviceProp_t& properties,
                                         std::string& arch,
                                         std::filesystem::path& directory)
{
    ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);

    arch = std::string(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));

    const std::filesystem::path candidate = root / arch;
    directory = std::filesystem::is_directory(candidate) ? candidate : std::filesystem::path{};
}

/// findPackedArchDirectoryUnder() for the packed set inside this binary's own discovery
/// root.
inline void findPackedArchDirectory(hipDeviceProp_t& properties,
                                    std::string& arch,
                                    std::filesystem::path& directory)
{
    findPackedArchDirectoryUnder(unitKpackRoot(), properties, arch, directory);
}

/// Reads `kernel_source` out of a built descriptor. A .kdp.json nests it under its first
/// inline kernel descriptor; a .ukd.json carries it at the top level. Parsed directly
/// rather than through DescriptorLoader, whose contract the integration tier covers.
///
/// Found by RECURSIVE search rather than a join on the arch root: the packer preserves
/// each descriptor's authored subpath, so a descriptor sits wherever its source root put
/// it. Searching by filename keeps callers indifferent to that depth.
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
    ASSERT_TRUE(source.contains("sha256")) << descriptor;

    out.tocKey = source["toc_key"].get<std::string>();
    out.library = source["library"].get<std::string>();
    // Read rather than recomputed: recomputing would compare the loader's hash against
    // this test's hash of the same bytes, which passes however wrong both are. The shipped
    // field is the claim the loader actually checks.
    out.sha256 = source["sha256"].get<std::string>();
    out.originDirectory = descriptor.parent_path();
    // `library` is relative to the directory holding the descriptor that declared it --
    // the same anchoring KernelDefinition::originDirectory describes. That directory is
    // the descriptor's OWN parent, not the arch root, so a nested descriptor resolves
    // through the `..` segments the packer wrote.
    out.archive = out.originDirectory / out.library;
    ASSERT_TRUE(std::filesystem::exists(out.archive))
        << descriptor << " names an archive that is not on disk: " << out.archive;
}

/// The kernel a built descriptor declares, as a definition a pack can prepare: its id, its
/// name, its integer and string metadata, and its `kernel_source` as the descriptor loader
/// parses it, with the recorded signature. `treeRoot` is @p directory, which the loader
/// stamps as the containment boundary.
///
/// Asserts rather than skips. Call through ASSERT_NO_FATAL_FAILURE.
inline void readPackedKernelDefinition(const std::filesystem::path& directory,
                                       const std::string& descriptorFile,
                                       hipdnn_plugin_sdk::ingestor::KernelDefinition& out)
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
    ASSERT_TRUE(kernel.is_object()) << descriptor;
    ASSERT_TRUE(kernel.contains("id") && kernel.contains("name")
                && kernel.contains("kernel_source"))
        << descriptor;

    ASSERT_NO_THROW(out.source = hipdnn_plugin_sdk::ingestor::detail::parseKernelSource(
                        kernel["kernel_source"], descriptor.string()))
        << descriptor;
    out.kernelId = hipdnn_flatbuffers_sdk::utilities::parseUuid(kernel["id"].get<std::string>());
    out.name = kernel["name"].get<std::string>();
    out.metadata.clear();
    if(kernel.contains("metadata"))
    {
        for(const auto& item : kernel["metadata"].items())
        {
            if(item.value().is_number_integer())
            {
                out.metadata[item.key()] = item.value().get<int64_t>();
            }
            else if(item.value().is_string())
            {
                out.metadata[item.key()] = item.value().get<std::string>();
            }
        }
    }
    out.originDirectory = descriptor.parent_path();
    out.treeRoot = directory;
}

} // namespace hip_kernel_provider::testing
