/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace hip_kernel_provider::testing
{

/// The descriptor set @p relativeSubdir names, resolved from this binary's own location.
///
/// The build tree lays the descriptor trees out under the same subdirectories the install
/// tree uses -- both derive from HIPDNN_PLUGIN_ROOTDIR and CMAKE_INSTALL_BINDIR -- so a
/// single binary-relative offset reaches them in either.
///
/// Measured from this function's own address rather than from argv[0] or the working
/// directory, so the answer does not depend on how the binary was invoked.
///
/// That address is always in the TEST EXECUTABLE, in both the unit and the integration
/// binary: the function is inline, so each one compiles its own copy. It is never the
/// provider plugin, even in the integration binary that dlopens it -- so @p relativeSubdir
/// is an offset from bin/, not from the engines directory the plugin sits in. The engine's
/// own lookup in KernelIngestorEngine.cpp measures from the plugin instead, and the two
/// land in different places.
///
/// Returns an empty path when there is no module to measure from; callers already test the
/// result for a directory and report their own absence.
inline std::filesystem::path descriptorSetRoot(const char* relativeSubdir)
{
    try
    {
        const auto resolved = hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(
                                  reinterpret_cast<const void*>(&descriptorSetRoot))
                              / relativeSubdir;
        std::error_code failed;
        auto collapsed = std::filesystem::weakly_canonical(resolved, failed);
        return failed ? resolved : collapsed;
    }
    catch(const std::runtime_error&)
    {
        return {};
    }
}

/// The reason @p root cannot serve as a descriptor root, or an empty string.
///
/// An absent or empty root gives a caller the same answer as a root holding nothing for
/// this device: every case skips and the binary reports success. A test main() calls this
/// and fails the process instead, so the two are told apart.
inline std::string describeUnusableDescriptorRoot(const std::filesystem::path& root)
{
    std::error_code failed;
    if(!std::filesystem::is_directory(root, failed))
    {
        return "the descriptor root '" + root.string() + "' is not a directory";
    }

    for(const auto& entry : std::filesystem::recursive_directory_iterator(root, failed))
    {
        if(entry.is_regular_file(failed) && entry.path().extension() == ".json")
        {
            return {};
        }
    }

    return "the descriptor root '" + root.string() + "' holds no descriptor JSON";
}

#ifdef HIPKERNELPROVIDER_UNIT_KPACK_RELDIR
/// The packed set inside this binary's own discovery root, holding one subdirectory per
/// packed arch, each with a real archive under `kpack/`.
///
/// The kpack cases need staged output that a packer actually compiled: an archive to open,
/// a toc_key to resolve against, and a descriptor naming both. `unit/pointwise` cannot
/// serve them -- it is authored in the embedded_source dialect, so the packer compiles
/// nothing for it and its arch folders hold no archive at all.
inline const std::filesystem::path& unitKpackRoot()
{
    static const std::filesystem::path root
        = descriptorSetRoot(HIPKERNELPROVIDER_UNIT_KPACK_RELDIR);
    return root;
}
#endif

} // namespace hip_kernel_provider::testing
