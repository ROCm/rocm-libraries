/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <filesystem>
#include <stdexcept>
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

#ifdef HIPKERNELPROVIDER_TEST_SET_PACKED_FIXTURE_RELDIR
/// The packed-fixture set, holding one subdirectory per packed arch.
inline const std::filesystem::path& packedFixtureRoot()
{
    static const std::filesystem::path root
        = descriptorSetRoot(HIPKERNELPROVIDER_TEST_SET_PACKED_FIXTURE_RELDIR);
    return root;
}
#endif

} // namespace hip_kernel_provider::testing
