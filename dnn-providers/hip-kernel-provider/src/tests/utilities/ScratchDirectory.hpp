// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <filesystem>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

namespace hip_kernel_provider::tests
{

/// Claims a scratch directory under the temp dir that no other run can be holding.
///
/// ScopedDirectory refuses to adopt an existing directory, so a fixed name fails against
/// any leftover or concurrent copy of the suite. The counter walks past whatever is there,
/// and create_directory is atomic, so testing a name and taking it are one step. Retry
/// rather than clear: the name may belong to a live process whose fixture would go with it.
///
/// @param label Names the calling suite, so a directory left behind by a crash says which
///              binary made it.
[[nodiscard]] inline hipdnn_test_sdk::utilities::ScopedDirectory
    claimScratchDirectory(const std::string& label)
{
    // Drawn per process, so concurrent runs start from different names rather than both
    // walking up from zero.
    static const unsigned s_session = std::random_device{}();
    static std::atomic<unsigned> s_counter{0};

    const std::filesystem::path base = std::filesystem::temp_directory_path();

    std::ostringstream prefix;
    prefix << "hkp_" << label << '_' << std::hex << s_session << '_';

    for(int attempt = 0; attempt < 64; ++attempt)
    {
        const std::filesystem::path candidate
            = base / (prefix.str() + std::to_string(s_counter.fetch_add(1)));
        try
        {
            return {candidate};
        }
        // filesystem_error derives from runtime_error, so it has to be caught first: an
        // unwritable temp directory fails identically 64 times and must not be reported as
        // name exhaustion.
        catch(const std::filesystem::filesystem_error&)
        {
            throw;
        }
        catch(const std::runtime_error&)
        {
            continue;
        }
    }
    throw std::runtime_error("claimScratchDirectory: no free scratch name under the temp dir");
}

} // namespace hip_kernel_provider::tests
