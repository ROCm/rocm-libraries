// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportObservationEmitter.hpp"

#include <array>
#include <chrono>
#include <ctime>
#include <fstream>
#include <system_error>

#include "harness/BundleMetadata.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

/// The sidecar's directory relative to the bundle root, POSIX-spelled — the key
/// the Python consumer indexes bundles by.
std::string bundleKey(const std::filesystem::path& sidecarPath,
                      const std::filesystem::path& bundleRoot)
{
    const auto directory = sidecarPath.parent_path();

    std::error_code ec;
    const auto relative = std::filesystem::relative(directory, bundleRoot, ec);

    // relative() reports "..\..\elsewhere" for a path outside the root rather
    // than failing, so both the error and the escape have to be caught. Either
    // way the absolute path goes out and the consumer warns; guessing a key
    // would be worse than an orphan, because a wrong key writes a wrong file.
    if(ec || relative.empty() || *relative.begin() == "..")
    {
        return directory.generic_string();
    }
    return relative.generic_string();
}

} // namespace

std::string currentUtcTimestamp()
{
    const auto seconds = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif

    std::array<char, 32> buffer{};
    const std::size_t written
        = std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return {buffer.data(), written};
}

nlohmann::json toObservationRecord(const SupportObservation& observation,
                                   const std::filesystem::path& bundleRoot,
                                   const ObservationProvenance& provenance)
{
    const auto& locator = observation.claimLocator;

    nlohmann::json record;
    record["bundle"] = bundleKey(locator.sidecarPath, bundleRoot);
    record["engine"] = observation.engineName;
    record["arch"] = observation.arch;
    record["platform"] = observation.platform;
    record["verdict"] = toString(observation.support);
    record["enforcement_level"] = toString(observation.enforcementLevel);

    record["case_id"]
        = locator.isSweep() ? nlohmann::json(locator.caseId) : nlohmann::json(nullptr);

    record["provenance"] = {{"rocm_version", provenance.rocmVersion},
                            {"commit", provenance.commit},
                            {"run_id", provenance.runId},
                            {"timestamp", provenance.timestamp}};

    return record;
}

EmitSummary emitSupportObservations(const std::vector<SupportObservation>& observations,
                                    const std::filesystem::path& outputPath,
                                    const std::filesystem::path& bundleRoot,
                                    const ObservationProvenance& provenance)
{
    EmitSummary summary;

    if(const auto parent = outputPath.parent_path(); !parent.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if(ec)
        {
            summary.errors.push_back("could not create directory " + parent.string() + ": "
                                     + ec.message());
            return summary;
        }
    }

    std::ofstream file(outputPath, std::ios::app);
    if(!file)
    {
        summary.errors.push_back("could not open for appending: " + outputPath.string());
        return summary;
    }

    for(const auto& observation : observations)
    {
        // dump() with no indent: one record, one line, which is the whole
        // contract of JSONL. nlohmann's default object is std::map-backed, so
        // keys come out sorted without asking.
        const std::string line = toObservationRecord(observation, bundleRoot, provenance).dump();

        file << line << "\n";
        ++summary.recordsEmitted;
    }

    file.flush();
    if(!file)
    {
        summary.errors.push_back("write failed: " + outputPath.string());
    }

    return summary;
}

} // namespace hipdnn_integration_tests::bundle
