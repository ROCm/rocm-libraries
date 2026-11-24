// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

/**
 * @file UnpulledModelsRecord.hpp
 * @brief Reads hkp_pack's skip record: the trained models (UHDs) a pack left out because their
 *        `model.bin` was never pulled from DVC. The census and the GPU integration tests both
 *        accept a missing model only when this record names it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The skip record hkp_pack writes at the packed arch root when it leaves a UHD out for an
/// un-pulled model (hkp_pack.pipeline.UNPULLED_RECORD).
inline constexpr const char* UNPULLED_RECORD = "hkp-unpulled-models.record";

/// Whether the skip record at @p archRoot names @p uhdId. The packer writes the record only
/// when it skips, so a UHD or model the packer lost for any other reason is in no record.
inline bool unpulledRecordNames(const std::filesystem::path& archRoot, const std::string& uhdId)
{
    std::ifstream file(archRoot / UNPULLED_RECORD);
    if(!file)
    {
        return false;
    }
    const auto record = nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
    if(!record.is_object() || !record.contains("unpulled_models")
       || !record["unpulled_models"].is_array())
    {
        return false;
    }
    for(const auto& entry : record["unpulled_models"])
    {
        if(entry.is_object() && entry.value("uhd_id", std::string()) == uhdId)
        {
            return true;
        }
    }
    return false;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing
