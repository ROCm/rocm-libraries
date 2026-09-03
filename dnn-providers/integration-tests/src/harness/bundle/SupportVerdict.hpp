// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_frontend/Error.hpp>

#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

struct LoadedEngine;

enum class SupportVerdict
{
    NO_SIDECAR,
    SATISFIED,
    CLAIM_BROKEN,
    QUERY_ERRORED,
    ENGINE_NOT_LOADED,
    NOT_ENFORCED,
    UNCLAIMED_SUPPORT,
};

const char* toString(SupportVerdict verdict);

bool isFailure(SupportVerdict verdict);

struct SupportResult
{
    SupportVerdict verdict;
    std::string bundlePath;
    std::string engineName;
    std::string arch;
    std::string platform;
    std::string detail;

    hipdnn_frontend::ErrorCode queryStatus = hipdnn_frontend::ErrorCode::OK;
    std::string queryMessage;
};

SupportResult evaluateSupport(hipdnn_frontend::ErrorCode errorCode,
                              const std::vector<int64_t>& rankedIds,
                              int64_t engineId,
                              bool claimed,
                              bool hasSidecar,
                              const std::string& bundlePath,
                              const std::string& engineName,
                              const std::string& arch,
                              const std::string& platform,
                              std::string_view queryMessage = {});

std::string baseArchToken(std::string_view fullArch);
std::string formatVerdictMessage(const SupportResult& result);

std::vector<SupportResult> observeAllSupport(hipdnn_frontend::ErrorCode errorCode,
                                             const std::vector<int64_t>& rankedIds,
                                             const SupportClaimLocator& locator,
                                             const std::vector<LoadedEngine>& loadedEngines,
                                             std::string_view queryMessage = {});

} // namespace hipdnn_integration_tests::bundle
