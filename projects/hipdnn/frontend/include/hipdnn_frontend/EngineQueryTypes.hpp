// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// Internal generation-tool types. These decode the ENGINE/ENGINECFG inspection
/// attributes read by hipdnn_frontend::detail::getEngineCandidates and
/// hipdnn_frontend::detail::getEnginePrediction. They are not part of the public
/// Graph API and carry no Python bindings: consumers select engines through the
/// heuristic descriptor, not by walking a kernel catalog (RFC 0017 §2,
/// RFC 0019 Open Question 12).

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn_frontend
{

/** @brief Scope of a calibrated engine performance prediction. */
enum class PredictionKind
{
    ENGINE = 0, ///< Predict ordinary no-search execution for the engine.
    CONFIGURATION = 1, ///< Predict and identify an exact engine configuration.
};

/** @brief Availability of a prediction, independent of engine applicability. */
enum class PredictionStatus
{
    UNAVAILABLE = 0, ///< No applicable calibrated model is deployed.
    AVAILABLE = 1, ///< A finite nonnegative calibrated TFLOPS estimate is available.
    INVALID = 2, ///< The model or its binding is invalid or incompatible.
};

/** @brief Calibrated prediction and its model/feature provenance, independent of applicability. */
struct EnginePrediction
{
    int64_t engineId = -1; ///< Queried engine.
    PredictionKind kind = PredictionKind::ENGINE; ///< Prediction scope.
    PredictionStatus status = PredictionStatus::UNAVAILABLE; ///< Model availability.
    std::optional<double> tflops; ///< Physical TFLOPS, present only when available.
    std::string model; ///< UHD model identity, empty if none is bound.
    std::string reason; ///< Explanation when the prediction is not available.
    nlohmann::json binding = nlohmann::json::object(); ///< Provenance; describe to request it.
    nlohmann::json features = nlohmann::json::object(); ///< Feature map; describe to request it.
    std::optional<EngineVariant> configuration; ///< Exact scored configuration, if available.
};

/// One catalog candidate. Enroll `variant` through add_engine_variants(); the
/// complete knob tuple selects exactly this ID on this graph/device snapshot.
struct EngineCandidate
{
    std::string id;
    EngineVariant variant;
    nlohmann::json kernelFeatures = nlohmann::json::object();
};

/// Bounded matched-catalog page. Unsupported enumeration is an Error, not an
/// empty successful page. Keep graph/device identity fixed across a page walk.
struct EngineCandidatePage
{
    int64_t engineId = -1;
    std::string engineName;
    std::string engineDescriptorId; ///< UED UUID, empty for non-descriptor engines.
    std::string graphId;
    std::string deviceId;
    std::string deviceArch;
    nlohmann::json problemFeatures = nlohmann::json::object();
    nlohmann::json deviceFeatures = nlohmann::json::object();
    uint64_t totalCount = 0;
    uint64_t offset = 0;
    std::optional<uint64_t> nextOffset;
    std::vector<EngineCandidate> candidates;
};

} // namespace hipdnn_frontend
