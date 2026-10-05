// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// Internal generation-tool types decoded by the hipdnn_frontend::detail engine queries. Not
/// part of the public Graph API and not bound to Python (RFC 0019 Open Question 12).

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
    UNAVAILABLE = 0, ///< No applicable calibrated model is deployed for the requested metric.
    AVAILABLE = 1, ///< A valid calibrated value in the requested metric's units is available.
    INVALID = 2, ///< The model or its binding is invalid or incompatible.
};

/** @brief Calibrated prediction and its model/feature provenance, independent of applicability. */
struct EnginePrediction
{
    int64_t engineId = -1; ///< Queried engine.
    PredictionKind kind = PredictionKind::ENGINE; ///< Prediction scope.
    PredictionStatus status = PredictionStatus::UNAVAILABLE; ///< Model availability.
    std::string metric; ///< Registered ranking metric the query named (RFC 0019 §4.4).
    std::optional<double> value; ///< Value in @c metric's units, present only when available.
    std::string model; ///< UHD model identity, empty if none is bound.
    std::string reason; ///< Explanation when the prediction is not available.
    nlohmann::json binding = nlohmann::json::object(); ///< Provenance; describe to request it.
    nlohmann::json features = nlohmann::json::object(); ///< Feature map; describe to request it.
    std::optional<EngineVariant> configuration; ///< Exact scored configuration, if available.
};

/// A (kind, metric) prediction an engine can answer on this graph, i.e. a bound model. Says
/// nothing about whether evaluating it would currently succeed.
struct PredictionCapability
{
    PredictionKind kind = PredictionKind::ENGINE;
    std::string metric;
    std::string model; ///< UHD model identity bound for this kind and metric.
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
