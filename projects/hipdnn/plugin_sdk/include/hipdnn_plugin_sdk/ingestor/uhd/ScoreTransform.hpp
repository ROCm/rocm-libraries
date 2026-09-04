// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace hipdnn_plugin_sdk::ingestor::uhd
{

/// @brief Score transform utilities (RFC 0019 §5, §12.3).
///
/// Models may be trained on a transformed target (e.g. log1p(tflops)). The declared
/// `score.transform` names that transform so a consumer can invert it and recover the
/// declared `score.units` — which is what makes cross-engine comparison meaningful.
///
/// The set of transforms is closed: a descriptor naming one we cannot invert is
/// rejected at load (see EngineRegistry::registerEngine) rather than silently
/// reporting a transformed number as if it were in the declared units.
namespace score_transform
{

/// Transform names this runtime can invert. An empty name and "identity" both mean
/// the model was trained on the raw target.
///
/// Kept in sync with the `score.transform` vocabulary a UHD descriptor may declare
/// (RFC 0019 §4) — a name the format advertises but this list omits is a descriptor
/// that passes review and then fails to load.
inline constexpr std::array<const char*, 6> SUPPORTED_TRANSFORMS
    = {"", "identity", "log1p", "log", "exp", "sqrt"};

/// Whether `transform` is a name this runtime can invert.
inline bool isSupported(const std::string& transform)
{
    for(const auto* known : SUPPORTED_TRANSFORMS)
    {
        if(transform == known)
        {
            return true;
        }
    }
    return false;
}

/// Comma-separated list of supported names, for diagnostics.
inline std::string supportedTransformList()
{
    std::string out;
    for(const auto* known : SUPPORTED_TRANSFORMS)
    {
        if(!out.empty())
        {
            out += ", ";
        }
        out += (*known == '\0') ? "\"\"" : known;
    }
    return out;
}

/// Apply inverse transform to recover original scale.
///
/// @param rawScore Score from the model.
/// @param transform Transform name from UhdConfig::scoreTransform. Must be one of
///        SUPPORTED_TRANSFORMS; unknown names fall through unchanged, which is only
///        safe because registration rejects them first.
/// @returns Score in the units declared by the UHD.
inline double applyInverse(double rawScore, const std::string& transform)
{
    if(transform == "log1p")
    {
        // Inverse of log1p is expm1
        return std::expm1(rawScore);
    }
    if(transform == "log")
    {
        return std::exp(rawScore);
    }
    if(transform == "exp")
    {
        return std::log(rawScore);
    }
    if(transform == "sqrt")
    {
        // Squaring is the inverse only on the domain sqrt actually produces. A negative
        // prediction is out of that domain, and squaring it silently maps it to a *positive*
        // score -- so a model predicting -0.5 outranks one predicting +0.25. NaN says
        // out-of-domain, which is what the log branches above already say for the same reason.
        return rawScore < 0.0 ? std::numeric_limits<double>::quiet_NaN() : rawScore * rawScore;
    }
    // "" or "identity": the model was trained on the raw target.
    return rawScore;
}

/// Apply forward transform (for training/debugging).
inline double applyForward(double value, const std::string& transform)
{
    if(transform == "log1p")
    {
        return std::log1p(value);
    }
    if(transform == "log")
    {
        return std::log(value);
    }
    if(transform == "exp")
    {
        return std::exp(value);
    }
    if(transform == "sqrt")
    {
        return std::sqrt(value);
    }
    return value;
}

} // namespace score_transform

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
