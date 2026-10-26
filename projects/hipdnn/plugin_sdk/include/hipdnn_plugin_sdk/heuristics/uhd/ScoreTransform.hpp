// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace hipdnn_plugin_sdk::uhd
{

/// @brief Score transform utilities (RFC 0019 §5, §12.3).
///
/// The set is closed: `isSupported` rejects any other name at parse and at L1 binding, so a
/// transformed number is never reported as if it were in `score.metric`'s units.
namespace score_transform
{

/// Transform names this runtime can invert; empty and "identity" mean the raw target. Must
/// cover every `score.transform` the UHD format allows (RFC 0019 §4).
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

/// Inverts @p transform to recover the score in `score.metric`'s units. Unknown names pass
/// through unchanged; `isSupported` keeps them from reaching here.
inline double applyInverse(double rawScore, const std::string& transform)
{
    if(transform == "log1p")
    {
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
        // A negative prediction is outside sqrt's range; squaring would rank it as positive.
        return rawScore < 0.0 ? std::numeric_limits<double>::quiet_NaN() : rawScore * rawScore;
    }
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

/// Whether the recovered score is a physical quantity, which RFC 0019 §8.3 accepts only when
/// strictly positive: true when a metric is declared or the transform admits only positive
/// targets. uhd_gen's offline evaluators apply the same predicate.
inline bool isPhysicalScore(const std::string& metric, const std::string& transform)
{
    return !metric.empty() || transform == "log" || transform == "log1p" || transform == "sqrt";
}

/// RFC 0019 §8.3: rankable only when finite, and strictly positive when @p positiveRequired.
inline bool isRankableScore(double recovered, bool positiveRequired)
{
    return std::isfinite(recovered) && (!positiveRequired || recovered > 0.0);
}

} // namespace score_transform

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
