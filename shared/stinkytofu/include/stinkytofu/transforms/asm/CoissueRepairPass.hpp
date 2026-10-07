// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {
class Pass;

/// Overrides of the Coissue* knobs for one pass instance (stinkytofu-opt arguments); unset
/// fields take PassFeatureConfig::CoissueFeatures.
struct CoissueRepairOptions {
    std::optional<std::string> mode;
    std::optional<double> marginPercent;
    std::optional<int> maxMoves;
    std::optional<int> searchRadius;
    std::optional<std::string> profileSet;
    std::optional<std::string> patterns;
    std::optional<bool> trustUncalibrated;
    /// InsertWaitAlu and the prefetch bridge run after the repair, with va_vsrc tracking.
    std::optional<bool> esm2;
    std::optional<bool> trackValuVsrc;
    /// Record each loop's prediction for CoissueAuditPass.
    bool audit = false;
    /// Also refuse a VALU move that lands within 13 matrix ops before, or 11 after, a memory
    /// op reading what it writes. That is the Python prototype's stand-in for the s_wait_alu
    /// model; only for comparing the engine with it.
    bool prototypeWaitAluRule = false;
};

/// Reorders the SALU/VALU fillers of each innermost loop that has matrix ops, so that what
/// the passes after it insert costs less matrix-pipe time. Off unless CoissueRepairMode
/// (or `options.mode`) is shadow or apply; see docs/developer/coissue-repair-pass.md.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createCoissueRepairPass(CoissueRepairOptions options = {});

/// Debug pass that reports the co-issue timing of the code as it stands. With
/// `printTimeline`, it prints the steady trip of every innermost loop that has matrix ops,
/// under each profile of PassFeatureConfig::CoissueFeatures::profileSet; with
/// `traceProfile`, also the issue cycle of every instruction under that profile.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createCoissueAuditPass(bool printTimeline = false,
                                                               std::string traceProfile = "");

/// The Coissue* module options, as the pass reads them.
STINKYTOFU_EXPORT PassFeatureConfig::CoissueFeatures coissueFeaturesFromModuleOptions(
    const StinkyAsmModule::ModuleOptions& options);

/// Check every Coissue* knob for `arch`; returns a message for the first bad one. A bad
/// spec is an error, never ignored.
STINKYTOFU_EXPORT std::optional<std::string> validateCoissueFeatures(
    const PassFeatureConfig::CoissueFeatures& features, const std::array<int, 3>& arch);

/// The profiles CoissueRepairPass times under for `arch` with these knobs, one line per
/// profile with every number it uses. This is the text of the pass's profile remark.
STINKYTOFU_EXPORT std::vector<std::string> describeCoissueProfiles(
    const PassFeatureConfig::CoissueFeatures& features, const std::array<int, 3>& arch);

}  // namespace stinkytofu
