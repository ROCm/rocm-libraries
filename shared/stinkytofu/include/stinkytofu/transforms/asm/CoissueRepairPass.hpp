// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {

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
