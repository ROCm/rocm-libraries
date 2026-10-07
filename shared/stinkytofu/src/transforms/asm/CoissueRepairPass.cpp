// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/CoissueRepairPass.hpp"

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "coissue/TimingProfile.hpp"

namespace stinkytofu {

PassFeatureConfig::CoissueFeatures coissueFeaturesFromModuleOptions(
    const StinkyAsmModule::ModuleOptions& options) {
    PassFeatureConfig::CoissueFeatures f;
    f.repairMode = options.CoissueRepairMode;
    f.marginPercent = options.CoissueMarginPercent;
    f.profileSet = options.CoissueProfileSet;
    f.maxMoves = options.CoissueMaxMoves;
    f.searchRadius = options.CoissueSearchRadius;
    f.trustUncalibrated = options.CoissueTrustUncalibrated;
    f.patterns = options.CoissuePatterns;
    f.waitcntIssueCycles = options.CoissueWaitcntIssueCycles;
    f.waitcntSettleCycles = options.CoissueWaitcntSettleCycles;
    f.issueCycles = options.CoissueIssueCycles;
    f.scalarLatency = options.CoissueScalarLatency;
    f.matrixQueueDepth = options.CoissueMatrixQueueDepth;
    return f;
}

std::optional<std::string> validateCoissueFeatures(
    const PassFeatureConfig::CoissueFeatures& features, const std::array<int, 3>& arch) {
    if (features.repairMode != "off" && features.repairMode != "shadow" &&
        features.repairMode != "apply")
        return "CoissueRepairMode: '" + features.repairMode + "' is not off, shadow or apply";
    if (!(features.marginPercent >= 0.0))
        return "CoissueMarginPercent: must be >= 0, got " + std::to_string(features.marginPercent);
    if (features.maxMoves < 0) return "CoissueMaxMoves: must be >= 0";
    if (features.searchRadius < 0) return "CoissueSearchRadius: must be >= 0";
    if (features.waitcntIssueCycles < -1) return "CoissueWaitcntIssueCycles: must be >= -1";
    if (features.waitcntSettleCycles < -1) return "CoissueWaitcntSettleCycles: must be >= -1";
    if (features.matrixQueueDepth < -1) return "CoissueMatrixQueueDepth: must be >= -1";
    coissue::ProfileSet set;
    return coissue::resolveProfileSet(hwModelForArch(arch), getGfxArchID(arch[0], arch[1], arch[2]),
                                      features, set);
}

std::vector<std::string> describeCoissueProfiles(const PassFeatureConfig::CoissueFeatures& features,
                                                 const std::array<int, 3>& arch) {
    const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
    coissue::ProfileSet set;
    if (auto err = coissue::resolveProfileSet(hwModelForArch(arch), archId, features, set))
        return {*err};
    std::vector<std::string> lines;
    for (const coissue::TimingProfile& p : set.profiles) lines.push_back(coissue::describe(p, archId));
    return lines;
}

}  // namespace stinkytofu
