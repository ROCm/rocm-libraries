// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// What CoissueRepairPass predicted the passes after it would leave in each loop it looked
// at, kept for CoissueAuditPass to compare with what they really left. Debug only: the
// repair records only when asked to.

#include <optional>
#include <string>
#include <vector>

namespace stinkytofu {
struct StinkyInstruction;
}

namespace stinkytofu::coissue {

/// An instruction as the audit compares it: the mnemonic, plus the operand an inserting pass
/// chooses (bank-switch immediate, s_wait_alu fields, s_nop count).
std::string auditSignature(const StinkyInstruction& inst);

struct PredictedLoop {
    std::vector<std::string> signatures;
};

/// Record the prediction for loop `header` of `function`, replacing an older one.
void recordPrediction(const std::string& function, const std::string& header,
                      PredictedLoop prediction);
/// The recorded prediction, removed from the log.
std::optional<PredictedLoop> takePrediction(const std::string& function, const std::string& header);

}  // namespace stinkytofu::coissue
