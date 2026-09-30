// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace stinkytofu {

/// Declares one region-clone job for RegionClonePass.
///
/// Set from the producer side (e.g. Tensile/rocisa) and shipped to the pass via
/// ModuleOptions::CloneList. The producer names the start BB; RegionClonePass
/// computes the region end in-pass and dispatches a per-kind post-clone
/// transform keyed on `name`.
///
/// Example:
///     CloneSpec{"InitCIterWmma", "LoopBeginL"}
/// means: from the BB labeled "LoopBeginL", clone up to the last chain head
/// (the last MFMA whose src C acc first appears), then run the "InitCIterWmma"
/// post-clone transform (zero src C on each chain-head MFMA in the clone).
struct CloneSpec {
    std::string name;        ///< Recipe key for postCloneTransform dispatch
    std::string startLabel;  ///< Label name (LabelData::label) marking region start
};

}  // namespace stinkytofu
