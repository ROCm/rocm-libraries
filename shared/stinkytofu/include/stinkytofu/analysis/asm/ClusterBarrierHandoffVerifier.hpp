// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <string>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Function;
class Pass;

/// Verify the final cluster wait -> workgroup signal/wait -> cluster signal
/// ordering on reachable control-flow paths. Empty means valid. Scalar eq/ne
/// guards may exclude paths only while their register facts remain valid.
/// This checks the local handoff, not cluster arrival counts or workgroup-uniform
/// participation. Known calls may be crossed only when their bodies are barrier-free.
STINKYTOFU_EXPORT std::string verifyClusterBarrierHandoffs(Function& func);
STINKYTOFU_EXPORT std::unique_ptr<Pass> createClusterBarrierHandoffVerifierPass();
}  // namespace stinkytofu
