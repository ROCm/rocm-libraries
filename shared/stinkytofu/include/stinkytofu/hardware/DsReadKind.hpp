// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace stinkytofu {

/// Hardware-timing classes for LDS read instructions.
///
/// This is intentionally independent of ISA opcodes so hardware models can
/// select type-specific drain formulas without depending on the asm IR.
enum class DsReadKind : uint8_t {
    Unknown,
    B32,
    B64,
    B128,
    Tr8B64,
    Tr16B128,
};

}  // namespace stinkytofu
