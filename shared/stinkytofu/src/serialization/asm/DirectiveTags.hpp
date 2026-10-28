// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Internal use only: how STIR spells a directive the emitter writes as its text.

#pragma once

#include <string_view>
#include <utility>

#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"

namespace stinkytofu {

/// The emitter writes these directives as their `value`. STIR prints the tag in
/// place of a name, followed by that text: `"st.asm_directive"("TEXTBLOCK", "...")`.
inline constexpr std::pair<AsmDirectiveKind, std::string_view> kTextDirectiveTags[] = {
    {AsmDirectiveKind::TEXTBLOCK, "TEXTBLOCK"},
    {AsmDirectiveKind::MACRO, "MACRO"},
    {AsmDirectiveKind::IF, "IF"},
    {AsmDirectiveKind::ENDIF, "ENDIF"},
};

}  // namespace stinkytofu
