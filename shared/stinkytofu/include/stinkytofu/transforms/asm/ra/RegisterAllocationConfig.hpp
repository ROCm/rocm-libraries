// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// One spelling for every register-allocation setting.
//
// TensileLite's StinkyTofuRegisterAllocation kernel parameter, the gfx1250
// pipeline and stinkytofu-opt's --RegisterAllocationPass all read the same
// comma-separated list, so a configuration tried in one runs unchanged in the
// others:
//
//     apply,classes=vs,allocator=greedy-compact-program,unbankable=hold
//
// The first word may be a bare mode (off, shadow, apply, force, or 0..3); the
// rest are `key=value` pairs and flags. docs/developer/register-allocation.md,
// "Configuring allocation", lists every key.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"

namespace stinkytofu {

/// What register allocation does to a kernel. The numbering is TensileLite's.
enum class RegisterAllocationMode : int {
    Off = 0,     ///< No allocation passes run.
    Shadow = 1,  ///< Colour and report; operands keep the producer's registers.
    Apply = 2,   ///< Write the colouring to the operands.
    Force = 3,   ///< Apply; TensileLite also re-judges an over-budget kernel afterwards.
};

/// Who wrote the configuration. It decides the defaults and the accepted keys.
enum class RegisterAllocationDialect {
    /// A kernel's StinkyTofuRegisterAllocation value, as the pipeline runs it.
    /// Defaults: mode off, classes vs, allocator
    /// greedy-compact+greedy-compact-largest+greedy-compact-program. The keys that
    /// only shape diagnostics (report, emitRegisterMap, emitSymbolBreadcrumbs,
    /// dumpBlocks) are refused, because the pipeline sets those itself.
    Kernel,
    /// stinkytofu-opt's --RegisterAllocationPass arguments. Defaults: mode
    /// shadow, classes v, allocator greedy, as the tool always had. Every key is
    /// accepted.
    Tool,
};

struct RegisterAllocationConfig {
    RegisterAllocationMode mode = RegisterAllocationMode::Off;

    /// What the pass is given. `options.allocate` is also the set of classes
    /// to lift, and `options.applyToOperands` follows `mode`.
    RegisterAllocationOptions options;
};

/// Parse \p tokens, the words of the list already split on commas. An empty
/// list is the dialect's defaults. A word the dialect does not accept, a value
/// a key does not accept, or two different modes is an error naming the word.
STINKYTOFU_EXPORT Expected<RegisterAllocationConfig> parseRegisterAllocationConfig(
    const std::vector<std::string>& tokens, RegisterAllocationDialect dialect);

/// Parse \p text, the comma-separated list. Blanks around a word are ignored.
STINKYTOFU_EXPORT Expected<RegisterAllocationConfig> parseRegisterAllocationConfig(
    std::string_view text, RegisterAllocationDialect dialect = RegisterAllocationDialect::Kernel);

/// The configuration a kernel runs under: \p text in the Kernel dialect, with
/// \p mode in place of whatever mode \p text names. TensileLite works the mode
/// out itself -- it turns allocation off where StinkyTofu does not run -- so
/// the number it passes is the one that holds. An empty \p text is every
/// default at that mode.
STINKYTOFU_EXPORT Expected<RegisterAllocationConfig> kernelRegisterAllocationConfig(
    int mode, std::string_view text);

/// The shortest spelling that parses back to \p config in the Kernel dialect.
///
/// The mode comes first as a digit, then only the settings that differ from
/// their Kernel defaults, in a fixed order, with list items sorted. So a
/// configuration that is only a mode is just the digit ("apply" is "2"), and
/// two spellings of one configuration canonicalize alike. Mode off is "0"
/// whatever else is set, since nothing runs. Keys the Kernel dialect refuses
/// are not part of a kernel's configuration and are left out.
STINKYTOFU_EXPORT std::string canonicalRegisterAllocationConfig(
    const RegisterAllocationConfig& config);

/// True when a kernel's \p mode and \p text, read as kernelRegisterAllocationConfig()
/// reads them, write new registers into operands of \p regClass: the mode
/// applies and the class is allocated. Passes that run after allocation ask
/// this to learn whether the producer's register numbering still holds. A
/// configuration that does not parse runs no allocation, so it rewrites nothing.
STINKYTOFU_EXPORT bool kernelRewritesRegisters(int mode, std::string_view text, RegType regClass);

/// How the configuration spells \p policy: "hold", "allocate" or "auto".
STINKYTOFU_EXPORT const char* unbankableOperandsName(
    RegisterAllocationOptions::UnbankableOperands policy);

/// Parse a class list such as "vs", "v" or "s". Empty, or any other letter,
/// is nullopt, so a typo is reported rather than selecting something else.
STINKYTOFU_EXPORT std::optional<RegClassSet> parseRegClasses(std::string_view spec);

/// Parse "s0" as one register or "s0:19" as the inclusive run s0..s19.
/// Anything else is nullopt, so a typo is reported rather than holding
/// registers nobody named.
STINKYTOFU_EXPORT std::optional<AllocationScope::HeldRange> parseHeldRange(std::string_view spec);

}  // namespace stinkytofu
