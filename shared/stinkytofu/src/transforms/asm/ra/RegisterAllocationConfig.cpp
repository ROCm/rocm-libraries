// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/ra/RegisterAllocationConfig.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/support/PassArguments.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"

namespace stinkytofu {
namespace {

using Mode = RegisterAllocationMode;
using Unbankable = RegisterAllocationOptions::UnbankableOperands;
using Quadrant = RegisterAllocationOptions::Quadrant;

// What a kernel gets when its configuration names no classes or allocator.
constexpr RegClassSet kKernelClasses = RegClassSet::all();
constexpr const char* kKernelAllocator =
    "greedy-compact+greedy-compact-largest+greedy-compact-program";

std::string_view trimmed(std::string_view text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    const size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::optional<Mode> modeNamed(std::string_view word) {
    if (word == "off" || word == "0") return Mode::Off;
    if (word == "shadow" || word == "1") return Mode::Shadow;
    if (word == "apply" || word == "2") return Mode::Apply;
    if (word == "force" || word == "3") return Mode::Force;
    return std::nullopt;
}

std::string classesName(RegClassSet classes) {
    std::string text;
    if (classes.contains(RegType::V)) text += "v";
    if (classes.contains(RegType::S)) text += "s";
    return text;
}

std::string heldRangeName(const AllocationScope::HeldRange& range) {
    std::string text = regTypeToString(range.regClass) + std::to_string(range.start);
    if (range.end != range.start) text += ":" + std::to_string(range.end);
    return text;
}

std::string joined(const std::vector<std::string>& items, const char* separator) {
    std::string text;
    for (const std::string& item : items) {
        if (!text.empty()) text += separator;
        text += item;
    }
    return text;
}

std::vector<std::string> sortedUnique(std::vector<std::string> items) {
    std::sort(items.begin(), items.end());
    items.erase(std::unique(items.begin(), items.end()), items.end());
    return items;
}

/// Every name in \p chain is registered, or a message naming the first that is not.
std::optional<std::string> unregisteredAllocator(const std::vector<std::string>& chain) {
    AllocatorRegistry::registerAllAllocators();
    const std::vector<std::string> names = AllocatorRegistry::registeredAllocatorNames();
    for (const std::string& name : chain) {
        if (std::find(names.begin(), names.end(), name) != names.end()) continue;
        return "allocator '" + name + "' is not registered; registered: " + joined(names, ", ");
    }
    return std::nullopt;
}

/// Keys that only shape diagnostics. stinkytofu-opt takes them; for a kernel
/// the pipeline chooses its own.
bool isToolOnlyKey(std::string_view key) {
    return key == "report" || key == "emitRegisterMap" || key == "emitSymbolBreadcrumbs" ||
           key == "dumpBlocks";
}

/// Set \p config's mode, and with it whether the pass writes the colouring.
void setModeOf(RegisterAllocationConfig& config, Mode mode) {
    config.mode = mode;
    config.options.applyToOperands = mode >= Mode::Apply;
}

}  // namespace

const char* unbankableOperandsName(Unbankable policy) {
    switch (policy) {
        case Unbankable::Hold:
            return "hold";
        case Unbankable::Allocate:
            return "allocate";
        case Unbankable::Auto:
            return "auto";
    }
    return "auto";
}

std::optional<RegClassSet> parseRegClasses(std::string_view spec) {
    RegClassSet classes;
    for (char c : spec) {
        switch (c) {
            case 'v':
                classes.add(RegType::V);
                break;
            case 's':
                classes.add(RegType::S);
                break;
            default:
                return std::nullopt;
        }
    }
    if (classes.empty()) return std::nullopt;
    return classes;
}

std::optional<AllocationScope::HeldRange> parseHeldRange(std::string_view spec) {
    if (spec.size() < 2) return std::nullopt;
    RegType type = RegType::UNKNOWN;
    if (spec[0] == 's')
        type = RegType::S;
    else if (spec[0] == 'v')
        type = RegType::V;
    else
        return std::nullopt;

    const auto digits = [](std::string_view text) -> std::optional<uint32_t> {
        if (text.empty()) return std::nullopt;
        uint32_t value = 0;
        for (char c : text) {
            if (c < '0' || c > '9') return std::nullopt;
            value = value * 10 + static_cast<uint32_t>(c - '0');
        }
        return value;
    };

    const std::string_view body = spec.substr(1);
    const size_t colon = body.find(':');
    if (colon == std::string_view::npos) {
        const std::optional<uint32_t> only = digits(body);
        if (!only.has_value()) return std::nullopt;
        return AllocationScope::HeldRange{type, *only, *only};
    }

    const std::optional<uint32_t> start = digits(body.substr(0, colon));
    const std::optional<uint32_t> end = digits(body.substr(colon + 1));
    if (!start.has_value() || !end.has_value() || *end < *start) return std::nullopt;
    return AllocationScope::HeldRange{type, *start, *end};
}

Expected<RegisterAllocationConfig> parseRegisterAllocationConfig(
    const std::vector<std::string>& tokens, RegisterAllocationDialect dialect) {
    using Result = Expected<RegisterAllocationConfig>;
    const bool kernel = dialect == RegisterAllocationDialect::Kernel;

    RegisterAllocationConfig config;
    RegisterAllocationOptions& options = config.options;
    if (kernel) {
        config.mode = Mode::Off;
        options.allocate = kKernelClasses;
        options.allocator = kKernelAllocator;
    } else {
        config.mode = Mode::Shadow;
        options.allocate = RegClassSet::only(RegType::V);
    }

    std::optional<std::string> modeWord;
    const auto setMode = [&](Mode mode, std::string_view word) -> std::optional<std::string> {
        if (modeWord.has_value() && modeNamed(*modeWord) != mode)
            return "two modes: '" + *modeWord + "' and '" + std::string(word) + "'";
        modeWord = std::string(word);
        config.mode = mode;
        return std::nullopt;
    };

    for (const std::string& raw : tokens) {
        const std::string_view token = trimmed(raw);
        if (token.empty()) continue;
        const size_t equals = token.find('=');
        const std::string_view key = trimmed(token.substr(0, equals));
        const auto unknown = [&] {
            return Result::Error("unknown register allocation setting '" + std::string(token) +
                                 "'");
        };
        if (kernel && isToolOnlyKey(key)) {
            return Result::Error("'" + std::string(key) +
                                 "' is a stinkytofu-opt setting; for a kernel the pipeline "
                                 "chooses its diagnostics (StinkyTofuEnableRemarks, "
                                 "StinkyTofuRegisterAllocationDump)");
        }

        if (equals == std::string_view::npos) {
            if (const std::optional<Mode> mode = modeNamed(key)) {
                if (std::optional<std::string> problem = setMode(*mode, key))
                    return Result::Error(*problem);
                continue;
            }
            if (key == "noVerify") {
                options.verify = false;
            } else if (key == "noRules") {
                options.rules.disableAll = true;
            } else if (key == "ruleAudit") {
                options.rules.auditAll = true;
            } else if (key == "report") {
                options.report = true;
            } else if (key == "emitRegisterMap") {
                options.emitRegisterMap = true;
            } else if (key == "emitSymbolBreadcrumbs") {
                options.emitSymbolBreadcrumbs = true;
            } else {
                return unknown();
            }
            continue;
        }

        const std::string_view value = trimmed(token.substr(equals + 1));
        const auto badValue = [&](const char* accepted) {
            return Result::Error("'" + std::string(key) + "' takes " + accepted + ", not '" +
                                 std::string(value) + "'");
        };
        if (key == "mode") {
            const std::optional<Mode> mode = modeNamed(value);
            if (!mode.has_value()) return badValue("off, shadow, apply, force or 0..3");
            if (std::optional<std::string> problem = setMode(*mode, value))
                return Result::Error(*problem);
        } else if (key == "classes") {
            const std::optional<RegClassSet> classes = parseRegClasses(value);
            if (!classes.has_value()) return badValue("s, v or vs");
            options.allocate = *classes;
        } else if (key == "allocator") {
            const std::vector<std::string> chain = splitList(value, '+');
            if (chain.empty()) return badValue("an allocator name, or several joined with '+'");
            if (std::optional<std::string> problem = unregisteredAllocator(chain))
                return Result::Error(*problem);
            options.allocator = joined(chain, "+");
        } else if (key == "unbankable") {
            std::optional<Unbankable> policy;
            for (const Unbankable candidate :
                 {Unbankable::Hold, Unbankable::Allocate, Unbankable::Auto}) {
                if (value == unbankableOperandsName(candidate)) policy = candidate;
            }
            if (!policy.has_value()) return badValue("hold, allocate or auto");
            options.unbankableOperands = *policy;
        } else if (key == "quadrant") {
            if (value == "any")
                options.quadrant = Quadrant::Any;
            else if (value == "prefer")
                options.quadrant = Quadrant::Prefer;
            else
                return badValue("any or prefer");
        } else if (key == "rules" || key == "noRule") {
            const std::vector<std::string> names = splitList(value, '+');
            if (names.empty()) return badValue("rule names joined with '+'");
            for (const std::string& name : names) {
                if (key == "noRule")
                    options.rules.disable.push_back(name);
                else if (name == "all")
                    options.rules.activateAll = true;
                else
                    options.rules.activate.push_back(name);
            }
        } else if (key == "pinReg") {
            const std::vector<std::string> ranges = splitList(value, '+');
            if (ranges.empty()) return badValue("registers such as s0 or s0:19, joined with '+'");
            for (const std::string& spec : ranges) {
                const std::optional<AllocationScope::HeldRange> range = parseHeldRange(spec);
                if (!range.has_value()) return badValue("registers such as s0 or s0:19");
                options.pinRegisters.push_back(*range);
            }
        } else if (key == "regionEnd") {
            if (value.empty()) return badValue("a block label");
            options.regionEnd = std::string(value);
        } else if (key == "dumpBlocks") {
            if (value.empty()) return badValue("a path prefix");
            options.blockDumpPrefix = std::string(value);
        } else {
            return unknown();
        }
    }

    if (!kernel && config.mode == Mode::Off)
        return Result::Error("mode off runs no register allocation; leave the pass out instead");
    setModeOf(config, config.mode);
    return config;
}

Expected<RegisterAllocationConfig> parseRegisterAllocationConfig(
    std::string_view text, RegisterAllocationDialect dialect) {
    return parseRegisterAllocationConfig(splitList(text, ','), dialect);
}

Expected<RegisterAllocationConfig> kernelRegisterAllocationConfig(int mode, std::string_view text) {
    Expected<RegisterAllocationConfig> config =
        parseRegisterAllocationConfig(text, RegisterAllocationDialect::Kernel);
    if (config.hasError()) return config;
    setModeOf(*config, static_cast<Mode>(std::clamp(mode, 0, 3)));
    return config;
}

std::string canonicalRegisterAllocationConfig(const RegisterAllocationConfig& config) {
    std::string mode = std::to_string(static_cast<int>(config.mode));
    if (config.mode == Mode::Off) return mode;

    const RegisterAllocationOptions& options = config.options;
    const RegisterAllocationOptions defaults;
    std::vector<std::string> words{mode};

    if (options.allocate != kKernelClasses)
        words.push_back("classes=" + classesName(options.allocate));
    if (options.allocator != kKernelAllocator) words.push_back("allocator=" + options.allocator);
    if (options.unbankableOperands != defaults.unbankableOperands)
        words.push_back(std::string("unbankable=") +
                        unbankableOperandsName(options.unbankableOperands));
    if (options.quadrant != defaults.quadrant) words.push_back("quadrant=prefer");

    std::vector<std::string> activate = sortedUnique(options.rules.activate);
    if (options.rules.activateAll) activate.insert(activate.begin(), "all");
    if (!activate.empty()) words.push_back("rules=" + joined(activate, "+"));
    if (const std::vector<std::string> disable = sortedUnique(options.rules.disable);
        !disable.empty())
        words.push_back("noRule=" + joined(disable, "+"));
    if (options.rules.disableAll) words.push_back("noRules");
    if (options.rules.auditAll) words.push_back("ruleAudit");

    if (!options.pinRegisters.empty()) {
        std::vector<AllocationScope::HeldRange> ranges = options.pinRegisters;
        const auto key = [](const AllocationScope::HeldRange& range) {
            return std::tuple(regTypeToString(range.regClass), range.start, range.end);
        };
        std::sort(ranges.begin(), ranges.end(),
                  [&](const auto& a, const auto& b) { return key(a) < key(b); });
        ranges.erase(std::unique(ranges.begin(), ranges.end(),
                                 [&](const auto& a, const auto& b) { return key(a) == key(b); }),
                     ranges.end());
        std::vector<std::string> names;
        names.reserve(ranges.size());
        for (const AllocationScope::HeldRange& range : ranges)
            names.push_back(heldRangeName(range));
        words.push_back("pinReg=" + joined(names, "+"));
    }
    if (!options.regionEnd.empty()) words.push_back("regionEnd=" + options.regionEnd);
    if (!options.verify) words.push_back("noVerify");
    return joined(words, ",");
}

bool kernelRewritesRegisters(int mode, std::string_view text, RegType regClass) {
    if (mode < static_cast<int>(Mode::Apply)) return false;
    const Expected<RegisterAllocationConfig> config = kernelRegisterAllocationConfig(mode, text);
    return !config.hasError() && config->options.allocate.contains(regClass);
}

}  // namespace stinkytofu
