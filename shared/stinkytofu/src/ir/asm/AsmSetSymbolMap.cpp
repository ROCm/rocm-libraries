/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated software files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"

#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <ostream>
#include <unordered_set>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/SymbolicRegName.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {
std::string trimAsmToken(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

bool isAsmSetIdentifier(const std::string& s) {
    if (s.empty()) return false;
    const unsigned char c0 = static_cast<unsigned char>(s[0]);
    if (!std::isalpha(c0) && c0 != '_' && c0 != '.') return false;
    for (size_t i = 1; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (!std::isalnum(c) && c != '_' && c != '.') return false;
    }
    return true;
}

/// Parse a `.set` RHS token as hex or decimal integer. Unsigned hex uses full
/// 32-bit range.
bool parseAsmSetNumericToken(const std::string& s, int64_t& out) {
    if (s.empty()) return false;
    bool negative = (s[0] == '-');
    size_t start = negative ? 1u : 0u;
    if (s.size() >= start + 2 && s[start] == '0' && (s[start + 1] == 'x' || s[start + 1] == 'X')) {
        if (start + 2 > s.size()) return false;
        char* end = nullptr;
        // Use unsigned 64-bit parse so 0xffffffff fits before narrowing semantics.
        unsigned long long v = std::strtoull(s.c_str() + start, &end, 16);
        if (end != s.c_str() + s.size() || end == s.c_str() + start) return false;
        if (negative) {
            if (v > 0x8000000000000000ull) return false;
            out = -static_cast<int64_t>(v);
        } else
            out = static_cast<int64_t>(v);
        return true;
    }
    for (size_t i = start; i < s.size(); ++i)
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    char* end = nullptr;
    long long val = std::strtoll(s.c_str(), &end, 10);
    if (end != s.c_str() + s.size() || end == s.c_str()) return false;
    out = static_cast<int64_t>(val);
    return true;
}

/// Resolve a `.set` right-hand side to an integer. Three forms are accepted:
///   * a bare decimal/hex literal (optionally signed),
///   * a single identifier referring to another `.set` symbol (resolved
///     recursively, with cycle detection), and
///   * a sum/difference of the above, e.g. `vgprBase+34` or `A-B+2`.
/// rocisa defines VGPR bases two levels deep with offset arithmetic
/// (`.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0`, `.set ..._BASE, vgprBase+34`),
/// so symbol+offset support is required to resolve VGPR names; SGPR `.set` are
/// plain literals and keep resolving via the first fast path. Returns false if
/// any term is unresolvable, which (as before) leaves the symbol unresolved.
bool resolveAsmSetRhs(const std::string& valstr,
                      const std::unordered_map<std::string, std::string>& raw,
                      std::unordered_set<std::string>& expanding, int depth, int64_t& out) {
    constexpr int kMaxDepth = 128;
    if (depth > kMaxDepth) return false;
    const std::string t = trimAsmToken(valstr);
    if (t.empty()) return false;

    // Fast path: a bare literal (handles its own leading sign and hex).
    if (parseAsmSetNumericToken(t, out)) return true;

    // Fast path: a single symbol reference, resolved recursively.
    if (isAsmSetIdentifier(t)) {
        if (expanding.contains(t)) return false;
        auto it = raw.find(t);
        if (it == raw.end()) return false;
        expanding.insert(t);
        const bool ok = resolveAsmSetRhs(it->second, raw, expanding, depth + 1, out);
        expanding.erase(t);
        return ok;
    }

    // General case: a sum/difference of terms, each a literal or a symbol.
    // Identifiers never contain '+'/'-' and a literal carries no interior sign,
    // so splitting on top-level '+'/'-' is unambiguous; each term's own leading
    // sign (if any) is consumed here and reapplied, then the unsigned remainder
    // is resolved recursively.
    int64_t total = 0;
    size_t i = 0;
    const size_t n = t.size();
    while (i < n) {
        int sign = 1;
        if (t[i] == '+') {
            ++i;
        } else if (t[i] == '-') {
            sign = -1;
            ++i;
        }
        const size_t start = i;
        while (i < n && t[i] != '+' && t[i] != '-') ++i;
        const std::string term = trimAsmToken(t.substr(start, i - start));
        if (term.empty()) return false;  // dangling or doubled operator
        int64_t termVal = 0;
        if (!resolveAsmSetRhs(term, raw, expanding, depth + 1, termVal)) return false;
        total += sign * termVal;
    }
    out = total;
    return true;
}

static int32_t int64ToInt32ForValuLiteral(int64_t v) {
    if (v >= INT32_MIN && v <= INT32_MAX) return static_cast<int32_t>(v);
    return static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(v)));
}
}  // namespace

void collectAsmSetSymbolValues(const Function& func,
                               std::unordered_map<std::string, int64_t>& out) {
    std::unordered_map<std::string, AsmSetSymbolInfo> info;
    collectAsmSetSymbolInfo(func, info);
    out.clear();
    for (const auto& kv : info) out[kv.first] = kv.second.value;
}

void collectAsmSetSymbolInfo(const Function& func,
                             std::unordered_map<std::string, AsmSetSymbolInfo>& out) {
    std::unordered_map<std::string, std::string> raw;
    std::unordered_map<std::string, unsigned> counts;
    for (const BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            const IRBase* node = it.getNodePtr();
            if (node->getType() != IRBase::IRType::StinkyAsmDirective) continue;
            const auto* directive = dyn_cast<AsmDirective>(node);
            if (directive == nullptr || directive->kind != AsmDirectiveKind::SET) continue;
            if (directive->symbol.empty()) continue;
            raw[directive->symbol] = trimAsmToken(directive->value);
            ++counts[directive->symbol];
        }
    }

    out.clear();
    for (const auto& kv : raw) {
        std::unordered_set<std::string> expanding;
        int64_t v = 0;
        AsmSetSymbolInfo info;
        info.definitionCount = counts[kv.first];
        if (resolveAsmSetRhs(kv.second, raw, expanding, 0, v)) {
            info.value = v;
            info.resolved = true;
        }
        out[kv.first] = info;
    }
}

/// Walks the function in emission order, tracking `.set` bindings as they come
/// into force, and rewrites each symbolically-named operand's index to match.
size_t resolveSymbolicOperands(Function& func, std::vector<SymbolicOperandFix>& fixes,
                               bool resolveUnresolvedSentinel) {
    // The all-ones placeholder a rocisa regIdx == -1 widens to (static_cast<uint32_t>(-1)).
    // Distinct from a real virtual register, which carries kVirtualBit over a small index.
    static constexpr uint32_t kUnresolvedSentinelIdx = 0xFFFFFFFFu;
    std::unordered_map<std::string, std::string> raw;
    std::unordered_map<std::string, int64_t> live;
    bool liveIsStale = true;

    auto refreshLive = [&]() {
        if (!liveIsStale) return;
        live.clear();
        for (const auto& kv : raw) {
            std::unordered_set<std::string> expanding;
            int64_t value = 0;
            if (resolveAsmSetRhs(kv.second, raw, expanding, 0, value)) live[kv.first] = value;
        }
        liveIsStale = false;
    };

    size_t corrected = 0;
    for (BasicBlock& bb : func) {
        for (IRBase& ir : bb) {
            if (auto* directive = dyn_cast<AsmDirective>(&ir)) {
                if (directive->kind != AsmDirectiveKind::SET || directive->symbol.empty()) continue;
                raw[directive->symbol] = trimAsmToken(directive->value);
                liveIsStale = true;
                continue;
            }
            auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;

            auto materialise = [&](bool isDest, size_t operand) {
                const StinkyRegister& reg = isDest ? instruction->getDestRegs()[operand]
                                                   : instruction->getSrcRegs()[operand];
                if (!reg.isRegister() || !reg.hasSymbolicName()) return;
                if (resolveUnresolvedSentinel) {
                    // Migration mode: resolve only the all-ones placeholder left by an
                    // eagerly-converted logical leaf (rocisa regIdx == -1). Native operands
                    // already carry a resolved index and real virtual registers carry a
                    // small low-bit index, so neither matches and both are left untouched.
                    if (reg.reg.idx != kUnresolvedSentinelIdx) return;
                } else {
                    // Virtual registers are not yet indices, so a name cannot contradict them.
                    if (reg.isVirtualReg()) return;
                }
                refreshLive();
                // Read before the write below, which lands in `reg`.
                const std::string symbol = reg.getSymbolicName();
                const uint32_t oldIdx = reg.reg.idx;
                const std::optional<int64_t> named = resolveNamedIndex(symbol, live, reg.reg.num);
                // An unresolvable name says nothing about the index; leave it be.
                if (!named.has_value() || *named < 0) return;
                if (*named == static_cast<int64_t>(oldIdx)) return;

                StinkyRegister updated = reg;
                updated.reg.idx = static_cast<uint32_t>(*named);
                if (isDest)
                    instruction->setDestReg(operand, updated);
                else
                    instruction->setSrcReg(operand, updated);
                fixes.push_back({symbol, oldIdx, static_cast<uint32_t>(*named)});
                ++corrected;
            };

            for (size_t operand = 0; operand < instruction->getNumDestRegs(); ++operand)
                materialise(true, operand);
            for (size_t operand = 0; operand < instruction->getNumSrcRegs(); ++operand)
                materialise(false, operand);
        }
    }
    return corrected;
}

bool tryResolveAsmSetSymbolToInt32(const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                                   const std::string& name, int32_t& outInt32) {
    if (asmSetSymbols == nullptr) return false;
    auto it = asmSetSymbols->find(name);
    if (it == asmSetSymbols->end()) return false;
    outInt32 = int64ToInt32ForValuLiteral(it->second);
    return true;
}

void dumpAsmSetSymbolMap(std::ostream& os, const std::unordered_map<std::string, int64_t>& map) {
    os << "[AsmSetSymbolMap] .set symbol -> int64 (" << map.size() << " entries)\n";
    if (map.empty()) {
        os << "  (none)\n";
        return;
    }
    for (const auto& kv : map) {
        const int32_t i32 = int64ToInt32ForValuLiteral(kv.second);
        const uint32_t u32 = static_cast<uint32_t>(i32);
        os << "  \"" << kv.first << "\" -> " << kv.second << " | u32=0x" << std::hex << u32
           << std::dec << "\n";
    }
}
}  // namespace stinkytofu
