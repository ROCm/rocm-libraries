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
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <unordered_set>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
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

bool resolveAsmSetRhs(const std::string& valstr,
                      const std::unordered_map<std::string, std::string>& raw,
                      std::unordered_set<std::string>& expanding, int depth, int64_t& out) {
    constexpr int kMaxDepth = 128;
    if (depth > kMaxDepth) return false;
    const std::string t = trimAsmToken(valstr);
    if (t.empty()) return false;
    if (parseAsmSetNumericToken(t, out)) return true;
    if (!isAsmSetIdentifier(t)) return false;
    if (expanding.contains(t)) return false;
    auto it = raw.find(t);
    if (it == raw.end()) return false;
    expanding.insert(t);
    const bool ok = resolveAsmSetRhs(it->second, raw, expanding, depth + 1, out);
    expanding.erase(t);
    return ok;
}

static int32_t int64ToInt32ForValuLiteral(int64_t v) {
    if (v >= INT32_MIN && v <= INT32_MAX) return static_cast<int32_t>(v);
    return static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(v)));
}
}  // namespace

void collectAsmSetSymbolValues(const Function& func,
                               std::unordered_map<std::string, int64_t>& out) {
    std::unordered_map<std::string, std::string> raw;
    for (const BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            const IRBase* node = it.getNodePtr();
            if (node->getType() != IRBase::IRType::StinkyAsmDirective) continue;
            const auto* directive = dyn_cast<AsmDirective>(node);
            if (directive == nullptr || directive->kind != AsmDirectiveKind::SET) continue;
            if (directive->symbol.empty()) continue;
            raw[directive->symbol] = trimAsmToken(directive->value);
        }
    }

    out.clear();
    for (const auto& kv : raw) {
        std::unordered_set<std::string> expanding;
        int64_t v = 0;
        if (resolveAsmSetRhs(kv.second, raw, expanding, 0, v)) out[kv.first] = v;
    }
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
