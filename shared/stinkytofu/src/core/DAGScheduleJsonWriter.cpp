/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/support/DAGScheduleJsonWriter.hpp"

#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
std::string instructionJsonLabel(const StinkyInstruction& inst) {
    std::ostringstream oss;
    inst.dump(oss);
    std::string s = oss.str();
    std::string out;
    out.reserve(s.size());
    bool lastSpace = false;
    for (char c : s) {
        if (c == '\r' || c == '\n' || c == '\t') {
            if (!lastSpace && !out.empty()) {
                out += ' ';
                lastSpace = true;
            }
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!lastSpace && !out.empty()) {
                out += ' ';
                lastSpace = true;
            }
            continue;
        }
        out += c;
        lastSpace = false;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    constexpr size_t kMax = 512;
    if (out.size() > kMax) out.resize(kMax);
    return out;
}

DAGScheduleJsonCollector::DAGScheduleJsonCollector(std::string outputPath, std::string functionName)
    : outputPath_(std::move(outputPath)), functionName_(std::move(functionName)) {}

DAGScheduleJsonCollector::~DAGScheduleJsonCollector() {
    finalize();
}

std::string DAGScheduleJsonCollector::escapeJson(std::string_view s) const {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char uc : s) {
        char c = static_cast<char>(uc);
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (uc < 0x20u) {
                    std::ostringstream hex;
                    hex << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(uc);
                    out += hex.str();
                } else
                    out += c;
        }
    }
    return out;
}

static void appendIdArray(std::ostringstream& o, const std::vector<unsigned>& ids) {
    o << '[';
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) o << ',';
        o << '"' << ids[i] << '"';
    }
    o << ']';
}

void DAGScheduleJsonCollector::addRegion(
    const std::string& title, const std::vector<std::pair<unsigned, std::string>>& nodeIdAndLabel,
    const std::vector<std::pair<unsigned, unsigned>>& edges,
    const std::vector<unsigned>& programOrderNodeIds,
    const std::vector<unsigned>& scheduledOrderNodeIds) {
    std::ostringstream o;
    o << "{\"title\":\"" << escapeJson(title) << "\",\"nodes\":[";
    for (size_t i = 0; i < nodeIdAndLabel.size(); ++i) {
        if (i) o << ',';
        o << "{\"id\":\"" << nodeIdAndLabel[i].first << "\",\"label\":\""
          << escapeJson(nodeIdAndLabel[i].second) << "\"}";
    }
    o << "],\"before\":{\"edges\":[";
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i) o << ',';
        o << "{\"from\":\"" << edges[i].first << "\",\"to\":\"" << edges[i].second << "\"}";
    }
    o << "],\"order\":";
    appendIdArray(o, programOrderNodeIds);
    o << "},\"after\":{\"edges\":[";
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i) o << ',';
        o << "{\"from\":\"" << edges[i].first << "\",\"to\":\"" << edges[i].second << "\"}";
    }
    o << "],\"order\":";
    appendIdArray(o, scheduledOrderNodeIds);
    o << "}}";
    regionsJson_.push_back(o.str());
}

void DAGScheduleJsonCollector::finalize() {
    if (finalized_) return;
    finalized_ = true;
    std::ofstream out(outputPath_);
    if (!out) return;
    out << "{\"schema\":\"stinkytofu-dag-schedule-v1\",\"function\":\"" << escapeJson(functionName_)
        << "\",\"regions\":[";
    for (size_t i = 0; i < regionsJson_.size(); ++i) {
        if (i) out << ',';
        out << regionsJson_[i];
    }
    out << "]}\n";
}

}  // namespace stinkytofu
