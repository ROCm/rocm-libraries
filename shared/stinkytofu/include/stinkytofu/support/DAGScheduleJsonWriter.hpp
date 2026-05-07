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
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {

class StinkyInstruction;

/// One-line, JSON-safe label for a DAG node (opcode / mnemonic + trimmed operands).
STINKYTOFU_EXPORT std::string instructionJsonLabel(const StinkyInstruction& inst);

/// UTF-8 JSON for tools/stinkytofu-analysis: register-dependency edges plus before/after
/// instruction order per region (schema stinkytofu-dag-schedule-v1). Used by PassManager pass-order
/// snapshots.
class STINKYTOFU_EXPORT DAGScheduleJsonCollector {
   public:
    DAGScheduleJsonCollector(std::string outputPath, std::string functionName);
    ~DAGScheduleJsonCollector();

    DAGScheduleJsonCollector(const DAGScheduleJsonCollector&) = delete;
    DAGScheduleJsonCollector& operator=(const DAGScheduleJsonCollector&) = delete;

    void addRegion(const std::string& title,
                   const std::vector<std::pair<unsigned, std::string>>& nodeIdAndLabel,
                   const std::vector<std::pair<unsigned, unsigned>>& edges,
                   const std::vector<unsigned>& programOrderNodeIds,
                   const std::vector<unsigned>& scheduledOrderNodeIds);

    void finalize();

   private:
    std::string escapeJson(std::string_view s) const;

    std::string outputPath_;
    std::string functionName_;
    std::vector<std::string> regionsJson_;
    bool finalized_ = false;
};

}  // namespace stinkytofu
