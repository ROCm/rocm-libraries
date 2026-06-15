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
#include "stinkytofu/bindings/python/Module.hpp"

#include <sstream>
#include <unordered_map>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/pipeline/Backend.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"

namespace stinkytofu {
namespace {
/**
 * @brief Range of instructions that belongs to a group.
 * @note This is a helper struct to manage instruction groups.
 */
struct InstructionGroupRange {
    IntrusiveListIterator<IRBase> first; /**< The first instruction in the group */
    IntrusiveListIterator<IRBase> last;  /**< The last instruction in the group */

    InstructionGroupRange()
        : first(IntrusiveListIterator<IRBase>()), last(IntrusiveListIterator<IRBase>()) {}

    IntrusiveListIterator<IRBase> begin() const {
        return first;
    }

    IntrusiveListIterator<IRBase> end() const {
        return ++IntrusiveListIterator<IRBase>(last.getNodePtr());
    }
};

}  // anonymous namespace

struct StinkyAsmModule::Impl {
    std::string name;
    std::string outputName;  // If non-empty, used for cost file basename (full kernel name)
    std::string outputDir;   // If non-empty, cost file goes to outputDir/<kernel_name>/
    std::array<int, 3> arch;

    // This map maintains the defined group names and the range of instructions for each group.
    std::unordered_map<std::string, InstructionGroupRange> instructionGroups;

    Function function;

    // Total instruction encoding size in bytes (for .amdhsa_inst_pref_size). -1 if not set.
    int64_t totalInstructionBytes = -1;

    // Plugin data: opaque key-value stores for pass plugins.
    std::unordered_map<std::string, int64_t> pluginDataI64;
    std::unordered_map<std::string, std::string> pluginDataStr;

    // Per-module pass builder for plugin pass registration.
    PassBuilder passBuilder;

    Impl(const std::string& name, const std::array<int, 3>& arch) : name(name), arch(arch) {
        // Create a single BasicBlock to hold all instructions
        function.createBasicBlock("entry");
    }

    ~Impl() {
        // Function destructor will clean up BasicBlocks and their IRLists
        // The IRBases in the IRList will be deleted by the Function
    }
};

StinkyAsmModule::StinkyAsmModule(const std::string& name, const std::array<int, 3>& arch,
                                 const ModuleOptions& moduleOptions)
    : pImpl(std::make_unique<Impl>(name, arch)), moduleOptions(moduleOptions) {
    // SwPrefetchScratchSgpr == -1: do not run SwPrefetchInsertionPass (see Gfx1250Backend).
    this->moduleOptions.EnableSwPrefetchInsertion =
        (this->moduleOptions.SwPrefetchScratchSgpr != -1);
}

StinkyAsmModule::~StinkyAsmModule() = default;

StinkyAsmModule::StinkyAsmModule(StinkyAsmModule&&) noexcept = default;
StinkyAsmModule& StinkyAsmModule::operator=(StinkyAsmModule&&) noexcept = default;

std::string StinkyAsmModule::getName() const {
    return pImpl->name;
}

void StinkyAsmModule::setOutputName(const std::string& name) {
    pImpl->outputName = name;
}

std::string StinkyAsmModule::getOutputName() const {
    return pImpl->outputName;
}

void StinkyAsmModule::setOutputDir(const std::string& dir) {
    pImpl->outputDir = dir;
}

std::string StinkyAsmModule::getOutputDir() const {
    return pImpl->outputDir;
}

std::array<int, 3> StinkyAsmModule::getArch() const {
    return pImpl->arch;
}

Function& StinkyAsmModule::getFunction() {
    return pImpl->function;
}

const Function& StinkyAsmModule::getFunction() const {
    return pImpl->function;
}

std::string StinkyAsmModule::emitAssembly() const {
    // Configure the emitter with default options
    stinkytofu::AsmEmitterOptions options;
    options.emitComments = true;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;
    options.useSymbolicNames = true;  // Enable symbolic register names

    stinkytofu::StinkyAsmEmitter emitter(options);
    return emitter.emit(getFunction());
}

void StinkyAsmModule::runOptimizationPipeline() {
    Backend backend(*this);
    backend.runOptimization();
}

std::optional<uint64_t> StinkyAsmModule::getMetaDataU64(const std::string& key) const {
    return getFunction().getMetaData(key);
}

void StinkyAsmModule::addGroup(const std::string& name) {
    if (pImpl->instructionGroups.find(name) != pImpl->instructionGroups.end()) {
        return;
    }
    pImpl->instructionGroups.emplace(name, InstructionGroupRange());
}

bool StinkyAsmModule::hasGroup(const std::string& name) const {
    return pImpl->instructionGroups.find(name) != pImpl->instructionGroups.end();
}

std::optional<std::pair<IntrusiveListIterator<IRBase>, IntrusiveListIterator<IRBase>>>
StinkyAsmModule::findGroupRange(const std::string& groupName) const {
    if (!hasGroup(groupName)) {
        return std::nullopt;
    }

    const auto& range = pImpl->instructionGroups.at(groupName);
    // Return nullopt if the group was registered but never populated
    if (range.first == IntrusiveListIterator<IRBase>()) {
        return std::nullopt;
    }

    return std::make_optional(std::make_pair(range.begin(), range.end()));
}

void StinkyAsmModule::updateInstructionGroups(const std::vector<const std::string*>& groups,
                                              size_t instsCountBefore) {
    for (const auto& groupName : groups) {
        if (hasGroup(*groupName)) {
            BasicBlock& bb = *getFunction().getEntryBlock();
            auto newInstructionCount = bb.size() - instsCountBefore;
            auto& groupRange = pImpl->instructionGroups.at(*groupName);
            if (groupRange.first == IntrusiveListIterator<IRBase>()) {
                auto it = bb.rbegin();
                for (size_t i = 1; i < newInstructionCount; ++i) {
                    it++;
                }
                groupRange.first = IntrusiveListIterator<IRBase>(it.getNodePtr());
            }

            groupRange.last = IntrusiveListIterator<IRBase>(bb.rbegin().getNodePtr());
        }
    }
}

void StinkyAsmModule::setGroupRange(const std::string& groupName,
                                    IntrusiveListIterator<IRBase> first,
                                    IntrusiveListIterator<IRBase> last) {
    if (!hasGroup(groupName)) return;
    auto& groupRange = pImpl->instructionGroups.at(groupName);
    groupRange.first = first;
    groupRange.last = last;
}

const StinkyAsmModule::ModuleOptions& StinkyAsmModule::getModuleOptions() const {
    return this->moduleOptions;
}

void StinkyAsmModule::setModuleOptions(const ModuleOptions& moduleOptions) {
    this->moduleOptions = moduleOptions;
}

void StinkyAsmModule::setTotalInstructionBytes(int64_t totalBytes) {
    pImpl->totalInstructionBytes = totalBytes;
}

int64_t StinkyAsmModule::getTotalInstructionBytes() const {
    return pImpl->totalInstructionBytes;
}

void StinkyAsmModule::setPluginDataI64(const std::string& key, int64_t value) {
    pImpl->pluginDataI64[key] = value;
}

int64_t StinkyAsmModule::getPluginDataI64(const std::string& key, int64_t defaultVal) const {
    auto it = pImpl->pluginDataI64.find(key);
    return it != pImpl->pluginDataI64.end() ? it->second : defaultVal;
}

void StinkyAsmModule::setPluginDataStr(const std::string& key, const std::string& value) {
    pImpl->pluginDataStr[key] = value;
}

std::string StinkyAsmModule::getPluginDataStr(const std::string& key,
                                              const std::string& defaultVal) const {
    auto it = pImpl->pluginDataStr.find(key);
    return it != pImpl->pluginDataStr.end() ? it->second : defaultVal;
}

PassBuilder& StinkyAsmModule::getPassBuilder() {
    return pImpl->passBuilder;
}

const PassBuilder& StinkyAsmModule::getPassBuilder() const {
    return pImpl->passBuilder;
}

}  // namespace stinkytofu
