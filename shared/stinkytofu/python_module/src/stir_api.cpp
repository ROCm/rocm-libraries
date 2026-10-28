/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "stir_api.hpp"

#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "stinkytofu/Version.h"
#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/analysis/asm/HazardGapAnalysisPass.hpp"
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ComgrProbe.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/pipeline/PassCatalog.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/RawAsmParser.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"
#include "stinkytofu/support/Diagnostic.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/support/PassArguments.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"
#include "stinkytofu/transforms/asm/DefUseAnalysisCleanup.hpp"

namespace nb = nanobind;
using namespace stinkytofu;

namespace {

/// A query copies names, opcodes, and operands. The objects stay readable
/// after a later pass frees or rebuilds the function.
struct PyInst {
    std::string opcode;
    nb::list defs;
    nb::list uses;
    std::vector<std::string> modifiers;
};

struct PyBlock {
    std::string name;
    std::vector<std::string> successors;
    nb::list instructions;
};

struct PyFunction {
    std::string name;
    bool hasAttachedSSA = false;
    nb::list blocks;
};

struct PyRemark {
    std::string kind;
    std::string remarkName;
    std::string message;
};

/// What one pass did to one function.
struct PyPassRecord {
    std::string passName;
    std::string function;
    bool changed = false;
    nb::list remarks;
    std::optional<std::string> before;
    std::optional<std::string> after;
};

struct PyLiveRange {
    SSAValueID value = kInvalidSSAValueID;
    std::string regClass;
    int width = 1;
    nb::object reg;
    nb::list segments;
    SlotIndex start = 0;
    SlotIndex end = 0;
    SlotIndex length = 0;
    std::string line;
};

/// One function's live ranges, and the place of every slot they can name.
struct PyLiveRanges {
    nb::list ranges;
    std::vector<nb::object> places;
};

struct PyPeak {
    std::string regClass;
    uint32_t dwords = 0;
    SlotIndex slot = 0;
    nb::object place;
};

struct PyPressure {
    nb::dict peaks;
    uint32_t vgpr = 0;
    uint32_t sgpr = 0;
};

struct PyHazardGap {
    std::string rule;
    int required = 0;
    int gap = 0;
    bool ok = false;
    std::string block;
    int producer = -1;
    int consumer = -1;
    std::string producerOpcode;
    std::string consumerOpcode;
};

struct PyFunctionSize {
    int64_t start = 0;
    int64_t bytes = 0;
    int64_t cycles = 0;
    int instructions = 0;
    nb::dict labels;
};

struct PyRunInfo {
    std::array<int, 3> arch{};
    std::string version;
    std::vector<std::string> passes;
    std::optional<std::string> function;
    nb::dict features;
    nb::dict tile;
    std::string command;
    std::string line;
};

std::string trimCopy(std::string text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

bool isSsaForm(std::string_view text) {
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '%') continue;
        if (i + 1 < text.size() && text[i + 1] == '<') return true;
        size_t j = i + 1;
        if (j >= text.size() || !std::isdigit(static_cast<unsigned char>(text[j]))) continue;
        while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
        if (j < text.size() && text[j] == ':') return true;
    }
    return false;
}

std::string errorText(const std::vector<Diagnostic>& diagnostics) {
    std::ostringstream out;
    for (const Diagnostic& diagnostic : diagnostics) {
        if (diagnostic.getLevel() != Diagnostic::Level::Error) continue;
        if (out.tellp() > 0) out << '\n';
        out << diagnostic.format();
    }
    return out.str();
}

std::string catalogNames() {
    std::string names;
    for (const PassInfo& info : passCatalog()) {
        if (!names.empty()) names += ", ";
        names += info.name;
    }
    return names;
}

nb::object physicalOperand(const StinkyRegister& reg) {
    switch (reg.dataType) {
        case StinkyRegister::Type::LiteralInt:
            return nb::cast(reg.literalInt);
        case StinkyRegister::Type::LiteralDouble:
            return nb::cast(reg.literalDouble);
        case StinkyRegister::Type::LiteralString:
            return nb::cast(reg.literalValue);
        case StinkyRegister::Type::Register: {
            const uint32_t index = reg.reg.idx & ~StinkyRegister::kVirtualBit;
            return nb::make_tuple(regTypeToString(reg.reg.type), static_cast<int>(index),
                                  static_cast<int>(reg.reg.num));
        }
        case StinkyRegister::Type::HwReg:
            return nb::make_tuple("hwreg", static_cast<int>(reg.hwreg.id),
                                  static_cast<int>(reg.hwreg.offset),
                                  static_cast<int>(reg.hwreg.size));
        case StinkyRegister::Type::Invalid:
            return nb::none();
    }
    return nb::none();
}

void appendOperand(nb::list& out, nb::object operand) {
    if (!operand.is_none()) out.append(std::move(operand));
}

nb::object ssaValue(const StinkySSAValue* value) {
    if (value == nullptr) return nb::none();
    return nb::make_tuple(static_cast<int>(value->valueId()),
                          regTypeToString(value->type().regType));
}

nb::object ssaLiteral(const StinkyOpOperand& operand) {
    const LegacyImmPayload& imm = operand.imm();
    switch (operand.kind()) {
        case StinkyOpOperand::Kind::LiteralInt:
            if (const auto* value = std::get_if<int32_t>(&imm)) return nb::cast(*value);
            break;
        case StinkyOpOperand::Kind::LiteralDouble:
            if (const auto* value = std::get_if<double>(&imm)) return nb::cast(*value);
            break;
        case StinkyOpOperand::Kind::LiteralString:
            if (const auto* value = std::get_if<std::string>(&imm)) return nb::cast(*value);
            break;
        case StinkyOpOperand::Kind::HwReg:
            if (const auto* hw = std::get_if<HwRegPayload>(&imm)) {
                return nb::make_tuple("hwreg", static_cast<int>(hw->id),
                                      static_cast<int>(hw->offset), static_cast<int>(hw->size));
            }
            break;
        case StinkyOpOperand::Kind::Value:
        case StinkyOpOperand::Kind::Invalid:
            break;
    }
    return nb::none();
}

/// Whether an instruction's defs or uses list \p reg; set_operand counts
/// positions the same way.
bool isListedOperand(const StinkyRegister& reg) {
    return reg.dataType != StinkyRegister::Type::Invalid;
}

nb::list physicalList(const std::vector<StinkyRegister>& regs) {
    nb::list out;
    for (const StinkyRegister& reg : regs) {
        if (isListedOperand(reg)) out.append(physicalOperand(reg));
    }
    return out;
}

nb::list ssaDefs(const StinkyInstruction& inst) {
    nb::list out;
    for (size_t i = 0; i < inst.getNumSSAResults(); ++i)
        appendOperand(out, ssaValue(inst.getSSAResult(i)));
    return out;
}

nb::list ssaUses(const StinkyInstruction& inst) {
    nb::list out;
    for (size_t i = 0; i < inst.getNumSSAOperands(); ++i) {
        const StinkyOpOperand* operand = inst.getSSAOperand(i);
        if (operand == nullptr) continue;
        if (operand->kind() == StinkyOpOperand::Kind::Value)
            appendOperand(out, ssaValue(operand->value()));
        else
            appendOperand(out, ssaLiteral(*operand));
    }
    return out;
}

std::string opcodeOf(const StinkyInstruction& inst) {
    const HwInstDesc* desc = inst.getHwInstDesc();
    const char* mnemonic = desc != nullptr ? desc->mnemonic : nullptr;
    return std::string("st.") + (mnemonic != nullptr ? mnemonic : "unknown");
}

GfxArchID checkedArchId(const std::array<int, 3>& arch) {
    if (!BackendRegistry::isArchSupported(arch)) {
        throw std::invalid_argument("unknown arch (" + std::to_string(arch[0]) + ", " +
                                    std::to_string(arch[1]) + ", " + std::to_string(arch[2]) + ")");
    }
    return getGfxArchID(arch);
}

void raiseErrors(const std::vector<Diagnostic>& diagnostics) {
    const std::string errors = errorText(diagnostics);
    if (!errors.empty()) throw std::invalid_argument(errors);
}

/// Build one function per parsed function, entry first. Every function is
/// converted before raising, so one error lists every loss. The module keeps the
/// warnings.
void populateModule(StinkyAsmModule& module,
                    std::vector<std::unique_ptr<ParsedFunction>>& parsedFunctions, GfxArchID archId,
                    std::vector<Diagnostic> diagnostics) {
    bool first = true;
    for (auto& parsedFunc : parsedFunctions) {
        if (parsedFunc == nullptr) continue;
        Function* function = nullptr;
        if (first) {
            function = &module.getFunction();
            first = false;
        } else {
            if (parsedFunc->funcName.empty())
                throw std::invalid_argument("only the first function may be unnamed");
            if (parsedFunc->funcName == module.getFunction().getName() ||
                module.getFunction(parsedFunc->funcName) != nullptr) {
                throw std::invalid_argument("duplicate function '" + parsedFunc->funcName + "'");
            }
            function = &module.createFunction(parsedFunc->funcName);
        }
        const StinkyErrorCode result = StinkyIRConverter::populateFunctionFromParsed(
            *parsedFunc, *function, archId, diagnostics);
        if (result == StinkyErrorCode::PARSE_ERROR)
            throw std::invalid_argument("function '" + parsedFunc->funcName + "' has no blocks");
    }
    raiseErrors(diagnostics);
    std::vector<Diagnostic> warnings;
    for (Diagnostic& diagnostic : diagnostics) {
        if (diagnostic.getLevel() != Diagnostic::Level::Error)
            warnings.push_back(std::move(diagnostic));
    }
    module.setLoadWarnings(std::move(warnings));
}

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("cannot read " + path);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::unique_ptr<StinkyAsmModule> loadStir(const std::string& text, std::array<int, 3> arch) {
    if (isSsaForm(text))
        throw std::invalid_argument(
            "SSA form does not round-trip; load physical STIR and run LiftAsmRegistersToSSAPass");
    const GfxArchID archId = checkedArchId(arch);

    MultiParseResult parsed = parseAllSourceStringsWithDiagnostics(text);
    raiseErrors(parsed.diagnostics);
    if (parsed.functions.empty()) throw std::invalid_argument("no st.func in STIR text");

    const std::string& name =
        parsed.moduleName.empty() ? parsed.functions.front()->funcName : parsed.moduleName;
    auto module = std::make_unique<StinkyAsmModule>(name, arch, StinkyAsmModule::ModuleOptions{});
    populateModule(*module, parsed.functions, archId, std::move(parsed.diagnostics));
    return module;
}

std::unique_ptr<StinkyAsmModule> loadStirFile(const std::string& path, std::array<int, 3> arch) {
    return loadStir(readFile(path), arch);
}

std::unique_ptr<StinkyAsmModule> loadAsm(const std::string& text, std::array<int, 3> arch,
                                         bool preserveSymbolicRegs, bool preserveComments,
                                         const std::optional<std::string>& fromLabel,
                                         const std::optional<std::string>& toLabel) {
    if (fromLabel.has_value() != toLabel.has_value())
        throw std::invalid_argument("from_label and to_label must be given together");
    const GfxArchID archId = checkedArchId(arch);

    RawAsmParserOptions options;
    options.preserveSymbolicNames = preserveSymbolicRegs;
    options.preserveComments = preserveComments;
    RawAsmParseResult parsed = parseRawAsmString(text, archId, options);
    raiseErrors(parsed.diagnostics);

    std::vector<std::unique_ptr<ParsedFunction>> functions;
    functions.push_back(std::move(parsed.parsedFunction));
    if (fromLabel) {
        std::string error;
        std::optional<ParsedRegion> cut =
            cutParsedRegion(*functions.front(), *fromLabel, *toLabel, error);
        if (!cut) throw std::invalid_argument(error);
        functions.front() = std::move(cut->region);
    }

    auto module = std::make_unique<StinkyAsmModule>(functions.front()->funcName, arch,
                                                    StinkyAsmModule::ModuleOptions{});
    populateModule(*module, functions, archId, std::move(parsed.diagnostics));
    if (parsed.signature) {
        module->getFunction().setMetaData(
            kSigTotalVgprsMetaKey,
            static_cast<uint64_t>(parsed.signature->kernelDescriptor.totalVgprs));
        // A region is not a whole kernel, so it has no header to write back.
        if (!fromLabel) module->setSignature(std::move(parsed.signature));
    }
    return module;
}

std::unique_ptr<StinkyAsmModule> loadAsmFile(const std::string& path, std::array<int, 3> arch,
                                             bool preserveSymbolicRegs, bool preserveComments,
                                             const std::optional<std::string>& fromLabel,
                                             const std::optional<std::string>& toLabel) {
    return loadAsm(readFile(path), arch, preserveSymbolicRegs, preserveComments, fromLabel,
                   toLabel);
}

/// A copy for the analysis API. Output settings, plugin data, and instruction
/// groups are not copied; no call in this API sets them.
std::unique_ptr<StinkyAsmModule> copyModule(const StinkyAsmModule& module) {
    auto copy = std::make_unique<StinkyAsmModule>(module.getName(), module.getArch(),
                                                  module.getModuleOptions());
    bool entry = true;
    for (const Function* function : module.getFunctions()) {
        if (function == nullptr) continue;
        Function& target = entry ? copy->getFunction() : copy->createFunction(function->getName());
        entry = false;
        if (!target.copyFrom(*function))
            throw std::invalid_argument("function '" + function->getName() +
                                        "' holds IR that cannot be copied");
    }
    copy->setSignature(module.getSignature());
    copy->setLoadWarnings(module.getLoadWarnings());
    for (const PassRunRecord& run : module.getPassRuns()) copy->addPassRun(run);
    return copy;
}

/// \p word as one shell word: bare when it is plain, single-quoted otherwise.
std::string shellWord(const std::string& word) {
    const bool plain = !word.empty() && std::all_of(word.begin(), word.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) ||
               std::strchr("_@%+=:,./-", c) != nullptr;
    });
    if (plain) return word;
    std::string quoted = "'";
    for (char c : word) quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return quoted + "'";
}

/// The stinkytofu command line, after `stinkytofu`, that repeats \p run.
std::string runCommand(const std::array<int, 3>& arch, const PassRunRecord& run) {
    std::string text = "run --arch " + std::to_string(arch[0]) + "," + std::to_string(arch[1]) +
                       "," + std::to_string(arch[2]);
    for (const std::string& spec : run.passes) text += " --pass " + shellWord(spec);
    if (run.function) text += " --function " + shellWord(*run.function);
    for (const auto& [name, value] : run.features)
        text += " --feature " + name + (value ? "=true" : "=false");
    for (const auto& [name, value] : run.tile)
        text += " --tile " + name + "=" + std::to_string(value);
    return text;
}

/// `stinkytofu <version>: run ...`, what run_info() and the printed comment say.
std::string runLine(const std::array<int, 3>& arch, const PassRunRecord& run) {
    return "stinkytofu " + std::string(getRuntimeVersion()) + ": " + runCommand(arch, run);
}

/// One `// stinkytofu <version>: run ...` line per run, which STIR and
/// assembler source both read as a comment.
std::string runCommentLines(const StinkyAsmModule& module) {
    std::string lines;
    for (const PassRunRecord& run : module.getPassRuns())
        lines += "// " + runLine(module.getArch(), run) + "\n";
    return lines;
}

nb::list runInfo(const StinkyAsmModule& module) {
    nb::list out;
    for (const PassRunRecord& run : module.getPassRuns()) {
        PyRunInfo info;
        info.arch = module.getArch();
        info.version = getRuntimeVersion();
        info.passes = run.passes;
        info.function = run.function;
        for (const auto& [name, value] : run.features)
            info.features[name.c_str()] = nb::cast(value);
        for (const auto& [name, value] : run.tile) info.tile[name.c_str()] = nb::cast(value);
        info.command = "stinkytofu " + runCommand(module.getArch(), run);
        info.line = runLine(module.getArch(), run);
        out.append(std::move(info));
    }
    return out;
}

std::string serializeModule(const StinkyAsmModule& module, bool ssa, bool comments) {
    std::ostringstream out;
    if (comments) out << runCommentLines(module);
    AsmPrinterOptions options;
    options.ssaForm = ssa;
    AsmPrinter printer(out, options);
    printer.print(module);
    return out.str();
}

std::string emitModuleAsm(const StinkyAsmModule& module, bool comments, bool cycleInfo,
                          bool symbolicNames, bool signature) {
    std::ostringstream out;
    if (comments) out << runCommentLines(module);
    if (signature && module.getSignature()) out << module.getSignature()->toString();
    AsmEmitterOptions options;
    options.emitComments = comments;
    options.emitCycleInfo = cycleInfo;
    options.useSymbolicNames = symbolicNames;
    options.indent = 0;
    StinkyAsmEmitter emitter(options);
    for (const Function* function : module.getFunctions()) emitter.emit(out, *function);
    return out.str();
}

std::optional<std::string> assembleModule(const StinkyAsmModule& module) {
    if (!hasComgrSupport())
        throw std::runtime_error("assemble() is unsupported: this build has no amd_comgr");
    const std::array<int, 3> arch = module.getArch();
    const ArchHelper::ArchInfo* info = ArchHelper::getInstance().getArchInfo(
        static_cast<uint32_t>(arch[0]), static_cast<uint32_t>(arch[1]),
        static_cast<uint32_t>(arch[2]));
    if (info == nullptr) throw std::invalid_argument("unknown arch for this module");
    const std::string isaName = "amdgcn-amd-amdhsa--" + BackendRegistry::makeArchKey(arch);

    std::string log;
    const std::string text = emitModuleAsm(module, false, false, false, true);
    if (assembleWithComgr(text, isaName, info->waveFrontSize, log)) return std::nullopt;
    if (log.empty()) return std::string("the assembler rejected the text and wrote no log");
    return log;
}

std::vector<std::string> modifierTexts(const StinkyInstruction& inst) {
    std::vector<std::string> out;
    for (const auto& modifier : inst.getModifiers()) {
        std::ostringstream text;
        AsmPrinter printer(text);
        if (printer.printModifier(*modifier)) out.push_back(text.str());
    }
    return out;
}

/// The run_passes options, read and checked before any pass runs.
struct RunOptions {
    std::optional<bool> unrollGemm;
    std::optional<bool> distributeGlobalRead;
    std::vector<std::pair<const GemmTileField*, uint32_t>> tile;
    std::optional<DebugPrintInstrumentation::Format> printBefore;
    std::optional<DebugPrintInstrumentation::Format> printAfter;
};

/// The `features` keys: the names run_passes, run_info, and the command line use.
struct FeatureFlag {
    const char* name;
    std::optional<bool> RunOptions::*field;
};
constexpr FeatureFlag kFeatureFlags[] = {
    {"unroll_gemm", &RunOptions::unrollGemm},
    {"distribute_global_read", &RunOptions::distributeGlobalRead},
};

/// A Python int other than a bool, which nanobind's integer casts accept too.
bool isPlainInt(nb::handle value) {
    return nb::isinstance<nb::int_>(value) && !nb::isinstance<nb::bool_>(value);
}

void readFeatures(const nb::dict& features, RunOptions& options) {
    for (auto [key, value] : features) {
        const std::string name = nb::str(key).c_str();
        const FeatureFlag* flag = nullptr;
        std::string keys;
        for (const FeatureFlag& candidate : kFeatureFlags) {
            if (name == candidate.name) flag = &candidate;
            if (!keys.empty()) keys += ", ";
            keys += candidate.name;
        }
        if (flag == nullptr)
            throw std::invalid_argument("unknown features key '" + name + "'; keys: " + keys);
        if (!nb::isinstance<nb::bool_>(value))
            throw nb::type_error(("features['" + name + "'] must be a bool").c_str());
        options.*(flag->field) = nb::cast<bool>(value);
    }
}

void readTile(const nb::dict& tile, RunOptions& options) {
    for (auto [key, value] : tile) {
        const std::string name = nb::str(key).c_str();
        const GemmTileField* field = nullptr;
        std::string keys;
        for (const GemmTileField& candidate : kGemmTileFields) {
            if (name == candidate.name) field = &candidate;
            if (!keys.empty()) keys += ", ";
            keys += candidate.name;
        }
        if (field == nullptr)
            throw std::invalid_argument("unknown tile key '" + name + "'; keys: " + keys);
        if (!isPlainInt(value))
            throw nb::type_error(("tile['" + name + "'] must be an int").c_str());
        uint32_t number = 0;
        if (!nb::try_cast(value, number))
            throw std::invalid_argument("tile['" + name + "'] must be in [0, 4294967295]");
        options.tile.emplace_back(field, number);
    }
}

std::optional<DebugPrintInstrumentation::Format> dumpFormat(
    const char* option, const std::optional<std::string>& value) {
    if (!value) return std::nullopt;
    if (*value == "stir") return DebugPrintInstrumentation::Format::Stir;
    if (*value == "asm") return DebugPrintInstrumentation::Format::Asm;
    throw std::invalid_argument(std::string(option) + " must be \"stir\" or \"asm\", not \"" +
                                *value + "\"");
}

/// The manager runs one pass, so the stream holds that pass's dump only.
std::shared_ptr<std::ostringstream> captureDump(PassManager& passManager,
                                                const StinkyAsmModule& module,
                                                DebugPrintInstrumentation::Format format,
                                                bool before) {
    auto stream = std::make_shared<std::ostringstream>();
    auto config = std::make_unique<PassManagerDebugConfig>();
    if (before) {
        config->setPrintBeforeAll();
        config->setDumpStreamBefore(stream);
    } else {
        config->setPrintAfterAll();
        config->setDumpStreamAfter(stream);
    }
    passManager.addInstrumentation(
        std::make_shared<DebugPrintInstrumentation>(std::move(config), &module, format));
    return stream;
}

/// The factory's feature defaults and tile stay unless the caller named a field.
void applyOptions(PassManager& passManager, const RunOptions& options) {
    PassFeatureConfig features = passManager.getPassContext().getPassFeatureConfig();
    if (options.unrollGemm) features.loopConfig.unrollGemm = *options.unrollGemm;
    if (options.distributeGlobalRead)
        features.dagFeatures.distributeGlobalRead = *options.distributeGlobalRead;
    passManager.setPassFeatureConfig(features);

    GemmTileConfig tile = passManager.getPassContext().getGemmTileConfig();
    for (const auto& [field, value] : options.tile) tile.*(field->member) = value;
    passManager.setGemmTileConfig(tile);
}

/// Walk the functions instead of calling getFunction(name): that searches only
/// callees, and maps "" to the entry whatever the entry is called.
std::vector<Function*> selectFunctions(StinkyAsmModule& module,
                                       const std::optional<std::string>& name) {
    std::vector<Function*> selected;
    std::string names;
    for (Function* function : module.getFunctions()) {
        if (function == nullptr) continue;
        if (!names.empty()) names += ", ";
        names += "'" + function->getName() + "'";
        if (!name || function->getName() == *name) selected.push_back(function);
    }
    if (name && selected.empty())
        throw std::invalid_argument("no function '" + *name + "'; functions: " + names);
    return selected;
}

Function& selectFunction(StinkyAsmModule& module, const std::string& name) {
    return *selectFunctions(module, name).front();
}

const char* remarkKind(OptimizationRemark::Kind kind) {
    switch (kind) {
        case OptimizationRemark::Kind::Passed:
            return "passed";
        case OptimizationRemark::Kind::Missed:
            return "missed";
        case OptimizationRemark::Kind::Analysis:
            return "analysis";
    }
    return "";
}

std::string functionText(const Function& function) {
    std::ostringstream out;
    function.dump(out);
    return out.str();
}

PyPassRecord runOnePass(StinkyAsmModule& module, Function& function, const std::string& name,
                        const std::vector<std::string>& args, const RunOptions& options) {
    PassManager passManager = makeStandalonePassManager(module.getArch());
    applyOptions(passManager, options);
    PassContext& context = passManager.getPassContext();
    context.setRemarksEnabled(true);
    std::vector<OptimizationRemark> remarks;
    // The handler must not throw: it runs inside passes built without exceptions.
    context.setRemarkHandler([&](const OptimizationRemark& remark) { remarks.push_back(remark); });
    std::shared_ptr<std::ostringstream> before;
    std::shared_ptr<std::ostringstream> after;
    if (options.printBefore) before = captureDump(passManager, module, *options.printBefore, true);
    if (options.printAfter) after = captureDump(passManager, module, *options.printAfter, false);

    std::unique_ptr<Pass> pass = createCatalogPass(name, args);
    if (!pass) throw std::invalid_argument(name + " rejected its arguments");
    passManager.addPass(std::move(pass));
    const std::string textBefore = functionText(function);
    passManager.run(function);

    bool passed = false;
    std::string missed;
    for (const OptimizationRemark& remark : remarks) {
        if (remark.kind == OptimizationRemark::Kind::Passed) passed = true;
        if (remark.kind != OptimizationRemark::Kind::Missed) continue;
        if (!missed.empty()) missed += '\n';
        missed += "missed: " + remark.passName + ": " + remark.message;
    }
    if (!missed.empty() && !passed) throw std::runtime_error(missed);
    if (context.getAnalysisFailed())
        throw std::runtime_error(name + " reported an analysis failure");

    PyPassRecord record;
    record.passName = name;
    record.function = function.getName();
    record.changed = functionText(function) != textBefore;
    for (const OptimizationRemark& remark : remarks)
        record.remarks.append(PyRemark{remarkKind(remark.kind), remark.remarkName, remark.message});
    if (before) record.before = before->str();
    if (after) record.after = after->str();
    return record;
}

nb::list runPasses(StinkyAsmModule& module, const std::vector<std::string>& specs,
                   const std::optional<std::string>& functionName,
                   const std::optional<nb::dict>& features, const std::optional<nb::dict>& tile,
                   const std::optional<std::string>& printBefore,
                   const std::optional<std::string>& printAfter) {
    struct Requested {
        std::string name;
        std::vector<std::string> args;
    };
    std::vector<Requested> requested;
    requested.reserve(specs.size());
    for (const std::string& spec : specs) {
        const size_t eq = spec.find('=');
        Requested item;
        item.name = trimCopy(eq == std::string::npos ? spec : spec.substr(0, eq));
        if (eq != std::string::npos) item.args = splitPassArgs(spec.substr(eq + 1));
        if (findCatalogPass(item.name) == nullptr)
            throw std::invalid_argument("unknown pass '" + item.name +
                                        "'; passes: " + catalogNames());
        if (createCatalogPass(item.name, item.args) == nullptr)
            throw std::invalid_argument(item.name + " rejected its arguments");
        requested.push_back(std::move(item));
    }
    RunOptions options;
    if (features) readFeatures(*features, options);
    if (tile) readTile(*tile, options);
    options.printBefore = dumpFormat("print_before", printBefore);
    options.printAfter = dumpFormat("print_after", printAfter);
    const std::vector<Function*> functions = selectFunctions(module, functionName);

    if (!specs.empty()) {
        PassRunRecord run;
        run.passes = specs;
        run.function = functionName;
        for (const FeatureFlag& flag : kFeatureFlags) {
            if (const std::optional<bool>& value = options.*(flag.field))
                run.features.emplace_back(flag.name, *value);
        }
        for (const auto& [field, value] : options.tile) run.tile.emplace_back(field->name, value);
        module.addPassRun(std::move(run));
    }

    nb::list records;
    for (Function* function : functions) {
        for (const Requested& item : requested)
            records.append(runOnePass(module, *function, item.name, item.args, options));
    }
    return records;
}

PyInst snapshotInstruction(const StinkyInstruction& inst) {
    PyInst out;
    out.opcode = opcodeOf(inst);
    if (inst.hasAttachedSSA()) {
        out.defs = ssaDefs(inst);
        out.uses = ssaUses(inst);
    } else {
        out.defs = physicalList(inst.getDestRegs());
        out.uses = physicalList(inst.getSrcRegs());
    }
    out.modifiers = modifierTexts(inst);
    return out;
}

PyBlock snapshotBlock(const BasicBlock& block) {
    PyBlock out;
    out.name = block.getLabel();
    for (BasicBlock* succ : block.getSuccessors()) {
        if (succ != nullptr) out.successors.push_back(succ->getLabel());
    }
    for (const IRBase& ir : block) {
        if (isIndexedInstruction(ir))
            out.instructions.append(snapshotInstruction(*cast<StinkyInstruction>(&ir)));
    }
    return out;
}

nb::list moduleFunctions(nb::handle owner) {
    nb::list out;
    StinkyAsmModule& module = nb::cast<StinkyAsmModule&>(owner);
    for (Function* function : module.getFunctions()) {
        if (function == nullptr) continue;
        PyFunction snapshot;
        snapshot.name = function->getName();
        snapshot.hasAttachedSSA = function->hasAttachedSSA();
        for (const BasicBlock& block : *function) snapshot.blocks.append(snapshotBlock(block));
        out.append(std::move(snapshot));
    }
    return out;
}

BasicBlock& selectBlock(Function& function, const std::string& name) {
    std::string names;
    for (BasicBlock& block : function) {
        if (block.getLabel() == name) return block;
        if (!names.empty()) names += ", ";
        names += "'^" + block.getLabel() + "'";
    }
    throw std::invalid_argument("no block '^" + name + "' in function '" + function.getName() +
                                "'; blocks: " + names);
}

StinkyInstruction& selectInstruction(BasicBlock& block, int index) {
    int count = 0;
    for (IRBase& ir : block) {
        if (!isIndexedInstruction(ir)) continue;
        if (count++ == index) return *cast<StinkyInstruction>(&ir);
    }
    throw std::out_of_range("no instruction " + std::to_string(index) + " in block '^" +
                            block.getLabel() + "', which has " + std::to_string(count));
}

/// A physical operand from one of the forms functions() lists.
StinkyRegister operandFromPython(nb::handle value) {
    const std::string text = nb::repr(value).c_str();
    if (isPlainInt(value)) {
        int32_t literal = 0;
        if (!nb::try_cast(value, literal))
            throw std::invalid_argument(text + " does not fit a 32-bit literal");
        return StinkyRegister(static_cast<int>(literal));
    }
    if (nb::isinstance<nb::float_>(value)) return StinkyRegister(nb::cast<double>(value));
    if (nb::isinstance<nb::str>(value)) return StinkyRegister(nb::cast<std::string>(value));
    if (nb::isinstance<nb::tuple>(value)) {
        const auto tuple = nb::borrow<nb::tuple>(value);
        const std::string head = tuple.size() > 0 && nb::isinstance<nb::str>(tuple[0])
                                     ? nb::cast<std::string>(tuple[0])
                                     : std::string();
        auto field = [&tuple](size_t i, auto& out) {
            return isPlainInt(tuple[i]) && nb::try_cast(tuple[i], out);
        };
        if (tuple.size() == 4 && head == "hwreg") {
            uint16_t id = 0;
            uint16_t offset = 0;
            uint16_t size = 0;
            if (field(1, id) && field(2, offset) && field(3, size))
                return StinkyRegister::Hwreg(id, offset, size);
        } else if (tuple.size() == 3 && !head.empty()) {
            const std::optional<RegType> type = tryParseRegType(head);
            uint32_t index = 0;
            uint16_t count = 0;
            if (type && field(1, index) && field(2, count) && index < StinkyRegister::kVirtualBit &&
                count > 0)
                return StinkyRegister(*type, index, count);
        } else if (tuple.size() == 2) {
            throw std::invalid_argument(text +
                                        " is an SSA value; set_operand takes physical operands");
        }
    }
    throw std::invalid_argument(text +
                                " is not an operand; give (class, index, count), an int, a float, "
                                "a str, or (\"hwreg\", id, offset, size)");
}

std::vector<Diagnostic> verifyModule(StinkyAsmModule& module,
                                     const std::optional<std::string>& functionName) {
    std::vector<Diagnostic> found;
    for (Function* function : selectFunctions(module, functionName)) {
        std::vector<Diagnostic> more = verifyStinkyIR(*function);
        found.insert(found.end(), std::make_move_iterator(more.begin()),
                     std::make_move_iterator(more.end()));
    }
    return found;
}

std::vector<std::string> replaceBlock(StinkyAsmModule& module, const std::string& functionName,
                                      const std::string& blockName, const std::string& body) {
    BasicBlock& block = selectBlock(selectFunction(module, functionName), blockName);
    if (isSsaForm(body))
        throw std::invalid_argument("SSA form does not round-trip; give the body as physical STIR");
    ParseResult parsed = parseBlockBodyWithDiagnostics(body);
    raiseErrors(parsed.diagnostics);
    std::vector<Diagnostic> diagnostics = std::move(parsed.diagnostics);
    const StinkyErrorCode code =
        StinkyIRConverter::replaceBlockBody(*parsed.parsedFunction->blocks.front(), block,
                                            checkedArchId(module.getArch()), diagnostics);
    raiseErrors(diagnostics);
    if (code != StinkyErrorCode::SUCCESS)
        throw std::runtime_error("block '^" + blockName + "' could not be replaced");
    std::vector<std::string> warnings;
    for (const Diagnostic& diagnostic : diagnostics) warnings.push_back(diagnostic.format());
    return warnings;
}

void setOperand(StinkyAsmModule& module, const std::string& functionName,
                const std::string& blockName, int index, const nb::tuple& operand,
                nb::handle value) {
    Function& function = selectFunction(module, functionName);
    if (function.hasAttachedSSA())
        throw std::invalid_argument("function '" + functionName +
                                    "' has attached SSA, so functions() lists SSA values, not the "
                                    "physical operands set_operand edits");
    StinkyInstruction& inst = selectInstruction(selectBlock(function, blockName), index);

    const std::string side = operand.size() == 2 && nb::isinstance<nb::str>(operand[0])
                                 ? nb::cast<std::string>(operand[0])
                                 : std::string();
    int position = 0;
    if ((side != "defs" && side != "uses") || !isPlainInt(operand[1]) ||
        !nb::try_cast(operand[1], position))
        throw std::invalid_argument("operand must be (\"defs\", i) or (\"uses\", i), not " +
                                    std::string(nb::repr(operand).c_str()));
    const bool isDef = side == "defs";
    const std::vector<StinkyRegister>& regs = isDef ? inst.getDestRegs() : inst.getSrcRegs();
    std::optional<size_t> slot;
    int listed = 0;
    for (size_t i = 0; i < regs.size() && !slot; ++i) {
        if (isListedOperand(regs[i]) && listed++ == position) slot = i;
    }
    if (!slot)
        throw std::out_of_range("no " + side + "[" + std::to_string(position) +
                                "] on that instruction, which has " + std::to_string(listed));

    const StinkyRegister replacement = operandFromPython(value);
    discardDefUseAnalysis(function);
    if (isDef)
        inst.setDestReg(*slot, replacement);
    else
        inst.setSrcReg(*slot, replacement);
}

/// (block, index, "use" or "def"), with index None for a block's leading pair
/// or a label, and None for a slot past the function's last.
nb::object placeTuple(const SlotPlace& place) {
    if (place.block == nullptr) return nb::none();
    nb::object index = place.index < 0 ? nb::none() : nb::cast(place.index);
    return nb::make_tuple(place.block->getLabel(), index, place.def ? "def" : "use");
}

Function& liftedFunction(StinkyAsmModule& module, const std::string& name, const char* query) {
    Function& function = selectFunction(module, name);
    if (!function.hasAttachedSSA())
        throw std::invalid_argument(std::string(query) + "() reads attached SSA, and function '" +
                                    name + "' has none; run LiftAsmRegistersToSSAPass first");
    return function;
}

PyLiveRanges liveRanges(StinkyAsmModule& module, const std::string& name) {
    const Function& function = liftedFunction(module, name, "live_ranges");
    const SSALiveIntervals live = computeSSALiveIntervals(function);
    PyLiveRanges out;
    for (size_t id = 1; id <= live.valueCount(); ++id) {
        const auto value = static_cast<SSAValueID>(id);
        const LiveRange& range = live.rangeOf(value);
        if (range.empty()) continue;
        PyLiveRange entry;
        entry.value = value;
        entry.regClass = regTypeToString(live.classOf(value));
        entry.width = live.widthOf(value);
        entry.reg = nb::none();
        const StinkySSAValue* ssa = function.ssaArena().get(value);
        if (ssa != nullptr && ssa->hasPhysicalBinding()) {
            const StinkySSAValue::PhysicalBinding& bound = ssa->physical();
            entry.reg = physicalOperand(StinkyRegister(bound.type, bound.idx, bound.num));
        }
        for (const LiveSegment& segment : range.segments())
            entry.segments.append(nb::make_tuple(segment.start, segment.end));
        entry.start = range.start();
        entry.end = range.end();
        entry.length = range.length();
        entry.line = live.rangeLine(value);
        out.ranges.append(std::move(entry));
    }
    const SSASlotIndexes& slots = live.slots();
    out.places.reserve(slots.slotCount());
    for (SlotIndex slot = 0; slot < slots.slotCount(); ++slot)
        out.places.push_back(placeTuple(slots.placeOf(slot)));
    return out;
}

PyPressure pressure(StinkyAsmModule& module, const std::string& name) {
    const SSALiveIntervals live = computeSSALiveIntervals(liftedFunction(module, name, "pressure"));
    PyPressure out;
    for (const auto& [regClass, dwords] : live.peakPressure()) {
        const SlotIndex slot = live.peakSlot(regClass);
        const std::string className = regTypeToString(regClass);
        out.peaks[className.c_str()] =
            nb::cast(PyPeak{className, dwords, slot, placeTuple(live.slots().placeOf(slot))});
    }
    out.vgpr = live.peakPressure(RegType::V);
    out.sgpr = live.peakPressure(RegType::S);
    return out;
}

nb::list hazards(StinkyAsmModule& module, const std::string& name) {
    nb::list out;
    for (const HazardGap& pair : findHazardGaps(selectFunction(module, name))) {
        out.append(PyHazardGap{pair.rule, pair.required, pair.gap, pair.ok(),
                               pair.block->getLabel(), pair.producerIndex, pair.consumerIndex,
                               opcodeOf(*pair.producer), opcodeOf(*pair.consumer)});
    }
    return out;
}

PyFunctionSize functionSize(StinkyAsmModule& module, const std::string& name) {
    const Function& function = selectFunction(module, name);
    for (const FunctionSize& size : measureFunctionSizes(module.getFunctions())) {
        if (size.function != &function) continue;
        PyFunctionSize out{size.start, size.bytes, size.cycles, size.instructions, nb::dict()};
        for (const auto& [label, offset] : size.labels)
            out.labels[label.c_str()] = nb::cast(offset);
        return out;
    }
    throw std::runtime_error("function '" + name + "' is not laid out in its module");
}

}  // namespace

void bind_stir_api(nb::module_& m, nb::class_<StinkyAsmModule>& cls) {
    nb::class_<Diagnostic>(m, "Diagnostic")
        .def_prop_ro(
            "level",
            [](const Diagnostic& diagnostic) {
                return std::string(Diagnostic::levelName(diagnostic.getLevel()));
            },
            "\"error\", \"warning\", or \"note\"")
        .def_prop_ro(
            "id",
            [](const Diagnostic& diagnostic) {
                return std::string(diagIdName(diagnostic.getId()));
            },
            "The kind of failure, such as \"unknown-mnemonic\"; str() shows it in brackets")
        .def_prop_ro("message",
                     [](const Diagnostic& diagnostic) { return diagnostic.getMessage(); })
        .def_prop_ro(
            "line",
            [](const Diagnostic& diagnostic) -> std::optional<unsigned> {
                if (diagnostic.getIRLocation() != nullptr) return std::nullopt;
                return diagnostic.getLine();
            },
            "Line in the loaded text, or None for a diagnostic about built IR")
        .def_prop_ro("column",
                     [](const Diagnostic& diagnostic) -> std::optional<unsigned> {
                         if (diagnostic.getIRLocation() != nullptr) return std::nullopt;
                         return diagnostic.getColumn();
                     })
        .def_prop_ro(
            "function",
            [](const Diagnostic& diagnostic) -> std::optional<std::string> {
                const IRLocation* at = diagnostic.getIRLocation();
                if (at == nullptr) return std::nullopt;
                return at->function;
            },
            "Function name, or None for a diagnostic about text")
        .def_prop_ro(
            "block",
            [](const Diagnostic& diagnostic) -> std::optional<std::string> {
                const IRLocation* at = diagnostic.getIRLocation();
                if (at == nullptr || at->block.empty()) return std::nullopt;
                return at->block;
            },
            "Block name, or None")
        .def_prop_ro(
            "index",
            [](const Diagnostic& diagnostic) -> std::optional<int> {
                const IRLocation* at = diagnostic.getIRLocation();
                if (at == nullptr || at->index < 0) return std::nullopt;
                return at->index;
            },
            "Position in the block's instructions(), or None")
        .def("__str__", &Diagnostic::format)
        .def("__repr__", [](const Diagnostic& diagnostic) {
            return "<Diagnostic " + diagnostic.format() + ">";
        });

    nb::class_<PyFunction>(m, "Function")
        .def_prop_ro("name", [](const PyFunction& function) { return function.name; })
        .def_prop_ro(
            "has_attached_ssa", [](const PyFunction& function) { return function.hasAttachedSSA; },
            "True after a lift: defs and uses are SSA values, and live_ranges() and pressure() "
            "can read the function")
        .def(
            "blocks", [](const PyFunction& function) { return function.blocks; },
            "Basic blocks in layout order");

    nb::class_<PyBlock>(m, "Block")
        .def_prop_ro("name", [](const PyBlock& block) { return block.name; })
        .def(
            "successors", [](const PyBlock& block) { return block.successors; },
            "Labels of successor blocks")
        .def(
            "instructions", [](const PyBlock& block) { return block.instructions; },
            "Instructions in this block");

    nb::class_<PyInst>(m, "Instruction")
        .def_prop_ro("opcode", [](const PyInst& inst) { return inst.opcode; })
        .def_prop_ro("defs", [](const PyInst& inst) { return inst.defs; })
        .def_prop_ro("uses", [](const PyInst& inst) { return inst.uses; })
        .def_prop_ro(
            "modifiers", [](const PyInst& inst) { return inst.modifiers; },
            "One `mod.X = { ... }` string per modifier, as serialize() prints them");

    nb::class_<PyRemark>(m, "Remark")
        .def_prop_ro(
            "kind", [](const PyRemark& remark) { return remark.kind; },
            "\"passed\", \"missed\", or \"analysis\"")
        .def_prop_ro("remark_name", [](const PyRemark& remark) { return remark.remarkName; })
        .def_prop_ro("message", [](const PyRemark& remark) { return remark.message; });

    nb::class_<PyPassRecord>(m, "PassRecord")
        .def_prop_ro("pass_name", [](const PyPassRecord& record) { return record.passName; })
        .def_prop_ro("function", [](const PyPassRecord& record) { return record.function; })
        .def_prop_ro(
            "changed", [](const PyPassRecord& record) { return record.changed; },
            "True when the function prints different physical STIR after the pass")
        .def_prop_ro(
            "remarks", [](const PyPassRecord& record) { return record.remarks; },
            "Every remark the pass emitted on this function, in order")
        .def_prop_ro(
            "before", [](const PyPassRecord& record) { return record.before; },
            "The print_before dump, or None")
        .def_prop_ro(
            "after", [](const PyPassRecord& record) { return record.after; },
            "The print_after dump, or None");

    nb::class_<PyLiveRange>(m, "LiveRange")
        .def_prop_ro("value", [](const PyLiveRange& range) { return range.value; })
        .def_prop_ro("reg_class", [](const PyLiveRange& range) { return range.regClass; })
        .def_prop_ro(
            "width", [](const PyLiveRange& range) { return range.width; }, "DWORDs")
        .def_prop_ro(
            "register", [](const PyLiveRange& range) { return range.reg; },
            "The register the lift took the value from, as (class, index, count), or None")
        .def_prop_ro(
            "segments", [](const PyLiveRange& range) { return range.segments; },
            "Half-open (start, end) slot pairs, in order")
        .def_prop_ro("start", [](const PyLiveRange& range) { return range.start; })
        .def_prop_ro(
            "end", [](const PyLiveRange& range) { return range.end; },
            "One past the last live slot")
        .def_prop_ro(
            "length", [](const PyLiveRange& range) { return range.length; }, "Slots covered")
        .def("__str__", [](const PyLiveRange& range) { return range.line; })
        .def("__repr__", [](const PyLiveRange& range) { return "<LiveRange " + range.line + ">"; });

    nb::class_<PyLiveRanges>(m, "LiveRanges")
        .def_prop_ro(
            "ranges", [](const PyLiveRanges& live) { return live.ranges; },
            "One LiveRange per live value, by value id")
        .def_prop_ro("slot_count", [](const PyLiveRanges& live) { return live.places.size(); })
        .def(
            "place",
            [](const PyLiveRanges& live, int64_t slot) -> nb::object {
                if (slot < 0 || static_cast<size_t>(slot) >= live.places.size()) return nb::none();
                return live.places[static_cast<size_t>(slot)];
            },
            nb::arg("slot"),
            "(block, index, \"use\" or \"def\") for a slot, index as in instructions() and None "
            "for a block's leading pair or a label. None past the last slot");

    nb::class_<PyPeak>(m, "PressurePeak")
        .def_prop_ro("reg_class", [](const PyPeak& peak) { return peak.regClass; })
        .def_prop_ro(
            "dwords", [](const PyPeak& peak) { return peak.dwords; },
            "Most DWORDs of the class live at once")
        .def_prop_ro(
            "slot", [](const PyPeak& peak) { return peak.slot; }, "First slot at the peak")
        .def_prop_ro(
            "place", [](const PyPeak& peak) { return peak.place; },
            "Where that slot is, as LiveRanges.place gives it");

    nb::class_<PyPressure>(m, "Pressure")
        .def_prop_ro(
            "peaks", [](const PyPressure& pressure) { return pressure.peaks; },
            "PressurePeak by register class, for each class the lift covered")
        .def_prop_ro(
            "vgpr", [](const PyPressure& pressure) { return pressure.vgpr; },
            "Peak VGPR DWORDs, 0 when none are live")
        .def_prop_ro(
            "sgpr", [](const PyPressure& pressure) { return pressure.sgpr; },
            "Peak SGPR DWORDs, 0 when none are live");

    nb::class_<PyHazardGap>(m, "HazardGap")
        .def_prop_ro("rule", [](const PyHazardGap& gap) { return gap.rule; })
        .def_prop_ro(
            "required", [](const PyHazardGap& gap) { return gap.required; },
            "Cycles the rule needs")
        .def_prop_ro(
            "gap", [](const PyHazardGap& gap) { return gap.gap; },
            "Cycles of the instructions between producer and consumer")
        .def_prop_ro(
            "ok", [](const PyHazardGap& gap) { return gap.ok; }, "gap is at least required")
        .def_prop_ro("block", [](const PyHazardGap& gap) { return gap.block; })
        .def_prop_ro(
            "producer", [](const PyHazardGap& gap) { return gap.producer; },
            "Position in the block's instructions()")
        .def_prop_ro(
            "consumer", [](const PyHazardGap& gap) { return gap.consumer; },
            "Position in the block's instructions()")
        .def_prop_ro("producer_opcode", [](const PyHazardGap& gap) { return gap.producerOpcode; })
        .def_prop_ro("consumer_opcode", [](const PyHazardGap& gap) { return gap.consumerOpcode; });

    nb::class_<PyRunInfo>(m, "RunInfo")
        .def_prop_ro("arch",
                     [](const PyRunInfo& run) {
                         return nb::make_tuple(run.arch[0], run.arch[1], run.arch[2]);
                     })
        .def_prop_ro(
            "version", [](const PyRunInfo& run) { return run.version; },
            "The library version that made the run")
        .def_prop_ro(
            "passes", [](const PyRunInfo& run) { return run.passes; },
            "The pass specs, in order, as run_passes was given them")
        .def_prop_ro(
            "function", [](const PyRunInfo& run) { return run.function; },
            "The function the run was limited to, or None for every function")
        .def_prop_ro(
            "features", [](const PyRunInfo& run) { return run.features; },
            "The features the caller set")
        .def_prop_ro(
            "tile", [](const PyRunInfo& run) { return run.tile; }, "The tile fields the caller set")
        .def_prop_ro(
            "command", [](const PyRunInfo& run) { return run.command; },
            "The stinkytofu command that repeats the run, without the input file")
        .def("__str__", [](const PyRunInfo& run) { return run.line; })
        .def("__repr__", [](const PyRunInfo& run) { return "<RunInfo " + run.line + ">"; });

    nb::class_<PyFunctionSize>(m, "FunctionSize")
        .def_prop_ro(
            "start", [](const PyFunctionSize& size) { return size.start; },
            "Byte offset of the function in the kernel image")
        .def_prop_ro(
            "bytes", [](const PyFunctionSize& size) { return size.bytes; }, "Encoded bytes")
        .def_prop_ro("cycles", [](const PyFunctionSize& size) { return size.cycles; })
        .def_prop_ro(
            "instructions", [](const PyFunctionSize& size) { return size.instructions; },
            "Instructions counted, not labels or directives")
        .def_prop_ro(
            "labels", [](const PyFunctionSize& size) { return size.labels; },
            "Each label's byte offset in the image, in program order");

    cls.def("copy", &copyModule,
            "A separate module with the same functions, metadata, kernel header, and history. "
            "Attached SSA is not copied");
    cls.def("serialize", &serializeModule, nb::arg("ssa") = false, nb::kw_only(),
            nb::arg("comments") = false,
            "Print this module. ssa=True is a diagnostic view and does not round-trip. "
            "comments=True writes one // line per recorded run first");
    cls.def("emit_asm", &emitModuleAsm, nb::kw_only(), nb::arg("comments") = false,
            nb::arg("cycle_info") = false, nb::arg("symbolic_names") = false,
            nb::arg("signature") = true,
            "Assembler source for every function, entry first. signature writes the kernel "
            "header load_asm kept. comments writes the recorded runs, header comments, and "
            "kept source comments");
    cls.def("run_info", &runInfo,
            "One RunInfo per run_passes call that ran passes on this module, oldest first");
    cls.def("assemble", &assembleModule,
            "Assemble emit_asm() with amd_comgr. None when it assembles, else the assembler's "
            "messages");
    cls.def(
        "load_warnings",
        [](const StinkyAsmModule& module) {
            std::vector<std::string> out;
            for (const Diagnostic& diagnostic : module.getLoadWarnings())
                out.push_back(diagnostic.format());
            return out;
        },
        "Warnings from the load that built this module, one line each");
    cls.def("run_passes", &runPasses, nb::arg("passes"), nb::kw_only(),
            nb::arg("function") = nb::none(), nb::arg("features") = nb::none(),
            nb::arg("tile") = nb::none(), nb::arg("print_before") = nb::none(),
            nb::arg("print_after") = nb::none(),
            "Run named catalog passes, in order, on every function or the one named. Returns one "
            "PassRecord per pass on each function");
    cls.def("functions", &moduleFunctions, "Functions in this module, entry first");
    cls.def("verify", &verifyModule, nb::kw_only(), nb::arg("function") = nb::none(),
            "Every problem in every function, or the one named, as Diagnostics. Empty when "
            "nothing is wrong");
    cls.def("replace_block", &replaceBlock, nb::arg("function"), nb::arg("block"), nb::arg("body"),
            "Replace a block's instructions with STIR body text: the lines serialize() prints "
            "between its header and its Successors line. The block keeps its name, label, and "
            "edges. Returns the warnings; the module is unchanged when it raises");
    cls.def("set_operand", &setOperand, nb::arg("function"), nb::arg("block"), nb::arg("index"),
            nb::arg("operand"), nb::arg("value"),
            "Set one physical operand: operand is (\"defs\", i) or (\"uses\", i) of the "
            "instruction at index, and value takes a form functions() lists");
    cls.def("live_ranges", &liveRanges, nb::arg("function"),
            "Where each SSA value of a lifted function is live, in slots, with a place() lookup");
    cls.def("pressure", &pressure, nb::arg("function"),
            "Peak DWORDs live at once in each register class of a lifted function, and where");
    cls.def("hazards", &hazards, nb::arg("function"),
            "Every producer -> consumer pair HazardGapAnalysisPass measures, passing or not");
    cls.def("size", &functionSize, nb::arg("function"),
            "Encoded bytes and cycles of a function, laid out where the kernel image places it");

    m.def("load_stir", &loadStir, nb::arg("text"), nb::arg("arch") = std::array<int, 3>{12, 5, 0},
          "Parse physical STIR into a module that owns its functions");
    m.def("load_stir_file", &loadStirFile, nb::arg("path"),
          nb::arg("arch") = std::array<int, 3>{12, 5, 0}, "Load physical STIR from a file");
    m.def("load_asm", &loadAsm, nb::arg("text"), nb::arg("arch") = std::array<int, 3>{12, 5, 0},
          nb::kw_only(), nb::arg("preserve_symbolic_regs") = false,
          nb::arg("preserve_comments") = false, nb::arg("from_label") = nb::none(),
          nb::arg("to_label") = nb::none(),
          "Parse assembler source into a module. from_label and to_label keep only that region, "
          "without the kernel header");
    m.def("load_asm_file", &loadAsmFile, nb::arg("path"),
          nb::arg("arch") = std::array<int, 3>{12, 5, 0}, nb::kw_only(),
          nb::arg("preserve_symbolic_regs") = false, nb::arg("preserve_comments") = false,
          nb::arg("from_label") = nb::none(), nb::arg("to_label") = nb::none(),
          "Load assembler source from a file");
    m.def(
        "pass_names",
        []() {
            std::vector<std::string> names;
            for (const PassInfo& info : passCatalog()) names.emplace_back(info.name);
            return names;
        },
        "Names of the passes run_passes can run");
    m.def(
        "version", []() { return std::string(getRuntimeVersion()); },
        "The library version, as run_info() records it");
}
