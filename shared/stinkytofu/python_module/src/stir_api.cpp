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
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/pipeline/PassCatalog.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace nb = nanobind;
using namespace stinkytofu;

namespace {

/// A query copies names, opcodes, and operands. The objects stay readable
/// after a later pass frees or rebuilds the function.
struct PyInst {
    std::string opcode;
    nb::list defs;
    nb::list uses;
};

struct PyBlock {
    std::string name;
    std::vector<std::string> successors;
    nb::list instructions;
};

struct PyFunction {
    std::string name;
    nb::list blocks;
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

/// AsmPrinter wraps a module in `st.module @name { ... }`. The parser reads
/// `st.func` at the top level, so peel one matching wrapper before parsing.
std::string withoutModuleWrapper(const std::string& text) {
    size_t i = 0;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    if (text.compare(i, 9, "st.module") != 0) return text;
    const size_t open = text.find('{', i);
    if (open == std::string::npos) throw std::invalid_argument("st.module is missing '{'");
    size_t end = text.size();
    while (end > open + 1 && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    if (text[end - 1] != '}') throw std::invalid_argument("st.module is missing '}'");
    return text.substr(open + 1, end - open - 2);
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

std::vector<std::string> splitArgs(const std::string& text) {
    std::vector<std::string> args;
    std::string current;
    for (char ch : text) {
        if (ch == ',') {
            current = trimCopy(std::move(current));
            if (!current.empty()) args.push_back(std::move(current));
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    current = trimCopy(std::move(current));
    if (!current.empty()) args.push_back(std::move(current));
    return args;
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

nb::list physicalList(const std::vector<StinkyRegister>& regs) {
    nb::list out;
    for (const StinkyRegister& reg : regs) appendOperand(out, physicalOperand(reg));
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

std::unique_ptr<StinkyAsmModule> loadStir(const std::string& text, std::array<int, 3> arch) {
    if (isSsaForm(text))
        throw std::invalid_argument(
            "SSA form does not round-trip; load physical STIR and run LiftAsmRegistersToSSAPass");

    const ArchHelper& arches = ArchHelper::getInstance();
    if (arches.getArchInfo(static_cast<uint32_t>(arch[0]), static_cast<uint32_t>(arch[1]),
                           static_cast<uint32_t>(arch[2])) == nullptr) {
        throw std::invalid_argument("unknown arch (" + std::to_string(arch[0]) + ", " +
                                    std::to_string(arch[1]) + ", " + std::to_string(arch[2]) + ")");
    }
    const GfxArchID archId =
        getGfxArchID(static_cast<uint32_t>(arch[0]), static_cast<uint32_t>(arch[1]),
                     static_cast<uint32_t>(arch[2]));

    MultiParseResult parsed = parseAllSourceStringsWithDiagnostics(withoutModuleWrapper(text));
    const std::string errors = errorText(parsed.diagnostics);
    if (!errors.empty()) throw std::invalid_argument(errors);
    if (parsed.functions.empty()) throw std::invalid_argument("no st.func in STIR text");

    StinkyAsmModule::ModuleOptions options{};
    auto module =
        std::make_unique<StinkyAsmModule>(parsed.functions.front()->funcName, arch, options);
    bool first = true;
    for (auto& parsedFunc : parsed.functions) {
        if (parsedFunc == nullptr) continue;
        Function* function = nullptr;
        if (first) {
            function = &module->getFunction();
            first = false;
        } else {
            if (parsedFunc->funcName == module->getFunction().getName() ||
                module->getFunction(parsedFunc->funcName) != nullptr) {
                throw std::invalid_argument("duplicate function '" + parsedFunc->funcName + "'");
            }
            function = &module->createFunction(parsedFunc->funcName);
        }
        const StinkyErrorCode result =
            StinkyIRConverter::populateFunctionFromParsed(*parsedFunc, *function, archId);
        if (result != StinkyErrorCode::SUCCESS)
            throw std::invalid_argument("failed to build function '" + parsedFunc->funcName + "'");
    }
    return module;
}

std::unique_ptr<StinkyAsmModule> loadStirFile(const std::string& path, std::array<int, 3> arch) {
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("cannot read " + path);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return loadStir(buffer.str(), arch);
}

std::string serializeModule(const StinkyAsmModule& module, bool ssa) {
    std::ostringstream out;
    AsmPrinterOptions options;
    options.ssaForm = ssa;
    AsmPrinter printer(out, options);
    printer.print(module);
    return out.str();
}

/// Same defaults as stinkytofu-opt individual-pass mode
/// (getPassFeatureConfig in tools/stinkytofu-opt/stinkytofu-opt.hpp).
PassFeatureConfig optPassFeatureConfig() {
    PassFeatureConfig config;
    config.loopConfig.unrollGemm = true;
    config.dagFeatures.distributeGlobalRead = true;
    return config;
}

void runOnePass(StinkyAsmModule& module, Function& function, const std::string& name,
                const std::vector<std::string>& args) {
    const std::array<int, 3> arch = module.getArch();
    PassManager passManager;
    registerAllAnalyses(passManager.getAnalysisManager());
    passManager.setPassFeatureConfig(optPassFeatureConfig());
    GemmTileConfig tile;
    tile.arch = arch;
    passManager.setGemmTileConfig(tile);
    const GfxArchID archId =
        getGfxArchID(static_cast<uint32_t>(arch[0]), static_cast<uint32_t>(arch[1]),
                     static_cast<uint32_t>(arch[2]));
    AsmCapsConfig caps = ToolchainCaps::probe(archId);
    if (arch == std::array<int, 3>{12, 5, 0}) {
        caps.requiresXCntForVolatileVMEM = true;
        caps.enableXnackReplay = true;
    }
    passManager.setAsmCapsConfig(caps);
    passManager.getPassContext().setRemarksEnabled(true);

    bool missed = false;
    bool passed = false;
    std::vector<std::string> missedLines;
    // The handler must not throw: it runs inside passes built without exceptions.
    passManager.getPassContext().setRemarkHandler(
        [&](const char* prefix, const std::string& passName, const std::string& message) {
            const std::string line = std::string(prefix) + ": " + passName + ": " + message;
            if (std::string(prefix) == "remark") passed = true;
            if (std::string(prefix) == "missed") {
                missed = true;
                missedLines.push_back(line);
            }
        });

    bool known = false;
    for (const PassInfo& info : passCatalog()) {
        if (name == info.name) {
            known = true;
            break;
        }
    }
    if (!known)
        throw std::invalid_argument("unknown pass '" + name + "'; passes: " + catalogNames());

    std::unique_ptr<Pass> pass = createCatalogPass(name, args);
    if (!pass) throw std::invalid_argument(name + " rejected its arguments");
    passManager.addPass(std::move(pass));
    passManager.run(function);

    if (missed && !passed) {
        std::ostringstream out;
        for (size_t i = 0; i < missedLines.size(); ++i) {
            if (i > 0) out << '\n';
            out << missedLines[i];
        }
        throw std::runtime_error(out.str());
    }
    if (passManager.getPassContext().getAnalysisFailed())
        throw std::runtime_error(name + " reported an analysis failure");
}

void runPasses(StinkyAsmModule& module, const std::vector<std::string>& specs) {
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
        if (eq != std::string::npos) item.args = splitArgs(spec.substr(eq + 1));
        if (item.name.empty())
            throw std::invalid_argument("unknown pass ''; passes: " + catalogNames());
        requested.push_back(std::move(item));
    }

    for (Function* function : module.getFunctions()) {
        if (function == nullptr) continue;
        for (const Requested& item : requested) runOnePass(module, *function, item.name, item.args);
    }
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
    return out;
}

PyBlock snapshotBlock(const BasicBlock& block) {
    PyBlock out;
    out.name = block.getLabel();
    for (BasicBlock* succ : block.getSuccessors()) {
        if (succ != nullptr) out.successors.push_back(succ->getLabel());
    }
    for (const IRBase& ir : block) {
        const auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst == nullptr || inst->getUnifiedOpcode() == GFX::LABEL) continue;
        out.instructions.append(snapshotInstruction(*inst));
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
        for (const BasicBlock& block : *function) snapshot.blocks.append(snapshotBlock(block));
        out.append(std::move(snapshot));
    }
    return out;
}

}  // namespace

void bind_stir_api(nb::module_& m, nb::class_<StinkyAsmModule>& cls) {
    nb::class_<PyFunction>(m, "Function")
        .def_prop_ro("name", [](const PyFunction& function) { return function.name; })
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
        .def_prop_ro("uses", [](const PyInst& inst) { return inst.uses; });

    cls.def("serialize", &serializeModule, nb::arg("ssa") = false,
            "Print this module. ssa=True is a diagnostic view and does not round-trip");
    cls.def("run_passes", &runPasses, nb::arg("passes"),
            "Run named catalog passes, in order, on every function");
    cls.def("functions", &moduleFunctions, "Functions in this module, entry first");

    m.def("load_stir", &loadStir, nb::arg("text"), nb::arg("arch") = std::array<int, 3>{12, 5, 0},
          "Parse physical STIR into a module that owns its functions");
    m.def("load_stir_file", &loadStirFile, nb::arg("path"),
          nb::arg("arch") = std::array<int, 3>{12, 5, 0}, "Load physical STIR from a file");
    m.def(
        "pass_names",
        []() {
            std::vector<std::string> names;
            for (const PassInfo& info : passCatalog()) names.emplace_back(info.name);
            return names;
        },
        "Names of the passes run_passes can run");
}
