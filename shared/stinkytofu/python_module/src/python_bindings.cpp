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

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <cstdint>
#include <sstream>

#include "HardwareCaps.hpp"
#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ComgrProbe.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/ir/logical/IntrinsicCall.hpp"
#include "stinkytofu/ir/logical/IntrinsicLibrary.hpp"
#include "stinkytofu/ir/logical/IntrinsicRegistry.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/transforms/logical/LowerLogicalModulePipeline.hpp"

namespace nb = nanobind;
using namespace stinkytofu;

// Forward declaration for logical count bindings
void init_logical_count(nb::module_& m);

NB_MODULE(_stinkytofu, m) {
    BackendRegistry::registerAllBackends();
    m.doc() = "StinkyTofu: High-Level IR for AMDGPU Assembly Generation (internal C++ module)";

    // ========================================================================
    // Bind StinkyAsmModule Class
    // ========================================================================
    nb::class_<StinkyAsmModule>(m, "StinkyAsmModule")
        .def("getName", &StinkyAsmModule::getName, "Get the name of this module")
        .def("setOutputName", &StinkyAsmModule::setOutputName,
             "Set the name used for output files (e.g. cost file); use full kernel name to match "
             ".o basename")
        .def("getOutputName", &StinkyAsmModule::getOutputName,
             "Get the output file basename; empty means use getName()")
        .def("setOutputDir", &StinkyAsmModule::setOutputDir,
             "Set output directory for cost file: comparison_output/<yaml_name>; file goes to "
             "<outputDir>/<kernel_name>/aggregated_instruction_cost.txt")
        .def("getOutputDir", &StinkyAsmModule::getOutputDir,
             "Get the output directory; empty means current directory")
        .def("emitAssembly", &StinkyAsmModule::emitAssembly,
             "Emit the assembly code for all instructions in this module")
        .def("getMetaDataU64", &StinkyAsmModule::getMetaDataU64, nb::arg("key"),
             "Get uint64 metadata from function by key")
        .def("runOptimizationPipeline", &StinkyAsmModule::runOptimizationPipeline,
             "Run the optimization pipeline on this module")
        .def("setPluginDataI64", &StinkyAsmModule::setPluginDataI64, nb::arg("key"),
             nb::arg("value"), "Set an integer plugin data value accessible by plugin passes")
        .def("getPluginDataI64", &StinkyAsmModule::getPluginDataI64, nb::arg("key"),
             nb::arg("defaultVal") = 0, "Get an integer plugin data value")
        .def("setPluginDataStr", &StinkyAsmModule::setPluginDataStr, nb::arg("key"),
             nb::arg("value"), "Set a string plugin data value accessible by plugin passes")
        .def("getPluginDataStr", &StinkyAsmModule::getPluginDataStr, nb::arg("key"),
             nb::arg("defaultVal") = "", "Get a string plugin data value")
        .def("numFunctions", &StinkyAsmModule::numFunctions,
             "Number of Functions (entry + callable functions) in this module")
        .def(
            "getFunctionNames",
            [](StinkyAsmModule& self) {
                std::vector<std::string> names;
                for (const auto* function : self.getFunctions()) {
                    if (function) names.push_back(function->getName());
                }
                return names;
            },
            "List every Function name in this module (entry first, then callable functions)")
        .def(
            "hasFunction",
            [](StinkyAsmModule& self, const std::string& name) {
                return self.getFunction(name) != nullptr;
            },
            nb::arg("name"), "Return true when this module contains a Function with the name")
        .def(
            "registerPassAtExtensionPoint",
            [](StinkyAsmModule& self, PipelineExtensionPoint ep, const std::string& passName) {
                self.getPassBuilder().registerAtExtensionPoint(
                    ep, [passName](PassManager& PM, StinkyAsmModule& module) {
                        auto pass = PassBuilder::createPassByName(passName, module);
                        if (pass) PM.addPass(std::move(pass));
                    });
            },
            nb::arg("extensionPoint"), nb::arg("passName"),
            "Register a named C++ pass at a pipeline extension point");

    // Pipeline extension point enum
    nb::enum_<PipelineExtensionPoint>(m, "PipelineExtensionPoint")
        .value("BeforeRegionPasses", PipelineExtensionPoint::BeforeRegionPasses)
        .value("InnerRegionBegin", PipelineExtensionPoint::InnerRegionBegin)
        .value("InnerRegionEnd", PipelineExtensionPoint::InnerRegionEnd)
        .value("AfterRegionPasses", PipelineExtensionPoint::AfterRegionPasses);

    // ------------------------------------------------------------------------
    // Bind CloneSpec so Python can construct entries for ModuleOptions::CloneList.
    // ------------------------------------------------------------------------
    nb::class_<CloneSpec>(m, "CloneSpec")
        .def(nb::init<std::string, std::string>(), nb::arg("name"), nb::arg("startLabel"))
        .def_rw("name", &CloneSpec::name)
        .def_rw("startLabel", &CloneSpec::startLabel);

    // ------------------------------------------------------------------------
    // Logical-IR -> Asm-IR one-shot lowering helper
    // ------------------------------------------------------------------------
    // Left-path entry point for KernelWriter-style Python code: build a
    // LogicalModule, then call lower_logical_module(...) to get back a
    // StinkyAsmModule ready for .emitAssembly() / .runOptimizationPipeline().
    // Mirrors the right path (toStinkyTofuModule for rocisa::Module). The
    // returned StinkyAsmModule borrows the LogicalInstructions from @p module
    // for IR-list membership but does NOT own them: keep @p module alive for
    // as long as the returned asm module is in use.
    m.def(
        "lower_logical_module",
        [](PyLogicalModule& module, std::array<int, 3> arch, const nb::object& options_obj) {
            StinkyAsmModule::ModuleOptions moduleOptions{};
            // Abs SW-prefetch base SGPR defaults to -1 (off); Tensile overrides via the
            // options dict below. (Replaces the removed SwPrefetchScratchSgpr default.)
            moduleOptions.SwInstructionPrefetchAbsBaseSgpr = -1;
            if (nb::isinstance<nb::dict>(options_obj)) {
                nb::dict options = nb::cast<nb::dict>(options_obj);
#define SET_MODULE_OPTION_LLM(name, type) \
    if (options.contains(#name)) nb::try_cast<type>(options[#name], moduleOptions.name);
                MODULE_OPTIONS_LIST(SET_MODULE_OPTION_LLM)
#undef SET_MODULE_OPTION_LLM
            }
            return lowerLogicalModuleToAsm(module, arch, moduleOptions);
        },
        nb::arg("module"), nb::arg("arch"), nb::arg("options") = nb::none(),
        "Run the standard logical-IR lowering pipeline on @p module and "
        "return the resulting StinkyAsmModule. @p arch is "
        "[major, minor, stepping] (e.g. [12, 5, 0] for gfx1250) and is used "
        "to look up the per-arch logical-to-asm mnemonic map. Keep @p module "
        "alive while using the returned StinkyAsmModule (it borrows "
        "LogicalInstruction nodes from it for IR-list membership). "
        "Optional @p options dict sets ModuleOptions fields (same keys as "
        "toStinkyTofuModule: CloneList, OptLevel, wavefrontSize, etc.).");

    // ========================================================================
    // Register Types
    // ========================================================================
    nb::enum_<RegType>(m, "RegType", "Register type enumeration")
        .value("V", RegType::V, "Vector Register (VGPR)")
        .value("S", RegType::S, "Scalar Register (SGPR)")
        .value("A", RegType::A, "Accumulator Register (AGPR)")
        .value("M", RegType::M, "Memory descriptor register (MGPR)")
        .value("ACC", RegType::ACC, "Accumulator Register (alternative)")
        .value("AGPR", RegType::AGPR, "Accumulator GPR")
        .value("VCC", RegType::VCC, "Vector Condition Code")
        .value("VCC_LO", RegType::VCC_LO, "Vector Condition Code (low)")
        .value("VCC_HI", RegType::VCC_HI, "Vector Condition Code (high)")
        .value("EXEC", RegType::EXEC, "Execution Mask")
        .value("EXEC_LO", RegType::EXEC_LO, "Execution Mask (low)")
        .value("EXEC_HI", RegType::EXEC_HI, "Execution Mask (high)")
        .value("SCC", RegType::SCC, "Scalar Condition Code")
        .value("LDS", RegType::LDS, "LDS dependency tracking pseudo-register")
        .value("UNKNOWN", RegType::UNKNOWN, "Unknown Register Type");

    // ========================================================================
    // StinkyRegister Class
    // ========================================================================
    // Note: Unlike rocisa which uses strings, stinkytofu uses enum (RegType) internally.
    // We need to expose StinkyRegister so nanobind can convert it properly.
    // However, users typically don't construct Register directly - they use helper functions.
    nb::class_<StinkyRegister>(m, "Register")
        // --- Constructors ---------------------------------------------------
        .def(nb::init<>(), "Create a null register")
        // Reg ctor with validation: silently accepting unknown type strings (the
        // raw nb::init binding) lets typos like Register("vgprFoo", 0, 1) build
        // a UNKNOWN-typed register that fails far away from the call site.
        .def(
            "__init__",
            [](StinkyRegister* self, const std::string& type, uint32_t index, uint16_t count) {
                if (!isValidRegTypeString(type)) {
                    throw nb::value_error(("Register: unknown register type '" + type +
                                           "'; expected one of v/s/a/m/... (see RegisterType.def)")
                                              .c_str());
                }
                new (self) StinkyRegister(type, index, count);
            },
            nb::arg("type"), nb::arg("index"), nb::arg("count") = 1,
            "Create a register (e.g., Register('v', 0, 1) for v0). Raises on unknown type.")
        .def(nb::init<float>(), nb::arg("value"), "Create a float literal")
        .def(nb::init<int>(), nb::arg("value"), "Create an int literal")
        // Single-string ctor → LiteralString. Used for keywords like MUBUF "off".
        // Distinct from the (type,index,count) overload by arg count.
        .def(
            "__init__",
            [](StinkyRegister* self, const std::string& literal_string) {
                new (self) StinkyRegister(literal_string);
            },
            nb::arg("literal_string"),
            "Create a literal string register (e.g., Register('off') for the MUBUF "
            "'off' keyword). Stored as LiteralString.")

        // --- Type / identity properties ------------------------------------
        .def_prop_ro(
            "reg_type",
            [](const StinkyRegister& r) -> RegType {
                if (r.dataType == StinkyRegister::Type::Register) {
                    return r.reg.type;
                }
                return RegType::UNKNOWN;
            },
            "Get the register type (V, S, A, etc.). UNKNOWN for non-Register values.")
        .def_prop_ro(
            "index",
            [](const StinkyRegister& r) -> int {
                if (r.dataType == StinkyRegister::Type::Register) {
                    return r.reg.idx;
                }
                return -1;
            },
            "Get the register index (-1 for non-Register values).")
        .def_prop_ro(
            "count",
            [](const StinkyRegister& r) -> int {
                if (r.dataType == StinkyRegister::Type::Register) {
                    return r.reg.num;
                }
                return 0;
            },
            "Get the register count (number of consecutive registers).")
        .def_prop_ro(
            "is_register",
            [](const StinkyRegister& r) { return r.dataType == StinkyRegister::Type::Register; },
            "True iff this carries a register reference (not a literal).")
        .def_prop_ro(
            "is_literal",
            [](const StinkyRegister& r) -> bool {
                return r.dataType == StinkyRegister::Type::LiteralInt ||
                       r.dataType == StinkyRegister::Type::LiteralDouble ||
                       r.dataType == StinkyRegister::Type::LiteralString;
            },
            "True iff this is any literal (int/double/string).")
        .def_prop_ro(
            "is_literal_string",
            [](const StinkyRegister& r) {
                return r.dataType == StinkyRegister::Type::LiteralString;
            },
            "True iff this is a LiteralString (e.g., MUBUF 'off' keyword).")
        .def_prop_ro(
            "literal_string",
            [](const StinkyRegister& r) -> std::optional<std::string> {
                if (r.dataType == StinkyRegister::Type::LiteralString) {
                    return r.literalValue;
                }
                return std::nullopt;
            },
            "Get the literal string content, or None if not a LiteralString.")

        // --- RegName accessors (rocisa-style symbolic name carrier) --------
        // stinkytofu's literalValue field doubles as the symbolic name slot for
        // Register-typed values. We marshal rocisa's (name, offsets[]) struct
        // to a "name+1+2+..." string at the binding boundary so stinkytofu's
        // existing IR transforms (e.g. LegalizationUtils::adjustSymbolicRegName)
        // can act on it without learning a new struct type.
        .def(
            "set_reg_name",
            [](StinkyRegister& r, const std::string& name, const std::vector<int>& offsets) {
                std::string full = name;
                for (int o : offsets) {
                    full += "+" + std::to_string(o);
                }
                r.setSymbolicName(full);
            },
            nb::arg("name"), nb::arg("offsets") = std::vector<int>{},
            "Attach a rocisa-style RegName (name + offsets) to this register. "
            "Stored on `literalValue`; encoded as 'name+offset1+offset2+...'.")
        .def(
            "get_reg_name",
            [](const StinkyRegister& r) -> std::pair<std::string, std::vector<int>> {
                std::string s = r.getSymbolicName();
                std::vector<int> offsets;
                if (s.empty()) {
                    return {"", offsets};
                }
                auto pos = s.find('+');
                std::string base = (pos == std::string::npos) ? s : s.substr(0, pos);
                while (pos != std::string::npos) {
                    auto next = s.find('+', pos + 1);
                    std::string token = s.substr(
                        pos + 1, next == std::string::npos ? std::string::npos : next - pos - 1);
                    offsets.push_back(std::stoi(token));
                    pos = next;
                }
                return {base, offsets};
            },
            "Read the attached RegName as (name, offsets[]). Returns ('', []) when no name set.")
        .def(
            "has_reg_name", [](const StinkyRegister& r) { return r.hasSymbolicName(); },
            "Whether this register currently carries a symbolic RegName.")
        .def(
            "clear_reg_name", [](StinkyRegister& r) { r.setSymbolicName(""); },
            "Detach any RegName, leaving the numeric (type,index,count) identity.")

        // --- Modifier flags (isMinus / isAbs) ------------------------------
        // C++ already stores isMinus/isAbs as 1-bit fields on reg; we just
        // expose Python toggles. Both setters are no-ops on literals.
        .def_prop_ro(
            "is_minus",
            [](const StinkyRegister& r) -> bool {
                return r.dataType == StinkyRegister::Type::Register && r.reg.isMinus;
            },
            "Whether the `-` (negate) modifier is set (`-v0`).")
        .def(
            "set_minus",
            [](StinkyRegister& r, bool value) {
                if (r.dataType == StinkyRegister::Type::Register) {
                    r.reg.isMinus = value ? 1u : 0u;
                }
            },
            nb::arg("value"), "Toggle the `-` modifier. No-op on literals.")
        .def_prop_ro(
            "is_abs",
            [](const StinkyRegister& r) -> bool {
                return r.dataType == StinkyRegister::Type::Register && r.reg.isAbs;
            },
            "Whether the `abs(...)` modifier is set (`abs(v0)`).")
        .def(
            "set_abs",
            [](StinkyRegister& r, bool value) {
                if (r.dataType == StinkyRegister::Type::Register) {
                    r.reg.isAbs = value ? 1u : 0u;
                }
            },
            nb::arg("value"), "Toggle the `abs(...)` modifier. No-op on literals.")

        // --- MSB offset (VGPR bank selector for idx >= 256) -----------------
        .def_prop_ro(
            "offset",
            [](const StinkyRegister& r) -> int {
                return r.dataType == StinkyRegister::Type::Register ? r.reg.offset : 0;
            },
            "The register offset (e.g. -512 for VGPR MSB bank 2).")
        .def(
            "set_offset",
            [](StinkyRegister& r, int value) {
                if (r.dataType == StinkyRegister::Type::Register) {
                    r.reg.offset = static_cast<int16_t>(value);
                }
            },
            nb::arg("value"),
            "Set the register offset for MSB encoding (e.g. -512 = bank 2). No-op on literals.")

        // --- Hash / equality (KernelWriter uses Registers as dict keys) ----
        .def("__hash__",
             [](const StinkyRegister& r) -> Py_hash_t { return static_cast<Py_hash_t>(r.hash()); })
        .def(
            "__eq__",
            [](const StinkyRegister& r, const nb::object& other) -> bool {
                if (!nb::isinstance<StinkyRegister>(other)) {
                    return false;
                }
                return r == nb::cast<StinkyRegister>(other);
            },
            nb::arg("other").none(true))
        .def(
            "__ne__",
            [](const StinkyRegister& r, const nb::object& other) -> bool {
                if (!nb::isinstance<StinkyRegister>(other)) {
                    return true;
                }
                return r != nb::cast<StinkyRegister>(other);
            },
            nb::arg("other").none(true))

        // --- Copy semantics (KernelWriter does copy.deepcopy on registers) -
        // StinkyRegister is value-like (trivially copyable for the union; the
        // std::string literalValue copies fine). Same impl for both shallow
        // and deep copy because there are no nested references.
        .def("__copy__", [](const StinkyRegister& r) { return StinkyRegister(r); })
        .def(
            "__deepcopy__",
            [](const StinkyRegister& r, nb::handle /*memo*/) { return StinkyRegister(r); },
            nb::arg("memo"))

        // --- Debug repr -----------------------------------------------------
        .def("__repr__", [](const StinkyRegister& r) -> std::string {
            if (r.dataType == StinkyRegister::Type::Register) {
                std::string typeStr;
                switch (r.reg.type) {
                    case RegType::V:
                        typeStr = "v";
                        break;
                    case RegType::S:
                        typeStr = "s";
                        break;
                    case RegType::A:
                        typeStr = "a";
                        break;
                    default:
                        typeStr = "?";
                        break;
                }
                std::string body;
                if (r.reg.num == 1) {
                    body = typeStr + std::to_string(r.reg.idx);
                } else {
                    body = typeStr + "[" + std::to_string(r.reg.idx) + ":" +
                           std::to_string(r.reg.idx + r.reg.num - 1) + "]";
                }
                std::string mods;
                if (r.reg.isMinus) mods += " -";
                if (r.reg.isAbs) mods += " abs";
                std::string name;
                if (!r.literalValue.empty()) {
                    name = " name=" + r.literalValue;
                }
                return "<Register " + body + mods + name + ">";
            } else if (r.dataType == StinkyRegister::Type::LiteralInt) {
                return "<Literal " + std::to_string(r.literalInt) + ">";
            } else if (r.dataType == StinkyRegister::Type::LiteralDouble) {
                return "<Literal " + std::to_string(r.literalDouble) + ">";
            } else if (r.dataType == StinkyRegister::Type::LiteralString) {
                return "<LiteralString \"" + r.literalValue + "\">";
            }
            return std::string("<Register (invalid)>");
        });

    // ========================================================================
    // Register Helper Functions (rocisa-style API)
    // ========================================================================
    m.def(
        "vgpr", [](int index, int count) { return StinkyRegister("v", index, count); },
        nb::arg("index"), nb::arg("count") = 1, "Create a VGPR register");

    m.def(
        "sgpr", [](int index, int count) { return StinkyRegister("s", index, count); },
        nb::arg("index"), nb::arg("count") = 1, "Create an SGPR register");

    m.def(
        "agpr", [](int index, int count) { return StinkyRegister("a", index, count); },
        nb::arg("index"), nb::arg("count") = 1, "Create an AGPR (Accumulator) register");

    m.def(
        "accvgpr", [](int index, int count) { return StinkyRegister("a", index, count); },
        nb::arg("index"), nb::arg("count") = 1,
        "Create an accumulator VGPR register (alias for agpr)");

    m.def(
        "mgpr", [](int index, int count) { return StinkyRegister("m", index, count); },
        nb::arg("index"), nb::arg("count") = 1, "Create an MGPR (Memory descriptor) register");

    m.def(
        "literal", [](float value) { return StinkyRegister(value); }, nb::arg("value"),
        "Create a float literal");

    // ========================================================================
    // Architecture IDs
    // ========================================================================
    // Only the architectures this library was built for exist in GfxArchID, so the enum is
    // exposed from the same list rather than hand-maintained.
    auto gfxArch = nb::enum_<GfxArchID>(m, "GfxArch");
#define STINKYTOFU_ARCH(archName) gfxArch.value(#archName, GfxArchID::archName);
#include "Config/Archs.def"

    // ========================================================================
    // Toolchain capability probing (via comgr)
    // ========================================================================
    m.def("hasComgrSupport", &hasComgrSupport,
          "Return True if this build was compiled with comgr support");

    m.def(
        "probeToolchainCaps",
        [](std::array<int, 3> arch) {
            auto* info = ArchHelper::getInstance().getArchInfo(arch[0], arch[1], arch[2]);
            if (!info)
                throw nb::value_error(("Unsupported architecture: gfx" + std::to_string(arch[0]) +
                                       std::to_string(arch[1]) + std::to_string(arch[2]))
                                          .c_str());
            GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
            AsmCapsConfig caps = ToolchainCaps::probe(archId);
            nb::dict result;
            result["VgprMsbMode"] = static_cast<int>(caps.vgprMsbMode);
            return result;
        },
        nb::arg("arch"),
        "Probe toolchain capabilities for [major, minor, stepping]. Results are cached.");

    m.def(
        "tryAssemble",
        [](const std::string& asmString, std::array<int, 3> arch) {
            auto* info = ArchHelper::getInstance().getArchInfo(arch[0], arch[1], arch[2]);
            if (!info)
                throw nb::value_error(("Unsupported architecture: gfx" + std::to_string(arch[0]) +
                                       std::to_string(arch[1]) + std::to_string(arch[2]))
                                          .c_str());
            // Build ISA name: amdgcn-amd-amdhsa--gfxMAJORMINORSTEPPING
            static constexpr char kHex[] = "0123456789abcdef";
            std::string isaName = "amdgcn-amd-amdhsa--gfx";
            isaName += std::to_string(info->major);
            isaName += std::to_string(info->minor);
            isaName += kHex[info->stepping & 0xF];
            return tryAssembleWithComgr(asmString, isaName, info->waveFrontSize);
        },
        nb::arg("asm_string"), nb::arg("arch"),
        "Try to assemble the given string for [major, minor, stepping]. Returns True if assembly "
        "succeeds.");

    nb::enum_<VgprMsbMode>(m, "VgprMsbMode")
        .value("NONE", VgprMsbMode::None)
        .value("MSB8", VgprMsbMode::Msb8)
        .value("MSB16", VgprMsbMode::Msb16);

    nb::enum_<MUBUFScope>(m, "MUBUFScope")
        .value("SCOPE_NONE", MUBUFScope::SCOPE_NONE)
        .value("SCOPE_CU", MUBUFScope::SCOPE_CU)
        .value("SCOPE_SE", MUBUFScope::SCOPE_SE)
        .value("SCOPE_DEV", MUBUFScope::SCOPE_DEV)
        .value("SCOPE_SYS", MUBUFScope::SCOPE_SYS);

    // ========================================================================
    // PyLogicalModule - Python-Specific High-Level IR Container
    // ========================================================================
    // Note: Exposed as "LogicalModule" in Python for backward compatibility
    nb::class_<PyLogicalModule>(m, "LogicalModule")
        .def(nb::init<const std::string&>(), nb::arg("name"),
             "Create a new IR module with the given kernel name")
        .def("add", &PyLogicalModule::add, nb::arg("instruction"),
             "Add a high-level IR instruction to the module (shared ownership)")
        .def("add_set_directive", &PyLogicalModule::addSetDirective, nb::arg("symbol"),
             nb::arg("value"),
             "Record a .set directive at the current position in the instruction stream")
        .def("add_label", &PyLogicalModule::addLabel, nb::arg("label_name"),
             nb::arg("alignment") = 1, nb::arg("comment") = "",
             "Record a label at the current position in the instruction stream")
        .def("add_textblock", &PyLogicalModule::addTextBlock, nb::arg("text"),
             "Record a textblock (comment/raw text) at the current position")
        .def("begin_group", &PyLogicalModule::beginGroup, nb::arg("name"),
             "Mark the beginning of a named instruction-group scope")
        .def("end_group", &PyLogicalModule::endGroup, nb::arg("name"),
             "Mark the end of a named instruction-group scope")
        .def("getName", &PyLogicalModule::getName, "Get the kernel name")
        .def(
            "dump",
            [](const PyLogicalModule& module) {
                std::ostringstream oss;
                module.dump(oss);
                return oss.str();
            },
            "Dump the IR module to a string");

    // ========================================================================
    // LogicalInstruction Base Class (polymorphic with virtual functions)
    // ========================================================================
    // Note: No trampoline needed. Trampolines are for Python subclasses that
    // override virtual methods. We only create concrete C++ classes (VAddF32, etc.)
    // that already implement all virtual methods, so no trampoline is needed.
    // Rocisa uses trampolines because it allows Python subclassing of instructions.
    nb::class_<LogicalInstruction>(m, "LogicalInstruction")
        .def_rw("comment", &LogicalInstruction::comment, "Optional comment")
        .def("get_logical_name", &LogicalInstruction::getLogicalName,
             "Get the logical name of this instruction")
        .def("is_composite", &LogicalInstruction::isComposite,
             "Check if this is a composite instruction")
        .def(
            "dump",
            [](const LogicalInstruction& inst) {
                std::ostringstream oss;
                inst.dump(oss);
                return oss.str();
            },
            "Dump the instruction to a string")
        .def(
            "set_ds",
            [](LogicalInstruction& inst, int na, int offset, int offset0, int offset1, bool gds) {
                inst.ds = DSModifiers(na, offset, offset0, offset1, gds);
            },
            nb::arg("na") = 1, nb::arg("offset") = 0, nb::arg("offset0") = 0,
            nb::arg("offset1") = 0, nb::arg("gds") = false, "Set DS (LDS/GDS) modifiers")
        .def(
            "set_mubuf",
            [](LogicalInstruction& inst, bool offen, int offset, bool glc, bool slc, bool nt,
               int scope, int th, bool isStore) {
                MUBUFScope mScope = static_cast<MUBUFScope>(scope);
                TemporalHint mTh = static_cast<TemporalHint>(th);
                inst.mubuf = MUBUFModifiers(
                    offen, offset, glc, slc, nt, /*lds=*/false, isStore, /*hasMUBUFConst=*/false,
                    /*hasGLCModifier=*/false, /*hasSC0Modifier=*/false, mScope, mTh);
            },
            nb::arg("offen") = false, nb::arg("offset") = 0, nb::arg("glc") = false,
            nb::arg("slc") = false, nb::arg("nt") = false, nb::arg("scope") = 0, nb::arg("th") = -1,
            nb::arg("is_store") = false,
            "Set MUBUF modifiers (offen, offset, glc, slc, nt, scope, th, is_store)")
        .def(
            "add_src",
            [](LogicalInstruction& inst, const StinkyRegister& reg) { inst.srcs.push_back(reg); },
            nb::arg("reg"), "Add an additional source register operand")
        .def(
            "set_vop3",
            [](LogicalInstruction& inst, const std::vector<int>& op_sel,
               const std::vector<int>& op_sel_hi, const std::vector<int>& byte_sel) {
                inst.vop3 = VOP3PModifiers(op_sel, op_sel_hi, byte_sel);
            },
            nb::arg("op_sel") = std::vector<int>{}, nb::arg("op_sel_hi") = std::vector<int>{},
            nb::arg("byte_sel") = std::vector<int>{},
            "Set VOP3P (op_sel/op_sel_hi/byte_sel) modifiers");

    // ========================================================================
    // Auto-generated Python bindings for all IR instructions (~273 classes)
    // ========================================================================
#include "PythonBindings_generated.inc"

    // ========================================================================
    // Special Instruction Classes (manually defined)
    // ========================================================================

    // MFMA - Matrix Fused Multiply-Add
    m.def(
        "MFMA",
        [](const std::string& instType, const std::string& accType, int m, int n, int k, int blocks,
           bool mfma1k, const StinkyRegister& acc, const StinkyRegister& a, const StinkyRegister& b,
           std::optional<StinkyRegister> acc2, bool neg, const std::string& matrixAFmt,
           const std::string& matrixBFmt, bool scaled, bool scaleOperands,
           const std::string& comment) {
            return makeLogicalInstructionShared(MFMA(
                instType, accType, m, n, k, blocks, mfma1k, acc, a, b, acc2 ? &(*acc2) : nullptr,
                neg, matrixAFmt, matrixBFmt, scaled, scaleOperands, comment));
        },
        nb::arg("instType"), nb::arg("accType"), nb::arg("m"), nb::arg("n"), nb::arg("k"),
        nb::arg("blocks"), nb::arg("mfma1k"), nb::arg("acc"), nb::arg("a"), nb::arg("b"),
        nb::arg("acc2") = std::nullopt, nb::arg("neg") = false, nb::arg("matrixAFmt") = "",
        nb::arg("matrixBFmt") = "", nb::arg("scaled") = false, nb::arg("scaleOperands") = false,
        nb::arg("comment") = "", "Create an MFMA instruction");

    // MXMFMA - Mixed-precision Matrix Fused Multiply-Add
    m.def(
        "MXMFMA",
        [](const std::string& instType, const std::string& accType,
           const std::string& mxScaleATypeStr, const std::string& mxScaleBTypeStr, int m, int n,
           int k, int block, const StinkyRegister& acc, const StinkyRegister& a,
           const StinkyRegister& b, const StinkyRegister& acc2, const StinkyRegister& mxsa,
           const StinkyRegister& mxsb, bool reuseA, bool reuseB, const std::string& matrixAFmt,
           const std::string& matrixBFmt, const std::string& comment) {
            return makeLogicalInstructionShared(
                MXMFMA(instType, accType, mxScaleATypeStr, mxScaleBTypeStr, m, n, k, block, acc, a,
                       b, acc2, mxsa, mxsb, reuseA, reuseB, matrixAFmt, matrixBFmt, comment));
        },
        nb::arg("instType"), nb::arg("accType"), nb::arg("mxScaleATypeStr"),
        nb::arg("mxScaleBTypeStr"), nb::arg("m"), nb::arg("n"), nb::arg("k"), nb::arg("block"),
        nb::arg("acc"), nb::arg("a"), nb::arg("b"), nb::arg("acc2"), nb::arg("mxsa"),
        nb::arg("mxsb"), nb::arg("reuseA") = false, nb::arg("reuseB") = false,
        nb::arg("matrixAFmt") = "", nb::arg("matrixBFmt") = "", nb::arg("comment") = "",
        "Create an MXMFMA instruction");

    // SMFMA - Sparse Matrix Fused Multiply-Add
    m.def(
        "SMFMA",
        [](const std::string& instType, const std::string& accType, int m, int n, int k, int blocks,
           bool mfma1k, const StinkyRegister& acc, const StinkyRegister& a, const StinkyRegister& b,
           const StinkyRegister& metadata, bool neg, const std::string& comment) {
            return makeLogicalInstructionShared(SMFMA(instType, accType, m, n, k, blocks, mfma1k,
                                                      acc, a, b, metadata, neg, comment));
        },
        nb::arg("instType"), nb::arg("accType"), nb::arg("m"), nb::arg("n"), nb::arg("k"),
        nb::arg("blocks"), nb::arg("mfma1k"), nb::arg("acc"), nb::arg("a"), nb::arg("b"),
        nb::arg("metadata"), nb::arg("neg") = false, nb::arg("comment") = "",
        "Create an SMFMA instruction");

    // SWaitAlu - Dependency counter wait instruction
    m.def(
        "SWaitAlu",
        [](int va_vdst, int va_sdst, int va_ssrc, int hold_cnt, int vm_vsrc, int va_vcc,
           int sa_sdst, const std::string& comment) {
            auto* inst = IRBase::createIR<LogicalInstruction>(logical::SWaitAlu);
            auto* data = new SWaitAluLogicalData(va_vdst, va_sdst, va_ssrc, hold_cnt, vm_vsrc,
                                                 va_vcc, sa_sdst);
            inst->setSpecialData(data);
            inst->comment = comment;
            return makeLogicalInstructionShared(inst);
        },
        nb::arg("va_vdst") = -1, nb::arg("va_sdst") = -1, nb::arg("va_ssrc") = -1,
        nb::arg("hold_cnt") = -1, nb::arg("vm_vsrc") = -1, nb::arg("va_vcc") = -1,
        nb::arg("sa_sdst") = -1, nb::arg("comment") = "", "Create an SWaitAlu instruction");

    // SchedulingFence - Scheduling barrier pseudo-instruction
    m.def(
        "SchedulingFence",
        [](const std::string& comment) {
            auto* inst = IRBase::createIR<LogicalInstruction>(logical::SchedulingFence);
            inst->comment = comment;
            return makeLogicalInstructionShared(inst);
        },
        nb::arg("comment") = "", "Create a SchedulingFence pseudo-instruction");

    // TensorLoadToLds - Higher-level tensor load operation
    m.def(
        "TensorLoadToLds",
        [](const StinkyRegister& group0, const StinkyRegister& group1,
           std::optional<StinkyRegister> group2, std::optional<StinkyRegister> group3,
           const std::string& comment) {
            return makeLogicalInstructionShared(
                TensorLoadToLds(group0, group1, group2 ? &(*group2) : nullptr,
                                group3 ? &(*group3) : nullptr, comment));
        },
        nb::arg("group0"), nb::arg("group1"), nb::arg("group2") = std::nullopt,
        nb::arg("group3") = std::nullopt, nb::arg("comment") = "",
        "Create a TensorLoadToLds instruction");

    // Label - Control flow label
    m.def(
        "Label",
        [](const std::string& labelName) { return makeLogicalInstructionShared(Label(labelName)); },
        nb::arg("labelName"), "Create a Label");

    // ========================================================================
    // IntrinsicCall - Placeholder for intrinsic function calls
    // ========================================================================
    nb::class_<IntrinsicCall, LogicalInstruction>(m, "IntrinsicCall")
        .def("get_function_name", &IntrinsicCall::getFunctionName, "Get intrinsic name")
        .def_rw("function_name", &IntrinsicCall::functionName, "Intrinsic function name");

    // ========================================================================
    // IntrinsicArgument - Intrinsic argument metadata
    // ========================================================================
    nb::class_<IntrinsicArgument>(m, "IntrinsicArgument")
        .def_ro("name", &IntrinsicArgument::name, "Argument name")
        .def_ro("reg_type", &IntrinsicArgument::regType, "Register type (vgpr, sgpr, etc.)");

    // ========================================================================
    // IntrinsicLibrary - Runtime library for intrinsic definitions
    // ========================================================================
    nb::class_<IntrinsicLibrary>(m, "IntrinsicLibrary")
        .def_static("load", &IntrinsicLibrary::loadFromFile, nb::arg("path"),
                    "Load intrinsic library from .st.bc file")
        .def("has_intrinsic", &IntrinsicLibrary::hasIntrinsic, nb::arg("name"),
             "Check if intrinsic exists")
        .def("get_intrinsic_names", &IntrinsicLibrary::getIntrinsicNames,
             "Get list of all intrinsic names")
        .def("get_arguments", &IntrinsicLibrary::getArguments, nb::arg("name"),
             "Get arguments for an intrinsic")
        .def("get_comment", &IntrinsicLibrary::getComment, nb::arg("name"),
             "Get comment for an intrinsic")
        .def("size", &IntrinsicLibrary::size, "Get number of intrinsics");

    // ========================================================================
    // IntrinsicRegistry - Singleton with automatic loading
    // ========================================================================
    nb::class_<IntrinsicRegistry>(m, "IntrinsicRegistry")
        .def_static("instance", &IntrinsicRegistry::instance, nb::rv_policy::reference,
                    "Get singleton instance (auto-loads intrinsics.st.bc)")
        .def("is_initialized", &IntrinsicRegistry::isInitialized,
             "Check if intrinsics were loaded successfully")
        .def("has_intrinsic", &IntrinsicRegistry::hasIntrinsic, nb::arg("name"),
             "Check if intrinsic exists")
        .def("get_intrinsic_names", &IntrinsicRegistry::getIntrinsicNames,
             "Get list of all intrinsic names")
        .def("get_loaded_path", &IntrinsicRegistry::getLoadedPath,
             "Get path where intrinsics.st.bc was loaded from")
        .def("get_library", &IntrinsicRegistry::getLibrary, "Get the underlying IntrinsicLibrary")
        .def("reload", &IntrinsicRegistry::reload, nb::arg("path"),
             "Reload intrinsics from a specific path");

    // Intrinsic helper function - cleaner API with kwargs
    // NOTE: This is a low-level function. Use the Python wrapper in __init__.py
    // which provides validation and automatic argument reordering.
    m.def(
        "Intrinsic",
        [](const std::string& name, const nb::kwargs& kwargs) {
            // Convert kwargs to ordered vector of registers
            // Note: Python dicts maintain insertion order (Python 3.7+)
            std::vector<StinkyRegister> args;
            args.reserve(kwargs.size());

            for (auto item : kwargs) {
                auto value = item.second;

                // Try to cast to StinkyRegister first
                if (nb::isinstance<StinkyRegister>(value)) {
                    args.push_back(nb::cast<StinkyRegister>(value));
                }
                // Handle int literals
                else if (nb::isinstance<nb::int_>(value)) {
                    args.push_back(StinkyRegister(nb::cast<int>(value)));
                }
                // Handle float literals
                else if (nb::isinstance<nb::float_>(value)) {
                    args.push_back(StinkyRegister(nb::cast<double>(value)));
                }
                // Handle string literals
                else if (nb::isinstance<nb::str>(value)) {
                    args.push_back(StinkyRegister(nb::cast<std::string>(value)));
                } else {
                    throw std::runtime_error(
                        "Intrinsic argument must be Register, int, float, or string");
                }
            }

            return makeLogicalInstructionShared(IRBase::createIR<IntrinsicCall>(name, args));
        },
        "Create an intrinsic call with named arguments\n\n"
        "Example:\n"
        "    module.add(st.Intrinsic('ReluF32', dest=st.vgpr(0), src=st.vgpr(1), "
        "temp=st.vgpr(2)))\n"
        "    module.add(st.Intrinsic('ClampF32', dest=v0, src=v1, min_val=0.0, max_val=1.0, "
        "temp=v2))\n\n"
        "Arguments can be:\n"
        "    - Register objects (st.vgpr(n), st.sgpr(n), etc.)\n"
        "    - int literals (0, 1, 255, etc.)\n"
        "    - float literals (0.0, 1.0, 3.14, etc.)\n"
        "    - string literals (for special values)\n\n"
        "The intrinsic will be expanded during optimization by IntrinsicExpansionPass.");

    // ========================================================================
    // SRD Upper Value (rocisa.code.SrdUpperValue replacement)
    // ========================================================================
    // Mirrors rocisa::SrdUpperValue (rocisa/src/code.cpp:56-82). Tensile
    // calls SrdUpperValue(IsaVersion) -> BitfieldUnion and immediately
    // reads .desc() / .getValue() to embed the SRD upper 32 bits as a
    // packed literal in the kernel signature (KernelWriterAssembly.py:1497).
    //
    // ISA dispatch lives in createSrdUpperValue(); Python receives a thin
    // BitfieldUnion handle with the rocisa-shaped surface API.
    nb::class_<BitfieldUnion>(m, "BitfieldUnion")
        .def("__str__", &BitfieldUnion::toString)
        .def("getValue", &BitfieldUnion::getValue)
        .def("toString", &BitfieldUnion::toString)
        .def("desc", &BitfieldUnion::desc);

    m.def(
        "SrdUpperValue",
        [](const nb::object& isa) {
            std::array<int, 3> version{};
            if (nb::isinstance<nb::tuple>(isa) || nb::isinstance<nb::list>(isa)) {
                version = {nb::cast<int>(isa[0]), nb::cast<int>(isa[1]), nb::cast<int>(isa[2])};
            } else {
                version = nb::cast<std::array<int, 3>>(isa);
            }
            return createSrdUpperValue(version);
        },
        nb::arg("isa"),
        "Create the SRD upper-32-bit literal for the given ISA "
        "(major, minor, stepping) tuple or 3-element array.");

    // ========================================================================
    // Architecture support query
    // ========================================================================
    m.def(
        "isSupportedByStinkyTofu",
        [](std::array<int, 3> arch) { return BackendRegistry::getArchPipeline(arch) != nullptr; },
        nb::arg("arch"),
        "Return True if the given architecture [major, minor, stepping] has a StinkyTofu backend");
    m.def("getRegisteredArchKeys", &BackendRegistry::getRegisteredArchKeys,
          "Return a list of arch name strings for all registered StinkyTofu backends (e.g. "
          "[\"gfx1250\"]).");
    m.def(
        "isMnemonicSupported",
        [](const std::string& mnemonic, std::array<int, 3> arch) {
            const auto* info = ArchHelper::getInstance().getArchInfo(arch[0], arch[1], arch[2]);
            if (!info) return false;
            const auto& map = info->getMnemonicToIsaOpcodeMap();
            return map.find(mnemonic) != map.end();
        },
        nb::arg("mnemonic"), nb::arg("arch"),
        "Return True if StinkyTofu has a hardware instruction definition for *mnemonic* on "
        "[major, minor, stepping]. Lets a generator ask before emitting, instead of finding "
        "out when the mnemonic fails to lower.");

    // ========================================================================
    // Hardware capability dictionaries (replaces rocisa getAsmCaps/etc.)
    // ========================================================================
    m.def(
        "getHardwareCaps",
        [](std::array<int, 3> arch) {
            auto caps = HardwareCaps::query(arch[0], arch[1], arch[2]);

            nb::dict asmCaps, archCaps, regCaps, asmBugs;
            for (auto& [k, v] : caps.asmCaps) asmCaps[k.c_str()] = v;
            for (auto& [k, v] : caps.archCaps) archCaps[k.c_str()] = v;
            for (auto& [k, v] : caps.regCaps) regCaps[k.c_str()] = v;
            for (auto& [k, v] : caps.asmBugs) asmBugs[k.c_str()] = v;

            nb::dict result;
            result["asmCaps"] = asmCaps;
            result["archCaps"] = archCaps;
            result["regCaps"] = regCaps;
            result["asmBugs"] = asmBugs;
            return result;
        },
        nb::arg("arch"),
        "Return hardware capability dicts for [major, minor, stepping].\n\n"
        "Returns a dict with keys 'asmCaps', 'archCaps', 'regCaps', 'asmBugs',\n"
        "matching the rocisa getAsmCaps()/getArchCaps()/getRegCaps()/getAsmBugs() API.");

    // ========================================================================
    // Logical Instruction Counting (ported from rocisa)
    // ========================================================================
    init_logical_count(m);
}
