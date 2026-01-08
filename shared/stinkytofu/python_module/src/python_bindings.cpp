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

#include "ir/IRModule.hpp"
#include "ir/StinkyInstructions.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/passes/CompositeInstructionLoweringPass.hpp"
#include "ir/passes/PassManager.hpp"
#include "ir/passes/ToStinkyAsmPass.hpp"
#include "isa/gfx/GfxIsa.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>

namespace nb = nanobind;
using namespace stinkytofu;

NB_MODULE(stinkytofu, m)
{
    m.doc() = "StinkyTofu: High-Level IR for AMDGPU Assembly Generation";

    // ========================================================================
    // Register Types
    // ========================================================================
    nb::enum_<RegType>(m, "RegType", "Register type enumeration")
        .value("V", RegType::V, "Vector Register (VGPR)")
        .value("S", RegType::S, "Scalar Register (SGPR)")
        .value("A", RegType::A, "Accumulator Register (AGPR)")
        .value("ACC", RegType::ACC, "Accumulator Register (alternative)")
        .value("AGPR", RegType::AGPR, "Accumulator GPR")
        .value("VCC", RegType::VCC, "Vector Condition Code")
        .value("VCC_LO", RegType::VCC_LO, "Vector Condition Code (low)")
        .value("VCC_HI", RegType::VCC_HI, "Vector Condition Code (high)")
        .value("EXEC", RegType::EXEC, "Execution Mask")
        .value("EXEC_LO", RegType::EXEC_LO, "Execution Mask (low)")
        .value("EXEC_HI", RegType::EXEC_HI, "Execution Mask (high)")
        .value("SCC", RegType::SCC, "Scalar Condition Code")
        .value("UNKNOWN", RegType::UNKNOWN, "Unknown Register Type");

    // ========================================================================
    // StinkyRegister Class
    // ========================================================================
    // Note: Unlike rocisa which uses strings, stinkytofu uses enum (RegType) internally.
    // We need to expose StinkyRegister so nanobind can convert it properly.
    // However, users typically don't construct Register directly - they use helper functions.
    nb::class_<StinkyRegister>(m, "Register")
        .def(nb::init<>(), "Create a null register")
        .def(nb::init<const std::string&, int, int>(),
             nb::arg("type"),
             nb::arg("index"),
             nb::arg("count") = 1,
             "Create a register (e.g., Register('v', 0, 1) for v0)")
        .def(nb::init<float>(), nb::arg("value"), "Create a float literal")
        .def(nb::init<int>(), nb::arg("value"), "Create an int literal")
        .def_prop_ro(
            "reg_type",
            [](const StinkyRegister& r) -> RegType {
                if(r.dataType == StinkyRegister::Type::Register)
                {
                    return r.reg.type;
                }
                return RegType::UNKNOWN;
            },
            "Get the register type (V, S, A, etc.)")
        .def_prop_ro(
            "index",
            [](const StinkyRegister& r) -> int {
                if(r.dataType == StinkyRegister::Type::Register)
                {
                    return r.reg.idx;
                }
                return -1;
            },
            "Get the register index")
        .def_prop_ro(
            "count",
            [](const StinkyRegister& r) -> int {
                if(r.dataType == StinkyRegister::Type::Register)
                {
                    return r.reg.num;
                }
                return 0;
            },
            "Get the register count (number of consecutive registers)")
        .def_prop_ro(
            "is_literal",
            [](const StinkyRegister& r) -> bool {
                return r.dataType == StinkyRegister::Type::LiteralInt
                       || r.dataType == StinkyRegister::Type::LiteralDouble;
            },
            "Check if this is a literal value")
        .def("__repr__", [](const StinkyRegister& r) -> std::string {
            if(r.dataType == StinkyRegister::Type::Register)
            {
                std::string typeStr;
                switch(r.reg.type)
                {
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
                if(r.reg.num == 1)
                {
                    return "<Register " + typeStr + std::to_string(r.reg.idx) + ">";
                }
                else
                {
                    return "<Register " + typeStr + "[" + std::to_string(r.reg.idx) + ":"
                           + std::to_string(r.reg.idx + r.reg.num - 1) + "]>";
                }
            }
            else if(r.dataType == StinkyRegister::Type::LiteralInt)
            {
                return "<Literal " + std::to_string(r.literalInt) + ">";
            }
            else if(r.dataType == StinkyRegister::Type::LiteralDouble)
            {
                return "<Literal " + std::to_string(r.literalDouble) + ">";
            }
            return std::string("<Register (invalid)>");
        });

    // ========================================================================
    // Register Helper Functions (rocisa-style API)
    // ========================================================================
    m.def(
        "vgpr",
        [](int index, int count) { return StinkyRegister("v", index, count); },
        nb::arg("index"),
        nb::arg("count") = 1,
        "Create a VGPR register");

    m.def(
        "sgpr",
        [](int index, int count) { return StinkyRegister("s", index, count); },
        nb::arg("index"),
        nb::arg("count") = 1,
        "Create an SGPR register");

    m.def(
        "agpr",
        [](int index, int count) { return StinkyRegister("a", index, count); },
        nb::arg("index"),
        nb::arg("count") = 1,
        "Create an AGPR (Accumulator) register");

    m.def(
        "accvgpr",
        [](int index, int count) { return StinkyRegister("a", index, count); },
        nb::arg("index"),
        nb::arg("count") = 1,
        "Create an accumulator VGPR register (alias for agpr)");

    m.def(
        "literal",
        [](float value) { return StinkyRegister(value); },
        nb::arg("value"),
        "Create a float literal");

    // ========================================================================
    // Architecture IDs
    // ========================================================================
    nb::enum_<GfxArchID>(m, "GfxArch")
        .value("Gfx942", GfxArchID::Gfx942, "GFX9.4.2 (MI210/MI250)")
        .value("Gfx950", GfxArchID::Gfx950, "GFX9.5.0 (MI300)")
        .value("Gfx1250", GfxArchID::Gfx1250, "GFX12.5.0 (RDNA4)");

    // ========================================================================
    // IRModule - High-Level IR Container
    // ========================================================================
    nb::class_<IRModule>(m, "IRModule")
        .def(nb::init<const std::string&>(),
             nb::arg("name"),
             "Create a new IR module with the given kernel name")
        .def("add",
             &IRModule::add,
             nb::arg("instruction"),
             "Add a high-level IR instruction to the module")
        .def("getName", &IRModule::getName, "Get the kernel name")
        .def(
            "dump",
            [](const IRModule& module) {
                std::ostringstream oss;
                module.dump(oss);
                return oss.str();
            },
            "Dump the IR module to a string");

    // ========================================================================
    // IRInstruction Base Class
    // ========================================================================
    nb::class_<IRInstruction>(m, "IRInstruction")
        .def_rw("comment", &IRInstruction::comment, "Optional comment")
        .def("get_logical_name",
             &IRInstruction::getLogicalName,
             "Get the logical name of this instruction")
        .def("is_composite",
             &IRInstruction::isComposite,
             "Check if this is a composite instruction");

    // ========================================================================
    // TODO: Auto-generate Python bindings for all IR instructions
    // ========================================================================
    //
    // For now, we'll manually expose a few key instructions as examples.
    // Future work: Generate these bindings automatically from TableGen,
    // similar to how we generate the C++ classes.
    //
    // Each instruction class needs:
    // 1. Constructor bindings with proper argument names
    // 2. Inheritance from IRInstruction
    // 3. Documentation strings
    //
    // Example instruction classes to expose:
    // - VAddF32, VAddF16, VAddU32, VAddI32
    // - VMulF32, VMulF16, VMulU32, VMulI32
    // - VFmaF32, VFmaF16
    // - DSLoadB32, DSLoadB64, DSStoreB32, DSStoreB64
    // - MFMA, MXMFMA, SMFMA
    // - TensorLoadToLds
    // - And ~273 more...
    //
    // ========================================================================

    // Example: VAddF32
    nb::class_<VAddF32, IRInstruction>(m, "VAddF32")
        .def(nb::init<const StinkyRegister&,
                      const StinkyRegister&,
                      const StinkyRegister&,
                      std::optional<DPPModifiers>,
                      std::optional<SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dest"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("dpp")     = std::nullopt,
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "",
             "32-bit floating point add: dest = src0 + src1");

    // Example: VMulF32
    nb::class_<VMulF32, IRInstruction>(m, "VMulF32")
        .def(nb::init<const StinkyRegister&,
                      const StinkyRegister&,
                      const StinkyRegister&,
                      std::optional<DPPModifiers>,
                      std::optional<SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dest"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("dpp")     = std::nullopt,
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "",
             "32-bit floating point multiply: dest = src0 * src1");

    // Example: VFmaF32
    nb::class_<VFmaF32, IRInstruction>(m, "VFmaF32")
        .def(nb::init<const StinkyRegister&,
                      const StinkyRegister&,
                      const StinkyRegister&,
                      const StinkyRegister&,
                      std::optional<DPPModifiers>,
                      std::optional<SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dest"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("dpp")     = std::nullopt,
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "",
             "32-bit floating point fused multiply-add: dest = src0 * src1 + src2");

    // Example: TensorLoadToLds
    nb::class_<TensorLoadToLds, IRInstruction>(m, "TensorLoadToLds")
        .def(nb::init<const StinkyRegister&,
                      const StinkyRegister&,
                      const StinkyRegister*,
                      const StinkyRegister*,
                      const std::string&>(),
             nb::arg("group0"),
             nb::arg("group1"),
             nb::arg("group2")  = nullptr,
             nb::arg("group3")  = nullptr,
             nb::arg("comment") = "",
             "Load tensor data to LDS (2-4 SGPR groups)");

    // ========================================================================
    // Pass Manager and Lowering
    // ========================================================================
    // TODO: Expose IRInstPassManager for lowering IR to assembly
    // TODO: Expose assembly emission (IRListModule::emitAssembly())
    //
    // Example future API:
    //   module = IRModule("kernel")
    //   module.add(VAddF32(vgpr(0), vgpr(1), vgpr(2)))
    //
    //   pm = PassManager(GfxArch.Gfx942)
    //   pm.add_pass(CompositeInstructionLoweringPass())
    //   pm.add_pass(ToStinkyAsmPass())
    //   asm_module = pm.run(module)
    //
    //   asm_code = asm_module.emit_assembly()
    // ========================================================================
}
