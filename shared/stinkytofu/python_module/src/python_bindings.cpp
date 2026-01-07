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

#include "ErrorHandling.hpp"
#include "StinkyBuilder.hpp"
#include "ir/StinkyIR.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyModifiers.hpp"
#include "ir/asm/StinkySignature.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace stinkytofu;

// ========================================================================
// Error Handling Helper - Convert Expected<T> to Python Exceptions
// ========================================================================

/// @brief Unwrap an Expected<T> or throw a Python exception if it contains an error
/// @tparam T The success type
/// @param result The Expected<T> to unwrap
/// @return The unwrapped value
/// @throws std::runtime_error (converted to Python RuntimeError) if Expected contains an error
template <typename T>
T unwrapExpected(Expected<T> result)
{
    if(!result)
    {
        throw std::runtime_error(result.getError());
    }
    return std::move(*result);
}

// Test functions for Expected<T> error handling
namespace test
{
    Expected<int> testExpectedSuccess(int value)
    {
        return value * 2;
    }

    Expected<int> testExpectedError(const std::string& errorMsg)
    {
        return Expected<int>::Error(errorMsg);
    }

    Expected<std::vector<int>> testExpectedVectorSuccess()
    {
        return std::vector<int>{1, 2, 3, 4, 5};
    }

    Expected<std::vector<int>> testExpectedVectorError()
    {
        return Expected<std::vector<int>>::Error("Vector creation failed");
    }
} // namespace test

// ========================================================================
// Helper Functions for Reducing Binding Boilerplate
// ========================================================================

// Helper for binding functions with signature: (dst, src0, src1, comment) -> vector<Instruction*>
template <typename Class>
Class& bindDSS(Class&      cls,
               const char* name,
               std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                   const StinkyRegister&,
                                                                   const StinkyRegister&,
                                                                   const std::string&),
               const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("dst"),
            nb::arg("src0"),
            nb::arg("src1"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (dst, src0, src1, src2, comment) -> vector<Instruction*>
template <typename Class>
Class& bindDSSS(Class&      cls,
                const char* name,
                std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                    const StinkyRegister&,
                                                                    const StinkyRegister&,
                                                                    const StinkyRegister&,
                                                                    const std::string&),
                const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("dst"),
            nb::arg("src0"),
            nb::arg("src1"),
            nb::arg("src2"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (dst, src, comment) -> vector<Instruction*>
template <typename Class>
Class& bindDS(Class&      cls,
              const char* name,
              std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                  const StinkyRegister&,
                                                                  const std::string&),
              const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("dst"),
            nb::arg("src"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (src0, src1, comment) -> vector<Instruction*> (no dst, like compare)
template <typename Class>
Class& bindSS(Class&      cls,
              const char* name,
              std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                  const StinkyRegister&,
                                                                  const std::string&),
              const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("src0"),
            nb::arg("src1"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding load functions with signature: (dst, addr, comment) -> vector<Instruction*>
template <typename Class>
Class& bindDA(Class&      cls,
              const char* name,
              std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                  const StinkyRegister&,
                                                                  const std::string&),
              const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("dst"),
            nb::arg("addr"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding store functions with signature: (addr, src, comment) -> vector<Instruction*>
template <typename Class>
Class& bindAS(Class&      cls,
              const char* name,
              std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                  const StinkyRegister&,
                                                                  const std::string&),
              const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("addr"),
            nb::arg("src"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding branch functions with signature: (labelName, comment) -> vector<Instruction*>
template <typename Class>
Class& bindBranch(Class&      cls,
                  const char* name,
                  std::vector<StinkyInstruction*> (StinkyTofu::*func)(const std::string&,
                                                                      const std::string&),
                  const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("labelName"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (src, comment) -> vector<Instruction*>
template <typename Class>
Class& bindS(Class&      cls,
             const char* name,
             std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                 const std::string&),
             const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("src"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (src, int, comment) -> vector<Instruction*>
template <typename Class>
Class& bindSI(Class&      cls,
              const char* name,
              std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                  int,
                                                                  const std::string&),
              const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("src"),
            nb::arg("simm16"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (dst, addr0, addr1, comment) -> vector<Instruction*>
template <typename Class>
Class& bindDAA(Class&      cls,
               const char* name,
               std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                   const StinkyRegister&,
                                                                   const StinkyRegister&,
                                                                   const std::string&),
               const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("dst"),
            nb::arg("addr0"),
            nb::arg("addr1"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (addr0, addr1, src, comment) -> vector<Instruction*>
template <typename Class>
Class& bindAAS(Class&      cls,
               const char* name,
               std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                   const StinkyRegister&,
                                                                   const StinkyRegister&,
                                                                   const std::string&),
               const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("addr0"),
            nb::arg("addr1"),
            nb::arg("src"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (dst, comment) -> vector<Instruction*>
template <typename Class>
Class& bindD(Class&      cls,
             const char* name,
             std::vector<StinkyInstruction*> (StinkyTofu::*func)(const StinkyRegister&,
                                                                 const std::string&),
             const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("dst"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding SMEM load functions with signature: (dst, base, offset, comment) -> vector<Instruction*>
template <typename Class>
Class& bindSMemLoad(Class&      cls,
                    const char* name,
                    std::vector<StinkyInstruction*> (StinkyTofu::*func)(
                        const StinkyRegister&, const StinkyRegister&, int, const std::string&),
                    const char* doc = nullptr)
{
    cls.def(name,
            func,
            nb::arg("dst"),
            nb::arg("base"),
            nb::arg("offset")  = 0,
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            doc ? doc : "");
    return cls;
}

// Helper for binding functions with signature: (comment) -> vector<Instruction*>
template <typename Class>
Class& bindNoArg(Class&      cls,
                 const char* name,
                 std::vector<StinkyInstruction*> (StinkyTofu::*func)(const std::string&),
                 const char* doc = nullptr)
{
    cls.def(name, func, nb::arg("comment") = "", nb::rv_policy::reference, doc ? doc : "");
    return cls;
}

NB_MODULE(stinkytofu, m)
{
    m.doc() = "StinkyTofu: A standalone Python module for GPU assembly generation";

    // ========================================================================
    // Bind StinkyRegister
    // ========================================================================

    nb::class_<StinkyRegister>(m, "StinkyRegister")
        .def(nb::init<const std::string&, int, int>(),
             nb::arg("regType"),
             nb::arg("regIdx"),
             nb::arg("regNum") = 1,
             "Create a register with the given type, index, and count")
        .def_static("Virtual",
                    &StinkyRegister::Virtual,
                    nb::arg("idx"),
                    nb::arg("num") = 1,
                    "Create a virtual VGPR with the given index and count")
        .def_static("VirtualSGPR",
                    &StinkyRegister::VirtualSGPR,
                    nb::arg("idx"),
                    nb::arg("num") = 1,
                    "Create a virtual SGPR with the given index and count")
        .def("isVirtualRegister",
             &StinkyRegister::isVirtualRegister,
             "Check if this register is a virtual register")
        .def("withOffset",
             &StinkyRegister::withOffset,
             nb::arg("offset"),
             "Apply an offset to this virtual register's index and mark it as physical")
        .def_prop_ro(
            "regType",
            [](const StinkyRegister& r) { return regTypeToString(r.reg.type); },
            "Register type (e.g., 'v', 's', 'a')")
        .def_prop_ro(
            "regIdx", [](const StinkyRegister& r) { return r.reg.idx; }, "Register index")
        .def_prop_ro(
            "regNum",
            [](const StinkyRegister& r) { return r.reg.num; },
            "Number of consecutive registers")
        .def("__repr__", [](const StinkyRegister& r) {
            if(r.reg.num == 1)
            {
                return regTypeToString(r.reg.type) + std::to_string(r.reg.idx);
            }
            else
            {
                return regTypeToString(r.reg.type) + "[" + std::to_string(r.reg.idx) + ":"
                       + std::to_string(r.reg.idx + r.reg.num - 1) + "]";
            }
        });

    // ========================================================================
    // Bind Modifier Classes
    // ========================================================================

    nb::class_<DSModifiers>(m, "DSModifiers")
        .def(nb::init<int, int, int, int, bool>(),
             nb::arg("na")      = 1,
             nb::arg("offset")  = 0,
             nb::arg("offset0") = 0,
             nb::arg("offset1") = 0,
             nb::arg("gds")     = false,
             "DS (Data Share / LDS) instruction modifiers")
        .def_rw("na", &DSModifiers::na, "Non-aligned flag")
        .def_rw("offset", &DSModifiers::offset, "Offset for single-offset instructions")
        .def_rw("offset0", &DSModifiers::offset0, "First offset for dual-offset instructions")
        .def_rw("offset1", &DSModifiers::offset1, "Second offset for dual-offset instructions")
        .def_rw("gds", &DSModifiers::gds, "Global Data Share flag");

    nb::class_<MUBUFModifiers>(m, "MUBUFModifiers")
        .def(nb::init<bool, int, bool, bool, bool, bool, bool>(),
             nb::arg("offen")    = false,
             nb::arg("offset12") = 0,
             nb::arg("glc")      = false,
             nb::arg("slc")      = false,
             nb::arg("nt")       = false,
             nb::arg("lds")      = false,
             nb::arg("isStore")  = false,
             "MUBUF (Buffer) instruction modifiers")
        .def_prop_rw(
            "offen",
            [](const MUBUFModifiers& m) { return (bool)m.offen; },
            [](MUBUFModifiers& m, bool v) { m.offen = v; },
            "Offset enable flag")
        .def_rw("offset12", &MUBUFModifiers::offset12, "12-bit offset")
        .def_prop_rw(
            "glc",
            [](const MUBUFModifiers& m) { return (bool)m.glc; },
            [](MUBUFModifiers& m, bool v) { m.glc = v; },
            "Globally coherent flag")
        .def_prop_rw(
            "slc",
            [](const MUBUFModifiers& m) { return (bool)m.slc; },
            [](MUBUFModifiers& m, bool v) { m.slc = v; },
            "System level coherent flag")
        .def_prop_rw(
            "nt",
            [](const MUBUFModifiers& m) { return (bool)m.nt; },
            [](MUBUFModifiers& m, bool v) { m.nt = v; },
            "Non-temporal flag")
        .def_prop_rw(
            "lds",
            [](const MUBUFModifiers& m) { return (bool)m.lds; },
            [](MUBUFModifiers& m, bool v) { m.lds = v; },
            "Load to LDS flag")
        .def_prop_rw(
            "isStore",
            [](const MUBUFModifiers& m) { return (bool)m.isStore; },
            [](MUBUFModifiers& m, bool v) { m.isStore = v; },
            "Is store operation flag");

    nb::class_<FLATModifiers>(m, "FLATModifiers")
        .def(nb::init<int, bool, bool, bool, bool>(),
             nb::arg("offset12") = 0,
             nb::arg("glc")      = false,
             nb::arg("slc")      = false,
             nb::arg("lds")      = false,
             nb::arg("isStore")  = false,
             "FLAT (Flat memory) instruction modifiers")
        .def_rw("offset12", &FLATModifiers::offset12, "12-bit offset")
        .def_prop_rw(
            "glc",
            [](const FLATModifiers& m) { return (bool)m.glc; },
            [](FLATModifiers& m, bool v) { m.glc = v; },
            "Globally coherent flag")
        .def_prop_rw(
            "slc",
            [](const FLATModifiers& m) { return (bool)m.slc; },
            [](FLATModifiers& m, bool v) { m.slc = v; },
            "System level coherent flag")
        .def_prop_rw(
            "lds",
            [](const FLATModifiers& m) { return (bool)m.lds; },
            [](FLATModifiers& m, bool v) { m.lds = v; },
            "Load to LDS flag")
        .def_prop_rw(
            "isStore",
            [](const FLATModifiers& m) { return (bool)m.isStore; },
            [](FLATModifiers& m, bool v) { m.isStore = v; },
            "Is store operation flag");

    nb::class_<SMEMModifiers>(m, "SMEMModifiers")
        .def(nb::init<bool, bool, int>(),
             nb::arg("glc")    = false,
             nb::arg("nv")     = false,
             nb::arg("offset") = 0,
             "SMEM (Scalar memory) instruction modifiers")
        .def_prop_rw(
            "glc",
            [](const SMEMModifiers& m) { return (bool)m.glc; },
            [](SMEMModifiers& m, bool v) { m.glc = v; },
            "Globally coherent flag")
        .def_prop_rw(
            "nv",
            [](const SMEMModifiers& m) { return (bool)m.nv; },
            [](SMEMModifiers& m, bool v) { m.nv = v; },
            "Non-volatile flag")
        .def_rw("offset", &SMEMModifiers::offset, "Offset");

    nb::class_<GLOBALModifiers>(m, "GLOBALModifiers")
        .def(nb::init<int>(), nb::arg("offset") = 0, "GLOBAL memory instruction modifiers")
        .def_rw("offset", &GLOBALModifiers::offset, "Offset");

    // SDWA (Sub-DWord Addressing) Modifiers
    nb::enum_<SDWAModifiers::SelectBit>(m, "SDWASelectBit")
        .value("SEL_NONE", SDWAModifiers::SelectBit::SEL_NONE)
        .value("DWORD", SDWAModifiers::SelectBit::DWORD)
        .value("BYTE_0", SDWAModifiers::SelectBit::BYTE_0)
        .value("BYTE_1", SDWAModifiers::SelectBit::BYTE_1)
        .value("BYTE_2", SDWAModifiers::SelectBit::BYTE_2)
        .value("BYTE_3", SDWAModifiers::SelectBit::BYTE_3)
        .value("WORD_0", SDWAModifiers::SelectBit::WORD_0)
        .value("WORD_1", SDWAModifiers::SelectBit::WORD_1);

    nb::enum_<SDWAModifiers::UnusedBit>(m, "SDWAUnusedBit")
        .value("UNUSED_NONE", SDWAModifiers::UnusedBit::UNUSED_NONE)
        .value("UNUSED_PAD", SDWAModifiers::UnusedBit::UNUSED_PAD)
        .value("UNUSED_SEXT", SDWAModifiers::UnusedBit::UNUSED_SEXT)
        .value("UNUSED_PRESERVE", SDWAModifiers::UnusedBit::UNUSED_PRESERVE);

    nb::class_<SDWAModifiers>(m, "SDWAModifiers")
        .def(nb::init<SDWAModifiers::SelectBit,
                      SDWAModifiers::UnusedBit,
                      SDWAModifiers::SelectBit,
                      SDWAModifiers::SelectBit>(),
             nb::arg("dst_sel")    = SDWAModifiers::SelectBit::SEL_NONE,
             nb::arg("dst_unused") = SDWAModifiers::UnusedBit::UNUSED_NONE,
             nb::arg("src0_sel")   = SDWAModifiers::SelectBit::SEL_NONE,
             nb::arg("src1_sel")   = SDWAModifiers::SelectBit::SEL_NONE,
             "SDWA (Sub-DWord Addressing) instruction modifiers")
        .def_rw("dst_sel", &SDWAModifiers::dst_sel, "Destination select")
        .def_rw("dst_unused", &SDWAModifiers::dst_unused, "Destination unused bits handling")
        .def_rw("src0_sel", &SDWAModifiers::src0_sel, "Source 0 select")
        .def_rw("src1_sel", &SDWAModifiers::src1_sel, "Source 1 select");

    nb::class_<DPPModifiers>(m, "DPPModifiers")
        .def(nb::init<int, int, int>(),
             nb::arg("row_shr")    = -1,
             nb::arg("row_bcast")  = -1,
             nb::arg("bound_ctrl") = -1,
             "DPP (Data Parallel Primitives) instruction modifiers")
        .def_rw("row_shr", &DPPModifiers::row_shr, "Row shift right")
        .def_rw("row_bcast", &DPPModifiers::row_bcast, "Row broadcast")
        .def_rw("bound_ctrl", &DPPModifiers::bound_ctrl, "Boundary control");

    nb::class_<VOP3PModifiers>(m, "VOP3PModifiers")
        .def(nb::init<const std::vector<int>&, const std::vector<int>&, const std::vector<int>&>(),
             nb::arg("op_sel")    = std::vector<int>{},
             nb::arg("op_sel_hi") = std::vector<int>{},
             nb::arg("byte_sel")  = std::vector<int>{},
             "VOP3P (Vector Operation 3 Packed) instruction modifiers")
        .def_rw("op_sel", &VOP3PModifiers::op_sel, "Operand select")
        .def_rw("op_sel_hi", &VOP3PModifiers::op_sel_hi, "Operand select high")
        .def_rw("byte_sel", &VOP3PModifiers::byte_sel, "Byte select");

    // HighBitSel enum for True16 modifiers
    nb::enum_<HighBitSel>(m, "HighBitSel")
        .value("NONE", HighBitSel::NONE)
        .value("LOW", HighBitSel::LOW)
        .value("HIGH", HighBitSel::HIGH);

    nb::class_<True16Modifiers>(m, "True16Modifiers")
        .def(nb::init<HighBitSel>(),
             nb::arg("high_bit") = HighBitSel::NONE,
             "True16 instruction modifiers for selecting high or low 16 bits")
        .def(nb::init<int>(), nb::arg("high_bit_int"), "True16 modifier from integer")
        .def_rw("high_bit", &True16Modifiers::high_bit, "High/low bit selection");

    // ========================================================================
    // Bind StinkyInstruction (exposing rocisa-compatible API)
    // ========================================================================

    nb::class_<StinkyInstruction>(m, "StinkyInstruction", "An instruction in the StinkyTofu IR")
        // Comment property (read/write)
        .def_prop_rw(
            "comment",
            [](const StinkyInstruction& inst) -> std::string {
                const CommentData* comment = inst.getModifier<CommentData>();
                return comment ? comment->comment : "";
            },
            [](StinkyInstruction& inst, const std::string& comment) {
                // This is a simplified version - actual implementation would need
                // to handle modifier mutation properly through StinkyInstruction API
                if(!comment.empty())
                {
                    inst.addModifier(CommentData(comment));
                }
            },
            "Get/set instruction comment")

        // Destination registers (read-only)
        .def_prop_ro(
            "dst",
            [](const StinkyInstruction& inst) -> std::vector<StinkyRegister> {
                return inst.getDestRegs();
            },
            "Get destination registers")

        // Source registers (read-only)
        .def_prop_ro(
            "srcs",
            [](const StinkyInstruction& inst) -> std::vector<StinkyRegister> {
                return inst.getSrcRegs();
            },
            "Get source registers")

        // DPP modifier (optional)
        .def_prop_ro(
            "dpp",
            [](const StinkyInstruction& inst) -> std::optional<DPPModifiers> {
                const DPPModifiers* dpp = inst.getModifier<DPPModifiers>();
                return dpp ? std::optional<DPPModifiers>(*dpp) : std::nullopt;
            },
            "Get DPP modifier if present")

        // SDWA modifier (optional)
        .def_prop_ro(
            "sdwa",
            [](const StinkyInstruction& inst) -> std::optional<SDWAModifiers> {
                const SDWAModifiers* sdwa = inst.getModifier<SDWAModifiers>();
                return sdwa ? std::optional<SDWAModifiers>(*sdwa) : std::nullopt;
            },
            "Get SDWA modifier if present")

        // VOP3P modifier (optional)
        .def_prop_ro(
            "vop3",
            [](const StinkyInstruction& inst) -> std::optional<VOP3PModifiers> {
                const VOP3PModifiers* vop3 = inst.getModifier<VOP3PModifiers>();
                return vop3 ? std::optional<VOP3PModifiers>(*vop3) : std::nullopt;
            },
            "Get VOP3P modifier if present")

        // getParams() - return tuple of (dests, srcs) like rocisa
        .def(
            "getParams",
            [](const StinkyInstruction& inst)
                -> std::tuple<std::vector<StinkyRegister>, std::vector<StinkyRegister>> {
                return std::make_tuple(inst.getDestRegs(), inst.getSrcRegs());
            },
            "Get instruction parameters as (dests, srcs) tuple")

        // String representation
        .def(
            "__str__",
            [](const StinkyInstruction& inst) {
                std::ostringstream oss;
                inst.dump(oss, false);
                return oss.str();
            },
            "Get string representation of instruction")

        // Prevent deepcopy and pickling (same as rocisa::Instruction)
        .def("__deepcopy__",
             [](const StinkyInstruction& self, const nb::dict&) {
                 throw std::runtime_error("Deepcopy not supported for StinkyInstruction");
                 return nullptr;
             })
        .def("__reduce__", [](const StinkyInstruction& self) {
            throw std::runtime_error("Pickling not supported for StinkyInstruction");
            return nullptr;
        });

    // ========================================================================
    // Bind IRListModule Class
    // ========================================================================

    nb::class_<IRListModule>(m, "IRListModule")
        .def("getName", &IRListModule::getName, "Get the name of this module")
        .def("add",
             &IRListModule::add,
             nb::arg("insts"),
             nb::rv_policy::reference,
             "Add instruction(s) to this module (accepts a list, returns the same list for "
             "chaining)")
        .def("addModule",
             &IRListModule::addModule,
             nb::arg("module"),
             nb::rv_policy::reference,
             "Add all instructions from another module to this module")
        .def("emitAssembly",
             &IRListModule::emitAssembly,
             nb::arg("emit_cycle_info") = false,
             nb::arg("emit_comments")   = true,
             "Emit the assembly code for all instructions in this module")
        .def("remapVirtualRegisters",
             &IRListModule::remapVirtualRegisters,
             nb::arg("vgprOffset"),
             nb::arg("sgprOffset"),
             "Remap all virtual registers in this module to physical registers (in-place)")
        .def(
            "cloneAndRemap",
            [](const IRListModule& self, int vgprOffset, int sgprOffset) {
                return unwrapExpected(self.cloneAndRemap(vgprOffset, sgprOffset));
            },
            nb::arg("vgprOffset"),
            nb::arg("sgprOffset"),
            "Create a deep copy of this module and remap its virtual registers (for template "
            "reuse)")
        .def("__str__", &IRListModule::toString, "Get a string representation of this module");

    // ========================================================================
    // Bind StinkyAsmIR Class (Low-Level Assembly IR)
    // ========================================================================

    auto cls = nb::class_<StinkyTofu>(m, "StinkyAsmIR");
    cls.def(nb::init<std::array<int, 3>>(),
            nb::arg("arch"),
            "Create a StinkyAsmIR builder for the specified architecture (e.g., [9, 4, 2])")

        .def("createIRList",
             &StinkyTofu::createIRList,
             nb::arg("name") = "",
             "Create a new IRList module with the given name");

    // Vector ALU Instructions
    bindDSS(cls,
            "VAddU32",
            &StinkyTofu::VAddU32,
            "Create a V_ADD_U32 instruction (32-bit unsigned integer add)");
    bindDSS(cls,
            "VMulF32",
            &StinkyTofu::VMulF32,
            "Create a V_MUL_F32 instruction (32-bit float multiply)");

    // Scalar Instructions
    bindDS(cls,
           "SAbsI32",
           &StinkyTofu::SAbsI32,
           "Create an S_ABS_I32 instruction (scalar absolute value)");
    bindNoArg(cls,
              "SBarrier",
              &StinkyTofu::SBarrier,
              "Create an S_BARRIER instruction (synchronization barrier)");

    // Composite Instructions (may return multiple instructions)
    bindDSS(cls,
            "VAddPKF32",
            &StinkyTofu::VAddPKF32,
            "Create V_PK_ADD_F32 instruction(s) - returns 1 instruction if supported, 2 V_ADD_F32 "
            "otherwise");

    // ========================================================================
    // Vector Arithmetic Instructions
    // ========================================================================

    bindDSS(cls, "VAddF16", &StinkyTofu::VAddF16);
    bindDSS(cls, "VAddF32", &StinkyTofu::VAddF32);
    bindDSS(cls, "VAddF64", &StinkyTofu::VAddF64);
    bindDSS(cls, "VAddI32", &StinkyTofu::VAddI32);
    bindDSS(cls, "VAddCOU32", &StinkyTofu::VAddCOU32);
    bindDSS(cls, "VAddCCOU32", &StinkyTofu::VAddCCOU32);
    bindDSS(cls, "VAddPKF16", &StinkyTofu::VAddPKF16);
    bindDSS(cls, "VAdd3U32", &StinkyTofu::VAdd3U32);
    bindDSS(cls, "VSubF32", &StinkyTofu::VSubF32);
    bindDSS(cls, "VSubI32", &StinkyTofu::VSubI32);
    bindDSS(cls, "VSubU32", &StinkyTofu::VSubU32);
    bindDSS(cls, "VSubCoU32", &StinkyTofu::VSubCoU32);

    // Vector Multiply Instructions
    bindDSS(cls, "VMulF16", &StinkyTofu::VMulF16);
    bindDSS(cls, "VMulF64", &StinkyTofu::VMulF64);
    bindDSS(cls, "VMulPKF16", &StinkyTofu::VMulPKF16);
    bindDSS(cls, "VMulPKF32S", &StinkyTofu::VMulPKF32S);
    bindDSS(cls, "VMulLOU32", &StinkyTofu::VMulLOU32);
    bindDSS(cls, "VMulHII32", &StinkyTofu::VMulHII32);
    bindDSS(cls, "VMulHIU32", &StinkyTofu::VMulHIU32);
    bindDSS(cls, "VMulI32I24", &StinkyTofu::VMulI32I24);
    bindDSS(cls, "VMulU32U24", &StinkyTofu::VMulU32U24);

    // Vector MAC/FMA Instructions
    bindDSS(cls, "VMacF32", &StinkyTofu::VMacF32);
    bindDSSS(cls, "VFmaF16", &StinkyTofu::VFmaF16);
    bindDSSS(cls, "VFmaF32", &StinkyTofu::VFmaF32);
    bindDSSS(cls, "VFmaF64", &StinkyTofu::VFmaF64);
    bindDSSS(cls, "VFmaPKF16", &StinkyTofu::VFmaPKF16);
    cls.def(
        "VFmaMixF32",
        [](StinkyTofu&           self,
           const StinkyRegister& dst,
           const StinkyRegister& src0,
           const StinkyRegister& src1,
           const StinkyRegister& src2,
           const std::string&    comment) {
            return unwrapExpected(self.VFmaMixF32(dst, src0, src1, src2, comment));
        },
        nb::arg("dst"),
        nb::arg("src0"),
        nb::arg("src1"),
        nb::arg("src2"),
        nb::arg("comment") = "",
        nb::rv_policy::reference,
        "Vector FMA with mixed precision (gfx1250+ only)");
    bindDSSS(cls, "VMadI32I24", &StinkyTofu::VMadI32I24);
    bindDSSS(cls, "VMadU32U24", &StinkyTofu::VMadU32U24);
    bindDSSS(cls, "VMadMixF32", &StinkyTofu::VMadMixF32);

    // Vector Dot Product Instructions
    bindDSSS(cls, "VDot2CF32F16", &StinkyTofu::VDot2CF32F16);
    bindDSSS(cls, "VDot2F32F16", &StinkyTofu::VDot2F32F16);
    bindDSSS(cls, "VDot2F32BF16", &StinkyTofu::VDot2F32BF16);
    bindDSSS(cls, "VDot2CF32BF16", &StinkyTofu::VDot2CF32BF16);

    // Vector Transcendental Instructions
    bindDS(cls, "VExpF16", &StinkyTofu::VExpF16);
    bindDS(cls, "VExpF32", &StinkyTofu::VExpF32);
    bindDS(cls, "VRcpF16", &StinkyTofu::VRcpF16);
    bindDS(cls, "VRcpF32", &StinkyTofu::VRcpF32);
    bindDS(cls, "VRcpIFlagF32", &StinkyTofu::VRcpIFlagF32);
    bindDS(cls, "VRsqF16", &StinkyTofu::VRsqF16);
    bindDS(cls, "VRsqF32", &StinkyTofu::VRsqF32);
    cls.def(
        "VRsqIFlagF32",
        [](StinkyTofu&           self,
           const StinkyRegister& dst,
           const StinkyRegister& src,
           const std::string&    comment) {
            return unwrapExpected(self.VRsqIFlagF32(dst, src, comment));
        },
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("comment") = "",
        nb::rv_policy::reference,
        "Vector reciprocal square root with integer flag (gfx1250+ only)");
    bindDS(cls, "VRndneF32", &StinkyTofu::VRndneF32);

    // Vector Min/Max/Med Instructions
    bindDSS(cls, "VMaxF16", &StinkyTofu::VMaxF16);
    bindDSS(cls, "VMaxF32", &StinkyTofu::VMaxF32);
    bindDSS(cls, "VMaxF64", &StinkyTofu::VMaxF64);
    bindDSS(cls, "VMaxI32", &StinkyTofu::VMaxI32);
    bindDSS(cls, "VMaxPKF16", &StinkyTofu::VMaxPKF16);
    bindDSS(cls, "VMinF16", &StinkyTofu::VMinF16);
    bindDSS(cls, "VMinF32", &StinkyTofu::VMinF32);
    bindDSS(cls, "VMinF64", &StinkyTofu::VMinF64);
    bindDSS(cls, "VMinI32", &StinkyTofu::VMinI32);
    bindDSSS(cls, "VMed3I32", &StinkyTofu::VMed3I32);
    bindDSSS(cls, "VMed3F32", &StinkyTofu::VMed3F32);

    // Vector Bitwise Instructions
    bindDSS(cls, "VAndB32", &StinkyTofu::VAndB32);
    bindDSSS(cls, "VAndOrB32", &StinkyTofu::VAndOrB32);
    bindDSS(cls, "VOrB32", &StinkyTofu::VOrB32);
    bindDSS(cls, "VXorB32", &StinkyTofu::VXorB32);
    bindDS(cls, "VNotB32", &StinkyTofu::VNotB32);
    bindDS(cls, "VPrngB32", &StinkyTofu::VPrngB32);
    bindDSS(cls, "VCndMaskB32", &StinkyTofu::VCndMaskB32);

    // Vector Shift Instructions
    bindDSS(cls, "VLShiftLeftB16", &StinkyTofu::VLShiftLeftB16);
    bindDSS(cls, "VLShiftLeftB32", &StinkyTofu::VLShiftLeftB32);
    bindDSS(cls, "VLShiftRightB32", &StinkyTofu::VLShiftRightB32);
    bindDSS(cls, "VLShiftLeftB64", &StinkyTofu::VLShiftLeftB64);
    bindDSS(cls, "VLShiftRightB64", &StinkyTofu::VLShiftRightB64);
    bindDSS(cls, "VAShiftRightI32", &StinkyTofu::VAShiftRightI32);

    // Vector Move/Utility Instructions
    bindDS(cls, "VMovB32", &StinkyTofu::VMovB32);
    bindDS(cls, "VSwapB32", &StinkyTofu::VSwapB32);
    bindDSS(cls, "VPackF16toB32", &StinkyTofu::VPackF16toB32);
    bindDSSS(cls, "VPermB32", &StinkyTofu::VPermB32);

    // Vector Bit Field Instructions
    bindDSSS(cls, "VBfeI32", &StinkyTofu::VBfeI32);
    bindDSSS(cls, "VBfeU32", &StinkyTofu::VBfeU32);
    bindDSSS(cls, "VBfiB32", &StinkyTofu::VBfiB32);

    // Vector Accumulator Access Instructions
    bindDS(cls, "VAccvgprReadB32", &StinkyTofu::VAccvgprReadB32);
    bindDS(cls, "VAccvgprWrite", &StinkyTofu::VAccvgprWrite);
    bindDS(cls, "VAccvgprWriteB32", &StinkyTofu::VAccvgprWriteB32);
    bindDS(cls, "VReadfirstlaneB32", &StinkyTofu::VReadfirstlaneB32);

    // ========================================================================
    // Branch Instructions
    // ========================================================================
    bindBranch(cls, "SBranch", &StinkyTofu::SBranch, "Unconditional branch to label");
    bindBranch(cls, "SCBranchSCC0", &StinkyTofu::SCBranchSCC0, "Conditional branch if SCC == 0");
    bindBranch(cls, "SCBranchSCC1", &StinkyTofu::SCBranchSCC1, "Conditional branch if SCC == 1");
    bindBranch(cls, "SCBranchVCCNZ", &StinkyTofu::SCBranchVCCNZ, "Conditional branch if VCC != 0");
    bindBranch(cls, "SCBranchVCCZ", &StinkyTofu::SCBranchVCCZ, "Conditional branch if VCC == 0");
    bindS(cls, "SSetPCB64", &StinkyTofu::SSetPCB64, "Set program counter to value in sgpr");
    bindDS(cls, "SSwapPCB64", &StinkyTofu::SSwapPCB64, "Swap program counter with value in sgpr");
    bindBranch(cls, "SCBranchExecZ", &StinkyTofu::SCBranchExecZ, "Conditional branch if EXEC == 0");
    bindBranch(
        cls, "SCBranchExecNZ", &StinkyTofu::SCBranchExecNZ, "Conditional branch if EXEC != 0");

    // ========================================================================
    // Compare Instructions (from cmp.hpp)
    // ========================================================================

    // Scalar Compare Instructions
    bindSS(cls, "SCmpEQI32", &StinkyTofu::SCmpEQI32, "Scalar compare equal (signed 32-bit)");
    bindSS(cls, "SCmpEQU32", &StinkyTofu::SCmpEQU32, "Scalar compare equal (unsigned 32-bit)");
    bindSS(cls, "SCmpEQU64", &StinkyTofu::SCmpEQU64, "Scalar compare equal (unsigned 64-bit)");
    bindSS(cls,
           "SCmpGeI32",
           &StinkyTofu::SCmpGeI32,
           "Scalar compare greater or equal (signed 32-bit)");
    bindSS(cls,
           "SCmpGeU32",
           &StinkyTofu::SCmpGeU32,
           "Scalar compare greater or equal (unsigned 32-bit)");
    bindSS(cls, "SCmpGtI32", &StinkyTofu::SCmpGtI32, "Scalar compare greater than (signed 32-bit)");
    bindSS(
        cls, "SCmpGtU32", &StinkyTofu::SCmpGtU32, "Scalar compare greater than (unsigned 32-bit)");
    bindSS(
        cls, "SCmpLeI32", &StinkyTofu::SCmpLeI32, "Scalar compare less or equal (signed 32-bit)");
    bindSS(
        cls, "SCmpLeU32", &StinkyTofu::SCmpLeU32, "Scalar compare less or equal (unsigned 32-bit)");
    bindSS(cls, "SCmpLgU32", &StinkyTofu::SCmpLgU32, "Scalar compare not equal (unsigned 32-bit)");
    bindSS(cls, "SCmpLgI32", &StinkyTofu::SCmpLgI32, "Scalar compare not equal (signed 32-bit)");
    bindSS(cls, "SCmpLgU64", &StinkyTofu::SCmpLgU64, "Scalar compare not equal (unsigned 64-bit)");
    bindSS(cls, "SCmpLtI32", &StinkyTofu::SCmpLtI32, "Scalar compare less than (signed 32-bit)");
    bindSS(cls, "SCmpLtU32", &StinkyTofu::SCmpLtU32, "Scalar compare less than (unsigned 32-bit)");
    bindSS(cls, "SBitcmp1B32", &StinkyTofu::SBitcmp1B32, "Scalar bit compare (32-bit)");

    // Scalar Compare with Immediate (SCMPK)
    bindSI(cls,
           "SCmpKEQU32",
           &StinkyTofu::SCmpKEQU32,
           "Scalar compare with immediate equal (unsigned 32-bit)");
    bindSI(cls,
           "SCmpKGeU32",
           &StinkyTofu::SCmpKGeU32,
           "Scalar compare with immediate greater or equal (unsigned 32-bit)");
    bindSI(cls,
           "SCmpKGtU32",
           &StinkyTofu::SCmpKGtU32,
           "Scalar compare with immediate greater than (unsigned 32-bit)");
    bindSI(cls,
           "SCmpKLGU32",
           &StinkyTofu::SCmpKLGU32,
           "Scalar compare with immediate not equal (unsigned 32-bit)");

    // Vector Compare Instructions
    bindDSS(cls, "VCmpEQF32", &StinkyTofu::VCmpEQF32, "Vector compare equal (F32)");
    bindDSS(cls, "VCmpEQF64", &StinkyTofu::VCmpEQF64, "Vector compare equal (F64)");
    bindDSS(cls, "VCmpEQU32", &StinkyTofu::VCmpEQU32, "Vector compare equal (U32)");
    bindDSS(cls, "VCmpEQI32", &StinkyTofu::VCmpEQI32, "Vector compare equal (I32)");
    bindDSS(cls, "VCmpGEF16", &StinkyTofu::VCmpGEF16, "Vector compare greater or equal (F16)");
    bindDSS(cls, "VCmpGTF16", &StinkyTofu::VCmpGTF16, "Vector compare greater than (F16)");
    bindDSS(cls, "VCmpGEF32", &StinkyTofu::VCmpGEF32, "Vector compare greater or equal (F32)");
    bindDSS(cls, "VCmpGTF32", &StinkyTofu::VCmpGTF32, "Vector compare greater than (F32)");
    bindDSS(cls, "VCmpGEF64", &StinkyTofu::VCmpGEF64, "Vector compare greater or equal (F64)");
    bindDSS(cls, "VCmpGTF64", &StinkyTofu::VCmpGTF64, "Vector compare greater than (F64)");
    bindDSS(cls, "VCmpGEI32", &StinkyTofu::VCmpGEI32, "Vector compare greater or equal (I32)");
    bindDSS(cls, "VCmpGTI32", &StinkyTofu::VCmpGTI32, "Vector compare greater than (I32)");
    bindDSS(cls, "VCmpGEU32", &StinkyTofu::VCmpGEU32, "Vector compare greater or equal (U32)");
    bindDSS(cls, "VCmpGtU32", &StinkyTofu::VCmpGtU32, "Vector compare greater than (U32)");
    bindDSS(cls, "VCmpLeU32", &StinkyTofu::VCmpLeU32, "Vector compare less or equal (U32)");
    bindDSS(cls, "VCmpLeI32", &StinkyTofu::VCmpLeI32, "Vector compare less or equal (I32)");
    bindDSS(cls, "VCmpLtI32", &StinkyTofu::VCmpLtI32, "Vector compare less than (I32)");
    bindDSS(cls, "VCmpLtU32", &StinkyTofu::VCmpLtU32, "Vector compare less than (U32)");
    bindDSS(cls, "VCmpUF32", &StinkyTofu::VCmpUF32, "Vector compare unordered (F32)");
    bindDSS(cls, "VCmpNeI32", &StinkyTofu::VCmpNeI32, "Vector compare not equal (I32)");
    bindDSS(cls, "VCmpNeU32", &StinkyTofu::VCmpNeU32, "Vector compare not equal (U32)");
    bindDSS(cls, "VCmpNeU64", &StinkyTofu::VCmpNeU64, "Vector compare not equal (U64)");
    bindDSS(cls, "VCmpClassF32", &StinkyTofu::VCmpClassF32, "Vector compare class (F32)");

    // Vector Compare with EXEC modification (VCmpX)
    bindDSS(cls,
            "VCmpXClassF32",
            &StinkyTofu::VCmpXClassF32,
            "Vector compare class with EXEC modification (F32)");
    bindDSS(cls,
            "VCmpXEqU32",
            &StinkyTofu::VCmpXEqU32,
            "Vector compare equal with EXEC modification (U32)");
    bindDSS(cls,
            "VCmpXGeU32",
            &StinkyTofu::VCmpXGeU32,
            "Vector compare greater or equal with EXEC modification (U32)");
    bindDSS(cls,
            "VCmpXGtU32",
            &StinkyTofu::VCmpXGtU32,
            "Vector compare greater than with EXEC modification (U32)");
    bindDSS(cls,
            "VCmpXLeU32",
            &StinkyTofu::VCmpXLeU32,
            "Vector compare less or equal with EXEC modification (U32)");
    bindDSS(cls,
            "VCmpXLeI32",
            &StinkyTofu::VCmpXLeI32,
            "Vector compare less or equal with EXEC modification (I32)");
    bindDSS(cls,
            "VCmpXLtF32",
            &StinkyTofu::VCmpXLtF32,
            "Vector compare less than with EXEC modification (F32)");
    bindDSS(cls,
            "VCmpXLtI32",
            &StinkyTofu::VCmpXLtI32,
            "Vector compare less than with EXEC modification (I32)");
    bindDSS(cls,
            "VCmpXLtU32",
            &StinkyTofu::VCmpXLtU32,
            "Vector compare less than with EXEC modification (U32)");
    bindDSS(cls,
            "VCmpXLtU64",
            &StinkyTofu::VCmpXLtU64,
            "Vector compare less than with EXEC modification (U64)");
    bindDSS(cls,
            "VCmpXNeU16",
            &StinkyTofu::VCmpXNeU16,
            "Vector compare not equal with EXEC modification (U16)");
    bindDSS(cls,
            "VCmpXNeU32",
            &StinkyTofu::VCmpXNeU32,
            "Vector compare not equal with EXEC modification (U32)");

    // ========================================================================
    // Conversion Instructions (from cvt.hpp)
    // ========================================================================

    bindDS(cls, "VCvtF16toF32", &StinkyTofu::VCvtF16toF32, "Convert F16 to F32");
    bindDS(cls, "VCvtF32toF16", &StinkyTofu::VCvtF32toF16, "Convert F32 to F16");
    bindDS(cls, "VCvtF32toU32", &StinkyTofu::VCvtF32toU32, "Convert F32 to U32");
    bindDS(cls, "VCvtU32toF32", &StinkyTofu::VCvtU32toF32, "Convert U32 to F32");
    bindDS(cls, "VCvtI32toF32", &StinkyTofu::VCvtI32toF32, "Convert I32 to F32");
    bindDS(cls, "VCvtF32toI32", &StinkyTofu::VCvtF32toI32, "Convert F32 to I32");
    bindDS(cls, "VCvtFP8toF32", &StinkyTofu::VCvtFP8toF32, "Convert FP8 to F32");
    bindDS(cls, "VCvtBF8toF32", &StinkyTofu::VCvtBF8toF32, "Convert BF8 to F32");
    bindDS(cls, "VCvtPkFP8toF32", &StinkyTofu::VCvtPkFP8toF32, "Convert packed FP8 to F32");
    bindDS(cls, "VCvtPkBF8toF32", &StinkyTofu::VCvtPkBF8toF32, "Convert packed BF8 to F32");
    bindDSS(cls, "VCvtPkF32toFP8", &StinkyTofu::VCvtPkF32toFP8, "Convert packed F32 to FP8");
    bindDSS(cls, "VCvtPkF32toBF8", &StinkyTofu::VCvtPkF32toBF8, "Convert packed F32 to BF8");
    bindDSS(cls,
            "VCvtSRF32toFP8",
            &StinkyTofu::VCvtSRF32toFP8,
            "Convert F32 to FP8 (stochastic rounding)");
    bindDSS(cls,
            "VCvtSRF32toBF8",
            &StinkyTofu::VCvtSRF32toBF8,
            "Convert F32 to BF8 (stochastic rounding)");
    bindDSS(cls,
            "VCvtScalePkFP8toF16",
            &StinkyTofu::VCvtScalePkFP8toF16,
            "Convert scaled packed FP8 to F16");
    bindDSS(cls,
            "VCvtScalePkBF8toF16",
            &StinkyTofu::VCvtScalePkBF8toF16,
            "Convert scaled packed BF8 to F16");
    bindDSS(cls, "VCvtScaleFP8toF16", &StinkyTofu::VCvtScaleFP8toF16, "Convert scaled FP8 to F16");
    bindDSS(cls,
            "VCvtScalePkF16toFP8",
            &StinkyTofu::VCvtScalePkF16toFP8,
            "Convert scaled packed F16 to FP8");
    bindDSS(cls,
            "VCvtScalePkF16toBF8",
            &StinkyTofu::VCvtScalePkF16toBF8,
            "Convert scaled packed F16 to BF8");
    bindDSS(cls,
            "VCvtScaleSRF16toFP8",
            &StinkyTofu::VCvtScaleSRF16toFP8,
            "Convert scaled F16 to FP8 (stochastic rounding)");
    bindDSS(cls,
            "VCvtScaleSRF16toBF8",
            &StinkyTofu::VCvtScaleSRF16toBF8,
            "Convert scaled F16 to BF8 (stochastic rounding)");
    // VCvtBF16toFP32 has extra optional parameters for vgprMask and vi
    cls.def(
        "VCvtBF16toFP32",
        [](StinkyTofu&           self,
           const StinkyRegister& dst,
           const StinkyRegister& src,
           const StinkyRegister* vgprMask,
           int                   vi,
           const std::string&    comment) {
            return unwrapExpected(self.VCvtBF16toFP32(dst, src, vgprMask, vi, comment));
        },
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("vgprMask") = nullptr,
        nb::arg("vi")       = 0,
        nb::arg("comment")  = "",
        nb::rv_policy::reference,
        "Convert BF16 to FP32 (gfx950+ only)");
    bindDSS(cls, "VCvtPkF32toBF16", &StinkyTofu::VCvtPkF32toBF16, "Convert packed F32 to BF16");

    // ========================================================================
    // Memory Instructions (from mem.hpp)
    // ========================================================================

    // DS (LDS) Instructions (using rocisa-compatible naming)
    bindDA(cls, "DSLoadU8", &StinkyTofu::DSLoadU8, "LDS load unsigned 8-bit");
    bindDA(cls, "DSLoadI8", &StinkyTofu::DSLoadI8, "LDS load signed 8-bit");
    bindDA(cls, "DSLoadU16", &StinkyTofu::DSLoadU16, "LDS load unsigned 16-bit");
    bindDA(cls, "DSLoadI16", &StinkyTofu::DSLoadI16, "LDS load signed 16-bit");
    bindDA(cls, "DSLoadB32", &StinkyTofu::DSLoadB32, "LDS load 32-bit");
    bindDA(cls, "DSLoadB64", &StinkyTofu::DSLoadB64, "LDS load 64-bit");
    bindDA(cls, "DSLoadB96", &StinkyTofu::DSLoadB96, "LDS load 96-bit");
    bindDA(cls, "DSLoadB128", &StinkyTofu::DSLoadB128, "LDS load 128-bit");
    bindDA(cls, "DSLoadD16HIU8", &StinkyTofu::DSLoadD16HIU8, "LDS load U8 to D16 high");
    bindDA(cls, "DSLoadD16HIU16", &StinkyTofu::DSLoadD16HIU16, "LDS load U16 to D16 high");
    bindDA(cls, "DSLoadB64TrB4", &StinkyTofu::DSLoadB64TrB4, "LDS load 64-bit transpose B4");
    bindDA(cls, "DSLoadB96TrB6", &StinkyTofu::DSLoadB96TrB6, "LDS load 96-bit transpose B6");
    bindDA(cls, "DSLoadB64TrB8", &StinkyTofu::DSLoadB64TrB8, "LDS load 64-bit transpose B8");
    bindDA(cls, "DSLoadB64TrB16", &StinkyTofu::DSLoadB64TrB16, "LDS load 64-bit transpose B16");
    bindDAA(cls, "DSLoad2B32", &StinkyTofu::DSLoad2B32, "LDS load two 32-bit");
    bindDAA(cls, "DSLoad2B64", &StinkyTofu::DSLoad2B64, "LDS load two 64-bit");
    bindAS(cls, "DSStoreB8", &StinkyTofu::DSStoreB8, "LDS store 8-bit");
    bindAS(cls, "DSStoreB16", &StinkyTofu::DSStoreB16, "LDS store 16-bit");
    bindAS(cls, "DSStoreB32", &StinkyTofu::DSStoreB32, "LDS store 32-bit");
    bindAS(cls, "DSStoreB64", &StinkyTofu::DSStoreB64, "LDS store 64-bit");
    bindAS(cls, "DSStoreB96", &StinkyTofu::DSStoreB96, "LDS store 96-bit");
    bindAS(cls, "DSStoreB128", &StinkyTofu::DSStoreB128, "LDS store 128-bit");
    bindAS(cls, "DSStoreD16HIB8", &StinkyTofu::DSStoreD16HIB8, "LDS store D16 high to B8");
    bindAS(cls, "DSStoreD16HIB16", &StinkyTofu::DSStoreD16HIB16, "LDS store D16 high to B16");
    bindAAS(cls, "DSStore2B32", &StinkyTofu::DSStore2B32, "LDS store two 32-bit");
    bindAAS(cls, "DSStore2B64", &StinkyTofu::DSStore2B64, "LDS store two 64-bit");
    bindDS(cls, "DSBPermuteB32", &StinkyTofu::DSBPermuteB32, "LDS byte permute 32-bit");

    // Buffer (MUBUF) Instructions
    bindDA(cls, "BufferLoadU8", &StinkyTofu::BufferLoadU8, "Buffer load unsigned byte");
    bindDA(cls, "BufferLoadI8", &StinkyTofu::BufferLoadI8, "Buffer load signed byte");
    bindDA(cls, "BufferLoadU16", &StinkyTofu::BufferLoadU16, "Buffer load unsigned short");
    bindDA(cls, "BufferLoadI16", &StinkyTofu::BufferLoadI16, "Buffer load signed short");
    bindDA(cls, "BufferLoadB32", &StinkyTofu::BufferLoadB32, "Buffer load 32-bit");
    bindDA(cls, "BufferLoadB64", &StinkyTofu::BufferLoadB64, "Buffer load 64-bit");
    bindDA(cls, "BufferLoadB96", &StinkyTofu::BufferLoadB96, "Buffer load 96-bit");
    bindDA(cls, "BufferLoadB128", &StinkyTofu::BufferLoadB128, "Buffer load 128-bit");
    bindDA(cls, "BufferLoadD16U8", &StinkyTofu::BufferLoadD16U8, "Buffer load U8 to D16");
    bindDA(cls, "BufferLoadD16HIU8", &StinkyTofu::BufferLoadD16HIU8, "Buffer load U8 to D16 high");
    bindDA(cls, "BufferLoadD16I8", &StinkyTofu::BufferLoadD16I8, "Buffer load I8 to D16");
    bindDA(cls, "BufferLoadD16HII8", &StinkyTofu::BufferLoadD16HII8, "Buffer load I8 to D16 high");
    bindDA(cls, "BufferLoadD16B16", &StinkyTofu::BufferLoadD16B16, "Buffer load B16 to D16");
    bindDA(
        cls, "BufferLoadD16HIB16", &StinkyTofu::BufferLoadD16HIB16, "Buffer load B16 to D16 high");
    bindAS(cls, "BufferStoreB8", &StinkyTofu::BufferStoreB8, "Buffer store byte");
    bindAS(cls,
           "BufferStoreD16HIU8",
           &StinkyTofu::BufferStoreD16HIU8,
           "Buffer store D16 high to byte");
    bindAS(cls, "BufferStoreB16", &StinkyTofu::BufferStoreB16, "Buffer store short");
    bindAS(cls,
           "BufferStoreD16HIB16",
           &StinkyTofu::BufferStoreD16HIB16,
           "Buffer store D16 high to short");
    bindAS(cls, "BufferStoreB32", &StinkyTofu::BufferStoreB32, "Buffer store 32-bit");
    bindAS(cls, "BufferStoreB64", &StinkyTofu::BufferStoreB64, "Buffer store 64-bit");
    bindAS(cls, "BufferStoreB96", &StinkyTofu::BufferStoreB96, "Buffer store 96-bit");
    bindAS(cls, "BufferStoreB128", &StinkyTofu::BufferStoreB128, "Buffer store 128-bit");
    bindDS(cls, "BufferAtomicAddF32", &StinkyTofu::BufferAtomicAddF32, "Buffer atomic add F32");
    bindDAA(cls,
            "BufferAtomicCmpswapB32",
            &StinkyTofu::BufferAtomicCmpswapB32,
            "Buffer atomic compare-swap 32-bit");
    bindDAA(cls,
            "BufferAtomicCmpswapB64",
            &StinkyTofu::BufferAtomicCmpswapB64,
            "Buffer atomic compare-swap 64-bit");

    // Scalar Memory (SMEM) Instructions
    // Scalar Memory Load Instructions
    // Size-based naming (B32/B64/etc.) is architecture-agnostic
    // Backend emits correct ISA: s_load_dword (GFX9) or s_load_b32 (GFX12)
    bindSMemLoad(cls,
                 "SLoadB32",
                 &StinkyTofu::SLoadB32,
                 "Scalar load 32-bit (1 dword) with optional offset\n"
                 "Emits s_load_dword (GFX9) or s_load_b32 (GFX12)");
    bindSMemLoad(cls,
                 "SLoadB64",
                 &StinkyTofu::SLoadB64,
                 "Scalar load 64-bit (2 dwords) with optional offset\n"
                 "Emits s_load_dwordx2 (GFX9) or s_load_b64 (GFX12)");
    bindSMemLoad(cls,
                 "SLoadB128",
                 &StinkyTofu::SLoadB128,
                 "Scalar load 128-bit (4 dwords) with optional offset\n"
                 "Emits s_load_dwordx4 (GFX9) or s_load_b128 (GFX12)");
    bindSMemLoad(cls,
                 "SLoadB256",
                 &StinkyTofu::SLoadB256,
                 "Scalar load 256-bit (8 dwords) with optional offset\n"
                 "Emits s_load_dwordx8 (GFX9) or s_load_b256 (GFX12)");
    bindSMemLoad(cls,
                 "SLoadB512",
                 &StinkyTofu::SLoadB512,
                 "Scalar load 512-bit (16 dwords) with optional offset\n"
                 "Emits s_load_dwordx16 (GFX9) or s_load_b512 (GFX12)");

    bindAS(cls, "SStoreB32", &StinkyTofu::SStoreB32, "Scalar store 32-bit");
    bindAS(cls, "SStoreB64", &StinkyTofu::SStoreB64, "Scalar store 64-bit");
    bindAS(cls, "SStoreB128", &StinkyTofu::SStoreB128, "Scalar store 128-bit");
    bindDS(cls, "SAtomicDec", &StinkyTofu::SAtomicDec, "Scalar atomic decrement");

    // Flat Memory Instructions
    bindDA(cls, "FlatLoadU8", &StinkyTofu::FlatLoadU8, "Flat load unsigned byte");
    bindDA(cls, "FlatLoadI8", &StinkyTofu::FlatLoadI8, "Flat load signed byte");
    bindDA(cls, "FlatLoadU16", &StinkyTofu::FlatLoadU16, "Flat load unsigned short");
    bindDA(cls, "FlatLoadI16", &StinkyTofu::FlatLoadI16, "Flat load signed short");
    bindDA(cls, "FlatLoadD16U8", &StinkyTofu::FlatLoadD16U8, "Flat load U8 to D16");
    bindDA(cls, "FlatLoadD16HIU8", &StinkyTofu::FlatLoadD16HIU8, "Flat load U8 to D16 high");
    bindDA(cls, "FlatLoadD16I8", &StinkyTofu::FlatLoadD16I8, "Flat load I8 to D16");
    bindDA(cls, "FlatLoadD16HII8", &StinkyTofu::FlatLoadD16HII8, "Flat load I8 to D16 high");
    bindDA(cls, "FlatLoadD16B16", &StinkyTofu::FlatLoadD16B16, "Flat load B16 to D16");
    bindDA(cls, "FlatLoadD16HIB16", &StinkyTofu::FlatLoadD16HIB16, "Flat load B16 to D16 high");
    bindDA(cls, "FlatLoadB32", &StinkyTofu::FlatLoadB32, "Flat load 32-bit");
    bindDA(cls, "FlatLoadB64", &StinkyTofu::FlatLoadB64, "Flat load 64-bit");
    bindDA(cls, "FlatLoadB96", &StinkyTofu::FlatLoadB96, "Flat load 96-bit");
    bindDA(cls, "FlatLoadB128", &StinkyTofu::FlatLoadB128, "Flat load 128-bit");
    bindAS(cls, "FlatStoreB8", &StinkyTofu::FlatStoreB8, "Flat store byte");
    bindAS(cls, "FlatStoreD16HIU8", &StinkyTofu::FlatStoreD16HIU8, "Flat store D16 high to byte");
    bindAS(cls, "FlatStoreB16", &StinkyTofu::FlatStoreB16, "Flat store short");
    bindAS(
        cls, "FlatStoreD16HIB16", &StinkyTofu::FlatStoreD16HIB16, "Flat store D16 high to short");
    bindAS(cls, "FlatStoreB32", &StinkyTofu::FlatStoreB32, "Flat store 32-bit");
    bindAS(cls, "FlatStoreB64", &StinkyTofu::FlatStoreB64, "Flat store 64-bit");
    bindAS(cls, "FlatStoreB96", &StinkyTofu::FlatStoreB96, "Flat store 96-bit");
    bindAS(cls, "FlatStoreB128", &StinkyTofu::FlatStoreB128, "Flat store 128-bit");
    bindDAA(
        cls, "FlatAtomicCmpswapB32", &StinkyTofu::FlatAtomicCmpswapB32, "Flat atomic compare-swap");

    // Composite Instructions
    // ========================================================================

    cls.def("VMulPKF32",
            &StinkyTofu::VMulPKF32,
            nb::arg("dst"),
            nb::arg("src0"),
            nb::arg("src1"),
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            "Composite: V_PK_MUL_F32 or 2x V_MUL_F32")
        .def("VMovB64",
             &StinkyTofu::VMovB64,
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "Composite: V_MOV_B64 or 2x V_MOV_B32")
        .def("VLShiftLeftOrB32",
             &StinkyTofu::VLShiftLeftOrB32,
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "Composite: (src0 << shift) | src1")
        .def("VAddLShiftLeftU32",
             &StinkyTofu::VAddLShiftLeftU32,
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "Composite: (src0 + src1) << shift")
        .def("VLShiftLeftAddU32",
             &StinkyTofu::VLShiftLeftAddU32,
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "Composite: (src0 << shift) + src1")
        .def("SWaitCnt",
             &StinkyTofu::SWaitCnt,
             nb::arg("vlcnt")   = -1,
             nb::arg("vscnt")   = -1,
             nb::arg("dscnt")   = -1,
             nb::arg("kmcnt")   = -1,
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "Composite: S_WAITCNT with architecture-specific counters")

        .def("SWaitAlu",
             &StinkyTofu::SWaitAlu,
             nb::arg("va_vdst")  = -1,
             nb::arg("va_sdst")  = -1,
             nb::arg("va_ssrc")  = -1,
             nb::arg("hold_cnt") = -1,
             nb::arg("vm_vsrc")  = -1,
             nb::arg("va_vcc")   = -1,
             nb::arg("sa_sdst")  = -1,
             nb::arg("comment")  = "",
             nb::rv_policy::reference,
             "S_WAIT_ALU: Wait for ALU completion")

        .def("SWaitTensorcnt",
             &StinkyTofu::SWaitTensorcnt,
             nb::arg("tensorcnt") = 0,
             nb::arg("comment")   = "",
             nb::rv_policy::reference,
             "S_WAIT_TENSORCNT: Wait for tensor operations")

        .def("SNop",
             &StinkyTofu::SNop,
             nb::arg("waitState"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "S_NOP: Scalar no operation")

        .def("VNop",
             &StinkyTofu::VNop,
             nb::arg("count"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "V_NOP: Vector no operation")

        .def("SEndpgm",
             &StinkyTofu::SEndpgm,
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "S_ENDPGM: End program")

        .def("SSleep",
             &StinkyTofu::SSleep,
             nb::arg("simm16"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "S_SLEEP: Sleep for specified cycles")

        .def("SDcacheWb",
             &StinkyTofu::SDcacheWb,
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "S_DCACHE_WB: Data cache writeback")

        .def("SDelayAlu",
             &StinkyTofu::SDelayAlu,
             nb::arg("instid0"),
             nb::arg("instid0cnt"),
             nb::arg("instskip")   = -1,
             nb::arg("instid1")    = -1,
             nb::arg("instid1cnt") = -1,
             nb::arg("comment")    = "",
             nb::rv_policy::reference,
             "S_DELAY_ALU: Delay ALU instruction scheduling")

        .def("SSetPrior",
             &StinkyTofu::SSetPrior,
             nb::arg("prior"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "S_SETPRIO: Set priority")

        .def("SSetVgprMsb",
             nb::overload_cast<int, const std::string&>(&StinkyTofu::SSetVgprMsb),
             nb::arg("simm16"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "S_SET_VGPR_MSB: Set VGPR MSB (encoded value)")

        .def("SSetVgprMsb",
             nb::overload_cast<int, int, int, int, const std::string&>(&StinkyTofu::SSetVgprMsb),
             nb::arg("msbSrc0"),
             nb::arg("msbSrc1"),
             nb::arg("msbSrc2"),
             nb::arg("msbDst"),
             nb::arg("comment") = "",
             nb::rv_policy::reference,
             "S_SET_VGPR_MSB: Set VGPR MSB (individual bits)");

    // ========================================================================
    // Scalar Arithmetic Instructions
    // ========================================================================

    bindDSS(cls, "SMaxI32", &StinkyTofu::SMaxI32);
    bindDSS(cls, "SMaxU32", &StinkyTofu::SMaxU32);
    bindDSS(cls, "SMinI32", &StinkyTofu::SMinI32);
    bindDSS(cls, "SMinU32", &StinkyTofu::SMinU32);
    bindDSS(cls, "SAddI32", &StinkyTofu::SAddI32);
    bindDSS(cls, "SAddU32", &StinkyTofu::SAddU32);
    bindDSS(cls, "SAddCU32", &StinkyTofu::SAddCU32);
    bindDSS(cls, "SMulI32", &StinkyTofu::SMulI32);
    bindDSS(cls, "SMulHII32", &StinkyTofu::SMulHII32);
    bindDSS(cls, "SMulHIU32", &StinkyTofu::SMulHIU32);
    // SMulLOU32 uses Expected<T> - unwrap to Python exception
    cls.def(
        "SMulLOU32",
        [](StinkyTofu&           self,
           const StinkyRegister& dst,
           const StinkyRegister& src0,
           const StinkyRegister& src1,
           const std::string&    comment) {
            return unwrapExpected(self.SMulLOU32(dst, src0, src1, comment));
        },
        nb::arg("dst"),
        nb::arg("src0"),
        nb::arg("src1"),
        nb::arg("comment") = "",
        nb::rv_policy::reference,
        "Create S_MUL_LO_U32 instruction (32-bit unsigned integer multiply, low 32 bits)\n\n"
        "Note: This instruction requires gfx1250+ architecture.\n"
        "Raises RuntimeError if not supported on current architecture.");
    bindDSS(cls, "SSubI32", &StinkyTofu::SSubI32);
    bindDSS(cls, "SSubU32", &StinkyTofu::SSubU32);
    bindDSS(cls, "SSubBU32", &StinkyTofu::SSubBU32);
    cls.def(
        "SSubU64",
        [](StinkyTofu&           self,
           const StinkyRegister& dst,
           const StinkyRegister& src0,
           const StinkyRegister& src1,
           const std::string&    comment) {
            return unwrapExpected(self.SSubU64(dst, src0, src1, comment));
        },
        nb::arg("dst"),
        nb::arg("src0"),
        nb::arg("src1"),
        nb::arg("comment") = "",
        nb::rv_policy::reference,
        "64-bit scalar unsigned subtraction (gfx1250+ only)");

    // Scalar Bitwise Instructions
    bindDSS(cls, "SAndB32", &StinkyTofu::SAndB32);
    bindDSS(cls, "SAndB64", &StinkyTofu::SAndB64);
    bindDSS(cls, "SAndN2B32", &StinkyTofu::SAndN2B32);
    bindDSS(cls, "SOrB32", &StinkyTofu::SOrB32);
    bindDSS(cls, "SOrB64", &StinkyTofu::SOrB64);
    bindDSS(cls, "SXorB32", &StinkyTofu::SXorB32);

    // Scalar Shift Instructions
    bindDSS(cls, "SLShiftLeftB32", &StinkyTofu::SLShiftLeftB32);
    bindDSS(cls, "SLShiftRightB32", &StinkyTofu::SLShiftRightB32);
    bindDSS(cls, "SLShiftLeftB64", &StinkyTofu::SLShiftLeftB64);
    bindDSS(cls, "SLShiftRightB64", &StinkyTofu::SLShiftRightB64);
    bindDSS(cls, "SAShiftRightI32", &StinkyTofu::SAShiftRightI32);
    bindDSS(cls, "SLShiftLeft1AddU32", &StinkyTofu::SLShiftLeft1AddU32);
    bindDSS(cls, "SLShiftLeft2AddU32", &StinkyTofu::SLShiftLeft2AddU32);
    bindDSS(cls, "SLShiftLeft3AddU32", &StinkyTofu::SLShiftLeft3AddU32);
    bindDSS(cls, "SLShiftLeft4AddU32", &StinkyTofu::SLShiftLeft4AddU32);

    // Scalar Move/Control Instructions
    bindDS(cls, "SMovB32", &StinkyTofu::SMovB32);
    bindDS(cls, "SMovB64", &StinkyTofu::SMovB64);
    bindDS(cls, "SCMovB32", &StinkyTofu::SCMovB32);
    bindDS(cls, "SCMovB64", &StinkyTofu::SCMovB64);
    bindDSS(cls, "SCSelectB32", &StinkyTofu::SCSelectB32);
    bindD(cls, "SGetPCB64", &StinkyTofu::SGetPCB64);
    bindDS(cls, "SSetMask", &StinkyTofu::SSetMask);

    // Scalar Bit Manipulation Instructions
    bindDS(cls, "SFf1B32", &StinkyTofu::SFf1B32);
    bindDSS(cls, "SBfmB32", &StinkyTofu::SBfmB32);
    bindDS(cls, "SMovkI32", &StinkyTofu::SMovkI32);
    bindDS(cls, "SSExtI16toI32", &StinkyTofu::SSExtI16toI32);

    // Scalar Exec Mask Instructions
    cls.def(
        "SAndSaveExecB32",
        [](StinkyTofu&           self,
           const StinkyRegister& dst,
           const StinkyRegister& src,
           const std::string&    comment) {
            return unwrapExpected(self.SAndSaveExecB32(dst, src, comment));
        },
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("comment") = "",
        nb::rv_policy::reference,
        "Save exec mask and perform bitwise AND (gfx1250+ only)");
    bindDS(cls, "SAndSaveExecB64", &StinkyTofu::SAndSaveExecB64);
    cls.def(
        "SOrSaveExecB32",
        [](StinkyTofu&           self,
           const StinkyRegister& dst,
           const StinkyRegister& src,
           const std::string&    comment) {
            return unwrapExpected(self.SOrSaveExecB32(dst, src, comment));
        },
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("comment") = "",
        nb::rv_policy::reference,
        "Save exec mask and perform bitwise OR (gfx1250+ only)");
    bindDS(cls, "SOrSaveExecB64", &StinkyTofu::SOrSaveExecB64);

    // Scalar Register Access Instructions
    bindDS(cls, "SGetRegB32", &StinkyTofu::SGetRegB32);
    bindDS(cls, "SSetRegB32", &StinkyTofu::SSetRegB32);
    bindDS(cls, "SSetRegIMM32B32", &StinkyTofu::SSetRegIMM32B32);

    // Label Creation
    cls.def("createLabel",
            &StinkyTofu::createLabel,
            nb::arg("label_name"),
            nb::rv_policy::reference,
            "Create a label instruction (emitted as 'label:' in assembly)")

        // Generic MFMA Creation Functions
        .def(
            "createMFMA",
            [](StinkyTofu&           self,
               const std::string&    instType,
               const std::string&    accType,
               int                   m,
               int                   n,
               int                   k,
               int                   blocks,
               bool                  mfma1k,
               const StinkyRegister& acc,
               const StinkyRegister& a,
               const StinkyRegister& b,
               const StinkyRegister* acc2,
               bool                  neg,
               const std::string&    comment) {
                return unwrapExpected(self.createMFMA(
                    instType, accType, m, n, k, blocks, mfma1k, acc, a, b, acc2, neg, comment));
            },
            nb::arg("instType"),
            nb::arg("accType"),
            nb::arg("m"),
            nb::arg("n"),
            nb::arg("k"),
            nb::arg("blocks"),
            nb::arg("mfma1k"),
            nb::arg("acc"),
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("acc2")    = nullptr,
            nb::arg("neg")     = false,
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            "Create a generic MFMA instruction (e.g., v_mfma_f32_32x32x8_bf16)\n"
            "Parameters:\n"
            "  instType: Input data type ('bf16', 'f16', 'i8', 'f8', 'bf8', etc.)\n"
            "  accType: Accumulator type ('f32', 'i32')\n"
            "  m, n, k: Matrix dimensions\n"
            "  blocks: Number of blocks (default 1)\n"
            "  mfma1k: Whether this is a _1k variant\n"
            "  acc: Accumulator destination register\n"
            "  a, b: Source registers\n"
            "  acc2: Optional accumulator source (defaults to acc if None)\n"
            "  neg: Negate operands\n"
            "  comment: Optional comment")
        .def(
            "createMXMFMA",
            [](StinkyTofu&           self,
               const std::string&    instType,
               const std::string&    accType,
               const std::string&    mxScaleATypeStr,
               const std::string&    mxScaleBTypeStr,
               int                   m,
               int                   n,
               int                   k,
               int                   block,
               const StinkyRegister& acc,
               const StinkyRegister& a,
               const StinkyRegister& b,
               const StinkyRegister& acc2,
               const StinkyRegister& mxsa,
               const StinkyRegister& mxsb,
               bool                  reuseA,
               bool                  reuseB,
               const std::string&    comment) {
                return unwrapExpected(self.createMXMFMA(instType,
                                                        accType,
                                                        mxScaleATypeStr,
                                                        mxScaleBTypeStr,
                                                        m,
                                                        n,
                                                        k,
                                                        block,
                                                        acc,
                                                        a,
                                                        b,
                                                        acc2,
                                                        mxsa,
                                                        mxsb,
                                                        reuseA,
                                                        reuseB,
                                                        comment));
            },
            nb::arg("instType"),
            nb::arg("accType"),
            nb::arg("mxScaleATypeStr"),
            nb::arg("mxScaleBTypeStr"),
            nb::arg("m"),
            nb::arg("n"),
            nb::arg("k"),
            nb::arg("block"),
            nb::arg("acc"),
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("acc2"),
            nb::arg("mxsa"),
            nb::arg("mxsb"),
            nb::arg("reuseA")  = false,
            nb::arg("reuseB")  = false,
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            "Create an MX format MFMA instruction (e.g., v_wmma_scale_f32_16x16x128_f8f6f4)\n"
            "Parameters:\n"
            "  instType: Input data type ('f8', 'f4', 'f6', 'bf8', etc.)\n"
            "  accType: Accumulator type (typically 'f32')\n"
            "  mxScaleATypeStr, mxScaleBTypeStr: Scale format ('e5m3', 'fp8')\n"
            "  m, n, k: Matrix dimensions\n"
            "  block: Block size (16 or other)\n"
            "  acc, a, b, acc2: Register operands\n"
            "  mxsa, mxsb: Scale factor registers\n"
            "  reuseA, reuseB: Matrix reuse flags\n"
            "  comment: Optional comment")
        .def(
            "createSMFMA",
            [](StinkyTofu&           self,
               const std::string&    instType,
               const std::string&    accType,
               int                   m,
               int                   n,
               int                   k,
               int                   blocks,
               bool                  mfma1k,
               const StinkyRegister& acc,
               const StinkyRegister& a,
               const StinkyRegister& b,
               const StinkyRegister& metadata,
               bool                  neg,
               const std::string&    comment) {
                return unwrapExpected(self.createSMFMA(
                    instType, accType, m, n, k, blocks, mfma1k, acc, a, b, metadata, neg, comment));
            },
            nb::arg("instType"),
            nb::arg("accType"),
            nb::arg("m"),
            nb::arg("n"),
            nb::arg("k"),
            nb::arg("blocks"),
            nb::arg("mfma1k"),
            nb::arg("acc"),
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("metadata"),
            nb::arg("neg")     = false,
            nb::arg("comment") = "",
            nb::rv_policy::reference,
            "Create a sparse MFMA instruction (e.g., v_smfmac_f32_16x16x32_bf16)\n"
            "Parameters:\n"
            "  instType: Input data type ('bf16', 'f16', 'i8', 'f8', 'bf8', etc.)\n"
            "  accType: Accumulator type ('f32', 'i32')\n"
            "  m, n, k: Matrix dimensions\n"
            "  blocks: Number of micro-blocks\n"
            "  mfma1k: Whether this is a _1k variant\n"
            "  acc: Accumulator destination register\n"
            "  a, b: Source registers\n"
            "  metadata: Sparsity metadata register\n"
            "  neg: Negate operands\n"
            "  comment: Optional comment");

    // ========================================================================
    // Bind StinkyIR (High-Level IR Builder)
    // ========================================================================

    nb::class_<StinkyIR>(m, "StinkyIR")
        .def(nb::init<std::array<int, 3>>(),
             nb::arg("arch"),
             "Create a StinkyIR high-level function generator for the specified architecture "
             "(e.g., [9, 4, 2])")

        // Division & Remainder Functions
        .def(
            "vectorStaticDivide",
            [](StinkyIR&                    self,
               StinkyTofu&                  builder,
               uint32_t                     qReg,
               uint32_t                     dReg,
               int                          divisor,
               const std::vector<uint32_t>& tmpVgpr,
               const std::string&           comment) {
                return unwrapExpected(
                    self.vectorStaticDivide(builder, qReg, dReg, divisor, tmpVgpr, comment));
            },
            nb::arg("builder"),
            nb::arg("qReg"),
            nb::arg("dReg"),
            nb::arg("divisor"),
            nb::arg("tmpVgpr"),
            nb::arg("comment") = "",
            "Static vector division (@function)\n"
            "Divides vDReg by a constant divisor, storing quotient in vQReg")

        .def(
            "vectorStaticDivideAndRemainder",
            [](StinkyIR&                    self,
               StinkyTofu&                  builder,
               uint32_t                     qReg,
               uint32_t                     rReg,
               uint32_t                     dReg,
               int                          divisor,
               const std::vector<uint32_t>& tmpVgpr,
               bool                         doRemainder,
               const std::string&           comment) {
                return unwrapExpected(self.vectorStaticDivideAndRemainder(
                    builder, qReg, rReg, dReg, divisor, tmpVgpr, doRemainder, comment));
            },
            nb::arg("builder"),
            nb::arg("qReg"),
            nb::arg("rReg"),
            nb::arg("dReg"),
            nb::arg("divisor"),
            nb::arg("tmpVgpr"),
            nb::arg("doRemainder") = true,
            nb::arg("comment")     = "",
            "Static vector division with remainder (@function)")

        .def("vectorUInt32DivideAndRemainder",
             &StinkyIR::vectorUInt32DivideAndRemainder,
             nb::arg("builder"),
             nb::arg("qReg"),
             nb::arg("dReg"),
             nb::arg("divReg"),
             nb::arg("rReg"),
             nb::arg("doRemainder") = true,
             nb::arg("comment")     = "",
             "Dynamic vector division using FP32 reciprocal (@function)")

        .def(
            "scalarStaticDivideAndRemainder",
            [](StinkyIR&                    self,
               StinkyTofu&                  builder,
               uint32_t                     qReg,
               uint32_t                     rReg,
               uint32_t                     dReg,
               int                          divisor,
               const std::vector<uint32_t>& tmpSgpr,
               int                          doRemainder,
               const std::string&           comment) {
                return unwrapExpected(self.scalarStaticDivideAndRemainder(
                    builder, qReg, rReg, dReg, divisor, tmpSgpr, doRemainder, comment));
            },
            nb::arg("builder"),
            nb::arg("qReg"),
            nb::arg("rReg"),
            nb::arg("dReg"),
            nb::arg("divisor"),
            nb::arg("tmpSgpr"),
            nb::arg("doRemainder") = 1,
            nb::arg("comment")     = "",
            "Static scalar division with remainder (@function)")

        // Multiplication Functions
        .def(
            "vectorStaticMultiply",
            [](StinkyIR&                    self,
               StinkyTofu&                  builder,
               uint32_t                     productReg,
               uint32_t                     operandReg,
               int                          multiplier,
               const std::vector<uint32_t>& tmpSgpr,
               const std::string&           comment) {
                return unwrapExpected(self.vectorStaticMultiply(
                    builder, productReg, operandReg, multiplier, tmpSgpr, comment));
            },
            nb::arg("builder"),
            nb::arg("productReg"),
            nb::arg("operandReg"),
            nb::arg("multiplier"),
            nb::arg("tmpSgpr"),
            nb::arg("comment") = "",
            "Vector multiplication by constant (@function)")

        .def("vectorMultiplyBpe",
             &StinkyIR::vectorMultiplyBpe,
             nb::arg("builder"),
             nb::arg("dstReg"),
             nb::arg("srcReg"),
             nb::arg("bpe"),
             nb::arg("comment") = "",
             "Multiply vector by bytes-per-element (@function)")

        .def("vectorMultiply64Bpe",
             &StinkyIR::vectorMultiply64Bpe,
             nb::arg("builder"),
             nb::arg("dstReg"),
             nb::arg("srcReg"),
             nb::arg("bpe"),
             nb::arg("tmpReg"),
             nb::arg("comment") = "",
             "Multiply 64-bit vector by BPE (@function)")

        .def("scalarMultiplyBpe",
             &StinkyIR::scalarMultiplyBpe,
             nb::arg("builder"),
             nb::arg("dstReg"),
             nb::arg("srcReg"),
             nb::arg("bpe"),
             nb::arg("comment") = "",
             "Multiply scalar by bytes-per-element (@function)")

        // Branching Functions
        .def("BranchIfZero",
             &StinkyIR::BranchIfZero,
             nb::arg("builder"),
             nb::arg("sgprName"),
             nb::arg("tmpSgpr"),
             nb::arg("label"),
             nb::arg("comment") = "",
             "Branch if scalar register is zero (@function)")

        .def("BranchIfNotZero",
             &StinkyIR::BranchIfNotZero,
             nb::arg("builder"),
             nb::arg("sgprName"),
             nb::arg("label"),
             nb::arg("comment") = "",
             "Branch if scalar register is not zero (@function)")

        .def(
            "BranchIfZeroTyped",
            [](StinkyIR&          self,
               StinkyTofu&        builder,
               uint32_t           sgprName,
               const std::string& dataType,
               uint32_t           tmpVgpr,
               const std::string& label,
               const std::string& comment) {
                return unwrapExpected(
                    self.BranchIfZeroTyped(builder, sgprName, dataType, tmpVgpr, label, comment));
            },
            nb::arg("builder"),
            nb::arg("sgprName"),
            nb::arg("dataType"),
            nb::arg("tmpVgpr"),
            nb::arg("label"),
            nb::arg("comment") = "",
            "Branch if scalar register is zero with type support (@function)\n"
            "Supports: 'i32', 'i64', 'f32', 'f64'")

        .def(
            "BranchIfNotZeroTyped",
            [](StinkyIR&          self,
               StinkyTofu&        builder,
               uint32_t           sgprName,
               const std::string& dataType,
               const std::string& label,
               const std::string& comment) {
                return unwrapExpected(
                    self.BranchIfNotZeroTyped(builder, sgprName, dataType, label, comment));
            },
            nb::arg("builder"),
            nb::arg("sgprName"),
            nb::arg("dataType"),
            nb::arg("label"),
            nb::arg("comment") = "",
            "Branch if scalar register is not zero with type support (@function)\n"
            "Supports: 'i32', 'i64', 'f32', 'f64'")

        // Casting Functions
        .def(
            "VSaturateCastInt",
            [](StinkyIR&          self,
               StinkyTofu&        builder,
               uint32_t           valueReg,
               uint32_t           tmpVgpr,
               uint32_t           tmpSgpr,
               int32_t            lowerBound,
               int32_t            upperBound,
               const std::string& saturateType,
               bool               initGpr,
               const std::string& comment) {
                return unwrapExpected(self.VSaturateCastInt(builder,
                                                            valueReg,
                                                            tmpVgpr,
                                                            tmpSgpr,
                                                            lowerBound,
                                                            upperBound,
                                                            saturateType,
                                                            initGpr,
                                                            comment));
            },
            nb::arg("builder"),
            nb::arg("valueReg"),
            nb::arg("tmpVgpr"),
            nb::arg("tmpSgpr"),
            nb::arg("lowerBound"),
            nb::arg("upperBound"),
            nb::arg("saturateType") = "normal",
            nb::arg("initGpr")      = true,
            nb::arg("comment")      = "",
            "Saturate cast integer to bounds (@function)\n"
            "Clamps value to [lowerBound, upperBound] range\n"
            "saturateType: 'normal' (both bounds), 'upper' (max only), 'lower' (min only, not yet "
            "implemented), 'none'")

        // Memory & Synchronization Functions
        .def(
            "DSInit",
            [](StinkyIR&          self,
               StinkyTofu&        builder,
               uint32_t           tmpVgprStart,
               uint32_t           serialVgpr,
               uint32_t           numThreads,
               uint32_t           ldsNumElements,
               int32_t            initValue,
               const std::string& comment) {
                return unwrapExpected(self.DSInit(builder,
                                                  tmpVgprStart,
                                                  serialVgpr,
                                                  numThreads,
                                                  ldsNumElements,
                                                  initValue,
                                                  comment));
            },
            nb::arg("builder"),
            nb::arg("tmpVgprStart"),
            nb::arg("serialVgpr"),
            nb::arg("numThreads"),
            nb::arg("ldsNumElements"),
            nb::arg("initValue") = 0,
            nb::arg("comment")   = "",
            "Initialize LDS (Local Data Share) memory (@function)\n"
            "Synchronizes threads, writes init value to LDS in parallel, and syncs again\n"
            "Requires 2 consecutive VGPRs starting at tmpVgprStart");

    // ========================================================================
    // Bind ArgumentLoader Class
    // ========================================================================

    nb::class_<ArgumentLoader>(m, "ArgumentLoader")
        .def(nb::init<StinkyTofu&>(),
             nb::arg("builder"),
             "Create an ArgumentLoader for efficient kernel argument loading\n\n"
             "ArgumentLoader manages the state and offset tracking needed to load\n"
             "kernel arguments from memory into SGPRs using SLoadBX instructions.\n"
             "It automatically selects the appropriate instruction size and advances\n"
             "the offset after each load.\n\n"
             "Example:\n"
             "  st = StinkyAsmIR([9, 4, 2])\n"
             "  module = st.createIRList('kernel')\n"
             "  loader = ArgumentLoader(st)\n"
             "  # Load 32-bit argument: dst=s0, src=s[2:3], dword=1\n"
             "  module.add(loader.loadKernArg(0, 2, 1, True))\n"
             "  # Load 64-bit argument: dst=s[1:2], src=s[2:3], dword=2\n"
             "  module.add(loader.loadKernArg(1, 2, 2, True))")

        .def("resetOffset", &ArgumentLoader::resetOffset, "Reset the kernel argument offset to 0")

        .def("setOffset",
             &ArgumentLoader::setOffset,
             nb::arg("offset"),
             "Set the kernel argument offset in bytes")

        .def("getOffset",
             &ArgumentLoader::getOffset,
             "Get the current kernel argument offset in bytes")

        .def(
            "loadKernArg",
            [](ArgumentLoader&    self,
               uint32_t           dstSgpr,
               uint32_t           srcAddr,
               nb::object         sgprOffset = nb::none(),
               int                dword      = 1,
               bool               writeSgpr  = true,
               const std::string& comment    = "") {
                std::optional<int> offsetValue;

                // Handle sgprOffset: can be None, int, or hex string
                if(!sgprOffset.is_none())
                {
                    if(nb::isinstance<nb::int_>(sgprOffset))
                    {
                        offsetValue = nb::cast<int>(sgprOffset);
                    }
                    else if(nb::isinstance<nb::str>(sgprOffset))
                    {
                        std::string offsetStr = nb::cast<std::string>(sgprOffset);
                        // Parse hex string (e.g., "0x10" or "16")
                        offsetValue = std::stoi(offsetStr, nullptr, 0);
                    }
                }

                return unwrapExpected(
                    self.loadKernArg(dstSgpr, srcAddr, dword, writeSgpr, offsetValue, comment));
            },
            nb::arg("dstSgpr"),
            nb::arg("srcAddr"),
            nb::arg("sgprOffset") = nb::none(),
            nb::arg("dword")      = 1,
            nb::arg("writeSgpr")  = true,
            nb::arg("comment")    = "",
            nb::rv_policy::reference,
            "Load a kernel argument using SLoadBX instruction\n\n"
            "Generates an SLoadBX instruction (B32/B64/B128/B256/B512) based on\n"
            "the dword count. The instruction loads from kernarg memory at\n"
            "[srcAddr + kernArgOffset] into the destination SGPR(s).\n\n"
            "After loading, kernArgOffset is automatically advanced by (dword * 4)\n"
            "bytes, unless sgprOffset is provided.\n\n"
            "Args:\n"
            "  dstSgpr: Destination SGPR index or name\n"
            "  srcAddr: Source address SGPR pair index (e.g., 2 for s[2:3])\n"
            "  sgprOffset: Optional explicit offset (int or hex string like '0x10')\n"
            "              If provided, kernArgOffset is not auto-advanced\n"
            "  dword: Number of dwords to load (1, 2, 4, 8, 16), default=1\n"
            "  writeSgpr: If False, only advance offset without generating instruction, "
            "default=True\n"
            "  comment: Optional comment (defaults to showing current offset)\n\n"
            "Returns:\n"
            "  List of instructions (empty if writeSgpr=False)\n\n"
            "Example:\n"
            "  loader.loadKernArg(0, 2, dword=4)  # Auto-advance offset\n"
            "  loader.loadKernArg(4, 2, sgprOffset='0x10', dword=2)  # Explicit offset")

        .def(
            "loadAllKernArg",
            [](ArgumentLoader& self,
               uint32_t        sgprStartIndex,
               uint32_t        srcAddr,
               int             numSgprToLoad,
               int             numSgprPreload) {
                return unwrapExpected(
                    self.loadAllKernArg(sgprStartIndex, srcAddr, numSgprToLoad, numSgprPreload));
            },
            nb::arg("sgprStartIndex"),
            nb::arg("srcAddr"),
            nb::arg("numSgprToLoad"),
            nb::arg("numSgprPreload") = 0,
            nb::rv_policy::reference,
            "Load all kernel arguments efficiently\n\n"
            "Loads a contiguous range of SGPRs from kernel arguments, automatically\n"
            "selecting the largest possible SLoadBX instructions (B512 -> B256 -> B128 -> B64 -> "
            "B32)\n"
            "based on SGPR alignment and remaining count.\n\n"
            "This minimizes the number of load instructions by using wider loads when possible.\n\n"
            "Args:\n"
            "  sgprStartIndex: Starting SGPR index to load into\n"
            "  srcAddr: Source address SGPR pair index (64-bit pointer)\n"
            "  numSgprToLoad: Total number of SGPRs to load\n"
            "  numSgprPreload: Number of SGPRs already preloaded (skip these)\n\n"
            "Returns:\n"
            "  List of SLoadBX instructions");

    // ========================================================================
    // Bind Register Helper Functions
    // ========================================================================

    m.def("vgpr",
          &vgpr,
          nb::arg("idx"),
          nb::arg("count") = 1,
          "Create a VGPR (Vector General Purpose Register)\n"
          "Examples: vgpr(0) -> v0, vgpr(0, 4) -> v[0:3]");

    m.def("sgpr",
          &sgpr,
          nb::arg("idx"),
          nb::arg("count") = 1,
          "Create an SGPR (Scalar General Purpose Register)\n"
          "Examples: sgpr(10) -> s10, sgpr(10, 8) -> s[10:17]");

    m.def("acc",
          &acc,
          nb::arg("idx"),
          nb::arg("count") = 1,
          "Create an ACC (Accumulator Register)\n"
          "Examples: acc(0) -> a0, acc(0, 4) -> a[0:3]");

    // ========================================================================
    // Bind Signature/Metadata Classes
    // ========================================================================

    // SignatureValueKind enum
    nb::enum_<SignatureValueKind>(m, "SignatureValueKind")
        .value("SIG_GLOBALBUFFER", SignatureValueKind::SIG_GLOBALBUFFER)
        .value("SIG_VALUE", SignatureValueKind::SIG_VALUE)
        .export_values();

    // BitfieldUnion base class (opaque to Python - only exposed through shared_ptr)
    auto bitfield_class = nb::class_<BitfieldUnion>(m, "BitfieldUnion");
    bitfield_class.def("toString", &BitfieldUnion::toString, "Convert to hex string");
    bitfield_class.def("getValue", &BitfieldUnion::getValue, "Get raw value");
    bitfield_class.def("fieldsDesc", &BitfieldUnion::fieldsDesc, "Get field descriptions");
    bitfield_class.def("desc", &BitfieldUnion::desc, "Get complete description");

    // Factory function for SRD values (returns base class pointer)
    m.def("createSrdUpperValue",
          &createSrdUpperValue,
          nb::arg("isa"),
          "Create appropriate SRD upper value for ISA version\n\n"
          "Args:\n"
          "  isa: ISA version as [major, minor, patch]\n\n"
          "Returns:\n"
          "  BitfieldUnion instance appropriate for the architecture\n\n"
          "Example:\n"
          "  srd = createSrdUpperValue([9, 4, 2])  # For MI300\n"
          "  print(srd.toString())  # Get hex value");

    // SignatureArgument
    nb::class_<SignatureArgument>(m, "SignatureArgument")
        .def(nb::init<int,
                      const std::string&,
                      SignatureValueKind,
                      const std::string&,
                      const std::string&>(),
             nb::arg("offset"),
             nb::arg("name"),
             nb::arg("valueKind"),
             nb::arg("valueType"),
             nb::arg("addrSpaceQual") = "",
             "Create a kernel argument descriptor\n\n"
             "Args:\n"
             "  offset: Byte offset in kernel argument buffer\n"
             "  name: Argument name\n"
             "  valueKind: SIG_GLOBALBUFFER or SIG_VALUE\n"
             "  valueType: Type string (e.g., 'f32', 'u32', 'f64')\n"
             "  addrSpaceQual: Address space qualifier (optional)")
        .def_rw("valueKind", &SignatureArgument::valueKind)
        .def_rw("valueType", &SignatureArgument::valueType)
        .def_rw("name", &SignatureArgument::name)
        .def_rw("offset", &SignatureArgument::offset)
        .def_rw("size", &SignatureArgument::size)
        .def_rw("addrSpaceQual", &SignatureArgument::addrSpaceQual)
        .def("__str__", &SignatureArgument::toString);

    // SignatureKernelDescriptor
    nb::class_<SignatureKernelDescriptor>(m, "SignatureKernelDescriptor")
        .def(nb::init<const std::string&,
                      const std::array<int, 3>&,
                      int,
                      const std::array<int, 3>&,
                      int,
                      int,
                      int,
                      int,
                      bool>(),
             nb::arg("name"),
             nb::arg("isaVersion"),
             nb::arg("groupSegSize"),
             nb::arg("sgprWorkGroup"),
             nb::arg("vgprWorkItem"),
             nb::arg("totalVgprs")      = 0,
             nb::arg("totalAgprs")      = 0,
             nb::arg("totalSgprs")      = 0,
             nb::arg("preloadKernArgs") = false,
             "Create kernel descriptor for .amdhsa_kernel section")
        .def_rw("kernelName", &SignatureKernelDescriptor::kernelName)
        .def_rw("totalVgprs", &SignatureKernelDescriptor::totalVgprs)
        .def_rw("totalAgprs", &SignatureKernelDescriptor::totalAgprs)
        .def_rw("totalSgprs", &SignatureKernelDescriptor::totalSgprs)
        .def_rw("groupSegSize", &SignatureKernelDescriptor::groupSegSize)
        .def_rw("sgprWorkGroup", &SignatureKernelDescriptor::sgprWorkGroup)
        .def_rw("vgprWorkItem", &SignatureKernelDescriptor::vgprWorkItem)
        .def_rw("enablePreloadKernArgs", &SignatureKernelDescriptor::enablePreloadKernArgs)
        .def("setGprs",
             &SignatureKernelDescriptor::setGprs,
             nb::arg("totalVgprs"),
             nb::arg("totalAgprs"),
             nb::arg("totalSgprs"))
        .def("getNextFreeVgpr", &SignatureKernelDescriptor::getNextFreeVgpr)
        .def("getNextFreeSgpr", &SignatureKernelDescriptor::getNextFreeSgpr)
        .def("__str__", &SignatureKernelDescriptor::toString);

    // SignatureCodeMeta
    nb::class_<SignatureCodeMeta>(m, "SignatureCodeMeta")
        .def(nb::init<const std::string&, int, int, int, int, const std::string&, int, int>(),
             nb::arg("name"),
             nb::arg("kernArgsVersion"),
             nb::arg("groupSegSize"),
             nb::arg("flatWgSize"),
             nb::arg("wavefrontSize"),
             nb::arg("codeObjectVersion"),
             nb::arg("totalVgprs") = 0,
             nb::arg("totalSgprs") = 0,
             "Create code object metadata for .amdgpu_metadata section")
        .def_rw("kernelName", &SignatureCodeMeta::kernelName)
        .def_rw("kernArgsVersion", &SignatureCodeMeta::kernArgsVersion)
        .def_rw("groupSegSize", &SignatureCodeMeta::groupSegSize)
        .def_rw("flatWgSize", &SignatureCodeMeta::flatWgSize)
        .def_rw("codeObjectVersion", &SignatureCodeMeta::codeObjectVersion)
        .def_rw("totalVgprs", &SignatureCodeMeta::totalVgprs)
        .def_rw("totalSgprs", &SignatureCodeMeta::totalSgprs)
        .def_rw("wavefrontSize", &SignatureCodeMeta::wavefrontSize)
        .def("setGprs", &SignatureCodeMeta::setGprs, nb::arg("totalVgprs"), nb::arg("totalSgprs"))
        .def("addArg",
             &SignatureCodeMeta::addArg,
             nb::arg("name"),
             nb::arg("kind"),
             nb::arg("type"),
             nb::arg("addrSpaceQual") = "",
             "Add a kernel argument")
        .def("__str__", &SignatureCodeMeta::toString);

    // SignatureBase - Main signature class
    nb::class_<SignatureBase>(m, "SignatureBase")
        .def(nb::init<const std::string&,
                      const std::array<int, 3>&,
                      int,
                      const std::string&,
                      int,
                      const std::array<int, 3>&,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      bool>(),
             nb::arg("kernelName"),
             nb::arg("isaVersion"),
             nb::arg("kernArgsVersion"),
             nb::arg("codeObjectVersion"),
             nb::arg("groupSegmentSize"),
             nb::arg("sgprWorkGroup"),
             nb::arg("vgprWorkItem"),
             nb::arg("flatWorkGroupSize"),
             nb::arg("wavefrontSize")   = 64,
             nb::arg("totalVgprs")      = 0,
             nb::arg("totalAgprs")      = 0,
             nb::arg("totalSgprs")      = 0,
             nb::arg("preloadKernArgs") = false,
             "Create complete kernel signature with metadata\n\n"
             "This is the main class for generating AMD GPU kernel code object metadata.\n"
             "It combines kernel descriptor (.amdhsa_kernel) and code metadata "
             "(.amdgpu_metadata).\n\n"
             "Args:\n"
             "  kernelName: Name of the kernel function\n"
             "  isaVersion: ISA version [major, minor, patch] (e.g., [9, 4, 2])\n"
             "  kernArgsVersion: Kernel arguments version (usually 2)\n"
             "  codeObjectVersion: Code object version ('4' or '5')\n"
             "  groupSegmentSize: LDS size in bytes\n"
             "  sgprWorkGroup: Which workgroup IDs to enable [x, y, z] (0 or 1)\n"
             "  vgprWorkItem: Workitem ID dimension (0=x, 1=y, 2=z)\n"
             "  flatWorkGroupSize: Maximum workgroup size\n"
             "  wavefrontSize: Wavefront size (32 or 64)\n"
             "  totalVgprs: Total VGPRs used\n"
             "  totalAgprs: Total AGPRs used\n"
             "  totalSgprs: Total SGPRs used\n"
             "  preloadKernArgs: Enable kernel argument preloading (GFX12+)\n\n"
             "Example:\n"
             "  sig = SignatureBase(\n"
             "      kernelName='myGemm',\n"
             "      isaVersion=[9, 4, 2],\n"
             "      kernArgsVersion=2,\n"
             "      codeObjectVersion='5',\n"
             "      groupSegmentSize=65536,\n"
             "      sgprWorkGroup=[1, 1, 0],\n"
             "      vgprWorkItem=0,\n"
             "      flatWorkGroupSize=256,\n"
             "      wavefrontSize=64\n"
             "  )")
        .def_rw("kernelDescriptor", &SignatureBase::kernelDescriptor)
        .def_rw("codeMeta", &SignatureBase::codeMeta)
        .def("setGprs",
             &SignatureBase::setGprs,
             nb::arg("totalVgprs"),
             nb::arg("totalAgprs"),
             nb::arg("totalSgprs"),
             "Set final register counts\n\n"
             "Args:\n"
             "  totalVgprs: Total vector GPRs\n"
             "  totalAgprs: Total accumulator GPRs\n"
             "  totalSgprs: Total scalar GPRs")
        .def("addArg",
             &SignatureBase::addArg,
             nb::arg("name"),
             nb::arg("kind"),
             nb::arg("type"),
             nb::arg("addrSpaceQual") = "",
             "Add a kernel argument\n\n"
             "Args:\n"
             "  name: Argument name\n"
             "  kind: SignatureValueKind (SIG_GLOBALBUFFER or SIG_VALUE)\n"
             "  type: Type string (e.g., 'f32', 'f64', 'u32')\n"
             "  addrSpaceQual: Address space ('generic' for buffers)\n\n"
             "Example:\n"
             "  sig.addArg('A', SignatureValueKind.SIG_GLOBALBUFFER, 'f32', 'generic')\n"
             "  sig.addArg('alpha', SignatureValueKind.SIG_VALUE, 'f32')")
        .def("addDescriptionTopic",
             &SignatureBase::addDescriptionTopic,
             nb::arg("text"),
             "Add a description topic (3-line comment block)")
        .def("addDescriptionBlock",
             &SignatureBase::addDescriptionBlock,
             nb::arg("text"),
             "Add a description block (/* comment */)")
        .def("addDescription",
             &SignatureBase::addDescription,
             nb::arg("text"),
             "Add a description line (// comment)")
        .def("clearDescription", &SignatureBase::clearDescription, "Clear all descriptions")
        .def("getNextFreeVgpr", &SignatureBase::getNextFreeVgpr, "Get next free VGPR index")
        .def("getNextFreeSgpr", &SignatureBase::getNextFreeSgpr, "Get next free SGPR index")
        .def("__str__",
             &SignatureBase::toString,
             "Generate complete kernel metadata assembly\n\n"
             "Returns:\n"
             "  String containing .amdhsa_kernel and .amdgpu_metadata sections");

    // Helper functions
    m.def("isaVersionToGfx",
          &isaVersionToGfx,
          nb::arg("isa"),
          "Convert ISA version to gfx string (e.g., [9,4,2] -> 'gfx942')");

    // ========================================================================
    // Bind KernelBody - Complete kernel container
    // ========================================================================

    nb::class_<KernelBody>(m, "KernelBody")
        .def(nb::init<const std::string&>(),
             nb::arg("name"),
             "Create a complete kernel container\n\n"
             "KernelBody combines SignatureBase (metadata) with IRListModule (instructions)\n"
             "to create a self-contained, complete kernel that can be generated with a\n"
             "single toString() call.\n\n"
             "This is the recommended way to build complete kernels in StinkyTofu.\n\n"
             "Args:\n"
             "  name: Kernel name\n\n"
             "Example:\n"
             "  kernel = KernelBody('myKernel')\n"
             "  kernel.addSignature(sig)\n"
             "  kernel.addBody(ir_module)\n"
             "  kernel.setGprs(128, 64, 48)\n"
             "  complete_asm = str(kernel)")
        .def("addSignature",
             &KernelBody::addSignature,
             nb::arg("signature"),
             "Add signature (metadata) to the kernel\n\n"
             "Args:\n"
             "  signature: SignatureBase instance with kernel metadata")
        .def("addBody",
             &KernelBody::addBody,
             nb::arg("body"),
             "Add instruction body to the kernel\n\n"
             "Args:\n"
             "  body: IRListModule containing instructions")
        .def("setGprs",
             &KernelBody::setGprs,
             nb::arg("totalVgprs"),
             nb::arg("totalAgprs"),
             nb::arg("totalSgprs"),
             "Set register counts (updates both signature and tracking)\n\n"
             "Args:\n"
             "  totalVgprs: Total vector GPRs\n"
             "  totalAgprs: Total accumulator GPRs\n"
             "  totalSgprs: Total scalar GPRs")
        .def("getNextFreeVgpr", &KernelBody::getNextFreeVgpr, "Get next available VGPR index")
        .def("getNextFreeSgpr", &KernelBody::getNextFreeSgpr, "Get next available SGPR index")
        .def("getName", &KernelBody::getName, "Get kernel name")
        .def("getSignature", &KernelBody::getSignature, "Get signature (metadata)")
        .def("getBody", &KernelBody::getBody, "Get instruction body (IRListModule)")
        .def("toString",
             &KernelBody::toString,
             nb::arg("emitComments")  = true,
             nb::arg("emitCycleInfo") = false,
             "Generate complete kernel assembly\n\n"
             "Combines signature metadata with instruction body into a complete,\n"
             "assembler-ready kernel.\n\n"
             "Args:\n"
             "  emitComments: Include comments in instruction assembly\n"
             "  emitCycleInfo: Include cycle information\n\n"
             "Returns:\n"
             "  Complete kernel assembly string")
        .def(
            "__str__",
            [](const KernelBody& kb) { return kb.toString(); },
            "Generate complete kernel assembly (same as toString())");

    // ========================================================================
    // Test functions for Expected<T> error handling
    // ========================================================================

    m.def(
        "_testExpectedSuccess",
        [](int value) { return unwrapExpected(test::testExpectedSuccess(value)); },
        nb::arg("value"),
        "Test function: Returns value * 2 (should succeed)");

    m.def(
        "_testExpectedError",
        [](const std::string& errorMsg) {
            return unwrapExpected(test::testExpectedError(errorMsg));
        },
        nb::arg("errorMsg"),
        "Test function: Always returns error (should raise RuntimeError)");

    m.def(
        "_testExpectedVectorSuccess",
        []() { return unwrapExpected(test::testExpectedVectorSuccess()); },
        "Test function: Returns [1,2,3,4,5] (should succeed)");

    m.def(
        "_testExpectedVectorError",
        []() { return unwrapExpected(test::testExpectedVectorError()); },
        "Test function: Always returns error (should raise RuntimeError)");
}
