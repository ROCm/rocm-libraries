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
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"
#include "instruction/mfma.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmModule.hpp"
#include "ir/rocisa/AllHwMappings.hpp"
#include "isa/ArchHelper.hpp"
#include "stinkytofu.hpp"

namespace nb = nanobind;

namespace
{
    using namespace rocisa;
    using namespace stinkytofu;

    /**
     * @brief Check if an instruction reads the SCC register
     */
    bool doesReadSCC(const Instruction* inst)
    {
        if(dynamic_cast<const SCSelectB32*>(inst))
        {
            return true;
        }
        return false;
    }

    /**
     * @brief Check if an instruction writes the SCC register
     */
    bool doesWriteSCC(const Instruction* inst)
    {
        if(dynamic_cast<const SCmpEQI32*>(inst) || dynamic_cast<const SCmpEQU32*>(inst)
           || dynamic_cast<const SSubU32*>(inst) || dynamic_cast<const SAddU32*>(inst)
           || dynamic_cast<const SAddCU32*>(inst) || dynamic_cast<const SSubBU32*>(inst))
        {
            return true;
        }
        return false;
    }

    // Helper functions to convert rocisa modifiers to stinkytofu modifiers
    stinkytofu::DSModifiers convertDSModifiers(const rocisa::DSModifiers& rocMod)
    {
        return stinkytofu::DSModifiers(
            rocMod.na, rocMod.offset, rocMod.offset0, rocMod.offset1, rocMod.gds);
    }

    stinkytofu::FLATModifiers convertFLATModifiers(const rocisa::FLATModifiers& rocMod)
    {
        return stinkytofu::FLATModifiers(
            rocMod.offset12, rocMod.glc, rocMod.slc, rocMod.lds, rocMod.isStore);
    }

    stinkytofu::MUBUFModifiers convertMUBUFModifiers(const rocisa::MUBUFModifiers& rocMod)
    {
        return stinkytofu::MUBUFModifiers(rocMod.offen,
                                          rocMod.offset12,
                                          rocMod.glc,
                                          rocMod.slc,
                                          rocMod.nt,
                                          rocMod.lds,
                                          rocMod.isStore);
    }

    stinkytofu::SMEMModifiers convertSMEMModifiers(const rocisa::SMEMModifiers& rocMod)
    {
        return stinkytofu::SMEMModifiers(rocMod.glc, rocMod.nv, rocMod.offset);
    }

} // anonymous namespace

namespace stinkytofu
{
    /**
     * @brief Convert a rocisa::Container (or InstructionInput) to StinkyRegister
     *
     * This function takes a rocisa::Container pointer and converts it to a
     * StinkyRegister. It handles RegisterContainer types by extracting the
     * register type, index, and number of registers.
     *
     * @param container Pointer to rocisa::Container to convert
     * @return StinkyRegister representing the container, or invalid register if conversion fails
     */
    StinkyRegister toStinkyRegister(const rocisa::Container* container)
    {
        if(const rocisa::RegisterContainer* regCont
           = dynamic_cast<const rocisa::RegisterContainer*>(container))
        {
            // Convert string regType to RegType enum
            RegType        regType = stringToRegType(regCont->regType);
            StinkyRegister reg{regType, regCont->regIdx, regCont->regNum};

            // Capture symbolic register name if available
            // In rocisa, the symbolic name includes the type prefix and all offsets
            // (e.g., "vgprLocalWriteAddrA+0" or "vgprValuA_X0_I0+4")
            if(regCont->regName.has_value())
            {
                // regName->toString() includes the base name and all offsets
                std::string fullName = regCont->getCompleteRegNameWithType();
                reg.setSymbolicName(fullName);
            }

            return reg;
        }
        return StinkyRegister{};
    }

    /**
     * @brief Convert a rocisa::InstructionInput to StinkyRegister
     *
     * This overload handles InstructionInput variants which can contain:
     * - shared_ptr<Container> (converted via RegisterContainer)
     * - int literals
     * - double literals
     * - string literals
     *
     * @param input The InstructionInput variant to convert
     * @return StinkyRegister representing the input value
     */
    StinkyRegister toStinkyRegister(const InstructionInput& input)
    {
        if(auto pptr = std::get_if<std::shared_ptr<rocisa::Container>>(&input))
        {
            if(auto regContainer = std::dynamic_pointer_cast<RegisterContainer>(*pptr))
            {
                // Convert string regType to RegType enum
                RegType        regType = stringToRegType(regContainer->regType);
                StinkyRegister reg{regType, regContainer->regIdx, regContainer->regNum};

                // Capture symbolic register name if available
                // In rocisa, the symbolic name includes the type prefix and all offsets
                // (e.g., "vgprLocalWriteAddrA+0" or "vgprValuA_X0_I0+4")
                if(regContainer->regName.has_value())
                {
                    // regName->toString() includes the base name and all offsets
                    std::string fullName = regContainer->getCompleteRegNameWithType();
                    reg.setSymbolicName(fullName);
                }

                return reg;
            }
        }
        else if(const int* literalInt = std::get_if<int>(&input))
        {
            return StinkyRegister(*literalInt);
        }
        else if(const double* literalDouble = std::get_if<double>(&input))
        {
            return StinkyRegister(*literalDouble);
        }
        else if(const std::string* literalString = std::get_if<std::string>(&input))
        {
            return StinkyRegister(*literalString);
        }

        return StinkyRegister{};
    }

    std::shared_ptr<StinkyAsmModule> toStinkyTofuModule(const rocisa::Module& module,
                                                        std::array<int, 3>    arch,
                                                        const std::string&    moduleName)
    {
        // Get GfxArchID from architecture array
        GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        StinkyAsmModule stinkyAsmModule(moduleName, arch);
        IRList&         insts = stinkyAsmModule.getIRList();

        // Create IRBuilder for lower-level instruction creation
        StinkyInstIRBuilder irBuilder(insts, archId);

        // Process each item
        for(auto itemShared : module.flatitems())
        {
            rocisa::Item* item = itemShared.get();
            // Handle labels
            if(rocisa::Label* rocLabel = dynamic_cast<rocisa::Label*>(item))
            {
                irBuilder.createStinkyLabel(insts.end(), rocLabel->getLabelName());
                continue;
            }

            // Handle instructions
            rocisa::Instruction* inst = dynamic_cast<rocisa::Instruction*>(item);
            if(inst == nullptr)
            {
                // TODO: Remove this once we have a better way to handle non-instruction items
                std::cout << "Skipping non-instruction item: " << item->toString() << std::endl;
                continue;
            }

            // Get the type of the rocisa instruction
            std::type_index rocisaTy = std::type_index(typeid(*inst));

            // Try to get the hardware instruction descriptor
            const HwInstDesc* hwInstDesc = getRocisaToMCID(rocisaTy, archId);

            StinkyInstruction* stinkyInst = nullptr;

            if(hwInstDesc != nullptr)
            {
                // Direct mapping exists - create instruction directly
                stinkyInst = irBuilder.createStinkyInstBefore(insts.end(), hwInstDesc);
            }
            else
            {
                // Need conversion function
                ConvertRocisaToHwInstFunc convFn = getConvertRocisaToHwInstFunc(rocisaTy, archId);
                if(convFn == nullptr)
                {
                    // Unhandled instruction type - skip
                    continue;
                }

                // Call conversion function
                std::vector<StinkyInstruction*> stinkyInsts = convFn(*inst, irBuilder, insts);

                if(stinkyInsts.empty())
                {
                    continue;
                }

                // For now, only handle single instruction conversions
                // TODO: Handle multiple instructions
                stinkyInst = stinkyInsts[0];
            }

            if(stinkyInst == nullptr)
            {
                continue;
            }

            // Add source and destination registers
            // Use set to avoid duplicates
            std::set<StinkyRegister> uniqueSrcRegs, uniqueDstRegs;

            // Add destination registers
            for(const InstructionInput& dst : inst->getDstParams())
            {
                StinkyRegister reg = toStinkyRegister(dst);
                if(uniqueDstRegs.find(reg) == uniqueDstRegs.end())
                {
                    uniqueDstRegs.insert(reg);
                    stinkyInst->addDestReg(reg);
                }
            }

            // Add source registers
            for(const InstructionInput& src : inst->getSrcParams())
            {
                StinkyRegister reg = toStinkyRegister(src);
                if(reg.isValid() && uniqueSrcRegs.find(reg) == uniqueSrcRegs.end())
                {
                    uniqueSrcRegs.insert(reg);
                    stinkyInst->addSrcReg(reg);
                }
            }

            // Add SCC register if needed
            if(doesReadSCC(inst))
            {
                stinkyInst->addSrcReg(StinkyRegister::getSCCRegister());
            }

            if(doesWriteSCC(inst))
            {
                stinkyInst->addDestReg(StinkyRegister::getSCCRegister());
            }

            // Copy modifiers from rocisa instruction to StinkyInstruction
            // DS (Local Memory) modifiers
            if(auto dsLoad = dynamic_cast<const DSLoadInstruction*>(inst))
            {
                if(dsLoad->ds.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::DSModifiers>(
                        convertDSModifiers(dsLoad->ds.value()));
                }
            }
            else if(auto dsStore = dynamic_cast<const DSStoreInstruction*>(inst))
            {
                if(dsStore->ds.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::DSModifiers>(
                        convertDSModifiers(dsStore->ds.value()));
                }
            }
            // FLAT (Flat Memory) modifiers
            else if(auto flatRead = dynamic_cast<const FLATReadInstruction*>(inst))
            {
                if(flatRead->flat.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::FLATModifiers>(
                        convertFLATModifiers(flatRead->flat.value()));
                }
            }
            else if(auto flatStore = dynamic_cast<const FLATStoreInstruction*>(inst))
            {
                if(flatStore->flat.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::FLATModifiers>(
                        convertFLATModifiers(flatStore->flat.value()));
                }
            }
            // MUBUF (Buffer Memory) modifiers
            else if(auto mubufRead = dynamic_cast<const MUBUFReadInstruction*>(inst))
            {
                if(mubufRead->mubuf.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::MUBUFModifiers>(
                        convertMUBUFModifiers(mubufRead->mubuf.value()));
                }
            }
            else if(auto mubufStore = dynamic_cast<const MUBUFStoreInstruction*>(inst))
            {
                if(mubufStore->mubuf.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::MUBUFModifiers>(
                        convertMUBUFModifiers(mubufStore->mubuf.value()));
                }
            }
            // SMEM (Scalar Memory) modifiers
            else if(auto smemLoad = dynamic_cast<const SMemLoadInstruction*>(inst))
            {
                if(smemLoad->smem.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::SMEMModifiers>(
                        convertSMEMModifiers(smemLoad->smem.value()));
                }
            }
            else if(auto smemStore = dynamic_cast<const SMemStoreInstruction*>(inst))
            {
                if(smemStore->smem.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::SMEMModifiers>(
                        convertSMEMModifiers(smemStore->smem.value()));
                }
            }
            else if(auto smemAtomic = dynamic_cast<const SMemAtomicDecInstruction*>(inst))
            {
                if(smemAtomic->smem.has_value())
                {
                    stinkyInst->addModifier<stinkytofu::SMEMModifiers>(
                        convertSMEMModifiers(smemAtomic->smem.value()));
                }
            }

            // Copy comment from rocisa instruction to StinkyInstruction
            if(!inst->comment.empty())
            {
                stinkyInst->addModifier<CommentData>(CommentData{inst->comment});
            }

            // Handle branch instructions
            if(rocisa::BranchInstruction* branchInst
               = dynamic_cast<rocisa::BranchInstruction*>(inst))
            {
                stinkyInst->addModifier<LabelData>(
                    LabelData{Modifier::Type::LABEL_NAME, branchInst->labelName});
            }
        }

        // Iterate through the instructions in the IRList and add them to the module
        for(IRList::iterator it = insts.begin(); it != insts.end(); ++it)
        {
            StinkyInstruction* inst = cast<StinkyInstruction>(it.getNodePtr());
            stinkyAsmModule.add(inst);
        }

        return std::make_shared<StinkyAsmModule>(std::move(stinkyAsmModule));
    }

} // namespace stinkytofu

/**
 * @brief Initialize StinkyTofu Python bindings
 *
 * This function binds the rocisa to StinkyTofu utilities to Python, allowing
 * Python code to convert rocisa to StinkyTofu IR.
 *
 * @param m The nanobind module to add bindings to
 */
void init_stinkytofu(nb::module_ m)
{
    // Bind toStinkyRegister for Container pointer
    m.def("toStinkyRegister",
          nb::overload_cast<const rocisa::Container*>(&stinkytofu::toStinkyRegister),
          nb::arg("container"),
          "Convert a rocisa Container to a StinkyRegister");

    // Bind toStinkyRegister for InstructionInput
    m.def("toStinkyRegister",
          nb::overload_cast<const InstructionInput&>(&stinkytofu::toStinkyRegister),
          nb::arg("input"),
          "Convert a rocisa InstructionInput to a StinkyRegister");

    // Bind toStinkyTofuModule with wrapper to handle tuple/list conversion
    m.def(
        "toStinkyTofuModule",
        [](const rocisa::Module& module, nb::object arch_obj, const std::string& moduleName) {
            // Convert Python sequence (tuple or list) to std::array
            if(!nb::isinstance<nb::sequence>(arch_obj))
            {
                throw std::invalid_argument(
                    "arch must be a tuple or list of 3 integers [major, minor, stepping]");
            }

            auto arch_seq = nb::cast<nb::sequence>(arch_obj);
            if(nb::len(arch_seq) != 3)
            {
                throw std::invalid_argument(
                    "arch must have exactly 3 elements [major, minor, stepping]");
            }

            std::array<int, 3> archArray = {
                nb::cast<int>(arch_seq[0]), nb::cast<int>(arch_seq[1]), nb::cast<int>(arch_seq[2])};
            return stinkytofu::toStinkyTofuModule(module, archArray, moduleName);
        },
        nb::arg("module"),
        nb::arg("arch"),
        nb::arg("moduleName") = "",
        nb::rv_policy::reference,
        "Convert a rocisa.Module to a StinkyTofu StinkyAsmModule. "
        "This function flattens the module hierarchy, converts labels and instructions, "
        "and handles register mapping. "
        "Args: module (rocisa.Module), arch (tuple/list of 3 ints [major, minor, stepping]), "
        "moduleName (str, optional)");
}
