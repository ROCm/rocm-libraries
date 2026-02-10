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

#include <cstdint>
#include <ostream>

#include <iostream> // TODO: don't use iostream.

#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"

namespace stinkytofu
{
    //----------------------------------------------------------------------
    // StinkyRegister implementation
    //----------------------------------------------------------------------
    void StinkyRegister::dump(std::ostream& out, const std::string& prefix) const
    {
        out << prefix;
        RegisterPrinter printer(out);
        printer.print(*this);
    }

    void StinkyRegister::dump() const
    {
        dump(std::cerr);
    }

    //----------------------------------------------------------------------
    // StinkyInstruction implementation
    //----------------------------------------------------------------------
    void StinkyInstruction::dump(bool printDetails, const std::string& prefix) const
    {
        dump(std::cerr, printDetails, prefix);
    }

    void StinkyInstruction::dump(std::ostream&      out,
                                 bool               printDetails,
                                 const std::string& prefix) const
    {
        AsmPrinter printer(out);
        if(!prefix.empty())
            out << prefix;
        printer.print(*this);
    }

    //----------------------------------------------------------------------
    // StinkyInstIRBuilder implementation
    //----------------------------------------------------------------------
    IRBuilder::ID StinkyInstIRBuilder::ID = &StinkyInstIRBuilder::ID;

    StinkyInstruction* StinkyInstIRBuilder::createStinkyLabel(IRList::iterator   pos,
                                                              const std::string& label)
    {
        static const HwInstDesc labelMCID{
            GFX::LABEL, GFX::LABEL, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};

        StinkyInstruction* labelInst = createStinkyInstBefore(pos, &labelMCID);
        labelInst->addModifier<LabelData>(LabelData{Modifier::Type::LABEL_NAME, label});
        return labelInst;
    }

    void StinkyInstIRBuilder::erase(StinkyInstruction* stinkyInst)
    {
        // Clean up use-def chains before deletion
        stinkyInst->unlinkFromSources();
        stinkyInst->unlinkFromUsers();

        irlist->remove(stinkyInst);
        delete stinkyInst;
    }

    //----------------------------------------------------------------------
    // StinkyInstruction Use-Def Chain Maintenance Implementation
    //----------------------------------------------------------------------

    void StinkyInstruction::dump() const
    {
        dump(std::cerr, true);
    }

    void StinkyInstruction::setSrcRegs(const std::vector<StinkyRegister>& newSrcRegs)
    {
        // Remove old use-def links
        unlinkFromSources();

        // Set new source registers
        srcRegs = newSrcRegs;

        // Establish new use-def links
        updateSourcesForInst();
    }

    void StinkyInstruction::setDestRegs(const std::vector<StinkyRegister>& newDestRegs)
    {
        // Remove old use-def links
        unlinkFromUsers();

        // Set new destination registers
        destRegs = newDestRegs;

        // Establish new use-def links for future users
        updateUsersForInst();
    }

    void StinkyInstruction::addSrcRegAndUpdateUD(const StinkyRegister& srcReg)
    {
        srcRegs.push_back(srcReg);

        // Find the defining instruction for this register by scanning backwards
        if(!srcReg.isRegister())
            return;

        // Scan backwards using getPrev() to find the most recent definition
        IRBase* prevNode = this->getPrev();
        while(prevNode)
        {
            if(prevNode->getType() == IRBase::IRType::StinkyTofu)
            {
                auto* candidateInst = dyn_cast<StinkyInstruction>(prevNode);
                if(!candidateInst)
                {
                    continue;
                }
                // Check if this instruction defines the register we're looking for
                for(const auto& destReg : candidateInst->destRegs)
                {
                    if(destReg.isRegister() && srcReg.isOverlap(destReg))
                    {
                        // Found the definition - link it
                        sources.push_back(candidateInst);
                        candidateInst->users.push_back(this);
                        return; // Only link to most recent definition
                    }
                }
            }

            prevNode = prevNode->getPrev();
        }
    }

    void StinkyInstruction::addDestRegAndUpdateUD(const StinkyRegister& destReg)
    {
        destRegs.push_back(destReg);

        // Find future uses of this register by scanning forwards
        if(!destReg.isRegister())
            return;

        // Scan forwards using getNext() to find uses
        IRBase* nextNode = this->getNext();
        while(nextNode)
        {
            if(nextNode->getType() == IRBase::IRType::StinkyTofu)
            {
                auto* candidateInst = dyn_cast<StinkyInstruction>(nextNode);
                if(!candidateInst)
                {
                    continue;
                }

                // Check if this instruction uses the register we just defined
                for(const auto& srcReg : candidateInst->srcRegs)
                {
                    if(srcReg.isRegister() && destReg.isOverlap(srcReg))
                    {
                        // Found a use - link it
                        candidateInst->sources.push_back(this);
                        users.push_back(candidateInst);
                        break; // Move to next instruction after finding one use
                    }
                }

                // Stop scanning if we hit a new definition of this register
                for(const auto& otherDest : candidateInst->destRegs)
                {
                    if(otherDest.isRegister() && destReg.isOverlap(otherDest))
                    {
                        return; // Stop at next definition
                    }
                }
            }
            nextNode = nextNode->getNext();
        }
    }

    void StinkyInstruction::updateSourcesForInst()
    {
        // Clear existing source links
        sources.clear();

        // For each source register, find its definition by scanning backwards
        for(const auto& srcReg : srcRegs)
        {
            if(!srcReg.isRegister())
                continue;

            // Handle consecutive registers (e.g., v[0:3] means v0, v1, v2, v3)
            for(unsigned i = 0; i < srcReg.reg.num; ++i)
            {
                StinkyRegister individualReg(srcReg.reg.type, srcReg.reg.idx + i, 1);

                // Scan backwards using getPrev() to find the most recent definition
                IRBase* prevNode = this->getPrev();
                while(prevNode)
                {
                    if(prevNode->getType() == IRBase::IRType::StinkyTofu)
                    {
                        auto* candidateInst = dyn_cast<StinkyInstruction>(prevNode);
                        if(!candidateInst)
                        {
                            continue;
                        }

                        for(const auto& destReg : candidateInst->destRegs)
                        {
                            if(destReg.isRegister() && individualReg.isOverlap(destReg))
                            {
                                // Found definition - add bidirectional link
                                sources.push_back(candidateInst);
                                candidateInst->users.push_back(this);
                                goto next_individual_reg; // Break out of while loop
                            }
                        }
                    }

                    prevNode = prevNode->getPrev();
                }
            next_individual_reg:;
            }
        }
    }

    void StinkyInstruction::updateUsersForInst()
    {
        // Clear existing user links
        users.clear();

        // For each destination register, find its uses by scanning forwards
        for(const auto& destReg : destRegs)
        {
            if(!destReg.isRegister())
                continue;

            // Scan forwards using getNext() to find uses
            IRBase* nextNode = this->getNext();
            while(nextNode)
            {
                if(nextNode->getType() == IRBase::IRType::StinkyTofu)
                {
                    auto* candidateInst = dyn_cast<StinkyInstruction>(nextNode);
                    if(!candidateInst)
                    {
                        continue;
                    }

                    // Check if this instruction uses our destination register
                    for(const auto& srcReg : candidateInst->srcRegs)
                    {
                        if(srcReg.isRegister() && destReg.isOverlap(srcReg))
                        {
                            // Found a use - add bidirectional link
                            candidateInst->sources.push_back(this);
                            users.push_back(candidateInst);
                            break;
                        }
                    }

                    // Stop if we hit a redefinition of this register
                    for(const auto& otherDest : candidateInst->destRegs)
                    {
                        if(otherDest.isRegister() && destReg.isOverlap(otherDest))
                        {
                            goto next_dest_reg; // Break out of while loop
                        }
                    }
                }
                nextNode = nextNode->getNext();
            }
        next_dest_reg:;
        }
    }

    void StinkyInstruction::unlinkFromSources()
    {
        // Remove this instruction from all its sources' user lists
        for(StinkyInstruction* source : sources)
        {
            auto& sourceUsers = source->users;
            sourceUsers.erase(std::remove(sourceUsers.begin(), sourceUsers.end(), this),
                              sourceUsers.end());
        }
        sources.clear();
    }

    void StinkyInstruction::unlinkFromUsers()
    {
        // Remove this instruction from all its users' source lists
        for(StinkyInstruction* user : users)
        {
            auto& userSources = user->sources;
            userSources.erase(std::remove(userSources.begin(), userSources.end(), this),
                              userSources.end());
        }
        users.clear();
    }

    static const HwInstDesc* getMCIDTable(GfxArchID arch)
    {
        const auto* archInfo
            = ArchHelper::getInstance().getArchInfo(arch); // Ensure ArchHelper is initialized
        return archInfo->getMCIDTable(); // Call the virtual method to ensure proper initialization
    }

    const HwInstDesc* getMCIDByUOp(GFX unifiedOpcode, GfxArchID arch)
    {
        IsaOpcode isaOpcode = GFX::INVALID;

        const auto* archInfo
            = ArchHelper::getInstance().getArchInfo(arch); // Ensure ArchHelper is initialized
        isaOpcode = archInfo->getIsaOpcode(unifiedOpcode);

        if(isaOpcode == GFX::INVALID)
        {
            // Return nullptr if instruction not supported on this architecture
            // Callers use this to probe architecture support
            return nullptr;
        }
        return &getMCIDTable(arch)[isaOpcode];
    }

    const HwInstDesc* getMCIDByIsaOp(IsaOpcode isaOpcode, GfxArchID arch)
    {
        return &getMCIDTable(arch)[isaOpcode];
    }

    uint16_t getMnemonicToIsaOpcode(const std::string& mnemonic, GfxArchID arch)
    {
        auto get = [&](const std::unordered_map<std::string, uint16_t>& map,
                       const std::string&                               mnemonic) -> uint16_t {
            auto it = map.find(mnemonic);
#ifndef NDEBUG
            if(it == map.end())
            {
                std::cerr << "Error: No ISA opcode found for mnemonic " << mnemonic << " in arch "
                          << getArchName(arch) << "\n";
                return GFX::INVALID;
            }
#endif
            return it->second;
        };

        const auto* archInfo
            = ArchHelper::getInstance().getArchInfo(arch); // Ensure ArchHelper is initialized
        return get(archInfo->getMnemonicToIsaOpcodeMap(), mnemonic);
    }

    //----------------------------------------------------------------------
    // StinkyInstruction utilities
    //----------------------------------------------------------------------

    uint32_t getBytesPerGlobalLoad(const StinkyInstruction& inst)
    {
        auto op = inst.getUnifiedOpcode();

        // FIXME: Should we add metadata for the buffer load inst in tablegen to
        //        query the bytes per load?
        switch(op)
        {
        case GFX::buffer_load_ubyte:
            return 1;
        case GFX::buffer_load_ubyte_d16:
        case GFX::buffer_load_ubyte_d16_hi:
        case GFX::buffer_load_ushort:
        case GFX::buffer_load_short_d16_hi:
            return 2;
        case GFX::buffer_load_dword:
            return 4;
        case GFX::buffer_load_dwordx2:
            return 8;
        case GFX::buffer_load_dwordx4:
            return 16;

        default:
            assert("Calling getBytesPerGlobalLoad but input is not GlobalRead");
            return 0;
        }
    }
} // namespace stinkytofu
