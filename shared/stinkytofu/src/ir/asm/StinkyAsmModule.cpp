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
#include "ir/asm/StinkyAsmModule.hpp"
#include "ir/asm/StinkyAsmEmitter.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmPrinter.hpp"
#include "isa/ArchHelper.hpp"

#include <iostream>
#include <sstream>

namespace stinkytofu
{
    struct StinkyAsmModule::Impl
    {
        std::string                     name;
        std::array<int, 3>              arch;
        std::vector<StinkyInstruction*> instructions;

        // The IRList is owned by a Function/BasicBlock
        // We maintain a pointer to it for compatibility
        std::unique_ptr<Function> function;
        BasicBlock*               basicBlock;

        Impl(const std::string& name, const std::array<int, 3>& arch)
            : name(name)
            , arch(arch)
            , function(std::make_unique<Function>(name))
            , basicBlock(nullptr)
        {
            // Create a single BasicBlock to hold all instructions
            basicBlock = function->createBasicBlock("entry");
            function->setEntryBlock(basicBlock);
        }

        ~Impl()
        {
            // Function destructor will clean up BasicBlocks and their IRLists
            // The StinkyInstructions in the IRList will be deleted by the Function
        }
    };

    StinkyAsmModule::StinkyAsmModule(const std::string& name, const std::array<int, 3>& arch)
        : pImpl(std::make_unique<Impl>(name, arch))
    {
    }

    StinkyAsmModule::~StinkyAsmModule() = default;

    StinkyAsmModule::StinkyAsmModule(StinkyAsmModule&&) noexcept            = default;
    StinkyAsmModule& StinkyAsmModule::operator=(StinkyAsmModule&&) noexcept = default;

    std::string StinkyAsmModule::getName() const
    {
        return pImpl->name;
    }

    std::array<int, 3> StinkyAsmModule::getArch() const
    {
        return pImpl->arch;
    }

    void StinkyAsmModule::add(StinkyInstruction* inst)
    {
        if(inst)
        {
            pImpl->instructions.push_back(inst);
        }
    }

    void StinkyAsmModule::add(const std::vector<StinkyInstruction*>& insts)
    {
        for(StinkyInstruction* inst : insts)
        {
            add(inst);
        }
    }

    const std::vector<StinkyInstruction*>& StinkyAsmModule::getInstructions() const
    {
        return pImpl->instructions;
    }

    size_t StinkyAsmModule::size() const
    {
        return pImpl->instructions.size();
    }

    std::string StinkyAsmModule::toString() const
    {
        std::ostringstream oss;
        oss << "StinkyAsmModule: " << pImpl->name;
        oss << " (arch: " << pImpl->arch[0] << "." << pImpl->arch[1] << "." << pImpl->arch[2]
            << ")\n";
        oss << stinkytofu::toString(getIRList());

        return oss.str();
    }

    std::string StinkyAsmModule::emitAssembly() const
    {
        // Configure the emitter with default options
        stinkytofu::AsmEmitterOptions options;
        options.emitComments     = true;
        options.emitCycleInfo    = false;
        options.indent           = 0;
        options.emitBlankLines   = false;
        options.useSymbolicNames = true; // Enable symbolic register names

        stinkytofu::StinkyAsmEmitter emitter(options);
        return emitter.emit(getIRList());
    }

    void StinkyAsmModule::runOptimizationPipeline()
    {
        // TODO: Implement optimization pipeline
        std::cout << "Running optimization pipeline on " << pImpl->name << std::endl;
    }

    IRList& StinkyAsmModule::getIRList()
    {
        return pImpl->basicBlock->getIR();
    }

    const IRList& StinkyAsmModule::getIRList() const
    {
        return pImpl->basicBlock->getIR();
    }

} // namespace stinkytofu
