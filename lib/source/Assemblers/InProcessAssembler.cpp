/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <lld/Common/Driver.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectFileInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/MCAsmParser.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/MCTargetOptions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Option/Option.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

#include <rocRoller/Assemblers/InProcessAssembler.hpp>

#include <llvm/Config/llvm-config.h>
#if LLVM_VERSION_MAJOR >= 18
LLD_HAS_DRIVER(elf)
#endif

namespace rocRoller
{

    InProcessAssembler::InProcessAssembler()
    {
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUDisassembler();
        LLVMInitializeAMDGPUAsmParser();
        LLVMInitializeAMDGPUAsmPrinter();
    }

    InProcessAssembler::~InProcessAssembler() {}

    void InProcessAssembler::assemble(const char* machineCode,
                                      const char* target,
                                      const char* featureString,
                                      const char* output)
    {
        const std::string triple = "amdgcn-amd-amdhsa";

        // Create a Source Manager with the machine_code string
        std::unique_ptr<llvm::MemoryBuffer> Buffer
            = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(machineCode), "Input Assembly");
        llvm::SourceMgr SrcMgr;
        SrcMgr.AddNewSourceBuffer(std::move(Buffer), llvm::SMLoc());

        // Get the target for an AMD GPU
        std::string         error;
        const llvm::Target* TheTarget = llvm::TargetRegistry::lookupTarget(triple, error);
        if(!TheTarget)
            throw std::runtime_error("Error creating a target for " + triple);

        // Create a Machine Code Register Info instance
        std::unique_ptr<llvm::MCRegisterInfo> MRI(TheTarget->createMCRegInfo(triple));

        // Create a Machine Code Assembly Info instance
        llvm::MCTargetOptions            mcOptions;
        std::unique_ptr<llvm::MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, triple, mcOptions));

        // Create an output stream
        std::error_code EC;
        auto FDOS = std::make_unique<llvm::raw_fd_ostream>(output, EC, llvm::sys::fs::OF_None);

        // Create a Machine Code Object File Info instance
        std::unique_ptr<llvm::MCObjectFileInfo> MOFI(new llvm::MCObjectFileInfo());

        // Create a Machine Code Target Info instance
        std::unique_ptr<llvm::MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(
            triple, std::string(target), std::string(featureString)));

        // Create a Context
        llvm::MCContext Ctx(llvm::Triple(triple), MAI.get(), MRI.get(), STI.get(), &SrcMgr);
        Ctx.setObjectFileInfo(MOFI.get());
        MOFI->initMCObjectFileInfo(Ctx, true);

        // Create a Machine Code Streamer instance
        std::unique_ptr<llvm::MCStreamer>  Str;
        std::unique_ptr<llvm::MCInstrInfo> MCII(TheTarget->createMCInstrInfo());

        auto                     BOS = std::make_unique<llvm::buffer_ostream>(*FDOS);
        llvm::raw_pwrite_stream* Out = BOS.get();

        llvm::MCCodeEmitter*  CE = TheTarget->createMCCodeEmitter(*MCII, Ctx);
        llvm::MCTargetOptions Backend_Options;
        llvm::MCAsmBackend*   MAB = TheTarget->createMCAsmBackend(*STI, *MRI, Backend_Options);
        Str.reset(TheTarget->createMCObjectStreamer(llvm::Triple(triple),
                                                    Ctx,
                                                    std::unique_ptr<llvm::MCAsmBackend>(MAB),
                                                    MAB->createObjectWriter(*Out),
                                                    std::unique_ptr<llvm::MCCodeEmitter>(CE),
                                                    *STI
#if LLVM_VERSION_MAJOR < 19
                                                    ,
                                                    true,
                                                    true,
                                                    true
#endif
                                                    ));

        // Create a Parser instance
        std::unique_ptr<llvm::MCAsmParser> Parser(createMCAsmParser(SrcMgr, Ctx, *Str.get(), *MAI));

        llvm::MCTargetOptions                    Options;
        std::unique_ptr<llvm::MCTargetAsmParser> TAP(
            TheTarget->createMCAsmParser(*STI, *Parser, *MCII, Options));
        Parser->setTargetParser(*TAP.get());

        // Run the parser, which will generate machine code to the output stream.
        bool result = Parser->Run(true);
        if(result)
            throw std::runtime_error("Error assembling machine code");
    }

    void InProcessAssembler::link(const char* input, const char* output)
    {
        llvm::opt::ArgStringList LLDArgs;
        LLDArgs.push_back("lld");
        LLDArgs.push_back(input);
        LLDArgs.push_back("--threads=1");
        LLDArgs.push_back("-shared");
        LLDArgs.push_back("-o");
        LLDArgs.push_back(output);

        bool linkResult = lld::elf::link(LLDArgs, llvm::outs(), llvm::outs(), false, false);
        if(!linkResult)
            throw std::runtime_error("Error linking");
    }

}
