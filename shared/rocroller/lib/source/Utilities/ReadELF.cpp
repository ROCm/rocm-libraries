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
#ifdef ROCROLLER_USE_LLD
#include <llvm/BinaryFormat/AMDGPUMetadataVerifier.h>
#include <llvm/BinaryFormat/MsgPackDocument.h>
#include <llvm/Object/ELF.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/AMDGPUMetadata.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
using namespace llvm;
using namespace llvm::object;
#endif

#include <rocRoller/Utilities/Error.hpp>

using namespace rocRoller;

std::string rocRoller::readMetaDataFromCodeObject(std::string const& fileName)
{
    std::string yaml;
#ifdef ROCROLLER_USE_LLD
    auto maybeBuffer = MemoryBuffer::getFile(fileName);
    if(!maybeBuffer)
    {
        Throw<FatalError>("Error reading file: ", maybeBuffer.getError().message());
    }

    auto maybeELFObject = ObjectFile::createELFObjectFile(maybeBuffer->get()->getMemBufferRef());
    if(!maybeELFObject)
    {
        Throw<FatalError>("Error creating ELF object file: ", toString(maybeELFObject.takeError()));
    }

    auto elf64LE = dyn_cast<ELF64LEObjectFile>(maybeELFObject->get());
    if(!elf64LE)
    {
        Throw<FatalError>("Not an ELF file: ", fileName);
    }
    auto elf = elf64LE->getELFFile();

    auto maybeSections = elf.sections();
    if(!maybeSections)
    {
        Throw<FatalError>("Can not read ELF sections.");
    }

    auto sections = *maybeSections;
    for(const auto& section : sections)
    {
        if(section.sh_type != ELF::SHT_NOTE)
            continue;

        auto err = llvm::Error::success();
        for(auto note : elf.notes(section, err))
        {
            if(note.getName() != "AMDGPU")
                continue;

            auto type = note.getType();
            if(type != ELF::NT_AMDGPU_METADATA)
                continue;

#if LLVM_VERSION_MAJOR >= 18
            // See https://llvm.org/docs/AMDGPUUsage.html#note-records for note re: alignment
            auto desc = note.getDesc(4);
#else
            auto desc = note.getDesc();
#endif
            auto msg = StringRef(reinterpret_cast<const char*>(desc.data()), desc.size());

            msgpack::Document meta;
            if(!meta.readFromBlob(msg, false))
                continue;

            AMDGPU::HSAMD::V3::MetadataVerifier Verifier(true);
            if(!Verifier.verify(meta.getRoot()))
            {
                Throw<FatalError>("Invalid AMDGPU::HSAMD metadata");
            }

            raw_string_ostream os(yaml);
            meta.toYAML(os);

            // Currently we're only looking at the first note...
            break;
        }
    }
#endif

    return yaml;
}
