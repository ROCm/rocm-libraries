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

#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "GPUArchitectureGenerator_defs.hpp"

namespace GPUArchitectureGenerator
{
    inline void AddCapability(rocRoller::GPUArchitectureTarget const& isaVersion,
                              rocRoller::GPUCapability const&         capability,
                              int                                     value)
    {
        auto [iter, _] = GPUArchitectures.try_emplace(isaVersion, isaVersion);
        iter->second.AddCapability(capability, value);
    }

    inline int HasCapability(rocRoller::GPUArchitectureTarget const& isaVersion,
                             rocRoller::GPUCapability const&         capability)
    {
        return GPUArchitectures[isaVersion].HasCapability(capability);
    }

    inline void AddInstructionInfo(rocRoller::GPUArchitectureTarget const& isaVersion,
                                   rocRoller::GPUInstructionInfo const&    instruction_info,
                                   std::shared_ptr<amdisa::IsaSpec> const& spec,
                                   std::map<std::string, amdisa::Instruction> const& alias_lookup)
    {
        auto [iter, _] = GPUArchitectures.try_emplace(isaVersion, isaVersion);

        std::string instruction = instruction_info.getInstruction();
        bool        isBranch    = instruction_info.isBranch();

        if(spec)
        {
            if(instruction != "v_fma_mix_f32"
               && //TODO: Remove this when MRISA supports instruction, or instruction is not needed anymore
               alias_lookup.find(instruction) == alias_lookup.end())
            {

                std::cerr << "Instruction must be in MRISA: " << isaVersion << std::endl
                          << instruction << std::endl;
                std::abort();
            }
        }

        if(!isBranch && spec && alias_lookup.contains(instruction))
        {
            isBranch = alias_lookup.at(instruction).is_branch;
        }

        bool isImplicit
            = instruction_info.hasImplicitAccess()
              || (ImplicitReadInstructions.find(instruction) != ImplicitReadInstructions.end()
                  && (std::find(ImplicitReadInstructions.at(instruction).begin(),
                                ImplicitReadInstructions.at(instruction).end(),
                                isaVersion)
                      != ImplicitReadInstructions.at(instruction).end()));

        if(instruction_info.isBranch() != isBranch
           || instruction_info.hasImplicitAccess() != isImplicit)
        {
            iter->second.AddInstructionInfo(
                rocRoller::GPUInstructionInfo(instruction,
                                              instruction_info.getWaitCount(),
                                              instruction_info.getWaitQueues(),
                                              instruction_info.getLatency(),
                                              isImplicit,
                                              isBranch));
        }
        else
        {
            iter->second.AddInstructionInfo(instruction_info);
        }
    }

    inline std::tuple<int, std::string> Execute(std::string const& command)
    {
        std::array<char, 128> buffer;
        std::string           result;
        FILE*                 pipe(popen(command.c_str(), "r"));
        if(!pipe)
        {
            return {-1, ""};
        }
        while(fgets(buffer.data(), buffer.size(), pipe) != nullptr)
        {
            result += buffer.data();
        }
        int ret_code = pclose(pipe);

        return {ret_code, result};
    }

    bool CheckAssembler()
    {
        return CheckAssembler(DEFAULT_ASSEMBLER);
    }

    bool CheckAssembler(std::string const& hipcc)
    {
        std::string cmd    = hipcc + " --version 2>&1";
        auto        result = Execute(cmd);

        if(std::get<0>(result) != 0)
        {
            fprintf(stderr, "Error:\n%s\n", std::get<1>(result).c_str());
            return false;
        }

        return true;
    }

    bool TryAssembler(std::string const&                      hipcc,
                      rocRoller::GPUArchitectureTarget const& isaVersion,
                      std::string const&                      query,
                      std::string const&                      options)
    {

        auto tmpFolderTemplate
            = std::filesystem::temp_directory_path() / "rocroller_GPUArchitectureGenerator-XXXXXX";
        char* tmpFolder = mkdtemp(const_cast<char*>(tmpFolderTemplate.c_str()));

        std::string asmFileName   = std::string(tmpFolder) + "/tmp.asm";
        std::string ouputFileName = std::string(tmpFolder) + "/tmp.o";
        std::string cmd
            = hipcc
              + " -Wno-unused-command-line-argument -c -x assembler -target amdgcn-amdhsa -mcpu="
              + isaVersion.toString() + " -mcode-object-version=5 " + options + " -o "
              + ouputFileName + " " + asmFileName + " 2>&1";

        std::ofstream asmFile;
        try
        {
            asmFile.open(asmFileName);
            asmFile << query;
            asmFile.close();
        }
        catch(std::exception& exc)
        {
            std::filesystem::remove_all(tmpFolder);
            return false;
        }

        auto result = Execute(cmd);

        std::filesystem::remove_all(tmpFolder);
        return std::get<0>(result) == 0 && std::get<1>(result).length() == 0;
    }

    std::map<std::string, amdisa::Instruction> buildISALookup(amdisa::IsaSpec const& spec)
    {
        std::map<std::string, amdisa::Instruction> alias_lookup;
        for(auto const& specInstruction : spec.instructions)
        {
            auto instruction = specInstruction.name;
            for(auto const& alias : specInstruction.aliased_names)
            {
                auto lowerAlias = alias;
                std::transform(lowerAlias.begin(), lowerAlias.end(), lowerAlias.begin(), ::tolower);
                alias_lookup[lowerAlias] = specInstruction;
            }
            auto lowerInstruction = instruction;
            std::transform(lowerInstruction.begin(),
                           lowerInstruction.end(),
                           lowerInstruction.begin(),
                           ::tolower);
            alias_lookup[lowerInstruction] = specInstruction;
        }
        return alias_lookup;
    }

    std::map<std::string, std::tuple<amdisa::IsaSpec, std::map<std::string, amdisa::Instruction>>>
        LoadSpecs(std::string const& xmlDir)
    {
        std::map<std::string,
                 std::tuple<amdisa::IsaSpec, std::map<std::string, amdisa::Instruction>>>
            retval;

        if(!xmlDir.empty())
        {
            for(auto const& file : std::filesystem::directory_iterator(xmlDir))
            {
                if(!file.is_regular_file() || file.path().extension() != ".xml")
                {
                    continue;
                }
                amdisa::IsaSpec spec;

                std::string err_msg;
                if(!amdisa::IsaXmlReader::ReadSpec(file.path(), spec, err_msg))
                {
                    std::cerr << "Error reading ISA XML: " << file.path() << std::endl
                              << err_msg << std::endl;
                    std::abort();
                }

                retval[spec.architecture.name] = {spec, buildISALookup(spec)};
            }
        }

        return retval;
    }

    void FillArchitectures(std::string const& hipcc, std::string const& xmlDir)
    {
        GPUArchitectures.clear();

        auto specMap = LoadSpecs(xmlDir);

        for(auto const& isaVersion : rocRoller::SupportedArchitectures)
        {
            std::shared_ptr<amdisa::IsaSpec>           spec;
            std::map<std::string, amdisa::Instruction> alias_lookup;

            std::string archName = isaVersion.name();
            if(specMap.find(archName) != specMap.end())
            {
                spec         = std::make_shared<amdisa::IsaSpec>(std::get<0>(specMap.at(archName)));
                alias_lookup = std::get<1>(specMap.at(archName));
            }

            for(auto const& query : AssemblerQueries)
            {
                if(TryAssembler(
                       hipcc, isaVersion, std::get<0>(query.second), std::get<1>(query.second)))
                {
                    AddCapability(isaVersion, query.first, 0);
                }
            }
            for(auto const& cap : ArchSpecificCaps)
            {
                if(std::find(cap.second.begin(), cap.second.end(), isaVersion) != cap.second.end())
                {
                    AddCapability(isaVersion, cap.first, 0);
                }
            }
            for(auto const& cap : PredicateCaps)
            {
                if(cap.second(isaVersion))
                {
                    AddCapability(isaVersion, cap.first, 0);
                }
            }

            if(TryAssembler(hipcc, isaVersion, "s_waitcnt_loadcnt 63", ""))
            {
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxVmcnt, 63);
            }
            else if(TryAssembler(hipcc, isaVersion, "s_waitcnt vmcnt(63)", ""))
            {
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxVmcnt, 63);
            }
            else if(TryAssembler(hipcc, isaVersion, "s_waitcnt vmcnt(15)", ""))
            {
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxVmcnt, 15);
            }
            else
            {
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxVmcnt, 0);
            }

            if(TryAssembler(hipcc, isaVersion, "s_waitcnt_kmcnt 31", ""))
            {
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxLgkmcnt, 31);
            }
            else if(TryAssembler(hipcc, isaVersion, "s_waitcnt lgkmcnt(15)", ""))
            {
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxLgkmcnt, 15);
            }
            else
            {
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxLgkmcnt, 0);
            }
            AddCapability(isaVersion, rocRoller::GPUCapability::MaxExpcnt, 7);
            AddCapability(isaVersion, rocRoller::GPUCapability::SupportedSource, 0);

            if(HasCapability(isaVersion, rocRoller::GPUCapability::HasWave32))
                AddCapability(isaVersion, rocRoller::GPUCapability::DefaultWavefrontSize, 32);
            else
                AddCapability(isaVersion, rocRoller::GPUCapability::DefaultWavefrontSize, 64);

            if(isaVersion.toString().starts_with("gfx95"))
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxLdsSize, 160 * (1 << 10));
            else
                AddCapability(isaVersion, rocRoller::GPUCapability::MaxLdsSize, 1 << 16);

            for(auto const& info : InstructionInfos)
            {
                if(std::find(std::get<0>(info).begin(), std::get<0>(info).end(), isaVersion)
                   != std::get<0>(info).end())
                {
                    for(auto const& instruction : std::get<1>(info))
                    {
                        AddInstructionInfo(isaVersion, instruction, spec, alias_lookup);
                    }
                }
            }

            for(auto const& group : GroupedInstructionInfos)
            {
                if(std::find(std::get<0>(group).begin(), std::get<0>(group).end(), isaVersion)
                   != std::get<0>(group).end())
                {
                    for(auto const& instruction : std::get<0>(std::get<1>(group)))
                    {
                        AddInstructionInfo(
                            isaVersion,
                            rocRoller::GPUInstructionInfo(instruction,
                                                          std::get<1>(std::get<1>(group)),
                                                          std::get<2>(std::get<1>(group)),
                                                          -1,
                                                          false,
                                                          false,
                                                          std::get<3>(std::get<1>(group))),
                            spec,
                            alias_lookup);
                    }
                }
            }

            for(auto const& instruction : ImplicitReadInstructions)
            {
                if(std::find(instruction.second.begin(), instruction.second.end(), isaVersion)
                   != instruction.second.end())
                {
                    if(!GPUArchitectures[isaVersion].HasInstructionInfo(instruction.first))
                    {
                        AddInstructionInfo(isaVersion,
                                           rocRoller::GPUInstructionInfo(
                                               instruction.first, -1, {}, -1, true, false),
                                           spec,
                                           alias_lookup);
                    }
                }
            }

            if(spec)
            {
                for(auto const& specInstruction : alias_lookup)
                {
                    auto converted
                        = ConvertSpecInstruction(specInstruction.second, specInstruction.first);
                    if(!GPUArchitectures[isaVersion].HasInstructionInfo(converted.getInstruction()))
                    {
                        AddInstructionInfo(isaVersion, converted, spec, alias_lookup);
                    }
                }
            }
        }
    }

    void FillArchitectures()
    {
        FillArchitectures(DEFAULT_ASSEMBLER, "");
    }

    void LoadYamls(std::vector<std::string> const& yamlIns)
    {
        for(const auto& yamlIn : yamlIns)
        {
            std::map<rocRoller::GPUArchitectureTarget, rocRoller::GPUArchitecture> splitArch
                = rocRoller::GPUArchitecture::readYaml(yamlIn);
            for(const auto& gpuArchitecture : splitArch)
            {
                GPUArchitectures[gpuArchitecture.first] = gpuArchitecture.second;
            }
        }
    }

    void GenerateFile(std::string const& fileName, bool asYAML, bool splitYAML)
    {
        if(asYAML)
        {
            if(!splitYAML)
            {
                std::ofstream outputFile;
                outputFile.open(fileName);
                outputFile << rocRoller::GPUArchitecture::writeYaml(GPUArchitectures) << std::endl;
                outputFile.close();
            }
            else
            {
                std::filesystem::path argPath(fileName);

                for(const auto& gpuArchitecture : GPUArchitectures)
                {
                    std::map<rocRoller::GPUArchitectureTarget, rocRoller::GPUArchitecture>
                        splitArch;
                    splitArch[gpuArchitecture.first] = gpuArchitecture.second;

                    std::string newFilename
                        = argPath.parent_path()
                          / (argPath.stem().string() + "_" + gpuArchitecture.first.toString()
                             + argPath.extension().string());
                    std::ranges::replace(newFilename, ':', '_');
                    std::ranges::replace(newFilename, '+', 't');

                    std::ofstream outputFile;
                    outputFile.open(newFilename);
                    outputFile << rocRoller::GPUArchitecture::writeYaml(splitArch) << std::endl;
                    outputFile.close();
                }
            }
        }
        else
        {
            // As msgpack
            std::ofstream outputFile;
            outputFile.open(fileName);
            outputFile << rocRoller::GPUArchitecture::writeMsgpack(GPUArchitectures) << std::endl;
            outputFile.close();
        }
    }

    rocRoller::GPUInstructionInfo ConvertSpecInstruction(const amdisa::Instruction& instruction,
                                                         const std::string&         name)
    {
        std::string inst_name = name.empty() ? instruction.name : name;
        std::transform(inst_name.begin(), inst_name.end(), inst_name.begin(), ::tolower);
        return rocRoller::GPUInstructionInfo(inst_name, -1, {}, 0, false, instruction.is_branch, 0);
    }
}
