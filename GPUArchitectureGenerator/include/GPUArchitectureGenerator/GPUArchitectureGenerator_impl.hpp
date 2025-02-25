
#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
                                   rocRoller::GPUInstructionInfo const&    instruction_info)
    {
        auto [iter, _] = GPUArchitectures.try_emplace(isaVersion, isaVersion);

        std::string instruction = instruction_info.getInstruction();
        bool        isBranch    = instruction_info.isBranch()
                        || (BranchInstructions.find(instruction) != BranchInstructions.end()
                            && (std::find(BranchInstructions.at(instruction).begin(),
                                          BranchInstructions.at(instruction).end(),
                                          isaVersion)
                                != BranchInstructions.at(instruction).end()));
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

    void FillArchitectures(std::string const& hipcc)
    {
        GPUArchitectures.clear();

        for(const auto& isaVersion : SupportedArchitectures)
        {
            for(const auto& query : AssemblerQueries)
            {
                if(TryAssembler(
                       hipcc, isaVersion, std::get<0>(query.second), std::get<1>(query.second)))
                {
                    AddCapability(isaVersion, query.first, 0);
                }
            }
            for(const auto& cap : ArchSpecificCaps)
            {
                if(std::find(cap.second.begin(), cap.second.end(), isaVersion) != cap.second.end())
                {
                    AddCapability(isaVersion, cap.first, 0);
                }
            }
            for(const auto& cap : PredicateCaps)
            {
                if(cap.second(isaVersion))
                {
                    AddCapability(isaVersion, cap.first, 0);
                }
            }

            if(TryAssembler(hipcc, isaVersion, "s_waitcnt vmcnt(63)", ""))
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
            AddCapability(isaVersion, rocRoller::GPUCapability::MaxLgkmcnt, 15);
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

            for(const auto& info : InstructionInfos)
            {
                if(std::find(std::get<0>(info).begin(), std::get<0>(info).end(), isaVersion)
                   != std::get<0>(info).end())
                {
                    for(const auto& instruction : std::get<1>(info))
                    {
                        AddInstructionInfo(isaVersion, instruction);
                    }
                }
            }

            for(const auto& group : GroupedInstructionInfos)
            {
                if(std::find(std::get<0>(group).begin(), std::get<0>(group).end(), isaVersion)
                   != std::get<0>(group).end())
                {
                    for(const auto& instruction : std::get<0>(std::get<1>(group)))
                    {
                        AddInstructionInfo(
                            isaVersion,
                            rocRoller::GPUInstructionInfo(instruction,
                                                          std::get<1>(std::get<1>(group)),
                                                          std::get<2>(std::get<1>(group)),
                                                          -1,
                                                          false,
                                                          false,
                                                          std::get<3>(std::get<1>(group))));
                    }
                }
            }

            for(const auto& instruction : BranchInstructions)
            {
                if(std::find(instruction.second.begin(), instruction.second.end(), isaVersion)
                   != instruction.second.end())
                {
                    if(!GPUArchitectures[isaVersion].HasInstructionInfo(instruction.first))
                    {
                        AddInstructionInfo(isaVersion,
                                           rocRoller::GPUInstructionInfo(
                                               instruction.first, -1, {}, -1, false, true));
                    }
                }
            }

            for(const auto& instruction : ImplicitReadInstructions)
            {
                if(std::find(instruction.second.begin(), instruction.second.end(), isaVersion)
                   != instruction.second.end())
                {
                    if(!GPUArchitectures[isaVersion].HasInstructionInfo(instruction.first))
                    {
                        AddInstructionInfo(isaVersion,
                                           rocRoller::GPUInstructionInfo(
                                               instruction.first, -1, {}, -1, true, false));
                    }
                }
            }
        }
    }

    void FillArchitectures()
    {
        FillArchitectures(DEFAULT_ASSEMBLER);
    }

    void GenerateFile(std::string const& fileName, bool asYAML)
    {
        std::ofstream outputFile;
        outputFile.open(fileName);
        if(asYAML)
        {
            outputFile << rocRoller::GPUArchitecture::writeYaml(GPUArchitectures) << std::endl;
        }
        else
        {
            // As msgpack
            outputFile << rocRoller::GPUArchitecture::writeMsgpack(GPUArchitectures) << std::endl;
        }
        outputFile.close();
    }
}
