/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#include <algorithm>
#include <filesystem>
#include <iostream> // todo: don't use iostream.
#include <sys/types.h>
#include <yaml-cpp/yaml.h>

#include "gfx/InstDefDSL.hpp"

namespace stinkytofu
{
    std::string getReducedFilename(const char* filename, unsigned keepDepth)
    {
        // only preserve the filename and the last `keepDepth` directories
        namespace fs = std::filesystem;
        fs::path p(filename);

        // Always include the filename itself
        fs::path result = p.filename();

        // Append up to `keepDepth` last parent directories (from innermost outward)
        auto parent = p.parent_path();
        for(unsigned i = 0; i < keepDepth && !parent.empty(); ++i)
        {
            result = parent.filename() / result;
            parent = parent.parent_path();
        }

        return result.string();
    }

    bool GpuArch::add(std::unique_ptr<GfxInstDef> inst)
    {
        assert(!finalized && "GpuArch already finalized");
        if(added.find(inst->name) != added.end())
        {
            std::cerr << "Error: Instruction " << inst->name << " already added!\n";
            error = true;
            return false;
        }
        added.insert(std::make_pair(inst->name, inst.get()));
        instructions.push_back(std::move(inst));
        return true;
    }

    bool GpuArch::erase(const std::string& name)
    {
        assert(!finalized && "GpuArch already finalized");
        auto it = std::remove_if(
            instructions.begin(), instructions.end(), [&](const std::unique_ptr<GfxInstDef>& inst) {
                return inst->name == name;
            });

        if(it == instructions.end())
        {
            std::cerr << "Error: Instruction " << name << " to erase is not found!\n";
            error = true;
            return false;
        }

        instructions.erase(it, instructions.end());
        added.erase(name);
        return true;
    }

    // yaml format:
    // - target: [9,4,2]
    // - instructions:
    //   - default_cycle: 4
    //   - cycle:
    //     - ds_store_b8: 8
    //     - ds_store_b16: 8
    //     - ...
    //   - latency:
    //     - ds_load_u8: 48
    //     - ...
    bool loadInstructionInfoFromYaml(GpuArch::InstructionInfo& info, const YAML::Node& instNode)
    {
        static_assert(sizeof(uint16_t) == sizeof(HwInstDesc::issue), "default_cycle type mismatch");
        static_assert(sizeof(uint16_t) == sizeof(HwInstDesc::issue), "cycle type mismatch");
        static_assert(sizeof(uint16_t) == sizeof(HwInstDesc::latency), "latency type mismatch");

        bool success = true;

        auto getValueAndCheckRange = [&](const YAML::Node& node) -> uint16_t {
            // Note that yaml will throw an exception if the parsed value is not in
            // the expected range of the type, but we want to handle the error ourselves.
            int value = node.as<int>();
            if(value < 0 || value > UINT16_MAX)
            {
                std::cerr << "Error: Value " << value << " is out of range for uint16_t\n";
                success = false;
                return 0;
            }
            return static_cast<uint16_t>(value);
        };

        for(const auto& item : instNode)
        {
            if(item["default_cycle"])
            {
                info.default_cycle = getValueAndCheckRange(item["default_cycle"]);
            }
            else if(item["cycle"])
            {
                for(const auto& c : item["cycle"])
                    for(auto it = c.begin(); it != c.end(); ++it)
                        info.cycle.push_back(std::make_pair(it->first.as<std::string>(),
                                                            getValueAndCheckRange(it->second)));
            }
            else if(item["latency"])
            {
                for(const auto& l : item["latency"])
                    for(auto it = l.begin(); it != l.end(); ++it)
                        info.latency.push_back(std::make_pair(it->first.as<std::string>(),
                                                              getValueAndCheckRange(it->second)));
            }
        }
        return success;
    }

    GfxInstDef* GpuArch::getInst(const std::string& name)
    {
        auto it = added.find(name);
        if(it != added.end())
            return it->second;

        std::cerr << "Error: Instruction " << name << " not found!\n";
        error = true;
        return nullptr;
    }

    void GpuArch::updateCycleAndLatency(const InstructionInfo& info)
    {
        // set default cycle to all instructions
        for(const auto& inst : instructions)
        {
            // only update the issue and latency if the instruction is not already set
            if(inst->hwInstDesc.issue == 0)
                inst->hwInstDesc.issue = info.default_cycle;
            if(inst->hwInstDesc.latency == 0)
                inst->hwInstDesc.latency = info.default_cycle;
        }

        // override the issue and latency for each instruction if specified in the yaml file
        for(const auto& cycle : info.cycle)
            if(GfxInstDef* inst = getInst(cycle.first))
                inst->hwInstDesc.issue = cycle.second;

        for(const auto& latency : info.latency)
            if(GfxInstDef* inst = getInst(latency.first))
                inst->hwInstDesc.latency = latency.second;
    }

    bool GpuArch::loadHardwareDataFromYaml(const std::string& yamlPath)
    {
        assert(!finalized && "GpuArch already finalized");
        std::filesystem::path yamlFile = std::filesystem::path(yamlPath);

        if(!std::filesystem::exists(yamlFile))
        {
            std::cerr << "Error: YAML file " << yamlFile << " not found!\n";
            error = true;
            return false;
        }

        std::cerr << "Load hardware yaml data in " << yamlFile << "\n";

        YAML::Node root = YAML::LoadFile(yamlPath);

        bool success = true;
        for(const auto& entry : root)
        {
            // if the entry is an instructions node, load the instruction info
            if(entry["instructions"])
            {
                InstructionInfo info;
                if(!loadInstructionInfoFromYaml(info, entry["instructions"]))
                    success = false;

                updateCycleAndLatency(info);
            }
        }

        error |= !success;

        return success;
    }

}
