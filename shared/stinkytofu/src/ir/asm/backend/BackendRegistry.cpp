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
#include "ir/asm/BackendRegistry.hpp"

#include "isa/ArchHelper.hpp"

#include <unordered_map>

namespace stinkytofu
{
    struct BackendRegistry::Registry
    {
        // Map from arch key to vector of pipeline factories
        std::unordered_map<std::string, std::vector<PipelineFactory>> pipelines;
    };

    BackendRegistry::Registry& BackendRegistry::getRegistry()
    {
        static Registry registry;
        return registry;
    }

    void BackendRegistry::addArchPipeline(const std::array<int, 3>& arch,
                                          PipelineConfigBuilder     builder,
                                          const std::string&        groupName)
    {
        auto& reg = getRegistry();
        reg.pipelines[makeArchKey(arch)].push_back(PipelineFactory{builder, groupName});
    }

    void BackendRegistry::setArchPipelines(const std::array<int, 3>&           arch,
                                           const std::vector<PipelineFactory>& factories)
    {
        auto& reg                        = getRegistry();
        reg.pipelines[makeArchKey(arch)] = factories;
    }

    std::vector<BackendRegistry::PipelineFactory>
        BackendRegistry::getPipelineFactories(const std::array<int, 3>& arch)
    {
        auto& reg = getRegistry();
        auto  it  = reg.pipelines.find(makeArchKey(arch));
        if(it != reg.pipelines.end())
        {
            return it->second; // Return copy of pipeline factories
        }
        return {};
    }

    bool BackendRegistry::hasPipelines(const std::array<int, 3>& arch)
    {
        auto& reg = getRegistry();
        auto  it  = reg.pipelines.find(makeArchKey(arch));
        return it != reg.pipelines.end() && !it->second.empty();
    }

    size_t BackendRegistry::getPipelineCount(const std::array<int, 3>& arch)
    {
        auto& reg = getRegistry();
        auto  it  = reg.pipelines.find(makeArchKey(arch));
        return (it != reg.pipelines.end()) ? it->second.size() : 0;
    }

    void BackendRegistry::clear()
    {
        auto& reg = getRegistry();
        reg.pipelines.clear();
    }

    void BackendRegistry::clearArch(const std::array<int, 3>& arch)
    {
        auto& reg = getRegistry();
        reg.pipelines.erase(makeArchKey(arch));
    }

    std::string BackendRegistry::makeArchKey(const std::array<int, 3>& arch)
    {
        return getArchName(getGfxArchID(arch[0], arch[1], arch[2]));
    }

} // namespace stinkytofu
