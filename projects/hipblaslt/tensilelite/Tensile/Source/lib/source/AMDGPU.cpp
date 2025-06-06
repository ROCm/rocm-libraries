/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>

namespace TensileLite
{
    static const std::map<AMDGPU::Processor, std::vector<int>>& NonStandardCUMap()
    {
        static const std::map<AMDGPU::Processor, std::vector<int>> NonStandardCU = {
            {AMDGPU::Processor::gfx90a, {104}}, {AMDGPU::Processor::gfx942, {20, 38, 64, 80, 228}}};
        return NonStandardCU;
    }

    TENSILE_API std::string AMDGPU::type() const
    {
        return Type();
    }

    TENSILE_API AMDGPU::AMDGPU() {}

    TENSILE_API AMDGPU::AMDGPU(AMDGPU::Processor p, int cus, std::string const& name)
        : processor(p)
        , computeUnitCount(cus)
        , deviceName(name)
        , skDynamicGrid(getSKDynamicGrid())
        , skDynamicWGM(getSKDynamicWGM())
        , skMaxCUs(getSKMaxCUs())
        , skGridMultiplier(getSKGridMultiplier())
        , skFixedGrid(getSKFixedGrid())
        , skFullTiles(getSKFullTiles())
    {
    }

    TENSILE_API AMDGPU::~AMDGPU() = default;

    TENSILE_API bool AMDGPU::isStandardCU() const
    {
        // return the result if we already tested it.
        if(isStandardCUs != -1)
            return (isStandardCUs == 1);

        // assume current device is a standard cu device.
        isStandardCUs = 1;

        auto mapIter = NonStandardCUMap().find(processor);
        if(mapIter != NonStandardCUMap().end())
        {
            auto list = mapIter->second;
            // current device is a speicalized cu device.
            if(std::count(list.begin(), list.end(), computeUnitCount) > 0)
                isStandardCUs = 0;
        }

        return (isStandardCUs == 1);
    }

    TENSILE_API bool AMDGPU::runsKernelTargeting(AMDGPU::Processor other) const
    {
        if(other > this->processor)
            return false;
        if(other == this->processor)
            return true;

        if(other == Processor::gfx803)
            return false;

        if(other == Processor::gfx900)
            return true;

        return false;
    }

    std::ostream& operator<<(std::ostream& stream, AMDGPU::Processor p)
    {
        stream << AMDGPU::toString(p);
        return stream;
    }

    TENSILE_API std::string AMDGPU::description() const
    {
        std::ostringstream rv;

        rv << deviceName << "(" << computeUnitCount << "-CU " << processor << ")";

        return rv.str();
    }

    TENSILE_API std::ostream& operator<<(std::ostream& stream, AMDGPU g)
    {
        return stream << g.description();
    }
} // namespace TensileLite
