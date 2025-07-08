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

#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>

namespace rocRoller
{
    namespace Serialization
    {

        template <typename IO>
        struct CustomMappingTraits<std::map<int, int>, IO>
            : public DefaultCustomMappingTraits<std::map<int, int>, IO, false, true>
        {
        };

        template <typename IO>
        struct MappingTraits<KernelGraph::UnrollColouring, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::UnrollColouring& colouring)
            {
                iot::mapRequired(io, "separators", colouring.separators);
                // iot::mapRequired(io, "coordinates", kern.coordinates);
                // iot::mapRequired(io, "mapper", kern.mapper);
            }

            static void mapping(IO& io, KernelGraph::KernelGraph& kern, EmptyContext& ctx)
            {
                mapping(io, kern);
            }
        };
    }
}
