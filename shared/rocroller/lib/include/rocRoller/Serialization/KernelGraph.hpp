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

#ifdef ROCROLLER_USE_LLVM
#include <llvm/ObjectYAML/YAML.h>
#endif

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/ControlGraph.hpp>
#include <rocRoller/Serialization/ControlToCoordinateMapper.hpp>
#include <rocRoller/Serialization/CoordinateGraph.hpp>

namespace rocRoller
{
    namespace Serialization
    {

        template <typename IO>
        struct MappingTraits<KernelGraph::KernelGraph, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::KernelGraph& kern)
            {
                iot::mapRequired(io, "control", kern.control);
                iot::mapRequired(io, "coordinates", kern.coordinates);
                iot::mapRequired(io, "mapper", kern.mapper);
            }

            static void mapping(IO& io, KernelGraph::KernelGraph& kern, EmptyContext& ctx)
            {
                mapping(io, kern);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::KernelGraphPtr, IO, Context>
            : public SharedPointerMappingTraits<KernelGraph::KernelGraphPtr, IO, Context, true>
        {
        };

    }
}
