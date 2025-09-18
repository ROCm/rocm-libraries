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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Command.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/Enum.hpp>
#include <rocRoller/Serialization/Expression.hpp>
#include <rocRoller/Serialization/KernelGraph.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Version.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<AssemblyKernelArgument, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, AssemblyKernelArgument& arg)
            {
                iot::mapRequired(io, ".name", arg.name);
                iot::mapRequired(io, ".size", arg.size);
                iot::mapRequired(io, ".offset", arg.offset);

                iot::mapOptional(io, ".expression", arg.expression);
                iot::mapOptional(io, ".variableType", arg.variableType);

                std::string valueKind = "by_value";
                if(arg.variableType.isGlobalPointer())
                {
                    valueKind = "global_buffer";

                    std::string addressSpace = "global";
                    iot::mapRequired(io, ".address_space", addressSpace);
                    iot::mapRequired(io, ".actual_access", arg.dataDirection);
                }

                iot::mapRequired(io, ".value_kind", valueKind);
            }

            static void mapping(IO& io, AssemblyKernelArgument& arg, EmptyContext& ctx)
            {
                mapping(io, arg);
            }
        };

        template <typename IO>
        struct MappingTraits<AssemblyKernel, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, AssemblyKernel& kern)
            {
                iot::mapRequired(io, ".name", kern.m_kernelName);

                //
                // Serialized but not de-serialized
                //
                std::string symbol = kern.kernelName() + ".kd";
                iot::mapRequired(io, ".symbol", symbol);

                int                       kernarg_segment_size;
                int                       group_segment_fixed_size;
                int                       private_segment_fixed_size;
                int                       kernarg_segment_align;
                int                       sgpr_count;
                int                       vgpr_count;
                int                       agpr_count;
                int                       max_flat_workgroup_size;
                std::vector<unsigned int> workgroupSize;

                if(iot::outputting(io))
                {
                    kernarg_segment_size       = kern.kernarg_segment_size();
                    group_segment_fixed_size   = kern.group_segment_fixed_size();
                    private_segment_fixed_size = kern.private_segment_fixed_size();
                    kernarg_segment_align      = kern.kernarg_segment_align();
                    sgpr_count                 = kern.sgpr_count();
                    vgpr_count                 = kern.vgpr_count();
                    agpr_count                 = kern.agpr_count();
                    max_flat_workgroup_size    = kern.max_flat_workgroup_size();
                    workgroupSize              = {
                        kern.m_workgroupSize[0],
                        kern.m_workgroupSize[1],
                        kern.m_workgroupSize[2],
                    };
                }

                iot::mapRequired(io, ".kernarg_segment_size", kernarg_segment_size);
                iot::mapRequired(io, ".group_segment_fixed_size", group_segment_fixed_size);
                iot::mapRequired(io, ".private_segment_fixed_size", private_segment_fixed_size);
                iot::mapRequired(io, ".kernarg_segment_align", kernarg_segment_align);
                iot::mapRequired(io, ".sgpr_count", sgpr_count);
                iot::mapRequired(io, ".vgpr_count", vgpr_count);
                iot::mapRequired(io, ".agpr_count", agpr_count);
                iot::mapRequired(io, ".max_flat_workgroup_size", max_flat_workgroup_size);
                iot::mapRequired(io, ".workgroup_size", workgroupSize);
                if(not iot::outputting(io))
                {
                    AssertFatal(workgroupSize.size() == 3);
                    kern.m_workgroupSize = {workgroupSize[0], workgroupSize[1], workgroupSize[2]};
                }
                iot::mapRequired(io, ".kernel_dimensions", kern.m_kernelDimensions);
                iot::mapRequired(io, ".wavefront_size", kern.m_wavefrontSize);
                iot::mapRequired(io, ".workitem_count", kern.m_workitemCount);
                iot::mapRequired(io, ".dynamic_sharedmemory_bytes", kern.m_dynamicSharedMemBytes);

                iot::mapRequired(io, ".args", kern.m_arguments);

                //
                // Meta info
                //
                if(iot::outputting(io))
                {
                    if(Settings::getInstance()->get(Settings::SerializeKernelGraph))
                    {
                        iot::mapOptional(io, ".kernel_graph", kern.m_kernelGraph);
                    }

                    if(kern.m_kernelGraph
                       && Settings::getInstance()->get(Settings::SerializeKernelGraphDOT))
                    {
                        auto dot = kern.m_kernelGraph->toDOT();
                        iot::mapOptional(io, ".kernel_graph_dot", dot);
                    }
                }
                else
                {
                    iot::mapOptional(io, ".kernel_graph", kern.m_kernelGraph);
                }

                iot::mapRequired(io, ".command", kern.m_command);
            }

            static void mapping(IO& io, AssemblyKernel& kern, EmptyContext& ctx)
            {
                mapping(io, kern);
            }
        };

        template <typename IO>
        struct MappingTraits<AssemblyKernels, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, AssemblyKernels& kern)
            {
                std::vector<int> hsa_version;

                if(iot::outputting(io))
                {
                    AssertFatal(kern.hsa_version().size() == 2);
                    hsa_version = {kern.hsa_version()[0], kern.hsa_version()[1]};
                }

                iot::mapRequired(io, "amdhsa.version", hsa_version);
                iot::mapRequired(io, "amdhsa.kernels", kern.kernels);

                if(!iot::outputting(io))
                {
                    AssertFatal(hsa_version.size() == 2);
                    // TODO: Set hsa_version from YAML input
                }
            }

            static void mapping(IO& io, AssemblyKernels& kern, EmptyContext& ctx)
            {
                mapping(io, kern);
            }
        };

        ROCROLLER_SERIALIZE_VECTOR(false, AssemblyKernelArgument);
        ROCROLLER_SERIALIZE_VECTOR(false, AssemblyKernel);
    }
}
