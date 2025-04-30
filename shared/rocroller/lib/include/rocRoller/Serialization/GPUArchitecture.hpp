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

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/Enum.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        const std::string KeyValue            = "Value";
        const std::string KeySramecc          = "Sramecc";
        const std::string KeyXnack            = "Xnack";
        const std::string KeyInstruction      = "Instruction";
        const std::string KeyWaitCount        = "WaitCount";
        const std::string KeyWaitQueues       = "WaitQueues";
        const std::string KeyLatency          = "Latency";
        const std::string KeyImplicitAccess   = "ImplicitAccess";
        const std::string KeyIsBranch         = "IsBranch";
        const std::string KeyMaxLiteral       = "MaxLiteral";
        const std::string KeyISAVersion       = "ISAVersion";
        const std::string KeyInstructionInfos = "InstructionInfos";
        const std::string KeyCapabilities     = "Capabilities";
        const std::string KeyArchitectures    = "Architectures";
        const std::string KeyArchString       = "ArchString";

        /**
         * GPUWaitQueueType is actually a class that look like an enum, so it is not handled by the
         * generic enum serialization.
         */
        template <>
        struct ScalarTraits<GPUWaitQueueType>
        {
            static std::string output(const GPUWaitQueueType& value)
            {
                return toString(value);
            }

            static void input(std::string const& scalar, GPUWaitQueueType& value)
            {
                value = GPUWaitQueueType(scalar);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUCapability, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUCapability& cap)
            {
                iot::mapRequired(io, KeyValue.c_str(), cap.m_value);
            }

            static void mapping(IO& io, GPUCapability& info, EmptyContext& ctx)
            {
                mapping(io, info);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUArchitectureTarget, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUArchitectureTarget& target)
            {
                iot::mapRequired(io, KeyArchString.c_str(), target.gfx);
                iot::mapRequired(io, KeyXnack.c_str(), target.features.xnack);
                iot::mapRequired(io, KeySramecc.c_str(), target.features.sramecc);
            }

            static void mapping(IO& io, GPUArchitectureTarget& target, EmptyContext& ctx)
            {
                mapping(io, target);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUInstructionInfo, IO, EmptyContext>
        {
            static const bool flow = true;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUInstructionInfo& info)
            {
                iot::mapRequired(io, KeyInstruction.c_str(), info.m_instruction);
                iot::mapRequired(io, KeyWaitCount.c_str(), info.m_waitCount);
                iot::mapRequired(io, KeyWaitQueues.c_str(), info.m_waitQueues);
                iot::mapRequired(io, KeyLatency.c_str(), info.m_latency);
                iot::mapRequired(io, KeyImplicitAccess.c_str(), info.m_implicitAccess);
                iot::mapRequired(io, KeyIsBranch.c_str(), info.m_isBranch);
                iot::mapRequired(io, KeyMaxLiteral.c_str(), info.m_maxOffsetValue);
            }

            static void mapping(IO& io, GPUInstructionInfo& info, EmptyContext& ctx)
            {
                mapping(io, info);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUArchitecture, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUArchitecture& arch)
            {
                iot::mapRequired(io, KeyISAVersion.c_str(), arch.m_archTarget);
                iot::mapRequired(io, KeyInstructionInfos.c_str(), arch.m_instructionInfos);
                iot::mapRequired(io, KeyCapabilities.c_str(), arch.m_capabilities);
            }

            static void mapping(IO& io, GPUArchitecture& arch, EmptyContext& ctx)
            {
                mapping(io, arch);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUArchitecturesStruct, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUArchitecturesStruct& arch)
            {
                iot::mapRequired(io, KeyArchitectures.c_str(), arch.architectures);
            }

            static void mapping(IO& io, GPUArchitecturesStruct& arch, EmptyContext& ctx)
            {
                mapping(io, arch);
            }
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<std::string, GPUInstructionInfo>, IO>
            : public DefaultCustomMappingTraits<std::map<std::string, GPUInstructionInfo>,
                                                IO,
                                                false,
                                                true>
        {
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<GPUCapability, int>, IO>
            : public DefaultCustomMappingTraits<std::map<GPUCapability, int>, IO, false, true>
        {
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<GPUArchitectureTarget, GPUArchitecture>, IO>
            : public DefaultCustomMappingTraits<std::map<GPUArchitectureTarget, GPUArchitecture>,
                                                IO,
                                                false,
                                                true>
        {
        };

        ROCROLLER_SERIALIZE_VECTOR(true, GPUWaitQueueType);
    }
}
