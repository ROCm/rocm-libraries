
#pragma once

#ifdef ROCROLLER_USE_LLVM
#include <llvm/ObjectYAML/YAML.h>
#endif

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>

#include "Base.hpp"
#include "Containers.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        const std::string KeyValue            = "Value";
        const std::string KeyMajorVersion     = "MajorVersion";
        const std::string KeyMinorVersion     = "MinorVersion";
        const std::string KeyPointVersion     = "PointVersion";
        const std::string KeySramecc          = "Sramecc";
        const std::string KeyStringRep        = "StringRep";
        const std::string KeyVersionRep       = "VersionRep";
        const std::string KeyLLVMFeaturesRep  = "LLVMFeaturesRep";
        const std::string KeyInstruction      = "Instruction";
        const std::string KeyWaitCount        = "WaitCount";
        const std::string KeyWaitQueues       = "WaitQueues";
        const std::string KeyLatency          = "Latency";
        const std::string KeyISAVersion       = "ISAVersion";
        const std::string KeyInstructionInfos = "InstructionInfos";
        const std::string KeyCapabilities     = "Capabilities";
        const std::string KeyArchitectures    = "Architectures";

        template <typename IO>
        struct EnumTraits<GPUWaitQueueType, IO>
        {
            using iot = IOTraits<IO>;

            static void enumeration(IO& io, GPUWaitQueueType& value)
            {
                for(int i = 0; i < static_cast<int>(GPUWaitQueueType::Count); i++)
                {
                    auto dir = static_cast<GPUWaitQueueType>(i);
                    auto str = dir.ToString();
                    iot::enumCase(io, value, str.c_str(), dir);
                }
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

            static void mapping(IO& io, GPUArchitectureTarget& arch)
            {
                iot::mapRequired(io, KeyMajorVersion.c_str(), arch.m_majorVersion);
                iot::mapRequired(io, KeyMinorVersion.c_str(), arch.m_minorVersion);
                iot::mapRequired(io, KeyPointVersion.c_str(), arch.m_pointVersion);
                iot::mapRequired(io, KeySramecc.c_str(), arch.m_sramecc);
                iot::mapRequired(io, KeyStringRep.c_str(), arch.m_string_rep);
                iot::mapRequired(io, KeyVersionRep.c_str(), arch.m_version_rep);
                iot::mapRequired(io, KeyLLVMFeaturesRep.c_str(), arch.m_llvm_features_rep);
            }

            static void mapping(IO& io, GPUArchitectureTarget& arch, EmptyContext& ctx)
            {
                mapping(io, arch);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUInstructionInfo, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUInstructionInfo& info)
            {
                iot::mapRequired(io, KeyInstruction.c_str(), info.m_instruction);
                iot::mapRequired(io, KeyWaitCount.c_str(), info.m_waitCount);
                iot::mapRequired(io, KeyWaitQueues.c_str(), info.m_waitQueues);
                iot::mapRequired(io, KeyLatency.c_str(), info.m_latency);
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
                iot::mapRequired(io, KeyISAVersion.c_str(), arch.m_isaVersion);
                iot::mapRequired(io, KeyInstructionInfos.c_str(), arch.m_instruction_infos);
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

        ROCROLLER_SERIALIZE_VECTOR(false, GPUWaitQueueType);
    }
}

#ifdef ROCROLLER_USE_LLVM
namespace llvm
{
    namespace yaml
    {
        template <rocRoller::Serialization::MappedType<IO> T>
        struct MappingTraits<T>
        {
            using obj        = T;
            using TheMapping = rocRoller::Serialization::MappingTraits<obj, IO>;

            static void mapping(IO& io, obj& o)
            {
                mapping(io, o, nullptr);
            }

            static void mapping(IO& io, obj& o, void*)
            {
                rocRoller::Serialization::EmptyContext ctx;
                TheMapping::mapping(io, o, ctx);
            }
        };

        template <rocRoller::Serialization::MappedType<IO> T>
        struct MappingTraits<Hide<T>>
        {
            using obj        = Hide<T>;
            using TheMapping = rocRoller::Serialization::MappingTraits<T, IO>;

            static void mapping(IO& io, obj& o)
            {
                mapping(io, o, nullptr);
            }

            static void mapping(IO& io, obj& o, void*)
            {
                rocRoller::Serialization::EmptyContext ctx;
                TheMapping::mapping(io, *o, ctx);
            }
        };
    }
}
static_assert(rocRoller::Serialization::CustomMappingType<
              std::map<rocRoller::GPUArchitectureTarget, rocRoller::GPUArchitecture>,
              llvm::yaml::IO>);
static_assert(rocRoller::Serialization::CustomMappingType<std::map<rocRoller::GPUCapability, int>,
                                                          llvm::yaml::IO>);
static_assert(rocRoller::Serialization::MappedType<rocRoller::GPUArchitecture, llvm::yaml::IO>);
static_assert(!llvm::yaml::has_FlowTraits<rocRoller::GPUCapability>::value);
#endif

#ifdef ROCROLLER_USE_YAML_CPP

namespace rocRoller
{
    void operator>>(const YAML::Node& node, GPUWaitQueueType& result)
    {
        GPUWaitQueueType rv(node.as<std::string>());
        result = std::move(rv);
    }

    void operator>>(const YAML::Node& node, std::vector<GPUWaitQueueType>& result)
    {
        for(unsigned i = 0; i < node.size(); i++)
        {
            GPUWaitQueueType element;
            node[i] >> element;
            result.push_back(element);
        }
    }

    void operator>>(const YAML::Node& node, GPUInstructionInfo& result)
    {
        std::string instruction = node[Serialization::KeyInstruction].as<std::string>();
        int         waitCount   = node[Serialization::KeyWaitCount].as<int>();
        std::vector<GPUWaitQueueType> waitQueueTypes;
        node[Serialization::KeyWaitQueues] >> waitQueueTypes;
        int                latency = node[Serialization::KeyLatency].as<int>();
        GPUInstructionInfo rv(instruction, waitCount, waitQueueTypes, latency);
        result = std::move(rv);
    }

    void operator>>(const YAML::Node& node, GPUArchitectureTarget& result)
    {
        GPUArchitectureTarget rv(node[Serialization::KeyStringRep].as<std::string>());
        result = std::move(rv);
    }

    void operator>>(const YAML::Node& node, std::map<GPUCapability, int>& result)
    {
        for(YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            GPUCapability key(it->first.as<std::string>());
            result[key] = it->second.as<int>();
        }
    }

    void operator>>(const YAML::Node& node, std::map<std::string, GPUInstructionInfo>& result)
    {
        for(YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            it->second >> result[it->first.as<std::string>()];
        }
    }

    void operator>>(const YAML::Node& node, GPUArchitecture& result)
    {
        GPUArchitectureTarget isaVersion;
        node[Serialization::KeyISAVersion] >> isaVersion;
        std::map<GPUCapability, int> capabilities;
        node[Serialization::KeyCapabilities] >> capabilities;
        std::map<std::string, GPUInstructionInfo> instructionInfos;
        node[Serialization::KeyInstructionInfos] >> instructionInfos;
        GPUArchitecture rv(isaVersion, capabilities, instructionInfos);
        result = std::move(rv);
    }

    void operator>>(const YAML::Node&                                 node,
                    std::map<GPUArchitectureTarget, GPUArchitecture>& result)
    {
        for(YAML::const_iterator it = node.begin(); it != node.end(); ++it)
        {
            GPUArchitectureTarget key(it->first.as<std::string>());
            GPUArchitecture       value;
            it->second >> value;
            result[key] = value;
        }
    }

    void operator>>(const YAML::Node& node, GPUArchitecturesStruct& result)
    {
        node[Serialization::KeyArchitectures] >> result.architectures;
    }
}

#endif
