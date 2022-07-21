#pragma once

#ifdef ROCROLLER_USE_LLVM
#include <llvm/ObjectYAML/YAML.h>
#endif

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>

#include "Base.hpp"
#include "Containers.hpp"

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
                AssertFatal(iot::outputting(io));

                iot::mapRequired(io, ".name", arg.name);
                iot::mapRequired(io, ".size", arg.size);
                iot::mapRequired(io, ".offset", arg.offset);

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
                AssertFatal(iot::outputting(io));

                std::string name = kern.kernelName();
                iot::mapRequired(io, ".name", name);

                std::string symbol = name + ".kd";
                iot::mapRequired(io, ".symbol", symbol);

                iot::mapRequired(io, ".kernarg_segment_size", kern.kernarg_segment_size());
                iot::mapRequired(io, ".group_segment_fixed_size", kern.group_segment_fixed_size());
                iot::mapRequired(
                    io, ".private_segment_fixed_size", kern.private_segment_fixed_size());
                iot::mapRequired(io, ".kernarg_segment_align", kern.kernarg_segment_align());
                iot::mapRequired(io, ".wavefront_size", kern.wavefront_size());
                iot::mapRequired(io, ".sgpr_count", kern.sgpr_count());
                iot::mapRequired(io, ".vgpr_count", kern.vgpr_count());
                iot::mapRequired(io, ".agpr_count", kern.agpr_count());
                iot::mapRequired(io, ".max_flat_workgroup_size", kern.max_flat_workgroup_size());

                iot::mapRequired(io, ".args", kern.arguments());

                // TODO: only do this for appropriate logging levels
                auto graph = kern.kernel_graph();
                if(graph)
                    iot::mapRequired(io, ".kernel_graph", *graph);
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
                AssertFatal(iot::outputting(io));
                iot::mapRequired(io, "amdhsa.version", kern.hsa_version());
                iot::mapRequired(io, "amdhsa.kernels", kern.kernels);
            }

            static void mapping(IO& io, AssemblyKernels& kern, EmptyContext& ctx)
            {
                mapping(io, kern);
            }
        };

        template <typename IO>
        struct EnumTraits<DataType, IO>
        {
            using iot = IOTraits<IO>;

            static void enumeration(IO& io, DataType& value)
            {
                for(int i = 0; i < static_cast<int>(DataType::Count); i++)
                {
                    auto const& info = DataTypeInfo::Get(i);
                    iot::enumCase(io, value, info.name.c_str(), info.variableType.dataType);
                }
            }
        };

        template <typename IO>
        struct EnumTraits<DataDirection, IO>
        {
            using iot = IOTraits<IO>;

            static void enumeration(IO& io, DataDirection& value)
            {
                for(int i = 0; i < static_cast<int>(DataDirection::Count); i++)
                {
                    auto        dir  = static_cast<DataDirection>(i);
                    auto        str  = ToString(dir);
                    auto const& info = DataTypeInfo::Get(i);
                    iot::enumCase(io, value, str.c_str(), dir);
                }
            }
        };

        ROCROLLER_SERIALIZE_VECTOR(false, AssemblyKernelArgument);
        ROCROLLER_SERIALIZE_VECTOR(false, AssemblyKernel);
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

        template <rocRoller::Serialization::SequenceType<IO> T>
        struct SequenceTraits<Hide<T>>
        {
            using obj         = Hide<T>;
            using TheSequence = rocRoller::Serialization::SequenceTraits<T, IO>;

            static size_t size(IO& io, obj& list)
            {
                return TheSequence::size(io, *list);
            }
            static auto& element(IO& io, obj& list, size_t index)
            {
                return TheSequence::element(io, *list, index);
            }
        };
    }
}
static_assert(rocRoller::Serialization::MappedType<rocRoller::AssemblyKernels, llvm::yaml::IO>);
static_assert(!llvm::yaml::has_FlowTraits<rocRoller::AssemblyKernels>::value);
#endif
