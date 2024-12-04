#pragma once

#ifdef ROCROLLER_USE_LLVM
#include <llvm/ObjectYAML/YAML.h>
#endif

#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations_fwd.hpp>
#include <rocRoller/Operations/T_Execute.hpp>

#include <rocRoller/Utilities/Utils.hpp>

#include "Base.hpp"
#include "Containers.hpp"

#include "Expression.hpp"
#include "Operations.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO>
        struct SequenceTraits<std::vector<CommandArgumentPtr>, IO>
            : public DefaultSequenceTraits<std::vector<CommandArgumentPtr>, IO, false>
        {
        };

        template <>
        struct KeyConversion<ArgumentOffsetMap::key_type>
        {
            static std::string toString(ArgumentOffsetMap::key_type const& value)
            {
                std::stringstream ss;
                ss << std::get<0>(value) << " " << std::get<1>(value) << " " << std::get<2>(value);
                return ss.str();
            }

            static ArgumentOffsetMap::key_type fromString(std::string const& value)
            {
                std::stringstream ss(value);

                int         tag, dim;
                std::string argTypeStr;

                ss >> tag;
                ss >> argTypeStr;
                ss >> dim;

                auto argType = rocRoller::fromString<ArgumentType>(argTypeStr);

                return {Operations::OperationTag{tag}, argType, dim};
            }
        };

        template <typename IO>
        struct CustomMappingTraits<ArgumentOffsetMap, IO>
            : public DefaultCustomMappingTraits<ArgumentOffsetMap, IO, false, false>
        {
        };

        template <typename IO>
        struct MappingTraits<Command, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, Command& command)
            {
                iot::mapRequired(io, "sync", command.m_sync);
                iot::mapRequired(io, "operations", command.m_operations);
                iot::mapRequired(io, "argOffsetMap", command.m_argOffsetMap);
                iot::mapRequired(io, "commandArgs", command.m_commandArgs);
                iot::mapRequired(io, "nextTag", command.m_nextTagValue);
                iot::mapRequired(io, "runtimeArgsOffset", command.m_runtimeArgsOffset);

                // XXX reconstruct m_tagMap when !outputting
            }

            static void mapping(IO& io, Command& command, EmptyContext& ctx)
            {
                mapping(io, command);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<CommandPtr, IO, Context>
            : public SharedPointerMappingTraits<CommandPtr, IO, Context, true>
        {
        };
    }
}
