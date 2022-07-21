
#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "GPUArchitectureTarget.hpp"
#include "GPUCapability.hpp"
#include "GPUInstructionInfo.hpp"

#include "../InstructionValues/Register_fwd.hpp"
#include "../Serialization/GPUArchitecture_fwd.hpp"

#include <msgpack.hpp>

namespace rocRoller
{
    class GPUArchitecture
    {
    public:
        void                          AddCapability(GPUCapability, int);
        void                          AddInstructionInfo(GPUInstructionInfo);
        bool                          HasCapability(GPUCapability) const;
        int                           GetCapability(GPUCapability) const;
        bool                          HasCapability(std::string) const;
        int                           GetCapability(std::string) const;
        rocRoller::GPUInstructionInfo GetInstructionInfo(std::string) const;

        GPUArchitecture();
        GPUArchitecture(GPUArchitectureTarget);
        GPUArchitecture(GPUArchitectureTarget const&,
                        std::map<GPUCapability, int> const&,
                        std::map<std::string, GPUInstructionInfo> const&);

        friend std::ostream& operator<<(std::ostream& os, const GPUArchitecture& d);

        /**
         * Returns true if `reg` is a Literal value that can be represented as a
         * constant operand in an instruction.  This is one of only a limited set of
         * values that can be encoded directly in the instruction, as opposed to
         * actual literals, which must be encoded as a separate 32-bit word in the
         * instruction stream.
         *
         * Many instructions support constant values but not literal values.
         */
        bool isSupportedConstantValue(Register::ValuePtr reg) const;

        /**
         * Returns true iff `value` can be represented as an iconst value in an
         * instruction. For all supported architectures, currently this is -16..64
         * inclusive
         */
        template <std::integral T>
        bool isSupportedConstantValue(T value) const;

        /**
         * Returns true iff `value` can be represented as an fconst value in an
         * instruction. For all supported architectures, currently this includes:
         *
         *  * 0, 0.5, 1.0, 2.0, 4.0
         *  * -0.5, -1.0, -2.0, -4.0
         *  * 1/(2*pi)
         */
        template <std::floating_point T>
        bool isSupportedConstantValue(T value) const;

        GPUArchitectureTarget const& target() const;

        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

        static std::string writeYaml(std::map<GPUArchitectureTarget, GPUArchitecture> const&);
        static std::map<GPUArchitectureTarget, GPUArchitecture> readYaml(std::string const&);
        static std::string writeMsgpack(std::map<GPUArchitectureTarget, GPUArchitecture> const&);
        static std::map<GPUArchitectureTarget, GPUArchitecture> readMsgpack(std::string const&);

        MSGPACK_DEFINE(m_isaVersion, m_instruction_infos, m_capabilities);

    private:
        GPUArchitectureTarget                     m_isaVersion;
        std::map<GPUCapability, int>              m_capabilities;
        std::map<std::string, GPUInstructionInfo> m_instruction_infos;
    };

    //Used as a container for serialization.
    struct GPUArchitecturesStruct
    {
        std::map<GPUArchitectureTarget, GPUArchitecture> architectures;
    };
}

#include "GPUArchitecture_impl.hpp"
