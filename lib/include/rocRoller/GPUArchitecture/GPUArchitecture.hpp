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

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    class GPUArchitecture
    {
    public:
        GPUArchitecture();
        explicit GPUArchitecture(GPUArchitectureTarget const&);
        GPUArchitecture(GPUArchitectureTarget const&,
                        std::map<GPUCapability, int> const&,
                        std::map<std::string, GPUInstructionInfo> const&);

        void                          AddCapability(GPUCapability const&, int);
        void                          AddInstructionInfo(GPUInstructionInfo const&);
        bool                          HasCapability(GPUCapability const&) const;
        int                           GetCapability(GPUCapability const&) const;
        bool                          HasCapability(std::string const&) const;
        int                           GetCapability(std::string const&) const;
        rocRoller::GPUInstructionInfo GetInstructionInfo(std::string const&) const;
        bool                          HasInstructionInfo(std::string const&) const;

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
        requires(!std::same_as<bool, T>) bool isSupportedConstantValue(T value) const;

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

        constexpr GPUArchitectureTarget const& target() const;

        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

        static std::string writeYaml(std::map<GPUArchitectureTarget, GPUArchitecture> const&);
        static std::map<GPUArchitectureTarget, GPUArchitecture> readYaml(std::string const&);
        static std::string writeMsgpack(std::map<GPUArchitectureTarget, GPUArchitecture> const&);
        static std::map<GPUArchitectureTarget, GPUArchitecture> readMsgpack(std::string const&);
        static std::map<GPUArchitectureTarget, GPUArchitecture> readEmbeddedMsgpack();

        std::map<GPUCapability, int> const&              getAllCapabilities() const;
        std::map<std::string, GPUInstructionInfo> const& getAllIntructionInfo() const;

    private:
        GPUArchitectureTarget                     m_archTarget;
        std::map<GPUCapability, int>              m_capabilities;
        std::map<std::string, GPUInstructionInfo> m_instructionInfos;

        template <std::floating_point T>
        // cppcheck-suppress functionStatic
        std::unordered_set<T> supportedConstantValues() const;
        template <std::integral T>
        // cppcheck-suppress functionStatic
        std::pair<T, T> supportedConstantRange() const;
    };

    //Used as a container for serialization.
    struct GPUArchitecturesStruct
    {
        std::map<GPUArchitectureTarget, GPUArchitecture> architectures;
    };
}

#include <rocRoller/GPUArchitecture/GPUArchitecture_impl.hpp>
