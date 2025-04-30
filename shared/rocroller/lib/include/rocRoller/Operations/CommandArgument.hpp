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

/**
 */

#pragma once

#include <optional>
#include <string>

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/KernelArguments_fwd.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    // CommandArgumentValue: type alias for std::variant of all value type.
    // Defined in CommandArgument_fwd.hpp.
    // RuntimeArguments: type alias for a container that holds runtime arguments.

    /**
     * An incoming argument.  Obtaining the underlying value is
     * intentionally difficult, to prevent accidentally generating a
     * kernel dependent on an argument having a specific value.
     *
     * Any specific optimizations based on an argument having a
     * specific value or type of value should be implemented through
     * the predication mechanism.
     */
    class CommandArgument : public std::enable_shared_from_this<CommandArgument>
    {
    public:
        CommandArgument(CommandPtr    com,
                        VariableType  variableType,
                        size_t        offset,
                        DataDirection direction = DataDirection::ReadWrite,
                        std::string   name      = "");

        VariableType variableType() const;

        /**
         *  Size & offset into the argument packet.   May not be set
         */
        int           size() const;
        int           offset() const;
        std::string   name() const;
        DataDirection direction() const;

        std::string toString() const;

        CommandArgumentValue getValue(RuntimeArguments const& args) const;

        Expression::ExpressionPtr expression();

        template <typename T>
        requires(!std::is_pointer_v<T>) CommandArgumentValue getValue(RuntimeArguments const& args)
        const;

        bool operator==(CommandArgument const& rhs) const;

    private:
        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

        std::weak_ptr<Command> m_command;

        int           m_size;
        size_t        m_offset;
        VariableType  m_variableType;
        DataDirection m_direction;
        std::string   m_name;
    };

    std::ostream& operator<<(std::ostream&, CommandArgument const&);
    std::ostream& operator<<(std::ostream&, CommandArgumentPtr const&);
    std::ostream& operator<<(std::ostream&, std::vector<CommandArgumentPtr> const&);

    VariableType  variableType(CommandArgumentValue const& val);
    std::string   name(CommandArgumentValue const& val);
    std::string   toString(CommandArgumentValue const& val);
    std::ostream& operator<<(std::ostream&, CommandArgumentValue const&);

    /**
     * Returns an unsigned integer from a CommandArgumentValue.
     * If the CommandArgumentValue is not an integer, an exception will be thrown.
     */
    unsigned int getUnsignedInt(CommandArgumentValue val);

    /**
     * Returns true if a CommandArgumentValue is an integer type.
     */
    bool isInteger(CommandArgumentValue val);
}

#include <rocRoller/Operations/CommandArgument_impl.hpp>
