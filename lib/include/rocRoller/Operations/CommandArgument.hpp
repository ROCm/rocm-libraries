/**
 */

#pragma once

#include <optional>
#include <string>

#include "CommandArgument_fwd.hpp"
#include "Command_fwd.hpp"

#include "../AssemblyKernel_fwd.hpp"
#include "../DataTypes/DataTypes.hpp"
#include "../Expression_fwd.hpp"
#include "../InstructionValues/Register_fwd.hpp"
#include "../KernelArguments_fwd.hpp"

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
        CommandArgument(std::shared_ptr<Command> com,
                        VariableType             variableType,
                        size_t                   offset,
                        DataDirection            direction = DataDirection::ReadWrite,
                        std::string              name      = "");

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
        std::weak_ptr<Command> m_command;

        int           m_size;
        size_t        m_offset;
        VariableType  m_variableType;
        DataDirection m_direction;
        std::string   m_name;
    };

    std::ostream& operator<<(std::ostream&, CommandArgument const&);
    std::ostream& operator<<(std::ostream&, std::shared_ptr<CommandArgument> const&);
    std::ostream& operator<<(std::ostream&, std::vector<std::shared_ptr<CommandArgument>> const&);

    VariableType  variableType(CommandArgumentValue const& val);
    std::string   name(CommandArgumentValue const& val);
    std::string   toString(CommandArgumentValue const& val);
    std::ostream& operator<<(std::ostream&, CommandArgumentValue const&);

    /**
     * Returns an unsigned integer from a CommandArgumentValue.
     * If the ComandArgumentValue is not an integer, an exception will be thrown.
     */
    unsigned int getUnsignedInt(CommandArgumentValue val);

    /**
     * Returns true if a CommandArgumentValue is an integer type.
     */
    bool isInteger(CommandArgumentValue val);
}

#include "CommandArgument_impl.hpp"
