/**
 */

#pragma once

#include <map>
#include <memory>
#include <vector>

#include "CommandArgument_fwd.hpp"
#include "CommandArguments.hpp"
#include "OperationTag.hpp"
#include "Operations_fwd.hpp"

#include "../DataTypes/DataTypes.hpp"
#include "../Expression.hpp"

namespace rocRoller
{

    struct Command : public std::enable_shared_from_this<Command>
    {
    public:
        using OperationList = std::vector<std::shared_ptr<Operations::Operation>>;

        Command();
        Command(bool sync);
        ~Command();

        Command(Command const& rhs);
        Command(Command&& rhs);

        Operations::OperationTag addOperation(std::shared_ptr<Operations::Operation> op);
        template <Operations::CConcreteOperation T>
        Operations::OperationTag addOperation(T&& op);

        CommandArgumentPtr allocateArgument(VariableType variableType,
                                            Operations::OperationTag const&,
                                            ArgumentType  argType,
                                            DataDirection direction = DataDirection::ReadWrite);
        CommandArgumentPtr allocateArgument(VariableType variableType,
                                            Operations::OperationTag const&,
                                            ArgumentType       argType,
                                            DataDirection      direction,
                                            std::string const& name);

        std::vector<CommandArgumentPtr> allocateArgumentVector(DataType dataType,
                                                               int      length,
                                                               Operations::OperationTag const& tag,
                                                               ArgumentType  argType,
                                                               DataDirection direction
                                                               = DataDirection::ReadWrite);

        std::vector<CommandArgumentPtr> allocateArgumentVector(DataType dataType,
                                                               int      length,
                                                               Operations::OperationTag const& tag,
                                                               ArgumentType       argType,
                                                               DataDirection      direction,
                                                               std::string const& name);

        std::shared_ptr<Operations::Operation> findTag(Operations::OperationTag const& tag) const;

        template <Operations::CConcreteOperation T>
        T getOperation(Operations::OperationTag const& tag) const;

        Operations::OperationTag getNextTag() const;
        Operations::OperationTag allocateTag();

        OperationList const& operations() const;

        std::string toString() const;
        std::string toString(const unsigned char*) const;
        std::string argInfo() const;

        /**
         * Returns a list of all of the CommandArguments within a Command.
         */
        std::vector<CommandArgumentPtr> getArguments() const;

        /**
         * NOTE: Debugging & testing purposes only.
         *
         * Reads the values of each of the command arguments out of `args` and returns them in
         * a map based on the name of each argument.
         */
        std::map<std::string, CommandArgumentValue>
            readArguments(RuntimeArguments const& args) const;

        /**
         * Returns the expected workItemCount for a command.
         * This is determined by finding the sizes for the first T_LOAD_LINEAR
         * command appearing in the command object.
         */
        std::array<Expression::ExpressionPtr, 3> createWorkItemCount() const;

        CommandArguments createArguments() const;

    private:
        /// Does this command need to synchronize before continuing the calling CPU thread?
        bool m_sync = false;

        OperationList m_operations;

        std::map<Operations::OperationTag, std::shared_ptr<Operations::Operation>> m_tagMap;

        ArgumentOffsetMap m_argOffsetMap;

        std::vector<CommandArgumentPtr> m_commandArgs;

        Operations::OperationTag m_nextTagValue{0};

        int m_runtimeArgsOffset = 0;
    };

    std::ostream& operator<<(std::ostream& stream, Command const& command);

}

#include "Command_impl.hpp"
