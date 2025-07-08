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

#include <map>
#include <memory>
#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Operations/CommandArguments.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Operations/Operations_fwd.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{

    struct Command : public std::enable_shared_from_this<Command>
    {
    public:
        using OperationList = std::vector<std::shared_ptr<Operations::Operation>>;

        Command();
        explicit Command(bool sync);
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

        static std::string toYAML(Command const&);
        static Command     fromYAML(std::string const& str);

        bool operator==(Command const& rhs) const;

    private:
        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

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

#include <rocRoller/Operations/Command_impl.hpp>
