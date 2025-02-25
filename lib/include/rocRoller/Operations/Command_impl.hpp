/**
 */

#pragma once

#include "Command.hpp"
#include "Operations.hpp"
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    inline Command::Command() = default;

    inline Command::Command(bool sync)
        : m_sync(sync)
    {
    }

    inline Command::~Command() = default;

    inline Command::Command(Command const& rhs) = default;
    inline Command::Command(Command&& rhs)      = default;

    inline Operations::OperationTag Command::addOperation(std::shared_ptr<Operations::Operation> op)
    {
        Operations::OperationTag rv;

        if(op == nullptr)
            return rv;

        AssertFatal(std::find(m_operations.begin(), m_operations.end(), op) == m_operations.end());

        Operations::AssignOutputs assignOutputsVisitor;
        auto                      outputs = assignOutputsVisitor.call(*op, m_nextTagValue);

        for(auto const& tag : outputs)
        {
            AssertFatal(m_tagMap.find(tag) == m_tagMap.end());
            rv = tag;

            m_tagMap[tag] = op;

            m_nextTagValue = std::max(m_nextTagValue, tag + 1);
        }

        m_operations.emplace_back(op);

        auto set = Operations::SetCommand(shared_from_this());
        set.call(*op);

        Operations::AllocateArguments allocate;
        allocate.call(*op);
        return rv;
    }

    template <Operations::CConcreteOperation T>
    inline Operations::OperationTag Command::addOperation(T&& op)
    {
        return addOperation(std::make_shared<Operations::Operation>(std::forward<T>(op)));
    }

    // Allocate a single command argument by incrementing the most recent offset.
    inline CommandArgumentPtr Command::allocateArgument(VariableType variableType,
                                                        Operations::OperationTag const& tag,
                                                        ArgumentType                    argType,
                                                        DataDirection                   direction)
    {
        std::string name
            = concatenate("user_", variableType.dataType, "_", variableType.pointerType, "_", tag);

        return allocateArgument(variableType, tag, argType, direction, name);
    }

    inline CommandArgumentPtr Command::allocateArgument(VariableType variableType,
                                                        Operations::OperationTag const& tag,
                                                        ArgumentType                    argType,
                                                        DataDirection                   direction,
                                                        std::string const&              name)
    {
        // TODO Fix argument alignment
        auto info      = DataTypeInfo::Get(variableType.dataType);
        int  alignment = info.alignment;
        if(variableType.isPointer())
            alignment = 8;
        m_runtimeArgsOffset = RoundUpToMultiple(m_runtimeArgsOffset, alignment);

        int old_offset = m_runtimeArgsOffset;
        m_runtimeArgsOffset += variableType.getElementSize();
        m_commandArgs.emplace_back(std::make_shared<CommandArgument>(
            shared_from_this(), variableType, old_offset, direction, name));
        m_argOffsetMap[std::make_tuple(tag, argType, -1)] = old_offset;
        return m_commandArgs[m_commandArgs.size() - 1];
    }

    inline std::vector<CommandArgumentPtr> Command::getArguments() const
    {
        return m_commandArgs;
    }

    inline std::vector<CommandArgumentPtr>
        Command::allocateArgumentVector(DataType                        dataType,
                                        int                             length,
                                        Operations::OperationTag const& tag,
                                        ArgumentType                    argType,
                                        DataDirection                   direction)
    {
        std::string name = concatenate("user", m_commandArgs.size());
        return allocateArgumentVector(dataType, length, tag, argType, direction, name);
    }

    // Allocate a vector of command arguments by incrementing the most recent offset.
    inline std::vector<CommandArgumentPtr>
        Command::allocateArgumentVector(DataType                        dataType,
                                        int                             length,
                                        Operations::OperationTag const& tag,
                                        ArgumentType                    argType,
                                        DataDirection                   direction,
                                        std::string const&              name)
    {
        std::vector<CommandArgumentPtr> args;
        for(int i = 0; i < length; i++)
        {
            m_commandArgs.emplace_back(
                std::make_shared<CommandArgument>(shared_from_this(),
                                                  dataType,
                                                  m_runtimeArgsOffset,
                                                  direction,
                                                  concatenate(name, "_", i)));
            args.push_back(m_commandArgs[m_commandArgs.size() - 1]);
            m_argOffsetMap[std::make_tuple(tag, argType, i)] = m_runtimeArgsOffset;
            m_runtimeArgsOffset += CeilDivide(DataTypeInfo::Get(dataType).elementBits, 8u);
        }

        return args;
    }

    // TODO: More advanced version of createWorkItemCount
    //       Right now, workItemCount is determined by the size of the first
    //       T_LOAD_LINEAR appearing in the command.
    inline std::array<Expression::ExpressionPtr, 3> Command::createWorkItemCount() const
    {

        bool found = false;
        auto one   = std::make_shared<Expression::Expression>(static_cast<int64_t>(1));

        std::array<Expression::ExpressionPtr, 3> result({one, one, one});

        for(auto operation : m_operations)
        {
            visit(rocRoller::overloaded{
                      [&](auto op) {},
                      [&](Operations::T_Load_Linear const& op) {
                          auto tensor = getOperation<Operations::Tensor>(op.getSrcTag());
                          auto sizes  = tensor.sizes();
                          for(size_t i = 0; i < sizes.size() && i < 3; i++)
                          {
                              result[i] = std::make_shared<Expression::Expression>(sizes[i]);
                          }

                          found = true;
                      },
                  },
                  *operation);

            if(found)
                break;
        }

        return result;
    }

    inline CommandArguments Command::createArguments() const
    {
        auto argOffsetMapPtr = std::make_shared<const ArgumentOffsetMap>(m_argOffsetMap);
        return CommandArguments(argOffsetMapPtr, m_runtimeArgsOffset);
    }

    inline std::shared_ptr<Operations::Operation>
        Command::findTag(Operations::OperationTag const& tag) const
    {
        auto iter = m_tagMap.find(tag);
        if(iter == m_tagMap.end())
            return nullptr;

        return iter->second;
    }

    template <Operations::CConcreteOperation T>
    T Command::getOperation(Operations::OperationTag const& tag) const
    {
        return std::get<T>(*findTag(tag));
    }

    inline Operations::OperationTag Command::getNextTag() const
    {
        return m_nextTagValue;
    }

    inline Operations::OperationTag Command::allocateTag()
    {
        return m_nextTagValue++;
    }

    inline std::string Command::toString() const
    {
        return toString(nullptr);
    }

    inline std::string Command::toString(const unsigned char* runtime_args) const
    {
        std::string rv;

        Operations::ToStringVisitor toStringVisitor;

        for(auto const& op : m_operations)
        {
            std::string op_string = toStringVisitor.call(*op, runtime_args);
            if(op_string.size() > 0)
                rv += op_string + "\n";
        }

        return rv;
    }

    inline std::string Command::argInfo() const
    {
        std::string rv;

        for(auto const& arg : m_commandArgs)
        {
            rv += arg->toString() + '\n';
        }

        return rv;
    }

    inline std::map<std::string, CommandArgumentValue>
        Command::readArguments(RuntimeArguments const& args) const
    {
        std::map<std::string, CommandArgumentValue> rv;

        for(auto const& ca : m_commandArgs)
        {
            rv[ca->name()] = ca->getValue(args);
        }

        return rv;
    }

    inline Command::OperationList const& Command::operations() const
    {
        return m_operations;
    }

    inline std::ostream& operator<<(std::ostream& stream, Command const& command)
    {
        return stream << command.toString();
    }

}
