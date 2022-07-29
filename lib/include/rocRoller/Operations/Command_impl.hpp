/**
 */

#pragma once

#include "Command.hpp"
#include "Operations.hpp"
#include "Utilities/Utils.hpp"

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

    inline void Command::addOperation(std::shared_ptr<Operations::Operation> op)
    {
        if(op == nullptr)
            return;

        AssertFatal(std::find(m_operations.begin(), m_operations.end(), op) == m_operations.end());

        auto outputs = Operations::AssignOutputs()(*op, m_nextTagValue);

        for(auto const& tag : outputs)
        {
            AssertFatal(m_tagMap.find(tag) == m_tagMap.end());

            m_tagMap[tag] = op;

            m_nextTagValue = std::max(m_nextTagValue, tag + 1);
        }

        m_operations.emplace_back(op);

        auto set = Operations::SetCommand(shared_from_this());
        set(*op);

        auto allocate = Operations::AllocateArguments();
        allocate(*op);
    }

    // Allocate a single command argument by incrementing the most recent offset.
    inline std::shared_ptr<CommandArgument> Command::allocateArgument(VariableType  variableType,
                                                                      DataDirection direction)
    {
        std::string name = concatenate("user_",
                                       variableType.dataType,
                                       "_",
                                       variableType.pointerType,
                                       "_",
                                       m_command_args.size());

        return allocateArgument(variableType, direction, name);
    }

    inline std::shared_ptr<CommandArgument> Command::allocateArgument(VariableType  variableType,
                                                                      DataDirection direction,
                                                                      std::string const& name)
    {
        // TODO Fix argument alignment
        auto info      = DataTypeInfo::Get(variableType.dataType);
        int  alignment = info.alignment;
        if(variableType.isPointer())
            alignment = 8;
        m_runtime_args_offset = RoundUpToMultiple(m_runtime_args_offset, alignment);

        int old_offset = m_runtime_args_offset;
        m_runtime_args_offset += variableType.getElementSize();
        m_command_args.emplace_back(std::make_shared<CommandArgument>(
            shared_from_this(), variableType, old_offset, direction, name));
        return m_command_args[m_command_args.size() - 1];
    }

    inline std::vector<std::shared_ptr<CommandArgument>> Command::getArguments() const
    {
        return m_command_args;
    }

    inline std::vector<std::shared_ptr<CommandArgument>>
        Command::allocateArgumentVector(DataType dataType, int length, DataDirection direction)
    {
        std::string name = concatenate("user", m_command_args.size());
        return allocateArgumentVector(dataType, length, direction, name);
    }

    // Allocate a vector of command arguments by incrementing the most recent offset.
    inline std::vector<std::shared_ptr<CommandArgument>> Command::allocateArgumentVector(
        DataType dataType, int length, DataDirection direction, std::string const& name)
    {
        std::vector<std::shared_ptr<CommandArgument>> args;
        for(int i = 0; i < length; i++)
        {
            m_command_args.emplace_back(
                std::make_shared<CommandArgument>(shared_from_this(),
                                                  dataType,
                                                  m_runtime_args_offset,
                                                  direction,
                                                  concatenate(name, "_", i)));
            args.push_back(m_command_args[m_command_args.size() - 1]);
            m_runtime_args_offset += DataTypeInfo::Get(dataType).elementSize;
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
                          auto sizes = op.sizes();
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

    inline std::shared_ptr<Operations::Operation> Command::findTag(int tag)
    {
        auto iter = m_tagMap.find(tag);
        if(iter == m_tagMap.end())
            return nullptr;

        return iter->second;
    }

    inline int Command::getNextTag() const
    {
        return m_nextTagValue;
    }

    inline std::string Command::toString() const
    {
        return toString(nullptr);
    }

    inline std::string Command::toString(const unsigned char* runtime_args) const
    {
        std::string rv;

        auto func = Operations::ToString();

        for(auto const& op : m_operations)
        {
            std::string op_string = func(*op, runtime_args);
            if(op_string.size() > 0)
                rv += op_string + "\n";
        }

        return rv;
    }

    inline std::string Command::argInfo() const
    {
        std::string rv;

        for(auto const& arg : m_command_args)
        {
            rv += arg->toString() + '\n';
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
