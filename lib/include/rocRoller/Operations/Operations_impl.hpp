/**
 *
 */

#pragma once

#include "Operations.hpp"
#include "T_Execute.hpp"
#include "T_Mul.hpp"

namespace rocRoller
{
    namespace Operations
    {
        inline std::unordered_set<OperationTag> Inputs::call(Operation const& op)
        {
            return std::visit(*this, op);
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(Tensor const& tensor)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(Scalar const& scalar)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(Literal const& literal)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(BlockScale const& blockScale)
        {
            return blockScale.getInputs();
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(T_Load_Linear const& load)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(T_Load_Scalar const& load)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(T_Load_Tiled const& load)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(T_Mul const& mul)
        {
            return mul.getInputs();
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(T_Store_Linear const& store)
        {
            return {store.getTag()};
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(T_Store_Tiled const& store)
        {
            return {store.getTag()};
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(T_Execute const& exec)
        {
            return exec.getInputs();
        }

        inline std::unordered_set<OperationTag> Inputs::call(XOp const& op)
        {
            return std::visit(*this, op);
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(E_Unary const& unary)
        {
            return unary.getInputs();
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(E_Binary const& binary)
        {
            return binary.getInputs();
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(E_Ternary const& ternary)
        {
            return ternary.getInputs();
        }

        inline std::unordered_set<OperationTag> Inputs::operator()(Nop const& exec)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Outputs::call(Operation const& op)
        {
            return std::visit(*this, op);
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(Tensor const& tensor)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(Scalar const& scalar)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(Literal const& literal)
        {
            return {literal.getTag()};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(BlockScale const& blockScale)
        {
            return {blockScale.getTag()};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(T_Load_Linear const& load)
        {
            return {load.getTag()};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(T_Load_Scalar const& load)
        {
            return {load.getTag()};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(T_Load_Tiled const& load)
        {
            return {load.getTag()};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(T_Mul const& mul)
        {
            return {mul.getTag()};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(T_Store_Linear const& load)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(T_Store_Tiled const& load)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(T_Execute const& exec)
        {
            return exec.getOutputs();
        }

        inline std::unordered_set<OperationTag> Outputs::call(XOp const& op)
        {
            return std::visit(*this, op);
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(E_Unary const& unary)
        {
            return unary.getOutputs();
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(E_Binary const& binary)
        {
            return binary.getOutputs();
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(E_Ternary const& ternary)
        {
            return ternary.getOutputs();
        }

        inline std::unordered_set<OperationTag> Outputs::operator()(Nop const& exec)
        {
            return {};
        }

        inline OperationTag TagVisitor::call(XOp const& op)
        {
            return std::visit(*this, op);
        }

        inline OperationTag TagVisitor::operator()(E_Unary const& unary)
        {
            return unary.getTag();
        }

        inline OperationTag TagVisitor::operator()(E_Binary const& binary)
        {
            return binary.getTag();
        }

        inline OperationTag TagVisitor::operator()(E_Ternary const& ternary)
        {
            return ternary.getTag();
        }

        inline std::unordered_set<OperationTag> AssignOutputs::call(Operation&   op,
                                                                    OperationTag nextTagValue)
        {
            m_nextTagValue = nextTagValue;

            return std::visit(*this, op);
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(Tensor& tensor)
        {
            if(tensor.getTag().uninitialized())
            {
                tensor.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {tensor.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(Scalar& scalar)
        {
            if(scalar.getTag().uninitialized())
            {
                scalar.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {scalar.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(Literal& literal)
        {
            if(literal.getTag().uninitialized())
            {
                literal.setTag(m_nextTagValue);
                m_nextTagValue++;
            }
            return {literal.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(BlockScale& blockScale)
        {
            if(blockScale.getTag().uninitialized())
            {
                blockScale.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {blockScale.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(T_Load_Linear& load)
        {
            if(load.getTag().uninitialized())
            {
                load.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {load.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(T_Load_Scalar& load)
        {
            if(load.getTag().uninitialized())
            {
                load.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {load.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(T_Load_Tiled& load)
        {
            if(load.getTag().uninitialized())
            {
                load.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {load.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(T_Mul& mul)
        {
            if(mul.getTag().uninitialized())
            {
                mul.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {mul.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(T_Store_Linear& store)
        {
            if(store.getTag().uninitialized())
            {
                store.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {store.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(T_Store_Tiled& store)
        {
            if(store.getTag().uninitialized())
            {
                store.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {store.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(T_Execute& exec)
        {
            if(exec.getTag().uninitialized())
            {
                exec.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {exec.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(Nop& nop)
        {
            return {};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::call(XOp&         op,
                                                                    OperationTag nextTagValue)
        {
            m_nextTagValue = nextTagValue;

            return std::visit(*this, op);
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(E_Ternary& ternary)
        {
            if(ternary.getTag().uninitialized())
            {
                ternary.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {ternary.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(E_Binary& binary)
        {
            if(binary.getTag().uninitialized())
            {
                binary.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {binary.getTag()};
        }

        inline std::unordered_set<OperationTag> AssignOutputs::operator()(E_Unary& unary)
        {
            if(unary.getTag().uninitialized())
            {
                unary.setTag(m_nextTagValue);
                m_nextTagValue++;
            }

            return {unary.getTag()};
        }

        inline std::string ToStringVisitor::call(Operation const&     op,
                                                 const unsigned char* runtime_args)
        {
            m_runtimeArgs = runtime_args;
            return std::visit(*this, op);
        }

        inline std::string ToStringVisitor::operator()(Tensor const& tensor)
        {
            return tensor.toString(m_runtimeArgs);
        }

        inline std::string ToStringVisitor::operator()(Scalar const& scalar)
        {
            return scalar.toString(m_runtimeArgs);
        }

        inline std::string ToStringVisitor::operator()(Literal const& literal)
        {
            return literal.toString();
        }

        inline std::string ToStringVisitor::operator()(BlockScale const& blockScale)
        {
            return blockScale.toString();
        }

        inline std::string ToStringVisitor::operator()(T_Load_Linear const& load)
        {
            return load.toString();
        }

        inline std::string ToStringVisitor::operator()(T_Load_Scalar const& load)
        {
            return load.toString();
        }

        inline std::string ToStringVisitor::operator()(T_Load_Tiled const& load)
        {
            return load.toString();
        }

        inline std::string ToStringVisitor::operator()(T_Mul const& mul)
        {
            return mul.toString();
        }

        inline std::string ToStringVisitor::operator()(T_Store_Linear const& store)
        {
            return store.toString();
        }

        inline std::string ToStringVisitor::operator()(T_Store_Tiled const& store)
        {
            return store.toString();
        }

        inline std::string ToStringVisitor::operator()(T_Execute const& exec)
        {
            return exec.toString();
        }

        inline std::string ToStringVisitor::call(XOp const& op)
        {
            return std::visit(*this, op);
        }

        inline std::string ToStringVisitor::operator()(E_Unary const& unary)
        {
            return unary.toString();
        }

        inline std::string ToStringVisitor::operator()(E_Binary const& binary)
        {
            return binary.toString();
        }

        inline std::string ToStringVisitor::operator()(E_Ternary const& ternary)
        {
            return ternary.toString();
        }

        inline std::string ToStringVisitor::operator()(Nop const& exec)
        {
            return "";
        }

        inline SetCommand::SetCommand(CommandPtr com)
            : command(com)
        {
        }

        inline void SetCommand::call(Operation& op)
        {
            std::visit(*this, op);
        }

        inline void SetCommand::operator()(Tensor& tensor)
        {
            tensor.setCommand(command);
        }

        inline void SetCommand::operator()(Scalar& scalar)
        {
            scalar.setCommand(command);
        }

        inline void SetCommand::operator()(Literal& literal)
        {
            literal.setCommand(command);
        }

        inline void SetCommand::operator()(BlockScale& blockScale)
        {
            blockScale.setCommand(command);
        }

        inline void SetCommand::operator()(T_Load_Linear& load)
        {
            load.setCommand(command);
        }

        inline void SetCommand::operator()(T_Load_Scalar& load)
        {
            load.setCommand(command);
        }

        inline void SetCommand::operator()(T_Load_Tiled& load)
        {
            load.setCommand(command);
        }

        inline void SetCommand::operator()(T_Mul& mul)
        {
            mul.setCommand(command);
        }

        inline void SetCommand::operator()(T_Store_Linear& store)
        {
            store.setCommand(command);
        }

        inline void SetCommand::operator()(T_Store_Tiled& store)
        {
            store.setCommand(command);
        }

        inline void SetCommand::operator()(T_Execute& exec)
        {
            exec.setCommand(command);
        }

        inline void SetCommand::operator()(Nop& exec) {}

        inline void AllocateArguments::call(Operation& op)
        {
            return std::visit(*this, op);
        }

        inline void AllocateArguments::operator()(Tensor& tensor)
        {
            tensor.allocateArguments();
        }

        inline void AllocateArguments::operator()(Scalar& scalar)
        {
            scalar.allocateArguments();
        }

        inline void AllocateArguments::operator()(Literal& literal) {}

        inline void AllocateArguments::operator()(BlockScale& blockScale) {}

        inline void AllocateArguments::operator()(T_Load_Linear& load) {}

        inline void AllocateArguments::operator()(T_Load_Scalar& load) {}

        inline void AllocateArguments::operator()(T_Load_Tiled& load) {}

        inline void AllocateArguments::operator()(T_Mul& mul) {}

        inline void AllocateArguments::operator()(T_Store_Linear& store) {}

        inline void AllocateArguments::operator()(T_Store_Tiled& store) {}

        inline void AllocateArguments::operator()(T_Execute& exec) {}

        inline void AllocateArguments::operator()(Nop& nop) {}

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::call(Operation& op)
        {
            return std::visit(*this, op);
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(Tensor& tensor)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(Scalar& scalar)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(Literal& literal)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(BlockScale&)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(T_Load_Linear& load)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(T_Load_Scalar& load)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(T_Load_Tiled& load)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(T_Mul&)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(T_Store_Linear& store)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(T_Store_Tiled& store)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(T_Execute& exec)
        {
            return {rocRoller::DataType::None};
        }

        inline rocRoller::VariableType
            rocRoller::Operations::VariableTypeVisitor::operator()(Nop& nop)
        {
            return {rocRoller::DataType::None};
        }

    }
}
