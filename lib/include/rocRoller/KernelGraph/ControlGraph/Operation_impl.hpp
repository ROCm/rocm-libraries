#pragma once

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        template <CConcreteOperation Op>
        inline std::string name(const Op& x)
        {
            return x.name();
        }

        /*
         * Helpers
         */
        inline std::string name(const Operation& x)
        {
            return std::visit([](const auto& a) { return a.name(); }, x);
        }

        struct OperationToStringVisitor
        {
            template <CHasToStringMember Op>
            std::string operator()(Op const& op)
            {
                return op.toString();
            }

            template <typename Op>
            requires(!CHasToStringMember<Op>) std::string operator()(Op const& op)
            {
                return op.name();
            }
        };

        inline std::string toString(const Operation& x)
        {
            return std::visit(OperationToStringVisitor{}, x);
        }

        template <typename T>
        concept CHasDataTypeMember = requires(T const& op)
        {
            {
                op.dataType
                } -> std::convertible_to<DataType>;
        };

        template <typename T>
        concept CHasVarTypeMember = requires(T const& op)
        {
            {
                op.varType
                } -> std::convertible_to<VariableType>;
        };

        struct OperationDataTypeVisitor
        {
            template <CHasDataTypeMember Op>
            DataType operator()(Op const& op)
            {
                return op.dataType;
            }

            template <CHasVarTypeMember Op>
            DataType operator()(Op const& op)
            {
                return op.varType.dataType;
            }

            template <typename Op>
            requires(!CHasDataTypeMember<Op> && !CHasVarTypeMember<Op>) DataType
                operator()(Op const& op)
            {
                return DataType::None;
            }
        };

        inline DataType getDataType(const Operation& x)
        {
            return std::visit(OperationDataTypeVisitor(), x);
        }
    }
}
