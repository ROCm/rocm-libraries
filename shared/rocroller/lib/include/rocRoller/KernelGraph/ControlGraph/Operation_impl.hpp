// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        inline std::string toString(ConditionalMode m)
        {
            switch(m)
            {
            case ConditionalMode::Branch:
                return "Branch";
            case ConditionalMode::Exec:
                return "Exec";
            case ConditionalMode::BranchAndExec:
                return "BranchAndExec";

            case ConditionalMode::Count:
            default:
                break;
            }
            Throw<FatalError>("Invalid ConditionalMode");
        }

        inline std::ostream& operator<<(std::ostream& stream, ConditionalMode m)
        {
            return stream << toString(m);
        }

        template <typename T>
        concept CHasVarTypeMember = requires(T const& op)
        {
            {
                op.varType
                } -> std::convertible_to<VariableType>;
        };

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
                auto rv = op.name();

                if constexpr(CHasVarTypeMember<Op>)
                {
                    rv += " " + toString(op.varType);
                }

                return rv;
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
