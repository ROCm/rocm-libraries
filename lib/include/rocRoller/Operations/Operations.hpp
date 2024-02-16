/**
 *
 */

#pragma once

#include <unordered_set>
#include <variant>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Operations/T_Execute_fwd.hpp>
#include <rocRoller/Operations/T_LoadStore.hpp>
#include <rocRoller/Operations/T_Mul.hpp>

namespace rocRoller
{
    namespace Operations
    {
        struct Nop
        {
            Nop() {}
            template <typename... Args>
            Nop(Args&&... i)
            {
            }
        };

        struct Inputs
        {
            std::unordered_set<int> call(Operation const&);

            std::unordered_set<int> operator()(T_Load_Linear const&);
            std::unordered_set<int> operator()(T_Load_Scalar const&);
            std::unordered_set<int> operator()(T_Load_Tiled const&);
            std::unordered_set<int> operator()(T_Mul const&);
            std::unordered_set<int> operator()(T_Store_Linear const&);
            std::unordered_set<int> operator()(T_Store_Tiled const&);
            std::unordered_set<int> operator()(T_Execute const&);

            std::unordered_set<int> call(XOp const&);
            std::unordered_set<int> operator()(E_Unary const&);
            std::unordered_set<int> operator()(E_Binary const&);
            std::unordered_set<int> operator()(E_Ternary const&);
            std::unordered_set<int> operator()(Nop const&);
        };

        struct Outputs
        {
            std::unordered_set<int> call(Operation const&);

            std::unordered_set<int> operator()(T_Load_Linear const&);
            std::unordered_set<int> operator()(T_Load_Scalar const&);
            std::unordered_set<int> operator()(T_Load_Tiled const&);
            std::unordered_set<int> operator()(T_Mul const&);
            std::unordered_set<int> operator()(T_Store_Linear const&);
            std::unordered_set<int> operator()(T_Store_Tiled const&);
            std::unordered_set<int> operator()(T_Execute const&);

            std::unordered_set<int> call(XOp const&);
            std::unordered_set<int> operator()(E_Unary const&);
            std::unordered_set<int> operator()(E_Binary const&);
            std::unordered_set<int> operator()(E_Ternary const&);
            std::unordered_set<int> operator()(Nop const&);
        };

        struct TagVisitor
        {
            int call(XOp const&);
            int operator()(E_Unary const&);
            int operator()(E_Binary const&);
            int operator()(E_Ternary const&);
        };

        struct AssignOutputs
        {
            std::unordered_set<int> call(Operation&, int);

            std::unordered_set<int> operator()(T_Load_Linear&);
            std::unordered_set<int> operator()(T_Load_Scalar&);
            std::unordered_set<int> operator()(T_Load_Tiled&);
            std::unordered_set<int> operator()(T_Mul&);
            std::unordered_set<int> operator()(T_Store_Linear&);
            std::unordered_set<int> operator()(T_Store_Tiled&);
            std::unordered_set<int> operator()(T_Execute&);

            std::unordered_set<int> call(XOp&, int);
            std::unordered_set<int> operator()(E_Unary&);
            std::unordered_set<int> operator()(E_Binary&);
            std::unordered_set<int> operator()(E_Ternary&);
            std::unordered_set<int> operator()(Nop&);

        private:
            int m_nextTagValue = -1;
        };

        struct ToStringVisitor
        {
            std::string call(Operation const&, const unsigned char*);

            std::string operator()(T_Load_Linear const&);
            std::string operator()(T_Load_Scalar const&);
            std::string operator()(T_Load_Tiled const&);
            std::string operator()(T_Mul const&);
            std::string operator()(T_Store_Linear const&);
            std::string operator()(T_Store_Tiled const&);
            std::string operator()(T_Execute const&);

            std::string call(XOp const&);
            std::string operator()(E_Unary const&);
            std::string operator()(E_Binary const&);
            std::string operator()(E_Ternary const&);
            std::string operator()(Nop const&);

        private:
            const unsigned char* m_runtimeArgs;
        };

        struct SetCommand
        {
            SetCommand(CommandPtr);

            void call(Operation&);

            void operator()(T_Load_Linear&);
            void operator()(T_Load_Scalar&);
            void operator()(T_Load_Tiled&);
            void operator()(T_Mul&);
            void operator()(T_Store_Linear&);
            void operator()(T_Store_Tiled&);
            void operator()(T_Execute&);
            void operator()(Nop&);

            CommandPtr command;
        };

        struct AllocateArguments
        {
            void call(Operation&);

            void operator()(T_Load_Linear&);
            void operator()(T_Load_Scalar&);
            void operator()(T_Load_Tiled&);
            void operator()(T_Mul&);
            void operator()(T_Store_Linear&);
            void operator()(T_Store_Tiled&);
            void operator()(T_Execute&);
            void operator()(Nop&);
        };

        struct VariableTypeVisitor
        {
            rocRoller::VariableType call(Operation&);

            rocRoller::VariableType operator()(T_Load_Linear&);
            rocRoller::VariableType operator()(T_Load_Scalar&);
            rocRoller::VariableType operator()(T_Load_Tiled&);
            rocRoller::VariableType operator()(T_Mul&);
            rocRoller::VariableType operator()(T_Store_Linear&);
            rocRoller::VariableType operator()(T_Store_Tiled&);
            rocRoller::VariableType operator()(T_Execute&);
            rocRoller::VariableType operator()(Nop&);
        };

    }
}

#include "Operations_impl.hpp"
