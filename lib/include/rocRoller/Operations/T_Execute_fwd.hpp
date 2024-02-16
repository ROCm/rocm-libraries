#pragma once

#include <variant>

namespace rocRoller
{
    namespace Operations
    {
        struct T_Execute;
        struct E_Unary;
        struct E_Binary;
        struct E_Ternary;
        struct E_Neg;
        struct E_Abs;
        struct E_Not;
        struct E_Add;
        struct E_Sub;
        struct E_Mul;
        struct E_Div;
        struct E_And;
        struct E_Or;
        struct E_GreaterThan;
        struct E_Conditional;

        using XOp = std::variant<E_Neg,
                                 E_Abs,
                                 E_Not,
                                 E_Add,
                                 E_Sub,
                                 E_Mul,
                                 E_Div,
                                 E_And,
                                 E_Or,
                                 E_GreaterThan,
                                 E_Conditional>;
    }
}
