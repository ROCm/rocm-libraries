/**
 *
 */

#pragma once
#include <variant>

namespace rocRoller
{
    namespace Operations
    {
        struct T_Load_Linear;
        struct T_Load_Scalar;
        struct T_Load_Tiled;
        struct T_Mul;
        struct T_Store_Linear;
        struct T_Store_Tiled;
        struct T_Execute;
        struct Nop;
        using Operation = std::variant<T_Load_Linear,
                                       T_Load_Scalar,
                                       T_Load_Tiled,
                                       T_Mul,
                                       T_Store_Linear,
                                       T_Store_Tiled,
                                       T_Execute,
                                       Nop>;

        template <typename T>
        concept COperation = requires
        {
            requires std::constructible_from<Operation, T>;
            requires !std::same_as<Operation, T>;
        };

        struct Inputs;
        struct Outputs;
        struct TagVisitor;
    }
}
