/**
 *
 */

#pragma once
#include <string>
#include <variant>

namespace rocRoller
{
    namespace Operations
    {
        struct Tensor;
        struct Scalar;
        struct Literal;
        struct BlockScale;
        struct T_Load_Linear;
        struct T_Load_Scalar;
        struct T_Load_Tiled;
        struct T_Mul;
        struct T_Store_Linear;
        struct T_Store_Tiled;
        struct T_Execute;
        struct Nop;
        struct RandomNumberGenerator;
        using Operation = std::variant<Tensor,
                                       Scalar,
                                       Literal,
                                       BlockScale,
                                       T_Load_Linear,
                                       T_Load_Scalar,
                                       T_Load_Tiled,
                                       T_Mul,
                                       T_Store_Linear,
                                       T_Store_Tiled,
                                       T_Execute,
                                       Nop,
                                       RandomNumberGenerator>;

        template <typename T>
        concept COperation = std::constructible_from<Operation, T>;

        template <typename T>
        concept CConcreteOperation = (COperation<T> && !std::same_as<Operation, T>);

        struct Inputs;
        struct Outputs;
        struct TagVisitor;

        std::string name(Operation const&);

        template <CConcreteOperation T>
        std::string name();

    }
}
