// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
        class Tensor;
        class Scalar;
        class Literal;
        class BlockScale;
        class Scratch;
        class SubTileTranspose;
        class T_Load_Linear;
        class T_Load_Scalar;
        class T_Load_Tiled;
        class T_Mul;
        class T_Store_Linear;
        class T_Store_Tiled;
        class T_Execute;
        struct Nop;
        class RandomNumberGenerator;
        using Operation = std::variant<Tensor,
                                       Scalar,
                                       Literal,
                                       BlockScale,
                                       Scratch,
                                       SubTileTranspose,
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
