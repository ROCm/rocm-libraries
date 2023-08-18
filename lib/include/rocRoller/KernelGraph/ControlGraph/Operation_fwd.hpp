#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        struct Assign;
        struct Barrier;
        struct ComputeIndex;
        struct ConditionalOp;
        struct Deallocate;
        struct DoWhileOp;
        struct ForLoopOp;
        struct Kernel;
        struct LoadLDSTile;
        struct LoadLinear;
        struct LoadVGPR;
        struct LoadTiled;
        struct Multiply;
        struct NOP;
        struct Scope;
        struct SetCoordinate;
        struct StoreLDSTile;
        struct StoreLinear;
        struct StoreTiled;
        struct StoreVGPR;
        struct TensorContraction;
        struct UnrollOp;
        struct WaitZero;

        using Operation = std::variant<Assign,
                                       Barrier,
                                       ComputeIndex,
                                       ConditionalOp,
                                       Deallocate,
                                       DoWhileOp,
                                       ForLoopOp,
                                       Kernel,
                                       LoadLDSTile,
                                       LoadLinear,
                                       LoadTiled,
                                       LoadVGPR,
                                       Multiply,
                                       NOP,
                                       Scope,
                                       SetCoordinate,
                                       StoreLDSTile,
                                       StoreLinear,
                                       StoreTiled,
                                       StoreVGPR,
                                       TensorContraction,
                                       UnrollOp,
                                       WaitZero>;

        template <typename T>
        concept COperation = std::constructible_from<Operation, T>;

        template <typename T>
        concept CConcreteOperation = (COperation<T> && !std::same_as<Operation, T>);
    }
}
