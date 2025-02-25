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
        struct AssertOp;
        struct Deallocate;
        struct DoWhileOp;
        struct ForLoopOp;
        struct Kernel;
        struct LoadLDSTile;
        struct LoadLinear;
        struct LoadVGPR;
        struct LoadSGPR;
        struct LoadTiled;
        struct Multiply;
        struct NOP;
        struct Block;
        struct Scope;
        struct SetCoordinate;
        struct StoreLDSTile;
        struct StoreLinear;
        struct StoreTiled;
        struct StoreVGPR;
        struct StoreSGPR;
        struct TensorContraction;
        struct UnrollOp;
        struct WaitZero;
        struct SeedPRNG;

        using Operation = std::variant<Assign,
                                       Barrier,
                                       ComputeIndex,
                                       ConditionalOp,
                                       AssertOp,
                                       Deallocate,
                                       DoWhileOp,
                                       ForLoopOp,
                                       Kernel,
                                       LoadLDSTile,
                                       LoadLinear,
                                       LoadTiled,
                                       LoadVGPR,
                                       LoadSGPR,
                                       Multiply,
                                       NOP,
                                       Block,
                                       Scope,
                                       SetCoordinate,
                                       StoreLDSTile,
                                       StoreLinear,
                                       StoreTiled,
                                       StoreVGPR,
                                       StoreSGPR,
                                       TensorContraction,
                                       UnrollOp,
                                       WaitZero,
                                       SeedPRNG>;

        template <typename T>
        concept COperation = std::constructible_from<Operation, T>;

        template <typename T>
        concept CConcreteOperation = (COperation<T> && !std::same_as<Operation, T>);
    }
}
