#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        struct Assign;
        struct Barrier;
        struct ComputeIndex;
        struct Deallocate;
        struct ElementOp;
        struct ForLoopOp;
        struct Kernel;
        struct LoadLDSTile;
        struct LoadLinear;
        struct LoadVGPR;
        struct LoadTiled;
        struct Multiply;
        struct Scope;
        struct SetCoordinate;
        struct StoreLDSTile;
        struct StoreLinear;
        struct StoreTiled;
        struct StoreVGPR;
        struct TensorContraction;
        struct UnrollOp;

        using Operation = std::variant<Assign,
                                       Barrier,
                                       ComputeIndex,
                                       ElementOp,
                                       Deallocate,
                                       ForLoopOp,
                                       Kernel,
                                       LoadLDSTile,
                                       LoadLinear,
                                       LoadTiled,
                                       LoadVGPR,
                                       Multiply,
                                       Scope,
                                       SetCoordinate,
                                       StoreLDSTile,
                                       StoreLinear,
                                       StoreTiled,
                                       StoreVGPR,
                                       TensorContraction,
                                       UnrollOp>;

        template <typename T>
        concept COperation = std::constructible_from<Operation, T>;
    }
}
