#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        struct ForLoopOp;
        struct Assign;
        struct UnrollOp;
        struct Barrier;
        struct ElementOp;
        struct LoadLDSTile;
        struct LoadLinear;
        struct LoadVGPR;
        struct LoadTiled;
        struct Multiply;
        struct TensorContraction;
        struct StoreLDSTile;
        struct StoreLinear;
        struct StoreTiled;
        struct StoreVGPR;
        struct Kernel;

        using Operation = std::variant<Kernel,
                                       ForLoopOp,
                                       Assign,
                                       UnrollOp,
                                       Barrier,
                                       ElementOp,
                                       LoadLDSTile,
                                       LoadLinear,
                                       LoadTiled,
                                       LoadVGPR,
                                       Multiply,
                                       TensorContraction,
                                       StoreLDSTile,
                                       StoreLinear,
                                       StoreTiled,
                                       StoreVGPR>;
    }
}
