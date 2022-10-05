#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::ControlHypergraph
    {
        struct Assign;
        struct Barrier;
        struct ElementOp;
        struct ForLoopOp;
        struct Kernel;
        struct LoadLDSTile;
        struct LoadLinear;
        struct LoadVGPR;
        struct LoadTiled;
        struct Multiply;
        struct StoreLDSTile;
        struct StoreLinear;
        struct StoreTiled;
        struct StoreVGPR;
        struct TensorContraction;
        struct UnrollOp;

        using Operation = std::variant<Assign,
                                       Barrier,
                                       ElementOp,
                                       ForLoopOp,
                                       Kernel,
                                       LoadLDSTile,
                                       LoadLinear,
                                       LoadTiled,
                                       LoadVGPR,
                                       Multiply,
                                       StoreLDSTile,
                                       StoreLinear,
                                       StoreTiled,
                                       StoreVGPR,
                                       TensorContraction,
                                       UnrollOp>;
    }
}
