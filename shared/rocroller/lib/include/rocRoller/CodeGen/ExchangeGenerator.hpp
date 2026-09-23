// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class ExchangeGenerator
        {
        public:
            ExchangeGenerator(KernelGraphPtr, ContextPtr);

            Generator<Instruction> genExchange(int                           tag,
                                               ControlGraph::Exchange const& exchange,
                                               CoordinateGraph::Transformer  coords);

        private:
            ContextPtr                       m_context;
            KernelGraphPtr                   m_graph;
            Expression::ExpressionTransducer m_fastArith;
        };
    }
}
