// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>

#include <rocRoller/Graph/Hypergraph.hpp>

namespace rocRoller
{
    namespace Graph
    {
        /**
         * `graph` must be an instantiation of Hypergraph which is calm (i.e.
         * not hyper).
         *
         * For each edge in `graph` that matches `edgePredicate`, delete that
         * edge if its destination node would still be reachable from the
         * source, only following edges that satisfy edgePredicate.
         */
        template <CCalmGraph AGraph, std::predicate<int> EdgePredicate>
        void removeRedundantEdges(AGraph& graph, EdgePredicate edgePredicate);

        /**
         * `graph` must be an instantiation of Hypergraph which is calm (i.e.
         * not hyper).
         *
         * For each edge in `graph` that matches `edgePredicate`, delete that
         * edge if its destination node would still be reachable from the
         * source, only following edges that satisfy edgePredicate.
         */
        template <CCalmGraph AGraph, std::predicate<int> EdgePredicate>
        Generator<int> findRedundantEdges(AGraph const& graph, EdgePredicate edgePredicate);
    }
}

#include "GraphUtilities_impl.hpp"
