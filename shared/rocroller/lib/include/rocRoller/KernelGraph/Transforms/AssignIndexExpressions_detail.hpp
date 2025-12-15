/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller::KernelGraph
{
    namespace AssignIndexExpressionsDetail
    {
        /**
         * @brief Parameters for index computation.
         */
        struct IndexComputeParams
        {
            bool     forward                  = false;
            bool     isStorePartOfGlobalToLDS = false;
            DataType valueType                = DataType::Count;
            DataType offsetType               = DataType::Count;
            DataType strideType               = DataType::Count;
        };

        inline Expression::ExpressionPtr L(auto const& x)
        {
            return Expression::literal(x);
        }

        /**
         * @brief Get element block values for stride computation.
         *
         * @param graph
         * @param target Target coordinate
         * @param isTransposed Whether load/store is transposed
         * @return Pair of (elementBlockNumber, elementBlockIndex)
         */
        std::pair<uint, uint>
            getElementBlockValues(KernelGraph const& graph, int target, const bool isTransposed);

        /**
         * @brief Create an Assign node for base offset computation.
         *
         * @param graph
         * @param params Index computation parameters
         * @param target Target coordinate
         * @param offset Offset coordinate
         * @param maybeLDS Whether target is LDS
         * @param isTransposed Whether load/store is transposed
         * @param context
         * @param coords Coordinate transformer
         * @return Tag of the created Assign node
         */
        int makeAssignBase(KernelGraph&                  graph,
                           IndexComputeParams const&     params,
                           int                           target,
                           int                           offset,
                           bool                          maybeLDS,
                           bool                          isTransposed,
                           ContextPtr                    context,
                           CoordinateGraph::Transformer& coords);

        /**
         * @brief Create an Assign node for stride computation.
         *
         * @param graph
         * @param params Index computation parameters
         * @param target Target coordinate
         * @param stride Stride coordinate
         * @param increment Increment coordinate
         * @param maybeLDS Whether target is LDS
         * @param isTransposed Whether load/store is transposed
         * @param context
         * @param coords Coordinate transformer
         * @return Tag of the created Assign node
         */
        int makeAssignStride(KernelGraph&                  graph,
                             IndexComputeParams const&     params,
                             int                           target,
                             int                           stride,
                             int                           increment,
                             bool                          maybeLDS,
                             bool                          isTransposed,
                             ContextPtr                    context,
                             CoordinateGraph::Transformer& coords);

        /**
         * @brief Create an Assign node for buffer descriptor.
         *
         * @param graph
         * @param params Index computation parameters
         * @param target Target coordinate
         * @param buffer Buffer coordinate
         * @param context
         * @param command
         * @return Tag of the created Assign node, or -1 if target has no User coordinate
         */
        int makeBuffer(KernelGraph&              graph,
                       IndexComputeParams const& params,
                       int                       target,
                       int                       buffer,
                       ContextPtr                context,
                       const CommandPtr          command);

    } // namespace AssignIndexExpressionsDetail
} // namespace rocRoller::KernelGraph
