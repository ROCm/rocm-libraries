// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

        /**
         * @brief Create a literal expression.
         */
        inline Expression::ExpressionPtr L(auto const& x)
        {
            return Expression::literal(x);
        }

        /**
         * @brief Convert element count to byte count based on data type.
         *
         * @param expr Element count expression
         * @param valueType Data type of elements
         * @return Expression representing byte count
         */
        inline Expression::ExpressionPtr ToBytes(Expression::ExpressionPtr expr, DataType valueType)
        {
            uint numBits = DataTypeInfo::Get(valueType).elementBits;
            Log::debug("  toBytes: {}: numBits {}", toString(valueType), numBits);

            // If numBits is not a multiple of 8, the caller must ensure
            // that (expr * numBits) is a multiple of 8.
            if(numBits % 8u == 0)
                return expr * L(numBits / 8u);
            return (expr * L(numBits)) / L(8u);
        }

        /**
         * @brief Find the corresponding KLoopTail for a given KLoop.
         *
         * Searches downstream via Sequence edges (UnrollLoops case) and
         * for siblings under common parent Scope (AddPrefetch case).
         *
         * @param kgraph The kernel graph
         * @param kLoop The KLoop tag
         * @return The corresponding KLoopTail tag, or std::nullopt if none exists
         */
        std::optional<int> FindCorrespondingKLoopTail(KernelGraph const& kgraph, int kLoop);

        /**
         * @brief Find the closest common ancestor Scope between two nodes.
         *
         * @param kgraph The kernel graph
         * @param nodeA First node
         * @param nodeB Second node
         * @return The tag of the common ancestor Scope, or std::nullopt if none exists
         */
        std::optional<int> FindCommonAncestorScope(KernelGraph const& kgraph, int nodeA, int nodeB);

        /**
         * @brief Get element block values for stride computation.
         *
         * Determines the number of elements per block for load/store operations,
         * which is needed for computing strides correctly for sub-dword types.
         *
         * @param graph The kernel graph
         * @param target Target coordinate
         * @param isTransposed Whether load/store is transposed
         * @return Pair of (elementBlockNumber, elementBlockIndex)
         */
        std::pair<uint, uint>
            GetElementBlockValues(KernelGraph const& graph, int target, bool isTransposed);

        /**
         * @brief Create an Assign node for base offset computation.
         *
         * Computes the initial memory offset for a load/store operation by
         * transforming coordinate indices to byte offsets. Handles LDS padding
         * for transposed loads of sub-dword types.
         *
         * @param graph The kernel graph to modify
         * @param params Index computation parameters
         * @param target Target coordinate (memory location)
         * @param offset Offset coordinate to connect the Assign to
         * @param baseAddress Base address coordinate (-1 if not applicable)
         * @param isLDS Whether target is LDS memory
         * @param isTransposed Whether load/store is transposed
         * @param context GPU context
         * @param command Command containing argument info
         * @param coords Coordinate transformer for index computation
         * @return Tag of the created Assign node
         */
        int MakeAssignBase(KernelGraph&                  graph,
                           IndexComputeParams const&     params,
                           int                           target,
                           int                           offset,
                           int                           baseAddress,
                           bool                          isLDS,
                           bool                          isTransposed,
                           ContextPtr                    context,
                           CommandPtr                    command,
                           CoordinateGraph::Transformer& coords);

        /**
         * @brief Create an Assign node for stride computation.
         *
         * Computes the memory stride for iterating through a tile. Handles
         * special cases for sub-dword types (FP6, FP4, etc.) and transposed
         * LDS loads that require padding.
         *
         * @param graph The kernel graph to modify
         * @param params Index computation parameters
         * @param target Target coordinate (memory location)
         * @param stride Stride coordinate to connect the Assign to
         * @param increment Coordinate representing the loop increment dimension
         * @param isLDS Whether target is LDS memory
         * @param isTransposed Whether load/store is transposed
         * @param context GPU context
         * @param coords Coordinate transformer for stride computation
         * @return Tag of the created Assign node
         */
        int MakeAssignStride(KernelGraph&                  graph,
                             IndexComputeParams const&     params,
                             int                           target,
                             int                           stride,
                             int                           increment,
                             bool                          isLDS,
                             bool                          isTransposed,
                             ContextPtr                    context,
                             CoordinateGraph::Transformer& coords);

        /**
         * @brief Create an Assign node for buffer descriptor.
         *
         * Creates a buffer descriptor for global memory operations, which
         * enables hardware bounds checking. The descriptor contains the
         * base pointer, size, and access options.
         *
         * @param graph The kernel graph to modify
         * @param params Index computation parameters
         * @param target Target coordinate (must have a User coordinate with size set)
         * @param buffer Buffer coordinate to connect the Assign to
         * @param context GPU context
         * @param command Command containing argument info
         * @return Tag of the created Assign node, or -1 if target has no User coordinate
         */
        int MakeBuffer(KernelGraph&              graph,
                       IndexComputeParams const& params,
                       int                       target,
                       int                       buffer,
                       ContextPtr                context,
                       CommandPtr                command);

        /**
         * @brief Detect if a LoadLDSTile candidate's index path contains
         * a non-affine LDS swizzle edge and extract the unroll coordinate
         * and its compile-time value from the upstream SetCoordinate.
         *
         * @param kgraph The kernel graph
         * @param candidate Control graph tag of the candidate operation
         * @return (unrollCoord, unrollValue) or (-1, -1) if not applicable
         */
        std::pair<int, int> GetInlineUnrollInfo(KernelGraph const& kgraph, int candidate);
    } // namespace AssignIndexExpressionsDetail
} // namespace rocRoller::KernelGraph
