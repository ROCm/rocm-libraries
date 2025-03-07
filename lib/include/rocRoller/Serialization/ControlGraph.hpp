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

#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Serialization/AssemblyKernel.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/Enum.hpp>
#include <rocRoller/Serialization/HasTraits.hpp>
#include <rocRoller/Serialization/Variant.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename T, typename IO, typename Context>
        requires(std::constructible_from<KernelGraph::ControlGraph::ControlEdge,
                                         T>) struct MappingTraits<T, IO, Context>
            : public EmptyMappingTraits<T, IO, Context>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<BufferInstructionOptions, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, BufferInstructionOptions& opt, Context&)
            {
                iot::mapRequired(io, "offen", opt.offen);
                iot::mapRequired(io, "glc", opt.glc);
                iot::mapRequired(io, "slc", opt.slc);
                iot::mapRequired(io, "sc1", opt.sc1);
                iot::mapRequired(io, "lds", opt.lds);
            }

            static void mapping(IO& io, BufferInstructionOptions& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::SetCoordinate, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::SetCoordinate& op, Context& ctx)
            {
                iot::mapRequired(io, "value", op.value);

                std::string valueStr;
                if(iot::outputting(io))
                    valueStr = toString(op.value);

                iot::mapRequired(io, "valueStr", valueStr);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::SetCoordinate& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::ForLoopOp, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::ForLoopOp& op, Context&)
            {
                iot::mapRequired(io, "loopName", op.loopName);
                iot::mapRequired(io, "condition", op.condition);

                std::string conditionStr;
                if(iot::outputting(io))
                    conditionStr = toString(op.condition);

                iot::mapRequired(io, "conditionStr", conditionStr);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::ForLoopOp& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::ConditionalOp, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::ConditionalOp& op, Context&)
            {
                iot::mapRequired(io, "conditionName", op.conditionName);
                iot::mapRequired(io, "condition", op.condition);

                std::string conditionStr;
                if(iot::outputting(io))
                    conditionStr = toString(op.condition);

                iot::mapRequired(io, "conditionStr", conditionStr);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::ConditionalOp& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::AssertOp, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::AssertOp& op, Context&)
            {
                iot::mapRequired(io, "assertName", op.assertName);
                iot::mapRequired(io, "condition", op.condition);

                std::string conditionStr;
                if(iot::outputting(io))
                    conditionStr = toString(op.condition);

                iot::mapRequired(io, "conditionStr", conditionStr);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::AssertOp& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::DoWhileOp, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::DoWhileOp& op, Context&)
            {
                iot::mapRequired(io, "loopName", op.loopName);
                iot::mapRequired(io, "condition", op.condition);

                std::string conditionStr;
                if(iot::outputting(io))
                    conditionStr = toString(op.condition);

                iot::mapRequired(io, "conditionStr", conditionStr);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::DoWhileOp& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::UnrollOp, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::UnrollOp& op, Context&)
            {
                iot::mapRequired(io, "size", op.size);

                std::string sizeStr;
                if(iot::outputting(io))
                    sizeStr = toString(op.size);

                iot::mapRequired(io, "sizeStr", sizeStr);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::UnrollOp& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::SeedPRNG, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::SeedPRNG& op, Context&)
            {
                iot::mapRequired(io, "addTID", op.addTID);

                std::string addTIDStr;
                if(iot::outputting(io))
                    addTIDStr = toString(op.addTID);

                iot::mapRequired(io, "addTIDStr", addTIDStr);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::SeedPRNG& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::Assign, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::Assign& op, Context&)
            {
                iot::mapRequired(io, "regType", op.regType);
                iot::mapRequired(io, "expression", op.expression);
                iot::mapRequired(io, "valueCount", op.valueCount);

                std::string expressionStr;
                if(iot::outputting(io))
                    expressionStr = toString(op.expression);

                iot::mapRequired(io, "expressionStr", expressionStr);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::Assign& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::ComputeIndex, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::ComputeIndex& op, Context&)
            {
                iot::mapRequired(io, "forward", op.forward);
                iot::mapRequired(io, "valueType", op.valueType);
                iot::mapRequired(io, "offsetType", op.offsetType);
                iot::mapRequired(io, "strideType", op.strideType);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::ComputeIndex& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename Op, typename IO, typename Context>
        requires(
            CIsAnyOf<Op,
                     KernelGraph::ControlGraph::Exchange,
                     KernelGraph::ControlGraph::LoadLinear,
                     KernelGraph::ControlGraph::LoadTiled,
                     KernelGraph::ControlGraph::LoadVGPR,
                     KernelGraph::ControlGraph::LoadSGPR,
                     KernelGraph::ControlGraph::LoadTileDirect2LDS,
                     KernelGraph::ControlGraph::LoadLDSTile>) struct MappingTraits<Op, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, Op& op, Context&)
            {
                iot::mapRequired(io, "varType", op.varType);

                if constexpr(std::same_as<Op, KernelGraph::ControlGraph::LoadVGPR>)
                {
                    iot::mapRequired(io, "scalar", op.scalar);
                }
                else if constexpr(std::same_as<Op, KernelGraph::ControlGraph::LoadSGPR>)
                {
                    //iot::mapRequired(io, "bufOpts", op.bufOpts);
                }
                else if constexpr(CIsAnyOf<Op,
                                           KernelGraph::ControlGraph::LoadTiled,
                                           KernelGraph::ControlGraph::LoadLDSTile>)
                {
                    iot::mapRequired(io, "isTransposedTile", op.isTransposedTile);
                }
                else
                {
                    // For all Ops other than above, they should have the same memeber (size).
                    // If this assertion fails, the Op might have new members added.
                    static_assert(sizeof(Op) == sizeof(KernelGraph::ControlGraph::LoadLinear));
                }
            }

            static void mapping(IO& io, Op& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::Multiply, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::Multiply& op, Context&)
            {
                iot::mapRequired(io, "scaleA", op.scaleA);
                iot::mapRequired(io, "scaleB", op.scaleB);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::Multiply& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename Op, typename IO, typename Context>
        requires(
            CIsAnyOf<Op,
                     KernelGraph::ControlGraph::StoreTiled,
                     KernelGraph::ControlGraph::StoreSGPR,
                     KernelGraph::ControlGraph::StoreLDSTile>) struct MappingTraits<Op, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, Op& op, Context&)
            {
                iot::mapRequired(io, "dataType", op.dataType);
                if constexpr(std::same_as<Op, KernelGraph::ControlGraph::StoreSGPR>)
                {
                    iot::mapRequired(io, "bufOpts", op.bufOpts);
                }
                if constexpr(std::same_as<Op, KernelGraph::ControlGraph::StoreTiled>)
                {
                    iot::mapRequired(io, "bufOpts", op.bufOpts);
                }
            }

            static void mapping(IO& io, Op& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::TensorContraction, IO, Context>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, KernelGraph::ControlGraph::TensorContraction& op, Context&)
            {
                iot::mapRequired(io, "aDims", op.aDims);
                iot::mapRequired(io, "bDims", op.bDims);
                iot::mapRequired(io, "scaleModeA", op.scaleModeA);
                iot::mapRequired(io, "scaleModeB", op.scaleModeB);
                iot::mapRequired(io, "scaleStridesA", op.scaleStridesA);
                iot::mapRequired(io, "scaleStridesB", op.scaleStridesB);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::TensorContraction& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename T, typename IO, typename Context>
        requires(std::constructible_from<KernelGraph::ControlGraph::Operation, T>&& T::HasValue
                 == false) struct MappingTraits<T, IO, Context>
            : public EmptyMappingTraits<T, IO, Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::ControlGraph::ControlEdge>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::ControlEdge, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::ControlGraph::ControlEdge,
                                                 IO,
                                                 Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::ControlGraph::Operation>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::Operation, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::ControlGraph::Operation, IO, Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::ControlGraph::ControlGraph::Element>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::ControlGraph::Element, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::ControlGraph::ControlGraph::Element,
                                                 IO,
                                                 Context>
        {
        };

        /**
         * Compress the node ordering table to reduce duplication.
         * Format is:
         *
         * {NodeOrdering: {left: [right, right, right]}}
         *
         * e.g.:
         *   LeftFirst:
         *     27: [944, 1371, 1377]
         *     106: [358, 813, 944, 1371, 1377]
         *   RightFirst:
         *     11: [62, 798]
         *
         * As with the storage in the ControlGraph, the left entry is always less than the
         * right entry.
         */

        using CompressedTableEntry = std::map<int, std::vector<int>>;
        using CompressedTable      = std::map<std::string, CompressedTableEntry>;

        inline CompressedTable compressedNodeOrderTable(
            std::unordered_map<std::tuple<int, int>, KernelGraph::ControlGraph::NodeOrdering> table)
        {
            CompressedTable rv;

            for(auto const& pair : table)
            {
                auto key = toString(pair.second);
                rv[key][std::get<0>(pair.first)].push_back(std::get<1>(pair.first));
            }

            return rv;
        }

        template <typename IO>
        struct CustomMappingTraits<CompressedTable, IO>
            : public DefaultCustomMappingTraits<CompressedTable, IO, false, false>
        {
        };
        template <typename IO>
        struct CustomMappingTraits<CompressedTableEntry, IO>
            : public DefaultCustomMappingTraits<CompressedTableEntry, IO, false, false>
        {
        };

#ifdef ROCROLLER_USE_YAML_CPP
        // LLVM serialization defines traits for vector<int>.
        ROCROLLER_SERIALIZE_VECTOR(true, int);
#endif

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::ControlGraph::ControlGraph, IO, Context>
        {
            using iot = IOTraits<IO>;
            using HG  = typename KernelGraph::ControlGraph::ControlGraph::Base;

            static void
                mapping(IO& io, KernelGraph::ControlGraph::ControlGraph& graph, Context& ctx)
            {
                MappingTraits<HG, IO, Context>::mapping(io, graph, ctx);

                CompressedTable table;

                if(iot::outputting(io))
                {
                    table = compressedNodeOrderTable(graph.nodeOrderTable());
                }

                iot::mapOptional(io, "orderTable", table);
            }

            static void mapping(IO& io, KernelGraph::ControlGraph::ControlGraph& graph)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, graph, ctx);
            }
        };
    }
}
