#pragma once

#include "../DataTypes/DataTypes.hpp"

#include "AssemblyKernel.hpp"
#include "Base.hpp"
#include "Containers.hpp"
#include "Enum.hpp"
#include "HasTraits.hpp"
#include "Variant.hpp"

#include "Hypergraph.hpp"

#include "../Graph/Hypergraph.hpp"
#include "../KernelGraph/ControlGraph/ControlGraph.hpp"

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
            std::same_as<
                Op,
                KernelGraph::ControlGraph::
                    LoadLinear> || std::same_as<Op, KernelGraph::ControlGraph::LoadTiled> || std::same_as<Op, KernelGraph::ControlGraph::LoadVGPR> || std::same_as<Op, KernelGraph::ControlGraph::LoadLDSTile>) struct
            MappingTraits<Op, IO, Context>
        {
            // If this assertion starts failing, it's likely one of these classes has had a member added.
            static_assert(
                std::same_as<
                    Op,
                    KernelGraph::ControlGraph::
                        LoadVGPR> || sizeof(Op) == sizeof(KernelGraph::ControlGraph::LoadLinear));

            using iot = IOTraits<IO>;
            static void mapping(IO& io, Op& op, Context&)
            {
                iot::mapRequired(io, "varType", op.varType);

                if constexpr(std::same_as<Op, KernelGraph::ControlGraph::LoadVGPR>)
                {
                    iot::mapRequired(io, "scalar", op.scalar);
                }
            }

            static void mapping(IO& io, Op& op)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, op, ctx);
            }
        };

        template <typename Op, typename IO, typename Context>
        requires(
            std::same_as<
                Op,
                KernelGraph::ControlGraph::
                    StoreTiled> || std::same_as<Op, KernelGraph::ControlGraph::StoreLDSTile>) struct
            MappingTraits<Op, IO, Context>
        {
            // If this assertion starts failing, it's likely one of these classes has had a member added.
            static_assert(sizeof(Op) == sizeof(KernelGraph::ControlGraph::StoreTiled));

            using iot = IOTraits<IO>;
            static void mapping(IO& io, Op& op, Context&)
            {
                iot::mapRequired(io, "dataType", op.dataType);
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
            std::map<std::pair<int, int>, KernelGraph::ControlGraph::NodeOrdering> table)
        {
            CompressedTable rv;

            for(auto const& pair : table)
            {
                auto key = toString(pair.second);
                rv[key][pair.first.first].push_back(pair.first.second);
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
