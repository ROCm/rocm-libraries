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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/Enum.hpp>
#include <rocRoller/Serialization/HasTraits.hpp>
#include <rocRoller/Serialization/Hypergraph.hpp>
#include <rocRoller/Serialization/Variant.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void
                mapping(IO& io, KernelGraph::CoordinateGraph::BaseDimension& dim, Context& ctx)
            {
                iot::mapRequired(io, "size", dim.size);
                iot::mapRequired(io, "stride", dim.stride);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::BaseDimension& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::LDS, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::LDS& lds, Context& ctx)
            {
                iot::mapRequired(io, "size", lds.size);
                iot::mapRequired(io, "stride", lds.stride);
                iot::mapRequired(io, "isDirect2LDS", lds.isDirect2LDS);
                iot::mapRequired(io, "holdsTransposedTile", lds.holdsTransposedTile);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::LDS& lds)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, lds, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::Adhoc, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::Adhoc& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                // Outputting of `name` should be handled by the variant.

                if(!iot::outputting(io))
                {
                    if constexpr(std::same_as<Context, std::string>)
                    {
                        dim.m_name = ctx;
                    }
                }
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::Adhoc& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::SubDimension, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void
                mapping(IO& io, KernelGraph::CoordinateGraph::SubDimension& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "dim", dim.dim);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::SubDimension& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::User, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::User& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "argumentName", dim.argumentName);
                iot::mapRequired(io, "needsPadding", dim.needsPadding);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::User& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::MacroTile, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::MacroTile& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "rank", dim.rank);
                iot::mapRequired(io, "memoryType", dim.memoryType);
                iot::mapRequired(io, "layoutType", dim.layoutType);

                iot::mapRequired(io, "sizes", dim.sizes);
                iot::mapRequired(io, "subTileSizes", dim.subTileSizes);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::MacroTile& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::ThreadTile, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::ThreadTile& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "rank", dim.rank);

                iot::mapRequired(io, "sizes", dim.sizes);
                iot::mapRequired(io, "wsizes", dim.wsizes);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::ThreadTile& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::WaveTile, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::WaveTile& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "rank", dim.rank);

                iot::mapRequired(io, "sizes", dim.sizes);
                iot::mapRequired(io, "layout", dim.layout);
                iot::mapRequired(io, "vgpr", dim.vgpr);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::WaveTile& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename T, typename IO, typename Context>
        requires(std::constructible_from<KernelGraph::CoordinateGraph::Edge, T>&& T::HasValue
                 == false) struct MappingTraits<T, IO, Context>
            : public EmptyMappingTraits<T, IO, Context>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::Index, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::Index& edge, Context& ctx)
            {
                iot::mapRequired(io, "index", edge.index);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::Index& edge)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, edge, ctx);
            }
        };

        template <typename T, typename IO, typename Context>
        requires(std::constructible_from<KernelGraph::CoordinateGraph::Dimension, T>&&
                     std::derived_from<T, KernelGraph::CoordinateGraph::SubDimension>&& T::HasValue
                 == false) struct MappingTraits<T, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, T& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::SubDimension, IO, Context>::mapping(
                    io, dim, ctx);
            }

            static void mapping(IO& io, T& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename T, typename IO, typename Context>
        requires(
            std::constructible_from<KernelGraph::CoordinateGraph::Dimension, T>&& std::derived_from<
                T,
                KernelGraph::CoordinateGraph::
                    BaseDimension> && !std::derived_from<T, KernelGraph::CoordinateGraph::SubDimension> && T::HasValue == false) struct
            MappingTraits<T, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, T& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);
            }

            static void mapping(IO& io, T& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::CoordinateTransformEdge>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::CoordinateTransformEdge, IO, Context>
            : public DefaultVariantMappingTraits<
                  KernelGraph::CoordinateGraph::CoordinateTransformEdge,
                  IO,
                  Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::DataFlowEdge>);

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::DataFlowEdge, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::CoordinateGraph::DataFlowEdge,
                                                 IO,
                                                 Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::Edge>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::Edge, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::CoordinateGraph::Edge, IO, Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::Dimension>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::Dimension, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::CoordinateGraph::Dimension,
                                                 IO,
                                                 Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::CoordinateGraph::Element>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::CoordinateGraph::Element, IO, Context>
            : public DefaultVariantMappingTraits<
                  KernelGraph::CoordinateGraph::CoordinateGraph::Element,
                  IO,
                  Context>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::CoordinateGraph, IO, Context>
        {
            using iot = IOTraits<IO>;
            using HG  = typename KernelGraph::CoordinateGraph::CoordinateGraph::Base;

            static void
                mapping(IO& io, KernelGraph::CoordinateGraph::CoordinateGraph& graph, Context& ctx)
            {
                MappingTraits<HG, IO, Context>::mapping(io, graph, ctx);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::CoordinateGraph& graph)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, graph, ctx);
            }
        };
    }
}
