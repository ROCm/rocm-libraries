#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/TagType.hpp>

#include "Dimension_fwd.hpp"
#include "InstructionValues/Register_fwd.hpp"
#include "Utilities/Utils.hpp"

namespace rocRoller
{
    namespace KernelGraph::CoordinateTransform
    {
        /*
         * Nodes (Dimensions)
         */

        struct BaseDimension
        {
            int                       tag;
            bool                      output;
            Expression::ExpressionPtr size, stride;

            BaseDimension() = delete;

            BaseDimension(int tag)
                : tag(tag)
                , size(nullptr)
                , stride(nullptr)
                , output(false)
            {
            }

            BaseDimension(int tag, bool output)
                : tag(tag)
                , size(nullptr)
                , stride(nullptr)
                , output(output)
            {
            }

            BaseDimension(int                       tag,
                          Expression::ExpressionPtr size,
                          Expression::ExpressionPtr stride,
                          bool                      output = false)
                : tag(tag)
                , size(size)
                , stride(stride)
                , output(output)
            {
            }

            std::string toString() const;

            virtual std::string name() const = 0;
        };

        /**
         * Adhoc - represents a temporary "internal" dimension.
         *
         * Dimensions in the Coordinate Transform graph often have C++
         * structs associated with them.  This facilitates writing
         * visitors, querying the graph, and operations like setting
         * coordinates.
         *
         * For dimensions that are specific (or "internal") to a given
         * coordinate transform, and that won't need to be referenced
         * in other parts of the code, the Adhoc dimension can be
         * used.
         */
        struct Adhoc : public BaseDimension
        {
            Adhoc() = delete;

            /**
             * Create an Adhoc dimension with a specific name, command
             * tag, size and stride.
             */
            Adhoc(std::string               name,
                  int                       tag,
                  Expression::ExpressionPtr size,
                  Expression::ExpressionPtr stride,
                  bool                      output = false)
                : BaseDimension(tag, size, stride, output)
                , m_name(name)
            {
                m_hash = std::hash<std::string>()(m_name);
            }

            /**
             * Create an Adhoc dimension with a specific name and
             * command tag.
             */
            Adhoc(std::string name, int tag, bool output = false)
                : Adhoc(name, tag, nullptr, nullptr, output)
            {
            }

            virtual std::string name() const override
            {
                return m_name;
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0, m_hash};
            }

        private:
            size_t      m_hash;
            std::string m_name;
        };

        /**
         * SubDimension - represents a single dimension of a tensor.
         *
         * Encodes size and stride info.
         */
        struct SubDimension : public BaseDimension
        {
            int dim;

            SubDimension() = delete;
            SubDimension(int                       tag,
                         int                       dim,
                         Expression::ExpressionPtr size,
                         Expression::ExpressionPtr stride,
                         bool                      output = false)
                : BaseDimension(tag, size, stride, output)
                , dim(dim)
            {
            }

            SubDimension(int tag, int dim, bool output = false)
                : BaseDimension(tag, output)
                , dim(dim)
            {
            }

            SubDimension(int tag)
                : BaseDimension(tag, false)
                , dim(0)
            {
            }

            virtual std::string toString() const;

            virtual std::string name() const
            {
                return "SubDimension";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }
        };

        /**
         * User - represents tensor from the user.
         *
         * Usually split into SubDimensions.  The subdimensions carry
         * sizes and strides.
         */
        struct User : public BaseDimension
        {
            using BaseDimension::BaseDimension;

            User(int tag, std::string name, bool output = false)
                : BaseDimension(tag, output)
                , m_argument_name(name)
            {
            }

            virtual std::string name() const override
            {
                return "User";
            }

            std::string argumentName() const
            {
                return m_argument_name;
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }

        private:
            std::string m_argument_name;
        };

        /**
         * Linear dimension.  Usually flattened subdimenions.
         */
        struct Linear : public BaseDimension
        {
            Linear(int tag)
                : BaseDimension(tag, false)
            {
            }

            Linear(int tag, bool output)
                : BaseDimension(tag, output)
            {
            }

            Linear(int                       tag,
                   Expression::ExpressionPtr size,
                   Expression::ExpressionPtr stride,
                   bool                      output = false)
                : BaseDimension(tag, size, stride, output)
            {
            }

            virtual std::string name() const override
            {
                return "Linear";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }
        };

        /**
         * Wavefront - represents wavefronts within a workgroup.
         */
        struct Wavefront : public SubDimension
        {
            using SubDimension::SubDimension;

            virtual std::string name() const override
            {
                return "Wavefront";
            }
        };

        /**
         * Lane - represents a lane within a wavefront.
         */
        struct Lane : public BaseDimension
        {
            using BaseDimension::BaseDimension;

            virtual std::string name() const override
            {
                return "Lane";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }
        };

        /**
         * Workgroup - typically represents workgroups on a GPU.
         *
         * Sub-dimensions 0, 1, and 2 coorespond to the x, y and z
         * kernel launch dimensions.
         */
        struct Workgroup : public SubDimension
        {
            Workgroup(int tag, int dim = 0, bool output = false)
                : SubDimension(tag, dim, output)
            {
            }

            virtual std::string name() const override
            {
                return "Workgroup";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }

            virtual std::string toString() const override
            {
                return SubDimension::toString();
            }
        };

        /**
         * Workitem - typically represents threads within a workgroup.
         *
         * Sub-dimensions 0, 1, and 2 coorespond to the x, y and z
         * kernel launch dimensions.
         */
        struct Workitem : public SubDimension
        {
            Workitem(int                       tag,
                     int                       dim    = 0,
                     Expression::ExpressionPtr size   = nullptr,
                     bool                      output = false)
                : SubDimension(tag, dim, size, Expression::literal(1u), output)
            {
            }

            virtual std::string name() const override
            {
                return "Workitem";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }
        };

        /**
         * VGPR - represents (small) thread local scalar/array.
         */
        struct VGPR : public BaseDimension
        {
            using BaseDimension::BaseDimension;

            virtual std::string name() const override
            {
                return "VGPR";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }
        };

        /**
         * LDS - represents local memory.
         */
        struct LDS : public BaseDimension
        {
            using BaseDimension::BaseDimension;

            virtual std::string name() const override
            {
                return "LDS";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }
        };

        /**
         * ForLoop -
         */
        struct ForLoop : public BaseDimension
        {
            using BaseDimension::BaseDimension;

            virtual std::string name() const override
            {
                return "ForLoop";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }
        };

        struct Unroll : public BaseDimension
        {
            Unroll() = delete;

            Unroll(int tag)
                : BaseDimension(tag, false)
            {
            }

            Unroll(int tag, uint usize)
                : BaseDimension(tag, nullptr, nullptr)
            {
                size   = rocRoller::Expression::literal(usize);
                stride = rocRoller::Expression::literal(1);
            }

            Unroll(int tag, Expression::ExpressionPtr usize)
                : BaseDimension(tag, nullptr, nullptr)
            {
                size   = usize;
                stride = rocRoller::Expression::literal(1);
            }

            virtual std::string name() const override
            {
                return "Unroll";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }
        };
        /**
         * MacroTileIndex - sub-dimension of a tile.  See MacroTile.
         */
        struct MacroTileIndex : public SubDimension
        {
            using SubDimension::SubDimension;

            virtual std::string name() const
            {
                return "MacroTileIndex";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }
        };

        /**
         * MacroTileNumber.  See MacroTile.
         */
        struct MacroTileNumber : public SubDimension
        {
            using SubDimension::SubDimension;

            virtual std::string name() const
            {
                return "MacroTileNumber";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }
        };

        /**
         * MacroTile - a tensor tile owned by a workgroup.
         *
         * The storage location (eg, VGPRs vs LDS) is specified by
         * `MemoryType`.
         */
        struct MacroTile : public BaseDimension
        {
            int        rank;
            MemoryType memoryType;
            LayoutType layoutType;

            std::vector<int> sizes;

            MacroTile() = delete;

            /**
             * Construct MacroTile dimension with deferred rank etc.
             */
            MacroTile(int tag)
                : BaseDimension(tag, false)
                , rank(0)
                , memoryType(MemoryType::None)
                , layoutType(LayoutType::None)
            {
            }

            /**
             * Construct MacroTile dimension with deferred sizes and
             * memory type.
             */
            MacroTile(int tag, int rank, bool output = false)
                : BaseDimension(tag, output)
                , rank(rank)
                , memoryType(MemoryType::None)
                , layoutType(LayoutType::None)
            {
            }

            /**
             * Construct MacroTile dimension with fully specified sizes
             * and memory type (ie, LDS vs VGPR).
             */
            MacroTile(int              tag,
                      std::vector<int> sizes,
                      MemoryType       memoryType,
                      std::vector<int> subTileSizes = {},
                      bool             output       = false)
                : BaseDimension(tag, output)
                , rank(sizes.size())
                , sizes(sizes)
                , memoryType(memoryType)
                , layoutType(LayoutType::None)
                , subTileSizes(subTileSizes)
            {
            }

            /**
             * Construct MacroTile dimension with fully specified sizes
             * and memory type (ie, LDS vs VGPR).
             *
             * Memory type is WAVE.
             */
            MacroTile(int              tag,
                      std::vector<int> sizes,
                      LayoutType       layoutType,
                      std::vector<int> subTileSizes = {},
                      bool             output       = false)
                : BaseDimension(tag, output)
                , rank(sizes.size())
                , sizes(sizes)
                , memoryType(MemoryType::WAVE)
                , layoutType(layoutType)
                , subTileSizes(subTileSizes)
            {
                AssertFatal(layoutType != LayoutType::None, "Invalid layout type.");
            }

            virtual std::string name() const
            {
                return "MacroTile";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }

            /**
             * Return MacroTileNumber cooresponding to sub-dimension `sdim` of this tile.
             */
            MacroTileNumber tileNumber(int sdim, bool output = false) const;

            /**
             * Return MacroTileIndex cooresponding to sub-dimension `sdim` of this tile.
             */
            MacroTileIndex tileIndex(int sdim, bool output = false) const;

            /**
             * Return total number of elements.
             */
            int elements() const;

            /**
             * Size of thread tiles.
             *
             * Sizes of -1 represent a "to be determined size".
             */
            std::vector<int> subTileSizes;
        };

        /**
         * ThreadTileIndex - sub-dimension of a tile.  See ThreadTile.
         */
        struct ThreadTileIndex : public SubDimension
        {
            using SubDimension::SubDimension;

            virtual std::string name() const
            {
                return "ThreadTileIndex";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }
        };

        /**
         * ThreadTileNumber.  See ThreadTile.
         */
        struct ThreadTileNumber : public SubDimension
        {
            using SubDimension::SubDimension;

            virtual std::string name() const
            {
                return "ThreadTileNumber";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }
        };

        /**
         * ThreadTile - a tensor tile owned by a thread.
         *
         * The storage location (eg, VGPRs vs LDS) is specified by
         * `MemoryType`.
         */
        struct ThreadTile : public BaseDimension
        {
            int rank;

            // -1 is used to represent a "to be determined" size.
            std::vector<int> sizes;

            ThreadTile() = delete;

            /**
             * Construct ThreadTile dimension with fully specified sizes
             * and memory type (ie, LDS vs VGPR).
             */
            ThreadTile(int tag, std::vector<int> sizes, bool output = false)
                : BaseDimension(tag, output)
                , rank(sizes.size())
                , sizes(sizes)
            {
            }

            virtual std::string name() const
            {
                return "ThreadTile";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }

            /**
             * Return ThreadTileNumber cooresponding to sub-dimension `sdim` of this tile.
             */
            ThreadTileNumber tileNumber(int sdim, bool output = false) const;

            /**
             * Return ThreadTileIndex cooresponding to sub-dimension `sdim` of this tile.
             */
            ThreadTileIndex tileIndex(int sdim, bool output = false) const;
        };

        /**
         * WaveTileIndex - sub-dimension of a tile.  See WaveTile.
         */
        struct WaveTileIndex : public SubDimension
        {
            using SubDimension::SubDimension;

            virtual std::string name() const
            {
                return "WaveTileIndex";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }
        };

        /**
         * WaveTileNumber.  See WaveTile.
         */
        struct WaveTileNumber : public SubDimension
        {
            using SubDimension::SubDimension;

            virtual std::string name() const
            {
                return "WaveTileNumber";
            }

            TagType getTag() const
            {
                return {tag, dim, output, 0};
            }
        };

        /**
         * WaveTile - a tensor tile owned by a wave in GPRs.
         */
        struct WaveTile : public BaseDimension
        {
            int rank;

            std::vector<int>   sizes;
            LayoutType         layout;
            Register::ValuePtr vgpr; // TODO: Does this belong here?  Move to "getVGPR"?

            WaveTile() = delete;

            /**
             * Construct WaveTile dimension with deferred rank and size.
             */
            WaveTile(int tag)
                : BaseDimension(tag, false)
                , rank(0)
                , layout(LayoutType::None)
            {
            }

            /**
             * Construct WaveTile dimension with deferred size and layout
             */
            WaveTile(int tag, int rank, bool output = false)
                : BaseDimension(tag, output)
                , rank(rank)
                , layout(LayoutType::None)
            {
            }

            /**
             * Construct WaveTile dimension with fully specified sizes.
             */
            WaveTile(int tag, std::vector<int> sizes, LayoutType layout, bool output = false)
                : BaseDimension(
                    tag, Expression::literal(product(sizes)), Expression::literal(1u), output)
                , rank(sizes.size())
                , sizes(sizes)
                , layout(layout)
            {
            }

            virtual std::string name() const
            {
                return "WaveTile";
            }

            TagType getTag() const
            {
                return {tag, 0, output, 0};
            }

            /**
             * Return WaveTileNumber cooresponding to sub-dimension `sdim` of this tile.
             */
            WaveTileNumber tileNumber(int sdim, bool output = false) const;

            /**
             * Return WaveTileIndex cooresponding to sub-dimension `sdim` of this tile.
             */
            WaveTileIndex tileIndex(int sdim, bool output = false) const;

            /**
             * Return total number of elements.
             */
            int elements() const;
        };

        /*
         * Helpers
         */

        inline TagType getTag(const Dimension& x)
        {
            auto ptag  = std::visit([](const auto a) { return a.getTag(); }, x);
            ptag.index = x.index();
            return ptag;
        }

        inline std::string toString(const Dimension& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        template <typename T>
        inline Expression::ExpressionPtr getSize(const T& x)
        {
            auto rv = std::visit([](const auto a) { return a.size; }, x);
            AssertFatal(rv, "Unable to get valid size for dimension: ", toString(x));
            return rv;
        }

        template <typename T>
        inline void setSize(T& x, Expression::ExpressionPtr size)
        {
            std::visit([size](auto& a) { a.size = size; }, x);
        }

        template <typename T>
        inline Expression::ExpressionPtr getStride(const T& x)
        {
            auto rv = std::visit([](const auto a) { return a.stride; }, x);
            AssertFatal(rv, "Unable to get valid stride for dimension: ", toString(x));
            return rv;
        }

        template <typename T>
        inline void setStride(T& x, Expression::ExpressionPtr stride)
        {
            std::visit([stride](auto& a) { a.stride = stride; }, x);
        }
    }
}
