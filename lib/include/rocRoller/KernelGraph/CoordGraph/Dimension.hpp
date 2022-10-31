#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>

#include "Dimension_fwd.hpp"

namespace rocRoller
{
    namespace KernelGraph::CoordGraph
    {
        /*
         * Nodes (Dimensions)
         */

        struct BaseDimension
        {
            Expression::ExpressionPtr size, stride;

            BaseDimension()
                : size(nullptr)
                , stride(nullptr)
            {
            }

            BaseDimension(Expression::ExpressionPtr size, Expression::ExpressionPtr stride)
                : size(size)
                , stride(stride)
            {
            }

            virtual std::string toString() const;

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
             * Create an Adhoc dimension with a specific name,
             * size and stride.
             */
            Adhoc(std::string const&        name,
                  Expression::ExpressionPtr size,
                  Expression::ExpressionPtr stride)
                : BaseDimension(size, stride)
                , m_name(name)
            {
                m_hash = std::hash<std::string>()(m_name);
            }

            /**
             * Create an Adhoc dimension with a specific name and
             * command tag.
             */
            Adhoc(std::string const& name)
                : Adhoc(name, nullptr, nullptr)
            {
            }

            virtual std::string name() const override
            {
                return m_name;
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

            SubDimension(int const                 dim,
                         Expression::ExpressionPtr size,
                         Expression::ExpressionPtr stride)
                : BaseDimension(size, stride)
                , dim(dim)
            {
            }

            SubDimension(int const dim = 0)
                : BaseDimension()
                , dim(dim)
            {
            }

            virtual std::string toString() const;

            virtual std::string name() const
            {
                return "SubDimension";
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

            User(std::string const& name)
                : BaseDimension()
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

        private:
            std::string m_argument_name;
        };

        /**
         * Linear dimension.  Usually flattened subdimenions.
         */
        struct Linear : public BaseDimension
        {
            using BaseDimension::BaseDimension;

            virtual std::string name() const override
            {
                return "Linear";
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
        };

        /**
         * Workgroup - typically represents workgroups on a GPU.
         *
         * Sub-dimensions 0, 1, and 2 coorespond to the x, y and z
         * kernel launch dimensions.
         */
        struct Workgroup : public SubDimension
        {
            Workgroup(int const dim = 0)
                : SubDimension(dim)
            {
            }

            virtual std::string name() const override
            {
                return "Workgroup";
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
            Workitem(int const dim = 0, Expression::ExpressionPtr size = nullptr)
                : SubDimension(dim, size, Expression::literal(1u))
            {
            }

            virtual std::string name() const override
            {
                return "Workitem";
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
        };

        struct Unroll : public BaseDimension
        {
            Unroll()
                : BaseDimension()
            {
            }

            Unroll(uint const usize)
                : BaseDimension(nullptr, nullptr)
            {
                size   = rocRoller::Expression::literal(usize);
                stride = rocRoller::Expression::literal(1);
            }

            Unroll(Expression::ExpressionPtr usize)
                : BaseDimension(nullptr, nullptr)
            {
                size   = usize;
                stride = rocRoller::Expression::literal(1);
            }

            virtual std::string name() const override
            {
                return "Unroll";
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

            /**
             * Construct MacroTile dimension with deferred rank etc.
             */
            MacroTile()
                : BaseDimension()
                , rank(0)
                , memoryType(MemoryType::None)
                , layoutType(LayoutType::None)
            {
            }

            /**
             * Construct MacroTile dimension with deferred sizes and
             * memory type.
             */
            MacroTile(int const rank)
                : BaseDimension()
                , rank(rank)
                , memoryType(MemoryType::None)
                , layoutType(LayoutType::None)
            {
            }

            /**
             * Construct MacroTile dimension with fully specified sizes
             * and memory type (ie, LDS vs VGPR).
             */
            MacroTile(std::vector<int> const& sizes,
                      MemoryType const        memoryType,
                      std::vector<int> const& subTileSizes = {})
                : BaseDimension()
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
            MacroTile(std::vector<int> const& sizes,
                      LayoutType const        layoutType,
                      std::vector<int> const& subTileSizes = {})
                : BaseDimension()
                , rank(sizes.size())
                , sizes(sizes)
                , memoryType(MemoryType::WAVE)
                , layoutType(layoutType)
                , subTileSizes(subTileSizes)
            {
                AssertFatal(layoutType != LayoutType::None, "Invalid layout type.");
            }

            virtual std::string toString() const override
            {
                if(!sizes.empty())
                {
                    std::string _sizes = "{";
                    for(int s : sizes)
                        _sizes += std::to_string(s) + ",";
                    _sizes[_sizes.length() - 1] = '}';
                    return name() + _sizes;
                }

                return BaseDimension::toString();
            }

            virtual std::string name() const override
            {
                return "MacroTile";
            }

            /**
             * Return MacroTileNumber cooresponding to sub-dimension `sdim` of this tile.
             */
            MacroTileNumber tileNumber(int sdim) const;

            /**
             * Return MacroTileIndex cooresponding to sub-dimension `sdim` of this tile.
             */
            MacroTileIndex tileIndex(int sdim) const;

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
            ThreadTile(std::vector<int> const& sizes)
                : BaseDimension()
                , rank(sizes.size())
                , sizes(sizes)
            {
            }

            virtual std::string name() const
            {
                return "ThreadTile";
            }

            /**
             * Return ThreadTileNumber cooresponding to sub-dimension `sdim` of this tile.
             */
            ThreadTileNumber tileNumber(int const sdim) const;

            /**
             * Return ThreadTileIndex cooresponding to sub-dimension `sdim` of this tile.
             */
            ThreadTileIndex tileIndex(int const sdim) const;
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

            /**
             * Construct WaveTile dimension with deferred rank and size.
             */
            WaveTile()
                : BaseDimension()
                , rank(0)
                , layout(LayoutType::None)
            {
            }

            /**
             * Construct WaveTile dimension with deferred size and layout
             */
            WaveTile(int const rank)
                : BaseDimension()
                , rank(rank)
                , layout(LayoutType::None)
            {
            }

            /**
             * Construct WaveTile dimension with fully specified sizes.
             */
            WaveTile(std::vector<int> const& sizes, LayoutType const layout)
                : BaseDimension(Expression::literal(product(sizes)), Expression::literal(1u))
                , rank(sizes.size())
                , sizes(sizes)
                , layout(layout)
            {
            }

            virtual std::string name() const
            {
                return "WaveTile";
            }

            /**
             * Return WaveTileNumber cooresponding to sub-dimension `sdim` of this tile.
             */
            WaveTileNumber tileNumber(int const sdim) const;

            /**
             * Return WaveTileIndex cooresponding to sub-dimension `sdim` of this tile.
             */
            WaveTileIndex tileIndex(int const sdim) const;

            /**
             * Return total number of elements.
             */
            int elements() const;
        };

        /**
         * JammedWaveTileNumber - Number of wave tiles to execute per wavefront
         */
        struct JammedWaveTileNumber : public SubDimension
        {
            using SubDimension::SubDimension;

            virtual std::string name() const
            {
                return "WaveTilePerWorkGroup";
            }
        };

        /*
         * Helpers
         */

        inline std::string toString(const Dimension& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        template <typename T>
        inline Expression::ExpressionPtr getSize(const T& x)
        {
            Expression::ExpressionPtr rv = std::visit([](auto const& a) { return a.size; }, x);
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
