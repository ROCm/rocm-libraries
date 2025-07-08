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

#include <memory>
#include <string>
#include <vector>

#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        /*
         * Nodes (Dimensions)
         *
         * Used for two different purposes:
         * - Coordinates (integer indices)
         * - Storage (Registers, e.g. MacroTile)
         */

        struct BaseDimension
        {
            Expression::ExpressionPtr size, stride, offset;

            Operations::OperationTag commandTag;

            BaseDimension() noexcept;
            BaseDimension(Operations::OperationTag commandTag);
            BaseDimension(Operations::OperationTag  commandTag,
                          Expression::ExpressionPtr size,
                          Expression::ExpressionPtr stride);
            BaseDimension(Expression::ExpressionPtr size, Expression::ExpressionPtr stride);
            BaseDimension(Expression::ExpressionPtr size,
                          Expression::ExpressionPtr stride,
                          Expression::ExpressionPtr offset);

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
         *
         * Can exist in the final graph.
         */
        struct Adhoc : public BaseDimension
        {
            Adhoc();

            /**
             * Create an Adhoc dimension with a specific name,
             * size and stride.
             */
            Adhoc(std::string const&        name,
                  Expression::ExpressionPtr size,
                  Expression::ExpressionPtr stride);

            /**
             * Create an Adhoc dimension with a specific name.
             */
            explicit Adhoc(std::string const& name);

            std::string name() const override;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

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
                         Expression::ExpressionPtr stride);

            explicit SubDimension(int const dim = 0);

            virtual std::string toString() const override;
            virtual std::string name() const override;
        };

        /**
         * User - represents tensor from the user.
         *
         * Usually split into SubDimensions.  The subdimensions carry
         * sizes and strides.
         */
        struct User : public BaseDimension
        {
            std::string argumentName;

            bool needsPadding = false;

            using BaseDimension::BaseDimension;

            explicit User(std::string const& name);
            User(std::string const& name, Expression::ExpressionPtr size);
            User(Operations::OperationTag commandTag, std::string const& name);
            User(Operations::OperationTag  commandTag,
                 std::string const&        name,
                 Expression::ExpressionPtr size);

            /**
             * @brief Constructor for a User dimension that is part of the scratch space.
             *
             * @param size How many elements make up the User dimension.
             * @param offset Location of data within the scratch space
             */
            User(Expression::ExpressionPtr size, Expression::ExpressionPtr offset);

            std::string name() const override;
        };

        /**
         * Linear dimension.  Usually flattened subdimenions.
         */
        struct Linear : public BaseDimension
        {
            static constexpr bool HasValue = false;
            using BaseDimension::BaseDimension;

            std::string name() const override;
        };

        /**
         * Wavefront - represents wavefronts within a workgroup.
         */
        struct Wavefront : public SubDimension
        {
            static constexpr bool HasValue = false;
            using SubDimension::SubDimension;

            std::string name() const override;
        };

        /**
         * Lane - represents a lane within a wavefront.
         */
        struct Lane : public BaseDimension
        {
            static constexpr bool HasValue = false;
            using BaseDimension::BaseDimension;

            std::string name() const override;
        };

        /**
         * Workgroup - typically represents workgroups on a GPU.
         *
         * Sub-dimensions 0, 1, and 2 correspond to the x, y and z
         * kernel launch dimensions.
         */
        struct Workgroup : public SubDimension
        {
            static constexpr bool HasValue = false;

            explicit Workgroup(int const dim = 0);
            Workgroup(int const dim, Expression::ExpressionPtr size);
            std::string name() const override;
        };

        /**
         * Workitem - typically represents threads within a workgroup.
         *
         * Sub-dimensions 0, 1, and 2 correspond to the x, y and z
         * kernel launch dimensions.
         */
        struct Workitem : public SubDimension
        {
            static constexpr bool HasValue = false;

            Workitem();
            explicit Workitem(int const dim, Expression::ExpressionPtr size = nullptr);

            std::string name() const override;
        };

        /**
         * VGPR - represents (small) thread local scalar/array.
         */
        struct VGPR : public BaseDimension
        {
            static constexpr bool HasValue = false;

            using BaseDimension::BaseDimension;

            std::string name() const override;
        };

        struct VGPRBlockNumber : public BaseDimension
        {
            static constexpr bool HasValue = false;

            using BaseDimension::BaseDimension;

            std::string name() const override;
        };

        struct VGPRBlockIndex : public BaseDimension
        {
            static constexpr bool HasValue = false;

            using BaseDimension::BaseDimension;

            std::string name() const override;
        };

        /**
         * LDS - represents local memory.
         *
         * Multipurpose:
         * - Represents storage
         * - Represents address coordinate information
         */
        struct LDS : public BaseDimension
        {
            static constexpr bool HasValue = false;
            using BaseDimension::BaseDimension;

            bool isDirect2LDS        = false;
            bool holdsTransposedTile = false;

            LDS();
            explicit LDS(bool const isDirect2LDS, bool const holdsTransposedTile = false);

            std::string name() const override;
        };

        /**
         * ForLoop - represents the coordinate value associated with
         * the iterations of a for-loop.
         *
         * ForLoop dimensions elucidate how indexes depend on which
         * for-loop iteration is being executed.
         */
        struct ForLoop : public BaseDimension
        {
            static constexpr bool HasValue = false;

            using BaseDimension::BaseDimension;

            std::string name() const override;
        };

        /**
         * Unroll - represents the coordinate value associated with
         * the unrolled iterations of a for-loop.
         *
         * Unroll dimensions elucidate how indexes depend on which
         * inner-iteration of an unrolled for-loop is being executed.
         */
        struct Unroll : public BaseDimension
        {
            static constexpr bool HasValue = false;

            Unroll();
            explicit Unroll(uint const usize);
            explicit Unroll(Expression::ExpressionPtr usize);

            std::string name() const override;
        };

        /**
         * MacroTileIndex - sub-dimension of a tile.  See MacroTile.
         */
        struct MacroTileIndex : public SubDimension
        {
            static constexpr bool HasValue = false;

            using SubDimension::SubDimension;

            std::string name() const override;
        };

        /**
         * MacroTileNumber.  See MacroTile.
         */
        struct MacroTileNumber : public SubDimension
        {
            static constexpr bool HasValue = false;

            using SubDimension::SubDimension;

            std::string name() const override;
        };

        /**
         * MacroTile - a tensor tile owned by a workgroup.
         *
         * The storage location (eg, VGPRs vs LDS) is specified by
         * `MemoryType`.
         */
        struct MacroTile : public BaseDimension
        {
            int        rank       = 0;
            MemoryType memoryType = MemoryType::None;
            LayoutType layoutType = LayoutType::None;

            std::vector<int> sizes;

            /**
             * Size of thread tiles.
             *
             * Sizes of -1 represent a "to be determined size".
             */
            std::vector<int> subTileSizes;

            /**
             * Size of matrix instruction tile.
             */
            std::vector<int> miTileSizes;

            /**
             * Number of bytes padding each dimension.
             *
             * For example, a MxN Macrotile padded with [[x y]] requires
             * M * N * elementBytes + N * x + M * y bytes of storage.
             */
            std::vector<uint> padBytesOfDim;

            /**
             * Construct MacroTile dimension with deferred rank etc.
             */
            MacroTile();

            /**
             * Construct MacroTile dimension with deferred sizes and
             * memory type.
             */
            explicit MacroTile(Operations::OperationTag commandTag);
            MacroTile(Operations::OperationTag commandTag, int rank);

            /**
             * Construct MacroTile dimension with fully specified sizes
             * and memory type (ie, LDS vs VGPR).
             */
            MacroTile(std::vector<int> const& sizes,
                      MemoryType              memoryType,
                      std::vector<int> const& subTileSizes = {});

            /**
             * Construct MacroTile dimension with fully specified sizes,
             * layout type (i.e. MATRIX_A, MATRIX_B or MATRIX_ACCUMULATOR) and
             * memory type (i.e. WAVE or LDS (internally represented as WAVE_LDS)).
             *
             * Memory type is WAVE (by default) for VGPRs or WAVE_LDS for LDS.
             */
            MacroTile(std::vector<int> const& sizes,
                      LayoutType const        layoutType,
                      std::vector<int> const& subTileSizes = {},
                      MemoryType const        memoryType   = MemoryType::WAVE,
                      std::vector<int> const& miTileSizes  = {});

            /**
             * Construct MacroTile dimension that is padded.
             *
             * Layout type must be either MATRIX_A or MATRIX_B.
             */
            MacroTile(MacroTile& macTile, std::vector<uint> const& padBytesOfDim);

            std::string toString() const override;

            std::string name() const override;

            /**
             * Return MacroTileNumber corresponding to sub-dimension `sdim` of this tile.
             */
            MacroTileNumber tileNumber(int sdim, Expression::ExpressionPtr size) const;

            /**
             * Return MacroTileIndex corresponding to sub-dimension `sdim` of this tile.
             */
            MacroTileIndex tileIndex(int sdim, uint jamming = 1) const;

            /**
             * Return total number of elements.
             */
            int elements() const;

            /**
             * Return total number of padding bytes.
             */
            uint paddingBytes() const;
        };

        /**
         * ThreadTileIndex - sub-dimension of a tile (fast-moving).
         */
        struct ThreadTileIndex : public SubDimension
        {
            static constexpr bool HasValue = false;

            ThreadTileIndex();
            explicit ThreadTileIndex(int const dim, Expression::ExpressionPtr size = nullptr);

            std::string name() const override;
        };

        /**
         * ThreadTileNumber - sub-dimension of a tile (slow-moving).
         */
        struct ThreadTileNumber : public SubDimension
        {
            static constexpr bool HasValue = false;

            ThreadTileNumber();
            explicit ThreadTileNumber(int const dim, Expression::ExpressionPtr size = nullptr);

            std::string name() const override;
        };

        /**
         * ThreadTile - a tensor tile owned by a thread.
         *
         * The storage location (eg, VGPRs vs LDS) is specified by
         * `MemoryType`.
         */
        struct ThreadTile : public BaseDimension
        {
            int rank = -1;

            // -1 is used to represent a "to be determined" size.
            std::vector<int> sizes;
            std::vector<int> wsizes;

            ThreadTile();
            /**
             * Construct ThreadTile dimension with fully specified sizes
             * and memory type (ie, LDS vs VGPR).
             */
            explicit ThreadTile(MacroTile const& mac_tile);

            std::string name() const override;
        };

        /**
         * WaveTileIndex - sub-dimension of a tile.  See WaveTile.
         */
        struct WaveTileIndex : public SubDimension
        {
            static constexpr bool HasValue = false;
            using SubDimension::SubDimension;

            std::string name() const override;
        };

        /**
         * WaveTileNumber.  See WaveTile.
         */
        struct WaveTileNumber : public SubDimension
        {
            static constexpr bool HasValue = false;
            using SubDimension::SubDimension;

            std::string name() const override;
        };

        /**
         * WaveTile - a tensor tile owned by a wave in GPRs.
         */
        struct WaveTile : public BaseDimension
        {
            int rank = 0;

            std::vector<int> sizes;
            std::vector<int> wsizes;

            LayoutType         layout = LayoutType::None;
            Register::ValuePtr vgpr; // TODO: Does this belong here?  Move to "getVGPR"?

            /**
             * Number of bytes padding each dimension.
             *
             * For example, a MxN Macrotile padded with [[x y]] requires
             * M * N * elementBytes + N * x + M * y bytes of storage.
             */
            std::vector<uint> padBytesOfDim;

            WaveTile() = default;

            /**
             * Construct WaveTile dimension with fully specified sizes.
             */
            explicit WaveTile(MacroTile const&);

            std::string name() const override;

            /**
             * Return WaveTileNumber corresponding to sub-dimension `sdim` of this tile.
             */
            WaveTileNumber tileNumber(int const sdim) const;

            /**
             * Return WaveTileIndex corresponding to sub-dimension `sdim` of this tile.
             */
            WaveTileIndex tileIndex(int const sdim) const;

            /**
             * Return total number of elements.
             */
            int elements() const;

            /**
             * Return total number of padding bytes.
             */
            uint paddingBytes() const;
        };

        /**
         * JammedWaveTileNumber - Number of wave tiles to execute per wavefront
         */
        struct JammedWaveTileNumber : public SubDimension
        {
            static constexpr bool HasValue = false;
            using SubDimension::SubDimension;

            std::string name() const override;
        };

        /**
         * ElementNumber - represents the value(s) from a ThreadTile to be stored in the VGPR(s).
         */
        struct ElementNumber : public SubDimension
        {
            static constexpr bool HasValue = false;

            ElementNumber();
            explicit ElementNumber(int const dim, Expression::ExpressionPtr size = nullptr);

            std::string name() const override;
        };

        /*
         * Helpers
         */

        inline std::string toString(const Dimension& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        template <CConcreteDimension Dim>
        inline std::string name(Dim const& d)
        {
            return d.name();
        }

        inline std::string name(const Dimension& x)
        {
            return std::visit([](const auto& a) { return a.name(); }, x);
        }

        template <typename T>
        inline Expression::ExpressionPtr getSize(const T& x)
        {
            Expression::ExpressionPtr rv = std::visit([](auto const& a) { return a.size; }, x);
            return rv;
        }

        template <typename T>
        inline void setSize(T& x, Expression::ExpressionPtr size)
        {
            std::visit([size](auto& a) { a.size = size; }, x);
        }

        template <typename T>
        inline Operations::OperationTag getCommandTag(const T& x)
        {
            return std::visit(
                [](auto const& a) -> Operations::OperationTag { return a.commandTag; }, x);
        }

        template <typename T>
        inline void setCommandTag(T& x, Operations::OperationTag commandTag)
        {
            std::visit([commandTag](auto& a) { a.commandTag = commandTag; }, x);
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

        template <CDimension D>
        inline bool isDimension(const Dimension& x)
        {
            if(std::holds_alternative<D>(x))
                return true;
            return false;
        }
    }
}
