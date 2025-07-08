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

#include <string>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {

        BaseDimension::BaseDimension() noexcept = default;

        BaseDimension::BaseDimension(Operations::OperationTag commandTag)
            : commandTag(commandTag)
        {
        }

        BaseDimension::BaseDimension(Operations::OperationTag  commandTag,
                                     Expression::ExpressionPtr size,
                                     Expression::ExpressionPtr stride)
            : size(size)
            , stride(stride)
            , commandTag(commandTag)
        {
        }

        BaseDimension::BaseDimension(Expression::ExpressionPtr size,
                                     Expression::ExpressionPtr stride)
            : size(size)
            , stride(stride)
        {
        }

        BaseDimension::BaseDimension(Expression::ExpressionPtr size,
                                     Expression::ExpressionPtr stride,
                                     Expression::ExpressionPtr offset)
            : size(size)
            , stride(stride)
            , offset(offset)
        {
        }

        std::string BaseDimension::toString() const
        {
            auto _size = size ? rocRoller::Expression::toString(size) : "NA";
            auto stag  = "{" + _size + "}";
            return name() + stag;
        }

        Adhoc::Adhoc() = default;

        Adhoc::Adhoc(std::string const&        name,
                     Expression::ExpressionPtr size,
                     Expression::ExpressionPtr stride)
            : BaseDimension(size, stride)
            , m_name(name)
        {
        }

        Adhoc::Adhoc(std::string const& name)
            : Adhoc(name, nullptr, nullptr)
        {
        }

        SubDimension::SubDimension(int const                 dim,
                                   Expression::ExpressionPtr size,
                                   Expression::ExpressionPtr stride)
            : BaseDimension(size, stride)
            , dim(dim)
        {
        }

        SubDimension::SubDimension(int const dim)
            : BaseDimension()
            , dim(dim)
        {
        }

        std::string SubDimension::name() const
        {
            return "SubDimension";
        }

        std::string SubDimension::toString() const
        {
            auto _size = size ? rocRoller::Expression::toString(size) : "NA";
            auto _sdim = std::to_string(dim);
            auto stag  = "{" + _sdim + ", " + _size + "}";
            return name() + stag;
        }

        User::User(Expression::ExpressionPtr size, Expression::ExpressionPtr offset)
            : BaseDimension(size, Expression::literal(1u), offset)
            , argumentName(rocRoller::SCRATCH)
        {
        }

        User::User(std::string const& name)
            : BaseDimension()
            , argumentName(name)
        {
        }

        User::User(Operations::OperationTag commandTag, std::string const& name)
            : BaseDimension(commandTag)
            , argumentName(name)
        {
        }

        User::User(Operations::OperationTag  commandTag,
                   std::string const&        name,
                   Expression::ExpressionPtr size)
            : BaseDimension(commandTag, size, Expression::literal(1u))
            , argumentName(name)
        {
        }

        User::User(std::string const& name, Expression::ExpressionPtr size)
            : BaseDimension(size, Expression::literal(1u))
            , argumentName(name)
        {
        }

        Workgroup::Workgroup(int const dim)
            : SubDimension(dim, nullptr, Expression::literal(1u))
        {
        }

        Workgroup::Workgroup(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        Workitem::Workitem() = default;

        Workitem::Workitem(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        LDS::LDS() = default;

        LDS::LDS(bool const isDirect2LDS, bool const holdsTransposedTile)
            : BaseDimension()
            , isDirect2LDS(isDirect2LDS)
            , holdsTransposedTile(holdsTransposedTile)
        {
        }

        Unroll::Unroll() = default;

        Unroll::Unroll(uint const usize)
            : BaseDimension(Expression::literal(usize), Expression::literal(1))
        {
        }

        Unroll::Unroll(Expression::ExpressionPtr usize)
            : BaseDimension(usize, Expression::literal(1))
        {
        }

        MacroTile::MacroTile() = default;

        MacroTile::MacroTile(Operations::OperationTag commandTag)
            : BaseDimension(commandTag)
        {
        }

        MacroTile::MacroTile(Operations::OperationTag commandTag, int rank)
            : BaseDimension(commandTag)
            , rank(rank)
        {
        }

        MacroTile::MacroTile(std::vector<int> const& sizes,
                             MemoryType              memoryType,
                             std::vector<int> const& subTileSizes)
            : BaseDimension()
            , rank(sizes.size())
            , sizes(sizes)
            , memoryType(memoryType)
            , layoutType(LayoutType::None)
            , subTileSizes(subTileSizes)
            , miTileSizes(subTileSizes)
        {
        }

        MacroTile::MacroTile(std::vector<int> const& sizes,
                             LayoutType              layoutType,
                             std::vector<int> const& subTileSizes,
                             MemoryType              memoryType,
                             std::vector<int> const& miTileSizes)
            : BaseDimension()
            , rank(sizes.size())
            , sizes(sizes)
            , memoryType(memoryType)
            , layoutType(layoutType)
            , subTileSizes(subTileSizes)
            , miTileSizes{miTileSizes.empty() ? subTileSizes : miTileSizes}
        {
            if(this->memoryType == MemoryType::LDS)
                this->memoryType = MemoryType::WAVE_LDS;
            AssertFatal(layoutType != LayoutType::None, "Invalid layout type.");
        }

        MacroTile::MacroTile(MacroTile& macTile, std::vector<uint> const& padBytesOfDim)
            : MacroTile(macTile)
        {
            AssertFatal(this->layoutType == LayoutType::MATRIX_A
                            || this->layoutType == LayoutType::MATRIX_B,
                        "Only MacroTiles for A or B can be padded.");
            this->padBytesOfDim = padBytesOfDim;
        }

        std::string MacroTile::toString() const
        {
            std::ostringstream msg;
            msg << BaseDimension::toString() << "(" << rank << "/" << memoryType << "/"
                << layoutType << ")"
                << "{";

            streamJoin(msg, sizes, ",");

            msg << "}-(";
            streamJoin(msg, subTileSizes, ",");
            msg << ")";

            return msg.str();
        }

        MacroTileNumber MacroTile::tileNumber(int sdim, Expression::ExpressionPtr size) const
        {
            return MacroTileNumber(sdim, size, Expression::literal(1u));
        }

        MacroTileIndex MacroTile::tileIndex(int sdim, uint jamming) const
        {
            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return MacroTileIndex(sdim,
                                  Expression::literal(static_cast<uint>(sizes.at(sdim)) * jamming),
                                  Expression::literal(stride));
        }

        int MacroTile::elements() const
        {
            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            return product(sizes);
        }

        uint MacroTile::paddingBytes() const
        {
            if(padBytesOfDim.empty())
                return 0;

            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            AssertFatal(sizes.size() == padBytesOfDim.size(),
                        "MacroTile sizes and padBytesOfDim must have the same rank.");
            return std::inner_product(sizes.rbegin(), sizes.rend(), padBytesOfDim.begin(), 0);
        }

        ThreadTileIndex::ThreadTileIndex() = default;
        ThreadTileIndex::ThreadTileIndex(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        ThreadTileNumber::ThreadTileNumber() = default;
        ThreadTileNumber::ThreadTileNumber(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        ThreadTile::ThreadTile() = default;

        ThreadTile::ThreadTile(MacroTile const& mac_tile)
            : BaseDimension()
            , rank(mac_tile.rank)
            , sizes(mac_tile.subTileSizes)
        {
            wsizes.resize(rank);
            for(int i = 0; i < rank; ++i)
            {
                wsizes[i] = mac_tile.sizes[i] / sizes[i];
            }
        }

        /**
         * Construct WaveTile dimension with fully specified sizes.
         */
        WaveTile::WaveTile(MacroTile const& macTile)
        {
            auto waveM = macTile.subTileSizes[0];
            auto waveN = macTile.subTileSizes[1];
            auto waveK = macTile.subTileSizes[2];

            padBytesOfDim = {};

            if(macTile.layoutType == LayoutType::MATRIX_A)
            {
                auto macM = macTile.sizes[0];
                auto macK = macTile.sizes[1];

                sizes  = {waveM, waveK};
                wsizes = {macM / waveM, macK / waveK};
                if(!macTile.padBytesOfDim.empty())
                {
                    auto padBytes0 = macTile.padBytesOfDim[0];
                    auto padBytes1 = macTile.padBytesOfDim[1];
                    padBytesOfDim  = {(macM / waveM) * padBytes0, (macK / waveK) * padBytes1};
                }
            }
            if(macTile.layoutType == LayoutType::MATRIX_B)
            {
                auto macK = macTile.sizes[0];
                auto macN = macTile.sizes[1];

                sizes  = {waveK, waveN};
                wsizes = {macK / waveK, macN / waveN};
                if(!macTile.padBytesOfDim.empty())
                {
                    auto padBytes0 = macTile.padBytesOfDim[0];
                    auto padBytes1 = macTile.padBytesOfDim[1];
                    padBytesOfDim  = {(macK / waveK) * padBytes0, (macN / waveN) * padBytes1};
                }
            }
            if(macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
            {
                auto macM = macTile.sizes[0];
                auto macN = macTile.sizes[1];

                sizes  = {waveM, waveN};
                wsizes = {macM / waveM, macN / waveN};
            }

            rank   = 2;
            size   = Expression::literal(product(sizes));
            stride = Expression::literal(1u);
            layout = macTile.layoutType;
        }

        WaveTileNumber WaveTile::tileNumber(int sdim) const
        {
            return WaveTileNumber(sdim,
                                  Expression::literal(static_cast<uint>(wsizes.at(sdim))),
                                  Expression::literal(1u));
        }

        WaveTileIndex WaveTile::tileIndex(int sdim) const
        {
            AssertFatal(!sizes.empty(), "WaveTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return WaveTileIndex(sdim,
                                 Expression::literal(static_cast<uint>(sizes.at(sdim))),
                                 Expression::literal(stride));
        }

        int WaveTile::elements() const
        {
            return product(sizes);
        }

        uint WaveTile::paddingBytes() const
        {
            if(padBytesOfDim.empty())
                return 0;

            AssertFatal(!sizes.empty(), "WaveTile doesn't have sizes set.");
            AssertFatal(sizes.size() == padBytesOfDim.size(),
                        "WaveTile sizes and padBytesOfDim must have the same rank.");
            return std::inner_product(sizes.rbegin(), sizes.rend(), padBytesOfDim.begin(), 0);
        }

        ElementNumber::ElementNumber() = default;
        ElementNumber::ElementNumber(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        std::string Adhoc::name() const
        {
            if(m_name.empty())
                return "Adhoc";
            else
                return "Adhoc." + m_name;
        }

#define DEFAULT_DIM_NAME(cls)     \
    std::string cls::name() const \
    {                             \
        return #cls;              \
    }

        DEFAULT_DIM_NAME(User);
        DEFAULT_DIM_NAME(Linear);
        DEFAULT_DIM_NAME(Wavefront);
        DEFAULT_DIM_NAME(Lane);
        DEFAULT_DIM_NAME(Workgroup);
        DEFAULT_DIM_NAME(Workitem);
        DEFAULT_DIM_NAME(VGPR);
        DEFAULT_DIM_NAME(VGPRBlockNumber);
        DEFAULT_DIM_NAME(VGPRBlockIndex);
        DEFAULT_DIM_NAME(LDS);
        DEFAULT_DIM_NAME(ForLoop);
        DEFAULT_DIM_NAME(Unroll);
        DEFAULT_DIM_NAME(MacroTileIndex);
        DEFAULT_DIM_NAME(MacroTileNumber);
        DEFAULT_DIM_NAME(MacroTile);
        DEFAULT_DIM_NAME(ThreadTileIndex);
        DEFAULT_DIM_NAME(ThreadTileNumber);
        DEFAULT_DIM_NAME(ThreadTile);
        DEFAULT_DIM_NAME(WaveTileIndex);
        DEFAULT_DIM_NAME(WaveTileNumber);
        DEFAULT_DIM_NAME(WaveTile);
        DEFAULT_DIM_NAME(JammedWaveTileNumber);
        DEFAULT_DIM_NAME(ElementNumber);
    }
}
