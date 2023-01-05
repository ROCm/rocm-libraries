#include <string>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        /*
         * Dimension methods
         */
        std::string BaseDimension::toString() const
        {
            auto _size = size ? rocRoller::Expression::toString(size) : "NA";
            auto stag  = "{" + _size + "}";
            return name() + stag;
        }

        std::string SubDimension::toString() const
        {
            auto _size = size ? rocRoller::Expression::toString(size) : "NA";
            auto _sdim = std::to_string(dim);
            auto stag  = "{" + _sdim + ", " + _size + "}";
            return name() + stag;
        }

        MacroTileNumber MacroTile::tileNumber(int sdim) const
        {
            return MacroTileNumber(sdim, Expression::literal(1u), Expression::literal(1u));
        }

        MacroTileIndex MacroTile::tileIndex(int sdim) const
        {
            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return MacroTileIndex(sdim,
                                  Expression::literal(static_cast<uint>(sizes.at(sdim))),
                                  Expression::literal(stride));
        }

        ThreadTileNumber ThreadTile::tileNumber(int sdim) const
        {
            return ThreadTileNumber(sdim, Expression::literal(1u), Expression::literal(1u));
        }

        ThreadTileIndex ThreadTile::tileIndex(int sdim) const
        {
            AssertFatal(!sizes.empty(), "ThreadTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return ThreadTileIndex(sdim,
                                   Expression::literal(static_cast<uint>(sizes.at(sdim))),
                                   Expression::literal(stride));
        }

        WaveTileNumber WaveTile::tileNumber(int sdim) const
        {
            return WaveTileNumber(sdim, Expression::literal(1u), Expression::literal(1u));
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

        int MacroTile::elements() const
        {
            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            return product(sizes);
        }
    }
}
