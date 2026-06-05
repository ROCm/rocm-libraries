// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile_details.hpp>

TEST_CASE("LDS swizzle round-trip: forward then inverse recovers original column", "[lds-swizzle]")
{
    using namespace rocRoller;
    using namespace rocRoller::Expression;
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::CoordinateGraph;
    using namespace rocRoller::KernelGraph::LDSSwizzleDetail;

    struct Config
    {
        unsigned int numCols;
        unsigned int rowsPerBankRow;
        unsigned int elementsPerChunk;
        unsigned int columnsPerBankRow;
    };

    // FP4 macK=256 (8 cols, 2 rows/bank), FP4 macK=128 (4 cols, 4 rows/bank)
    Config configs[] = {
        {8, 2, 32, 16},
        {4, 4, 32, 16},
    };

    for(auto cfg : configs)
    {
        DYNAMIC_SECTION("numCols=" << cfg.numCols << " rowsPerBankRow=" << cfg.rowsPerBankRow)
        {
            auto params = LDSSwizzleParams{
                cfg.numCols, cfg.rowsPerBankRow, cfg.elementsPerChunk, cfg.columnsPerBankRow};

            unsigned int totalElems = cfg.numCols * cfg.elementsPerChunk;
            unsigned int totalRows  = cfg.numCols * cfg.rowsPerBankRow;

            // Build a coordinate graph with LoadTiled swizzle edges
            rocRoller::KernelGraph::KernelGraph grGraph;
            auto                                grCol = grGraph.coordinates.addElement(
                MacroTileIndex(0, literal(cfg.numCols), literal(1u)));
            auto grRow = grGraph.coordinates.addElement(
                MacroTileIndex(1, literal(totalRows), literal(1u)));
            auto grSwizzled = addLoadTiledSwizzleEdges(grGraph, grCol, grRow, 0, params);

            // Build a coordinate graph with LoadLDSTile unswizzle edges
            rocRoller::KernelGraph::KernelGraph lrGraph;
            auto                                lrCol = lrGraph.coordinates.addElement(
                MacroTileIndex(0, literal(totalElems), literal(1u)));
            auto lrRow = lrGraph.coordinates.addElement(
                MacroTileIndex(1, literal(totalRows), literal(1u)));
            auto lrUnswizzled = addLoadLDSTileSwizzleEdges(lrGraph, lrCol, lrRow, 0, params);

            for(unsigned int row = 0; row < totalRows; ++row)
            {
                for(unsigned int col = 0; col < cfg.numCols; ++col)
                {
                    // LoadTiled forward: (col, row) -> swizzledCol
                    Transformer grXform(&grGraph.coordinates);
                    grXform.setCoordinate(grCol, literal(col));
                    grXform.setCoordinate(grRow, literal(row));
                    auto grResult = grXform.reverse({grSwizzled});
                    auto swizzledCol
                        = static_cast<unsigned int>(getUnsignedInt(evaluate(grResult[0])));

                    // LoadLDSTile inverse: (swizzledCol * ePC, row) -> rawColElem
                    // The LoadLDSTile side works on element granularity, so multiply by ePC
                    unsigned int swizzledElem = swizzledCol * cfg.elementsPerChunk;
                    Transformer  lrXform(&lrGraph.coordinates);
                    lrXform.setCoordinate(lrCol, literal(swizzledElem));
                    lrXform.setCoordinate(lrRow, literal(row));
                    auto lrResult = lrXform.reverse({lrUnswizzled});
                    auto rawElem = static_cast<unsigned int>(getUnsignedInt(evaluate(lrResult[0])));

                    // Convert back to chunk index
                    auto rawCol = rawElem / cfg.elementsPerChunk;
                    CHECK(rawCol == col);
                }
            }
        }
    }
}
