// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// nanobind bindings for the TensileLite "tensile_writer" C++ migration.
//
// This is the minimal scaffold for the incremental KernelWriter port: it
// exposes the pure subtile *geometry* math (no writer state, no register
// allocation, no rocisa emission) as
//
//     _tensile_writer.subtile.geometry
//
// The Python package tensile_writer/ re-exports this. The geometry value/query
// layer, the read-only TileInfo query layer (tile_info), the MFMA F8F6F4
// instType selection (emit), the subtile instruction-scheduling slot-placement
// algorithm (instruction_scheduler), the LogicalScheduler value/config
// helpers (fmt_mt, SchedulerConfig.get_partition_candidates) and the GR/LR
// offset-assignment plans (on ABTileInfoQuery) are all C++-only — the Python
// side forwards to them unconditionally with no Python fallback. The
// writer-free LogicalScheduler pass pipeline (place_LRs
// through remove_unnecessary_wait_lr_sync, plus assign_vgpr_tiles) is now the
// live implementation: the Python LogicalScheduler delegates those passes here
// and rebuilds its dataclass partitions from value_partitions(). The rocisa
// writer integration (emit/build_*/populate_instructions/alloc) stays Python.

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "tensile_writer/emit_leaves.hpp"
#include "tensile_writer/instruction_scheduler.hpp"
#include "tensile_writer/logical_scheduler.hpp"
#include "tensile_writer/logical_scheduler_passes.hpp"
#include "tensile_writer/subtile_geometry.hpp"
#include "tensile_writer/tile_info.hpp"

namespace nb = nanobind;
using namespace tw::subtile;

namespace {

// Tag marker identities — empty types analogous to C++ tag-dispatch tags.
struct GRTag_1x1 {};
struct GRTag_1x2 {};
struct GRTag_2x2 {};
struct GRTag_TLU1 {};
struct LRTag_1x1 {};
struct LRTag_1x2 {};
struct LRTag_TLU1 {};

// Read kernel[key] as a long, accepting any Python mapping (dict, Solution,
// etc.). Mirrors Python's kernel["..."] item access.
long kernel_int(nb::handle kernel, const std::string& key) {
  return nb::cast<long>(kernel[nb::str(key.c_str())]);
}

void bind_geometry(nb::module_& g) {
  // -- LoadShape --------------------------------------------------------
  nb::class_<LoadShape>(g, "LoadShape")
      .def(nb::init<int, int>(), nb::arg("m"), nb::arg("k"))
      .def_ro("m", &LoadShape::m)
      .def_ro("k", &LoadShape::k)
      .def("__eq__",
           [](const LoadShape& a, const LoadShape& b) { return a == b; },
           nb::arg("other").none())
      .def("__repr__", [](const LoadShape& s) {
        return "LoadShape(m=" + std::to_string(s.m) +
               ", k=" + std::to_string(s.k) + ")";
      });

  // -- MMALayout --------------------------------------------------------
  nb::class_<MMALayout>(g, "MMALayout")
      .def(nb::init<int, int, int, int>(), nb::arg("instM"),
           nb::arg("blocks") = -1, nb::arg("vgprs") = -1,
           nb::arg("waveSize") = -1)
      .def_ro("instM", &MMALayout::instM)
      .def_ro("blocks", &MMALayout::blocks)
      .def_ro("vgprs", &MMALayout::vgprs)
      .def_ro("waveSize", &MMALayout::waveSize)
      .def_ro("contiguousLanes", &MMALayout::contiguousLanes)
      .def_ro("kGroups", &MMALayout::kGroups)
      .def_ro("elementsPerLaneNonK", &MMALayout::elementsPerLaneNonK)
      .def("inputBytesPerLane", &MMALayout::inputBytesPerLane)
      .def("tileSizeBytes", &MMALayout::tileSizeBytes, nb::arg("instK"),
           nb::arg("elementBytes"))
      .def("regsPerTile", &MMALayout::regsPerTile, nb::arg("instK"),
           nb::arg("elementBytes"));

  // -- MMAScaleLayout ---------------------------------------------------
  nb::class_<MMAScaleLayout>(g, "MMAScaleLayout")
      .def(nb::init<int, int, double, int, int>(), nb::arg("instM"),
           nb::arg("blocks") = -1, nb::arg("vgprs") = -1.0,
           nb::arg("mxBlock") = -1, nb::arg("waveSize") = -1)
      .def_ro("instM", &MMAScaleLayout::instM)
      .def_ro("blocks", &MMAScaleLayout::blocks)
      .def_ro("vgprs", &MMAScaleLayout::vgprs)
      .def_ro("mxBlock", &MMAScaleLayout::mxBlock)
      .def_ro("waveSize", &MMAScaleLayout::waveSize)
      .def_ro("contiguousLanes", &MMAScaleLayout::contiguousLanes);

  // -- ABGRGeometry -----------------------------------------------------
  nb::class_<ABGRGeometry>(g, "ABGRGeometry")
      .def(nb::init<const MMALayout&, int, double, const LoadShape&,
                    std::pair<int, int>, std::optional<int>, std::optional<int>,
                    bool, int>(),
           nb::arg("mmaLayout"), nb::arg("instK"), nb::arg("bpe"),
           nb::arg("loadShape"), nb::arg("subtileShape") = std::pair<int, int>{1, 1},
           nb::arg("subtileCount") = nb::none(),
           nb::arg("subtileStride") = nb::none(), nb::arg("tlu") = false,
           nb::arg("loadWidth") = 16)
      .def_ro("mmaLayout", &ABGRGeometry::mmaLayout)
      .def_ro("instK", &ABGRGeometry::instK)
      .def_ro("bpe", &ABGRGeometry::bpe)
      .def_ro("tlu", &ABGRGeometry::tlu)
      .def_ro("loadShape", &ABGRGeometry::loadShape)
      .def_ro("loadWidth", &ABGRGeometry::loadWidth)
      .def_ro("subtileShape", &ABGRGeometry::subtileShape)
      .def_prop_ro("subtileCount",
                   [](const ABGRGeometry& s) -> nb::object {
                     if (s.subtileCount.has_value())
                       return nb::cast(*s.subtileCount);
                     return nb::none();
                   })
      .def_prop_ro("subtileStride",
                   [](const ABGRGeometry& s) -> nb::object {
                     if (s.subtileStride.has_value())
                       return nb::cast(*s.subtileStride);
                     return nb::none();
                   })
      .def_ro("mmaTileShape", &ABGRGeometry::mmaTileShape)
      .def_ro("mmaTileSize", &ABGRGeometry::mmaTileSize)
      .def_ro("mmaTileRegCount", &ABGRGeometry::mmaTileRegCount)
      .def("globalMMATileGrid", &ABGRGeometry::globalMMATileGrid,
           nb::arg("macroTile"), nb::arg("depthU"))
      .def("localMMATileGrid", &ABGRGeometry::localMMATileGrid,
           nb::arg("macroTile"), nb::arg("depthU"), nb::arg("waveGroupSize"))
      .def("globalSubtileGrid", &ABGRGeometry::globalSubtileGrid,
           nb::arg("macroTile"), nb::arg("depthU"))
      .def("subtileSizeBytes", &ABGRGeometry::subtileSizeBytes)
      .def("bytesPerLoad", &ABGRGeometry::bytesPerLoad, nb::arg("numWaves"))
      .def("loadsPerStrip", &ABGRGeometry::loadsPerStrip, nb::arg("numWaves"))
      .def("localGRGranularity", &ABGRGeometry::localGRGranularity,
           nb::arg("numWaves"))
      .def(
          "for_kernel",
          [](const ABGRGeometry& self, nb::handle kernel, const std::string& tc) {
            int wg_idx = (tc == "A") ? 0 : 1;
            auto wg = nb::cast<std::vector<int>>(kernel[nb::str("MIWaveGroup")]);
            int wg_m = wg.at(wg_idx);
            long mt = kernel_int(kernel, "MacroTile" + tc);
            long mt_mma = floordiv(mt, self.mmaTileShape.first);
            return self.forKernel(wg_m, mt_mma);
          },
          nb::arg("kernel"), nb::arg("tc"))
      .def("forKernelParams", &ABGRGeometry::forKernelParams, nb::arg("wg_m"),
           nb::arg("mt_mma"))
      .def(
          "subtileForMmaTile",
          [](const ABGRGeometry& self, long r, long c) -> nb::object {
            if (!self.subtileCount.has_value() ||
                !self.subtileStride.has_value()) {
              throw std::runtime_error(
                  "subtileForMmaTile requires for_kernel() to be called first");
            }
            auto res = self.subtileForMmaTile(r, c);
            nb::list tiles;
            for (auto& t : res.mma_tiles)
              tiles.append(nb::make_tuple(t.first, t.second));
            return nb::make_tuple(
                nb::make_tuple(res.subtile_id.first, res.subtile_id.second),
                nb::make_tuple(res.block_shape.first, res.block_shape.second),
                tiles);
          },
          nb::arg("r"), nb::arg("c"));

  // -- ABLRGeometry -----------------------------------------------------
  nb::class_<ABLRGeometry>(g, "ABLRGeometry")
      .def(nb::init<const MMALayout&, int, double, const LoadShape&,
                    std::pair<int, int>, bool, int>(),
           nb::arg("mmaLayout"), nb::arg("instK"), nb::arg("bpe"),
           nb::arg("loadShape"),
           nb::arg("subtileShape") = std::pair<int, int>{1, 1},
           nb::arg("tlu") = false, nb::arg("loadWidth") = 16)
      .def_ro("mmaLayout", &ABLRGeometry::mmaLayout)
      .def_ro("instK", &ABLRGeometry::instK)
      .def_ro("bpe", &ABLRGeometry::bpe)
      .def_ro("tlu", &ABLRGeometry::tlu)
      .def_ro("loadShape", &ABLRGeometry::loadShape)
      .def_ro("loadWidth", &ABLRGeometry::loadWidth)
      .def_ro("subtileShape", &ABLRGeometry::subtileShape)
      .def_ro("mmaTileShape", &ABLRGeometry::mmaTileShape)
      .def_ro("mmaTileSize", &ABLRGeometry::mmaTileSize)
      .def_ro("mmaTileRegCount", &ABLRGeometry::mmaTileRegCount)
      .def("globalMMATileGrid", &ABLRGeometry::globalMMATileGrid,
           nb::arg("macroTile"), nb::arg("depthU"))
      .def("localMMATileGrid", &ABLRGeometry::localMMATileGrid,
           nb::arg("macroTile"), nb::arg("depthU"), nb::arg("waveGroupSize"))
      .def("globalSubtileGrid", &ABLRGeometry::globalSubtileGrid,
           nb::arg("macroTile"), nb::arg("depthU"))
      .def("subtileSizeBytes", &ABLRGeometry::subtileSizeBytes);

  // -- CDTileGeometry ---------------------------------------------------
  nb::class_<CDTileGeometry>(g, "CDTileGeometry")
      .def(nb::init<const MMALayout&, double, const LoadShape&>(),
           nb::arg("mmaLayout"), nb::arg("bpe"),
           nb::arg("storeShape") = LoadShape(1, 1))
      .def_ro("mmaLayout", &CDTileGeometry::mmaLayout)
      .def_ro("bpe", &CDTileGeometry::bpe)
      .def_ro("storeShape", &CDTileGeometry::storeShape)
      .def_ro("mmaTileShape", &CDTileGeometry::mmaTileShape)
      .def_ro("mmaTileSize", &CDTileGeometry::mmaTileSize)
      .def_ro("mmaTileRegCount", &CDTileGeometry::mmaTileRegCount)
      .def("globalMMATileGrid", &CDTileGeometry::globalMMATileGrid,
           nb::arg("macroTile0"), nb::arg("macroTile1"))
      .def("localMMATileGrid", &CDTileGeometry::localMMATileGrid,
           nb::arg("macroTile0"), nb::arg("macroTile1"), nb::arg("waveGroup"))
      .def("globalSubtileGrid", &CDTileGeometry::globalSubtileGrid,
           nb::arg("macroTile0"), nb::arg("macroTile1"), nb::arg("subtileShape"))
      .def("localSubtileGrid", &CDTileGeometry::localSubtileGrid,
           nb::arg("macroTile0"), nb::arg("macroTile1"), nb::arg("waveGroup"),
           nb::arg("subtileShape"));

  // -- MXScaleGRGeometry ------------------------------------------------
  nb::class_<MXScaleGRGeometry>(g, "MXScaleGRGeometry")
      .def(nb::init<const MMAScaleLayout&, int, double, int,
                    std::optional<std::pair<int, int>>>(),
           nb::arg("scaleLayout"), nb::arg("instK"), nb::arg("bpe"),
           nb::arg("loadWidth") = 16, nb::arg("subtileShape") = nb::none())
      .def_ro("scaleLayout", &MXScaleGRGeometry::scaleLayout)
      .def_ro("instK", &MXScaleGRGeometry::instK)
      .def_ro("bpe", &MXScaleGRGeometry::bpe)
      .def_ro("loadWidth", &MXScaleGRGeometry::loadWidth)
      .def_prop_ro("subtileShape",
                   [](const MXScaleGRGeometry& s) -> nb::object {
                     if (s.subtileShape.has_value())
                       return nb::cast(*s.subtileShape);
                     return nb::none();
                   })
      .def_ro("mmaTileShape", &MXScaleGRGeometry::mmaTileShape)
      .def_ro("mmaTileSize", &MXScaleGRGeometry::mmaTileSize)
      .def_ro("mmaTileRegCount", &MXScaleGRGeometry::mmaTileRegCount)
      .def("globalMMATileGrid", &MXScaleGRGeometry::globalMMATileGrid,
           nb::arg("macroTile"), nb::arg("depthU"))
      .def(
          "for_kernel",
          [](const MXScaleGRGeometry& self, nb::handle kernel,
             const std::string& tc) {
            if (self.subtileShape.has_value()) return self;
            long mt = kernel_int(kernel, "MacroTile" + tc);
            long du = kernel_int(kernel, "_DepthU" + tc);
            return self.forKernel(mt, du);
          },
          nb::arg("kernel"), nb::arg("tc"));

  // -- MXScaleLRGeometry ------------------------------------------------
  nb::class_<MXScaleLRGeometry>(g, "MXScaleLRGeometry")
      .def(nb::init<const MMAScaleLayout&, int, double, int,
                    std::pair<int, int>>(),
           nb::arg("scaleLayout"), nb::arg("instK"), nb::arg("bpe"),
           nb::arg("loadWidth") = 16,
           nb::arg("subtileShape") = std::pair<int, int>{2, 2})
      .def_ro("scaleLayout", &MXScaleLRGeometry::scaleLayout)
      .def_ro("instK", &MXScaleLRGeometry::instK)
      .def_ro("bpe", &MXScaleLRGeometry::bpe)
      .def_ro("loadWidth", &MXScaleLRGeometry::loadWidth)
      .def_ro("subtileShape", &MXScaleLRGeometry::subtileShape)
      .def_ro("mmaTileShape", &MXScaleLRGeometry::mmaTileShape)
      .def_ro("mmaTileSize", &MXScaleLRGeometry::mmaTileSize)
      .def_ro("mmaTileRegCount", &MXScaleLRGeometry::mmaTileRegCount)
      .def("globalMMATileGrid", &MXScaleLRGeometry::globalMMATileGrid,
           nb::arg("macroTile"), nb::arg("depthU"))
      .def("globalSubtileGrid", &MXScaleLRGeometry::globalSubtileGrid,
           nb::arg("macroTile"), nb::arg("depthU"))
      .def("subtileSizeBytes", &MXScaleLRGeometry::subtileSizeBytes);

  // -- Tag marker identities --------------------------------------------
  // Empty types analogous to C++ tag-dispatch types. They carry no data and
  // exist only so Python dispatch/shims can key on a stable C++ identity.
  nb::class_<GRTag_1x1>(g, "GRTag_1x1").def(nb::init<>());
  nb::class_<GRTag_1x2>(g, "GRTag_1x2").def(nb::init<>());
  nb::class_<GRTag_2x2>(g, "GRTag_2x2").def(nb::init<>());
  nb::class_<GRTag_TLU1>(g, "GRTag_TLU1").def(nb::init<>());
  nb::class_<LRTag_1x1>(g, "LRTag_1x1").def(nb::init<>());
  nb::class_<LRTag_1x2>(g, "LRTag_1x2").def(nb::init<>());
  nb::class_<LRTag_TLU1>(g, "LRTag_TLU1").def(nb::init<>());

  // -- Pre-defined gfx950 layout constants ------------------------------
  g.attr("MFMA_16x16_1B_4K_4V") = nb::cast(MMALayout(16, 1, 4, 64));
  g.attr("MFMA_16x16_1B_4K_8V") = nb::cast(MMALayout(16, 1, 8, 64));
  g.attr("MFMA_16x16_1B_4N_4V") = nb::cast(MMALayout(16, 1, 4, 64));
  g.attr("MFMA_SCALE_16x16_1B_MX32_8V") =
      nb::cast(MMAScaleLayout(16, 1, 0.25, 32, 64));
}

// ---------------------------------------------------------------------------
// TileInfo query layer (read-only) — AB (ABTilePair) case.
// ---------------------------------------------------------------------------
void bind_tile_info(nb::module_& t) {
  // -- Emit-leaf plan value types (instruction shape only) --------------
  nb::class_<SingleBufferLoadPlan>(
      t, "SingleBufferLoadPlan",
      "Data-only plan for SubtileGREmit.emitSingleBufferLoad: the skip "
      "decision, MUBUF offsetK, and per-load m0 offsets. Register state "
      "(soffset/voff) stays in Python.")
      .def_ro("skip", &SingleBufferLoadPlan::skip)
      .def_ro("grBaseId", &SingleBufferLoadPlan::grBaseId)
      .def_ro("offsetK", &SingleBufferLoadPlan::offsetK)
      .def_ro("m0Offsets", &SingleBufferLoadPlan::m0Offsets);

  nb::class_<DsReadEntry>(t, "DsReadEntry",
                          "One ds_read within a tile: destination VGPR offset "
                          "and sharedVgprLROffset index.")
      .def_ro("dstRegOffset", &DsReadEntry::dstRegOffset)
      .def_ro("addrIdx", &DsReadEntry::addrIdx);

  nb::class_<SingleDsReadPlan>(
      t, "SingleDsReadPlan",
      "Data-only plan for SubtileLREmit.emitSingleDsRead: the DS offset, "
      "register stride, and per-read structure. Destination/address VGPR "
      "base indices stay in Python.")
      .def_ro("regsPerDsRead", &SingleDsReadPlan::regsPerDsRead)
      .def_ro("mfmaId", &SingleDsReadPlan::mfmaId)
      .def_ro("offset", &SingleDsReadPlan::offset)
      .def_ro("numReadsForTile", &SingleDsReadPlan::numReadsForTile)
      .def_ro("reads", &SingleDsReadPlan::reads);

  // -- GR / LR offset-assignment plans (B16 / TLU0) ---------------------
  nb::class_<GROffsetAssignPlan>(
      t, "GROffsetAssignPlan",
      "Data-only scalar math for SubtileGREmit.graTileAssignment (B16/TLU0): "
      "derived block/row/partition sizes and per-load offset strides. No "
      "rocisa objects and no writer register state are computed here.")
      .def_ro("subIterKBytes", &GROffsetAssignPlan::subIterKBytes)
      .def_ro("loadWidth", &GROffsetAssignPlan::loadWidth)
      .def_ro("blockSize", &GROffsetAssignPlan::blockSize)
      .def_ro("numRowsPerLDSBanks", &GROffsetAssignPlan::numRowsPerLDSBanks)
      .def_ro("numRowsPerWave", &GROffsetAssignPlan::numRowsPerWave)
      .def_ro("partitionOffset", &GROffsetAssignPlan::partitionOffset)
      .def_ro("partitionMode", &GROffsetAssignPlan::partitionMode)
      .def_ro("subtileSizeElems", &GROffsetAssignPlan::subtileSizeElems)
      .def_ro("grAdvanceOffset", &GROffsetAssignPlan::grAdvanceOffset)
      .def_ro("bpeBits", &GROffsetAssignPlan::bpeBits)
      .def_ro("grSubtileRowOffset", &GROffsetAssignPlan::grSubtileRowOffset)
      .def_ro("sStride", &GROffsetAssignPlan::sStride)
      .def_ro("numGRPerSubtile", &GROffsetAssignPlan::numGRPerSubtile)
      .def_ro("loadRatioGR", &GROffsetAssignPlan::loadRatioGR)
      .def_ro("isFp8", &GROffsetAssignPlan::isFp8);

  nb::class_<LROffsetAssignPlan>(
      t, "LROffsetAssignPlan",
      "Data-only scalar math for SubtileLREmit.lraTileAssignment (B16/TLU0): "
      "derived block/row sizes, MFMA column stride, and the wave-partition "
      "selector. No rocisa objects and no writer register state are computed "
      "here.")
      .def_ro("subIterKBytes", &LROffsetAssignPlan::subIterKBytes)
      .def_ro("loadWidthLR", &LROffsetAssignPlan::loadWidthLR)
      .def_ro("loadWidthGR", &LROffsetAssignPlan::loadWidthGR)
      .def_ro("blockSize", &LROffsetAssignPlan::blockSize)
      .def_ro("numRowsPerLDSBanks", &LROffsetAssignPlan::numRowsPerLDSBanks)
      .def_ro("miM", &LROffsetAssignPlan::miM)
      .def_ro("numMFMACols", &LROffsetAssignPlan::numMFMACols)
      .def_ro("partitionOffset", &LROffsetAssignPlan::partitionOffset)
      .def_ro("sInterval", &LROffsetAssignPlan::sInterval)
      .def_ro("mWavesM", &LROffsetAssignPlan::mWavesM)
      .def_ro("wavePartMode", &LROffsetAssignPlan::wavePartMode)
      .def_ro("loadRatioGR", &LROffsetAssignPlan::loadRatioGR)
      .def_ro("isFp8", &LROffsetAssignPlan::isFp8);

  nb::class_<ABTileInfoQuery>(t, "ABTileInfoQuery")
      .def(nb::init<const ABGRGeometry&, const ABLRGeometry&, long, long, long,
                    long, long>(),
           nb::arg("gr"), nb::arg("lr"), nb::arg("macroTile"),
           nb::arg("depthU"), nb::arg("waveGroupSize"), nb::arg("waveSize"),
           nb::arg("numWaves"))
      // Inputs
      .def_ro("gr", &ABTileInfoQuery::gr)
      .def_ro("lr", &ABTileInfoQuery::lr)
      .def_ro("macroTile", &ABTileInfoQuery::macroTile)
      .def_ro("depthU", &ABTileInfoQuery::depthU)
      .def_ro("waveGroupSize", &ABTileInfoQuery::waveGroupSize)
      .def_ro("waveSize", &ABTileInfoQuery::waveSize)
      .def_ro("numWaves", &ABTileInfoQuery::numWaves)
      // Derived grids / ratios
      .def_ro("globalMMATileGrid", &ABTileInfoQuery::globalMMATileGrid)
      .def_ro("localMMATileGrid", &ABTileInfoQuery::localMMATileGrid)
      .def_ro("subtileShape", &ABTileInfoQuery::subtileShape)
      .def_prop_ro("subtileCount",
                   [](const ABTileInfoQuery& s) -> nb::object {
                     if (s.subtileCount.has_value())
                       return nb::cast(*s.subtileCount);
                     return nb::none();
                   })
      .def_prop_ro("subtileStride",
                   [](const ABTileInfoQuery& s) -> nb::object {
                     if (s.subtileStride.has_value())
                       return nb::cast(*s.subtileStride);
                     return nb::none();
                   })
      .def_ro("globalSubtileGrid", &ABTileInfoQuery::globalSubtileGrid)
      .def_ro("localSubtileGrid", &ABTileInfoQuery::localSubtileGrid)
      .def_ro("subtileSize", &ABTileInfoQuery::subtileSize)
      .def_ro("loadRatioGR", &ABTileInfoQuery::loadRatioGR)
      .def_ro("lrSubtileShape", &ABTileInfoQuery::lrSubtileShape)
      .def_ro("lrSubtileSize", &ABTileInfoQuery::lrSubtileSize)
      .def_ro("lrGlobalSubtileGrid", &ABTileInfoQuery::lrGlobalSubtileGrid)
      .def_ro("lrLocalSubtileGrid", &ABTileInfoQuery::lrLocalSubtileGrid)
      .def_ro("loadRatioLR", &ABTileInfoQuery::loadRatioLR)
      // Count properties
      .def_prop_ro("numMFMATiles", &ABTileInfoQuery::numMFMATiles)
      .def_prop_ro("numGlobalSubtiles", &ABTileInfoQuery::numGlobalSubtiles)
      .def_prop_ro("numLocalSubtiles", &ABTileInfoQuery::numLocalSubtiles)
      // Grid / index query helpers
      .def("getLocalSubtileLinearId", &ABTileInfoQuery::getLocalSubtileLinearId,
           nb::arg("sId0"), nb::arg("sId1"))
      .def("grLoadIndexForSubtile", &ABTileInfoQuery::grLoadIndexForSubtile,
           nb::arg("sId0"), nb::arg("sId1"), nb::arg("loadIdx") = 0)
      .def("lrTileIndexForSubtile", &ABTileInfoQuery::lrTileIndexForSubtile,
           nb::arg("sId0"), nb::arg("sId1"), nb::arg("mfmaId") = 0)
      .def("globalMmaTilesForSubtile",
           &ABTileInfoQuery::globalMmaTilesForSubtile, nb::arg("sId0"),
           nb::arg("sId1"))
      .def("waveMmaTilesForSubtile", &ABTileInfoQuery::waveMmaTilesForSubtile,
           nb::arg("sId0"), nb::arg("sId1"))
      .def("grRegGroupForSubtileRow",
           &ABTileInfoQuery::grRegGroupForSubtileRow, nb::arg("sId0"))
      .def("getSubtileShapeLinearId",
           &ABTileInfoQuery::getSubtileShapeLinearId, nb::arg("k0"),
           nb::arg("k1"))
      .def_prop_ro("numGRPerSubtile", &ABTileInfoQuery::numGRPerSubtile)
      // Emit-leaf plans (instruction shape only).
      .def("singleBufferLoadPlan", &ABTileInfoQuery::singleBufferLoadPlan,
           nb::arg("sId0"), nb::arg("sId1"))
      .def("singleDsReadPlan", &ABTileInfoQuery::singleDsReadPlan,
           nb::arg("sId0"), nb::arg("sId1"), nb::arg("subIterK"),
           nb::arg("numRegs"))
      // Offset-assignment scalar math (B16 / TLU0).
      .def("grOffsetAssignPlan", &ABTileInfoQuery::grOffsetAssignPlan,
           nb::arg("ldsRowBankSize"))
      .def("lrOffsetAssignPlan", &ABTileInfoQuery::lrOffsetAssignPlan,
           nb::arg("ldsRowBankSize"), nb::arg("mWavesM"));
}

// ---------------------------------------------------------------------------
// Subtile emit-leaf data-only decisions (no rocisa). MFMA instType selection.
// ---------------------------------------------------------------------------
void bind_emit(nb::module_& e) {
  using namespace tw::subtile::emit;
  e.def("mfma_f8f6f4_inst_type", &mfma_f8f6f4_inst_type, nb::arg("aIsF8"),
        nb::arg("aIsBF8"), nb::arg("aIsF4"), nb::arg("bIsF8"),
        nb::arg("bIsBF8"), nb::arg("bIsF4"), nb::arg("sourceSwap") = false,
        "Select the rocisa InstType member name for the "
        "V_MFMA_SCALE_F32_16x16x128_F8F6F4 family from per-operand 8-bit/4-bit "
        "float predicates. Backs Kernel._selectF8F6F4InstType (called "
        "unconditionally); raises on unsupported combinations (surfaced as a "
        "RuntimeError).");
}

// ---------------------------------------------------------------------------
// Subtile InstructionScheduler slot-placement algorithm (data-only model).
// ---------------------------------------------------------------------------
void bind_instruction_scheduler(nb::module_& s) {
  using namespace tw::subtile::insched;

  nb::enum_<InstKind>(s, "InstKind",
                      "Instruction classification used by the scheduler. "
                      "Mirrors the isinstance() predicates of the Python "
                      "InstructionScheduler.")
      .value("Mfma", InstKind::Mfma)
      .value("LocalRead", InstKind::LocalRead)
      .value("GlobalRead", InstKind::GlobalRead)
      .value("WaitCnt", InstKind::WaitCnt)
      .value("M0Update", InstKind::M0Update)
      .value("Other", InstKind::Other);

  nb::class_<InstRef>(s, "Instruction",
                      "Data-only view of one rocisa instruction: its kind plus "
                      "the waitcnt fields the vmcnt post-pass needs.")
      .def(nb::init<InstKind, long, bool>(), nb::arg("kind"),
           nb::arg("vlcnt") = -1, nb::arg("adjustVmcnt") = true)
      .def_ro("kind", &InstRef::kind)
      .def_ro("vlcnt", &InstRef::vlcnt)
      .def_ro("adjustVmcnt", &InstRef::adjustVmcnt);

  nb::class_<ModuleRef>(s, "ModuleRef",
                        "Data-only view of one LogicalScheduler.EmittedModule.")
      .def(nb::init<int, std::string, std::optional<int>,
                    std::vector<InstRef>>(),
           nb::arg("moduleId"), nb::arg("opType"),
           nb::arg("before") = nb::none(),
           nb::arg("instructions") = std::vector<InstRef>{})
      .def_ro("moduleId", &ModuleRef::moduleId)
      .def_ro("opType", &ModuleRef::opType)
      .def_prop_ro("before",
                   [](const ModuleRef& m) -> nb::object {
                     if (m.before.has_value()) return nb::cast(*m.before);
                     return nb::none();
                   })
      .def_ro("instructions", &ModuleRef::instructions);

  nb::class_<ScheduleResult>(s, "ScheduleResult",
                             "Result of the slot-placement algorithm.")
      // Final emission order as a list of (moduleIndex, instIdx) tuples.
      .def_prop_ro("order",
                   [](const ScheduleResult& r) {
                     nb::list out;
                     for (const auto& p : r.order)
                       out.append(nb::make_tuple(p.first, p.second));
                     return out;
                   })
      .def_ro("kinds", &ScheduleResult::kinds)
      .def_ro("vlcnt", &ScheduleResult::vlcnt)
      // (orderIndex, delta) pairs the shim applies to live waitcnt objects.
      .def_prop_ro("vmcntAdjustments", [](const ScheduleResult& r) {
        nb::list out;
        for (const auto& p : r.vmcntAdjustments)
          out.append(nb::make_tuple(p.first, p.second));
        return out;
      });

  s.def("schedule", &schedule, nb::arg("modules"),
        "Run the subtile instruction-scheduling slot-placement algorithm over "
        "a data-only emitted-module chain and return the final emission order "
        "plus the vmcnt post-pass result. Raises ValueError on structural "
        "precondition violations (e.g. not exactly one MFMA module).");
}

// ---------------------------------------------------------------------------
// Subtile LogicalScheduler data/config primitives (pure value types).
// ---------------------------------------------------------------------------
void bind_logical_scheduler(nb::module_& s) {
  using namespace tw::subtile::lsched;

  // -- Pass enum --------------------------------------------------------
  nb::enum_<Pass>(s, "Pass",
                  "Scheduler passes in dependency order. The numeric value "
                  "defines topological order. Mirrors LogicalScheduler.Pass.")
      .value("LR", Pass::LR)
      .value("VGPR_TILES", Pass::VGPR_TILES)
      .value("GR", Pass::GR)
      .value("DEPS", Pass::DEPS)
      .value("REMOVE_GR_DEPS", Pass::REMOVE_GR_DEPS)
      .value("REMOVE_LR_DEPS", Pass::REMOVE_LR_DEPS)
      .value("REMOVE_DEPS", Pass::REMOVE_DEPS)
      .value("GR_INC", Pass::GR_INC)
      .value("GROUP_LR_GR", Pass::GROUP_LR_GR)
      .value("REMOVE_WAIT_LR_SYNC", Pass::REMOVE_WAIT_LR_SYNC)
      .value("EMIT", Pass::EMIT)
      .value("BUILD", Pass::BUILD)
      .value("POPULATE", Pass::POPULATE);

  // -- free helpers -----------------------------------------------------
  s.def("fmt_mt", &fmt_mt, nb::arg("mt"),
        "Format an MT iteration integer as a display string "
        "(0 -> 'n', 1 -> 'n+1').");

  // -- MFMATileRange ----------------------------------------------------
  nb::class_<MFMATileRange>(s, "MFMATileRange",
                            "A rectangular range of MFMA tile coordinates for "
                            "one read.")
      .def(nb::init<int, int, int, int>(), nb::arg("subIterK_start"),
           nb::arg("subIterK_end"), nb::arg("tileId_start"),
           nb::arg("tileId_end"))
      .def_ro("subIterK_start", &MFMATileRange::subIterK_start)
      .def_ro("subIterK_end", &MFMATileRange::subIterK_end)
      .def_ro("tileId_start", &MFMATileRange::tileId_start)
      .def_ro("tileId_end", &MFMATileRange::tileId_end)
      .def_prop_ro("subIterK_list", &MFMATileRange::subIterK_list)
      .def_prop_ro("tileId_list", &MFMATileRange::tileId_list)
      .def("fmt_k", &MFMATileRange::fmt_k)
      .def("fmt_tiles", &MFMATileRange::fmt_tiles);

  // -- ReadGranularity --------------------------------------------------
  nb::class_<ReadGranularity>(s, "ReadGranularity",
                              "Load granularity for one operation on one "
                              "tensor, measured in MFMA tiles.")
      .def(nb::init<int, int>(), nb::arg("mn"), nb::arg("k"))
      .def_ro("mn", &ReadGranularity::mn)
      .def_ro("k", &ReadGranularity::k)
      .def("tile_range", &ReadGranularity::tile_range, nb::arg("k"),
           nb::arg("t_start"), nb::arg("t_end"));

  // -- SchedulerConfig --------------------------------------------------
  nb::class_<SchedulerConfig>(s, "SchedulerConfig",
                              "Configuration for the MFMATile-based scheduler.")
      .def(
          "__init__",
          [](SchedulerConfig* self, int numMFMATilesM, int numMFMATilesN,
             int numSubIterK, ReadGranularity lrA, ReadGranularity lrB,
             ReadGranularity grA, ReadGranularity grB,
             std::optional<ReadGranularity> lrSA,
             std::optional<ReadGranularity> lrSB,
             std::optional<ReadGranularity> grSA,
             std::optional<ReadGranularity> grSB, PartitionSpec partitionSizeM,
             PartitionSpec partitionSizeN, int pgr) {
            new (self) SchedulerConfig();
            self->numMFMATilesM = numMFMATilesM;
            self->numMFMATilesN = numMFMATilesN;
            self->numSubIterK = numSubIterK;
            self->lrA = lrA;
            self->lrB = lrB;
            self->grA = grA;
            self->grB = grB;
            self->lrSA = lrSA;
            self->lrSB = lrSB;
            self->grSA = grSA;
            self->grSB = grSB;
            self->partitionSizeM = partitionSizeM;
            self->partitionSizeN = partitionSizeN;
            self->pgr = pgr;
            self->post_init();
          },
          nb::arg("numMFMATilesM"), nb::arg("numMFMATilesN"),
          nb::arg("numSubIterK"), nb::arg("lrA"), nb::arg("lrB"),
          nb::arg("grA"), nb::arg("grB"), nb::arg("lrSA") = nb::none(),
          nb::arg("lrSB") = nb::none(), nb::arg("grSA") = nb::none(),
          nb::arg("grSB") = nb::none(), nb::arg("partitionSizeM") = 0,
          nb::arg("partitionSizeN") = 0, nb::arg("pgr") = 2)
      .def_ro("numMFMATilesM", &SchedulerConfig::numMFMATilesM)
      .def_ro("numMFMATilesN", &SchedulerConfig::numMFMATilesN)
      .def_ro("numSubIterK", &SchedulerConfig::numSubIterK)
      .def_ro("lrA", &SchedulerConfig::lrA)
      .def_ro("lrB", &SchedulerConfig::lrB)
      .def_ro("grA", &SchedulerConfig::grA)
      .def_ro("grB", &SchedulerConfig::grB)
      .def_ro("lrSA", &SchedulerConfig::lrSA)
      .def_ro("lrSB", &SchedulerConfig::lrSB)
      .def_ro("grSA", &SchedulerConfig::grSA)
      .def_ro("grSB", &SchedulerConfig::grSB)
      // Raw input partition specs (int or explicit per-partition list),
      // preserved verbatim like the Python dataclass fields. post_init consumes
      // them into the derived partitionSizes{M,N}; these stay readable for
      // parity with the Python SchedulerConfig.
      .def_ro("partitionSizeM", &SchedulerConfig::partitionSizeM)
      .def_ro("partitionSizeN", &SchedulerConfig::partitionSizeN)
      .def_ro("pgr", &SchedulerConfig::pgr)
      .def_ro("plr", &SchedulerConfig::plr)
      .def_ro("offsetPartition", &SchedulerConfig::offsetPartition)
      .def_prop_ro("partitionSizesM", &SchedulerConfig::partitionSizesM)
      .def_prop_ro("partitionSizesN", &SchedulerConfig::partitionSizesN)
      .def_prop_ro("prefixM",
                   [](const SchedulerConfig& c) { return c._prefixM; })
      .def_prop_ro("prefixN",
                   [](const SchedulerConfig& c) { return c._prefixN; })
      .def_prop_ro("hasScale", &SchedulerConfig::hasScale)
      .def_prop_ro("numPartitionsM", &SchedulerConfig::numPartitionsM)
      .def_prop_ro("numPartitionsN", &SchedulerConfig::numPartitionsN)
      .def_prop_ro("numPartitions", &SchedulerConfig::numPartitions)
      .def_static(
          "_normalize_partition_sizes",
          [](PartitionSpec spec, int total, const std::string& dim, int mn) {
            return SchedulerConfig::normalize_partition_sizes(spec, total, dim,
                                                              mn);
          },
          nb::arg("spec"), nb::arg("total"), nb::arg("dim"), nb::arg("mn") = 1)
      .def_static("get_partition_candidates",
                  &SchedulerConfig::get_partition_candidates, nb::arg("M"),
                  nb::arg("N"),
                  "Return partition candidates as [(sizeM, sizeN), ...] given "
                  "the two localMMATileGrid[0] values M and N.");

  // -- Placement value types -------------------------------------------
  nb::class_<MFMAPlacement>(s, "MFMAPlacement",
                            "MFMA operation consuming data for one subIterK.")
      .def(nb::init<int, MFMATileRange, MFMATileRange>(), nb::arg("subIterK"),
           nb::arg("tileA"), nb::arg("tileB"))
      .def_ro("subIterK", &MFMAPlacement::subIterK)
      .def_ro("tileA", &MFMAPlacement::tileA)
      .def_ro("tileB", &MFMAPlacement::tileB)
      // Pass-populated fields (default empty; filled by the Python passes).
      .def_rw("deps", &MFMAPlacement::deps)
      .def_rw("preOps", &MFMAPlacement::preOps)
      .def_rw("postOps", &MFMAPlacement::postOps)
      .def_rw("vgpr_tile_maps", &MFMAPlacement::vgpr_tile_maps)
      .def_ro("kind", &MFMAPlacement::kind)
      .def("__str__", &MFMAPlacement::str);

  nb::class_<LRPlacement>(s, "LRPlacement",
                          "Local Read placement for one tensor in one subIterK "
                          "slot.")
      .def(nb::init<std::string, int, MFMATileRange, int, int>(),
           nb::arg("tensor"), nb::arg("mtIteration"), nb::arg("tiles"),
           nb::arg("subIterK_slot"), nb::arg("partition") = 0)
      .def_ro("tensor", &LRPlacement::tensor)
      .def_ro("mtIteration", &LRPlacement::mtIteration)
      .def_ro("tiles", &LRPlacement::tiles)
      .def_ro("subIterK_slot", &LRPlacement::subIterK_slot)
      .def_ro("partition", &LRPlacement::partition)
      // Pass-populated fields (default empty; filled by the Python passes).
      .def_rw("deps", &LRPlacement::deps)
      .def_rw("preOps", &LRPlacement::preOps)
      .def_rw("postOps", &LRPlacement::postOps)
      .def_rw("vgpr_tile_map", &LRPlacement::vgpr_tile_map)
      .def_ro("kind", &LRPlacement::kind)
      .def("__str__", &LRPlacement::str);

  nb::class_<GRPlacement>(s, "GRPlacement",
                          "Global Read placement for one tensor in one "
                          "subIterK slot.")
      .def(nb::init<std::string, int, MFMATileRange, int, int>(),
           nb::arg("tensor"), nb::arg("mtIteration"), nb::arg("tiles"),
           nb::arg("subIterK_slot"), nb::arg("partition") = 0)
      .def_ro("tensor", &GRPlacement::tensor)
      .def_ro("mtIteration", &GRPlacement::mtIteration)
      .def_ro("tiles", &GRPlacement::tiles)
      .def_ro("subIterK_slot", &GRPlacement::subIterK_slot)
      .def_ro("partition", &GRPlacement::partition)
      // Pass-populated fields (default empty; filled by the Python passes).
      .def_rw("deps", &GRPlacement::deps)
      .def_rw("preOps", &GRPlacement::preOps)
      .def_rw("postOps", &GRPlacement::postOps)
      .def_ro("kind", &GRPlacement::kind)
      .def("__str__", &GRPlacement::str);

  // -- Dependency / before-chain op value types ------------------------
  nb::class_<WaitGRCounts>(s, "WaitGRCounts",
                           "Per-tensor inflight load counts for wait_gr preOp.")
      .def(nb::init<int, int, int, int>(), nb::arg("A") = 0, nb::arg("B") = 0,
           nb::arg("SA") = 0, nb::arg("SB") = 0)
      .def_ro("A", &WaitGRCounts::A)
      .def_ro("B", &WaitGRCounts::B)
      .def_ro("SA", &WaitGRCounts::SA)
      .def_ro("SB", &WaitGRCounts::SB)
      .def("__str__", &WaitGRCounts::str);

  nb::class_<WaitGROp>(s, "WaitGROp",
                       "Wait for global reads to complete. Optionally includes "
                       "a sync barrier.")
      .def(nb::init<std::optional<WaitGRCounts>, bool, bool>(),
           nb::arg("wait_gr_counts") = nb::none(), nb::arg("has_sync") = false,
           nb::arg("adjustVmcnt") = true)
      .def_ro("wait_gr_counts", &WaitGROp::wait_gr_counts)
      .def_ro("has_sync", &WaitGROp::has_sync)
      .def_ro("adjustVmcnt", &WaitGROp::adjustVmcnt)
      .def_ro("kind", &WaitGROp::kind)
      .def("__str__", &WaitGROp::str);

  nb::class_<WaitLROp>(s, "WaitLROp",
                       "Wait for local reads to complete. Optionally includes "
                       "a sync barrier.")
      .def(nb::init<bool>(), nb::arg("has_sync") = false)
      .def_ro("has_sync", &WaitLROp::has_sync)
      .def_ro("kind", &WaitLROp::kind)
      .def("__str__", &WaitLROp::str);

  nb::class_<SyncOp>(s, "SyncOp", "Standalone sync barrier.")
      .def(nb::init<>())
      .def_ro("kind", &SyncOp::kind)
      .def("__str__", &SyncOp::str);

  nb::class_<MaskKOp>(s, "MaskKOp",
                      "Zero A/B vgprs whose K-index >= remaining tail K for one "
                      "subIterK group.")
      .def(nb::init<int>(), nb::arg("subIterK") = 0)
      .def_ro("subIterK", &MaskKOp::subIterK)
      .def_rw("vgpr_tile_map", &MaskKOp::vgpr_tile_map)
      .def_ro("kind", &MaskKOp::kind)
      .def("__str__", &MaskKOp::str);

  nb::class_<LRIncOp>(s, "LRIncOp",
                      "LDS buffer swap for local reads on a specific tensor.")
      .def(nb::init<std::string>(), nb::arg("tensor") = std::string())
      .def_ro("tensor", &LRIncOp::tensor)
      .def_ro("kind", &LRIncOp::kind)
      .def("__str__", &LRIncOp::str);

  nb::class_<GRIncOp>(s, "GRIncOp",
                      "Pointer update + LDS swap for global reads on a specific "
                      "tensor.")
      .def(nb::init<std::string>(), nb::arg("tensor") = std::string())
      .def_ro("tensor", &GRIncOp::tensor)
      .def_ro("kind", &GRIncOp::kind)
      .def("__str__", &GRIncOp::str);

  nb::class_<SkipOp>(s, "SkipOp",
                     "Skip guard: compare LoopCounter and branch.")
      .def(nb::init<std::string, int, std::string, bool, std::string>(),
           nb::arg("compare") = std::string(), nb::arg("value") = 0,
           nb::arg("target") = std::string(), nb::arg("rawLabel") = false,
           nb::arg("branchComment") = std::string())
      .def_ro("compare", &SkipOp::compare)
      .def_ro("value", &SkipOp::value)
      .def_ro("target", &SkipOp::target)
      .def_ro("rawLabel", &SkipOp::rawLabel)
      .def_ro("branchComment", &SkipOp::branchComment)
      .def_ro("kind", &SkipOp::kind)
      .def_prop_ro("tensor", &SkipOp::tensor)
      .def("__str__", &SkipOp::str);

  nb::class_<InlineModuleOp>(
      s, "InlineModuleOp",
      "Inline a writer-built Module at this point in the schedule. The Python "
      "`build` Callable is not modeled in C++; only label / kind / str() are "
      "ported.")
      .def(nb::init<std::string>(), nb::arg("label") = std::string("inline"))
      .def_rw("label", &InlineModuleOp::label)
      .def_ro("kind", &InlineModuleOp::kind)
      .def("__str__", &InlineModuleOp::str);

  // -- Dependency / slot / emitted-module value types ------------------
  nb::class_<Dep>(s, "Dep",
                  "Dependency on another placement (annotate_deps output). "
                  "`ref` is a value copy of the referenced LR/GR placement.")
      .def(nb::init<std::variant<LRPlacement, GRPlacement>, int>(),
           nb::arg("ref"), nb::arg("mt_offset") = 0)
      .def_rw("ref", &Dep::ref)
      .def_rw("mt_offset", &Dep::mt_offset);

  nb::class_<SubIterKSlot>(s, "SubIterKSlot",
                           "All operations placed in one subIterK step.")
      .def(nb::init<int>(), nb::arg("subIterK"))
      .def_rw("subIterK", &SubIterKSlot::subIterK)
      .def_rw("mfma", &SubIterKSlot::mfma)
      .def_rw("lrs", &SubIterKSlot::lrs)
      .def_rw("grs", &SubIterKSlot::grs);

  nb::class_<EmittedModule>(
      s, "EmittedModule",
      "One emitted module with before-link for instruction scheduling. The "
      "rocisa `instructions` list (filled during Python emission) is not "
      "modeled here; `opType` derives from the source's kind.")
      .def(nb::init<int, std::optional<int>, std::optional<Emittable>>(),
           nb::arg("moduleId") = -1, nb::arg("before") = nb::none(),
           nb::arg("source") = nb::none())
      .def_rw("moduleId", &EmittedModule::moduleId)
      .def_rw("before", &EmittedModule::before)
      .def_rw("source", &EmittedModule::source)
      .def_prop_ro("opType", &EmittedModule::opType);
}

// ---------------------------------------------------------------------------
// Subtile LogicalScheduler writer-free pass pipeline (data-only model).
// ---------------------------------------------------------------------------
void bind_logical_scheduler_passes(nb::module_& s) {
  using tw::subtile::lsched::SchedulerConfig;
  using LS = tw::subtile::lsched::passes::LogicalScheduler;

  nb::class_<LS>(
      s, "LogicalScheduler",
      "Writer-free C++ port of the subtile LogicalScheduler pass pipeline "
      "(place_LRs through emit/build). Operates purely on the data-only logical "
      "schedule; it does NOT populate rocisa instructions, allocate writer VGPR "
      "pools, or emit Kernel.mainLoop control flow. Each pass auto-runs its "
      "prerequisites, mirroring the Python LogicalScheduler. The print_* methods "
      "produce byte-identical output to the Python print_* helpers for "
      "pass-by-pass parity testing.")
      .def(nb::init<SchedulerConfig>(), nb::arg("config"))
      .def("place_LRs", &LS::place_LRs)
      .def("assign_vgpr_tiles", &LS::assign_vgpr_tiles)
      .def("place_GRs", &LS::place_GRs)
      .def("annotate_deps", &LS::annotate_deps)
      .def("remove_unnecessary_gr_deps", &LS::remove_unnecessary_gr_deps)
      .def("remove_unnecessary_lr_deps", &LS::remove_unnecessary_lr_deps)
      .def("remove_cross_deps", &LS::remove_cross_deps)
      .def("insert_gr_lr_inc", &LS::insert_gr_lr_inc)
      .def("group_lr_gr", &LS::group_lr_gr)
      .def("remove_unnecessary_wait_lr_sync",
           &LS::remove_unnecessary_wait_lr_sync)
      .def("emit", &LS::emit)
      .def("build", &LS::build)
      .def("build_ngll", &LS::build_ngll,
           "Build the NGLL (No Global Load Loop) variant by rewriting the "
           "mainloop emit graph (drop GR(n+2), zero wait_gr counts). Result is "
           "exposed via value_ngll().")
      .def("build_nll", &LS::build_nll,
           "Build the NLL (No Load Loop) variant by rewriting the mainloop "
           "emit graph (drop GR/LR(n+1)/gr_inc, zero wait_gr counts, prune "
           "orphan sync/wait_lr). Result is exposed via value_nll().")
      .def("build_preloop", &LS::build_preloop,
           "Build the preloop pipeline-priming sequence (fresh GR/LR "
           "placements + skip guards). Result is exposed via value_preloop().")
      .def("build_tailloop_pgr0", &LS::build_tailloop_pgr0, nb::arg("tile_maps"),
           nb::arg("bf16"), nb::arg("miK"),
           "Build the PGR0-template tail loop from the flat tail tile maps "
           "(per-partition {tensor: [{group: tileId}]}), the BF16 boundary "
           "flag, and MatrixInstK. The InlineModuleOp boundary build callback "
           "is attached by the Python wrapper. Result is exposed via "
           "value_tailloop().")
      .def("value_ngll", &LS::value_ngll,
           "Materialize the NGLL variant as value EmittedModules "
           "(list[partition][subIterK][EmittedModule]); coordinate-only "
           "placement sources, like value_emitted().")
      .def("value_nll", &LS::value_nll,
           "Materialize the NLL variant as value EmittedModules "
           "(coordinate-only placement sources, like value_emitted()).")
      .def("value_preloop", &LS::value_preloop,
           "Materialize the preloop variant as value EmittedModules; sources "
           "carry full placement data (incl. LR vgpr tile maps).")
      .def("value_tailloop", &LS::value_tailloop,
           "Materialize the tail-loop variant as value EmittedModules; sources "
           "carry full placement data (vgpr tile maps, mask tile map).")
      .def("value_partitions", &LS::value_partitions,
           "Materialize the computed schedule as bound value types "
           "(list[list[SubIterKSlot]]) so the Python writer can rebuild its "
           "dataclass partitions. Deps carry coordinate-only ref copies; the "
           "Python converter re-establishes object identity by coordinate "
           "match.")
      .def("value_emitted", &LS::value_emitted,
           "Materialize the emitted before-link graph as bound value "
           "EmittedModules (list[partition][subIterK][EmittedModule]) so the "
           "Python writer can rebuild its EmittedModule dataclasses. Each "
           "module's source is a coordinate-only value Emittable; the Python "
           "converter re-uses its persistent placement dataclasses for "
           "placement sources (preserving identity for vgpr-map flow-through) "
           "and builds fresh op dataclasses for before-chain ops.")
      .def_ro("needs_unrolling", &LS::needs_unrolling)
      .def_ro("unroll_factor", &LS::unroll_factor)
      .def_prop_ro("tile_peaks",
                   [](const LS& self) { return self.tile_peaks; })
      .def("print_lr", &LS::print_lr)
      .def("print_vgpr", &LS::print_vgpr)
      .def("print_gr", &LS::print_gr)
      .def("print_deps", &LS::print_deps)
      .def("print_remove_deps", &LS::print_remove_deps)
      .def("print_group_lr_gr", &LS::print_group_lr_gr)
      .def("print_emit", &LS::print_emit);
}

}  // namespace

NB_MODULE(_tensile_writer, m) {
  m.doc() =
      "TensileLite C++ migration scaffold (nanobind). Hosts the pure subtile "
      "geometry math under the subtile.geometry submodule and the read-only "
      "TileInfo query layer under the subtile.tile_info submodule.";

  nb::module_ subtile = m.def_submodule(
      "subtile", "Subtile-based kernel geometry (pure math).");
  nb::module_ geometry = subtile.def_submodule(
      "geometry", "Tile geometry value/query layer ported from "
                  "Tensile.Components.Subtile.SubtileGeometry.");
  bind_geometry(geometry);

  nb::module_ tile_info = subtile.def_submodule(
      "tile_info", "Read-only TileInfo construction + grid/index query layer "
                   "ported from Tensile.Components.Subtile.Kernel.TileInfo "
                   "(ABTilePair case).");
  bind_tile_info(tile_info);

  nb::module_ instruction_scheduler = subtile.def_submodule(
      "instruction_scheduler",
      "Subtile instruction-scheduling slot-placement algorithm ported from "
      "Tensile.Components.Subtile.InstructionScheduler. Operates on a "
      "data-only instruction/module model; a Python shim maps the resulting "
      "order back onto live rocisa objects.");
  bind_instruction_scheduler(instruction_scheduler);

  nb::module_ logical_scheduler = subtile.def_submodule(
      "logical_scheduler",
      "Subtile LogicalScheduler data/config primitives ported from "
      "Tensile.Components.Subtile.LogicalScheduler. Pure value/config types "
      "(Pass, ReadGranularity, MFMATileRange, SchedulerConfig, placement/op "
      "value types) only; the scheduling passes remain in Python.");
  bind_logical_scheduler(logical_scheduler);
  bind_logical_scheduler_passes(logical_scheduler);

  nb::module_ emit = subtile.def_submodule(
      "emit",
      "Subtile emit-leaf data-only decisions ported from "
      "Tensile.Components.Subtile (Kernel.emitMfmaInstruction instType "
      "selection). No rocisa objects are built in C++; the Python emit "
      "functions construct the rocisa Module from these decisions.");
  bind_emit(emit);
}
