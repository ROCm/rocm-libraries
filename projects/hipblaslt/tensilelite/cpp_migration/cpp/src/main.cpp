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
// The Python package tensile_writer/ re-exports this, and
// Tensile/Components/Subtile/SubtileGeometry.py can optionally delegate its
// pure-math query methods here (opt-in via TENSILE_WRITER_CPP).

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "tensile_writer/subtile_geometry.hpp"

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
            // Parity reads: Python reads these even though the derived result
            // depends only on wg_m and mt_mma.
            (void)(wg.at(0) * wg.at(1));
            (void)kernel_int(kernel, "WavefrontSize");
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

}  // namespace

NB_MODULE(_tensile_writer, m) {
  m.doc() =
      "TensileLite C++ migration scaffold (nanobind). Hosts the pure subtile "
      "geometry math under the subtile.geometry submodule.";

  nb::module_ subtile = m.def_submodule(
      "subtile", "Subtile-based kernel geometry (pure math).");
  nb::module_ geometry = subtile.def_submodule(
      "geometry", "Tile geometry value/query layer ported from "
                  "Tensile.Components.Subtile.SubtileGeometry.");
  bind_geometry(geometry);
}
