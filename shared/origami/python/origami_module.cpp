// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/hardware.hpp"
#include "origami/streamk.hpp"
#include "origami/utils.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

using Hardware = origami::hardware_t;

NB_MODULE(origami, m)
{
    nanobind::enum_<Hardware::architecture_t>(m, "architecture_t")
        .value("gfx942", Hardware::architecture_t::gfx942)
        .value("gfx950", Hardware::architecture_t::gfx950)
        .export_values();

    nanobind::enum_<origami::data_type_t>(m, "data_type_t")
        .value("Float", origami::data_type_t::Float)
        .value("ComplexFloat", origami::data_type_t::ComplexFloat)
        .value("ComplexDouble", origami::data_type_t::ComplexDouble)
        .value("Double", origami::data_type_t::Double)
        .value("Half", origami::data_type_t::Half)
        .value("Int8x4", origami::data_type_t::Int8x4)
        .value("Int32", origami::data_type_t::Int32)
        .value("BFloat16", origami::data_type_t::BFloat16)
        .value("Int8", origami::data_type_t::Int8)
        .value("Int64", origami::data_type_t::Int64)
        .value("XFloat32", origami::data_type_t::XFloat32)
        .value("Float8_fnuz", origami::data_type_t::Float8_fnuz)
        .value("BFloat8_fnuz", origami::data_type_t::BFloat8_fnuz)
        .value("Float8BFloat8_fnuz", origami::data_type_t::Float8BFloat8_fnuz)
        .value("BFloat8Float8_fnuz", origami::data_type_t::BFloat8Float8_fnuz)
        .value("Float8", origami::data_type_t::Float8)
        .value("BFloat8", origami::data_type_t::BFloat8)
        .value("Float8BFloat8", origami::data_type_t::Float8BFloat8)
        .value("BFloat8Float8", origami::data_type_t::BFloat8Float8)
        .value("Float6", origami::data_type_t::Float6)
        .value("BFloat6", origami::data_type_t::BFloat6)
        .value("Float4", origami::data_type_t::Float4)
        .export_values();

    m.def("int_to_data_type",
          &origami::int_to_data_type,
          "Convert int to data_type_t.");

    nanobind::class_<Hardware>(m, "Hardware")
        .def(nanobind::init<Hardware::architecture_t,
                            size_t,
                            size_t,
                            size_t,
                            double,
                            double,
                            double,
                            size_t,
                            double,
                            size_t,
                            double>())
        .def("print", &Hardware::print)
        .def("print_debug_info", &Hardware::print_debug_info)
        .def_rw("N_CU", &Hardware::N_CU)
        .def_rw("LDS_capacity", &Hardware::LDS_capacity)
        .def_rw("mem1_perf_ratio", &Hardware::mem1_perf_ratio)
        .def_rw("mem2_perf_ratio", &Hardware::mem2_perf_ratio)
        .def_rw("mem3_perf_ratio", &Hardware::mem3_perf_ratio)
        .def_rw("L2_capacity", &Hardware::L2_capacity)
        .def_rw("CU_per_L2", &Hardware::CU_per_L2)
        .def_rw("compute_clock_ghz", &Hardware::compute_clock_ghz)
        .def_rw("parallel_mi_cu", &Hardware::parallel_mi_cu)
        .def_rw("percent_bw_per_wg", &Hardware::percent_bw_per_wg)
        .def_rw("NUM_XCD", &Hardware::NUM_XCD);

    m.def("get_hardware_for_device",
          &Hardware::get_hardware_for_device,
          "This gets a hardware object for a device.");

    m.def("datatype_to_bits", &origami::data_type_to_bits, "Return the number of bits in a datatype");
    m.def("string_to_datatype", &origami::string_to_data_type, "Convert a string representation of a datatype into data_type_t enum");
    m.def("select_best_macro_tile_size",
          &origami::select_best_macro_tile_size,
          "Get best macro tile sizes.");
    m.def("select_grid", &origami::streamk::select_grid, "Select Best StreamK Grid Size");
    m.def("compute_total_latency", &origami::compute_total_latency, "compute_total_latency");
    m.def("select_best_wgm", &origami::select_best_wgm, "Get best workgroup mapping.");
}
