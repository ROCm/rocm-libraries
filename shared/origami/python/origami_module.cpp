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

using hardware_t = origami::hardware_t;
namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(origami, m)
{
    nanobind::enum_<hardware_t::architecture_t>(m, "architecture_t")
        .value("gfx942", hardware_t::architecture_t::gfx942)
        .value("gfx950", hardware_t::architecture_t::gfx950)
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

    nanobind::enum_<origami::streamk::reduction_type>(m, "reduction_type")
        .value("Tree", origami::streamk::reduction_type::Tree)
        .value("Parallel", origami::streamk::reduction_type::Parallel)
        .export_values();

    m.def("int_to_reduction_type",
          &origami::streamk::int_to_reduction_type,
          "Convert int to reduction_type.");

    nanobind::class_<hardware_t>(m, "hardware_t")
        .def(nanobind::init<hardware_t::architecture_t,
                            size_t,
                            size_t,
                            size_t,
                            double,
                            double,
                            double,
                            size_t,
                            double,
                            size_t,
                            std::tuple<double, double, double>>())
        .def("print", &hardware_t::print)
        .def("print_debug_info", &hardware_t::print_debug_info)
        .def_rw("N_CU", &hardware_t::N_CU)
        .def_rw("lds_capacity", &hardware_t::lds_capacity)
        .def_rw("mem1_perf_ratio", &hardware_t::mem1_perf_ratio)
        .def_rw("mem2_perf_ratio", &hardware_t::mem2_perf_ratio)
        .def_rw("mem3_perf_ratio", &hardware_t::mem3_perf_ratio)
        .def_rw("L2_capacity", &hardware_t::L2_capacity)
        .def_rw("CU_per_L2", &hardware_t::CU_per_L2)
        .def_rw("compute_clock_ghz", &hardware_t::compute_clock_ghz)
        .def_rw("parallel_mi_cu", &hardware_t::parallel_mi_cu)
        .def_rw("mem_bw_per_wg_coefficients", &hardware_t::mem_bw_per_wg_coefficients)
        .def_rw("NUM_XCD", &hardware_t::NUM_XCD)
        .def_rw("debug_info", &hardware_t::debug_info);

    m.def("get_hardware_for_device",
          &hardware_t::get_hardware_for_device,
          "This gets a hardware object for a device.");

    m.def("datatype_to_bits", &origami::data_type_to_bits, "Return the number of bits in a datatype");
    m.def("string_to_datatype", &origami::string_to_data_type, "Convert a string representation of a datatype into data_type_t enum");
    m.def("select_best_macro_tile_size",
          &origami::select_best_macro_tile_size,
          "M"_a,
          "N"_a,
          "K"_a,
          "batch"_a,
          "transA"_a,
          "transB"_a,
          "hardware"_a,
          "MT_list"_a,
          "element_size_A"_a,
          "element_size_B"_a,
          "element_size_out"_a,
          "mi_datatype"_a,
          "mx_block_size"_a,
          "H_L2"_a,
          "print"_a,
          "WGM"_a,
          "max_cus"_a = 0,
          "Get best macro tile sizes.");
    m.def("select_reduction", &origami::streamk::select_reduction, "Select best StreamK reduction strategy");
    m.def("select_grid", &origami::streamk::select_grid,
          "x"_a,
          "y"_a,
          "z"_a,
          "batch"_a,
          "trans_a"_a,
          "trans_b"_a,
          "element_size_A"_a,
          "element_size_B"_a,
          "element_size_out"_a,
          "mi_datatype"_a,
          "workspace_size"_a,
          "mt_m"_a,
          "mt_n"_a,
          "mt_k"_a,
          "mi_m"_a,
          "mi_n"_a,
          "mi_k"_a,
          "workgroup_mapping"_a,
          "workspace_size_per_elem_c"_a,
          "occupancy"_a,
          "analytical_hardware"_a,
          "dynamic_grid_version"_a,
          "reduction_strategy"_a,
          "max_cus"_a = 0,
          "Select Best StreamK Grid Size");
    m.def("compute_total_latency", &origami::compute_total_latency,
          "hardware"_a,
          "M"_a,
          "N"_a,
          "K"_a,
          "batch"_a,
          "transA"_a,
          "transB"_a,
          "MT_M"_a,
          "MT_N"_a,
          "MT_K"_a,
          "MI_M"_a,
          "MI_N"_a,
          "MI_K"_a,
          "element_size_A"_a,
          "element_size_B"_a,
          "element_size_out"_a,
          "mi_datatype"_a,
          "mx_block_size"_a,
          "WGM"_a,
          "non_temporal_a"_a = 0,
          "non_temporal_b"_a = 0,
          "occupancy"_a = 1,
          "split"_a = 0,
          "max_cus"_a = 0,
          "Compute the total latency of a gemm");
    m.def("select_best_wgm", &origami::select_best_wgm, "Get best workgroup mapping.");
}
