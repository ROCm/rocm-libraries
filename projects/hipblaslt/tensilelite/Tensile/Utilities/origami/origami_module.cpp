#include <Tensile/analytical/Hardware.hpp>
#include <Tensile/analytical/StreamK.hpp>
#include <Tensile/analytical/Utils.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Origami = TensileLite::analytical;
using Hardware    = TensileLite::analytical::Hardware;

PYBIND11_MODULE(origami, m)
{
    pybind11::enum_<Hardware::Architecture>(m, "Architecture")
        .value("gfx942", Hardware::Architecture::gfx942)
        .value("gfx950", Hardware::Architecture::gfx950)
        .export_values();

    pybind11::enum_<Origami::DataType>(m, "DataType")
        .value("Float", Origami::DataType::Float)
        .value("ComplexFloat", Origami::DataType::ComplexFloat)
        .value("ComplexDouble", Origami::DataType::ComplexDouble)
        .value("Double", Origami::DataType::Double)
        .value("Half", Origami::DataType::Half)
        .value("Int8x4", Origami::DataType::Int8x4)
        .value("Int32", Origami::DataType::Int32)
        .value("BFloat16", Origami::DataType::BFloat16)
        .value("Int8", Origami::DataType::Int8)
        .value("Int64", Origami::DataType::Int64)
        .value("XFloat32", Origami::DataType::XFloat32)
        .value("Float8_fnuz", Origami::DataType::Float8_fnuz)
        .value("BFloat8_fnuz", Origami::DataType::BFloat8_fnuz)
        .value("Float8BFloat8_fnuz", Origami::DataType::Float8BFloat8_fnuz)
        .value("BFloat8Float8_fnuz", Origami::DataType::BFloat8Float8_fnuz)
        .value("Float8", Origami::DataType::Float8)
        .value("BFloat8", Origami::DataType::BFloat8)
        .value("Float8BFloat8", Origami::DataType::Float8BFloat8)
        .value("BFloat8Float8", Origami::DataType::BFloat8Float8)
        .value("Float6", Origami::DataType::Float6)
        .value("BFloat6", Origami::DataType::BFloat6)
        .value("Float4", Origami::DataType::Float4)
        .export_values();

    m.def("intToDataType",
          &Origami::intToDataType,
          "Convert int to DataType.");

    pybind11::class_<Hardware>(m, "Hardware")
        .def(pybind11::init<Hardware::Architecture,
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
        .def_readwrite("N_CU", &Hardware::N_CU)
        .def_readwrite("LDS_capacity", &Hardware::LDS_capacity)
        .def_readwrite("mem1_perf_ratio", &Hardware::mem1_perf_ratio)
        .def_readwrite("mem2_perf_ratio", &Hardware::mem2_perf_ratio)
        .def_readwrite("mem3_perf_ratio", &Hardware::mem3_perf_ratio)
        .def_readwrite("L2_capacity", &Hardware::L2_capacity)
        .def_readwrite("CU_per_L2", &Hardware::CU_per_L2)
        .def_readwrite("compute_clock_ghz", &Hardware::compute_clock_ghz)
        .def_readwrite("parallel_MI_CU", &Hardware::parallel_MI_CU)
        .def_readwrite("percent_bw_per_wg", &Hardware::percent_bw_per_wg)
        .def_readwrite("NUM_XCD", &Hardware::NUM_XCD);

    m.def("getHardwareForDevice",
          &Hardware::getHardwareForDevice,
          "This gets a hardware object for a device.");

    m.def("datatype_to_bits", &Origami::dataTypeToBits, "Return the number of bits in a datatype");
    m.def("string_to_datatype", &Origami::stringToDatatype, "Convert a string representation of a datatype into DataType enum");
    m.def("select_best_macro_tile_size",
          &Origami::select_best_macro_tile_size,
          "Get best macro tile sizes.");
    m.def("select_streamk_grid", &Origami::streamk::select_streamk_grid, "Select Best StreamK Grid Size");
    m.def("compute_total_latency", &Origami::compute_total_latency, "compute_total_latency");
    m.def("select_best_wgm", &Origami::select_best_wgm, "Get best workgroup mapping.");
}