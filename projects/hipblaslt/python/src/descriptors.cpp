// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include "status.hpp"
#include "init.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

namespace {

class Handle
{
public:
    Handle() { HIPBLASLT_CHECK(hipblasLtCreate(&h_)); }
    ~Handle() { close(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    void close()
    {
        if(h_)
        {
            hipblasLtDestroy(h_);  // best-effort in destructor path
            h_ = nullptr;
        }
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(h_); }
    hipblasLtHandle_t raw() const { return h_; }

private:
    hipblasLtHandle_t h_ = nullptr;
};

class MatrixLayout
{
public:
    MatrixLayout(hipDataType dtype, uint64_t rows, uint64_t cols, int64_t ld)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&l_, dtype, rows, cols, ld));
    }
    ~MatrixLayout() { if(l_) hipblasLtMatrixLayoutDestroy(l_); }
    MatrixLayout(const MatrixLayout&) = delete;
    MatrixLayout& operator=(const MatrixLayout&) = delete;

    void set_attribute(hipblasLtMatrixLayoutAttribute_t attr, int32_t value)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(l_, attr, &value, sizeof(value)));
    }
    std::uintptr_t ptr() const { return reinterpret_cast<std::uintptr_t>(l_); }
    hipblasLtMatrixLayout_t raw() const { return l_; }

private:
    hipblasLtMatrixLayout_t l_ = nullptr;
};

} // namespace

void init_descriptors(nb::module_& m)
{
    nb::class_<Handle>(m, "Handle")
        .def(nb::init<>())
        .def("close", &Handle::close)
        .def_prop_ro("ptr", &Handle::ptr)
        .def("__enter__", [](Handle& self) -> Handle& { return self; },
             nb::rv_policy::reference_internal)
        .def("__exit__", [](Handle& self, nb::object, nb::object, nb::object) {
            self.close();
            return false;
        });

    nb::class_<MatrixLayout>(m, "MatrixLayout")
        .def(nb::init<hipDataType, uint64_t, uint64_t, int64_t>(),
             nb::arg("dtype"), nb::arg("rows"), nb::arg("cols"), nb::arg("ld"))
        .def("set_attribute", &MatrixLayout::set_attribute,
             nb::arg("attr"), nb::arg("value"))
        .def_prop_ro("ptr", &MatrixLayout::ptr);
}
