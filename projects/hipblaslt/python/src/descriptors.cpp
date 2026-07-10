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
}
