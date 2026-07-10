// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "device_array.hpp"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "status.hpp"
#include "init.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

DeviceArray::DeviceArray(size_t nbytes, hipDataType dtype,
                         std::vector<int64_t> shape, std::string host_dtype)
    : nbytes_(nbytes), dtype_(dtype), shape_(std::move(shape)),
      host_dtype_(std::move(host_dtype))
{
    HIP_CHECK(hipMalloc(&d_, nbytes_));
}

DeviceArray::~DeviceArray() { _free_nothrow(); }

void DeviceArray::_free_nothrow() noexcept
{
    if(d_) { hipFree(d_); d_ = nullptr; }  // best-effort; must not throw in destructor
}

void DeviceArray::free()
{
    if(d_) { HIP_CHECK(hipFree(d_)); d_ = nullptr; }
}

void DeviceArray::copy_from_host(const void* src, size_t nbytes)
{
    if(nbytes != nbytes_)
        throw HipblasLtError("copy_from_host size mismatch");
    HIP_CHECK(hipMemcpy(d_, src, nbytes, hipMemcpyHostToDevice));
}

void DeviceArray::copy_to_host(void* dst, size_t nbytes) const
{
    if(nbytes != nbytes_)
        throw HipblasLtError("copy_to_host size mismatch");
    HIP_CHECK(hipMemcpy(dst, d_, nbytes, hipMemcpyDeviceToHost));
}

using NpArray = nb::ndarray<nb::numpy, nb::c_contig>;

void init_device_array(nb::module_& m)
{
    nb::class_<DeviceArray>(m, "DeviceArray")
        .def_static("from_numpy", [](NpArray arr, hipDataType dtype) {
            std::vector<int64_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
            size_t nbytes = arr.nbytes();
            // host_dtype is empty string; to_numpy() in Python reconstructs it via _DTYPE_TO_NP.
            auto da = std::make_unique<DeviceArray>(nbytes, dtype, shape, "");
            da->copy_from_host(arr.data(), nbytes);
            return da.release();
        }, nb::arg("arr"), nb::arg("dtype"), nb::rv_policy::take_ownership)
        .def("copy_from_host", [](DeviceArray& self, NpArray arr) {
            self.copy_from_host(arr.data(), arr.nbytes());
        }, nb::arg("arr"))
        .def("copy_to_host", [](DeviceArray& self, NpArray out) {
            self.copy_to_host(out.data(), out.nbytes());
        }, nb::arg("out"))
        .def("free", &DeviceArray::free)
        .def_prop_ro("ptr", &DeviceArray::ptr)
        .def_prop_ro("nbytes", &DeviceArray::nbytes)
        .def_prop_ro("shape", [](DeviceArray& self) -> nb::object {
            nb::list lst;
            for(auto v : self.shape())
                lst.append(v);
            return nb::tuple(lst);
        })
        .def_prop_ro("dtype", &DeviceArray::dtype)
        .def("__enter__", [](DeviceArray& self) -> DeviceArray& { return self; },
             nb::rv_policy::reference_internal)
        .def("__exit__", [](DeviceArray& self, nb::object, nb::object, nb::object) {
            self.free(); return false;
        })
        .def_static("_alloc", [](size_t nbytes, hipDataType dtype, std::vector<int64_t> shape) {
            // Allocate device memory without initialising it — used by from_dlpack and
            // other callers that will fill the buffer themselves.
            auto da = std::make_unique<DeviceArray>(nbytes, dtype, shape, "");
            return da.release();
        }, nb::arg("nbytes"), nb::arg("dtype"), nb::arg("shape"),
           nb::rv_policy::take_ownership);
}
