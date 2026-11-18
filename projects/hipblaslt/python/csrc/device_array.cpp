// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// DeviceArray, with every host transfer validated here rather than in Python, so calling
// _core directly gets the same checks as the hipblaslt wrapper.
#include "device_array.hpp"
#include "init.hpp"
#include "status.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <memory>
#include <optional>
#include <string>

namespace nb = nanobind;

namespace hipblaslt_py
{

    DeviceArray::DeviceArray(size_t nbytes, hipDataType dtype, std::vector<int64_t> shape)
        : nbytes_(nbytes)
        , dtype_(dtype)
        , shape_(std::move(shape))
    {
        if(nbytes_ > 0)
            HIP_CHECK(hipMalloc(&d_, nbytes_));
    }

    DeviceArray::~DeviceArray()
    {
        free_nothrow();
    }

    void DeviceArray::free_nothrow() noexcept
    {
        if(d_)
        {
            hipFree(d_);
            d_ = nullptr;
        }
    }

    void DeviceArray::free()
    {
        if(d_)
        {
            void* p = d_;
            d_      = nullptr;
            HIP_CHECK(hipFree(p));
        }
    }

    void* DeviceArray::raw() const
    {
        if(!d_ && nbytes_ > 0)
            throw HipblasLtError("this DeviceArray has been freed");
        return d_;
    }

    void DeviceArray::copy_from_host(const void* src, size_t nbytes)
    {
        if(nbytes != nbytes_)
            throw nb::value_error(("copy_from_host: the host array has " + std::to_string(nbytes)
                                   + " bytes, and this DeviceArray has " + std::to_string(nbytes_))
                                      .c_str());
        if(nbytes)
            HIP_CHECK(hipMemcpy(raw(), src, nbytes, hipMemcpyHostToDevice));
    }

    void DeviceArray::copy_to_host(void* dst, size_t nbytes) const
    {
        if(nbytes != nbytes_)
            throw nb::value_error(("copy_to_host: the host array has " + std::to_string(nbytes)
                                   + " bytes, and this DeviceArray has " + std::to_string(nbytes_))
                                      .c_str());
        if(nbytes)
            HIP_CHECK(hipMemcpy(dst, raw(), nbytes, hipMemcpyDeviceToHost));
    }

} // namespace hipblaslt_py

namespace
{
    using namespace hipblaslt_py;

    // The NumPy dtype name each supported hipDataType holds, and its size. ml_dtypes names
    // (bfloat16, float8_*) arrive as a same-width unsigned view plus the name, since nanobind
    // only accepts NumPy's own scalar types.
    struct HostType
    {
        const char* name;
        size_t      itemsize;
    };

    const std::map<hipDataType, HostType>& host_types()
    {
        static const std::map<hipDataType, HostType> t = {
            {HIP_R_32F, {"float32", 4}},
            {HIP_R_64F, {"float64", 8}},
            {HIP_R_16F, {"float16", 2}},
            {HIP_R_16BF, {"bfloat16", 2}},
            {HIP_R_8I, {"int8", 1}},
            {HIP_R_32I, {"int32", 4}},
            {HIP_R_8F_E4M3, {"float8_e4m3fn", 1}},
            {HIP_R_8F_E5M2, {"float8_e5m2", 1}},
            {HIP_R_8F_E4M3_FNUZ, {"float8_e4m3fnuz", 1}},
            {HIP_R_8F_E5M2_FNUZ, {"float8_e5m2fnuz", 1}},
        };
        return t;
    }

    // An R_8I buffer also takes uint8, for opaque byte buffers such as workspaces.
    bool also_accepted(hipDataType dtype, const std::string& name)
    {
        return dtype == HIP_R_8I && name == "uint8";
    }

    std::string numpy_name(const nb::dlpack::dtype& d)
    {
        using nb::dlpack::dtype_code;
        std::string base;
        switch(static_cast<dtype_code>(d.code))
        {
        case dtype_code::Float:
            base = "float";
            break;
        case dtype_code::Int:
            base = "int";
            break;
        case dtype_code::UInt:
            base = "uint";
            break;
        case dtype_code::Bool:
            return "bool";
        default:
            return "other";
        }
        return base + std::to_string(d.bits);
    }

    using HostArray = nb::ndarray<nb::numpy, nb::c_contig, nb::device::cpu>;

    std::string type_name(hipDataType dtype)
    {
        return nb::cast<std::string>(nb::str(nb::cast(dtype)));
    }

    // Check a host array against a DeviceArray's type, before any HIP call.
    void check_host_array(const HostArray&                  arr,
                          hipDataType                       dtype,
                          const std::optional<std::string>& host_dtype,
                          const char*                       what)
    {
        auto it = host_types().find(dtype);
        if(it == host_types().end())
            throw nb::value_error(
                (std::string(what) + ": DeviceArray doesn't support " + type_name(dtype) + " yet")
                    .c_str());
        const HostType& want = it->second;
        std::string     got  = numpy_name(arr.dtype());
        if(host_dtype)
        {
            // An ml_dtypes array, passed as its same-width unsigned view.
            std::string wire = "uint" + std::to_string(want.itemsize * 8);
            if(got != wire)
                throw nb::value_error((std::string(what) + ": host_dtype is given, so the array "
                                       + "must be its " + wire + " view, not " + got)
                                          .c_str());
            got = *host_dtype;
        }
        if(got != want.name && !also_accepted(dtype, got))
            throw nb::value_error((std::string(what) + ": a " + got + " array can't go into a "
                                   + type_name(dtype) + " DeviceArray, which holds " + want.name)
                                      .c_str());
    }

    // Contiguity is enforced by the binding (nb::c_contig with noconvert), so a strided array
    // raises a TypeError instead of being copied silently.
} // namespace

void init_device_array(nb::module_& m)
{
    nb::class_<DeviceArray>(m, "DeviceArray", "Device memory holding one hipBLASLt data type.")
        .def_static(
            "from_numpy",
            [](HostArray arr, hipDataType dtype, std::optional<std::string> host_dtype) {
                check_host_array(arr, dtype, host_dtype, "from_numpy");
                std::vector<int64_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
                auto                 da = std::make_unique<DeviceArray>(arr.nbytes(), dtype, shape);
                da->copy_from_host(arr.data(), arr.nbytes());
                return da.release();
            },
            nb::arg("arr").noconvert(),
            nb::arg("dtype"),
            nb::arg("host_dtype") = nb::none(),
            nb::rv_policy::take_ownership,
            "Copy a C-contiguous NumPy array to the device. Its dtype must match `dtype`. For an "
            "ml_dtypes array, pass its same-width unsigned view and its dtype name as `host_dtype` "
            "(hipblaslt.from_numpy does this).")
        .def(
            "copy_from_host",
            [](DeviceArray& self, HostArray arr, std::optional<std::string> host_dtype) {
                check_host_array(arr, self.dtype(), host_dtype, "copy_from_host");
                self.copy_from_host(arr.data(), arr.nbytes());
            },
            nb::arg("arr").noconvert(),
            nb::arg("host_dtype") = nb::none())
        .def(
            "copy_to_host",
            [](DeviceArray& self, HostArray out, std::optional<std::string> host_dtype) {
                check_host_array(out, self.dtype(), host_dtype, "copy_to_host");
                self.copy_to_host(out.data(), out.nbytes());
            },
            nb::arg("out").noconvert(),
            nb::arg("host_dtype") = nb::none())
        .def("free", &DeviceArray::free, "Release the device memory now.")
        .def_prop_ro("freed", &DeviceArray::freed)
        .def_prop_ro("ptr",
                     [](const DeviceArray& a) { return reinterpret_cast<uintptr_t>(a.raw()); })
        .def_prop_ro("nbytes", &DeviceArray::nbytes)
        .def_prop_ro("dtype", &DeviceArray::dtype)
        .def_prop_ro("shape",
                     [](const DeviceArray& a) {
                         nb::list out;
                         for(auto v : a.shape())
                             out.append(v);
                         return nb::tuple(out);
                     })
        .def(
            "__enter__",
            [](DeviceArray& self) -> DeviceArray& { return self; },
            nb::rv_policy::reference)
        .def("__exit__", [](DeviceArray& self, nb::args) { self.free(); });

    m.def(
        "_host_types",
        []() {
            std::map<int, std::pair<std::string, size_t>> out;
            for(const auto& [dtype, t] : host_types())
                out[static_cast<int>(dtype)] = {t.name, t.itemsize};
            return out;
        },
        "{DataType value: (NumPy dtype name, itemsize)} for the types DeviceArray holds.");
}
