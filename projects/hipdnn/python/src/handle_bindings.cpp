// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_frontend.hpp>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <stdexcept>

namespace nb = nanobind;
using namespace hipdnn_frontend;

class HandleWrapper
{
private:
    HipdnnHandlePtr _handle;

public:
    HandleWrapper()
        : _handle(createHipdnnHandle())
    {
    }

    explicit HandleWrapper(uintptr_t streamPtr)
        : _handle(createHipdnnHandle(reinterpret_cast<hipStream_t>(streamPtr)))
    {
    }

    hipdnnHandle_t get() const
    {
        return _handle.get();
    }

    void setStream(uintptr_t streamPtr)
    {
        auto error = setHipdnnHandleStream(_handle, reinterpret_cast<hipStream_t>(streamPtr));
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to set stream on hipdnn handle: "
                                     + error.get_message());
        }
    }

    uintptr_t getStream() const
    {
        hipStream_t stream = nullptr;
        auto error = getHipdnnHandleStream(_handle, &stream);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to get stream from hipdnn handle: "
                                     + error.get_message());
        }
        return reinterpret_cast<uintptr_t>(stream);
    }
};

void handle_bindings(nb::module_& m)
{
    nb::class_<HandleWrapper>(m, "Handle")
        .def(nb::init<>(), "Create a new hipdnn handle")
        .def(nb::init<uintptr_t>(), nb::arg("stream"), "Create a handle with a stream")
        .def(
            "get",
            [](const HandleWrapper& h) { return reinterpret_cast<uintptr_t>(h.get()); },
            "Get the handle pointer as an integer")
        .def("set_stream",
             &HandleWrapper::setStream,
             nb::arg("stream"),
             "Set the HIP stream (as integer pointer)")
        .def("get_stream", &HandleWrapper::getStream, "Get the HIP stream (as integer pointer)")
        .def("__repr__", [](const HandleWrapper& h) {
            return "<hipdnn_frontend.Handle at "
                   + std::to_string(reinterpret_cast<uintptr_t>(h.get())) + ">";
        });

    // Convenience functions to create handles
    m.def(
        "create_handle",
        []() { return std::make_shared<HandleWrapper>(); },
        "Create a new hipdnn handle");
    m.def(
        "create_handle",
        [](uintptr_t stream) { return std::make_shared<HandleWrapper>(stream); },
        nb::arg("stream"),
        "Create a new hipdnn handle with a stream");
}
