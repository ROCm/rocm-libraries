// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

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

    void checkNotDestroyed() const
    {
        if(!_handle)
        {
            throw std::runtime_error("Handle has been destroyed");
        }
    }

public:
    HandleWrapper()
    {
        const auto error = createHipdnnHandle(_handle);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to create hipdnn handle: " + error.get_message());
        }
    }

    explicit HandleWrapper(uintptr_t streamPtr)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto error = createHipdnnHandle(_handle, reinterpret_cast<hipStream_t>(streamPtr));
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to create hipdnn handle: " + error.get_message());
        }
    }

    hipdnnHandle_t get() const
    {
        checkNotDestroyed();
        return *_handle;
    }

    bool isValid() const
    {
        return _handle != nullptr;
    }

    void destroy()
    {
        _handle.reset();
    }

    void setStream(uintptr_t streamPtr)
    {
        checkNotDestroyed();
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto error = setHipdnnHandleStream(_handle, reinterpret_cast<hipStream_t>(streamPtr));
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to set stream on hipdnn handle: "
                                     + error.get_message());
        }
    }

    uintptr_t getStream() const
    {
        checkNotDestroyed();
        hipStream_t stream = nullptr;
        const auto error = getHipdnnHandleStream(_handle, &stream);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to get stream from hipdnn handle: "
                                     + error.get_message());
        }
        return reinterpret_cast<uintptr_t>(stream);
    }
};

void handleBindings(nb::module_& m)
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
        .def("__int__", [](const HandleWrapper& h) { return reinterpret_cast<uintptr_t>(h.get()); })
        .def("__index__",
             [](const HandleWrapper& h) { return reinterpret_cast<uintptr_t>(h.get()); })
        .def("__repr__", [](const HandleWrapper& h) {
            if(!h.isValid())
            {
                return std::string("<hipdnn_frontend.Handle (destroyed)>");
            }
            return "<hipdnn_frontend.Handle at "
                   + std::to_string(reinterpret_cast<uintptr_t>(h.get())) + ">";
        });

    m.def(
        "create_handle",
        []() { return std::make_shared<HandleWrapper>(); },
        "Create a new hipdnn handle");
    m.def(
        "create_handle",
        [](uintptr_t stream) { return std::make_shared<HandleWrapper>(stream); },
        nb::arg("stream"),
        "Create a new hipdnn handle with a stream");
    m.def(
        "destroy_handle",
        [](HandleWrapper& h) { h.destroy(); },
        nb::arg("handle"),
        "Destroy a hipdnn handle. The handle object should not be used after this call.");
    m.def(
        "set_stream",
        [](HandleWrapper& h, uintptr_t stream) { h.setStream(stream); },
        nb::arg("handle"),
        nb::arg("stream"),
        "Set the HIP stream on a handle");
    m.def(
        "get_stream",
        [](const HandleWrapper& h) { return h.getStream(); },
        nb::arg("handle"),
        "Get the HIP stream from a handle");
}
