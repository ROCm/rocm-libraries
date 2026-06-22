// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <hip/hip_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace
{

void throwOnHipError(hipError_t status, const char* operation)
{
    if(status == hipSuccess)
    {
        return;
    }

    const auto* operationName = operation == nullptr ? "HIP operation" : operation;
    const auto* errorString = hipGetErrorString(status);
    if(errorString == nullptr)
    {
        errorString = "unknown HIP error";
    }

    throw std::runtime_error(std::string(operationName) + " failed: " + errorString);
}

hipStream_t toHipStream(uintptr_t stream)
{
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<hipStream_t>(stream);
}

class HipEvent
{
private:
    hipEvent_t _event = nullptr;

    hipEvent_t getChecked() const
    {
        if(_event == nullptr)
        {
            throw std::runtime_error("HIP event has been destroyed");
        }
        return _event;
    }

public:
    HipEvent()
    {
        throwOnHipError(hipEventCreate(&_event), "hipEventCreate");
    }

    ~HipEvent()
    {
        destroy();
    }

    HipEvent(const HipEvent&) = delete;
    HipEvent& operator=(const HipEvent&) = delete;

    HipEvent(HipEvent&& other) noexcept
        : _event(other._event)
    {
        other._event = nullptr;
    }

    HipEvent& operator=(HipEvent&& other) noexcept
    {
        if(this != &other)
        {
            destroy();
            _event = other._event;
            other._event = nullptr;
        }
        return *this;
    }

    void destroy() noexcept
    {
        if(_event != nullptr)
        {
            (void)hipEventDestroy(_event);
            _event = nullptr;
        }
    }

    void record(uintptr_t stream)
    {
        throwOnHipError(hipEventRecord(getChecked(), toHipStream(stream)), "hipEventRecord");
    }

    void synchronize() const
    {
        throwOnHipError(hipEventSynchronize(getChecked()), "hipEventSynchronize");
    }

    float elapsedTime(const HipEvent& stop) const
    {
        float milliseconds = 0.0F;
        throwOnHipError(hipEventElapsedTime(&milliseconds, getChecked(), stop.getChecked()),
                        "hipEventElapsedTime");
        return milliseconds;
    }

    uintptr_t ptr() const
    {
        return reinterpret_cast<uintptr_t>(getChecked());
    }
};

int getDeviceCount()
{
    int count = 0;
    const auto status = hipGetDeviceCount(&count);
    if(status == hipErrorNoDevice)
    {
        return 0;
    }
    throwOnHipError(status, "hipGetDeviceCount");
    return count;
}

void streamSynchronize(uintptr_t stream)
{
    throwOnHipError(hipStreamSynchronize(toHipStream(stream)), "hipStreamSynchronize");
}

} // namespace

void hipBindings(nb::module_& m)
{
    nb::class_<HipEvent>(m, "HipEvent")
        .def(nb::init<>(), "Create a HIP event")
        .def("record",
             &HipEvent::record,
             nb::arg("stream") = 0,
             "Record the event on a HIP stream pointer encoded as an integer")
        .def("synchronize",
             &HipEvent::synchronize,
             nb::call_guard<nb::gil_scoped_release>(),
             "Block until the event has completed")
        .def("elapsed_time",
             &HipEvent::elapsedTime,
             nb::arg("stop_event"),
             "Return elapsed time in milliseconds from this event to stop_event")
        .def("destroy", &HipEvent::destroy, "Destroy the HIP event")
        .def("ptr", &HipEvent::ptr, "Return the hipEvent_t pointer as an integer")
        .def("__int__", &HipEvent::ptr)
        .def("__index__", &HipEvent::ptr)
        .def("__repr__", [](const HipEvent& event) {
            return "<hipdnn_frontend.HipEvent at " + std::to_string(event.ptr()) + ">";
        });

    m.def("hip_stream_synchronize",
          &streamSynchronize,
          nb::arg("stream") = 0,
          nb::call_guard<nb::gil_scoped_release>(),
          "Block until a HIP stream pointer encoded as an integer is idle");
    m.def("hip_get_device_count", &getDeviceCount, "Return the number of visible HIP devices");
}
