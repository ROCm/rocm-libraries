// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
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

bool canUseStreamWaitValue()
{
    int dev = 0;
    throwOnHipError(hipGetDevice(&dev), "hipGetDevice");
    int value = 0;
    throwOnHipError(hipDeviceGetAttribute(&value, hipDeviceAttributeCanUseStreamWaitValue, dev),
                    "hipDeviceGetAttribute");
    return value != 0;
}

void deviceSynchronize()
{
    throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

using StallGate = hipdnn_data_sdk::utilities::StallGate;

std::unique_ptr<StallGate> createStallGate()
{
    auto gate = std::make_unique<StallGate>();
    if(gate->isUsable())
    {
        return gate;
    }

    // hipSuccess means no HIP call failed, so the device simply lacks support.
    if(gate->lastError() == hipSuccess)
    {
        throw std::runtime_error("hipStreamWaitValue32 unsupported on this device");
    }
    throwOnHipError(gate->lastError(), gate->lastOperation());
    return nullptr;
}

void armStallGate(StallGate& gate, uintptr_t stream)
{
    if(gate.arm(toHipStream(stream)))
    {
        return;
    }

    throwOnHipError(gate.lastError(), gate.lastOperation());
    // Reached only when no HIP call failed, so an earlier watchdog timeout
    // disabled stalling for this shared object. Raising is the only way the caller
    // can tell that the stream is unstalled and the next span includes host time.
    throw std::runtime_error("HIP stall gate is disabled after a stall watchdog timeout");
}

} // namespace

// NOTE: HipEvent, StallGate, and the hip_* stream/device helpers are HIP
// primitives, not hipDNN concepts. They are exposed through the hipDNN frontend
// bindings only provisionally; treat them as an internal, unstable surface and
// avoid depending on them.
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

    nb::class_<StallGate>(m, "HipStallGate")
        .def(nb::new_(&createStallGate), "Create a host-released device-side stall gate")
        .def("arm",
             &armStallGate,
             nb::arg("stream") = 0,
             nb::call_guard<nb::gil_scoped_release>(),
             "Stall a HIP stream pointer encoded as an integer until release() is called")
        .def("release", &StallGate::release, "Release the gate so stalled work proceeds")
        .def("timed_out",
             &StallGate::timedOut,
             "Return whether the stall watchdog, not release(), ended the last arm()");

    m.def("hip_device_synchronize",
          &deviceSynchronize,
          nb::call_guard<nb::gil_scoped_release>(),
          "Block until all work on the current device has completed");
    m.def("hip_can_use_stream_wait_value",
          &canUseStreamWaitValue,
          "Return whether the current device supports hipStreamWaitValue32");
}
