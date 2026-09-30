// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>
#include <memory>
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
        // HIP can report success with a garbage duration (e.g. an event never
        // recorded on a stream that never ran). Zero is a legitimate back-to-back
        // measurement, so only non-finite or negative values are rejected.
        if(!std::isfinite(milliseconds) || milliseconds < 0.0F)
        {
            throw std::runtime_error("hipEventElapsedTime returned a non-finite or negative "
                                     "elapsed time");
        }
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

void armStallGate(StallGate& gate, uintptr_t stream)
{
    if(gate.arm(toHipStream(stream)))
    {
        return;
    }

    throwOnHipError(gate.lastError(), gate.lastOperation());
    throw std::runtime_error(
        std::string("HIP stall gate declined arm: ")
        + (gate.lastOperation() != nullptr ? gate.lastOperation() : "unknown operation"));
}

// Binding-local owner for a StallGate. StallGate is non-movable, so the wrapper
// owns it indirectly and can detach that ownership before releasing the GIL.
//
// close()/__exit__ are the deterministic teardown path. The plain destructor is
// the GC fallback. StallGate destruction joins its watchdog thread and frees
// signal memory with hipFree, which can synchronize the device. Teardown
// therefore releases the GIL.
class PyStallGate
{
public:
    PyStallGate()
        : _gate(std::make_unique<StallGate>())
    {
        if(_gate->isUsable())
        {
            return;
        }

        // hipSuccess means no HIP call failed, so the device simply lacks support.
        const auto lastError = _gate->lastError();
        const auto* lastOperation = _gate->lastOperation();
        _gate.reset();
        if(lastError == hipSuccess)
        {
            throw std::runtime_error("hipStreamWaitValue32 unsupported on this device");
        }
        throwOnHipError(lastError, lastOperation);
    }

    ~PyStallGate()
    {
        closeGate();
    }

    PyStallGate(const PyStallGate&) = delete;
    PyStallGate& operator=(const PyStallGate&) = delete;
    PyStallGate(PyStallGate&&) = delete;
    PyStallGate& operator=(PyStallGate&&) = delete;

    void arm(uintptr_t stream)
    {
        armStallGate(checkOpen(), stream);
    }

    void release()
    {
        checkOpen().release();
    }

    bool timedOut()
    {
        return checkOpen().timedOut();
    }

    // Idempotent: a second close() is a no-op, matching StallGate::release()'s own
    // idempotence and file.close()'s in the standard library.
    void close()
    {
        closeGate();
    }

    PyStallGate& enter()
    {
        checkOpen();
        return *this;
    }

    void exit(const nb::object& /*excType*/,
              const nb::object& /*excValue*/,
              const nb::object& /*traceback*/)
    {
        closeGate();
    }

private:
    StallGate& checkOpen()
    {
        if(_gate == nullptr)
        {
            throw std::runtime_error("HipStallGate is closed");
        }
        return *_gate;
    }

    void closeGate() noexcept
    {
        if(_gate == nullptr)
        {
            return;
        }

        // Detach while Python still serializes access to this wrapper, then destroy
        // the gate without the GIL. Both explicit close() and nanobind deallocation
        // enter here with the GIL held.
        auto gate = std::move(_gate);
        const nb::gil_scoped_release release;
        gate.reset();
    }

    std::unique_ptr<StallGate> _gate;
};

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

    nb::class_<PyStallGate>(m, "HipStallGate")
        .def(nb::init<>(), "Create a host-released device-side stall gate")
        .def("arm",
             &PyStallGate::arm,
             nb::arg("stream") = 0,
             "Stall a HIP stream pointer encoded as an integer until release() is called.\n"
             "Before re-arming, synchronize the previously armed stream so its wait "
             "packet has retired; release() alone does not guarantee retirement. "
             "The stream must remain valid until the measurement has finished.")
        .def("release", &PyStallGate::release, "Release the gate so stalled work proceeds")
        .def("timed_out",
             &PyStallGate::timedOut,
             "Return whether the stall watchdog, not release(), ended the last arm()")
        .def("close",
             &PyStallGate::close,
             "Idempotently release pending waits, join the watchdog, and free its signal. "
             "Freeing the signal can synchronize the device. "
             "Every other method raises RuntimeError once closed.")
        .def("__enter__", &PyStallGate::enter, nb::rv_policy::reference_internal)
        .def("__exit__",
             &PyStallGate::exit,
             nb::arg("exc_type").none(),
             nb::arg("exc_value").none(),
             nb::arg("traceback").none());

    m.def("hip_device_synchronize",
          &deviceSynchronize,
          nb::call_guard<nb::gil_scoped_release>(),
          "Block until all work on the current device has completed");
    m.def("hip_can_use_stream_wait_value",
          &canUseStreamWaitValue,
          "Return whether the current device supports hipStreamWaitValue32");
}
