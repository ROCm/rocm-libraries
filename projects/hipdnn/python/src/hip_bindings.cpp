// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <cstdint>
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

// Host-released device-side stall: the work stream waits on a host-writable
// signal value; the host releases by writing the value from a private control
// stream. Uses hipStreamWaitValue32/hipStreamWriteValue32 on hipMallocSignalMemory
// so the bindings stay CXX-only (no device-code compilation).
class HipStallGate
{
private:
    uint32_t* _signal = nullptr;
    hipStream_t _control = nullptr;

    // A destroyed gate has null signal/control; arm/release would otherwise
    // issue stream-wait/write ops on a null signal pointer (UB-adjacent).
    void throwIfDestroyed() const
    {
        if(_signal == nullptr || _control == nullptr)
        {
            throw std::runtime_error("HIP stall gate has been destroyed");
        }
    }

public:
    HipStallGate()
    {
        if(!canUseStreamWaitValue())
        {
            throw std::runtime_error("hipStreamWaitValue32 unsupported on this device");
        }
        // Signal memory is an 8-byte HSA signal; a smaller size is rejected with
        // hipErrorInvalidValue. The 32-bit wait/write ops act on its low word.
        throwOnHipError(hipExtMallocWithFlags(reinterpret_cast<void**>(&_signal),
                                              sizeof(uint64_t),
                                              hipMallocSignalMemory),
                        "hipExtMallocWithFlags");
        // Any failure after the allocation above must free what was already
        // acquired: a throwing constructor does not run the destructor, so
        // _signal (and _control, once created) would otherwise leak.
        try
        {
            // Non-blocking so the release write runs concurrently with a stalled
            // work stream; a blocking control stream would implicitly serialize
            // with the legacy default stream and deadlock when the gate stalls it.
            throwOnHipError(hipStreamCreateWithFlags(&_control, hipStreamNonBlocking),
                            "hipStreamCreateWithFlags");
            throwOnHipError(hipStreamWriteValue32(_control, _signal, 0U, 0),
                            "hipStreamWriteValue32");
            throwOnHipError(hipStreamSynchronize(_control), "hipStreamSynchronize");
        }
        catch(...)
        {
            destroy();
            throw;
        }
    }

    ~HipStallGate()
    {
        destroy();
    }

    HipStallGate(const HipStallGate&) = delete;
    HipStallGate& operator=(const HipStallGate&) = delete;

    HipStallGate(HipStallGate&& other) noexcept
        : _signal(other._signal)
        , _control(other._control)
    {
        other._signal = nullptr;
        other._control = nullptr;
    }

    HipStallGate& operator=(HipStallGate&& other) noexcept
    {
        if(this != &other)
        {
            destroy();
            _signal = other._signal;
            _control = other._control;
            other._signal = nullptr;
            other._control = nullptr;
        }
        return *this;
    }

    // Reset the signal, then enqueue a wait packet on the work stream that blocks
    // all later work on that stream until the host releases the gate.
    void arm(uintptr_t stream)
    {
        throwIfDestroyed();
        throwOnHipError(hipStreamWriteValue32(_control, _signal, 0U, 0), "hipStreamWriteValue32");
        throwOnHipError(hipStreamSynchronize(_control), "hipStreamSynchronize");
        throwOnHipError(hipStreamWaitValue32(
                            toHipStream(stream), _signal, 1U, hipStreamWaitValueGte, 0xFFFFFFFFU),
                        "hipStreamWaitValue32");
    }

    // Release the gate from the (otherwise idle) control stream; the work stream
    // proceeds device-side, no host sync needed.
    void release()
    {
        throwIfDestroyed();
        throwOnHipError(hipStreamWriteValue32(_control, _signal, 1U, 0), "hipStreamWriteValue32");
    }

    void destroy() noexcept
    {
        if(_control != nullptr)
        {
            (void)hipStreamDestroy(_control);
            _control = nullptr;
        }
        if(_signal != nullptr)
        {
            (void)hipFree(_signal);
            _signal = nullptr;
        }
    }
};

} // namespace

// NOTE: HipEvent, HipStallGate, and the hip_* stream/device helpers are HIP
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

    nb::class_<HipStallGate>(m, "HipStallGate")
        .def(nb::init<>(), "Create a host-released device-side stall gate")
        .def("arm",
             &HipStallGate::arm,
             nb::arg("stream") = 0,
             nb::call_guard<nb::gil_scoped_release>(),
             "Stall a HIP stream pointer encoded as an integer until release() is called")
        .def("release",
             &HipStallGate::release,
             "Release the gate so stalled work on the stream proceeds")
        .def("destroy", &HipStallGate::destroy, "Destroy the stall gate");

    m.def("hip_device_synchronize",
          &deviceSynchronize,
          nb::call_guard<nb::gil_scoped_release>(),
          "Block until all work on the current device has completed");
    m.def("hip_can_use_stream_wait_value",
          &canUseStreamWaitValue,
          "Return whether the current device supports hipStreamWaitValue32");
}
