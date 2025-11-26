// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <memory>
#include <nanobind/nanobind.h>
#include <stdexcept>

namespace nb = nanobind;
using namespace hipdnn_frontend;

class HandleWrapper
{
private:
    hipdnnHandle_t _handle;
    bool _ownsHandle;

public:
    HandleWrapper()
        : _handle(nullptr)
        , _ownsHandle(true)
    {
        auto backend = hipdnnBackend();
        auto status = backend->create(&_handle);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to create hipdnn handle");
        }
    }

    ~HandleWrapper()
    {
        if(_ownsHandle && _handle)
        {
            auto backend = hipdnnBackend();
            backend->destroy(_handle);
        }
    }

    // Disable copy
    HandleWrapper(const HandleWrapper&) = delete;
    HandleWrapper& operator=(const HandleWrapper&) = delete;

    // Enable move
    HandleWrapper(HandleWrapper&& other) noexcept
        : _handle(other._handle)
        , _ownsHandle(other._ownsHandle)
    {
        other._handle = nullptr;
        other._ownsHandle = false;
    }

    HandleWrapper& operator=(HandleWrapper&& other) noexcept
    {
        if(this != &other)
        {
            if(_ownsHandle && _handle)
            {
                auto backend = hipdnnBackend();
                backend->destroy(_handle);
            }
            _handle = other._handle;
            _ownsHandle = other._ownsHandle;
            other._handle = nullptr;
            other._ownsHandle = false;
        }
        return *this;
    }

    hipdnnHandle_t get() const
    {
        return _handle;
    }
};

void handle_bindings(nb::module_& m)
{
    nb::class_<HandleWrapper>(m, "Handle")
        .def(nb::init<>(), "Create a new hipdnn handle")
        .def(
            "get",
            [](const HandleWrapper& h) { return reinterpret_cast<uintptr_t>(h.get()); },
            "Get the handle pointer as an integer")
        .def("__repr__", [](const HandleWrapper& h) {
            return "<hipdnn_frontend.Handle at "
                   + std::to_string(reinterpret_cast<uintptr_t>(h.get())) + ">";
        });

    // Add a convenience function to create handles
    m.def(
        "create_handle",
        []() { return std::make_shared<HandleWrapper>(); },
        "Create a new hipdnn handle");
}
