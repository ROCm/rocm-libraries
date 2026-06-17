// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <cstring>
#include <hip/hip_runtime.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

class DeviceBuffer
{
private:
    void* _devicePtr = nullptr;
    size_t _sizeBytes = 0;

public:
    DeviceBuffer(size_t sizeBytes)
        : _sizeBytes(sizeBytes)
    {
        if(_sizeBytes > 0)
        {
            const auto status = hipMalloc(&_devicePtr, _sizeBytes);
            if(status != hipSuccess)
            {
                throw std::runtime_error("Failed to allocate device memory: "
                                         + std::string(hipGetErrorString(status)));
            }
        }
    }

    ~DeviceBuffer()
    {
        if(_devicePtr != nullptr)
        {
            (void)hipFree(_devicePtr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : _devicePtr(other._devicePtr)
        , _sizeBytes(other._sizeBytes)
    {
        other._devicePtr = nullptr;
        other._sizeBytes = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if(this != &other)
        {
            if(_devicePtr != nullptr)
            {
                (void)hipFree(_devicePtr);
            }
            _devicePtr = other._devicePtr;
            _sizeBytes = other._sizeBytes;
            other._devicePtr = nullptr;
            other._sizeBytes = 0;
        }
        return *this;
    }

    void copyFromHost(const void* hostPtr)
    {
        if(_devicePtr == nullptr || hostPtr == nullptr)
        {
            throw std::runtime_error("Invalid pointers for copy operation");
        }
        const auto status = hipMemcpy(_devicePtr, hostPtr, _sizeBytes, hipMemcpyHostToDevice);
        if(status != hipSuccess)
        {
            throw std::runtime_error("Failed to copy from host: "
                                     + std::string(hipGetErrorString(status)));
        }
    }

    void copyToHost(void* hostPtr)
    {
        if(_devicePtr == nullptr || hostPtr == nullptr)
        {
            throw std::runtime_error("Invalid pointers for copy operation");
        }
        const auto status = hipMemcpy(hostPtr, _devicePtr, _sizeBytes, hipMemcpyDeviceToHost);
        if(status != hipSuccess)
        {
            throw std::runtime_error("Failed to copy to host: "
                                     + std::string(hipGetErrorString(status)));
        }
    }

    void* ptr()
    {
        return _devicePtr;
    }
    size_t size() const
    {
        return _sizeBytes;
    }

    // Fill with zeros
    void zeros()
    {
        if(_devicePtr != nullptr)
        {
            const auto status = hipMemset(_devicePtr, 0, _sizeBytes);
            if(status != hipSuccess)
            {
                throw std::runtime_error("Failed to zero memory: "
                                         + std::string(hipGetErrorString(status)));
            }
        }
    }
};

void memoryBindings(nb::module_& m)
{
    nb::class_<DeviceBuffer>(m, "DeviceBuffer")
        .def(nb::init<size_t>(),
             nb::arg("size_bytes"),
             "Create a device buffer with the given size in bytes")
        .def(
            "copy_from_host",
            [](DeviceBuffer& self, const nb::bytes& data) {
                if(data.size() != self.size())
                {
                    throw std::runtime_error("Data size (" + std::to_string(data.size())
                                             + ") doesn't match buffer size ("
                                             + std::to_string(self.size()) + ")");
                }
                self.copyFromHost(data.c_str());
            },
            nb::arg("data"),
            "Copy data from bytes to device memory")
        .def(
            "copy_to_host",
            [](DeviceBuffer& self) -> nb::bytes {
                std::vector<uint8_t> hostData(self.size());
                self.copyToHost(hostData.data());
                return nb::bytes(reinterpret_cast<const char*>(hostData.data()), hostData.size());
            },
            "Copy data from device memory to bytes")
        .def("zeros", &DeviceBuffer::zeros, "Fill the buffer with zeros")
        .def("size", &DeviceBuffer::size, "Get the size of the buffer in bytes")
        .def(
            "ptr",
            [](DeviceBuffer& self) { return reinterpret_cast<uintptr_t>(self.ptr()); },
            "Get the device pointer as an integer")
        .def("__repr__", [](const DeviceBuffer& buf) {
            return "<hipdnn_frontend.DeviceBuffer size=" + std::to_string(buf.size()) + " bytes>";
        });

    // Utility function to get element size for different data types
    m.def(
        "get_dtype_size",
        [](const nb::object& dtype) -> size_t {
            const std::string dtypeStr = nb::str(dtype).c_str();

            if(dtypeStr == "<f4" || dtypeStr == "float32")
            {
                return sizeof(float);
            }
            if(dtypeStr == "<f2" || dtypeStr == "float16")
            {
                return sizeof(uint16_t);
            }
            if(dtypeStr == "<f8" || dtypeStr == "float64")
            {
                return sizeof(double);
            }
            if(dtypeStr == "<i4" || dtypeStr == "int32")
            {
                return sizeof(int32_t);
            }
            if(dtypeStr == "<u1" || dtypeStr == "uint8")
            {
                return sizeof(uint8_t);
            }
            if(dtypeStr == "<i1" || dtypeStr == "int8")
            {
                return sizeof(int8_t);
            }
            throw std::runtime_error("Unsupported dtype: " + dtypeStr);
        },
        nb::arg("dtype"),
        "Get the size in bytes of a numpy dtype");
}
