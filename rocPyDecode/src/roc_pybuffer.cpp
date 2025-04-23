/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "roc_pybuffer.h"
#include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
using namespace py::literals;

static void CheckValidBuffer(const void *ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("NULL buffer not accepted");
    }
}

BufferInterface::BufferInterface(DLPackPyTensor &&dlTensor) {
    if (dlTensor->data != nullptr) {
        CheckValidBuffer(dlTensor->data);
    }
    m_dlTensor = std::move(dlTensor);
}

py::tuple BufferInterface::shape() const {
    py::tuple shape(m_dlTensor->ndim);
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] = m_dlTensor->shape[i];
    }
    return shape;
}

py::tuple BufferInterface::strides() const {
    py::tuple strides(m_dlTensor->ndim);

    for (size_t i = 0; i < strides.size(); ++i) {
        strides[i] = m_dlTensor->strides[i];
    }
    return strides;
}

std::string BufferInterface::dtype() const {
    if (m_dlTensor->dtype.bits == 8)
        return std::string("|u1");
    else if (m_dlTensor->dtype.bits == 16)
        return std::string("|u2");
    return std::string("|u1"); // non-void function must ret value
}

void *BufferInterface::data() const {
    return m_dlTensor->data;
}

py::capsule BufferInterface::dlpack(py::object stream) const {
    
    struct ManagerCtx {
        DLManagedTensor tensor;
        std::shared_ptr<const BufferInterface> extBuffer;
    };

    auto ctx = std::make_unique<ManagerCtx>();

    // Set up tensor deleter to delete the ManagerCtx
    ctx->tensor.manager_ctx = ctx.get();
    ctx->tensor.deleter = [](DLManagedTensor *tensor) {
        auto *ctx = static_cast<ManagerCtx *>(tensor->manager_ctx);
        delete ctx;
    };

    // Copy tensor data
    ctx->tensor.dl_tensor = *m_dlTensor;

    // Manager context holds a reference to this External Buffer so that
    // GC doesn't delete this buffer while the dlpack tensor still refers to it.
    ctx->extBuffer = this->shared_from_this();

    // Creates the python capsule with the DLManagedTensor instance we're returning.
    py::capsule cap(&ctx->tensor, "dltensor", [](PyObject *ptr) {
                        if(PyCapsule_IsValid(ptr, "dltensor")) {
                            // If consumer didn't delete the tensor,
                            if(auto *dlTensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(ptr, "dltensor"))) {
                                // Delete the tensor.
                                if(dlTensor->deleter != nullptr) {
                                    dlTensor->deleter(dlTensor);
                                }
                            }
                        } });

    // Now that the capsule is created and the manager ctx was transfered to it,
    // we can release the unique_ptr.
    ctx.release();

    return cap;
}

py::tuple BufferInterface::dlpackDevice() const {
    return py::make_tuple(py::int_(static_cast<int>(m_dlTensor->device.device_type)),
                          py::int_(static_cast<int>(m_dlTensor->device.device_id)));
}

const DLTensor &BufferInterface::dlTensor() const {
    return *m_dlTensor;
}

void BufferInterface::Export(py::module &m) {
    py::class_<BufferInterface, std::shared_ptr<BufferInterface>>(m, "BufferInterface", py::dynamic_attr())
        .def_property_readonly("shape", &BufferInterface::shape, "Get the shape of the buffer as an array")
        .def_property_readonly("strides", &BufferInterface::strides, "Get the strides of the buffer")
        .def_property_readonly("dtype", &BufferInterface::dtype, "Get the data type of the buffer")
        .def("__dlpack__", &BufferInterface::dlpack, "stream"_a=1, "Export the buffer as a DLPack tensor")
        .def("__dlpack_device__", &BufferInterface::dlpackDevice, "Get the device associated with the buffer");
}

void PyExportInitializer(py::module& m) {  
    BufferInterface::Export(m);
}

int BufferInterface::LoadDLPack(std::vector<size_t>& _shape, std::vector<size_t>& _stride, uint32_t bit_depth, std::string& _type_str, void* _data, int device_id_) {
    m_dlTensor->byte_offset = 0;
    m_dlTensor->device.device_type = kDLROCM;   // TODO: infer the device type from the memory buffer
    m_dlTensor->device.device_id = device_id_;

    // Convert data
    void* ptr = _data;
    CheckValidBuffer(ptr);
    m_dlTensor->data = ptr;

    // Convert DataType
    if (_type_str != "|u1" && _type_str != "|u2") {  // TODO: can also be other letters
        throw std::runtime_error("Could not create DL Pack tensor! Invalid typstr: " + _type_str);
        return -1;
    }

    int itemSizeDT;

    m_dlTensor->dtype.code = kDLUInt;

    if (bit_depth == 8) {
        m_dlTensor->dtype.bits = 8;
        itemSizeDT = sizeof(uint8_t);
    } else if (bit_depth == 10) {
        m_dlTensor->dtype.bits = 16;
        itemSizeDT = sizeof(uint16_t);
    }
    m_dlTensor->dtype.lanes = 1;

    // Convert ndim
    m_dlTensor->ndim = _shape.size();

    // Convert shape
    m_dlTensor->shape = new int64_t[m_dlTensor->ndim];
    for (int i = 0; i < m_dlTensor->ndim; ++i) {
        m_dlTensor->shape[i] = _shape[i];
    }
    
    // Convert strides
    int strides_dim = _stride.size();
    m_dlTensor->strides = new int64_t[strides_dim];
    for (int i = 0; i < strides_dim; ++i) {
        m_dlTensor->strides[i] = _stride[i];
        if (m_dlTensor->strides[i] % itemSizeDT != 0) {
            throw std::runtime_error("Stride must be a multiple of the element size in bytes");
            return -1;
        }
        m_dlTensor->strides[i] /= itemSizeDT;
    }
    return 0;
}
