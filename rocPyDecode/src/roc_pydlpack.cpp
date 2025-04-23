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

//#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
#include "roc_pydlpack.h"

 

DLPackPyTensor::DLPackPyTensor() noexcept : m_tensor{} {
}

DLPackPyTensor::DLPackPyTensor(DLManagedTensor &&managedTensor) : m_tensor{std::move(managedTensor)} {
    managedTensor = {};
}

DLPackPyTensor::DLPackPyTensor(const DLTensor &tensor) : DLPackPyTensor(DLManagedTensor{tensor}) {
}

DLPackPyTensor::DLPackPyTensor(const py::buffer_info &info, const DLDevice &dev) : m_tensor{} {
    DLTensor &dlTensor = m_tensor.dl_tensor;
    dlTensor.data      = info.ptr;
    //TBD dtype
    dlTensor.dtype.code = kDLInt;
    dlTensor.dtype.bits = 8;
    dlTensor.dtype.lanes = 1;
    dlTensor.ndim        = info.ndim;
    dlTensor.device      = dev;
    dlTensor.byte_offset = 0;

    m_tensor.deleter = [](DLManagedTensor *self) {
        delete[] self->dl_tensor.shape;
        self->dl_tensor.shape = nullptr;
        delete[] self->dl_tensor.strides;
        self->dl_tensor.strides = nullptr;
    };

    try {
        dlTensor.shape = new int64_t[info.ndim];
        std::copy_n(info.shape.begin(), info.shape.size(), dlTensor.shape);

        dlTensor.strides = new int64_t[info.ndim];
        for (int i = 0; i < info.ndim; ++i) {
            if (info.strides[i] % info.itemsize != 0) {
                throw std::runtime_error("Stride must be a multiple of the element size in bytes");
            }

            dlTensor.strides[i] = info.strides[i] / info.itemsize;
        }
    } catch (...) {
        m_tensor.deleter(&m_tensor);
        throw;
    }
}

DLPackPyTensor::DLPackPyTensor(DLPackPyTensor &&that) noexcept : m_tensor{std::move(that.m_tensor)} {
    that.m_tensor = {};
}

DLPackPyTensor::~DLPackPyTensor() {
    if (m_tensor.deleter) {
        m_tensor.deleter(&m_tensor);
    }
}

DLPackPyTensor &DLPackPyTensor::operator=(DLPackPyTensor &&that) noexcept {
    if (this != &that) {
        if (m_tensor.deleter) {
            m_tensor.deleter(&m_tensor);
        }
        m_tensor = std::move(that.m_tensor);
        that.m_tensor = {};
    }
    return *this;
}

const DLTensor *DLPackPyTensor::operator->() const {
    return &m_tensor.dl_tensor;
}

DLTensor *DLPackPyTensor::operator->() {
    return &m_tensor.dl_tensor;
}

const DLTensor &DLPackPyTensor::operator*() const {
    return m_tensor.dl_tensor;
}

DLTensor &DLPackPyTensor::operator*() {
    return m_tensor.dl_tensor;
}
