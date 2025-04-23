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

#ifndef USE_DLPACK_HEADER
#define USE_DLPACK_HEADER

#include <pybind11/buffer_info.h>
#include <dlpack/dlpack.h>

namespace py = pybind11;

class DLPackPyTensor final {
    public:
        DLPackPyTensor() noexcept;
        explicit DLPackPyTensor(const DLTensor &tensor);
        explicit DLPackPyTensor(DLManagedTensor &&tensor);
        explicit DLPackPyTensor(const py::buffer_info &info, const DLDevice &dev);

        DLPackPyTensor(DLPackPyTensor &&that) noexcept;
        ~DLPackPyTensor();

        DLPackPyTensor &operator=(DLPackPyTensor &&that) noexcept;

        const DLTensor *operator->() const;
        DLTensor       *operator->();

        const DLTensor &operator*() const;
        DLTensor       &operator*();

    private:
        DLManagedTensor m_tensor;
};
 
#endif // USE_DLPACK_HEADER
