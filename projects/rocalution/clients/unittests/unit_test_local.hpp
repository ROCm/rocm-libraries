/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

// LocalMatrix and LocalVector from and to the host references, for the tests
// of the Local* objects and of the solvers on them

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "unit_test_common.hpp"

#include "base/local_matrix.hpp"
#include "base/local_vector.hpp"

namespace rocalution_unit
{
    // Built and converted to f on the host, then moved to b
    template <typename T>
    std::unique_ptr<rocalution::LocalMatrix<T>>
        make_local_matrix(Backend b, const Csr<T>& ref, Format f = {rocalution::CSR, 1})
    {
        auto A = std::make_unique<rocalution::LocalMatrix<T>>();
        A->AllocateCSR("A", ref.nnz(), ref.n, ref.m);
        A->CopyFromCSR(ref.row_offset.data(), ref.col.data(), ref.val.data());
        A->ConvertTo(f.id, f.blockdim);
        if(b == Backend::Accelerator)
        {
            A->MoveToAccelerator();
        }
        return A;
    }

    // Through a host array, as std::vector<bool> has no data()
    template <typename V>
    void fill_local_vector(rocalution::LocalVector<V>& v, const std::vector<V>& values)
    {
        std::unique_ptr<V[]> data(new V[values.size()]);
        std::copy(values.begin(), values.end(), data.get());

        v.Allocate("v", static_cast<int64_t>(values.size()));
        v.CopyFromHostData(data.get());
    }

    // Moved to b, then allocated and filled
    template <typename V>
    std::unique_ptr<rocalution::LocalVector<V>> make_local_vector(Backend               b,
                                                                  const std::vector<V>& values)
    {
        auto v = std::make_unique<rocalution::LocalVector<V>>();
        if(b == Backend::Accelerator)
        {
            v->MoveToAccelerator();
        }
        fill_local_vector(*v, values);
        return v;
    }

    // On the backend of like. V is T when values is a braced list.
    template <typename T, typename V = T>
    std::unique_ptr<rocalution::LocalVector<V>>
        make_local_vector(const rocalution::BaseRocalution<T>& like, const std::vector<V>& values)
    {
        auto v = std::make_unique<rocalution::LocalVector<V>>();
        v->CloneBackend(like);
        fill_local_vector(*v, values);
        return v;
    }

    template <typename V>
    std::vector<V> values_of(const rocalution::LocalVector<V>& v)
    {
        std::unique_ptr<V[]> data(new V[v.GetSize()]);
        v.CopyToHostData(data.get());
        return std::vector<V>(data.get(), data.get() + v.GetSize());
    }

    // Host copy in CSR, with the explicit zeros that the conversion keeps
    template <typename T>
    Csr<T> csr_of(const rocalution::LocalMatrix<T>& A)
    {
        rocalution::LocalMatrix<T> host;
        host.CloneFrom(A);
        host.MoveToHost();
        host.ConvertToCSR();

        Csr<T> C(static_cast<int>(host.GetM()), static_cast<int>(host.GetN()));
        C.row_offset.resize(C.n + 1);
        C.col.resize(host.GetNnz());
        C.val.resize(host.GetNnz());
        if(host.GetNnz() > 0)
        {
            host.CopyToCSR(C.row_offset.data(), C.col.data(), C.val.data());
        }
        else
        {
            std::fill(C.row_offset.begin(), C.row_offset.end(), 0);
        }
        return C;
    }

    // LocalMatrix and LocalVector keep the query of their backend protected
    template <typename Object>
    struct BackendProbe : Object
    {
        static bool on_accelerator(const Object& x)
        {
            return (x.*&BackendProbe::is_accel_)();
        }
    };

    template <typename Object>
    bool on_accelerator(const Object& x)
    {
        return BackendProbe<Object>::on_accelerator(x);
    }

    template <typename Object>
    std::string backend_of(const Object& x)
    {
        return backend_name(on_accelerator(x) ? Backend::Accelerator : Backend::Host);
    }
}
