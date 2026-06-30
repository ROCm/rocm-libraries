/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "base_stencil.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "backend_manager.hpp"
#include "base_vector.hpp"

#include <complex>
#include <stdlib.h>

namespace rocalution
{

    template <typename ValueType>
    BaseStencil<ValueType>::BaseStencil()
    {
        log_debug(this, "BaseStencil::BaseStencil()");

        this->ndim_ = 0;
        this->size_ = 0;
    }

    template <typename ValueType>
    BaseStencil<ValueType>::~BaseStencil()
    {
        log_debug(this, "BaseStencil::~BaseStencil()");
    }

    template <typename ValueType>
    int BaseStencil<ValueType>::GetM(void) const
    {
        int dim = 1;

        if(this->GetNDim() > 0)
        {
            for(int i = 0; i < ndim_; ++i)
            {
                dim *= this->size_;
            }
        }

        return dim;
    }

    template <typename ValueType>
    int BaseStencil<ValueType>::GetN(void) const
    {
        return this->GetM();
    }

    template <typename ValueType>
    int BaseStencil<ValueType>::GetNDim(void) const
    {
        return this->ndim_;
    }

    template <typename ValueType>
    void BaseStencil<ValueType>::set_backend(const Rocalution_Backend_Descriptor& local_backend)
    {
        this->local_backend_ = local_backend;
    }

    template <typename ValueType>
    void BaseStencil<ValueType>::SetGrid(int size)
    {
        assert(size >= 0);
        this->size_ = size;
    }

    template <typename ValueType>
    HostStencil<ValueType>::HostStencil()
    {
    }

    template <typename ValueType>
    HostStencil<ValueType>::~HostStencil()
    {
    }

    template <typename ValueType>
    AcceleratorStencil<ValueType>::AcceleratorStencil()
    {
    }

    template <typename ValueType>
    AcceleratorStencil<ValueType>::~AcceleratorStencil()
    {
    }

    template <typename ValueType>
    HIPAcceleratorStencil<ValueType>::HIPAcceleratorStencil()
    {
    }

    template <typename ValueType>
    HIPAcceleratorStencil<ValueType>::~HIPAcceleratorStencil()
    {
    }

    template class BaseStencil<double>;
    template class BaseStencil<float>;
#ifdef SUPPORT_COMPLEX
    template class BaseStencil<std::complex<double>>;
    template class BaseStencil<std::complex<float>>;
#endif
    template class BaseStencil<int>;

    template class HostStencil<double>;
    template class HostStencil<float>;
#ifdef SUPPORT_COMPLEX
    template class HostStencil<std::complex<double>>;
    template class HostStencil<std::complex<float>>;
#endif
    template class HostStencil<int>;

    template class AcceleratorStencil<double>;
    template class AcceleratorStencil<float>;
#ifdef SUPPORT_COMPLEX
    template class AcceleratorStencil<std::complex<double>>;
    template class AcceleratorStencil<std::complex<float>>;
#endif
    template class AcceleratorStencil<int>;

    template class HIPAcceleratorStencil<double>;
    template class HIPAcceleratorStencil<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorStencil<std::complex<double>>;
    template class HIPAcceleratorStencil<std::complex<float>>;
#endif
    template class HIPAcceleratorStencil<int>;

} // namespace rocalution
