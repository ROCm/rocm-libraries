/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "local_stencil.hpp"
#include "../utils/def.hpp"
#include "host/host_stencil_laplace2d.hpp"
#include "host/host_vector.hpp"
#include "local_vector.hpp"
#include "stencil_types.hpp"

#include "../utils/log.hpp"

#include <complex>

namespace rocalution
{

    // LCOV_EXCL_START
    template <typename ValueType>
    LocalStencil<ValueType>::LocalStencil()
    {
        log_debug(this, "LocalStencil::LocalStencil()");

        this->object_name_ = "";

        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    LocalStencil<ValueType>::~LocalStencil()
    {
        log_debug(this, "LocalStencil::~LocalStencil()");

        delete this->stencil_;
    }

    template <typename ValueType>
    LocalStencil<ValueType>::LocalStencil(unsigned int type)
    {
        log_debug(this, "LocalStencil::LocalStencil()", type);

        assert(type == Laplace2D); // the only one at the moment

        this->object_name_ = _stencil_type_names[type];

        this->stencil_host_ = new HostStencilLaplace2D<ValueType>(this->local_backend_);
        this->stencil_      = this->stencil_host_;
    }

    template <typename ValueType>
    int64_t LocalStencil<ValueType>::GetNDim(void) const
    {
        return this->stencil_->GetNDim();
    }

    template <typename ValueType>
    int64_t LocalStencil<ValueType>::GetNnz(void) const
    {
        return this->stencil_->GetNnz();
    }

    template <typename ValueType>
    int64_t LocalStencil<ValueType>::GetM(void) const
    {
        return this->stencil_->GetM();
    }

    template <typename ValueType>
    int64_t LocalStencil<ValueType>::GetN(void) const
    {
        return this->stencil_->GetN();
    }

    template <typename ValueType>
    void LocalStencil<ValueType>::Info(void) const
    {
        this->stencil_->Info();
    }

    template <typename ValueType>
    void LocalStencil<ValueType>::Clear(void)
    {
        log_debug(this, "LocalStencil::Clear()");

        this->stencil_->SetGrid(0);
    }

    template <typename ValueType>
    void LocalStencil<ValueType>::SetGrid(int size)
    {
        log_debug(this, "LocalStencil::SetGrid()", size);

        assert(size >= 0);

        this->stencil_->SetGrid(size);
    }

    template <typename ValueType>
    void LocalStencil<ValueType>::Apply(const LocalVector<ValueType>& in,
                                        LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalStencil::Apply()", (const void*&)in, out);

        assert(out != NULL);

        assert(((this->stencil_ == this->stencil_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->stencil_ == this->stencil_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

        this->stencil_->Apply(*in.vector_, out->vector_);
    }

    template <typename ValueType>
    void LocalStencil<ValueType>::ApplyAdd(const LocalVector<ValueType>& in,
                                           ValueType                     scalar,
                                           LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalStencil::ApplyAdd()", (const void*&)in, scalar, out);

        assert(out != NULL);

        assert(((this->stencil_ == this->stencil_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->stencil_ == this->stencil_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

        this->stencil_->Apply(*in.vector_, out->vector_);
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    void LocalStencil<ValueType>::MoveToAccelerator(void)
    {
        LOG_INFO("The function is not implemented (yet)!");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void LocalStencil<ValueType>::MoveToHost(void)
    {
        LOG_INFO("The function is not implemented (yet)!");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template class LocalStencil<double>;
    template class LocalStencil<float>;
#ifdef SUPPORT_COMPLEX
    template class LocalStencil<std::complex<double>>;
    template class LocalStencil<std::complex<float>>;
#endif

} // namespace rocalution
