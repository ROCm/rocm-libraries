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

#include "host_stencil_laplace2d.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../stencil_types.hpp"
#include "host_vector.hpp"

#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution
{

    // LCOV_EXCL_START
    template <typename ValueType>
    HostStencilLaplace2D<ValueType>::HostStencilLaplace2D()
    {

        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostStencilLaplace2D<ValueType>::HostStencilLaplace2D(
        const Rocalution_Backend_Descriptor& local_backend)
    {

        log_debug(
            this, "HostStencilLaplace2D::HostStencilLaplace2D()", "constructor with local_backend");

        this->set_backend(local_backend);

        this->ndim_ = 2;
    }

    template <typename ValueType>
    HostStencilLaplace2D<ValueType>::~HostStencilLaplace2D()
    {

        log_debug(this, "HostStencilLaplace2D::~HostStencilLaplace2D()", "destructor");
    }

    template <typename ValueType>
    void HostStencilLaplace2D<ValueType>::Info(void) const
    {

        LOG_INFO("Stencil 2D Laplace (Host) size=" << this->size_ << " dim=" << this->GetNDim());
    }

    template <typename ValueType>
    int64_t HostStencilLaplace2D<ValueType>::GetNnz(void) const
    {

        return 5;
    }

    template <typename ValueType>
    void HostStencilLaplace2D<ValueType>::Apply(const BaseVector<ValueType>& in,
                                                BaseVector<ValueType>*       out) const
    {

        if((this->ndim_ > 0) && (this->size_ > 0))
        {

            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            int nrow = this->GetM();
            assert(in.GetSize() == nrow);
            assert(out->GetSize() == nrow);
            assert(out->GetSize() == in.GetSize());

            const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
            HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            _set_omp_backend_threads(this->local_backend_, nrow);

            int idx = 0;

// interior
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 1; i < this->size_ - 1; ++i)
                for(int j = 1; j < this->size_ - 1; ++j)
                {
                    idx = i * this->size_ + j;

                    cast_out->vec_[idx]
                        = static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_] // i-1
                          + static_cast<ValueType>(-1) * cast_in->vec_[idx - 1] // j-1
                          + static_cast<ValueType>(4) * cast_in->vec_[idx] // i,j
                          + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1] // j+1
                          + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_]; // i+1
                }

                // boundary layers

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int j = 1; j < this->size_ - 1; ++j)
            {
                idx = 0 * this->size_ + j;

                cast_out->vec_[idx]
                    = static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                      + static_cast<ValueType>(4) * cast_in->vec_[idx]
                      + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1]
                      + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];

                idx = (this->size_ - 1) * this->size_ + j;

                cast_out->vec_[idx] = static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                                      + static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                                      + static_cast<ValueType>(4) * cast_in->vec_[idx]
                                      + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1];
            }

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 1; i < this->size_ - 1; ++i)
            {
                idx = i * this->size_ + 0;

                cast_out->vec_[idx]
                    = static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                      + static_cast<ValueType>(4) * cast_in->vec_[idx]
                      + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1]
                      + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];

                idx = i * this->size_ + this->size_ - 1;

                cast_out->vec_[idx]
                    = static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                      + static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                      + static_cast<ValueType>(4) * cast_in->vec_[idx]
                      + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];
            }

            // boundary points

            idx                 = 0 * (this->size_) + 0;
            cast_out->vec_[idx] = static_cast<ValueType>(4) * cast_in->vec_[idx]
                                  + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1]
                                  + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];

            idx                 = 0 * (this->size_) + this->size_ - 1;
            cast_out->vec_[idx] = static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                                  + static_cast<ValueType>(4) * cast_in->vec_[idx]
                                  + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];

            idx                 = (this->size_ - 1) * (this->size_) + 0;
            cast_out->vec_[idx] = static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                                  + static_cast<ValueType>(4) * cast_in->vec_[idx]
                                  + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1];

            idx                 = (this->size_ - 1) * (this->size_) + this->size_ - 1;
            cast_out->vec_[idx] = static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                                  + static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                                  + static_cast<ValueType>(4) * cast_in->vec_[idx];
        }
    }

    template <typename ValueType>
    void HostStencilLaplace2D<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
                                                   ValueType                    scalar,
                                                   BaseVector<ValueType>*       out) const
    {

        if((this->ndim_ > 0) && (this->size_ > 0))
        {

            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            int nrow = this->GetM();
            assert(in.GetSize() == nrow);
            assert(out->GetSize() == nrow);
            assert(out->GetSize() == in.GetSize());

            const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
            HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            _set_omp_backend_threads(this->local_backend_, nrow);

            int idx = 0;

// interior
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 1; i < this->size_ - 1; ++i)
                for(int j = 1; j < this->size_ - 1; ++j)
                {
                    idx = i * this->size_ + j;

                    cast_out->vec_[idx]
                        += static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_] // i-1
                           + static_cast<ValueType>(-1) * cast_in->vec_[idx - 1] // j-1
                           + static_cast<ValueType>(4) * cast_in->vec_[idx] // i,j
                           + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1] // j+1
                           + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_]; // i+1
                }

                // boundary layers

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int j = 1; j < this->size_ - 1; ++j)
            {
                idx = 0 * this->size_ + j;

                cast_out->vec_[idx]
                    += static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                       + static_cast<ValueType>(4) * cast_in->vec_[idx]
                       + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1]
                       + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];

                idx = (this->size_ - 1) * this->size_ + j;

                cast_out->vec_[idx] += static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                                       + static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                                       + static_cast<ValueType>(4) * cast_in->vec_[idx]
                                       + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1];
            }

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 1; i < this->size_ - 1; ++i)
            {
                idx = i * this->size_ + 0;

                cast_out->vec_[idx]
                    += static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                       + static_cast<ValueType>(4) * cast_in->vec_[idx]
                       + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1]
                       + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];

                idx = i * this->size_ + this->size_ - 1;

                cast_out->vec_[idx]
                    += static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                       + static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                       + static_cast<ValueType>(4) * cast_in->vec_[idx]
                       + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];
            }

            // boundary points

            idx = 0 * (this->size_) + 0;
            cast_out->vec_[idx] += static_cast<ValueType>(4) * cast_in->vec_[idx]
                                   + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1]
                                   + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];

            idx = 0 * (this->size_) + this->size_ - 1;
            cast_out->vec_[idx] += static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                                   + static_cast<ValueType>(4) * cast_in->vec_[idx]
                                   + static_cast<ValueType>(-1) * cast_in->vec_[idx + this->size_];

            idx = (this->size_ - 1) * (this->size_) + 0;
            cast_out->vec_[idx] += static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                                   + static_cast<ValueType>(4) * cast_in->vec_[idx]
                                   + static_cast<ValueType>(-1) * cast_in->vec_[idx + 1];

            idx = (this->size_ - 1) * (this->size_) + this->size_ - 1;
            cast_out->vec_[idx] += static_cast<ValueType>(-1) * cast_in->vec_[idx - this->size_]
                                   + static_cast<ValueType>(-1) * cast_in->vec_[idx - 1]
                                   + static_cast<ValueType>(4) * cast_in->vec_[idx];
        }
    }

    template class HostStencilLaplace2D<double>;
    template class HostStencilLaplace2D<float>;
#ifdef SUPPORT_COMPLEX
    template class HostStencilLaplace2D<std::complex<double>>;
    template class HostStencilLaplace2D<std::complex<float>>;
#endif

} // namespace rocalution
