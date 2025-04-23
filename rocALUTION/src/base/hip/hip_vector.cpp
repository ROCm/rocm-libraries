/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hip_vector.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../backend_manager.hpp"
#include "../base_vector.hpp"
#include "../host/host_vector.hpp"
#include "hip_allocate_free.hpp"
#include "hip_blas.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_utils.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#ifdef SUPPORT_COMPLEX
#include <complex>
#include <hip/hip_complex.h>
#endif

#include "hip_rand_normal.hpp"
#include "hip_rand_uniform.hpp"

namespace rocalution
{
    template <typename ValueType>
    HIPAcceleratorVector<ValueType>::HIPAcceleratorVector()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorVector<ValueType>::HIPAcceleratorVector(
        const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(
            this, "HIPAcceleratorVector::HIPAcceleratorVector()", "constructor with local_backend");

        this->vec_ = NULL;
        this->set_backend(local_backend);

        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorVector<ValueType>::~HIPAcceleratorVector()
    {
        log_debug(this, "HIPAcceleratorVector::~HIPAcceleratorVector()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorVector<ValueType>");
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Allocate(int64_t n)
    {
        assert(n >= 0);

        this->Clear();

        if(n > 0)
        {
            allocate_hip(n, &this->vec_);
            set_to_zero_hip(this->local_backend_.HIP_block_size, n, this->vec_);
        }

        this->size_ = n;

        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::SetDataPtr(ValueType** ptr, int64_t size)
    {
        assert(size >= 0);

        if(size > 0)
        {
            assert(*ptr != NULL);
        }

        hipDeviceSynchronize();

        this->vec_  = *ptr;
        this->size_ = size;
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::LeaveDataPtr(ValueType** ptr)
    {
        assert(this->size_ >= 0);

        hipDeviceSynchronize();
        *ptr       = this->vec_;
        this->vec_ = NULL;

        this->size_ = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Clear(void)
    {
        if(this->size_ > 0)
        {
            free_hip(&this->vec_);
            this->size_ = 0;
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFromHost(const HostVector<ValueType>& src)
    {
        // CPU to HIP copy
        const HostVector<ValueType>* cast_vec;
        if((cast_vec = dynamic_cast<const HostVector<ValueType>*>(&src)) != NULL)
        {
            if(this->size_ == 0)
            {
                // Allocate local structure
                this->Allocate(cast_vec->size_);
            }

            assert(cast_vec->size_ == this->size_);

            copy_h2d(this->size_, cast_vec->vec_, this->vec_);
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyToHost(HostVector<ValueType>* dst) const
    {
        // HIP to CPU copy
        HostVector<ValueType>* cast_vec;
        if((cast_vec = dynamic_cast<HostVector<ValueType>*>(dst)) != NULL)
        {
            if(cast_vec->size_ == 0)
            {
                // Allocate local vector
                cast_vec->Allocate(this->size_);
            }

            assert(cast_vec->size_ == this->size_);

            copy_d2h(this->size_, this->vec_, cast_vec->vec_);
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFromHostAsync(const HostVector<ValueType>& src)
    {
        // CPU to HIP copy
        const HostVector<ValueType>* cast_vec;
        if((cast_vec = dynamic_cast<const HostVector<ValueType>*>(&src)) != NULL)
        {
            if(this->size_ == 0)
            {
                // Allocate local vector
                this->Allocate(cast_vec->size_);
            }

            assert(cast_vec->size_ == this->size_);

            copy_h2d(this->size_,
                     cast_vec->vec_,
                     this->vec_,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyToHostAsync(HostVector<ValueType>* dst) const
    {
        // HIP to CPU copy
        HostVector<ValueType>* cast_vec;
        if((cast_vec = dynamic_cast<HostVector<ValueType>*>(dst)) != NULL)
        {
            if(cast_vec->size_ == 0)
            {
                // Allocate local vector
                cast_vec->Allocate(this->size_);
            }

            assert(cast_vec->size_ == this->size_);

            copy_d2h(this->size_,
                     this->vec_,
                     cast_vec->vec_,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType>& src)
    {
        const HIPAcceleratorVector<ValueType>* hip_cast_vec;
        const HostVector<ValueType>*           host_cast_vec;

        // HIP to HIP copy
        if((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src)) != NULL)
        {
            if(this->size_ == 0)
            {
                // Allocate local vector
                this->Allocate(hip_cast_vec->size_);
            }

            assert(hip_cast_vec->size_ == this->size_);

            if(this != hip_cast_vec)
            {
                copy_d2d(this->size_, hip_cast_vec->vec_, this->vec_);
            }
        }
        else
        {
            // HIP to CPU copy
            if((host_cast_vec = dynamic_cast<const HostVector<ValueType>*>(&src)) != NULL)
            {
                this->CopyFromHost(*host_cast_vec);
            }
            else
            {
                LOG_INFO("Error unsupported HIP vector type");
                this->Info();
                src.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFromAsync(const BaseVector<ValueType>& src)
    {
        const HIPAcceleratorVector<ValueType>* hip_cast_vec;
        const HostVector<ValueType>*           host_cast_vec;

        // HIP to HIP copy
        if((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src)) != NULL)
        {
            if(this->size_ == 0)
            {
                // Allocate local vector
                this->Allocate(hip_cast_vec->size_);
            }

            assert(hip_cast_vec->size_ == this->size_);

            if(this != hip_cast_vec)
            {
                copy_d2d(this->size_,
                         hip_cast_vec->vec_,
                         this->vec_,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }
        }
        else
        {
            // HIP to CPU copy
            if((host_cast_vec = dynamic_cast<const HostVector<ValueType>*>(&src)) != NULL)
            {
                this->CopyFromHostAsync(*host_cast_vec);
            }
            else
            {
                LOG_INFO("Error unsupported HIP vector type");
                this->Info();
                src.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType>& src,
                                                   int64_t                      src_offset,
                                                   int64_t                      dst_offset,
                                                   int64_t                      size)
    {
        assert(this->size_ > 0);
        assert(size > 0);
        assert(dst_offset + size <= this->size_);

        const HIPAcceleratorVector<ValueType>* cast_src
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src);

        assert(cast_src != NULL);
        assert(cast_src->size_ > 0);
        assert(src_offset + size <= cast_src->size_);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

        kernel_copy_offset_from<<<GridSize,
                                  BlockSize,
                                  0,
                                  HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            size, src_offset, dst_offset, cast_src->vec_, this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyTo(BaseVector<ValueType>* dst) const
    {
        HIPAcceleratorVector<ValueType>* hip_cast_vec;
        HostVector<ValueType>*           host_cast_vec;

        // HIP to HIP copy
        if((hip_cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*>(dst)) != NULL)
        {
            if(hip_cast_vec->size_ == 0)
            {
                // Allocate local vector
                hip_cast_vec->Allocate(this->size_);
            }

            assert(hip_cast_vec->size_ == this->size_);

            if(this != hip_cast_vec)
            {
                copy_d2d(this->size_, this->vec_, hip_cast_vec->vec_);
            }
        }
        else
        {
            // HIP to CPU copy
            if((host_cast_vec = dynamic_cast<HostVector<ValueType>*>(dst)) != NULL)
            {
                this->CopyToHost(host_cast_vec);
            }
            else
            {
                LOG_INFO("Error unsupported HIP vector type");
                this->Info();
                dst->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyToAsync(BaseVector<ValueType>* dst) const
    {
        HIPAcceleratorVector<ValueType>* hip_cast_vec;
        HostVector<ValueType>*           host_cast_vec;

        // HIP to HIP copy
        if((hip_cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*>(dst)) != NULL)
        {
            if(hip_cast_vec->size_ == 0)
            {
                // Allocate local vector
                hip_cast_vec->Allocate(this->size_);
            }

            assert(hip_cast_vec->size_ == this->size_);

            if(this != hip_cast_vec)
            {
                copy_d2d(this->size_, this->vec_, hip_cast_vec->vec_, true);
            }
        }
        else
        {
            // HIP to CPU copy
            if((host_cast_vec = dynamic_cast<HostVector<ValueType>*>(dst)) != NULL)
            {
                this->CopyToHostAsync(host_cast_vec);
            }
            else
            {
                LOG_INFO("Error unsupported HIP vector type");
                this->Info();
                dst->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFromFloat(const BaseVector<float>& src)
    {
        LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HIPAcceleratorVector<double>::CopyFromFloat(const BaseVector<float>& src)
    {
        const HIPAcceleratorVector<float>* hip_cast_vec;

        // HIP to HIP copy
        if((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<float>*>(&src)) != NULL)
        {
            if(this->size_ == 0)
            {
                this->Allocate(hip_cast_vec->size_);
            }

            assert(hip_cast_vec->size_ == this->size_);

            if(this->size_ > 0)
            {
                dim3 BlockSize(this->local_backend_.HIP_block_size);
                dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

                kernel_copy_from_float<<<GridSize,
                                         BlockSize,
                                         0,
                                         HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->size_, hip_cast_vec->vec_, this->vec_);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFromDouble(const BaseVector<double>& src)
    {
        LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HIPAcceleratorVector<float>::CopyFromDouble(const BaseVector<double>& src)
    {
        const HIPAcceleratorVector<double>* hip_cast_vec;

        // HIP to HIP copy
        if((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<double>*>(&src)) != NULL)
        {
            if(this->size_ == 0)
            {
                this->Allocate(hip_cast_vec->size_);
            }

            assert(hip_cast_vec->size_ == this->size_);

            if(this->size_ > 0)
            {
                dim3 BlockSize(this->local_backend_.HIP_block_size);
                dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

                kernel_copy_from_double<<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->size_, hip_cast_vec->vec_, this->vec_);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
        }
        else
        {
            LOG_INFO("Error unsupported HIP vector type");
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFromData(const ValueType* data)
    {
        copy_d2d(this->size_, data, this->vec_);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFromHostData(const ValueType* data)
    {
        copy_h2d(this->size_, data, this->vec_);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyToData(ValueType* data) const
    {
        copy_d2d(this->size_, this->vec_, data);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyToHostData(ValueType* data) const
    {
        copy_d2h(this->size_, this->vec_, data);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Zeros(void)
    {
        set_to_zero_hip(this->local_backend_.HIP_block_size, this->size_, this->vec_);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Ones(void)
    {
        set_to_one_hip(this->local_backend_.HIP_block_size, this->size_, this->vec_);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::SetValues(ValueType val)
    {
        set_to_value_hip(this->local_backend_.HIP_block_size, this->size_, this->vec_, val);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::AddScale(const BaseVector<ValueType>& x, ValueType alpha)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

            assert(cast_x != NULL);
            assert(this->size_ == cast_x->size_);

            rocblas_status status;
            status = rocblasTaxpy(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  this->size_,
                                  &alpha,
                                  cast_x->vec_,
                                  1,
                                  this->vec_,
                                  1);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
        }
    }

    template <>
    void HIPAcceleratorVector<bool>::AddScale(const BaseVector<bool>& x, bool alpha)
    {
        LOG_INFO("No bool axpy function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HIPAcceleratorVector<int>::AddScale(const BaseVector<int>& x, int alpha)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<int>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<int>*>(&x);

            assert(cast_x != NULL);
            assert(this->size_ == cast_x->size_);
            kernel_axpy<256><<<(this->size_ - 1) / 256 + 1,
                               256,
                               0,
                               HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, alpha, cast_x->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <>
    void HIPAcceleratorVector<int64_t>::AddScale(const BaseVector<int64_t>& x, int64_t alpha)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<int64_t>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<int64_t>*>(&x);

            assert(cast_x != NULL);
            assert(this->size_ == cast_x->size_);
            kernel_axpy<256><<<(this->size_ - 1) / 256 + 1,
                               256,
                               0,
                               HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, alpha, cast_x->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::ScaleAdd(ValueType alpha, const BaseVector<ValueType>& x)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

            assert(cast_x != NULL);
            assert(this->size_ == cast_x->size_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            kernel_scaleadd<<<GridSize,
                              BlockSize,
                              0,
                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, alpha, cast_x->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::ScaleAddScale(ValueType                    alpha,
                                                        const BaseVector<ValueType>& x,
                                                        ValueType                    beta)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

            assert(cast_x != NULL);
            assert(this->size_ == cast_x->size_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            kernel_scaleaddscale<<<GridSize,
                                   BlockSize,
                                   0,
                                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, alpha, beta, cast_x->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::ScaleAddScale(ValueType                    alpha,
                                                        const BaseVector<ValueType>& x,
                                                        ValueType                    beta,
                                                        int64_t                      src_offset,
                                                        int64_t                      dst_offset,
                                                        int64_t                      size)
    {
        if(this->size_ > 0)
        {
            assert(this->size_ > 0);
            assert(size > 0);
            assert(dst_offset + size <= this->size_);

            const HIPAcceleratorVector<ValueType>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

            assert(cast_x != NULL);
            assert(cast_x->size_ > 0);
            assert(src_offset + size <= cast_x->size_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

            kernel_scaleaddscale_offset<<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                size, src_offset, dst_offset, alpha, beta, cast_x->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::ScaleAdd2(ValueType                    alpha,
                                                    const BaseVector<ValueType>& x,
                                                    ValueType                    beta,
                                                    const BaseVector<ValueType>& y,
                                                    ValueType                    gamma)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);
            const HIPAcceleratorVector<ValueType>* cast_y
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&y);

            assert(cast_x != NULL);
            assert(cast_y != NULL);
            assert(this->size_ == cast_x->size_);
            assert(this->size_ == cast_y->size_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            kernel_scaleadd2<<<GridSize,
                               BlockSize,
                               0,
                               HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, alpha, beta, gamma, cast_x->vec_, cast_y->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Scale(ValueType alpha)
    {
        if(this->size_ > 0)
        {
            rocblas_status status;
            status = rocblasTscal(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  this->size_,
                                  &alpha,
                                  this->vec_,
                                  1);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
        }
    }

    template <>
    void HIPAcceleratorVector<bool>::Scale(bool alpha)
    {
        LOG_INFO("No bool rocBLAS scale function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HIPAcceleratorVector<int>::Scale(int alpha)
    {
        LOG_INFO("No int rocBLAS scale function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HIPAcceleratorVector<int64_t>::Scale(int64_t alpha)
    {
        LOG_INFO("No integral rocBLAS scale function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType HIPAcceleratorVector<ValueType>::Dot(const BaseVector<ValueType>& x) const
    {
        const HIPAcceleratorVector<ValueType>* cast_x
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        ValueType res = static_cast<ValueType>(0);

        if(this->size_ > 0)
        {
            rocblas_status status;
            status = rocblasTdotc(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  this->size_,
                                  this->vec_,
                                  1,
                                  cast_x->vec_,
                                  1,
                                  &res);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);

            // Synchronize stream to make sure, result is available on the host
            hipStreamSynchronize(HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return res;
    }

    template <>
    bool HIPAcceleratorVector<bool>::Dot(const BaseVector<bool>& x) const
    {
        LOG_INFO("No bool dot function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int HIPAcceleratorVector<int>::Dot(const BaseVector<int>& x) const
    {
        LOG_INFO("No int dot function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int64_t HIPAcceleratorVector<int64_t>::Dot(const BaseVector<int64_t>& x) const
    {
        LOG_INFO("No integral dot function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType HIPAcceleratorVector<ValueType>::DotNonConj(const BaseVector<ValueType>& x) const
    {
        const HIPAcceleratorVector<ValueType>* cast_x
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        ValueType res = static_cast<ValueType>(0);

        if(this->size_ > 0)
        {
            rocblas_status status;
            status = rocblasTdotu(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  this->size_,
                                  this->vec_,
                                  1,
                                  cast_x->vec_,
                                  1,
                                  &res);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);

            // Synchronize stream to make sure, result is available on the host
            hipStreamSynchronize(HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return res;
    }

    template <>
    bool HIPAcceleratorVector<bool>::DotNonConj(const BaseVector<bool>& x) const
    {
        LOG_INFO("No bool dotc function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int HIPAcceleratorVector<int>::DotNonConj(const BaseVector<int>& x) const
    {
        LOG_INFO("No int dotc function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int64_t HIPAcceleratorVector<int64_t>::DotNonConj(const BaseVector<int64_t>& x) const
    {
        LOG_INFO("No integral dotc function");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType HIPAcceleratorVector<ValueType>::Norm(void) const
    {
        ValueType res = static_cast<ValueType>(0);

        if(this->size_ > 0)
        {
            rocblas_status status;
            status = rocblasTnrm2(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  this->size_,
                                  this->vec_,
                                  1,
                                  &res);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);

            // Synchronize stream to make sure, result is available on the host
            hipStreamSynchronize(HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return res;
    }

    template <>
    bool HIPAcceleratorVector<bool>::Norm(void) const
    {
        LOG_INFO("What is bool HIPAcceleratorVector<ValueType>::Norm(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int HIPAcceleratorVector<int>::Norm(void) const
    {
        LOG_INFO("What is int HIPAcceleratorVector<ValueType>::Norm(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int64_t HIPAcceleratorVector<int64_t>::Norm(void) const
    {
        LOG_INFO("What is integral HIPAcceleratorVector<ValueType>::Norm(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType HIPAcceleratorVector<ValueType>::Reduce(void) const
    {
        ValueType res = static_cast<ValueType>(0);

        if(this->size_ > 0)
        {
            char*  buffer = NULL;
            size_t size   = 0;

            ValueType* dres = NULL;
            allocate_hip(1, &dres);

            rocprimTreduce(buffer,
                           size,
                           this->vec_,
                           dres,
                           this->size_,
                           HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            allocate_hip(size, &buffer);

            rocprimTreduce(buffer,
                           size,
                           this->vec_,
                           dres,
                           this->size_,
                           HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&buffer);

            copy_d2h(1, dres, &res);

            free_hip(&dres);
        }

        return res;
    }

    template <>
    bool HIPAcceleratorVector<bool>::Reduce(void) const
    {
        LOG_INFO("What is bool HIPAcceleratorVector::<ValueType>::Reduce(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType HIPAcceleratorVector<ValueType>::InclusiveSum(const BaseVector<ValueType>& vec)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&vec);

            assert(cast_vec != NULL);

            char*  buffer = NULL;
            size_t size   = 0;

            rocprimTinclusivesum(buffer,
                                 size,
                                 cast_vec->vec_,
                                 this->vec_,
                                 this->size_,
                                 HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            allocate_hip(size, &buffer);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            rocprimTinclusivesum(buffer,
                                 size,
                                 cast_vec->vec_,
                                 this->vec_,
                                 this->size_,
                                 HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&buffer);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            ValueType ret;
            copy_d2h(1, this->vec_ + this->size_ - 1, &ret);

            return ret;
        }

        return 0;
    }

    template <>
    bool HIPAcceleratorVector<bool>::InclusiveSum(const BaseVector<bool>& vec)
    {
        LOG_INFO("What is bool HIPAcceleratorVector::InclusiveSum()?");
        FATAL_ERROR(__FILE__, __LINE__);

        return false;
    }

    template <typename ValueType>
    ValueType HIPAcceleratorVector<ValueType>::ExclusiveSum(const BaseVector<ValueType>& vec)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&vec);

            assert(cast_vec != NULL);

            char*  buffer = NULL;
            size_t size   = 0;

            rocprimTexclusivesum(buffer,
                                 size,
                                 cast_vec->vec_,
                                 this->vec_,
                                 this->size_,
                                 HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            allocate_hip(size, &buffer);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            rocprimTexclusivesum(buffer,
                                 size,
                                 cast_vec->vec_,
                                 this->vec_,
                                 this->size_,
                                 HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&buffer);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            ValueType ret;
            copy_d2h(1, this->vec_ + this->size_ - 1, &ret);

            return ret;
        }

        return 0;
    }

    template <>
    bool HIPAcceleratorVector<bool>::ExclusiveSum(const BaseVector<bool>& vec)
    {
        LOG_INFO("What is bool HIPAcceleratorVector::ExclusiveSum()?");
        FATAL_ERROR(__FILE__, __LINE__);

        return false;
    }

    template <typename ValueType>
    ValueType HIPAcceleratorVector<ValueType>::Asum(void) const
    {
        ValueType res = static_cast<ValueType>(0);

        if(this->size_ > 0)
        {
            rocblas_status status;
            status = rocblasTasum(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  this->size_,
                                  this->vec_,
                                  1,
                                  &res);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);

            // Synchronize stream to make sure, result is available on the host
            hipStreamSynchronize(HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return res;
    }

    template <>
    bool HIPAcceleratorVector<bool>::Asum(void) const
    {
        LOG_INFO("Asum<bool> not implemented");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int HIPAcceleratorVector<int>::Asum(void) const
    {
        LOG_INFO("Asum<int> not implemented");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int64_t HIPAcceleratorVector<int64_t>::Asum(void) const
    {
        LOG_INFO("Asum<int64_t> not implemented");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    int64_t HIPAcceleratorVector<ValueType>::Amax(ValueType& value) const
    {
        int index = 0;
        value     = static_cast<ValueType>(0.0);

        if(this->size_ > 0)
        {
            rocblas_status status;
            status = rocblasTamax(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  this->size_,
                                  this->vec_,
                                  1,
                                  &index);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);

            copy_d2h(1, this->vec_ + index, &value);
        }

        value = std::abs(value);
        return index;
    }

    template <>
    int64_t HIPAcceleratorVector<bool>::Amax(bool& value) const
    {
        LOG_INFO("Amax<bool> not implemented");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int64_t HIPAcceleratorVector<int>::Amax(int& value) const
    {
        LOG_INFO("Amax<int> not implemented");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int64_t HIPAcceleratorVector<int64_t>::Amax(int64_t& value) const
    {
        LOG_INFO("Amax<int64_t> not implemented");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType>& x)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);

            assert(cast_x != NULL);
            assert(this->size_ == cast_x->size_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            kernel_pointwisemult<<<GridSize,
                                   BlockSize,
                                   0,
                                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, cast_x->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType>& x,
                                                        const BaseVector<ValueType>& y)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_x
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&x);
            const HIPAcceleratorVector<ValueType>* cast_y
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&y);

            assert(cast_x != NULL);
            assert(cast_y != NULL);
            assert(this->size_ == cast_x->size_);
            assert(this->size_ == cast_y->size_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            kernel_pointwisemult2<<<GridSize,
                                    BlockSize,
                                    0,
                                    HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, cast_x->vec_, cast_y->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Permute(const BaseVector<int>& permutation)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<int>* cast_perm
                = dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);

            assert(cast_perm != NULL);
            assert(this->size_ == cast_perm->size_);

            HIPAcceleratorVector<ValueType> vec_tmp(this->local_backend_);
            vec_tmp.Allocate(this->size_);
            vec_tmp.CopyFrom(*this);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            // this->vec_[ cast_perm->vec_[i] ] = vec_tmp.vec_[i];
            kernel_permute<<<GridSize,
                             BlockSize,
                             0,
                             HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, cast_perm->vec_, vec_tmp.vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::PermuteBackward(const BaseVector<int>& permutation)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<int>* cast_perm
                = dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);

            assert(cast_perm != NULL);
            assert(this->size_ == cast_perm->size_);

            HIPAcceleratorVector<ValueType> vec_tmp(this->local_backend_);
            vec_tmp.Allocate(this->size_);
            vec_tmp.CopyFrom(*this);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            //    this->vec_[i] = vec_tmp.vec_[ cast_perm->vec_[i] ];
            kernel_permute_backward<<<GridSize,
                                      BlockSize,
                                      0,
                                      HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, cast_perm->vec_, vec_tmp.vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::CopyFromPermute(const BaseVector<ValueType>& src,
                                                          const BaseVector<int>&       permutation)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src);
            const HIPAcceleratorVector<int>* cast_perm
                = dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);
            assert(cast_perm != NULL);
            assert(cast_vec != NULL);

            assert(cast_vec->size_ == this->size_);
            assert(cast_perm->size_ == this->size_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            //    this->vec_[ cast_perm->vec_[i] ] = cast_vec->vec_[i];
            kernel_permute<<<GridSize,
                             BlockSize,
                             0,
                             HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, cast_perm->vec_, cast_vec->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void
        HIPAcceleratorVector<ValueType>::CopyFromPermuteBackward(const BaseVector<ValueType>& src,
                                                                 const BaseVector<int>& permutation)
    {
        if(this->size_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&src);
            const HIPAcceleratorVector<int>* cast_perm
                = dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);
            assert(cast_perm != NULL);
            assert(cast_vec != NULL);

            assert(cast_vec->size_ == this->size_);
            assert(cast_perm->size_ == this->size_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            //    this->vec_[i] = cast_vec->vec_[ cast_perm->vec_[i] ];
            kernel_permute_backward<<<GridSize,
                                      BlockSize,
                                      0,
                                      HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, cast_perm->vec_, cast_vec->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::GetIndexValues(const BaseVector<int>& index,
                                                         BaseVector<ValueType>* values) const
    {
        assert(values != NULL);

        const HIPAcceleratorVector<int>* cast_idx
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&index);
        HIPAcceleratorVector<ValueType>* cast_vec
            = dynamic_cast<HIPAcceleratorVector<ValueType>*>(values);

        assert(cast_idx != NULL);
        assert(cast_vec != NULL);
        assert(cast_vec->size_ == cast_idx->size_);

        if(cast_idx->size_ > 0)
        {
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(cast_idx->size_ / this->local_backend_.HIP_block_size + 1);

            // Prepare send buffer
            kernel_get_index_values<<<GridSize,
                                      BlockSize,
                                      0,
                                      HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                cast_idx->size_, cast_idx->vec_, this->vec_, cast_vec->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::SetIndexValues(const BaseVector<int>&       index,
                                                         const BaseVector<ValueType>& values)
    {
        const HIPAcceleratorVector<int>* cast_idx
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&index);
        const HIPAcceleratorVector<ValueType>* cast_vec
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&values);

        assert(cast_idx != NULL);
        assert(cast_vec != NULL);
        assert(cast_vec->size_ == cast_idx->size_);

        if(cast_idx->size_ > 0)
        {
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(cast_idx->size_ / this->local_backend_.HIP_block_size + 1);

            kernel_set_index_values<<<GridSize,
                                      BlockSize,
                                      0,
                                      HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                cast_idx->size_, cast_idx->vec_, cast_vec->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::AddIndexValues(const BaseVector<int>&       index,
                                                         const BaseVector<ValueType>& values)
    {
        const HIPAcceleratorVector<int>* cast_idx
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&index);
        const HIPAcceleratorVector<ValueType>* cast_vec
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&values);

        assert(cast_idx != NULL);
        assert(cast_vec != NULL);
        assert(cast_vec->size_ == cast_idx->size_);

        if(cast_idx->size_ > 0)
        {
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(cast_idx->size_ / this->local_backend_.HIP_block_size + 1);

            kernel_add_index_values<<<GridSize,
                                      BlockSize,
                                      0,
                                      HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                cast_idx->size_, cast_idx->vec_, cast_vec->vec_, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <>
    void HIPAcceleratorVector<bool>::AddIndexValues(const BaseVector<int>&  index,
                                                    const BaseVector<bool>& values)
    {
        LOG_INFO("AddIndexValues<bool>() is not available");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::GetContinuousValues(int64_t    start,
                                                              int64_t    end,
                                                              ValueType* values) const
    {
        assert(start >= 0);
        assert(end >= start);
        assert(end <= this->size_);
        assert(values != NULL);

        // Asynchronous memcpy
        copy_d2h(end - start,
                 this->vec_ + start,
                 values,
                 true,
                 HIPSTREAM(this->local_backend_.HIP_stream_current));
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::SetContinuousValues(int64_t          start,
                                                              int64_t          end,
                                                              const ValueType* values)
    {
        assert(start >= 0);
        assert(end >= start);
        assert(end <= this->size_);

        // Asynchronous memcpy
        copy_h2d(end - start,
                 values,
                 this->vec_ + start,
                 true,
                 HIPSTREAM(this->local_backend_.HIP_stream_current));
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::RSPMISUpdateCFmap(const BaseVector<int>& index,
                                                            BaseVector<ValueType>* values)
    {
        LOG_INFO("RSPMISUpdateCFmap() is only available for int");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HIPAcceleratorVector<int>::RSPMISUpdateCFmap(const BaseVector<int>& index,
                                                      BaseVector<int>*       values)
    {
        assert(values != NULL);

        const HIPAcceleratorVector<int>* cast_idx
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&index);
        HIPAcceleratorVector<int>* cast_vec = dynamic_cast<HIPAcceleratorVector<int>*>(values);

        assert(cast_idx != NULL);
        assert(cast_vec != NULL);
        assert(cast_vec->size_ == cast_idx->size_);

        kernel_rs_pmis_cf_update_pack<256><<<(cast_idx->size_ - 1) / 256 + 1,
                                             256,
                                             0,
                                             HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            cast_idx->size_, cast_idx->vec_, cast_vec->vec_, this->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::ExtractCoarseMapping(
        int64_t start, int64_t end, const int* index, int nc, int* size, int* map) const
    {
        LOG_INFO("ExtractCoarseMapping() NYI for HIP");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::ExtractCoarseBoundary(
        int64_t start, int64_t end, const int* index, int nc, int* size, int* boundary) const
    {
        LOG_INFO("ExtractCoarseBoundary() NYI for HIP");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Power(double power)
    {
        if(this->size_ > 0)
        {
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->size_ / this->local_backend_.HIP_block_size + 1);

            kernel_power<<<GridSize,
                           BlockSize,
                           0,
                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->size_, power, this->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::SetRandomUniform(unsigned long long seed,
                                                           ValueType          a,
                                                           ValueType          b)
    {
        if(this->size_ > 0)
        {
            //
            // Create the random calculator.
            //
            HIPRandUniform_rocRAND<ValueType> rand_engine_uniform(
                seed, std::real(a), std::real(b), &this->local_backend_);

            //
            // Apply the random calculator.
            //
            this->SetRandom(rand_engine_uniform);
        }
    }

    //
    // No internal usage for integral types, so let's skip the implementation rather than providing one we do not use.
    //
    template <>
    void HIPAcceleratorVector<bool>::SetRandomUniform(unsigned long long seed, bool a, bool b)
    {
        LOG_INFO("HIPAcceleratorVector::SetRandomUniform(), available implementation are for "
                 "float, double, complex float and complex double only.");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    //
    // No internal usage for integral types, so let's skip the implementation rather than providing one we do not use.
    //
    template <>
    void HIPAcceleratorVector<int>::SetRandomUniform(unsigned long long seed, int a, int b)
    {
        LOG_INFO("HIPAcceleratorVector::SetRandomUniform(), available implementation are for "
                 "float, double, complex float and complex double only.");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    //
    // No internal usage for integral types, so let's skip the implementation rather than providing one we do not use.
    //
    template <>
    void HIPAcceleratorVector<int64_t>::SetRandomUniform(unsigned long long seed,
                                                         int64_t            a,
                                                         int64_t            b)
    {
        LOG_INFO("HIPAcceleratorVector::SetRandomUniform(), available implementation are for "
                 "float, double, complex float and complex double only.");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::SetRandomNormal(unsigned long long seed,
                                                          ValueType          mean,
                                                          ValueType          var)
    {
        //
        // Create the random calculator.
        //
        HIPRandNormal_rocRAND<ValueType> rand_engine_normal(seed, std::real(mean), std::real(var));

        //
        // Apply the random calculator.
        //
        this->SetRandom(rand_engine_normal);
    }

    //
    // No internal usage for integral types, so let's skip the implementation rather than providing one we do not use.
    //
    template <>
    void HIPAcceleratorVector<bool>::SetRandomNormal(unsigned long long seed, bool mean, bool var)
    {
        LOG_INFO("HIPAcceleratorVector::SetRandomNormal(), available implementation are for float, "
                 "double, complex float and complex double only.");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    //
    // No internal usage for integral types, so let's skip the implementation rather than providing one we do not use.
    //
    template <>
    void HIPAcceleratorVector<int>::SetRandomNormal(unsigned long long seed, int mean, int var)
    {
        LOG_INFO("HIPAcceleratorVector::SetRandomNormal(), available implementation are for float, "
                 "double, complex float and complex double only.");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    //
    // No internal usage for integral types, so let's skip the implementation rather than providing one we do not use.
    //
    template <>
    void HIPAcceleratorVector<int64_t>::SetRandomNormal(unsigned long long seed,
                                                        int64_t            mean,
                                                        int64_t            var)
    {
        LOG_INFO("HIPAcceleratorVector::SetRandomNormal(), available implementation are for float, "
                 "double, complex float and complex double only.");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HIPAcceleratorVector<std::complex<float>>::Power(double power)
    {
        if(this->size_ > 0)
        {
            LOG_INFO("HIPAcceleratorVector::Power(), no pow() for complex float in HIP");
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <>
    void HIPAcceleratorVector<std::complex<double>>::Power(double power)
    {
        if(this->size_ > 0)
        {
            LOG_INFO("HIPAcceleratorVector::Power(), no pow() for complex double in HIP");
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <>
    void HIPAcceleratorVector<int>::Power(double power)
    {
        if(this->size_ > 0)
        {
            LOG_INFO("HIPAcceleratorVector::Power(), no pow() for int in HIP");
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorVector<ValueType>::Sort(BaseVector<ValueType>* sorted,
                                               BaseVector<int>*       perm) const
    {
        if(this->size_ > 0)
        {
            assert(sorted != NULL);

            HIPAcceleratorVector<ValueType>* cast_sort
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(sorted);
            HIPAcceleratorVector<int>* cast_perm = dynamic_cast<HIPAcceleratorVector<int>*>(perm);

            assert(cast_sort != NULL);

            void*  buffer = NULL;
            size_t size;

            unsigned int begin_bit = 0;
            unsigned int end_bit   = 8 * sizeof(ValueType);

            if(cast_perm == NULL)
            {
                // Sort without permutation
                rocprim::radix_sort_keys(buffer,
                                         size,
                                         this->vec_,
                                         cast_sort->vec_,
                                         this->size_,
                                         begin_bit,
                                         end_bit,
                                         HIPSTREAM(this->local_backend_.HIP_stream_current));
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                hipMalloc(&buffer, size);
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                rocprim::radix_sort_keys(buffer,
                                         size,
                                         this->vec_,
                                         cast_sort->vec_,
                                         this->size_,
                                         begin_bit,
                                         end_bit,
                                         HIPSTREAM(this->local_backend_.HIP_stream_current));
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                hipFree(buffer);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
            else
            {
                assert(cast_perm != NULL);

                // workspace
                int* workspace = NULL;
                allocate_hip(this->size_, &workspace);

                // Create identity permutation
                rocsparse_status status = rocsparse_create_identity_permutation(
                    ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                    this->size_,
                    workspace);
                CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

                // Radix sort pairs
                rocprim::radix_sort_pairs(buffer,
                                          size,
                                          this->vec_,
                                          cast_sort->vec_,
                                          workspace,
                                          cast_perm->vec_,
                                          this->size_,
                                          begin_bit,
                                          end_bit,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current));
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                hipMalloc(&buffer, size);
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                rocprim::radix_sort_pairs(buffer,
                                          size,
                                          this->vec_,
                                          cast_sort->vec_,
                                          workspace,
                                          cast_perm->vec_,
                                          this->size_,
                                          begin_bit,
                                          end_bit,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current));
                CHECK_HIP_ERROR(__FILE__, __LINE__);

                hipFree(buffer);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <>
    void HIPAcceleratorVector<std::complex<float>>::Sort(BaseVector<std::complex<float>>* sorted,
                                                         BaseVector<int>* perm) const
    {
        LOG_INFO("HIPAcceleratorVector::Sort(), how to sort complex numbers?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HIPAcceleratorVector<std::complex<double>>::Sort(BaseVector<std::complex<double>>* sorted,
                                                          BaseVector<int>* perm) const
    {
        LOG_INFO("HIPAcceleratorVector::Sort(), how to sort complex numbers?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template class HIPAcceleratorVector<double>;
    template class HIPAcceleratorVector<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorVector<std::complex<double>>;
    template class HIPAcceleratorVector<std::complex<float>>;
#endif
    template class HIPAcceleratorVector<bool>;
    template class HIPAcceleratorVector<int>;
    template class HIPAcceleratorVector<int64_t>;

} // namespace rocalution
