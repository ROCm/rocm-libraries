/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#include "rocsparse-types.h"
#include "utility.h"
#include <iostream>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T, typename I>
    void set_identity_array(rocsparse_handle handle_, I size_, T* x_);
    template <uint32_t BLOCKSIZE, typename T, typename I>
    void get_permuted_array(rocsparse_handle handle_, I size_, const T* a_, T* x_, const I* perm_);
    template <uint32_t BLOCKSIZE, typename T, typename I>
    void set_permuted_array(rocsparse_handle handle_, I size_, T* a_, const T* x_, const I* perm_);

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status nrminf(rocsparse_handle          handle_,
                            size_t                    nitems_,
                            const T*                  x_,
                            floating_data_t<T>*       nrm_,
                            const floating_data_t<T>* nrm0_,
                            bool                      MX);

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status nrminf_diff(rocsparse_handle          handle_,
                                 size_t                    nitems_,
                                 const T*                  x_,
                                 const T*                  y_,
                                 floating_data_t<T>*       nrm_,
                                 const floating_data_t<T>* nrm0_,
                                 bool                      MX);

    //
    // Assign nitems of type T in the buffer.
    //
    template <typename T>
    static inline T* __restrict__ assign_b(size_t&             buffer_size_,
                                           void* __restrict__& buffer_,
                                           size_t              nitems_)
    {
        if(buffer_size_ > sizeof(T) * nitems_)
        {
            T* __restrict__ res = ((T* __restrict__)buffer_);
            buffer_             = (void* __restrict__)(res + nitems_);
            buffer_size_ -= sizeof(T) * nitems_;
            return res;
        }
        else
        {
#ifndef NDEBUG
            std::cerr << "assign buffer error " << std::endl;
            exit(1);
#endif
            return nullptr;
        }
    }

    //
    // Unssign nitems of type T in the buffer.
    //
    template <typename T>
    static inline T* __restrict__ unassign_b(size_t&             buffer_size_,
                                             void* __restrict__& buffer_,
                                             size_t              nitems_)
    {
        T* __restrict__ res = ((T* __restrict__)buffer_);
        buffer_             = (void* __restrict__)(res - nitems_);
        buffer_size_ += sizeof(T) * nitems_;
        return nullptr;
    }

    template <typename T>
    inline hipError_t stay_on_device(T* target_, const T* source_, hipStream_t stream_)
    {
        return hipMemcpyAsync(target_, source_, sizeof(T), hipMemcpyDeviceToDevice, stream_);
    }

    template <typename T>
    inline hipError_t on_host(T* target_, const T* source_, hipStream_t stream_)
    {
        return hipMemcpyAsync(target_, source_, sizeof(T), hipMemcpyDeviceToHost, stream_);
    }

    template <typename T>
    inline hipError_t on_device(T* target, const T* source_, hipStream_t stream_)
    {
        return hipMemcpyAsync(target, source_, sizeof(T), hipMemcpyHostToDevice, stream_);
    }

    template <typename IMPL>
    struct buffer_layout_crtp_t
    {
    public:
        static size_t get_sizeof_double()
        {
            return ((sizeof(IMPL) - 1) / sizeof(double) + 1);
        }

        typedef enum enum_ivalue_type
        {
            perm,
            lnnz,
            lptr,
            lptr_end,
            unnz,
            uptr,
            uptr_end,
        } ivalue_type;

        typedef enum enum_jvalue_type
        {
            ind,
        } jvalue_type;

        typedef enum enum_tvalue_type
        {
            x,
            buffer
        } tvalue_type;

        size_t get_size(ivalue_type i)
        {
            return static_cast<IMPL&>(*this).get_size(i);
        }

        size_t get_size(jvalue_type i)
        {
            return static_cast<IMPL&>(*this).get_size(i);
        }

        size_t get_size(tvalue_type i)
        {
            return static_cast<IMPL&>(*this).get_size(i);
        }

        void* get_pointer(ivalue_type i)
        {
            return static_cast<IMPL&>(*this).get_pointer(i);
        }

        void* get_pointer(jvalue_type j)
        {
            return static_cast<IMPL&>(*this).get_pointer(j);
        }

        void* get_pointer(tvalue_type i)
        {
            return static_cast<IMPL&>(*this).get_pointer(i);
        }

    protected:
        buffer_layout_crtp_t(){};
        void*  m_buffer;
        size_t m_buffer_size;
    };

    struct buffer_layout_contiguous_t : buffer_layout_crtp_t<buffer_layout_contiguous_t>
    {
    protected:
        using parent_t = buffer_layout_crtp_t<buffer_layout_contiguous_t>;

    public:
        size_t get_size(typename parent_t::ivalue_type i)
        {
            return m_isizes[i];
        }
        size_t get_size(typename parent_t::jvalue_type i)
        {
            return m_jsizes[i];
        }

        size_t get_size(typename parent_t::tvalue_type i)
        {
            return m_tsizes[i];
        }

        void* get_pointer(typename parent_t::ivalue_type i)
        {
            return m_ipointers[i];
        }

        void* get_pointer(typename parent_t::jvalue_type j)
        {
            return m_jpointers[j];
        }

        void* get_pointer(typename parent_t::tvalue_type i)
        {
            return m_tpointers[i];
        }

    protected:
        void set(typename parent_t::tvalue_type v,
                 size_t&                        buffer_size_,
                 void* __restrict__&            buffer_,
                 size_t                         nitems_,
                 size_t                         sizelm_)
        {
            m_tpointers[v]
                = (void*)rocsparse::assign_b<char>(buffer_size_, buffer_, nitems_ * sizelm_);
            m_tsizes[v] = sizelm_ * nitems_;
        }

        template <typename J>
        void set(typename parent_t::jvalue_type v,
                 size_t&                        buffer_size_,
                 void* __restrict__&            buffer_,
                 size_t                         nitems_)
        {
            m_jpointers[v] = rocsparse::assign_b<J>(buffer_size_, buffer_, nitems_);
            m_jsizes[v]    = sizeof(J) * nitems_;
        }

        template <typename I>
        void set(typename parent_t::ivalue_type v,
                 size_t&                        buffer_size_,
                 void* __restrict__&            buffer_,
                 size_t                         nitems_)
        {
            m_ipointers[v] = rocsparse::assign_b<I>(buffer_size_, buffer_, nitems_);
            m_isizes[v]    = sizeof(I) * nitems_;
        }

    public:
        template <typename I, typename J>
        void init(J                   m_,
                  I                   nnz_,
                  rocsparse_datatype  datatype_,
                  size_t&             buffer_size_,
                  void* __restrict__& buffer_)
        {
            const size_t parent_sizeof_double = parent_t::get_sizeof_double();
            parent_t::m_buffer_size = buffer_size_ - parent_sizeof_double * sizeof(double);
            parent_t::m_buffer      = (void*)(((double*)buffer_) + parent_sizeof_double);

            buffer_size_ = parent_t::m_buffer_size;
            buffer_      = parent_t::m_buffer;

            this->set<I>(parent_t::perm, buffer_size_, buffer_, nnz_);
            this->set<I>(parent_t::lnnz, buffer_size_, buffer_, 1);
            this->set<I>(parent_t::lptr, buffer_size_, buffer_, m_ + 1);

            this->set<I>(parent_t::unnz, buffer_size_, buffer_, 1);
            this->set<I>(parent_t::uptr, buffer_size_, buffer_, m_ + 1);
            this->set<J>(parent_t::ind, buffer_size_, buffer_, nnz_);

            //
            // Note if A required I to be int64_t, maybe not L or U ...
            //
            m_ipointers[parent_t::lptr_end] = ((I*)m_ipointers[parent_t::lptr]) + 1;
            m_ipointers[parent_t::uptr_end] = ((I*)m_ipointers[parent_t::uptr]) + 1;

            if(datatype_ == rocsparse_datatype_f32_r)
            {
                this->set(parent_t::x, buffer_size_, buffer_, nnz_, sizeof(float));
            }
            else if(datatype_ == rocsparse_datatype_f64_r)
            {
                this->set(parent_t::x, buffer_size_, buffer_, nnz_, sizeof(double));
            }
            else if(datatype_ == rocsparse_datatype_f32_c)
            {
                this->set(
                    parent_t::x, buffer_size_, buffer_, nnz_, sizeof(rocsparse_float_complex));
            }
            else if(datatype_ == rocsparse_datatype_f64_c)
            {
                this->set(
                    parent_t::x, buffer_size_, buffer_, nnz_, sizeof(rocsparse_double_complex));
            }

            m_tpointers[parent_t::buffer] = buffer_;
            m_tsizes[parent_t::buffer]    = buffer_size_;
        }
        buffer_layout_contiguous_t(){};

    private:
        size_t m_isizes[7]{};
        size_t m_jsizes[1]{};
        size_t m_tsizes[2]{};
        void*  m_ipointers[7]{};
        void*  m_jpointers[1]{};
        void*  m_tpointers[2]{};
    };
}
