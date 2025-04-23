/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_VECTOR_HPP
#define ROCSPARSE_VECTOR_HPP

#include "rocsparse_allocator.hpp"
#include "rocsparse_init.hpp"

template <memory_mode::value_t MODE, typename T>
struct dense_vector;
template <typename T>
struct host_vector;

template <typename T>
using host_dense_vector = dense_vector<memory_mode::host, T>;

template <typename T>
using device_dense_vector = dense_vector<memory_mode::device, T>;

template <typename T>
using managed_dense_vector = dense_vector<memory_mode::managed, T>;

template <memory_mode::value_t MODE, typename T>
struct dense_vector_t
{
protected:
    size_t m_size;
    T*     m_val;

public:
    using value_type = T;
    dense_vector_t(size_t size, T* val);
    dense_vector_t& operator()(size_t size, T* val)
    {
        m_size = size;
        m_val  = val;
        return *this;
    }
    size_t   size() const;
             operator T*();
             operator const T*() const;
    T*       data();
    const T* data() const;
    ~dense_vector_t();

    // Disallow copying or assigning
    dense_vector_t<MODE, T>(const dense_vector_t<MODE, T>&) = delete;
    template <memory_mode::value_t THAT_MODE>
    dense_vector_t<MODE, T>(const dense_vector_t<THAT_MODE, T>&) = delete;

    dense_vector_t<MODE, T>& operator=(const dense_vector_t<MODE, T>&);
    template <memory_mode::value_t THAT_MODE>
    dense_vector_t<MODE, T>& operator=(const dense_vector_t<THAT_MODE, T>&);
    template <memory_mode::value_t THAT_MODE>
    void unit_check(const dense_vector_t<THAT_MODE, T>& that_) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            host_dense_vector<T> on_host(*this);
            on_host.unit_check(that_);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::managed:
            case memory_mode::host:
            {
                unit_check_scalar<size_t>(this->size(), that_.size());
                unit_check_segments<T>(this->size(), this->data(), that_.data());
                break;
            }
            case memory_mode::device:
            {
                host_dense_vector<T> that(that_);
                this->unit_check(that);
                break;
            }
            }
            break;
        }
        }
    }
    template <memory_mode::value_t THAT_MODE>
    void near_check(const dense_vector_t<THAT_MODE, T>& that_,
                    floating_data_t<T>                  tol_ = default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            host_dense_vector<T> on_host(*this, true);
            on_host.near_check(that_, tol_);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::managed:
            case memory_mode::host:
            {
                unit_check_scalar<size_t>(this->size(), that_.size());
                near_check_segments<T>(this->size(), this->data(), that_.data(), tol_);
                break;
            }
            case memory_mode::device:
            {
                host_dense_vector<T> that(that_);
                this->near_check(that, tol_);
                break;
            }
            }
            break;
        }
        }
    }
    void print() const;

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const dense_vector_t<THAT_MODE, T>& that)
    {
        CHECK_HIP_THROW_ERROR(this->size() == that.size() ? hipSuccess : hipErrorInvalidValue);
        auto err = hipMemcpy(this->data(),
                             that.data(),
                             sizeof(T) * that.size(),
                             memory_mode::get_hipMemcpyKind(MODE, THAT_MODE));
        CHECK_HIP_THROW_ERROR(err);
    }

    void transfer_to(std::vector<T>& that) const;
    void transfer_from(const host_vector<T>& that);
    void unit_check(const host_vector<T>& that_) const;
    void near_check(const host_vector<T>& that_,
                    floating_data_t<T>    tol_ = default_tolerance<T>::value) const;
};

template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

    // Decay into pointer wherever pointer is expected
    operator T*()
    {
        return this->data();
    }
    operator const T*() const
    {
        return this->data();
    }

    T& operator[](int64_t index)
    {
        return this->data()[index];
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const dense_vector_t<THAT_MODE, T>& that)
    {
        CHECK_HIP_THROW_ERROR(this->size() == that.size() ? hipSuccess : hipErrorInvalidValue);
        CHECK_HIP_THROW_ERROR(
            hipMemcpy(this->data(),
                      that.data(),
                      sizeof(T) * that.size(),
                      memory_mode::get_hipMemcpyKind(memory_mode::host, THAT_MODE)));
    }
    void transfer_from(const host_vector<T>& that)
    {
        CHECK_HIP_THROW_ERROR(this->size() == that.size() ? hipSuccess : hipErrorInvalidValue);
        CHECK_HIP_THROW_ERROR(
            hipMemcpy(this->data(),
                      that.data(),
                      sizeof(T) * that.size(),
                      memory_mode::get_hipMemcpyKind(memory_mode::host, memory_mode::host)));
    }

    template <memory_mode::value_t THAT_MODE>
    void unit_check(const dense_vector_t<THAT_MODE, T>& that_) const
    {
        that_.unit_check(*this);
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const dense_vector_t<THAT_MODE, T>& that_,
                    floating_data_t<T>                  tol_ = default_tolerance<T>::value) const
    {
        that_.near_check(*this, tol_);
    }

    void unit_check(const host_vector<T>& that_) const
    {
        unit_check_scalar<rocsparse_int>(this->size(), that_.size());
        unit_check_segments<T>(this->size(), this->data(), that_);
    }

    void near_check(const host_vector<T>& that_,
                    floating_data_t<T>    tol_ = default_tolerance<T>::value) const
    {
        unit_check_scalar<rocsparse_int>(this->size(), that_.size());
        near_check_segments<T>(this->size(), this->data(), that_.data(), tol_);
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const T* that)
    {
        CHECK_HIP_THROW_ERROR(
            hipMemcpy(this->data(),
                      that,
                      sizeof(T) * this->size(),
                      memory_mode::get_hipMemcpyKind(memory_mode::host, THAT_MODE)));
    };

    void print(std::ostream& out) const
    {
        const size_t N = this->size();
        for(size_t i = 0; i < N; ++i)
        {
            out << "[" << i << "] = " << (*this)[i] << std::endl;
        }
    }

    void print_limited(std::ostream& out, size_t bound) const
    {
        const size_t N = this->size();
        for(size_t i = 0; i < N; ++i)
        {
            out << "[" << i << "] = " << (*this)[i] << std::endl;
            if(i >= bound)
                break;
        }
    }
};

template <memory_mode::value_t MODE, typename T>
struct dense_vector : dense_vector_t<MODE, T>
{
private:
    using allocator = rocsparse_allocator<MODE, T>;

public:
    dense_vector()
        : dense_vector_t<MODE, T>(0, nullptr){};
    ~dense_vector()
    {
#ifdef GOOGLE_TEST
        allocator::check_guards(this->data(), this->size());
#endif
        allocator::free(this->data());
    };

    hipError_t memcheck() const
    {
        return ((this->m_size == 0) && (this->data() == nullptr))
                   ? hipSuccess
                   : (((this->m_size > 0) && (this->data() != nullptr)) ? hipSuccess
                                                                        : hipErrorOutOfMemory);
    }

    // Tell whether malloc failed

    void resize(size_t s)
    {
        if(s != this->m_size)
        {
            if(this->m_val)
            {
                allocator::free(this->m_val);
            }
            this->m_val  = allocator::malloc(s);
            this->m_size = s;
        }
    }

    explicit dense_vector(size_t s)
        : dense_vector_t<MODE, T>(s, allocator::malloc(s))
    {
    }

    explicit dense_vector(const host_vector<T>& that, bool transfer = true)
        : dense_vector_t<MODE, T>(that.size(), allocator::malloc(that.size()))
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    explicit dense_vector(const dense_vector<MODE, T>& that, bool transfer = true)
        : dense_vector_t<MODE, T>(that.size(), allocator::malloc(that.size()))
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    explicit dense_vector(const dense_vector_t<MODE, T>& that, bool transfer = true)
        : dense_vector_t<MODE, T>(that.size(), allocator::malloc(that.size()))
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    explicit dense_vector(const dense_vector_t<THAT_MODE, T>& that, bool transfer = true)
        : dense_vector_t<MODE, T>(that.size(), allocator::malloc(that.size()))
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }
};

template <memory_mode::value_t MODE, typename T>
dense_vector_t<MODE, T>::dense_vector_t(size_t size, T* val)
    : m_size(size)
    , m_val(val)
{
}

template <memory_mode::value_t MODE, typename T>
size_t dense_vector_t<MODE, T>::size() const
{
    return this->m_size;
}

template <memory_mode::value_t MODE, typename T>
dense_vector_t<MODE, T>::operator T*()
{
    return this->m_val;
}

template <memory_mode::value_t MODE, typename T>
dense_vector_t<MODE, T>::operator const T*() const
{
    return this->m_val;
}

template <memory_mode::value_t MODE, typename T>
T* dense_vector_t<MODE, T>::data()
{
    return this->m_val;
}
template <memory_mode::value_t MODE, typename T>
const T* dense_vector_t<MODE, T>::data() const
{
    return this->m_val;
}
template <memory_mode::value_t MODE, typename T>
dense_vector_t<MODE, T>::~dense_vector_t()
{
}

template <memory_mode::value_t MODE, typename T>
void dense_vector_t<MODE, T>::unit_check(const host_vector<T>& that_) const
{
    switch(MODE)
    {
    case memory_mode::device:
    {
        host_dense_vector<T> on_host(*this);
        on_host.unit_check(that_);
        break;
    }

    case memory_mode::managed:
    case memory_mode::host:
    {
        unit_check_scalar<rocsparse_int>(this->size(), that_.size());
        unit_check_segments<T>(this->size(), this->data(), that_.data());
        break;
    }
    }
}

template <memory_mode::value_t MODE, typename T>
void dense_vector_t<MODE, T>::near_check(const host_vector<T>& that_, floating_data_t<T> tol_) const
{
    switch(MODE)
    {
    case memory_mode::device:
    {
        host_dense_vector<T> on_host(*this);
        on_host.near_check(that_, tol_);
        break;
    }

    case memory_mode::managed:
    case memory_mode::host:
    {
        unit_check_scalar<rocsparse_int>(this->size(), that_.size());
        near_check_segments<T>(this->size(), this->data(), that_.data(), tol_);
        break;
    }
    }
}

template <memory_mode::value_t MODE, typename T>
void dense_vector_t<MODE, T>::transfer_from(const host_vector<T>& that)
{
    CHECK_HIP_THROW_ERROR(this->size() == that.size() ? hipSuccess : hipErrorInvalidValue);
    auto err = hipMemcpy(this->data(),
                         that.data(),
                         sizeof(T) * that.size(),
                         memory_mode::get_hipMemcpyKind(MODE, memory_mode::host));
    CHECK_HIP_THROW_ERROR(err);
}

template <memory_mode::value_t MODE, typename T>
void dense_vector_t<MODE, T>::transfer_to(std::vector<T>& that) const
{
    that.resize(this->m_size);
    CHECK_HIP_THROW_ERROR(hipMemcpy(that.data(),
                                    this->data(),
                                    sizeof(T) * this->size(),
                                    memory_mode::get_hipMemcpyKind(memory_mode::host, MODE)));
}

template <memory_mode::value_t MODE, typename T>
void dense_vector_t<MODE, T>::print() const
{
    switch(MODE)
    {
    case memory_mode::host:
    case memory_mode::managed:
    {
        size_t   N = this->size();
        const T* x = this->data();
        for(size_t i = 0; i < N; ++i)
        {
            std::cout << " " << x[i] << std::endl;
        }
        break;
    }
    case memory_mode::device:
    {
        dense_vector<memory_mode::host, T> on_host(*this, true);
        on_host.print();
        break;
    }
    }
};

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses host memory */

template <memory_mode::value_t mode_>
struct memory_traits;

template <>
struct memory_traits<memory_mode::device>
{
    template <typename S>
    using array_t = device_dense_vector<S>;
};

template <>
struct memory_traits<memory_mode::managed>
{
    template <typename S>
    using array_t = managed_dense_vector<S>;
};

//
// For compatibility.
//

template <>
struct memory_traits<memory_mode::host>
{
    template <typename S>
    using array_t = host_vector<S>;
};

template <typename T>
using device_vector = device_dense_vector<T>;

#endif // ROCSPARSE_VECTOR_HPP
