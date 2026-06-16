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

#include "vector.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "global_vector.hpp"
#include "local_vector.hpp"

#include <complex>

namespace rocalution
{

    template <typename ValueType>
    Vector<ValueType>::Vector()
    {
        log_debug(this, "Vector::Vector()");

        this->object_name_ = "";
    }

    template <typename ValueType>
    Vector<ValueType>::~Vector()
    {
        log_debug(this, "Vector::~Vector()");
    }

    template <typename ValueType>
    int64_t Vector<ValueType>::GetLocalSize(void) const
    {
        return this->GetSize();
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    void Vector<ValueType>::CopyFrom(const LocalVector<ValueType>& src)
    {
        LOG_INFO("Vector<ValueType>::CopyFrom(const LocalVector<ValueType>& src)");
        LOG_INFO("Mismatched types:");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::CopyFrom(const GlobalVector<ValueType>& src)
    {
        LOG_INFO("Vector<ValueType>::CopyFrom(const GlobalVector<ValueType>& src)");
        LOG_INFO("Mismatched types:");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::CopyFromAsync(const LocalVector<ValueType>& src)
    {
        LOG_INFO("Vector<ValueType>::CopyFromAsync(const LocalVector<ValueType>& src)");
        LOG_INFO("Mismatched types:");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::CopyFromFloat(const LocalVector<float>& src)
    {
        LOG_INFO("Vector<ValueType>::CopyFromFloat(const LocalVector<float>& src)");
        LOG_INFO("Mismatched types:");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::CopyFromDouble(const LocalVector<double>& src)
    {
        LOG_INFO("Vector<ValueType>::CopyFromDouble(const LocalVector<double>& src)");
        LOG_INFO("Mismatched types:");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::CopyFrom(const LocalVector<ValueType>& src,
                                     int64_t                       src_offset,
                                     int64_t                       dst_offset,
                                     int64_t                       size)
    {
        LOG_INFO("Vector<ValueType>::CopyFrom(const LocalVector<ValueType>& src,"
                 "int64_t src_offset,"
                 "int64_t dst_offset,"
                 "int64_t size");
        LOG_INFO("Mismatched types:");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::CloneFrom(const LocalVector<ValueType>& src)
    {
        LOG_INFO("Vector<ValueType>::CloneFrom(const LocalVector<ValueType>& src)");
        LOG_INFO("Mismatched types:");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::CloneFrom(const GlobalVector<ValueType>& src)
    {
        LOG_INFO("Vector<ValueType>::CloneFrom(const GlobalVector<ValueType>& src)");
        LOG_INFO("Mismatched types:");
        this->Info();
        src.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::AddScale(const LocalVector<ValueType>& x, ValueType alpha)
    {
        LOG_INFO("Vector<ValueType>::AddScale(const LocalVector<ValueType>& x, ValueType alpha)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::AddScale(const GlobalVector<ValueType>& x, ValueType alpha)
    {
        LOG_INFO("Vector<ValueType>::AddScale(const GlobalVector<ValueType>& x, ValueType alpha)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x)
    {
        LOG_INFO("Vector<ValueType>::ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x)
    {
        LOG_INFO("Vector<ValueType>::ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::ScaleAddScale(ValueType                     alpha,
                                          const LocalVector<ValueType>& x,
                                          ValueType                     beta,
                                          int64_t                       src_offset,
                                          int64_t                       dst_offset,
                                          int64_t                       size)
    {
        LOG_INFO("Vector<ValueType>::ScaleAddScale(ValueType alpha,"
                 "const LocalVector<ValueType>& x,"
                 "ValueType beta,"
                 "int64_t src_offset,"
                 "int64_t dst_offset,"
                 "int64_t size)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::ScaleAddScale(ValueType                      alpha,
                                          const GlobalVector<ValueType>& x,
                                          ValueType                      beta,
                                          int64_t                        src_offset,
                                          int64_t                        dst_offset,
                                          int64_t                        size)
    {
        LOG_INFO("Vector<ValueType>::ScaleAddScale(ValueType alpha,"
                 "const GlobalVector<ValueType>& x,"
                 "ValueType beta,"
                 "int64_t src_offset,"
                 "int64_t dst_offset,"
                 "int64_t size)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType Vector<ValueType>::Dot(const LocalVector<ValueType>& x) const
    {
        LOG_INFO("Vector<ValueType>::Dot(const LocalVector<ValueType>& x) const");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType Vector<ValueType>::Dot(const GlobalVector<ValueType>& x) const
    {
        LOG_INFO("Vector<ValueType>::Dot(const GlobalVector<ValueType>& x) const");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType Vector<ValueType>::DotNonConj(const LocalVector<ValueType>& x) const
    {
        LOG_INFO("Vector<ValueType>::DotNonConj(const LocalVector<ValueType>& x) const");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType Vector<ValueType>::DotNonConj(const GlobalVector<ValueType>& x) const
    {
        LOG_INFO("Vector<ValueType>::DotNonConj(const GlobalVector<ValueType>& x) const");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType Vector<ValueType>::InclusiveSum(const LocalVector<ValueType>& vec)
    {
        LOG_INFO("Vector<ValueType>::InclusiveSum(const LocalVector<ValueType>& vec)");
        LOG_INFO("Mismatched types:");
        this->Info();
        vec.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType Vector<ValueType>::InclusiveSum(const GlobalVector<ValueType>& vec)
    {
        LOG_INFO("Vector<ValueType>::InclusiveSum(const GlobalVector<ValueType>& vec)");
        LOG_INFO("Mismatched types:");
        this->Info();
        vec.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType Vector<ValueType>::ExclusiveSum(const LocalVector<ValueType>& vec)
    {
        LOG_INFO("Vector<ValueType>::ExclusiveSum(const LocalVector<ValueType>& vec)");
        LOG_INFO("Mismatched types:");
        this->Info();
        vec.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    ValueType Vector<ValueType>::ExclusiveSum(const GlobalVector<ValueType>& vec)
    {
        LOG_INFO("Vector<ValueType>::ExclusiveSum(const GlobalVector<ValueType>& vec)");
        LOG_INFO("Mismatched types:");
        this->Info();
        vec.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x)
    {
        LOG_INFO("Vector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x)
    {
        LOG_INFO("Vector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x,
                                          const LocalVector<ValueType>& y)
    {
        LOG_INFO("Vector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x, const "
                 "LocalVector<ValueType>& y)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        y.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x,
                                          const GlobalVector<ValueType>& y)
    {
        LOG_INFO("Vector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x, const "
                 "GlobalVector<ValueType>& y)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        y.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::ScaleAddScale(ValueType                     alpha,
                                          const LocalVector<ValueType>& x,
                                          ValueType                     beta)
    {
        LOG_INFO("ScaleAddScale(ValueType alpha, const LocalVector<ValueType>& x, "
                 "ValueType beta)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::ScaleAddScale(ValueType                      alpha,
                                          const GlobalVector<ValueType>& x,
                                          ValueType                      beta)
    {
        LOG_INFO("ScaleAddScale(ValueType alpha, const GlobalVector<ValueType>& x, "
                 "ValueType beta)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::ScaleAdd2(ValueType                     alpha,
                                      const LocalVector<ValueType>& x,
                                      ValueType                     beta,
                                      const LocalVector<ValueType>& y,
                                      ValueType                     gamma)
    {
        LOG_INFO("ScaleAdd2(ValueType alpha, const LocalVector<ValueType>& x, ValueType "
                 "beta, const LocalVector<ValueType>& y, ValueType gamma)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        y.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Vector<ValueType>::ScaleAdd2(ValueType                      alpha,
                                      const GlobalVector<ValueType>& x,
                                      ValueType                      beta,
                                      const GlobalVector<ValueType>& y,
                                      ValueType                      gamma)
    {
        LOG_INFO("ScaleAdd2(ValueType alpha, const GlobalVector<ValueType>& x, ValueType "
                 "beta, const GlobalVector<ValueType>& y, ValueType gamma)");
        LOG_INFO("Mismatched types:");
        this->Info();
        x.Info();
        y.Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template class Vector<double>;
    template class Vector<float>;
#ifdef SUPPORT_COMPLEX
    template class Vector<std::complex<double>>;
    template class Vector<std::complex<float>>;
#endif
    template class Vector<bool>;
    template class Vector<int>;
    template class Vector<int64_t>;

} // namespace rocalution
