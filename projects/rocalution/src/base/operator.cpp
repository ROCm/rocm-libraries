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

#include "operator.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "global_vector.hpp"
#include "local_vector.hpp"
#include "vector.hpp"

#include <complex>

namespace rocalution
{

    template <typename ValueType>
    Operator<ValueType>::Operator()
    {
        log_debug(this, "Operator::Operator()");

        this->object_name_ = "";
    }

    template <typename ValueType>
    Operator<ValueType>::~Operator()
    {
        log_debug(this, "Operator::~Operator()");
    }

    template <typename ValueType>
    int64_t Operator<ValueType>::GetLocalM(void) const
    {
        return this->GetM();
    }

    template <typename ValueType>
    int64_t Operator<ValueType>::GetLocalN(void) const
    {
        return this->GetN();
    }

    template <typename ValueType>
    int64_t Operator<ValueType>::GetLocalNnz(void) const
    {
        return this->GetNnz();
    }

    template <typename ValueType>
    int64_t Operator<ValueType>::GetGhostM(void) const
    {
        return 0;
    }

    template <typename ValueType>
    int64_t Operator<ValueType>::GetGhostN(void) const
    {
        return 0;
    }

    template <typename ValueType>
    int64_t Operator<ValueType>::GetGhostNnz(void) const
    {
        return 0;
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    void Operator<ValueType>::Transpose(void)
    {
        LOG_INFO("Operator<ValueType>::Transpose() not implemented for this operator");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Operator<ValueType>::Apply(const GlobalVector<ValueType>& in,
                                    GlobalVector<ValueType>*       out) const
    {
        LOG_INFO("Operator<ValueType>::Apply(const GlobalVector<ValueType>& in, "
                 "GlobalVector<ValueType> *out)");
        LOG_INFO("Mismatched types:");
        this->Info();
        in.Info();
        out->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Operator<ValueType>::Apply(const LocalVector<ValueType>& in,
                                    LocalVector<ValueType>*       out) const
    {
        LOG_INFO(
            "Operator<ValueType>::Apply(const LocalVector<ValueType>& in, LocalVector<ValueType> "
            "*out)");
        LOG_INFO("Mismatched types:");
        this->Info();
        in.Info();
        out->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Operator<ValueType>::ApplyAdd(const GlobalVector<ValueType>& in,
                                       ValueType                      scalar,
                                       GlobalVector<ValueType>*       out) const
    {
        LOG_INFO("Operator<ValueType>::ApplyAdd(const GlobalVector<ValueType>& in, ValueType "
                 "scalar, GlobalVector<ValueType> *out)");
        LOG_INFO("Mismatched types:");
        this->Info();
        in.Info();
        out->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void Operator<ValueType>::ApplyAdd(const LocalVector<ValueType>& in,
                                       ValueType                     scalar,
                                       LocalVector<ValueType>*       out) const
    {
        LOG_INFO("Operator<ValueType>::ApplyAdd(const LocalVector<ValueType>& in, ValueType "
                 "scalar, LocalVector<ValueType> *out)");
        LOG_INFO("Mismatched types:");
        this->Info();
        in.Info();
        out->Info();
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template class Operator<double>;
    template class Operator<float>;
#ifdef SUPPORT_COMPLEX
    template class Operator<std::complex<double>>;
    template class Operator<std::complex<float>>;
#endif
    template class Operator<int>;

} // namespace rocalution
