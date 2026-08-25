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

#include "inversion.hpp"
#include "../../utils/def.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <complex>
#include <math.h>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    Inversion<OperatorType, VectorType, ValueType>::Inversion()
    {
        log_debug(this, "Inversion::Inversion()");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    Inversion<OperatorType, VectorType, ValueType>::~Inversion()
    {
        log_debug(this, "Inversion::~Inversion()");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void Inversion<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("Inversion solver");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Inversion<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        LOG_INFO("Inversion direct solver starts");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Inversion<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        LOG_INFO("Inversion ends");
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void Inversion<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "Inversion::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->op_ != NULL);
        assert(this->op_->GetM() == this->op_->GetN());
        assert(this->op_->GetM() > 0);

        this->inverse_.CloneFrom(*this->op_);
        this->inverse_.Invert();

        log_debug(this, "Inversion::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Inversion<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "Inversion::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->inverse_.Clear();
            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Inversion<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "Inversion::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->inverse_.MoveToHost();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Inversion<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "Inversion::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->inverse_.MoveToAccelerator();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Inversion<OperatorType, VectorType, ValueType>::Solve_(const VectorType& rhs,
                                                                VectorType*       x)
    {
        log_debug(this, "Inversion::Solve_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->build_ == true);

        this->inverse_.Apply(rhs, x);

        log_debug(this, "Inversion::Solve_()", " #*# end");
    }

    template class Inversion<LocalMatrix<double>, LocalVector<double>, double>;
    template class Inversion<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class Inversion<LocalMatrix<std::complex<double>>,
                             LocalVector<std::complex<double>>,
                             std::complex<double>>;
    template class Inversion<LocalMatrix<std::complex<float>>,
                             LocalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

} // namespace rocalution
