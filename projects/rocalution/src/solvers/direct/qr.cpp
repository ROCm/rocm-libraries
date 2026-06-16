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

#include "qr.hpp"
#include "../../utils/def.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <complex>
#include <math.h>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    QR<OperatorType, VectorType, ValueType>::QR()
    {
        log_debug(this, "QR::QR()");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    QR<OperatorType, VectorType, ValueType>::~QR()
    {
        log_debug(this, "QR::~QR()");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void QR<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("QR solver");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QR<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        LOG_INFO("QR direct solver starts");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QR<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        LOG_INFO("QR ends");
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void QR<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "QR::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->op_ != NULL);
        assert(this->op_->GetM() == this->op_->GetN());
        assert(this->op_->GetM() > 0);

        this->qr_.CloneFrom(*this->op_);
        this->qr_.QRDecompose();

        log_debug(this, "QR::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QR<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "QR::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->qr_.Clear();
            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QR<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "QR::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->qr_.MoveToHost();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QR<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "QR::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->qr_.MoveToAccelerator();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QR<OperatorType, VectorType, ValueType>::Solve_(const VectorType& rhs, VectorType* x)
    {
        log_debug(this, "QR::Solve_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->build_ == true);

        this->qr_.QRSolve(rhs, x);

        log_debug(this, "QR::Solve_()", " #*# end");
    }

    template class QR<LocalMatrix<double>, LocalVector<double>, double>;
    template class QR<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class QR<LocalMatrix<std::complex<double>>,
                      LocalVector<std::complex<double>>,
                      std::complex<double>>;
    template class QR<LocalMatrix<std::complex<float>>,
                      LocalVector<std::complex<float>>,
                      std::complex<float>>;
#endif

} // namespace rocalution
