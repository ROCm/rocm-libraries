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

#include "multigrid.hpp"
#include "../../utils/def.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/local_matrix.hpp"

#include "../../base/global_vector.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <complex>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    MultiGrid<OperatorType, VectorType, ValueType>::MultiGrid()
    {
        log_debug(this, "MultiGrid::MultiGrid()", "default constructor");

        this->scaling_ = true;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    MultiGrid<OperatorType, VectorType, ValueType>::~MultiGrid()
    {
        log_debug(this, "MultiGrid::~MultiGrid()", "destructor");

        delete[] this->restrict_op_level_;
        delete[] this->prolong_op_level_;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiGrid<OperatorType, VectorType, ValueType>::SetRestrictOperator(OperatorType** op)
    {
        log_debug(this, "MultiGrid::SetRestrictOperator()", op);

        assert(this->build_ == false);
        assert(op != NULL);
        assert(this->levels_ > 0);

        this->restrict_op_level_ = new OperatorType*[this->levels_];

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            this->restrict_op_level_[i] = op[i];
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiGrid<OperatorType, VectorType, ValueType>::SetProlongOperator(OperatorType** op)
    {
        log_debug(this, "MultiGrid::SetProlongOperator()", op);

        assert(this->build_ == false);
        assert(op != NULL);
        assert(this->levels_ > 0);

        this->prolong_op_level_ = new OperatorType*[this->levels_];

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            this->prolong_op_level_[i] = op[i];
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiGrid<OperatorType, VectorType, ValueType>::SetOperatorHierarchy(OperatorType** op)
    {
        log_debug(this, "MultiGrid::SetOperatorHierarchy()", op);

        assert(this->build_ == false);
        assert(op != NULL);

        this->op_level_ = op;
    }

    template class MultiGrid<LocalMatrix<double>, LocalVector<double>, double>;
    template class MultiGrid<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class MultiGrid<LocalMatrix<std::complex<double>>,
                             LocalVector<std::complex<double>>,
                             std::complex<double>>;
    template class MultiGrid<LocalMatrix<std::complex<float>>,
                             LocalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

    template class MultiGrid<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class MultiGrid<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class MultiGrid<GlobalMatrix<std::complex<double>>,
                             GlobalVector<std::complex<double>>,
                             std::complex<double>>;
    template class MultiGrid<GlobalMatrix<std::complex<float>>,
                             GlobalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

} // namespace rocalution
