/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_KRYLOV_CHEBYSHEV_HPP_
#define ROCALUTION_KRYLOV_CHEBYSHEV_HPP_

#include "rocalution/export.hpp"
#include "solver.hpp"

#include <vector>

namespace rocalution
{

    /** \ingroup solver_module
  * \class Chebyshev
  * \brief Chebyshev Iteration Scheme
  * \details
  * The Chebyshev Iteration scheme (also known as acceleration scheme) is similar to the
  * CG method but requires minimum and maximum eigenvalues of the operator.
  * \cite templates
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class Chebyshev : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        Chebyshev();
        ROCALUTION_EXPORT
        virtual ~Chebyshev();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        /** \brief Set the minimum and maximum eigenvalues of the operator */
        ROCALUTION_EXPORT
        void Set(ValueType lambda_min, ValueType lambda_max);

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void ReBuildNumeric(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
        virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

        virtual void PrintStart_(void) const;
        virtual void PrintEnd_(void) const;

        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        bool      init_lambda_;
        ValueType lambda_min_, lambda_max_;

        VectorType r_, z_;
        VectorType p_;
    };

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_CHEBYSHEV_HPP_
