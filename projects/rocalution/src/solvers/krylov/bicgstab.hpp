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

#ifndef ROCALUTION_KRYLOV_BICGSTAB_HPP_
#define ROCALUTION_KRYLOV_BICGSTAB_HPP_

#include "../solver.hpp"
#include "rocalution/export.hpp"

namespace rocalution
{

    /** \ingroup solver_module
  * \class BiCGStab
  * \brief Bi-Conjugate Gradient Stabilized Method
  * \details
  * The Bi-Conjugate Gradient Stabilized method is a variation of CGS and solves sparse
  * (non) symmetric linear systems \f$Ax=b\f$.
  * \cite SAAD
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class BiCGStab : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        BiCGStab();
        ROCALUTION_EXPORT
        virtual ~BiCGStab();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

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
        VectorType r_;
        VectorType r0_;
        VectorType p_;
        VectorType q_;
        VectorType t_;
        VectorType v_;
        VectorType z_;
    };

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_BICGSTAB_HPP_
