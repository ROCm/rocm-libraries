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

#ifndef ROCALUTION_FGMRES_FGMRES_HPP_
#define ROCALUTION_FGMRES_FGMRES_HPP_

#include "../solver.hpp"
#include "rocalution/export.hpp"

#include <vector>

namespace rocalution
{

    /** \ingroup solver_module
  * \class FGMRES
  * \brief Flexible Generalized Minimum Residual Method
  * \details
  * The Flexible Generalized Minimum Residual method (FGMRES) is a projection method for
  * solving sparse (non) symmetric linear systems \f$Ax=b\f$. It is similar to the GMRES
  * method with the only difference, the FGMRES is based on a window shifting of the
  * Krylov subspace and thus allows the preconditioner \f$M^{-1}\f$ to be not a constant
  * operator. This can be especially helpful if the operation \f$M^{-1}x\f$ is the result
  * of another iterative process and not a constant operator.
  * \cite SAAD
  *
  * The Krylov subspace basis
  * size can be set using SetBasisSize(). The default size is 30.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class FGMRES : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        FGMRES();
        ROCALUTION_EXPORT
        virtual ~FGMRES();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void ReBuildNumeric(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Set the size of the Krylov subspace basis */
        ROCALUTION_EXPORT
        virtual void SetBasisSize(int size_basis);

    protected:
        virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
        virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

        virtual void PrintStart_(void) const;
        virtual void PrintEnd_(void) const;

        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

        /** \brief Generate Givens rotation */
        static void GenerateGivensRotation_(ValueType dx, ValueType dy, ValueType& c, ValueType& s);
        /** \brief Apply Givens rotation */
        static void ApplyGivensRotation_(ValueType c, ValueType s, ValueType& dx, ValueType& dy);

    private:
        VectorType** v_;
        VectorType** z_;

        ValueType* c_;
        ValueType* s_;
        ValueType* r_;
        ValueType* H_;

        int size_basis_;
    };

} // namespace rocalution

#endif // ROCALUTION_FGMRES_FGMRES_HPP_
