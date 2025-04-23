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

#ifndef ROCALUTION_GMRES_GMRES_HPP_
#define ROCALUTION_GMRES_GMRES_HPP_

#include "../solver.hpp"
#include "rocalution/export.hpp"

#include <vector>

namespace rocalution
{

    /** \ingroup solver_module
  * \class GMRES
  * \brief Generalized Minimum Residual Method
  * \details
  * The Generalized Minimum Residual method (GMRES) is a projection method for solving
  * sparse (non) symmetric linear systems \f$Ax=b\f$, based on restarting technique. The
  * solution is approximated in a Krylov subspace \f$\mathcal{K}=\mathcal{K}_{m}\f$ and
  * \f$\mathcal{L}=A\mathcal{K}_{m}\f$ with minimal residual, where \f$\mathcal{K}_{m}\f$
  * is the \f$m\f$-th Krylov subspace with \f$v_{1} = r_{0}/||r_{0}||_{2}\f$.
  * \cite SAAD
  *
  * The Krylov subspace basis size can be set using SetBasisSize(). The default size is
  * 30.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class GMRES : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        GMRES();
        ROCALUTION_EXPORT
        virtual ~GMRES();

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
        VectorType   z_;

        ValueType* c_;
        ValueType* s_;
        ValueType* r_;
        ValueType* H_;

        int size_basis_;
    };

} // namespace rocalution

#endif // ROCALUTION_GMRES_GMRES_HPP_
