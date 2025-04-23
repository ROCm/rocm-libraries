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

#ifndef ROCALUTION_MIXED_PRECISION_HPP_
#define ROCALUTION_MIXED_PRECISION_HPP_

#include "rocalution/export.hpp"
#include "solver.hpp"

namespace rocalution
{

    /** \ingroup solver_module
  * \class MixedPrecisionDC
  * \brief Mixed-Precision Defect Correction Scheme
  * \details
  * The Mixed-Precision solver is based on a defect-correction scheme. The current
  * implementation of the library is using host based correction in double precision and
  * accelerator computation in single precision. The solver is implemeting the scheme
  * \f[
  *   x_{k+1} = x_{k} + A^{-1} r_{k},
  * \f]
  * where the computation of the residual \f$r_{k} = b - Ax_{k}\f$ and the update
  * \f$x_{k+1} = x_{k} + d_{k}\f$ are performed on the host in double precision. The
  * computation of the residual system \f$Ad_{k} = r_{k}\f$ is performed on the
  * accelerator in single precision. In addition to the setup functions of the iterative
  * solver, the user need to specify the inner (\f$Ad_{k} = r_{k}\f$) solver.
  *
  * \tparam OperatorTypeH - can be LocalMatrix
  * \tparam VectorTypeH - can be LocalVector
  * \tparam ValueTypeH - can be double
  * \tparam OperatorTypeL - can be LocalMatrix
  * \tparam VectorTypeL - can be LocalVector
  * \tparam ValueTypeL - can be float
  */
    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    class MixedPrecisionDC : public IterativeLinearSolver<OperatorTypeH, VectorTypeH, ValueTypeH>
    {
    public:
        ROCALUTION_EXPORT
        MixedPrecisionDC();
        ROCALUTION_EXPORT
        virtual ~MixedPrecisionDC();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        /** \brief Set the inner solver for \f$Ad_{k} = r_{k}\f$ */
        ROCALUTION_EXPORT
        void Set(Solver<OperatorTypeL, VectorTypeL, ValueTypeL>& Solver_L);

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void ReBuildNumeric(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void SolveNonPrecond_(const VectorTypeH& rhs, VectorTypeH* x);
        virtual void SolvePrecond_(const VectorTypeH& rhs, VectorTypeH* x);

        virtual void PrintStart_(void) const;
        virtual void PrintEnd_(void) const;

        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        Solver<OperatorTypeL, VectorTypeL, ValueTypeL>* Solver_L_;

        VectorTypeH r_h_;
        VectorTypeL r_l_;

        VectorTypeH* x_h_;
        VectorTypeL  d_l_;
        VectorTypeH  d_h_;

        const OperatorTypeH* op_h_;
        OperatorTypeL*       op_l_;
    };

} // namespace rocalution

#endif // ROCALUTION_MIXED_PRECISION_HPP_
