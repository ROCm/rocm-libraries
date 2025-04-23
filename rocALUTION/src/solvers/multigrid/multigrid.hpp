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

#ifndef ROCALUTION_MULTIGRID_HPP_
#define ROCALUTION_MULTIGRID_HPP_

#include "base_multigrid.hpp"
#include "rocalution/export.hpp"

namespace rocalution
{

    /** \ingroup solver_module
  * \class MultiGrid
  * \brief MultiGrid Method
  * \details
  * The MultiGrid method can be used with external data, such as externally computed
  * restriction, prolongation and operator hierarchy. The user need to pass all this
  * information for each level and for its construction. This includes smoothing step,
  * prolongation/restriction, grid traversing and coarse grid solver. This data need
  * to be passed to the solver.
  * \cite Trottenberg2003
  * - Restriction and prolongation operations can be performed in two ways, based on
  *   Restriction() and Prolongation() of the LocalVector class, or by matrix-vector
  *   multiplication. This is configured by a set function.
  * - Smoothers can be of any iterative linear solver. Valid options are Jacobi,
  *   Gauss-Seidel, ILU, etc. using a FixedPoint iteration scheme with pre-defined number
  *   of iterations. The smoothers could also be a solver such as CG, BiCGStab, etc.
  * - Coarse grid solver could be of any iterative linear solver type. The class also
  *   provides mechanisms to specify, where the coarse grid solver has to be performed,
  *   on the host or on the accelerator. The coarse grid solver can be preconditioned.
  * - Grid scaling based on a \f$L_2\f$ norm ratio.
  * - Operator matrices need to be passed on each grid level.
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class MultiGrid : public BaseMultiGrid<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        MultiGrid();
        ROCALUTION_EXPORT
        virtual ~MultiGrid();

        ROCALUTION_EXPORT
        virtual void SetRestrictOperator(OperatorType** op);
        ROCALUTION_EXPORT
        virtual void SetProlongOperator(OperatorType** op);
        ROCALUTION_EXPORT
        virtual void SetOperatorHierarchy(OperatorType** op);
    };

} // namespace rocalution

#endif // ROCALUTION_MULTIGRID_HPP_
