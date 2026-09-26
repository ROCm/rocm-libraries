/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_
#define ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"
#include "rocalution/export.hpp"

#include <vector>

namespace rocalution
{
    /** \ingroup solver_module
  * \class RugeStuebenAMG
  * \brief Ruge-Stueben Algebraic MultiGrid method.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class RugeStuebenAMG : public BaseAMG<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        RugeStuebenAMG();
        ROCALUTION_EXPORT
        virtual ~RugeStuebenAMG();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        /** \brief Set strength threshold */
        ROCALUTION_EXPORT
        void SetStrengthThreshold(float eps);

        /** \brief Set Coarsening strategy */
        ROCALUTION_EXPORT
        void SetCoarseningStrategy(CoarseningStrategy strat);

        /** \brief Set Interpolation type. */
        ROCALUTION_EXPORT
        void SetInterpolationType(InterpolationType type);

        /** \brief Enable FF1 interpolation limitation. This has no effect on MM
        * interpolation. */
        ROCALUTION_EXPORT
        void SetInterpolationFF1Limit(bool FF1);

        /** \brief Drop interpolation weights below \p factor times the largest magnitude
        * of their row and rescale the row to preserve its sum. \p factor must be in
        * [0, 1) (default: 0, no truncation). */
        ROCALUTION_EXPORT
        void SetInterpolationTruncationFactor(float factor);

        /** \brief Keep at most \p max_elmts weights per interpolation row and rescale the
        * row to preserve its sum (default: 0, unlimited). Only available for a single
        * process. */
        ROCALUTION_EXPORT
        void SetInterpolationMaxElmts(int max_elmts);

        ROCALUTION_EXPORT
        virtual void ReBuildNumeric(void);

    protected:
        virtual bool Aggregate_(const OperatorType& op,
                                OperatorType*       pro,
                                OperatorType*       res,
                                OperatorType*       coarse,
                                LocalVector<int>*   trans);

        virtual void PrintStart_(void) const;
        virtual void PrintEnd_(void) const;

    private:
        /** \brief Coupling strength */
        float eps_;

        /** \brief Flag to limit FF interpolation */
        bool FF1_;

        /** \brief Coarsening strategy */
        CoarseningStrategy coarsening_;

        /** \brief Interpolation type */
        InterpolationType interpolation_;

        /** \brief Interpolation truncation */
        float trunc_factor_;
        int   p_max_elmts_;
    };

} // namespace rocalution

#endif // ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_
