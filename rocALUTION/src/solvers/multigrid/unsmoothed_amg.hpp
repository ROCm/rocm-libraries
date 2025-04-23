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

#ifndef ROCALUTION_UNSMOOTHED_AMG_HPP_
#define ROCALUTION_UNSMOOTHED_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"
#include "rocalution/export.hpp"

#include <vector>

namespace rocalution
{

    /** \ingroup solver_module
  * \class UAAMG
  * \brief Unsmoothed Aggregation Algebraic MultiGrid Method
  * \details
  * The Unsmoothed Aggregation Algebraic MultiGrid method is based on unsmoothed
  * aggregation based interpolation scheme.
  * \cite stueben
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class UAAMG : public BaseAMG<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        UAAMG();
        ROCALUTION_EXPORT
        virtual ~UAAMG();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        /** \brief Set coupling strength */
        ROCALUTION_EXPORT
        void SetCouplingStrength(ValueType eps);
        /** \brief Set over-interpolation parameter for aggregation */
        ROCALUTION_EXPORT
        void SetOverInterp(ValueType overInterp);

        /** \brief Set Coarsening strategy */
        ROCALUTION_EXPORT
        void SetCoarseningStrategy(CoarseningStrategy strat);

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
        ValueType eps_;

        /** \brief Over-interpolation parameter for aggregation */
        ValueType over_interp_;

        /** \brief Coarsening strategy */
        CoarseningStrategy strat_;
    };

} // namespace rocalution

#endif // ROCALUTION_UNSMOOTHED_AMG_HPP_
