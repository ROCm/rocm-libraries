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

#ifndef ROCALUTION_PRECONDITIONER_MULTICOLORED_ILU_HPP_
#define ROCALUTION_PRECONDITIONER_MULTICOLORED_ILU_HPP_

#include "../../base/local_vector.hpp"
#include "../solver.hpp"
#include "preconditioner.hpp"
#include "preconditioner_multicolored.hpp"
#include "rocalution/export.hpp"

#include <vector>

namespace rocalution
{

    /** \ingroup precond_module
  * \class MultiColoredILU
  * \brief Multi-Colored Incomplete LU Factorization Preconditioner
  * \details
  * Multi-Colored Incomplete LU Factorization based on the ILU(p) factorization with a
  * power(q)-pattern method. This method provides a higher degree of parallelism of
  * forward and backward substitution compared to the standard ILU(p) preconditioner.
  * \cite Lukarski2012
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class MultiColoredILU : public MultiColored<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        MultiColoredILU();
        ROCALUTION_EXPORT
        virtual ~MultiColoredILU();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        ROCALUTION_EXPORT
        virtual void ReBuildNumeric(void);

        /** \brief Initialize a multi-colored ILU(p, p+1) preconditioner */
        ROCALUTION_EXPORT
        void Set(int p);

        /** \brief Initialize a multi-colored ILU(p, q) preconditioner
      * \details level = true will perform the factorization with levels <br>
      * level = false will perform the factorization only on the power(q)-pattern
      */
        ROCALUTION_EXPORT
        void Set(int p, int q, bool level = true);

    protected:
        virtual void Build_Analyser_(void);
        virtual void Factorize_(void);
        virtual void PostAnalyse_(void);

        virtual void SolveL_(void);
        virtual void SolveD_(void);
        virtual void SolveR_(void);
        virtual void Solve_(const VectorType& rhs, VectorType* x);

        /** \brief power(q) pattern parameter */
        int q_;
        /** \brief p-levels parameter */
        int p_;
        /** \brief Perform factorization with levels or not */
        bool level_;
        /** \brief Number of non-zeros */
        int64_t nnz_;
    };

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_MULTICOLORED_ILU_HPP_
