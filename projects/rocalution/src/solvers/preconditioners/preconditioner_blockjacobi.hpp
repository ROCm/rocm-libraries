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

#ifndef ROCALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_
#define ROCALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_

#include "preconditioner.hpp"
#include "rocalution/export.hpp"

namespace rocalution
{

    /** \ingroup precond_module
  * \class BlockJacobi
  * \brief Block-Jacobi Preconditioner
  * \details
  * The Block-Jacobi preconditioner is designed to wrap any local preconditioner and
  * apply it in a global block fashion locally on each interior matrix.
  *
  * \tparam OperatorType - can be GlobalMatrix
  * \tparam VectorType - can be GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class BlockJacobi : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        BlockJacobi();
        ROCALUTION_EXPORT
        virtual ~BlockJacobi();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        /** \brief Set local preconditioner */
        ROCALUTION_EXPORT
        void Set(Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>& precond);

        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        ROCALUTION_EXPORT
        virtual void SolveZeroSol(const VectorType& rhs, VectorType* x);

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void ReBuildNumeric(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>* local_precond_;
    };

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_
