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

#ifndef ROCALUTION_BASE_AMG_HPP_
#define ROCALUTION_BASE_AMG_HPP_

#include "../solver.hpp"
#include "base_multigrid.hpp"
#include "rocalution/export.hpp"

#include <vector>

namespace rocalution
{
    typedef enum _coarsening_strategy
    {
        Greedy = 0,
        PMIS   = 1
    } CoarseningStrategy;

    typedef enum _interpolation_type
    {
        Direct = 0,
        ExtPI  = 1
    } InterpolationType;

    typedef enum _lumping_strategy
    {
        AddWeakConnections      = 0,
        SubtractWeakConnections = 1
    } LumpingStrategy;

    /** \ingroup solver_module
  * \class BaseAMG
  * \brief Base class for all algebraic multigrid solvers
  * \details
  * The Algebraic MultiGrid solver is based on the BaseMultiGrid class. The coarsening
  * is obtained by different aggregation techniques. The smoothers can be constructed
  * inside or outside of the class.
  *
  * All parameters in the Algebraic MultiGrid class can be set externally, including
  * smoothers and coarse grid solver.
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class BaseAMG : public BaseMultiGrid<OperatorType, VectorType, ValueType>
    {
    public:
        BaseAMG();
        virtual ~BaseAMG();

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Clear all local data */
        virtual void ClearLocal(void);

        /** \brief Create AMG hierarchy */
        ROCALUTION_EXPORT
        virtual void BuildHierarchy(void);

        /** \brief Create AMG smoothers */
        virtual void BuildSmoothers(void);

        /** \brief Set coarsest level for hierarchy creation */
        ROCALUTION_EXPORT
        void SetCoarsestLevel(int coarse_size);

        /** \brief Set flag to pass smoothers manually for each level */
        ROCALUTION_EXPORT
        void SetManualSmoothers(bool sm_manual);
        /** \brief Set flag to pass coarse grid solver manually */
        ROCALUTION_EXPORT
        void SetManualSolver(bool s_manual);

        /** \brief Set the smoother operator format */
        ROCALUTION_EXPORT
        void SetDefaultSmootherFormat(unsigned int op_format);
        /** \brief Set the operator format */
        ROCALUTION_EXPORT
        void SetOperatorFormat(unsigned int op_format, int op_blockdim);

        /** \brief Returns the number of levels in hierarchy */
        ROCALUTION_EXPORT
        int GetNumLevels(void);

        /** \private */
        virtual void SetRestrictOperator(OperatorType** op);
        /** \private */
        virtual void SetProlongOperator(OperatorType** op);
        /** \private */
        virtual void SetOperatorHierarchy(OperatorType** op);

    protected:
        /** \brief Constructs the prolongation, restriction and coarse operator */
        virtual bool Aggregate_(const OperatorType& op,
                                OperatorType*       pro,
                                OperatorType*       res,
                                OperatorType*       coarse,
                                LocalVector<int>*   trans)
            = 0;

        /** \brief Maximal coarse grid size */
        int coarse_size_;

        /** \brief Smoother is set manually or not */
        bool set_sm_;
        /** \brief Smoother hierarchy */
        Solver<OperatorType, VectorType, ValueType>** sm_default_;

        /** \brief Coarse grid solver is set manually or not */
        bool set_s_;

        /** \brief Build flag for hierarchy */
        bool hierarchy_;

        /** \brief Smoother operator format */
        unsigned int sm_format_;
        /** \brief Operator format */
        unsigned int op_format_;
        /** \brief Operator block dimension */
        int op_blockdim_;
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_AMG_HPP_
