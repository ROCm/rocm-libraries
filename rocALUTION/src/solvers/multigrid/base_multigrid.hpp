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

#ifndef ROCALUTION_BASE_MULTIGRID_HPP_
#define ROCALUTION_BASE_MULTIGRID_HPP_

#include "../../base/operator.hpp"
#include "../solver.hpp"
#include "rocalution/export.hpp"

namespace rocalution
{

    enum _cycle
    {
        Vcycle = 0,
        Wcycle = 1,
        Kcycle = 2,
        Fcycle = 3
    };

    /** \ingroup solver_module
  * \class BaseMultiGrid
  * \brief Base class for all multigrid solvers
  * \cite Trottenberg2003
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class BaseMultiGrid : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
    {
    public:
        BaseMultiGrid();
        virtual ~BaseMultiGrid();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        /** \private */
        ROCALUTION_EXPORT
        virtual void SetPreconditioner(Solver<OperatorType, VectorType, ValueType>& precond);

        /** \brief Set the coarse grid solver */
        ROCALUTION_EXPORT
        void SetSolver(Solver<OperatorType, VectorType, ValueType>& solver);

        /** \brief Set the smoother for each level */
        ROCALUTION_EXPORT
        void SetSmoother(IterativeLinearSolver<OperatorType, VectorType, ValueType>** smoother);

        /** \brief Set the number of pre-smoothing steps */
        ROCALUTION_EXPORT
        void SetSmootherPreIter(int iter);

        /** \brief Set the number of post-smoothing steps */
        ROCALUTION_EXPORT
        void SetSmootherPostIter(int iter);

        /** \brief Set the restriction operator for each level */
        virtual void SetRestrictOperator(OperatorType** op) = 0;

        /** \brief Set the prolongation operator for each level */
        virtual void SetProlongOperator(OperatorType** op) = 0;

        /** \brief Set the operator for each level */
        virtual void SetOperatorHierarchy(OperatorType** op) = 0;

        /** \brief Enable/disable scaling of intergrid transfers */
        ROCALUTION_EXPORT
        void SetScaling(bool scaling);

        /** \brief Force computation of coarser levels on the host backend */
        ROCALUTION_EXPORT
        void SetHostLevels(int levels);

        /** \brief Set the MultiGrid Cycle (default: Vcycle) */
        ROCALUTION_EXPORT
        void SetCycle(unsigned int cycle);

        /** \brief Set the MultiGrid Kcycle on all levels or only on finest level */
        ROCALUTION_EXPORT
        void SetKcycleFull(bool kcycle_full);

        /** \brief Set the depth of the multigrid solver */
        ROCALUTION_EXPORT
        void InitLevels(int levels);

        /** \brief Perform one Vcycle solve iteration */
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        /** \brief Build multigrid solver */
        virtual void Build(void);
        /** \brief Initialize multigrid solver (called from Build()) */
        virtual void Initialize(void);
        /** \brief Clear multigrid solver */
        virtual void Clear(void);
        /** \brief Finalize multigrid solver (called from Clear()) */
        virtual void Finalize(void);

    protected:
        /** \brief Restricts a given fine vector to a coarse vector */
        void Restrict_(const VectorType& fine, VectorType* coarse);

        /** \brief Prolongs a given coarse vector to a fine vector */
        void Prolong_(const VectorType& coarse, VectorType* fine);

        /** \brief V-cycle */
        void Vcycle_(const VectorType& rhs, VectorType* x);
        /** \brief W-cycle */
        void Wcycle_(const VectorType& rhs, VectorType* x);
        /** \brief F-cycle */
        void Fcycle_(const VectorType& rhs, VectorType* x);
        /** \brief K-cycle */
        void Kcycle_(const VectorType& rhs, VectorType* x);

        /** \private */
        virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);

        /** \private */
        virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

        virtual void PrintStart_(void) const;
        virtual void PrintEnd_(void) const;

        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

        /** \brief Move all level data to the host */
        void MoveHostLevels_(void);

        /** \brief Number of levels in the hierarchy */
        int levels_;
        /** \brief Host levels */
        int host_level_;
        /** \brief Current level */
        int current_level_;
        /** \brief Intergrid transfer scaling */
        bool scaling_;
        /** \brief Number of pre-smoothing steps */
        int iter_pre_smooth_;
        /** \brief Number of post-smoothing steps */
        int iter_post_smooth_;
        /** \brief Cycle type */
        unsigned int cycle_;
        /** \brief K-cycle type */
        bool kcycle_full_;

        /** \brief Residual norm */
        double res_norm_;

        /** \brief Operator hierarchy */
        OperatorType** op_level_;

        /** \brief Restriction operator hierarchy */
        OperatorType** restrict_op_level_;
        /** \brief Prolongation operator hierarchy */
        OperatorType** prolong_op_level_;

        VectorType** d_level_; /**< \private */
        VectorType** r_level_; /**< \private */
        VectorType** t_level_; /**< \private */
        VectorType** s_level_; /**< \private */
        VectorType** q_level_; /**< \private */

        /** \brief Transfer mapping */
        LocalVector<int>** trans_level_;

        /** \brief Coarse grid solver */
        Solver<OperatorType, VectorType, ValueType>* solver_coarse_;

        /** \brief Smoother for each level */
        IterativeLinearSolver<OperatorType, VectorType, ValueType>** smoother_level_;
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_MULTIGRID_HPP_
