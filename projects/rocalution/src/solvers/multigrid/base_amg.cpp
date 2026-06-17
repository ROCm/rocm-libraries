/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "base_amg.hpp"
#include "../../utils/def.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"
#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"
#include "../iter_ctrl.hpp"

#include "../krylov/cg.hpp"
#include "../preconditioners/preconditioner.hpp"

#include "../../utils/log.hpp"

#include <list>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    BaseAMG<OperatorType, VectorType, ValueType>::BaseAMG()
    {
        log_debug(this, "BaseAMG::BaseAMG()", "default constructor");

        this->coarse_size_ = 300;

        // manual smoothers and coarse solver
        this->set_sm_ = false;
        this->set_s_  = false;

        // default smoother format
        this->sm_format_ = CSR;
        // default operator format
        this->op_format_ = CSR;
        // default operator block dimension
        this->op_blockdim_ = 1;

        // since hierarchy has not been built yet
        this->hierarchy_ = false;

        // initialize temp default smoother pointer
        this->sm_default_ = NULL;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    BaseAMG<OperatorType, VectorType, ValueType>::~BaseAMG()
    {
        log_debug(this, "BaseAMG::BaseAMG()", "destructor");

        this->Clear();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::SetCoarsestLevel(int coarse_size)
    {
        log_debug(this, "BaseAMG::SetCoarsestLevel()", coarse_size);

        assert(this->build_ == false);
        assert(this->hierarchy_ == false);
        assert(coarse_size > 1);

        this->coarse_size_ = coarse_size;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::SetManualSmoothers(bool sm_manual)
    {
        log_debug(this, "BaseAMG::SetManualSmoothers()", sm_manual);

        assert(this->build_ == false);

        this->set_sm_ = sm_manual;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::SetManualSolver(bool s_manual)
    {
        log_debug(this, "BaseAMG::SetManualSolver()", s_manual);

        assert(this->build_ == false);

        this->set_s_ = s_manual;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::SetDefaultSmootherFormat(
        unsigned int op_format)
    {
        log_debug(this, "BaseAMG::SetDefaultSmootherFormat()", op_format);

        assert(this->build_ == false);

        this->sm_format_ = op_format;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::SetOperatorFormat(unsigned int op_format,
                                                                         int          op_blockdim)
    {
        log_debug(this, "BaseAMG::SetOperatorFormat()", op_format, op_blockdim);

        this->op_format_   = op_format;
        this->op_blockdim_ = op_blockdim;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    int BaseAMG<OperatorType, VectorType, ValueType>::GetNumLevels(void)
    {
        assert(this->hierarchy_ != false);

        return this->levels_;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "BaseAMG::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);

        // Build hierarchy
        this->BuildHierarchy();

        // Build smoothers, if not passed by the user
        if(this->set_sm_ == false)
        {
            this->BuildSmoothers();
        }

        // Build coarse grid solver, if not passed by the user
        if(this->set_s_ == false)
        {
            // Coarse Grid Solver
            CG<OperatorType, VectorType, ValueType>* cgs
                = new CG<OperatorType, VectorType, ValueType>;

            // Set absolute tolerance to 0 to avoid issues with very small numbers
            cgs->Init(0.0, 1e-6, 1e+8, 1000);

            // No verbose output
            cgs->Verbose(0);

            this->solver_coarse_ = cgs;
        }

        // Initialize multigrid structures
        this->Initialize();

        // Convert operator to op_format
        if(this->op_format_ != CSR)
        {
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                this->op_level_[i]->ConvertTo(this->op_format_, this->op_blockdim_);
            }
        }

        this->build_ = true;

        log_debug(this, "BaseAMG::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::BuildHierarchy(void)
    {
        log_debug(this, "BaseAMG::BuildHierarchy()", " #*# begin");

        if(this->hierarchy_ == false)
        {
            assert(this->build_ == false);
            this->hierarchy_ = true;

            // AMG will use operators for inter grid transfers
            assert(this->op_ != NULL);
            assert(this->coarse_size_ > 0);

            if(this->op_->GetM() <= static_cast<int64_t>(this->coarse_size_))
            {
                // LCOV_EXCL_START
                LOG_INFO("Problem size too small for AMG, use Krylov solver instead");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }

            // Lists for the building procedure
            std::list<OperatorType*>     op_list_;
            std::list<OperatorType*>     restrict_list_;
            std::list<OperatorType*>     prolong_list_;
            std::list<LocalVector<int>*> trans_list_;

            this->levels_ = 1;

            // Build finest hierarchy
            op_list_.push_back(new OperatorType);
            restrict_list_.push_back(new OperatorType);
            prolong_list_.push_back(new OperatorType);
            trans_list_.push_back(new LocalVector<int>);

            op_list_.back()->CloneBackend(*this->op_);
            restrict_list_.back()->CloneBackend(*this->op_);
            prolong_list_.back()->CloneBackend(*this->op_);
            trans_list_.back()->CloneBackend(*this->op_);

            // Create prolongation and restriction operators
            bool success = this->Aggregate_(*this->op_,
                                            prolong_list_.back(),
                                            restrict_list_.back(),
                                            op_list_.back(),
                                            trans_list_.back());

            // The very first level is not allowed to fail
            if(success == false)
            {
                // LCOV_EXCL_START
                LOG_INFO("Could not build initial AMG level");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }

            ++this->levels_;

            while(op_list_.back()->GetM() > static_cast<int64_t>(this->coarse_size_))
            {
                // Add new list elements
                restrict_list_.push_back(new OperatorType);
                prolong_list_.push_back(new OperatorType);
                OperatorType* prev_op_ = op_list_.back();
                op_list_.push_back(new OperatorType);
                trans_list_.push_back(new LocalVector<int>);

                op_list_.back()->CloneBackend(*this->op_);
                restrict_list_.back()->CloneBackend(*this->op_);
                prolong_list_.back()->CloneBackend(*this->op_);
                trans_list_.back()->CloneBackend(*this->op_);

                bool success = this->Aggregate_(*prev_op_,
                                                prolong_list_.back(),
                                                restrict_list_.back(),
                                                op_list_.back(),
                                                trans_list_.back());

                // Check if aggregation was successful
                if(success == false)
                {
                    // Something went wrong with level creation
                    // Revert level and stop generating new levels
                    prolong_list_.back()->Clear();
                    restrict_list_.back()->Clear();
                    op_list_.back()->Clear();
                    trans_list_.back()->Clear();

                    delete prolong_list_.back();
                    delete restrict_list_.back();
                    delete op_list_.back();
                    delete trans_list_.back();

                    LOG_VERBOSE_INFO(2,
                                     "*** warning: BaseAMG::Build() Requested coarse operator "
                                     "size cannot be reached.");

                    break;
                }

                ++this->levels_;

                if(this->levels_ > 19)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: BaseAMG::Build() Current number of levels: "
                                         << this->levels_);
                }
            }

            // Allocate data structures
            this->op_level_          = new OperatorType*[this->levels_ - 1];
            this->restrict_op_level_ = new OperatorType*[this->levels_ - 1];
            this->prolong_op_level_  = new OperatorType*[this->levels_ - 1];
            this->trans_level_       = new LocalVector<int>*[this->levels_ - 1];

            typename std::list<OperatorType*>::iterator     op_it    = op_list_.begin();
            typename std::list<OperatorType*>::iterator     pro_it   = prolong_list_.begin();
            typename std::list<OperatorType*>::iterator     res_it   = restrict_list_.begin();
            typename std::list<LocalVector<int>*>::iterator trans_it = trans_list_.begin();

            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                this->op_level_[i] = *op_it;
                ++op_it;

                this->restrict_op_level_[i] = *res_it;
                ++res_it;

                this->prolong_op_level_[i] = *pro_it;
                ++pro_it;

                this->trans_level_[i] = *trans_it;
                ++trans_it;
            }
        }

        log_debug(this, "BaseAMG::BuildHierarchy()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::BuildSmoothers(void)
    {
        log_debug(this, "BaseAMG::BuildSmoothers()", " #*# begin");

        // Smoother for each level
        this->smoother_level_
            = new IterativeLinearSolver<OperatorType, VectorType, ValueType>*[this->levels_ - 1];
        this->sm_default_ = new Solver<OperatorType, VectorType, ValueType>*[this->levels_ - 1];

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            FixedPoint<OperatorType, VectorType, ValueType>* sm
                = new FixedPoint<OperatorType, VectorType, ValueType>;
            Jacobi<OperatorType, VectorType, ValueType>* jac
                = new Jacobi<OperatorType, VectorType, ValueType>;

            sm->SetRelaxation(static_cast<ValueType>(2.f / 3.f));
            sm->SetPreconditioner(*jac);
            sm->Verbose(0);
            this->smoother_level_[i] = sm;
            this->sm_default_[i]     = jac;
        }

        log_debug(this, "BaseAMG::BuildSmoothers()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "BaseAMG::Clear()", this->build_);

        if(this->build_ == true)
        {
            // Clear AMG specific data
            this->ClearLocal();

            // Uninitialize multigrid structures
            this->Finalize();

            // De-allocate operator data structures
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                // Clear operator data structure
                delete this->op_level_[i];
                delete this->restrict_op_level_[i];
                delete this->prolong_op_level_[i];
            }

            delete[] this->op_level_;
            delete[] this->restrict_op_level_;
            delete[] this->prolong_op_level_;

            // De-allocate smoothers, if not allocated by the user
            if(this->set_sm_ == false)
            {
                for(int i = 0; i < this->levels_ - 1; ++i)
                {
                    delete this->smoother_level_[i];
                    delete this->sm_default_[i];
                }

                delete[] this->smoother_level_;
                delete[] this->sm_default_;
            }

            // De-allocate coarse grid solver, if not allocated by user
            if(this->set_s_ == false)
            {
                delete this->solver_coarse_;
            }

            this->levels_    = -1;
            this->build_     = false;
            this->hierarchy_ = false;
        }
    }

    // do nothing
    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::ClearLocal(void)
    {
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::SetRestrictOperator(OperatorType** op)
    {
        LOG_INFO(
            "BaseAMG::SetRestrictOperator() Perhaps you want to use the MultiGrid class to set "
            "external restriction operators");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::SetProlongOperator(OperatorType** op)
    {
        LOG_INFO("BaseAMG::SetProlongOperator() Perhaps you want to use the MultiGrid class to set "
                 "external prolongation operators");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseAMG<OperatorType, VectorType, ValueType>::SetOperatorHierarchy(OperatorType** op)
    {
        LOG_INFO(
            "BaseAMG::SetOperatorHierarchy() Perhaps you want to use the MultiGrid class to set "
            "external operators");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template class BaseAMG<LocalMatrix<double>, LocalVector<double>, double>;
    template class BaseAMG<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class BaseAMG<LocalMatrix<std::complex<double>>,
                           LocalVector<std::complex<double>>,
                           std::complex<double>>;
    template class BaseAMG<LocalMatrix<std::complex<float>>,
                           LocalVector<std::complex<float>>,
                           std::complex<float>>;
#endif

    template class BaseAMG<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class BaseAMG<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class BaseAMG<GlobalMatrix<std::complex<double>>,
                           GlobalVector<std::complex<double>>,
                           std::complex<double>>;
    template class BaseAMG<GlobalMatrix<std::complex<float>>,
                           GlobalVector<std::complex<float>>,
                           std::complex<float>>;
#endif

} // namespace rocalution
