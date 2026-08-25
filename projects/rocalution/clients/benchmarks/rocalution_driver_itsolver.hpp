/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocalution_driver_itsolver_template.hpp"

//
// @brief This structure is responsible of providing a base implementation
// of \ref rocalution_driver_itsolver_template.
//
template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_driver_itsolver_base : rocalution_driver_itsolver_template<ITSOLVER, T>
{
protected:
    using traits_t         = rocalution_driver_itsolver_traits<ITSOLVER, T>;
    using solver_t         = typename traits_t::solver_t;
    using preconditioner_t = typename traits_t::preconditioner_t;
    using params_t         = rocalution_bench_solver_parameters;

protected:
    T*                cache_csr_val{};
    preconditioner_t* m_preconditioner{};

    //
    // Option for caching.
    //
    void Cache(int             m,
               int             n,
               int             nnz,
               const int*      csr_ptr,
               const int*      csr_ind,
               const T*        csr_val,
               const params_t& parameters)
    {
        this->cache_csr_val = new T[nnz];
        for(int i = 0; i < nnz; i++)
        {
            this->cache_csr_val[i] = csr_val[i];
        }
    }

public:
    //
    // @brief Destructor.
    //
    ~rocalution_driver_itsolver_base()
    {
        if(this->cache_csr_val != nullptr)
        {
            delete[] this->cache_csr_val;
            this->cache_csr_val = nullptr;
        }
        if(this->m_preconditioner != nullptr)
        {
            delete m_preconditioner;
            m_preconditioner = nullptr;
        }
    }

    virtual const preconditioner_t* GetPreconditioner() const override
    {
        return this->m_preconditioner;
    };

    virtual preconditioner_t* GetPreconditioner() override
    {
        return this->m_preconditioner;
    };

    virtual void SetPreconditioner(preconditioner_t* preconditioner) override
    {
        this->m_preconditioner = preconditioner;
    };

    // @brief Import matrix.
    // @param[in]
    // A           local matrix.
    // @param[in]
    // parameters  parameters to configure the operation.
    // @return true if successful, false otherwise.
    virtual bool ImportMatrix(LocalMatrix<T>& A, const params_t& parameters)
    {
        auto matrix_init = parameters.GetEnumMatrixInit();
        if(matrix_init.is_invalid())
        {
            rocalution_bench_errmsg << "matrix initialization is invalid" << std::endl;
            return false;
        }

        const auto rebuild_numeric = parameters.Get(params_t::rebuild_numeric);
        switch(matrix_init.value)
        {

        case rocalution_enum_matrix_init::laplacian:
        {
            int*      csr_ptr = NULL;
            int*      csr_col = NULL;
            T*        csr_val = NULL;
            const int ndim    = parameters.Get(params_t::ndim);
            auto      nrow    = gen_2d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
            if(rebuild_numeric)
            {
                this->Cache(nrow, nrow, csr_ptr[nrow], csr_ptr, csr_col, csr_val, parameters);
            }
            A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", csr_ptr[nrow], nrow, nrow);
            return true;
        }

        case rocalution_enum_matrix_init::permuted_identity:
        {
            int*      csr_ptr = NULL;
            int*      csr_col = NULL;
            T*        csr_val = NULL;
            const int ndim    = parameters.Get(params_t::ndim);
            auto      nrow    = gen_permuted_identity(ndim, &csr_ptr, &csr_col, &csr_val);
            if(rebuild_numeric)
            {
                this->Cache(nrow, nrow, csr_ptr[nrow], csr_ptr, csr_col, csr_val, parameters);
            }
            A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", csr_ptr[nrow], nrow, nrow);
            return true;
        }

        case rocalution_enum_matrix_init::file:
        {
            const std::string matrix_filename = parameters.Get(params_t::matrix_filename);
            if(matrix_filename == "")
            {
                rocalution_bench_errmsg << "no filename for matrix file initialization."
                                        << std::endl;
                return false;
            }
            else
            {
                A.ReadFileMTX(matrix_filename);

                // Apply reordering ?
                // LocalVector<int> rcmk;
                // A.RCMK(&rcmk);
                // A.Permute(rcmk);

                if(rebuild_numeric)
                {
                    //
                    // Since we want to cache
                    //
                    // Convert to csr format.
                    //
                    A.ConvertTo(1);

                    //
                    // Grab csr data.
                    //
                    int* csr_ptr = NULL;
                    int* csr_col = NULL;
                    T*   csr_val = NULL;
                    auto nrow    = A.GetM();
                    auto ncol    = A.GetN();
                    auto nnz     = A.GetNnz();
                    //
                    // Leave pointers, A is not usable after this until we re-set the pointers.
                    //
                    A.LeaveDataPtrCSR(&csr_ptr, &csr_col, &csr_val);

                    //
                    // Cache.
                    //
                    this->Cache(nrow, ncol, nnz, csr_ptr, csr_col, csr_val, parameters);

                    //
                    // Re-set the pointers.
                    //
                    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);
                }
                return true;
            }
        }
        }

        return true;
    }

    virtual bool ImportLinearSystem(LocalMatrix<T>& A,
                                    LocalVector<T>& B,
                                    LocalVector<T>& X,
                                    const params_t& parameters) override
    {
        //
        // Import matrix.
        //
        this->ImportMatrix(A, parameters);

        return true;
    }
};

//
// @Brief Default implementation.
//
template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_driver_itsolver_default : rocalution_driver_itsolver_base<ITSOLVER, T>
{
    using params_t = rocalution_bench_solver_parameters;

    virtual bool CreatePreconditioner(LocalMatrix<T>& A,
                                      LocalVector<T>& B,
                                      LocalVector<T>& X,
                                      const params_t& parameters) override
    {

        auto enum_preconditioner = parameters.GetEnumPreconditioner();
        if(enum_preconditioner.is_invalid())
        {
            rocalution_bench_errmsg << "enum preconditioner is invalid." << std::endl;
            return false;
        }
        this->m_preconditioner = nullptr;
        switch(enum_preconditioner.value)
        {
        case rocalution_enum_preconditioner::none:
        {
            return true;
        }
        case rocalution_enum_preconditioner::chebyshev:
        {
            // Chebyshev preconditioner

            // Determine min and max eigenvalues
            T lambda_min;
            T lambda_max;

            A.Gershgorin(lambda_min, lambda_max);

            auto* p = new rocalution::
                AIChebyshev<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            p->Set(3, lambda_max / 7.0, lambda_max);
            this->m_preconditioner = p;
            break;
        }

        case rocalution_enum_preconditioner::FSAI:
        {
            auto* p
                = new rocalution::FSAI<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            this->m_preconditioner = p;
            break;
        }

        case rocalution_enum_preconditioner::SPAI:
        {
            auto* p
                = new rocalution::SPAI<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            this->m_preconditioner = p;
            break;
        }
        case rocalution_enum_preconditioner::TNS:
        {
            auto* p
                = new rocalution::TNS<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            this->m_preconditioner = p;
            break;
        }

        case rocalution_enum_preconditioner::Jacobi:
        {
            auto* p
                = new rocalution::Jacobi<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            //
            // no specific parameters
            //
            this->m_preconditioner = p;
            break;
        }

        case rocalution_enum_preconditioner::GS:
        {
            auto* p = new rocalution::GS<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            //
            // no specific parameters
            //
            this->m_preconditioner = p;
            break;
        }

        case rocalution_enum_preconditioner::SGS:
        {
            auto* p
                = new rocalution::SGS<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            //
            // no specific parameters
            //
            this->m_preconditioner = p;
            break;
        }

        case rocalution_enum_preconditioner::ILU:
        {
            auto* p
                = new rocalution::ILU<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            //
            // no specific parameters
            //
            this->m_preconditioner = p;
            break;
        }
        case rocalution_enum_preconditioner::ItILU0:
        {
            auto enum_itilu0_alg = parameters.GetEnumItILU0Algorithm();
            if(enum_itilu0_alg.is_invalid())
            {
                rocalution_bench_errmsg << "enum_itilu0_alg is invalid." << std::endl;
                return false;
            }

            auto* p
                = new rocalution::ItILU0<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            p->SetMaxIter(parameters.Get(params_t::itilu0_max_iter));
            p->SetTolerance(parameters.Get(params_t::itilu0_tol));
            p->SetOptions(parameters.Get(params_t::itilu0_options));

            switch(enum_itilu0_alg.value)
            {
            case rocalution_enum_itilu0_alg::Default:
            {
                p->SetAlgorithm(ItILU0Algorithm::Default);
                break;
            }
            case rocalution_enum_itilu0_alg::AsyncInPlace:
            {
                p->SetAlgorithm(ItILU0Algorithm::AsyncInPlace);
                break;
            }
            case rocalution_enum_itilu0_alg::AsyncSplit:
            {
                p->SetAlgorithm(ItILU0Algorithm::AsyncSplit);
                break;
            }
            case rocalution_enum_itilu0_alg::SyncSplit:
            {
                p->SetAlgorithm(ItILU0Algorithm::SyncSplit);
                break;
            }
            case rocalution_enum_itilu0_alg::SyncSplitFusion:
            {
                p->SetAlgorithm(ItILU0Algorithm::SyncSplitFusion);
                break;
            }
            }

            this->m_preconditioner = p;
            return true;
        }
        case rocalution_enum_preconditioner::ILUT:
        {
            auto* p
                = new rocalution::ILUT<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            p->Set(parameters.Get(params_t::ilut_tol), parameters.Get(params_t::ilut_n));

            this->m_preconditioner = p;
            break;
        }
        case rocalution_enum_preconditioner::IC:
        {
            auto* p = new rocalution::IC<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            //
            // no specific parameters
            //
            this->m_preconditioner = p;
            break;
        }
        case rocalution_enum_preconditioner::MCGS:
        {
            auto* p = new rocalution::
                MultiColoredGS<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            p->SetRelaxation(parameters.Get(params_t::mcgs_relax));
            this->m_preconditioner = p;
            break;
        }
        case rocalution_enum_preconditioner::MCSGS:
        {
            auto* p = new rocalution::
                MultiColoredSGS<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;
            //
            // no specific parameters
            //
            this->m_preconditioner = p;
            break;
        }
        case rocalution_enum_preconditioner::MCILU:
        {
            auto* p = new rocalution::
                MultiColoredILU<rocalution::LocalMatrix<T>, rocalution::LocalVector<T>, T>;

            p->Set(parameters.Get(params_t::mcilu_p),
                   parameters.Get(params_t::mcilu_q),
                   parameters.Get(params_t::mcilu_use_level));

            this->m_preconditioner = p;
            break;
        }
        }

        if(this->m_preconditioner != nullptr)
        {
            const auto itsolve = parameters.Get(params_t::iterative_solve);

            rocalution::SolverDescr descr;

            if(itsolve)
            {
                descr.SetTriSolverAlg(TriSolverAlg_Iterative);
                descr.SetIterativeSolverMaxIteration(parameters.Get(params_t::itsolve_max_iter));
                descr.SetIterativeSolverTolerance(parameters.Get(params_t::itsolve_tol));
            }
            else
            {
                descr.SetTriSolverAlg(TriSolverAlg_Default);
            }

            this->m_preconditioner->SetSolverDescriptor(descr);

            return true;
        }

        return false;
    }
};

//
// @Brief Default definition.
//
template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_driver_itsolver : rocalution_driver_itsolver_default<ITSOLVER, T>
{
};
