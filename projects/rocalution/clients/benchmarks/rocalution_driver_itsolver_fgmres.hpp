/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocalution_driver_itsolver.hpp"

template <typename T>
struct rocalution_driver_itsolver<rocalution_enum_itsolver::fgmres, T>
    : rocalution_driver_itsolver_default<rocalution_enum_itsolver::fgmres, T>
{

    static constexpr auto ITSOLVER = rocalution_enum_itsolver::fgmres;
    using traits_t                 = rocalution_driver_itsolver_traits<ITSOLVER, T>;
    using solver_t                 = typename traits_t::solver_t;
    using params_t                 = rocalution_bench_solver_parameters;

    virtual bool PreprocessSolverBuild(LocalMatrix<T>& A,
                                       LocalVector<T>& B,
                                       LocalVector<T>& X,
                                       solver_t&       solver,
                                       const params_t& parameters) override
    {
        const auto krylov_basis = parameters.Get(params_t::krylov_basis);
        solver.SetBasisSize(krylov_basis);
        return true;
    };
};
