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

#include "rocalution_bench_itsolver.hpp"
#include "rocalution_bench_solver_parameters.hpp"

using namespace rocalution;

template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_driver_itsolver_traits;

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::ruge_stueben_amg, T>
{
    using solver_t         = BiCGStab<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::saamg, T>
{
    using solver_t         = FCG<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::uaamg, T>
{
    using solver_t         = FCG<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = UAAMG<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::qmrcgstab, T>
{
    using solver_t         = QMRCGStab<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::pairwise_amg, T>
{
    using solver_t         = CG<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::idr, T>
{
    using solver_t         = IDR<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::fcg, T>
{
    using solver_t         = FCG<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::cr, T>
{
    using solver_t         = CR<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::cg, T>
{
    using solver_t         = CG<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::gmres, T>
{
    using solver_t         = GMRES<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::fgmres, T>
{
    using solver_t         = FGMRES<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::bicgstab, T>
{
    using solver_t         = BiCGStab<LocalMatrix<T>, LocalVector<T>, T>;
    using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};
