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

#include "rocalution_bench_solver_template_base.hpp"
#include "rocalution_enum_directsolver.hpp"

using namespace rocalution;

template <rocalution_enum_directsolver::value_type DIRECTSOLVER, typename T>
struct rocalution_enum_directsolver_traits;

template <typename T>
struct rocalution_enum_directsolver_traits<rocalution_enum_directsolver::lu, T>
{
    using solver_t = LU<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_enum_directsolver_traits<rocalution_enum_directsolver::qr, T>
{
    using solver_t = QR<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_enum_directsolver_traits<rocalution_enum_directsolver::inversion, T>
{
    using solver_t = Inversion<LocalMatrix<T>, LocalVector<T>, T>;
};
