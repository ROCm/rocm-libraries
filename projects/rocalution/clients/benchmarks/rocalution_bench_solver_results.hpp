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

#include "rocalution_enum_itsolver.hpp"
#include "rocalution_enum_preconditioner.hpp"
#include "rocalution_enum_smoother.hpp"
#include <iomanip>

struct rocalution_bench_solver_results
{
    // clang-format off
#define RESBOOL_TRANSFORM_EACH			\
  RESBOOL_TRANSFORM(convergence)
    // clang-format on

#define RESBOOL_TRANSFORM(x_) x_,
    typedef enum e_bool_ : int
    {
        RESBOOL_TRANSFORM_EACH
    } e_bool;

    static constexpr e_bool e_bool_all[] = {RESBOOL_TRANSFORM_EACH};
#undef RESBOOL_TRANSFORM

    // clang-format off
#define RESINT_TRANSFORM_EACH			\
  RESINT_TRANSFORM(iter)
    // clang-format on

#define RESINT_TRANSFORM(x_) x_,
    typedef enum e_int_ : int
    {
        RESINT_TRANSFORM_EACH
    } e_int;

    static constexpr e_int e_int_all[] = {RESINT_TRANSFORM_EACH};
#undef RESINT_TRANSFORM

    // clang-format off
#define RESDOUBLE_TRANSFORM_EACH					\
  RESDOUBLE_TRANSFORM(time_import)					\
  RESDOUBLE_TRANSFORM(time_analyze)					\
  RESDOUBLE_TRANSFORM(time_solve)					\
  RESDOUBLE_TRANSFORM(time_global)					\
  RESDOUBLE_TRANSFORM(norm_residual)					\
  RESDOUBLE_TRANSFORM(nrmmax_err)					\
  RESDOUBLE_TRANSFORM(nrmmax95_err)					\
  RESDOUBLE_TRANSFORM(nrmmax75_err)					\
  RESDOUBLE_TRANSFORM(nrmmax50_err)
    // clang-format on

#define RESDOUBLE_TRANSFORM(x_) x_,
    typedef enum e_double_ : int
    {
        RESDOUBLE_TRANSFORM_EACH
    } e_double;
    static constexpr e_double e_double_all[] = {RESDOUBLE_TRANSFORM_EACH};
#undef RESDOUBLE_TRANSFORM

private:
    static constexpr std::size_t e_bool_size   = countof(e_bool_all);
    static constexpr std::size_t e_int_size    = countof(e_int_all);
    static constexpr std::size_t e_double_size = countof(e_double_all);

#define RESBOOL_TRANSFORM(x_) #x_,
    static constexpr const char* e_bool_names[e_bool_size]{RESBOOL_TRANSFORM_EACH};
#undef RESBOOL_TRANSFORM

#define RESINT_TRANSFORM(x_) #x_,
    static constexpr const char* e_int_names[e_int_size]{RESINT_TRANSFORM_EACH};
#undef RESINT_TRANSFORM

#define RESDOUBLE_TRANSFORM(x_) #x_,
    static constexpr const char* e_double_names[e_double_size]{RESDOUBLE_TRANSFORM_EACH};
#undef RESDOUBLE_TRANSFORM
    bool   bool_values[e_bool_size]{};
    int    int_values[e_int_size]{};
    double double_values[e_double_size]{};

public:
    static const char* Name(e_bool v)
    {
        return e_bool_names[v];
    };
    static const char* Name(e_double v)
    {
        return e_double_names[v];
    };
    static const char* Name(e_int v)
    {
        return e_int_names[v];
    };

    bool Get(e_bool v) const;
    void Set(e_bool v, bool s);

    int  Get(e_int v) const;
    void Set(e_int v, int s);

    double Get(e_double v) const;
    void   Set(e_double v, double s);

    void Info(std::ostream& out) const
    {
        out << "bool:  " << std::endl;
        for(auto e : e_bool_all)
        {
            out << std::setw(20) << e_bool_names[e] << std::setw(20) << bool_values[e] << std::endl;
        }
        out << "int:  " << std::endl;
        for(auto e : e_int_all)
        {
            out << std::setw(20) << e_int_names[e] << std::setw(20) << int_values[e] << std::endl;
        }
        out << "double:  " << std::endl;
        for(auto e : e_double_all)
        {
            out << std::setw(20) << e_double_names[e] << std::setw(20) << double_values[e]
                << std::endl;
        }
    }

    void WriteJson(std::ostream& out) const;
    void WriteNicely(std::ostream& out) const;
};
