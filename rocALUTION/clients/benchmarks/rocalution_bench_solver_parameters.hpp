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

#include "rocalution_enum_coarsening_strategy.hpp"
#include "rocalution_enum_directsolver.hpp"
#include "rocalution_enum_itilu0_alg.hpp"
#include "rocalution_enum_itsolver.hpp"
#include "rocalution_enum_matrix_init.hpp"
#include "rocalution_enum_preconditioner.hpp"
#include "rocalution_enum_smoother.hpp"

#include <iomanip>

//
// @brief Structure responsible of the parameter definitions and values.
//
struct rocalution_bench_solver_parameters
{
protected:
    //
    // @brief Which matrix initialization.
    //
    rocalution_enum_matrix_init m_enum_matrix_init{};

    //
    // @brief Which ItILU0 algorithm.
    //
    rocalution_enum_itilu0_alg m_enum_itilu0_alg{};

    //
    // @brief Which iterative solver.
    //
    rocalution_enum_itsolver m_enum_itsolver{};

    //
    // @brief Which preconditioner.
    //
    rocalution_enum_preconditioner m_enum_preconditioner{};

    //
    // @brief Which smoother.
    //
    rocalution_enum_smoother m_enum_smoother{};

    //
    // @brief Which direct solver.
    //
    rocalution_enum_directsolver m_enum_directsolver{};

    //
    // @brief Which coarsening_strategy.
    //
    rocalution_enum_coarsening_strategy m_enum_coarsening_strategy{};

public:
    //
    // @brief Get which direct solver.
    //
    rocalution_enum_directsolver GetEnumDirectSolver() const;
    //
    // @brief Get which smoother.
    //
    rocalution_enum_smoother GetEnumSmoother() const;
    //
    // @brief Get which coarsening_strategy.
    //
    rocalution_enum_coarsening_strategy GetEnumCoarseningStrategy() const;
    //
    // @brief Get which preconditioner
    //
    rocalution_enum_preconditioner GetEnumPreconditioner() const;
    //
    // @brief Get which iterative solver.
    //
    rocalution_enum_itsolver GetEnumIterativeSolver() const;
    //
    // @brief Get which ItILU0 algorithm.
    //
    rocalution_enum_itilu0_alg GetEnumItILU0Algorithm() const;
    //
    // @brief Get which matrix initialization
    //
    rocalution_enum_matrix_init GetEnumMatrixInit() const;

    //
    // @brief Define Boolean parameters
    //

    // clang-format off
#define PBOOL_TRANSFORM_EACH			\
  PBOOL_TRANSFORM(iterative_solve) \
  PBOOL_TRANSFORM(verbose)			\
  PBOOL_TRANSFORM(mcilu_use_level)
    // clang-format on

#define PBOOL_TRANSFORM(x_) x_,
    typedef enum e_bool_ : int
    {
        PBOOL_TRANSFORM_EACH
    } e_bool;

    static constexpr e_bool e_bool_all[] = {PBOOL_TRANSFORM_EACH};
#undef PBOOL_TRANSFORM

    //
    // @brief Define Integer parameters
    //

    // clang-format off
#define PINT_TRANSFORM_EACH						\
  PINT_TRANSFORM(krylov_basis)						\
  PINT_TRANSFORM(ndim)							\
  PINT_TRANSFORM(ilut_n)						\
  PINT_TRANSFORM(itsolve_max_iter)      \
  PINT_TRANSFORM(itilu0_max_iter)				\
  PINT_TRANSFORM(itilu0_options)				\
  PINT_TRANSFORM(mcilu_p)						\
  PINT_TRANSFORM(mcilu_q)						\
  PINT_TRANSFORM(max_iter)						\
  PINT_TRANSFORM(solver_pre_smooth)					\
  PINT_TRANSFORM(solver_post_smooth)					\
  PINT_TRANSFORM(solver_ordering)					\
  PINT_TRANSFORM(rebuild_numeric)					\
  PINT_TRANSFORM(cycle)							\
  PINT_TRANSFORM(solver_coarsest_level)					\
  PINT_TRANSFORM(blockdim)

    // clang-format on

#define PINT_TRANSFORM(x_) x_,
    typedef enum e_int_ : int
    {
        PINT_TRANSFORM_EACH
    } e_int;

    static constexpr e_int e_int_all[] = {PINT_TRANSFORM_EACH};
#undef PINT_TRANSFORM

    //
    // @brief Define String parameters
    //

    // clang-format off
#define PSTRING_TRANSFORM_EACH						\
  PSTRING_TRANSFORM(coarsening_strategy)				\
  PSTRING_TRANSFORM(direct_solver)					\
  PSTRING_TRANSFORM(iterative_solver)					\
  PSTRING_TRANSFORM(itilu0_alg)					\
  PSTRING_TRANSFORM(matrix)						\
  PSTRING_TRANSFORM(matrix_filename)					\
  PSTRING_TRANSFORM(preconditioner)					\
  PSTRING_TRANSFORM(smoother)
    // clang-format on

#define PSTRING_TRANSFORM(x_) x_,
    typedef enum e_string_ : int
    {
        PSTRING_TRANSFORM_EACH
    } e_string;

    static constexpr e_string e_string_all[] = {PSTRING_TRANSFORM_EACH};
#undef PSTRING_TRANSFORM

    //
    // @brief Define Unsigned integer parameters
    //

    // clang-format off
#define PUINT_TRANSFORM_EACH					\
  PUINT_TRANSFORM(format)
    // clang-format on

#define PUINT_TRANSFORM(x_) x_,
    typedef enum e_uint_ : int
    {
        PUINT_TRANSFORM_EACH
    } e_uint;

    static constexpr e_uint e_uint_all[] = {PUINT_TRANSFORM_EACH};
#undef PUINT_TRANSFORM

    //
    // @brief Define Double parameters
    //

    // clang-format off
#define PDOUBLE_TRANSFORM_EACH				\
  PDOUBLE_TRANSFORM(abs_tol)				\
  PDOUBLE_TRANSFORM(rel_tol)				\
  PDOUBLE_TRANSFORM(div_tol)				\
  PDOUBLE_TRANSFORM(residual_tol)			\
  PDOUBLE_TRANSFORM(ilut_tol)				\
  PDOUBLE_TRANSFORM(itsolve_tol)    \
  PDOUBLE_TRANSFORM(itilu0_tol)				\
  PDOUBLE_TRANSFORM(mcgs_relax)				\
  PDOUBLE_TRANSFORM(solver_over_interp)			\
  PDOUBLE_TRANSFORM(solver_coupling_strength) \
    // clang-format on

#define PDOUBLE_TRANSFORM(x_) x_,
    typedef enum e_double_ : int
    {
        PDOUBLE_TRANSFORM_EACH
    } e_double;
    static constexpr e_double e_double_all[] = {PDOUBLE_TRANSFORM_EACH};
#undef PDOUBLE_TRANSFORM

private:
    //
    // @brief Number of string parameters
    //
    static constexpr std::size_t e_string_size = countof(e_string_all);
    //
    // @brief Number of unsigned integer parameters
    //
    static constexpr std::size_t e_uint_size = countof(e_uint_all);
    //
    // @brief Number of Boolean parameters
    //
    static constexpr std::size_t e_bool_size = countof(e_bool_all);
    //
    // @brief Number of integer parameters
    //
    static constexpr std::size_t e_int_size = countof(e_int_all);
    //
    // @brief Number of double parameters
    //
    static constexpr std::size_t e_double_size = countof(e_double_all);

    //
    // @brief Array of Boolean parameter names.
    //
#define PBOOL_TRANSFORM(x_) #x_,
    static constexpr const char* e_bool_names[e_bool_size]{PBOOL_TRANSFORM_EACH};
#undef PBOOL_TRANSFORM

    //
    // @brief Array of unsigned integer parameter names.
    //
#define PUINT_TRANSFORM(x_) #x_,
    static constexpr const char* e_uint_names[e_uint_size]{PUINT_TRANSFORM_EACH};
#undef PUINT_TRANSFORM

    //
    // @brief Array of string parameter names.
    //
#define PSTRING_TRANSFORM(x_) #x_,
    static constexpr const char* e_string_names[e_string_size]{PSTRING_TRANSFORM_EACH};
#undef PSTRING_TRANSFORM

    //
    // @brief Array of integer  parameter names.
    //
#define PINT_TRANSFORM(x_) #x_,
    static constexpr const char* e_int_names[e_int_size]{PINT_TRANSFORM_EACH};
#undef PINT_TRANSFORM

    //
    // @brief Array of Double parameter names.
    //
#define PDOUBLE_TRANSFORM(x_) #x_,
    static constexpr const char* e_double_names[e_double_size]{PDOUBLE_TRANSFORM_EACH};
#undef PDOUBLE_TRANSFORM

    //
    // @brief Array of Boolean parameter values.
    //
    bool bool_values[e_bool_size]{};

    //
    // @brief Array of Unsigned integer parameter values.
    //
    unsigned int uint_values[e_uint_size]{};

    //
    // @brief Array of Integer parameter values.
    //
    int int_values[e_int_size]{};

    //
    // @brief Array of string parameter values.
    //
    std::string string_values[e_string_size]{};

    //
    // @brief Array of Double parameter values.
    //
    double double_values[e_double_size]{};

public:
    static const char* GetName(e_bool v)
    {
        return e_bool_names[v];
    }
    static const char* GetName(e_int v)
    {
        return e_int_names[v];
    }
    static const char* GetName(e_uint v)
    {
        return e_uint_names[v];
    }
    static const char* GetName(e_double v)
    {
        return e_double_names[v];
    }
    static const char* GetName(e_string v)
    {
        return e_string_names[v];
    }

    //
    // @brief Get pointer to string parameter value.
    //
    std::string* GetPointer(e_string v);

    //
    // @brief Get string parameter value.
    //
    std::string Get(e_string v) const;

    //
    // @brief Set string parameter value.
    //
    void Set(e_string v, const std::string& s);

    //
    // @brief Get unsigned int parameter value.
    //
    unsigned int Get(e_uint v) const;
    //
    // @brief Get pointer to unsigned int parameter value.
    //
    unsigned int* GetPointer(e_uint v);

    //
    // @brief Set unsigned int parameter value.
    //
    void Set(e_uint v, unsigned int s);

    //
    // @brief Get Boolean parameter value.
    //
    bool Get(e_bool v) const;
    //
    // @brief Get pointer to Boolean parameter value.
    //
    bool* GetPointer(e_bool v);

    //
    // @brief Set Boolean parameter value.
    //
    void Set(e_bool v, bool s);

    //
    // @brief Get integer parameter value.
    //
    int Get(e_int v) const;

    //
    // @brief Get pointer to integer parameter value.
    //
    int* GetPointer(e_int v);

    //
    // @brief Set integer parameter value.
    //
    void Set(e_int v, int s);

    //
    // @brief Get double parameter value.
    //
    double Get(e_double v) const;

    //
    // @brief Get pointer to double parameter value.
    //
    double* GetPointer(e_double v);

    //
    // @brief Set double parameter value.
    //
    void Set(e_double v, double s);

    //
    // @brief Write information
    //
    void Info(std::ostream& out) const
    {
        out.setf(std::ios::left);
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
        out << "uint:  " << std::endl;
        for(auto e : e_uint_all)
        {
            out << std::setw(20) << e_uint_names[e] << std::setw(20) << uint_values[e] << std::endl;
        }
        out << "double:  " << std::endl;
        for(auto e : e_double_all)
        {
            out << std::setw(20) << e_double_names[e] << std::setw(20) << double_values[e]
                << std::endl;
        }
        out << "string:  " << std::endl;
        for(auto e : e_string_all)
        {
            out << std::setw(20) << e_string_names[e] << "'" << string_values[e] << "'"
                << std::endl;
        }
    }

    void WriteJson(std::ostream& out) const;
    void WriteNicely(std::ostream& out) const;
};
