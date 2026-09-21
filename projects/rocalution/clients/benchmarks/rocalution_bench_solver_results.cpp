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
#include "rocalution_bench_solver_results.hpp"

constexpr rocalution_bench_solver_results::e_bool   rocalution_bench_solver_results::e_bool_all[];
constexpr rocalution_bench_solver_results::e_int    rocalution_bench_solver_results::e_int_all[];
constexpr rocalution_bench_solver_results::e_double rocalution_bench_solver_results::e_double_all[];

constexpr const char* rocalution_bench_solver_results::e_bool_names[];
constexpr const char* rocalution_bench_solver_results::e_int_names[];
constexpr const char* rocalution_bench_solver_results::e_double_names[];

bool rocalution_bench_solver_results::Get(e_bool v) const
{
    return bool_values[v];
}
void rocalution_bench_solver_results::Set(e_bool v, bool s)
{
    bool_values[v] = s;
}

int rocalution_bench_solver_results::Get(e_int v) const
{
    return int_values[v];
}
void rocalution_bench_solver_results::Set(e_int v, int s)
{
    int_values[v] = s;
}

double rocalution_bench_solver_results::Get(e_double v) const
{
    return double_values[v];
}
void rocalution_bench_solver_results::Set(e_double v, double s)
{
    double_values[v] = s;
}

void rocalution_bench_solver_results::WriteJson(std::ostream& out) const
{
    bool first = false;
    for(auto e : e_bool_all)
    {
        if(!first)
            first = true;
        else
            out << ",";
        out << "\"" << e_bool_names[e] << "\" : "
            << "\"" << bool_values[e] << "\"";
    }
    for(auto e : e_int_all)
    {
        if(!first)
            first = true;
        else
            out << ",";
        out << "\"" << e_int_names[e] << "\" : "
            << "\"" << int_values[e] << "\"";
    }
    for(auto e : e_double_all)
    {
        if(!first)
            first = true;
        else
            out << ",";
        out << "\"" << e_double_names[e] << "\" : "
            << "\"" << double_values[e] << "\"";
    }
}

void rocalution_bench_solver_results::WriteNicely(std::ostream& out) const
{
    out.precision(2);
    out.setf(std::ios::fixed);
    out.setf(std::ios::left);
    out << std::scientific;

    for(auto e : e_bool_all)
    {
        out << std::setw(14) << e_bool_names[e];
    }
    for(auto e : e_int_all)
    {
        out << std::setw(14) << e_int_names[e];
    }
    for(auto e : e_double_all)
    {
        out << std::setw(14) << e_double_names[e];
    }
    out << std::endl;
    for(auto e : e_bool_all)
    {
        out << std::setw(14) << bool_values[e];
    }
    for(auto e : e_int_all)
    {
        out << std::setw(14) << int_values[e];
    }
    for(auto e : e_double_all)
    {
        out << std::setw(14) << double_values[e];
    }
    out << std::endl;
}
