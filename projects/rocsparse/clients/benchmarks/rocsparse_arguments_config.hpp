/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <iostream>
#include <vector>

#include "program_options.hpp"
#include "rocsparse_arguments.hpp"

struct rocsparse_arguments_config : Arguments
{

public:
    char          precision{};
    char          indextype{};
    std::string   function_name{};
    rocsparse_int device_id{};

private:
    std::string   b_matrixmarket{};
    std::string   b_mlcsr{};
    std::string   b_mlbsr{};
    std::string   b_rocalution{};
    std::string   b_rocsparseio{};
    std::string   b_file{};
    std::string   b_matrices_dir{};
    char          b_transA{};
    char          b_transB{};
    int           b_baseA{};
    int           b_baseB{};
    int           b_baseC{};
    int           b_baseD{};
    int           b_action{};
    int           b_part{};
    int           b_matrix_type{};
    char          b_diag{};
    char          b_uplo{};
    int           b_storage{};
    char          b_apol{};
    rocsparse_int b_dir{};
    rocsparse_int b_order{};
    rocsparse_int b_orderB{};
    rocsparse_int b_orderC{};
    rocsparse_int b_formatA{};
    rocsparse_int b_formatB{};
    rocsparse_int b_formatC{};
    rocsparse_int b_itilu0_alg{};
    rocsparse_int b_spmv_alg{};
    rocsparse_int b_spmm_alg{};
    rocsparse_int b_sddmm_alg{};
    rocsparse_int b_gtsv_interleaved_alg{};
#ifdef ROCSPARSE_WITH_MEMSTAT
    std::string b_memory_report_filename{};
#endif
public:
    rocsparse_arguments_config();
    void set_description(options_description& desc);
    int  parse(int& argc, char**& argv, options_description& desc);
    int  parse_no_default(int& argc, char**& argv, options_description& desc);
};
