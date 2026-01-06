/*! \file */
/* ************************************************************************
* Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse_arguments.hpp"
#include "program_options.hpp"

struct hipsparse_arguments_config : Arguments
{

public:
    char        precision{};
    char        indextype{};
    int         device_id{};
    std::string function_name{};

private:
    std::string b_filename{};
    char        b_transA{};
    char        b_transB{};
    int         b_baseA{};
    int         b_baseB{};
    int         b_baseC{};
    int         b_baseD{};
    int         b_action{};
    int         b_part{};
    int         b_dir{};
    int         b_orderA{};
    int         b_orderB{};
    int         b_orderC{};
    int         b_formatA{};
    int         b_formatB{};

    int b_csr2csc_alg{};
    int b_dense2sparse_alg{};
    int b_sparse2dense_alg{};
    int b_sddmm_alg{};
    int b_spgemm_alg{};
    int b_spmm_alg{};
    int b_spmv_alg{};
    int b_spsm_alg{};
    int b_spsv_alg{};

    char b_diag{};
    char b_uplo{};
    char b_spol{};

public:
    hipsparse_arguments_config();
    void set_description(options_description& desc);
    int  parse(int& argc, char**& argv, options_description& desc);
    int  parse_no_default(int& argc, char**& argv, options_description& desc);
};
