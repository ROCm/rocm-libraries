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

/*! \file
 *  \brief hipsparse_arguments.hpp provides a class to parse command arguments in both,
 *  clients and gtest.
 */

#pragma once

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "hipsparse_arguments_support.hpp"
#include "hipsparse_datatype2string.hpp"

template <typename T>
static T convert_alpha_beta(double r, double i)
{
    return static_cast<T>(r);
}

struct Arguments
{
    int M;
    int N;
    int K;
    int nnz;
    int block_dim;
    int row_block_dimA;
    int col_block_dimA;
    int row_block_dimB;
    int col_block_dimB;

    int lda;
    int ldb;
    int ldc;

    int batch_count;

    hipsparseIndexType_t index_type_I;
    hipsparseIndexType_t index_type_J;
    hipDataType          compute_type;

    double alpha;
    double alphai;
    double beta;
    double betai;
    double threshold;
    double percentage;
    double c;
    double s;

    hipsparseOperation_t transA;
    hipsparseOperation_t transB;
    hipsparseIndexBase_t baseA;
    hipsparseIndexBase_t baseB;
    hipsparseIndexBase_t baseC;
    hipsparseIndexBase_t baseD;

    hipsparseAction_t       action;
    hipsparseHybPartition_t part;
    hipsparseDiagType_t     diag_type;
    hipsparseFillMode_t     fill_mode;
    hipsparseSolvePolicy_t  solve_policy;

    hipsparseDirection_t dirA;
    hipsparseOrder_t     orderA;
    hipsparseOrder_t     orderB;
    hipsparseOrder_t     orderC;
    hipsparseFormat_t    formatA;
    hipsparseFormat_t    formatB;

    hipsparseCsr2CscAlg_t       csr2csc_alg;
    hipsparseDenseToSparseAlg_t dense2sparse_alg;
    hipsparseSparseToDenseAlg_t sparse2dense_alg;
    hipsparseSDDMMAlg_t         sddmm_alg;
    hipsparseSpGEMMAlg_t        spgemm_alg;
    hipsparseSpMMAlg_t          spmm_alg;
    hipsparseSpMVAlg_t          spmv_alg;
    hipsparseSpSMAlg_t          spsm_alg;
    hipsparseSpSVAlg_t          spsv_alg;

    int    numericboost;
    double boosttol;
    double boostval;
    double boostvali;

    int ell_width;
    int permute;
    int gtsv_alg;
    int gpsv_alg;

    int unit_check;
    int timing;
    int iters;

    char filename[192]; // nos2.bin, bmwcra_1.bin, etc
    char function[64]; // axpby, spmv_csr, etc
    char category[32]; // quick, pre_checkin, etc

    Arguments()
    {
        this->M              = -1;
        this->N              = -1;
        this->K              = -1;
        this->nnz            = -1;
        this->block_dim      = 2;
        this->row_block_dimA = 2;
        this->col_block_dimA = 2;
        this->row_block_dimB = 2;
        this->col_block_dimB = 2;

        this->lda = -1;
        this->ldb = -1;
        this->ldc = -1;

        this->batch_count = -1;

        this->index_type_I = HIPSPARSE_INDEX_32I;
        this->index_type_J = HIPSPARSE_INDEX_32I;
        this->compute_type = HIP_R_32F;

        this->alpha      = 0.0;
        this->alphai     = 0.0;
        this->beta       = 0.0;
        this->betai      = 0.0;
        this->threshold  = 0.0;
        this->percentage = 0.0;
        this->c          = 1.0;
        this->s          = 1.0;

        this->transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        this->transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        this->baseA  = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseB  = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseC  = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseD  = HIPSPARSE_INDEX_BASE_ZERO;

        this->action       = HIPSPARSE_ACTION_NUMERIC;
        this->part         = HIPSPARSE_HYB_PARTITION_AUTO;
        this->diag_type    = HIPSPARSE_DIAG_TYPE_NON_UNIT;
        this->fill_mode    = HIPSPARSE_FILL_MODE_LOWER;
        this->solve_policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

        this->dirA    = HIPSPARSE_DIRECTION_ROW;
        this->orderA  = HIPSPARSE_ORDER_COL;
        this->orderB  = HIPSPARSE_ORDER_COL;
        this->orderC  = HIPSPARSE_ORDER_COL;
        this->formatA = HIPSPARSE_FORMAT_COO;
        this->formatB = HIPSPARSE_FORMAT_COO;

        this->csr2csc_alg      = csr2csc_alg_support::get_default_algorithm();
        this->dense2sparse_alg = dense2sparse_alg_support::get_default_algorithm();
        this->sparse2dense_alg = sparse2dense_alg_support::get_default_algorithm();
        this->sddmm_alg        = sddmm_alg_support::get_default_algorithm();
        this->spgemm_alg       = spgemm_alg_support::get_default_algorithm();
        this->spmm_alg         = spmm_alg_support::get_default_algorithm();
        this->spmv_alg         = spmv_alg_support::get_default_algorithm();
        this->spsm_alg         = spsm_alg_support::get_default_algorithm();
        this->spsv_alg         = spsv_alg_support::get_default_algorithm();

        this->numericboost = 0;
        this->boosttol     = 0.0;
        this->boostval     = 1.0;
        this->boostvali    = 0.0;

        this->ell_width = 0;
        this->permute   = 0;
        this->gtsv_alg  = 0;
        this->gpsv_alg  = 0;

        this->unit_check = 1;
        this->timing     = 0;
        this->iters      = 10;

        this->filename[0] = '\0';
        this->function[0] = '\0';
        this->category[0] = '\0';
    }

    template <typename T>
    T get_alpha() const
    {
        return convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return convert_alpha_beta<T>(beta, betai);
    }

    template <typename T>
    T get_threshold() const
    {
        return threshold;
    }

    template <typename T>
    T get_percentage() const
    {
        return percentage;
    }

    // Validate input format.
    // hipsparse_gentest.py is expected to conform to this format.
    // hipsparse_gentest.py uses hipsparse_common.yaml to generate this format.
    static void validate(std::istream& ifs)
    {
        auto error = [](auto name) {
            std::cerr << "Arguments field " << name << " does not match format.\n\n"
                      << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that hipsparse_arguments.hpp and hipsparse_common.yaml\n"
                         "define exactly the same Arguments, that hipsparse_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            abort();
        };

        char      header[10]{}, trailer[10]{};
        Arguments arg{};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));

        if(strcmp(header, "hipSPARSE"))
            error("header");
        else if(strcmp(trailer, "HIPsparse"))
            error("trailer");

        auto check_func = [&, sig = (uint8_t)0](const auto& elem, auto name) mutable {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(uint8_t i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const uint8_t*>(&elem)[i] ^ sig ^ i)
                    error(name);
            sig += 89;
        };

#define HIPSPARSE_FORMAT_CHECK(x) check_func(arg.x, #x)

        // Order is important
        HIPSPARSE_FORMAT_CHECK(M);
        HIPSPARSE_FORMAT_CHECK(N);
        HIPSPARSE_FORMAT_CHECK(K);
        HIPSPARSE_FORMAT_CHECK(nnz);
        HIPSPARSE_FORMAT_CHECK(block_dim);
        HIPSPARSE_FORMAT_CHECK(row_block_dimA);
        HIPSPARSE_FORMAT_CHECK(col_block_dimA);
        HIPSPARSE_FORMAT_CHECK(row_block_dimB);
        HIPSPARSE_FORMAT_CHECK(col_block_dimB);
        HIPSPARSE_FORMAT_CHECK(lda);
        HIPSPARSE_FORMAT_CHECK(ldb);
        HIPSPARSE_FORMAT_CHECK(ldc);
        HIPSPARSE_FORMAT_CHECK(batch_count);
        HIPSPARSE_FORMAT_CHECK(index_type_I);
        HIPSPARSE_FORMAT_CHECK(index_type_J);
        HIPSPARSE_FORMAT_CHECK(compute_type);
        HIPSPARSE_FORMAT_CHECK(alpha);
        HIPSPARSE_FORMAT_CHECK(alphai);
        HIPSPARSE_FORMAT_CHECK(beta);
        HIPSPARSE_FORMAT_CHECK(betai);
        HIPSPARSE_FORMAT_CHECK(threshold);
        HIPSPARSE_FORMAT_CHECK(percentage);
        HIPSPARSE_FORMAT_CHECK(c);
        HIPSPARSE_FORMAT_CHECK(s);
        HIPSPARSE_FORMAT_CHECK(transA);
        HIPSPARSE_FORMAT_CHECK(transB);
        HIPSPARSE_FORMAT_CHECK(baseA);
        HIPSPARSE_FORMAT_CHECK(baseB);
        HIPSPARSE_FORMAT_CHECK(baseC);
        HIPSPARSE_FORMAT_CHECK(baseD);
        HIPSPARSE_FORMAT_CHECK(action);
        HIPSPARSE_FORMAT_CHECK(part);
        HIPSPARSE_FORMAT_CHECK(diag_type);
        HIPSPARSE_FORMAT_CHECK(fill_mode);
        HIPSPARSE_FORMAT_CHECK(solve_policy);
        HIPSPARSE_FORMAT_CHECK(dirA);
        HIPSPARSE_FORMAT_CHECK(orderA);
        HIPSPARSE_FORMAT_CHECK(orderB);
        HIPSPARSE_FORMAT_CHECK(orderC);
        HIPSPARSE_FORMAT_CHECK(formatA);
        HIPSPARSE_FORMAT_CHECK(formatB);
        HIPSPARSE_FORMAT_CHECK(csr2csc_alg);
        HIPSPARSE_FORMAT_CHECK(dense2sparse_alg);
        HIPSPARSE_FORMAT_CHECK(sparse2dense_alg);
        HIPSPARSE_FORMAT_CHECK(sddmm_alg);
        HIPSPARSE_FORMAT_CHECK(spgemm_alg);
        HIPSPARSE_FORMAT_CHECK(spmm_alg);
        HIPSPARSE_FORMAT_CHECK(spmv_alg);
        HIPSPARSE_FORMAT_CHECK(spsm_alg);
        HIPSPARSE_FORMAT_CHECK(spsv_alg);
        HIPSPARSE_FORMAT_CHECK(numericboost);
        HIPSPARSE_FORMAT_CHECK(boosttol);
        HIPSPARSE_FORMAT_CHECK(boostval);
        HIPSPARSE_FORMAT_CHECK(boostvali);
        HIPSPARSE_FORMAT_CHECK(ell_width);
        HIPSPARSE_FORMAT_CHECK(permute);
        HIPSPARSE_FORMAT_CHECK(gtsv_alg);
        HIPSPARSE_FORMAT_CHECK(gpsv_alg);
        HIPSPARSE_FORMAT_CHECK(unit_check);
        HIPSPARSE_FORMAT_CHECK(timing);
        HIPSPARSE_FORMAT_CHECK(iters);
        HIPSPARSE_FORMAT_CHECK(filename);
        HIPSPARSE_FORMAT_CHECK(function);
        HIPSPARSE_FORMAT_CHECK(category);
    }

    void set_filename(const std::string& bin_file)
    {
        strncpy(this->filename, bin_file.c_str(), bin_file.length());
        this->filename[bin_file.length()] = '\0';
    }

private:
    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg)
    {
        str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return str;
    }

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream& str, double x)
    {
        if(std::isnan(x))
            str << ".nan";
        else if(std::isinf(x))
            str << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[]{c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream& str, const char* s)
    {
        str << std::quoted(s);
    }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

        print("filename", arg.filename);
        print("function", arg.function);
        print("category", arg.category);
        print("M", arg.M);
        print("N", arg.N);
        print("K", arg.K);
        print("nnz", arg.nnz);
        print("block_dim", arg.block_dim);
        print("row_block_dimA", arg.row_block_dimA);
        print("col_block_dimA", arg.col_block_dimA);
        print("row_block_dimB", arg.row_block_dimB);
        print("col_block_dimB", arg.col_block_dimB);
        print("lda", arg.lda);
        print("ldb", arg.ldb);
        print("ldc", arg.ldc);
        print("batch_count", arg.batch_count);
        print("index_type_I", hipsparse_indextype2string(arg.index_type_I));
        print("index_type_J", hipsparse_indextype2string(arg.index_type_J));
        print("compute_type", hipsparse_datatype2string(arg.compute_type));
        print("alpha", arg.alpha);
        print("alphai", arg.alphai);
        print("beta", arg.beta);
        print("betai", arg.betai);
        print("threshold", arg.threshold);
        print("percentage", arg.percentage);
        print("c", arg.c);
        print("s", arg.s);
        print("transA", hipsparse_operation2string(arg.transA));
        print("transB", hipsparse_operation2string(arg.transB));
        print("baseA", hipsparse_indexbase2string(arg.baseA));
        print("baseB", hipsparse_indexbase2string(arg.baseB));
        print("baseC", hipsparse_indexbase2string(arg.baseC));
        print("baseD", hipsparse_indexbase2string(arg.baseD));
        print("action", hipsparse_action2string(arg.action));
        print("part", hipsparse_partition2string(arg.part));
        print("diag_type", hipsparse_diagtype2string(arg.diag_type));
        print("fill_mode", hipsparse_fillmode2string(arg.fill_mode));
        print("solve_policy", hipsparse_solvepolicy2string(arg.solve_policy));
        print("dirA", hipsparse_direction2string(arg.dirA));
        print("orderA", hipsparse_order2string(arg.orderA));
        print("orderB", hipsparse_order2string(arg.orderB));
        print("orderC", hipsparse_order2string(arg.orderC));
        print("formatA", hipsparse_format2string(arg.formatA));
        print("formatB", hipsparse_format2string(arg.formatB));
        print("csr2csc_alg", hipsparse_csr2cscalg2string(arg.csr2csc_alg));
        print("dense2sparse_alg", hipsparse_densetosparsealg2string(arg.dense2sparse_alg));
        print("sparse2dense_alg", hipsparse_sparsetodensealg2string(arg.sparse2dense_alg));
        print("sddmm_alg", hipsparse_sddmmalg2string(arg.sddmm_alg));
        print("spgemm_alg", hipsparse_spgemmalg2string(arg.spgemm_alg));
        print("spmm_alg", hipsparse_spmmalg2string(arg.spmm_alg));
        print("spmv_alg", hipsparse_spmvalg2string(arg.spmv_alg));
        print("spsm_alg", hipsparse_spsmalg2string(arg.spsm_alg));
        print("spsv_alg", hipsparse_spsvalg2string(arg.spsv_alg));
        print("csr2csc_alg", arg.csr2csc_alg);
        print("dense2sparse_alg", arg.dense2sparse_alg);
        print("sparse2dense_alg", arg.sparse2dense_alg);
        print("sddmm_alg", arg.sddmm_alg);
        print("spgemm_alg", arg.spgemm_alg);
        print("spmm_alg", arg.spmm_alg);
        print("spmv_alg", arg.spmv_alg);
        print("spsm_alg", arg.spsm_alg);
        print("spsv_alg", arg.spsv_alg);
        print("numeric_boost", arg.numericboost);
        print("boosttol", arg.boosttol);
        print("boostval", arg.boostval);
        print("boostvali", arg.boostvali);
        print("ell_width", arg.ell_width);
        print("permute", arg.permute);
        print("gtsv_alg", arg.gtsv_alg);
        print("gpsv_alg", arg.gpsv_alg);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);
        return str << " }\n";
    }
};
