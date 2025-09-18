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

#include "rocsparse_arguments_config.hpp"
#include "rocsparse_clients_matrices_dir.hpp"
#include "rocsparse_enum.hpp"
#include "rocsparse_importer_format_t.hpp"

rocsparse_arguments_config::rocsparse_arguments_config()
{
    //
    // Arguments must be a C-compatible struct so cppcheck complains about non-initialized member variables.
    // Then we need to initialize.
    {
        this->M               = static_cast<rocsparse_int>(0);
        this->N               = static_cast<rocsparse_int>(0);
        this->K               = static_cast<rocsparse_int>(0);
        this->nnz             = static_cast<rocsparse_int>(0);
        this->block_dim       = static_cast<rocsparse_int>(0);
        this->row_block_dimA  = static_cast<rocsparse_int>(0);
        this->col_block_dimA  = static_cast<rocsparse_int>(0);
        this->row_block_dimB  = static_cast<rocsparse_int>(0);
        this->col_block_dimB  = static_cast<rocsparse_int>(0);
        this->dimx            = static_cast<rocsparse_int>(0);
        this->dimy            = static_cast<rocsparse_int>(0);
        this->dimz            = static_cast<rocsparse_int>(0);
        this->ll              = static_cast<rocsparse_int>(0);
        this->l               = static_cast<rocsparse_int>(0);
        this->u               = static_cast<rocsparse_int>(0);
        this->uu              = static_cast<rocsparse_int>(0);
        this->index_type_I    = static_cast<rocsparse_indextype>(0);
        this->index_type_J    = static_cast<rocsparse_indextype>(0);
        this->a_type          = static_cast<rocsparse_datatype>(0);
        this->b_type          = static_cast<rocsparse_datatype>(0);
        this->c_type          = static_cast<rocsparse_datatype>(0);
        this->x_type          = static_cast<rocsparse_datatype>(0);
        this->y_type          = static_cast<rocsparse_datatype>(0);
        this->compute_type    = static_cast<rocsparse_datatype>(0);
        this->A_row_indextype = static_cast<rocsparse_indextype>(0);
        this->A_col_indextype = static_cast<rocsparse_indextype>(0);
        this->B_row_indextype = static_cast<rocsparse_indextype>(0);
        this->B_col_indextype = static_cast<rocsparse_indextype>(0);
        this->C_row_indextype = static_cast<rocsparse_indextype>(0);
        this->C_col_indextype = static_cast<rocsparse_indextype>(0);
        this->alpha           = static_cast<double>(0);
        this->alphai          = static_cast<double>(0);
        this->beta            = static_cast<double>(0);
        this->betai           = static_cast<double>(0);
        this->threshold       = static_cast<double>(0);
        this->percentage      = static_cast<double>(0);
        this->transA          = static_cast<rocsparse_operation>(0);
        this->transB          = static_cast<rocsparse_operation>(0);
        this->baseA           = static_cast<rocsparse_index_base>(0);
        this->baseB           = static_cast<rocsparse_index_base>(0);
        this->baseC           = static_cast<rocsparse_index_base>(0);
        this->baseD           = static_cast<rocsparse_index_base>(0);
        this->action          = static_cast<rocsparse_action>(0);
        this->part            = static_cast<rocsparse_hyb_partition>(0);
        this->matrix_type     = static_cast<rocsparse_matrix_type>(0);
        this->diag            = static_cast<rocsparse_diag_type>(0);
        this->uplo            = static_cast<rocsparse_fill_mode>(0);
        this->storage         = static_cast<rocsparse_storage_mode>(0);
        this->apol            = static_cast<rocsparse_analysis_policy>(0);
        this->spol            = static_cast<rocsparse_solve_policy>(0);
        this->direction       = static_cast<rocsparse_direction>(0);
        this->order           = static_cast<rocsparse_order>(0);
        this->orderB          = static_cast<rocsparse_order>(0);
        this->orderC          = static_cast<rocsparse_order>(0);
        this->formatA         = static_cast<rocsparse_format>(0);
        this->formatB         = static_cast<rocsparse_format>(0);
        this->formatC         = static_cast<rocsparse_format>(0);

        this->itilu0_alg           = rocsparse_itilu0_alg_default;
        this->sddmm_alg            = rocsparse_sddmm_alg_default;
        this->spmv_alg             = rocsparse_spmv_alg_default;
        this->spsv_alg             = rocsparse_spsv_alg_default;
        this->sptrsv_alg           = rocsparse_sptrsv_alg_default;
        this->spitsv_alg           = rocsparse_spitsv_alg_default;
        this->spsm_alg             = rocsparse_spsm_alg_default;
        this->sptrsm_alg           = rocsparse_sptrsm_alg_default;
        this->spmm_alg             = rocsparse_spmm_alg_default;
        this->spgemm_alg           = rocsparse_spgemm_alg_default;
        this->spgeam_alg           = rocsparse_spgeam_alg_default;
        this->sparse_to_dense_alg  = rocsparse_sparse_to_dense_alg_default;
        this->dense_to_sparse_alg  = rocsparse_dense_to_sparse_alg_default;
        this->gtsv_interleaved_alg = static_cast<rocsparse_gtsv_interleaved_alg>(0);
        this->gpsv_interleaved_alg = static_cast<rocsparse_gpsv_interleaved_alg>(0);

        this->matrix                      = static_cast<rocsparse_matrix_init>(0);
        this->matrix_init_kind            = static_cast<rocsparse_matrix_init_kind>(0);
        this->unit_check                  = static_cast<rocsparse_int>(0);
        this->timing                      = static_cast<rocsparse_int>(1);
        this->iters                       = static_cast<rocsparse_int>(0);
        this->iters_inner                 = static_cast<rocsparse_int>(0);
        this->nfreeiter                   = static_cast<rocsparse_int>(0);
        this->nmaxiter                    = static_cast<rocsparse_int>(0);
        this->denseld                     = static_cast<int64_t>(0);
        this->batch_count                 = static_cast<rocsparse_int>(0);
        this->batch_count_A               = static_cast<rocsparse_int>(0);
        this->batch_count_B               = static_cast<rocsparse_int>(0);
        this->batch_count_C               = static_cast<rocsparse_int>(0);
        this->batch_stride                = static_cast<rocsparse_int>(0);
        this->ld_multiplier_B             = static_cast<rocsparse_int>(2);
        this->ld_multiplier_C             = static_cast<rocsparse_int>(2);
        this->algo                        = static_cast<uint32_t>(0);
        this->numericboost                = static_cast<int>(0);
        this->boosttol                    = static_cast<double>(0);
        this->boostval                    = static_cast<double>(0);
        this->boostvali                   = static_cast<double>(0);
        this->tolm                        = static_cast<double>(0);
        this->graph_test                  = false;
        this->skip_reproducibility        = false;
        this->sparsity_pattern_statistics = false;
        this->call_stage_analysis         = true;
        this->convert_to_int              = false;
        this->filename[0]                 = '\0';
        this->function[0]                 = '\0';
        this->name[0]                     = '\0';
        this->category[0]                 = '\0';
        this->hardware[0]                 = '\0';
        this->skip_hardware[0]            = '\0';
    }

    this->precision = 's';
    this->indextype = 's';
}

void rocsparse_arguments_config::set_description(options_description& desc)
{
    // clang-format off
    desc.add_options()("help,h", "produces this help message")

    ("sizem,m",
     value<rocsparse_int>(&this->M)->default_value(128),
     "Specific matrix size testing: sizem is only applicable to SPARSE-2 "
     "& SPARSE-3: the number of rows.")

    ("sizen,n",
     value<rocsparse_int>(&this->N)->default_value(128),
     "Specific matrix/vector size testing: SPARSE-1: the length of the "
     "dense vector. SPARSE-2 & SPARSE-3: the number of columns")

    ("sizek,k",
     value<rocsparse_int>(&this->K)->default_value(128),
     "Specific matrix/vector size testing: SPARSE-3: the number of columns")

    ("sizennz,z",
     value<rocsparse_int>(&this->nnz)->default_value(32),
     "Specific vector size testing, LEVEL-1: the number of non-zero elements "
     "of the sparse vector.")

    ("blockdim",
     value<rocsparse_int>(&this->block_dim)->default_value(2),
     "BSR block dimension (default: 2)")

    ("row-blockdimA",
     value<rocsparse_int>(&this->row_block_dimA)->default_value(2),
     "General BSR row block dimension (default: 2)")

    ("col-blockdimA",
     value<rocsparse_int>(&this->col_block_dimA)->default_value(2),
     "General BSR col block dimension (default: 2)")

    ("row-blockdimB",
     value<rocsparse_int>(&this->row_block_dimB)->default_value(2),
     "General BSR row block dimension (default: 2)")

    ("col-blockdimB",
     value<rocsparse_int>(&this->col_block_dimB)->default_value(2),
     "General BSR col block dimension (default: 2)")

    ("mtx",
     value<std::string>(&this->b_matrixmarket)->default_value(""), "read from matrix "
     "market (.mtx) format. This will override parameters -m, -n, and -z.")

    ("smtx",
     value<std::string>(&this->b_mlcsr)->default_value(""), "read from machine "
     "learning CSR (.smtx) format. This will override parameters -m, -n, and -z.")

    ("bsmtx",
     value<std::string>(&this->b_mlbsr)->default_value(""), "read from machine "
     "learning BSR (.bsmtx) format. This will override parameters -m, -n, and -z.")

    ("rocalution",
     value<std::string>(&this->b_rocalution)->default_value(""),
     "read from rocalution matrix binary file.")

    ("rocsparseio",
     value<std::string>(&this->b_rocsparseio)->default_value(""),
     "read from rocsparseio matrix binary file.")

    ("file",
     value<std::string>(&this->b_file)->default_value(""),
     "read from file with file extension detection.")

    ("matrices-dir",
     value<std::string>(&this->b_matrices_dir)->default_value(""),
     "Specify the matrix source directory.")

    ("dimx",
     value<rocsparse_int>(&this->dimx)->default_value(0), "assemble "
     "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
     "will override parameters -m, -n, -z and --mtx.")

    ("dimy",
     value<rocsparse_int>(&this->dimy)->default_value(0), "assemble "
     "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
     "will override parameters -m, -n, -z and --mtx.")

    ("dimz",
     value<rocsparse_int>(&this->dimz)->default_value(0), "assemble "
     "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
     "will override parameters -m, -n, -z and --mtx.")

    ("diag_ll",
     value<rocsparse_int>(&this->ll)->default_value(0), "assemble "
     "pentadiagonal matrix with stencil <ll l u, uu>.")

    ("diag_l",
     value<rocsparse_int>(&this->l)->default_value(0), "assemble "
     "tridiagonal matrix with stencil <l u> or pentadiagonal matrix with stencil <ll l u, uu>.")

    ("diag_u",
     value<rocsparse_int>(&this->u)->default_value(0), "assemble "
     "tridiagonal matrix with stencil <l u> or pentadiagonal matrix with stencil <ll l u, uu>.")

    ("diag_uu",
     value<rocsparse_int>(&this->uu)->default_value(0), "assemble "
     "pentadiagonal matrix with stencil <ll l u, uu>.")

    ("alpha",
     value<double>(&this->alpha)->default_value(1.0), "specifies the scalar alpha")

    ("beta",
     value<double>(&this->beta)->default_value(0.0), "specifies the scalar beta")

    ("threshold",
     value<double>(&this->threshold)->default_value(1.0), "specifies the scalar threshold")

    ("percentage",
     value<double>(&this->percentage)->default_value(0.0), "specifies the scalar percentage")

    ("transposeA",
     value<char>(&this->b_transA)->default_value('N'),
     "N = no transpose, T = transpose, C = conjugate transpose")

    ("transposeB",
     value<char>(&this->b_transB)->default_value('N'),
     "N = no transpose, T = transpose, C = conjugate transpose, (default = N)")

    ("indexbaseA",
     value<int>(&this->b_baseA)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseB",
     value<int>(&this->b_baseB)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseC",
     value<int>(&this->b_baseC)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseD",
     value<int>(&this->b_baseD)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("action",
     value<int>(&this->b_action)->default_value(0),
     "0 = rocsparse_action_numeric, 1 = rocsparse_action_symbolic, (default: 0)")

    ("hybpart",
     value<int>(&this->b_part)->default_value(0),
     "0 = rocsparse_hyb_partition_auto, 1 = rocsparse_hyb_partition_user,\n"
     "2 = rocsparse_hyb_partition_max, (default: 0)")

    ("hybellwidth",
     value<uint32_t>(&this->algo)->default_value(0),
     "ell width to use when hybpart is set to rocsparse_hyb_partition_user (=1), (default: 0)")

    ("matrix_type",
     value<int>(&this->b_matrix_type)->default_value(0),
     "0 = rocsparse_matrix_type_general, 1 = rocsparse_matrix_type_symmetric,\n"
     "2 = rocsparse_matrix_type_hermitian, 3 = rocsparse_matrix_type_triangular, (default: 0)")

    ("diag",
     value<char>(&this->b_diag)->default_value('N'),
     "N = non-unit diagonal, U = unit diagonal, (default = N)")

    ("uplo",
     value<char>(&this->b_uplo)->default_value('L'),
     "L = lower fill, U = upper fill, (default = L)")

    ("storage",
     value<int>(&this->b_storage)->default_value(0),
     "0 = rocsparse_storage_mode_sorted, 1 = rocsparse_storage_mode_unsorted, (default = 0)")

    ("apolicy",
     value<char>(&this->b_apol)->default_value('R'),
     "R = reuse meta data, F = force re-build, (default = R)")

    ("function,f",
     value<std::string>(&this->function_name)->default_value("axpyi"),
     "SPARSE function to test. Options:\n"
     "  Level1: axpyi, doti, dotci, gthr, gthrz, roti, sctr\n"
     "  Level2: bsrmv, bsrxmv, bsrsv, coomv, coomv_aos, csrmv, csrmv_managed, csrsv, csritsv, coosv, ellmv, hybmv, gebsrmv, gemvi\n"
     "  Level3: bsrmm, bsrsm, gebsrmm, csrmm, csrmm_batched, coomm, coomm_batched, cscmm, cscmm_batched, csrsm, coosm, gemmi, sddmm\n"
     "  Extra: bsrgeam, bsrgemm, csrgeam, csrgemm, csrgemm_reuse\n"
     "  Preconditioner: bsric0, bsrilu0, csric0, csrilu0, csritilu0, gtsv, gtsv_no_pivot, gtsv_no_pivot_strided_batch, gtsv_interleaved_batch, gpsv_interleaved_batch\n"
     "  Conversion: csr2coo, csr2csc, gebsr2gebsc, csr2ell, csr2hyb, csr2bsr, csr2gebsr\n"
     "              coo2csr, ell2csr, hyb2csr, dense2csr, dense2coo, prune_dense2csr, prune_dense2csr_by_percentage, dense2csc\n"
     "              csr2dense, csc2dense, coo2dense, bsr2csr, gebsr2csr, gebsr2gebsr, csr2csr_compress, prune_csr2csr, prune_csr2csr_by_percentage\n"
     "              sparse_to_dense_coo, sparse_to_dense_csr, sparse_to_dense_csc, dense_to_sparse_coo, dense_to_sparse_csr, dense_to_sparse_csc, sparse_to_sparse, extract\n"
     "  Sorting: cscsort, csrsort, coosort\n"
     "  Misc: identity, inverse_permutation, nnz\n"
     "  Util: check_matrix_csr, check_matrix_csc, check_matrix_coo, check_matrix_gebsr, check_matrix_gebsc, check_matrix_ell, check_matrix_hyb")

    ("indextype",
     value<char>(&this->indextype)->default_value('s'),
     "Specify index types to be int32_t (s), int64_t (d) or mixed (m). Options: s,d,m")

    ("precision,r",
     value<char>(&this->precision)->default_value('s'), "Options: s,d,c,z")

    ("verify,v",
     value<rocsparse_int>(&this->unit_check)->default_value(0),
     "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

    ("iters,i",
     value<rocsparse_int>(&this->iters)->default_value(10),
     "Total iterations to run inside timing loop")

    ("iters_inner",
      value<rocsparse_int>(&this->iters_inner)->default_value(50),
      "Inner iterations to run inside timing loop")

    ("nfreeiter",
     value<rocsparse_int>(&this->nfreeiter)->default_value(20),
     "Number of free iterations for iterative algorithms (default: 20)")

    ("nmaxiter",
     value<rocsparse_int>(&this->nmaxiter)->default_value(1000),
     "Maximum number of iterations for iterative algorithms (default: 1000)")

    ("device,d",
     value<rocsparse_int>(&this->device_id)->default_value(0),
     "Set default device to be used for subsequent program runs")

    ("direction",
     value<rocsparse_int>(&this->b_dir)->default_value(rocsparse_direction_row),
     "Indicates whether BSR blocks should be laid out in row-major storage or by column-major storage: row-major storage = 0, column-major storage = 1 (default: 0)")

    ("order",
     value<rocsparse_int>(&this->b_order)->default_value(rocsparse_order_column),
     "Indicates whether a dense matrix is laid out in column-major storage: 1, or row-major storage 0 (default: 1)")

    ("orderB",
     value<rocsparse_int>(&this->b_orderB)->default_value(rocsparse_order_column),
     "Indicates whether a dense matrix is laid out in column-major storage: 1, or row-major storage 0 (default: 1)")

    ("orderC",
     value<rocsparse_int>(&this->b_orderC)->default_value(rocsparse_order_column),
     "Indicates whether a dense matrix is laid out in column-major storage: 1, or row-major storage 0 (default: 1)")

    ("formatA",
     value<rocsparse_int>(&this->b_formatA)->default_value(rocsparse_format_coo),
     "Indicates whether a sparse matrix is laid out in coo format: 0, coo_aos format: 1, csr format: 2, csc format: 3, ell format: 4, bell format: 5, bsr format: 6 (default:0)")

    ("formatB",
     value<rocsparse_int>(&this->b_formatB)->default_value(rocsparse_format_coo),
     "Indicates whether a sparse matrix is laid out in coo format: 0, coo_aos format: 1, csr format: 2, csc format: 3, ell format: 4, bell format: 5, bsr format: 6 (default:0)")

    ("formatC",
      value<rocsparse_int>(&this->b_formatC)->default_value(rocsparse_format_coo),
      "Indicates whether a sparse matrix is laid out in coo format: 0, coo_aos format: 1, csr format: 2, csc format: 3, ell format: 4, bell format: 5, bsr format: 6 (default:0)")

    ("denseld",
     value<int64_t>(&this->denseld)->default_value(128),
     "Indicates the leading dimension of a dense matrix >= M, assuming a column-oriented storage.")

    ("batch_count",
     value<rocsparse_int>(&this->batch_count)->default_value(128),
     "Indicates the batch count for batched routines.")

    ("batch_count_A",
     value<rocsparse_int>(&this->batch_count_A)->default_value(128),
     "Indicates the batch count for the sparse A matrix in spmm batched routines.")

    ("batch_count_B",
     value<rocsparse_int>(&this->batch_count_B)->default_value(128),
     "Indicates the batch count for the dense B matrix in spmm batched routines.")

    ("batch_count_C",
     value<rocsparse_int>(&this->batch_count_C)->default_value(128),
     "Indicates the batch count for the dense C matrix in spmm batched routines.")

    ("batch_stride",
     value<rocsparse_int>(&this->batch_stride)->default_value(128),
     "Indicates the batch stride for batched routines.")

#ifdef ROCSPARSE_WITH_MEMSTAT
    ("memstat-report",
     value<std::string>(&this->b_memory_report_filename)->default_value("rocsparse_bench_memstat.json"),
     "Output filename for memory report.")
#endif

    ("spmv_alg",
      value<rocsparse_int>(&this->b_spmv_alg)->default_value(rocsparse_spmv_alg_default),
      "Indicates what algorithm to use when running SpMV. Possibly choices are default: 0, COO: 1, CSR adaptive: 2, CSR stream: 3, ELL: 4, COO atomic: 5, BSR: 6, CSR LRB: 7, CSR nnzsplit: 8 (default:0)")

    ("itilu0_alg",
      value<rocsparse_int>(&this->b_itilu0_alg)->default_value(rocsparse_itilu0_alg_default),
      "Indicates what algorithm to use when running Iterative ILU0. see documentation.")

    ("spmm_alg",
      value<rocsparse_int>(&this->b_spmm_alg)->default_value(rocsparse_spmm_alg_default),
      "Indicates what algorithm to use when running SpMM. Possibly choices are default: 0, CSR: 1, COO segmented: 2, COO atomic: 3, CSR row split: 4, CSR merge: 5, COO segmented atomic: 6, BELL: 7, BSR: 8, CSR merge-path: 9, CSR non-zero split: 5 (default:0)")

    ("sddmm_alg",
      value<rocsparse_int>(&this->b_sddmm_alg)->default_value(rocsparse_sddmm_alg_default),
      "Indicates what algorithm to use when running SDDMM. Possibly choices are rocsparse_sddmm_alg_default: 0, rocsparse_sddmm_alg_default: 1 (default: 0)")

    ("gtsv_interleaved_alg",
      value<rocsparse_int>(&this->b_gtsv_interleaved_alg)->default_value(rocsparse_gtsv_interleaved_alg_default),
      "Indicates what algorithm to use when running rocsparse_gtsv_interleaved_batch. Possibly choices are thomas: 1, lu: 2, qr: 3 (default:3)")

      ("sparsity-pattern-statistics",
       "enable sparsity pattern statistics: min,max and median of the number of non-zeros per row and per column of the sparsity pattern of the matrix A will be part of benchmarking results.")

      ("no-stage-analysis",
       "disable the stage analysis in algorithms where it applies.");
    // clang-format on
}

int rocsparse_arguments_config::parse(int& argc, char**& argv, options_description& desc)
{
    variables_map vm;
    store(parse_command_line(argc, argv, desc, sizeof(rocsparse_arguments_config)), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return -2;
    }

    if(this->b_dir != rocsparse_direction_row && this->b_dir != rocsparse_direction_column)
    {
        std::cerr << "Invalid value for --direction" << std::endl;
        return -1;
    }

    if(rocsparse_order_t::is_invalid(this->b_order))
    {
        std::cerr << "Invalid value for --order" << std::endl;
    }

    if(rocsparse_order_t::is_invalid(this->b_orderB))
    {
        std::cerr << "Invalid value for --orderB" << std::endl;
    }

    if(rocsparse_order_t::is_invalid(this->b_orderC))
    {
        std::cerr << "Invalid value for --orderC" << std::endl;
    }

    if(rocsparse_format_t::is_invalid(this->b_formatA))
    {
        std::cerr << "Invalid value for --formatA" << std::endl;
        return -1;
    }

    if(rocsparse_format_t::is_invalid(this->b_formatB))
    {
        std::cerr << "Invalid value for --formatB" << std::endl;
        return -1;
    }

    if(rocsparse_format_t::is_invalid(this->b_formatC))
    {
        std::cerr << "Invalid value for --formatC" << std::endl;
        return -1;
    }

    if(rocsparse_itilu0_alg_t::is_invalid(this->b_itilu0_alg))
    {
        std::cerr << "Invalid value '" << this->b_itilu0_alg
                  << "' for --itilu0_alg, valid values are : (";
        rocsparse_itilu0_alg_t::info(std::cerr);
        std::cerr << ")" << std::endl;
        return -1;
    }

    if(this->b_spmv_alg != rocsparse_spmv_alg_default && this->b_spmv_alg != rocsparse_spmv_alg_coo
       && this->b_spmv_alg != rocsparse_spmv_alg_csr_adaptive
       && this->b_spmv_alg != rocsparse_spmv_alg_csr_rowsplit
       && this->b_spmv_alg != rocsparse_spmv_alg_ell
       && this->b_spmv_alg != rocsparse_spmv_alg_coo_atomic
       && this->b_spmv_alg != rocsparse_spmv_alg_bsr
       && this->b_spmv_alg != rocsparse_spmv_alg_csr_lrb
       && this->b_spmv_alg != rocsparse_spmv_alg_csr_nnzsplit)
    {
        std::cerr << "Invalid value for --spmv_alg" << std::endl;
        return -1;
    }

    if(this->b_spmm_alg != rocsparse_spmm_alg_default && this->b_spmm_alg != rocsparse_spmm_alg_csr
       && this->b_spmm_alg != rocsparse_spmm_alg_bsr
       && this->b_spmm_alg != rocsparse_spmm_alg_coo_segmented
       && this->b_spmm_alg != rocsparse_spmm_alg_coo_atomic
       && this->b_spmm_alg != rocsparse_spmm_alg_csr_row_split
       && this->b_spmm_alg != rocsparse_spmm_alg_csr_nnz_split
       && this->b_spmm_alg != rocsparse_spmm_alg_csr_merge_path
       && this->b_spmm_alg != rocsparse_spmm_alg_coo_segmented_atomic
       && this->b_spmm_alg != rocsparse_spmm_alg_bell)
    {
        std::cerr << "Invalid value for --spmm_alg" << std::endl;
        return -1;
    }

    if(this->b_sddmm_alg != rocsparse_sddmm_alg_default
       && this->b_sddmm_alg != rocsparse_sddmm_alg_dense)
    {
        std::cerr << "Invalid value for --sddmm_alg" << std::endl;
        return -1;
    }

    if(this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_default
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_thomas
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_lu
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_qr)
    {
        std::cerr << "Invalid value for --gtsv_interleaved_alg" << std::endl;
        return -1;
    }

    if(vm.count("sparsity-pattern-statistics"))
    {
        this->sparsity_pattern_statistics = true;
    }

    if(vm.count("no-stage-analysis"))
    {
        this->call_stage_analysis = false;
    }

    if(this->b_transA == 'N')
    {
        this->transA = rocsparse_operation_none;
    }
    else if(this->b_transA == 'T')
    {
        this->transA = rocsparse_operation_transpose;
    }
    else if(this->b_transA == 'C')
    {
        this->transA = rocsparse_operation_conjugate_transpose;
    }

    if(this->b_transB == 'N')
    {
        this->transB = rocsparse_operation_none;
    }
    else if(this->b_transB == 'T')
    {
        this->transB = rocsparse_operation_transpose;
    }
    else if(this->b_transB == 'C')
    {
        this->transB = rocsparse_operation_conjugate_transpose;
    }
    sprintf(this->function, "%s", this->function_name.c_str());
    this->baseA = (this->b_baseA == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
    this->baseB = (this->b_baseB == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
    this->baseC = (this->b_baseC == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
    this->baseD = (this->b_baseD == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;

    this->action = (this->b_action == 0) ? rocsparse_action_numeric : rocsparse_action_symbolic;
    this->part   = (this->b_part == 0)   ? rocsparse_hyb_partition_auto
                   : (this->b_part == 1) ? rocsparse_hyb_partition_user
                                         : rocsparse_hyb_partition_max;
    this->matrix_type = (this->b_matrix_type == 0)   ? rocsparse_matrix_type_general
                        : (this->b_matrix_type == 1) ? rocsparse_matrix_type_symmetric
                        : (this->b_matrix_type == 2) ? rocsparse_matrix_type_hermitian
                                                     : rocsparse_matrix_type_triangular;
    this->diag = (this->b_diag == 'N') ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit;
    this->uplo = (this->b_uplo == 'L') ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
    this->storage
        = (this->b_storage == 0) ? rocsparse_storage_mode_sorted : rocsparse_storage_mode_unsorted;
    this->apol
        = (this->b_apol == 'R') ? rocsparse_analysis_policy_reuse : rocsparse_analysis_policy_force;
    this->spol      = rocsparse_solve_policy_auto;
    this->direction = (this->b_dir == rocsparse_direction_row) ? rocsparse_direction_row
                                                               : rocsparse_direction_column;
    this->order
        = (this->b_order == rocsparse_order_row) ? rocsparse_order_row : rocsparse_order_column;
    this->orderB
        = (this->b_orderB == rocsparse_order_row) ? rocsparse_order_row : rocsparse_order_column;
    this->orderC
        = (this->b_orderC == rocsparse_order_row) ? rocsparse_order_row : rocsparse_order_column;
    this->formatA              = (rocsparse_format)this->b_formatA;
    this->formatB              = (rocsparse_format)this->b_formatB;
    this->formatC              = (rocsparse_format)this->b_formatC;
    this->spmv_alg             = (rocsparse_spmv_alg)this->b_spmv_alg;
    this->itilu0_alg           = (rocsparse_itilu0_alg)this->b_itilu0_alg;
    this->spmm_alg             = (rocsparse_spmm_alg)this->b_spmm_alg;
    this->sddmm_alg            = (rocsparse_sddmm_alg)this->b_sddmm_alg;
    this->gtsv_interleaved_alg = (rocsparse_gtsv_interleaved_alg)this->b_gtsv_interleaved_alg;

#ifdef ROCSPARSE_WITH_MEMSTAT
    rocsparse_status status = rocsparse_memstat_report(this->b_memory_report_filename.c_str());
    if(status != rocsparse_status_success)
    {
        std::cerr << "rocsparse_memstat_report failed " << std::endl;
        return -1;
    }
#endif

    if(this->b_matrices_dir != "")
    {
        rocsparse_clients_matrices_dir_set(this->b_matrices_dir.c_str());
    }

    if(this->b_file != "")
    {
        strcpy(this->filename, this->b_file.c_str());

        rocsparse_importer_format_t importer_format;
        importer_format(this->filename);
        switch(importer_format.value)
        {
        case rocsparse_importer_format_t::matrixmarket:
        {
            this->matrix = rocsparse_matrix_file_mtx;
            break;
        }
        case rocsparse_importer_format_t::mlcsr:
        {
            this->matrix = rocsparse_matrix_file_smtx;
            break;
        }
        case rocsparse_importer_format_t::mlbsr:
        {
            this->matrix = rocsparse_matrix_file_bsmtx;
            break;
        }
        case rocsparse_importer_format_t::rocalution:
        {
            this->matrix = rocsparse_matrix_file_rocalution;
            break;
        }
        case rocsparse_importer_format_t::rocsparseio:
        {
            this->matrix = rocsparse_matrix_file_rocsparseio;
            break;
        }
        case rocsparse_importer_format_t::unknown:
        {
            std::cerr << "No extension is detected in the filename '" << b_file << "' "
                      << std::endl;
            std::cerr << "The list of detectable extensions is : " << std::endl;
            for(auto format : rocsparse_importer_format_t::all_formats)
            {
                if(rocsparse_importer_format_t::unknown != format)
                {
                    std::cerr << " - '" << rocsparse_importer_format_t::extension(format) << "'"
                              << std::endl;
                }
            }
            return -1;
        }
        }
    }
    else if(this->b_rocsparseio != "")
    {
        strcpy(this->filename, this->b_rocsparseio.c_str());
        this->matrix = rocsparse_matrix_file_rocsparseio;
    }
    else if(this->b_rocalution != "")
    {
        strcpy(this->filename, this->b_rocalution.c_str());
        this->matrix = rocsparse_matrix_file_rocalution;
    }
    else if(this->b_matrixmarket != "")
    {
        strcpy(this->filename, this->b_matrixmarket.c_str());
        this->matrix = rocsparse_matrix_file_mtx;
    }
    else if(this->b_mlcsr != "")
    {
        strcpy(this->filename, this->b_mlcsr.c_str());
        this->matrix = rocsparse_matrix_file_smtx;
    }
    else if(this->b_mlbsr != "")
    {
        strcpy(this->filename, this->b_mlbsr.c_str());
        this->matrix = rocsparse_matrix_file_bsmtx;
    }
    else if(this->dimx != 0 && this->dimy != 0 && this->dimz != 0)
    {
        this->matrix = rocsparse_matrix_laplace_3d;
    }
    else if(this->dimx != 0 && this->dimy != 0)
    {
        this->matrix = rocsparse_matrix_laplace_2d;
    }
    else if(this->ll == 0 && this->l != 0 && this->u != 0 && this->uu == 0)
    {
        this->matrix = rocsparse_matrix_tridiagonal;
    }
    else if(this->ll != 0 && this->l != 0 && this->u != 0 && this->uu != 0)
    {
        this->matrix = rocsparse_matrix_pentadiagonal;
    }
    else
    {
        this->matrix = rocsparse_matrix_random;
    }

    this->matrix_init_kind = rocsparse_matrix_init_kind_default;
    /* ============================================================================================
   */
    if(this->M < 0 || this->N < 0)
    {
        std::cerr << "Invalid dimension" << std::endl;
        return -1;
    }

    if(this->block_dim < 1)
    {
        std::cerr << "Invalid value for --blockdim" << std::endl;
        return -1;
    }

    if(this->row_block_dimA < 1)
    {
        std::cerr << "Invalid value for --row-blockdimA" << std::endl;
        return -1;
    }

    if(this->col_block_dimA < 1)
    {
        std::cerr << "Invalid value for --col-blockdimA" << std::endl;
        return -1;
    }

    if(this->row_block_dimB < 1)
    {
        std::cerr << "Invalid value for --row-blockdimB" << std::endl;
        return -1;
    }

    if(this->col_block_dimB < 1)
    {
        std::cerr << "Invalid value for --col-blockdimB" << std::endl;
        return -1;
    }

    switch(this->indextype)
    {
    case 's':
    {
        this->index_type_I = rocsparse_indextype_i32;
        this->index_type_J = rocsparse_indextype_i32;
        break;
    }
    case 'd':
    {
        this->index_type_I = rocsparse_indextype_i64;
        this->index_type_J = rocsparse_indextype_i64;
        break;
    }

    case 'm':
    {
        this->index_type_I = rocsparse_indextype_i64;
        this->index_type_J = rocsparse_indextype_i32;
        break;
    }
    default:
    {
        std::cerr << "Invalid value for --indextype" << std::endl;
        return -1;
    }
    }

    switch(this->precision)
    {
    case 's':
    {
        this->compute_type = rocsparse_datatype_f32_r;
        break;
    }
    case 'd':
    {
        this->compute_type = rocsparse_datatype_f64_r;
        break;
    }

    case 'c':
    {
        this->compute_type = rocsparse_datatype_f32_c;
        break;
    }
    case 'z':
    {
        this->compute_type = rocsparse_datatype_f64_c;
        break;
    }
    default:
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }
    }

    this->A_row_indextype = this->index_type_I;
    this->A_col_indextype = this->index_type_J;

    return 0;
}
