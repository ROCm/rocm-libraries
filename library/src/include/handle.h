/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-auxiliary.h"
#include "rocsparse-version.h"

#include "rocsparse_blas.h"
#include <fstream>
#include <hip/hip_runtime_api.h>

/*! \brief typedefs to opaque info structs */
typedef struct _rocsparse_trm_info*      rocsparse_trm_info;
typedef struct _rocsparse_csrmv_info*    rocsparse_csrmv_info;
typedef struct _rocsparse_csrgemm_info*  rocsparse_csrgemm_info;
typedef struct _rocsparse_csritsv_info*  rocsparse_csritsv_info;
typedef struct _rocsparse_adaptive_info* rocsparse_adaptive_info;
typedef struct _rocsparse_lrb_info*      rocsparse_lrb_info;
typedef struct _rocsparse_bsrmv_info*    rocsparse_bsrmv_info;
typedef struct _rocsparse_csrmv_info*    rocsparse_cscmv_info;

/********************************************************************************
 * \brief rocsparse_handle is a structure holding the rocsparse library context.
 * It must be initialized using rocsparse_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparse_destroy_handle().
 *******************************************************************************/
struct _rocsparse_handle
{
    // constructor
    _rocsparse_handle();
    // destructor
    ~_rocsparse_handle();

    // set stream
    rocsparse_status set_stream(hipStream_t user_stream);
    // get stream
    rocsparse_status get_stream(hipStream_t* user_stream) const;
    // set pointer mode
    rocsparse_status set_pointer_mode(rocsparse_pointer_mode user_mode);
    // get pointer mode
    rocsparse_status get_pointer_mode(rocsparse_pointer_mode* user_mode) const;

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device wavefront size
    int wavefront_size;
    // shared memory per block opt-in
    int shared_mem_per_block_optin;
    // asic revision
    int asic_rev;
    // stream ; default stream is system stream NULL
    hipStream_t stream = 0;
    // pointer mode ; default mode is host
    rocsparse_pointer_mode pointer_mode = rocsparse_pointer_mode_host;
    // logging mode
    rocsparse_layer_mode layer_mode;
    // device buffer
    size_t buffer_size{};
    void*  buffer{};

    void* alpha{};
    void* beta{};

    // device one
    void* sone{};
    void* done{};

    // blas handle
    rocsparse::blas_handle blas_handle;

    // logging streams
    std::ofstream log_trace_ofs;
    std::ofstream log_bench_ofs;
    std::ofstream log_debug_ofs;
    std::ostream* log_trace_os{};
    std::ostream* log_bench_os{};
    std::ostream* log_debug_os{};
};

/********************************************************************************
 * \brief rocsparse_mat_descr is a structure holding the rocsparse matrix
 * descriptor. It must be initialized using rocsparse_create_mat_descr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocsparse_destroy_mat_descr().
 *******************************************************************************/
struct _rocsparse_mat_descr
{
    // matrix type
    rocsparse_matrix_type type = rocsparse_matrix_type_general;
    // fill mode
    rocsparse_fill_mode fill_mode = rocsparse_fill_mode_lower;
    // diagonal type
    rocsparse_diag_type diag_type = rocsparse_diag_type_non_unit;
    // index base
    rocsparse_index_base base = rocsparse_index_base_zero;
    // storage mode
    rocsparse_storage_mode storage_mode = rocsparse_storage_mode_sorted;
    // maximum nnz per row
    int64_t max_nnz_per_row = 0;
};

/********************************************************************************
 * \brief rocsparse_hyb_mat is a structure holding the rocsparse HYB matrix.
 * It must be initialized using rocsparse_create_hyb_mat() and the returned
 * handle must be passed to all subsequent library function calls that involve
 * the HYB matrix.
 * It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *******************************************************************************/
struct _rocsparse_hyb_mat
{
    // num rows
    rocsparse_int m{};
    // num cols
    rocsparse_int n{};

    // partition type
    rocsparse_hyb_partition partition = rocsparse_hyb_partition_auto;

    // ELL matrix part
    rocsparse_int  ell_nnz{};
    rocsparse_int  ell_width{};
    rocsparse_int* ell_col_ind{};
    void*          ell_val{};

    // COO matrix part
    rocsparse_int  coo_nnz{};
    rocsparse_int* coo_row_ind{};
    rocsparse_int* coo_col_ind{};
    void*          coo_val{};

    rocsparse_datatype data_type_T = rocsparse_datatype_f32_r;
};

/********************************************************************************
 * \brief rocsparse_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_mat_info().
 *******************************************************************************/
struct _rocsparse_mat_info
{
protected:
    rocsparse_csrmv_info csrmv_info{};
    rocsparse_bsrmv_info bsrmv_info{};

public:
    rocsparse_trm_info     bsrsv_upper_info{};
    rocsparse_trm_info     bsrsv_lower_info{};
    rocsparse_trm_info     bsrsvt_upper_info{};
    rocsparse_trm_info     bsrsvt_lower_info{};
    rocsparse_trm_info     bsric0_info{};
    rocsparse_trm_info     bsrilu0_info{};
    rocsparse_trm_info     bsrsm_upper_info{};
    rocsparse_trm_info     bsrsm_lower_info{};
    rocsparse_trm_info     bsrsmt_upper_info{};
    rocsparse_trm_info     bsrsmt_lower_info{};
    rocsparse_trm_info     csric0_info{};
    rocsparse_trm_info     csrilu0_info{};
    rocsparse_trm_info     csrsv_upper_info{};
    rocsparse_trm_info     csrsv_lower_info{};
    rocsparse_trm_info     csrsvt_upper_info{};
    rocsparse_trm_info     csrsvt_lower_info{};
    rocsparse_trm_info     csrsm_upper_info{};
    rocsparse_trm_info     csrsm_lower_info{};
    rocsparse_trm_info     csrsmt_upper_info{};
    rocsparse_trm_info     csrsmt_lower_info{};
    rocsparse_csrgemm_info csrgemm_info{};
    rocsparse_csritsv_info csritsv_info{};

    // zero pivot for csrsv, csrsm, csrilu0, csric0
    void* zero_pivot{};

    // singular pivot for csric0
    void* singular_pivot{};

    // tolerance used for determining near singularity
    double singular_tol{};

    // numeric boost for ilu0
    int         boost_enable{};
    size_t      boost_tol_size{};
    const void* boost_tol{};
    const void* boost_val{};

    ~_rocsparse_mat_info();

    void                 set_csrmv_info(rocsparse_csrmv_info value);
    rocsparse_csrmv_info get_csrmv_info();

    void                 set_bsrmv_info(rocsparse_bsrmv_info value);
    rocsparse_bsrmv_info get_bsrmv_info();
};

/********************************************************************************
 * \brief rocsparse_color_info is a structure holding the color info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_color_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_color_info().
 *******************************************************************************/
struct _rocsparse_color_info
{
};

struct _rocsparse_adaptive_info
{

    size_t    size{}; // num row blocks
    int64_t   first_row{};
    int64_t   last_row{};
    void*     row_blocks{};
    uint32_t* wg_flags{};
    void*     wg_ids{};

public:
    ~_rocsparse_adaptive_info();
    void clear();
};

struct _rocsparse_lrb_info
{
    void* rows_offsets_scratch{}; // size of m
    void* rows_bins{}; // size of m
    void* n_rows_bins{}; // size of 32

    size_t    size{};
    uint32_t* wg_flags{};

    int64_t nRowsBins[32]{}; // host array

    ~_rocsparse_lrb_info();
    void clear();
};

/********************************************************************************
 * \brief rocsparse_csrmv_info is a structure holding the rocsparse csrmv info
 * data gathered during csrmv_analysis.
 *******************************************************************************/
struct _rocsparse_csrmv_info
{
    _rocsparse_adaptive_info adaptive{};
    _rocsparse_lrb_info      lrb{};

    // some data to verify correct execution
    rocsparse_operation         trans = rocsparse_operation_none;
    int64_t                     m{};
    int64_t                     n{};
    int64_t                     nnz{};
    int64_t                     max_rows{};
    const _rocsparse_mat_descr* descr{};
    const void*                 csr_row_ptr{};
    const void*                 csr_col_ind{};

    rocsparse_indextype index_type_I = rocsparse_indextype_u16;
    rocsparse_indextype index_type_J = rocsparse_indextype_u16;

    _rocsparse_csrmv_info() {}

    ~_rocsparse_csrmv_info()
    {
        this->clear();
    }

    void clear()
    {
        this->adaptive.clear();
        this->lrb.clear();
        this->trans        = rocsparse_operation_none;
        this->m            = 0;
        this->n            = 0;
        this->nnz          = 0;
        this->max_rows     = 0;
        this->descr        = nullptr;
        this->csr_row_ptr  = nullptr;
        this->csr_col_ind  = nullptr;
        this->index_type_I = rocsparse_indextype_u16;
        this->index_type_J = rocsparse_indextype_u16;
    }
};

/********************************************************************************
 * \brief rocsparse_bsrmv_info is a structure holding the rocsparse bsrmv info
 * data gathered during bsrmv_analysis.
 *******************************************************************************/
struct _rocsparse_bsrmv_info
{
protected:
    rocsparse_csrmv_info m_csrmv_info{};

public:
    _rocsparse_bsrmv_info() {}

    ~_rocsparse_bsrmv_info()
    {
        if(this->m_csrmv_info)
        {
            delete this->m_csrmv_info;
            this->m_csrmv_info = nullptr;
        }
    }

    rocsparse_csrmv_info get_csrmv_info()
    {
        return this->m_csrmv_info;
    }
    void set_csrmv_info(rocsparse_csrmv_info value)
    {
        this->m_csrmv_info = value;
    }
};

namespace rocsparse
{
    /********************************************************************************
   * \brief Copy csrmv info.
   *******************************************************************************/
    rocsparse_status copy_csrmv_info(rocsparse_csrmv_info dest, const rocsparse_csrmv_info src);
}

struct _rocsparse_trm_info
{
    // maximum non-zero entries per row
    int64_t max_nnz{};

    // device array to hold row permutation
    void* row_map{};
    // device array to hold pointer to diagonal entry
    void* trm_diag_ind{};
    // device pointers to hold transposed data
    void* trmt_perm{};
    void* trmt_row_ptr{};
    void* trmt_col_ind{};

    // some data to verify correct execution
    int64_t                     m{};
    int64_t                     nnz{};
    const _rocsparse_mat_descr* descr{};
    const void*                 trm_row_ptr{};
    const void*                 trm_col_ind{};

    rocsparse_indextype index_type_I = rocsparse_indextype_u16;
    rocsparse_indextype index_type_J = rocsparse_indextype_u16;
};

namespace rocsparse
{
    /********************************************************************************
 * \brief rocsparse_trm_info is a structure holding the rocsparse bsrsv, csrsv,
 * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
 * csrilu0_analysis and csric0_analysis. It must be initialized using the
 * create_trm_info() routine. It should be destroyed at the end
 * using destroy_trm_info().
 *******************************************************************************/
    rocsparse_status create_trm_info(rocsparse_trm_info* info);

    /********************************************************************************
 * \brief Copy trm info.
 *******************************************************************************/
    rocsparse_status copy_trm_info(rocsparse_trm_info dest, const rocsparse_trm_info src);

    /********************************************************************************
 * \brief Destroy trm info.
 *******************************************************************************/
    rocsparse_status destroy_trm_info(rocsparse_trm_info info);

    /********************************************************************************
 * \brief check_trm_shared checks if the given trm info structure
 * shares its meta data with another trm info structure.
 *******************************************************************************/
    bool check_trm_shared(const rocsparse_mat_info info, rocsparse_trm_info trm);
}

/********************************************************************************
 * \brief rocsparse_csritsv_info is a structure holding the rocsparse csritsv
 * info data gathered during csritsv_buffer_size. It must be initialized using
 * the create_csritsv_info() routine. It should be destroyed at the
 * end using destroy_csritsv_info().
 *******************************************************************************/
struct _rocsparse_csritsv_info
{
protected:
    rocsparse_csrmv_info m_csrmv_info{};

public:
    bool                is_submatrix;
    int64_t             ptr_end_size{};
    rocsparse_indextype ptr_end_indextype{};
    void*               ptr_end{};

    rocsparse_csrmv_info get_csrmv_info()
    {
        return this->m_csrmv_info;
    }
    void set_csrmv_info(rocsparse_csrmv_info value)
    {
        this->m_csrmv_info = value;
    }

    _rocsparse_csritsv_info() = default;

    ~_rocsparse_csritsv_info()
    {
        if(this->m_csrmv_info)
        {
            delete this->m_csrmv_info;
        }
    }
};

namespace rocsparse
{
    /********************************************************************************
 * \brief rocsparse_csritsv_info is a structure holding the rocsparse csritsv
 * info data gathered during csritsv_buffer_size. It must be initialized using
 * the create_csritsv_info() routine. It should be destroyed at the
 * end using destroy_csritsv_info().
 *******************************************************************************/
    rocsparse_status create_csritsv_info(rocsparse_csritsv_info* info);

    /********************************************************************************
 * \brief Copy csritsv info.
 *******************************************************************************/
    rocsparse_status copy_csritsv_info(rocsparse_csritsv_info       dest,
                                       const rocsparse_csritsv_info src);

    /********************************************************************************
 * \brief Destroy csritsv info.
 *******************************************************************************/
    rocsparse_status destroy_csritsv_info(rocsparse_csritsv_info info);
}

/********************************************************************************
 * \brief rocsparse_csrgemm_info is a structure holding the rocsparse csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the create_csrgemm_info() routine. It should be destroyed at the
 * end using destroy_csrgemm_info().
 *******************************************************************************/
struct _rocsparse_csrgemm_info
{
    size_t buffer_size{};
    bool   is_initialized{};
    // Perform alpha * A * B
    bool mul{true};
    // Perform beta * D
    bool add{true};
};

/********************************************************************************
 * \brief rocsparse_spgeam_descr is a structure holding the rocsparse spgeam
 * descriptor. It must be initialized using rocsparse_create_spgeam_descr().
 * It should be destroyed at the end using rocsparse_destroy_spgeam_descr().
 *******************************************************************************/
struct _rocsparse_spgeam_descr
{
    // C matrix row pointer data
    void*               csr_row_ptr_C{};
    int64_t             nnz_C{};
    int64_t             m{};
    rocsparse_indextype indextype{rocsparse_indextype_i32};

    // rocprim buffer data
    bool   rocprim_alloc{};
    size_t rocprim_size{};
    void*  rocprim_buffer{};

    // Perform alpha * A + beta * B
    bool alpha_mul{true};
    bool beta_mul{true};

    rocsparse_spgeam_alg alg{rocsparse_spgeam_alg_default};
    rocsparse_datatype   compute_type{rocsparse_datatype_f32_r};
    rocsparse_operation  trans_A{rocsparse_operation_none};
    rocsparse_operation  trans_B{rocsparse_operation_none};
};

namespace rocsparse
{
    /********************************************************************************
 * \brief rocsparse_csrgemm_info is a structure holding the rocsparse csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the create_csrgemm_info() routine. It should be destroyed at the
 * end using destroy_csrgemm_info().
 *******************************************************************************/
    rocsparse_status create_csrgemm_info(rocsparse_csrgemm_info* info);

    /********************************************************************************
 * \brief Copy csrgemm info.
 *******************************************************************************/
    rocsparse_status copy_csrgemm_info(rocsparse_csrgemm_info       dest,
                                       const rocsparse_csrgemm_info src);

    /********************************************************************************
 * \brief Destroy csrgemm info.
 *******************************************************************************/
    rocsparse_status destroy_csrgemm_info(rocsparse_csrgemm_info info);
}

/********************************************************************************
 * \brief ELL format indexing
 *******************************************************************************/
#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

struct _rocsparse_spvec_descr
{
    bool init{};

    int64_t size{};
    int64_t nnz{};

    void* idx_data{};
    void* val_data{};

    const void* const_idx_data{};
    const void* const_val_data{};

    rocsparse_indextype idx_type{};
    rocsparse_datatype  data_type{};

    rocsparse_index_base idx_base{};
};

struct _rocsparse_spmat_descr
{
    bool init{};

    mutable bool analysed{};

    int64_t rows{};
    int64_t cols{};
    int64_t nnz{};

    void* row_data{};
    void* col_data{};
    void* ind_data{};
    void* val_data{};

    const void* const_row_data{};
    const void* const_col_data{};
    const void* const_ind_data{};
    const void* const_val_data{};

    rocsparse_indextype row_type{};
    rocsparse_indextype col_type{};
    rocsparse_datatype  data_type{};

    rocsparse_index_base idx_base{};
    rocsparse_format     format{};

    rocsparse_mat_descr descr{};
    rocsparse_mat_info  info{};

    rocsparse_direction block_dir{};
    int64_t             block_dim{};
    int64_t             ell_cols{};
    int64_t             ell_width{};

    int64_t batch_count{};
    int64_t batch_stride{};
    int64_t offsets_batch_stride{};
    int64_t columns_values_batch_stride{};
};

struct _rocsparse_dnvec_descr
{
    bool init{};

    int64_t            size{};
    void*              values{};
    const void*        const_values{};
    rocsparse_datatype data_type{};
};

struct _rocsparse_dnmat_descr
{
    bool init{};

    int64_t rows{};
    int64_t cols{};
    int64_t ld{};

    void* values{};

    const void* const_values{};

    rocsparse_datatype data_type{};
    rocsparse_order    order{};

    int64_t batch_count{};
    int64_t batch_stride{};
};

namespace rocsparse
{
    //
    // Get architecture name.
    //
    std::string handle_get_arch_name(rocsparse_handle handle);

    struct rocpsarse_arch_names
    {
        static constexpr const char* gfx908 = "gfx908";
    };

    //
    // Get xnack mode.
    //
    std::string handle_get_xnack_mode(rocsparse_handle handle);
}
