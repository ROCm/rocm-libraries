/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include "internal/generic/rocsparse_v2_spmv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "../conversion/rocsparse_convert_scalar.hpp"
#include "rocsparse_bsrmv.hpp"
#include "rocsparse_coomv.hpp"
#include "rocsparse_coomv_aos.hpp"
#include "rocsparse_cscmv.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_ellmv.hpp"
#include "rocsparse_spmv.hpp"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spmv_alg value_)
{
    switch(value_)
    {
    case rocsparse_spmv_alg_default:
    case rocsparse_spmv_alg_coo:
    case rocsparse_spmv_alg_csr_adaptive:
    case rocsparse_spmv_alg_csr_rowsplit:
    case rocsparse_spmv_alg_ell:
    case rocsparse_spmv_alg_coo_atomic:
    case rocsparse_spmv_alg_bsr:
    case rocsparse_spmv_alg_csr_lrb:
    case rocsparse_spmv_alg_csr_nnzsplit:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spmv_input value_)
{
    switch(value_)
    {
    case rocsparse_spmv_input_alg:
    case rocsparse_spmv_input_operation:
    case rocsparse_spmv_input_compute_datatype:
    case rocsparse_spmv_input_scalar_datatype:
    case rocsparse_spmv_input_nnz_use_starting_block_ids:
    {
        return false;
    }
    }
    return true;
};

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_v2_spmv_stage value_)
{
    switch(value_)
    {
    case rocsparse_v2_spmv_stage_analysis:
    case rocsparse_v2_spmv_stage_compute:
    {
        return false;
    }
    }
    return true;
};

struct _rocsparse_spmv_descr
{
protected:
    rocsparse_v2_spmv_stage m_stage;
    rocsparse_spmv_alg      m_alg;
    rocsparse_operation     m_operation;
    rocsparse_datatype      m_scalar_datatype;
    rocsparse_datatype      m_compute_datatype;
    bool                    m_use_starting_block_ids;

    float m_local_host_alpha_value[4];
    float m_local_host_beta_value[4];

    rocsparse_csrmv_info m_csrmv_info{};
    rocsparse_cscmv_info m_cscmv_info{};
    rocsparse_bsrmv_info m_bsrmv_info{};

public:
    rocsparse_cscmv_info get_cscmv_info()
    {
        return this->m_cscmv_info;
    }
    void set_cscmv_info(rocsparse_cscmv_info value)
    {
        this->m_cscmv_info = value;
    }

    rocsparse_csrmv_info get_csrmv_info()
    {
        return this->m_csrmv_info;
    }
    void set_csrmv_info(rocsparse_csrmv_info value)
    {
        this->m_csrmv_info = value;
    }

    rocsparse_bsrmv_info get_bsrmv_info()
    {
        return this->m_bsrmv_info;
    }
    void set_bsrmv_info(rocsparse_bsrmv_info value)
    {
        this->m_bsrmv_info = value;
    }

    ~_rocsparse_spmv_descr()
    {
        if(this->m_csrmv_info != nullptr)
        {
            delete this->m_csrmv_info;
        }
        if(this->m_cscmv_info != nullptr)
        {
            delete this->m_cscmv_info;
        }
        if(this->m_bsrmv_info != nullptr)
        {
            delete this->m_bsrmv_info;
        }
    }

    _rocsparse_spmv_descr()
        : m_stage((rocsparse_v2_spmv_stage)-1)
        , m_alg((rocsparse_spmv_alg)-1)
        , m_operation((rocsparse_operation)-1)
        , m_scalar_datatype((rocsparse_datatype)-1)
        , m_compute_datatype((rocsparse_datatype)-1)
        , m_use_starting_block_ids(false)
    {
    }

    void* get_local_host_alpha()
    {
        return &this->m_local_host_alpha_value[0];
    }
    void* get_local_host_beta()
    {
        return &this->m_local_host_beta_value[0];
    }

    rocsparse_v2_spmv_stage get_stage() const
    {
        return this->m_stage;
    }
    rocsparse_spmv_alg get_alg() const
    {
        return this->m_alg;
    }
    rocsparse_operation get_operation() const
    {
        return this->m_operation;
    }
    rocsparse_datatype get_scalar_datatype() const
    {
        return this->m_scalar_datatype;
    }
    rocsparse_datatype get_compute_datatype() const
    {
        return this->m_compute_datatype;
    }

    void set_stage(rocsparse_v2_spmv_stage value)
    {
        this->m_stage = value;
    }
    void set_alg(rocsparse_spmv_alg value)
    {
        this->m_alg = value;
    }

    void set_operation(rocsparse_operation value)
    {
        this->m_operation = value;
    }

    void set_scalar_datatype(rocsparse_datatype value)
    {
        this->m_scalar_datatype = value;
    }

    void set_compute_datatype(rocsparse_datatype value)
    {
        this->m_compute_datatype = value;
    }
    void set_use_starting_block_ids(bool value)
    {
        this->m_use_starting_block_ids = value;
    }
    bool get_use_starting_block_ids() const
    {
        return this->m_use_starting_block_ids;
    }
};

extern "C" rocsparse_status rocsparse_create_spmv_descr(rocsparse_spmv_descr* descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_POINTER(0, descr);
    *descr = new _rocsparse_spmv_descr();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_destroy_spmv_descr(rocsparse_spmv_descr descr)
try
{

    ROCSPARSE_ROUTINE_TRACE;
    if(descr != nullptr)
    {
        delete descr;
    }
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spmv_set_input(rocsparse_handle     handle,
                                                     rocsparse_spmv_descr descr,
                                                     rocsparse_spmv_input input,
                                                     const void*          in,
                                                     size_t               size_in_bytes,
                                                     rocsparse_error*     p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, input);
    ROCSPARSE_CHECKARG_POINTER(3, in);

    switch(input)
    {
    case rocsparse_spmv_input_alg:
    {

        switch(descr->get_stage())
        {
        case rocsparse_v2_spmv_stage_analysis:
        case rocsparse_v2_spmv_stage_compute:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_internal_error,
                "The field 'rocsparse_spmv_input_alg' must be set before the stage "
                "'rocsparse_v2_spmv_stage_analysis' is executed.");
        }
        }

        ROCSPARSE_CHECKARG(4,
                           size_in_bytes,
                           size_in_bytes != sizeof(rocsparse_spmv_alg),
                           rocsparse_status_invalid_size);
        const rocsparse_spmv_alg alg = *reinterpret_cast<const rocsparse_spmv_alg*>(in);
        descr->set_alg(alg);
        return rocsparse_status_success;
    }

    case rocsparse_spmv_input_scalar_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           size_in_bytes,
                           size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(in);
        descr->set_scalar_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_spmv_input_compute_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           size_in_bytes,
                           size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(in);
        descr->set_compute_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_spmv_input_operation:
    {
        switch(descr->get_stage())
        {
        case rocsparse_v2_spmv_stage_analysis:
        case rocsparse_v2_spmv_stage_compute:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_internal_error,
                "The field 'rocsparse_spmv_input_operation' must be set before the stage "
                "'rocsparse_v2_spmv_stage_analysis' is executed.");
        }
        }

        ROCSPARSE_CHECKARG(4,
                           size_in_bytes,
                           size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op = *reinterpret_cast<const rocsparse_operation*>(in);
        descr->set_operation(op);
        return rocsparse_status_success;
    }

    case rocsparse_spmv_input_nnz_use_starting_block_ids:
    {
        ROCSPARSE_CHECKARG(
            4, size_in_bytes, size_in_bytes != sizeof(bool), rocsparse_status_invalid_size);
        const bool use_starting_block_ids = *reinterpret_cast<const bool*>(in);
        descr->set_use_starting_block_ids(use_starting_block_ids);
        return rocsparse_status_success;
    }
        // LCOV_EXCL_START
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{

    rocsparse_status v2_spmv(rocsparse_handle            handle,
                             rocsparse_spmv_descr        spmv_descr,
                             const void*                 alpha,
                             rocsparse_const_spmat_descr mat,
                             rocsparse_const_dnvec_descr x,
                             const void*                 beta,
                             const rocsparse_dnvec_descr y,
                             rocsparse_v2_spmv_stage     stage,
                             size_t                      buffer_size_in_bytes,
                             void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        //
        //
        //
        const rocsparse_format    format         = mat->format;
        const int64_t             rows           = mat->rows;
        const int64_t             cols           = mat->cols;
        const int64_t             nnz            = mat->nnz;
        rocsparse_mat_descr       mat_descr      = mat->descr;
        const rocsparse_datatype  data_type      = mat->data_type;
        const rocsparse_indextype row_type       = mat->row_type;
        const rocsparse_indextype col_type       = mat->col_type;
        const void*               const_val_data = mat->const_val_data;
        const void*               const_row_data = mat->const_row_data;
        const void*               const_ind_data = mat->const_ind_data;
        const void*               const_col_data = mat->const_col_data;
        const rocsparse_datatype  x_data_type    = x->data_type;
        const rocsparse_datatype  y_data_type    = y->data_type;
        const void*               x_const_values = x->const_values;
        void*                     y_values       = y->values;
        const bool                analysed       = mat->analysed;
        const int64_t             block_dim      = mat->block_dim;
        const int64_t             ell_width      = mat->ell_width;
        const rocsparse_direction block_dir      = mat->block_dir;

        //
        //
        //
        const rocsparse_operation operation = spmv_descr->get_operation();
        const rocsparse_spmv_alg  alg       = spmv_descr->get_alg();

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_spmv_alg(format, alg)));

        switch(stage)
        {
        case rocsparse_v2_spmv_stage_analysis:
        {
            switch(format)
            {

            case rocsparse_format_ell:
            case rocsparse_format_coo_aos:
            {
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {
                rocsparse_coomv_alg coomv_alg;
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2coomv_alg(alg, coomv_alg)));

                if(analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR((rocsparse::coomv_analysis(handle,
                                                                         operation,
                                                                         coomv_alg,
                                                                         rows,
                                                                         cols,
                                                                         nnz,
                                                                         mat_descr,
                                                                         data_type,
                                                                         const_val_data,
                                                                         row_type,
                                                                         const_row_data,
                                                                         col_type,
                                                                         const_col_data)));
                    mat->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_bsr:
            {
                //
                // If algorithm 1 or default is selected and analysis step is required
                //
                rocsparse_bsrmv_info bsrmv_info = spmv_descr->get_bsrmv_info();
                if(bsrmv_info == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrmv_analysis(handle,
                                                                         block_dir,
                                                                         operation,
                                                                         rows,
                                                                         cols,
                                                                         nnz,
                                                                         mat_descr,
                                                                         data_type,
                                                                         const_val_data,
                                                                         row_type,
                                                                         const_row_data,
                                                                         col_type,
                                                                         const_col_data,
                                                                         block_dim,
                                                                         &bsrmv_info)));
                    spmv_descr->set_bsrmv_info(bsrmv_info);
                }

                return rocsparse_status_success;
            }

            case rocsparse_format_csr:
            {
                rocsparse::csrmv_alg alg_csrmv;
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2csrmv_alg(alg, alg_csrmv)));

                rocsparse_csrmv_info csrmv_info = spmv_descr->get_csrmv_info();
                if(csrmv_info == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrmv_analysis(handle,
                                                                         operation,
                                                                         alg_csrmv,
                                                                         rows,
                                                                         cols,
                                                                         nnz,
                                                                         mat_descr,
                                                                         data_type,
                                                                         const_val_data,
                                                                         row_type,
                                                                         const_row_data,
                                                                         col_type,
                                                                         const_col_data,
                                                                         &csrmv_info)));
                    spmv_descr->set_csrmv_info(csrmv_info);
                }

                return rocsparse_status_success;
            }

            case rocsparse_format_csc:
            {
                rocsparse::csrmv_alg alg_csrmv;
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2csrmv_alg(alg, alg_csrmv)));
                rocsparse_cscmv_info cscmv_info = spmv_descr->get_cscmv_info();
                if(cscmv_info == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR((rocsparse::cscmv_analysis(handle,
                                                                         operation,
                                                                         alg_csrmv,
                                                                         rows,
                                                                         cols,
                                                                         nnz,
                                                                         mat_descr,
                                                                         data_type,
                                                                         const_val_data,
                                                                         col_type,
                                                                         const_col_data,
                                                                         row_type,
                                                                         const_row_data,
                                                                         &cscmv_info)));
                    spmv_descr->set_cscmv_info(cscmv_info);
                }
                return rocsparse_status_success;
            }

                // LCOV_EXCL_START
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
            // LCOV_EXCL_STOP

        case rocsparse_v2_spmv_stage_compute:
        {
            static constexpr bool    fallback_algorithm = false;
            const rocsparse_datatype scalar_datatype    = spmv_descr->get_scalar_datatype();
            const rocsparse_datatype compute_datatype   = spmv_descr->get_compute_datatype();
            const void*              local_alpha        = alpha;
            const void*              local_beta         = beta;

            if(scalar_datatype != compute_datatype)
            {
                switch(handle->pointer_mode)
                {
                case rocsparse_pointer_mode_host:
                {

                    //
                    // Convert scalars from scalar_datatype to compute_datatype
                    //
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::convert_host_scalars(scalar_datatype,
                                                        compute_datatype,
                                                        alpha,
                                                        spmv_descr->get_local_host_alpha(),
                                                        beta,
                                                        spmv_descr->get_local_host_beta()));

                    local_alpha = spmv_descr->get_local_host_alpha();
                    local_beta  = spmv_descr->get_local_host_beta();

                    break;
                }
                case rocsparse_pointer_mode_device:
                {

                    //
                    // Convert scalars from scalar_datatype to compute_datatype
                    //
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_device_scalars(handle->stream,
                                                                                scalar_datatype,
                                                                                compute_datatype,
                                                                                alpha,
                                                                                handle->alpha,
                                                                                beta,
                                                                                handle->beta));

                    local_alpha = handle->alpha;
                    local_beta  = handle->beta;

                    break;
                }
                }
            }

            switch(format)
            {
            case rocsparse_format_coo:
            {
                rocsparse_coomv_alg coomv_alg;
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2coomv_alg(alg, coomv_alg)));

                RETURN_IF_ROCSPARSE_ERROR((rocsparse::coomv(handle,
                                                            operation,
                                                            coomv_alg,
                                                            rows,
                                                            cols,
                                                            nnz,
                                                            compute_datatype,
                                                            local_alpha,
                                                            mat_descr,
                                                            data_type,
                                                            const_val_data,
                                                            row_type,
                                                            const_row_data,
                                                            col_type,
                                                            const_col_data,
                                                            x_data_type,
                                                            x_const_values,
                                                            compute_datatype,
                                                            local_beta,
                                                            y_data_type,
                                                            y_values,
                                                            fallback_algorithm)));
                return rocsparse_status_success;
            }

            case rocsparse_format_coo_aos:
            {
                rocsparse_coomv_aos_alg coomv_aos_alg;
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2coomv_aos_alg(alg, coomv_aos_alg)));
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::coomv_aos(handle,
                                                                operation,
                                                                coomv_aos_alg,
                                                                rows,
                                                                cols,
                                                                nnz,
                                                                compute_datatype,
                                                                local_alpha,
                                                                mat_descr,
                                                                data_type,
                                                                const_val_data,
                                                                row_type,
                                                                const_ind_data,
                                                                x_data_type,
                                                                x_const_values,
                                                                compute_datatype,
                                                                local_beta,
                                                                y_data_type,
                                                                y_values,
                                                                fallback_algorithm)));
                return rocsparse_status_success;
            }

            case rocsparse_format_bsr:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrmv(handle,
                                                            block_dir,
                                                            operation,
                                                            rows,
                                                            cols,
                                                            nnz,
                                                            compute_datatype,
                                                            local_alpha,
                                                            mat_descr,
                                                            data_type,
                                                            const_val_data,
                                                            row_type,
                                                            const_row_data,
                                                            col_type,
                                                            const_col_data,
                                                            block_dim,
                                                            spmv_descr->get_bsrmv_info(),
                                                            x_data_type,
                                                            x_const_values,
                                                            compute_datatype,
                                                            local_beta,
                                                            y_data_type,
                                                            y_values)));
                return rocsparse_status_success;
            }
            case rocsparse_format_csr:
            {
                rocsparse::csrmv_alg alg_csrmv;
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2csrmv_alg(alg, alg_csrmv)));
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::csrmv(handle,
                                      operation,
                                      alg_csrmv,
                                      rows,
                                      cols,
                                      nnz,
                                      compute_datatype,
                                      local_alpha,
                                      mat_descr,
                                      data_type,
                                      const_val_data,
                                      row_type,
                                      const_row_data,
                                      row_type,
                                      reinterpret_cast<const char*>(const_row_data)
                                          + rocsparse::indextype_sizeof(row_type),
                                      col_type,
                                      const_col_data,
                                      spmv_descr->get_csrmv_info(),
                                      x_data_type,
                                      x_const_values,
                                      compute_datatype,
                                      local_beta,
                                      y_data_type,
                                      y_values,
                                      fallback_algorithm)));
                return rocsparse_status_success;
            }
            case rocsparse_format_csc:
            {
                rocsparse::csrmv_alg alg_csrmv;
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2csrmv_alg(alg, alg_csrmv)));

                RETURN_IF_ROCSPARSE_ERROR((rocsparse::cscmv(handle,
                                                            operation,
                                                            alg_csrmv,
                                                            rows,
                                                            cols,
                                                            nnz,
                                                            compute_datatype,
                                                            local_alpha,
                                                            mat_descr,
                                                            data_type,
                                                            const_val_data,
                                                            col_type,
                                                            const_col_data,
                                                            row_type,
                                                            const_row_data,
                                                            spmv_descr->get_cscmv_info(),
                                                            x_data_type,
                                                            x_const_values,
                                                            compute_datatype,
                                                            local_beta,
                                                            y_data_type,
                                                            y_values,
                                                            fallback_algorithm)));
                return rocsparse_status_success;
            }
            case rocsparse_format_ell:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::ellmv(handle,
                                                            operation,
                                                            rows,
                                                            cols,
                                                            compute_datatype,
                                                            local_alpha,
                                                            mat_descr,
                                                            data_type,
                                                            const_val_data,
                                                            col_type,
                                                            const_col_data,
                                                            ell_width,
                                                            x_data_type,
                                                            x_const_values,
                                                            compute_datatype,
                                                            local_beta,
                                                            y_data_type,
                                                            y_values)));
                return rocsparse_status_success;
            }

                // LCOV_EXCL_START
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_v2_spmv_buffer_size(rocsparse_handle            handle, //0
                                                          rocsparse_spmv_descr        descr, //1
                                                          rocsparse_const_spmat_descr mat, //2
                                                          rocsparse_const_dnvec_descr x, //3
                                                          rocsparse_const_dnvec_descr y, //4
                                                          rocsparse_v2_spmv_stage     stage, // 5
                                                          size_t* buffer_size_in_bytes, // 6
                                                          rocsparse_error* p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, mat);
    ROCSPARSE_CHECKARG_POINTER(3, x);
    ROCSPARSE_CHECKARG_POINTER(4, y);
    ROCSPARSE_CHECKARG_ENUM(5, stage);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size_in_bytes);

    //
    // Validate spmv_inputs.
    //

    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_alg()),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_operation()),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_compute_datatype()),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_scalar_datatype()),
                       rocsparse_status_invalid_value);

    switch(mat->format)
    {
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_csr:
    case rocsparse_format_csc:
    case rocsparse_format_bsr:
    case rocsparse_format_ell:
    case rocsparse_format_bell:
    {
        switch(stage)
        {
        case rocsparse_v2_spmv_stage_compute:
        case rocsparse_v2_spmv_stage_analysis:
        {
            buffer_size_in_bytes[0] = 0;
            return rocsparse_status_success;
            // LCOV_EXCL_START
        }
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value, "invalid stage");
    }
    }
    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value, "invalid format");
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_v2_spmv(rocsparse_handle            handle, //0
                                              rocsparse_spmv_descr        descr, //1
                                              const void*                 alpha, //2
                                              rocsparse_const_spmat_descr mat, //3
                                              rocsparse_const_dnvec_descr x, //4
                                              const void*                 beta, //5
                                              rocsparse_dnvec_descr       y, //6
                                              rocsparse_v2_spmv_stage     stage, // 7
                                              size_t                      buffer_size_in_bytes, // 8
                                              void*                       buffer, // 9
                                              rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, alpha);
    ROCSPARSE_CHECKARG_POINTER(3, mat);
    ROCSPARSE_CHECKARG_POINTER(4, x);
    ROCSPARSE_CHECKARG_POINTER(5, beta);
    ROCSPARSE_CHECKARG_POINTER(6, y);
    ROCSPARSE_CHECKARG_ENUM(7, stage);
    ROCSPARSE_CHECKARG(8,
                       buffer_size_in_bytes,
                       (buffer_size_in_bytes == 0 && buffer != nullptr),
                       rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(9,
                       buffer,
                       (buffer == nullptr && buffer_size_in_bytes > 0),
                       rocsparse_status_invalid_pointer);

    //
    // Validate spmv_inputs.
    //
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_alg()),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_operation()),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_compute_datatype()),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_scalar_datatype()),
                       rocsparse_status_invalid_value);

    //
    // Validate the stage.
    //
    const rocsparse_v2_spmv_stage current_stage = descr->get_stage();
    switch(stage)
    {
    case rocsparse_v2_spmv_stage_analysis:
    {
        if(current_stage == rocsparse_v2_spmv_stage_compute)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_v2_spmv_stage_analysis cannot be called after "
                "the stage rocsparse_v2_spmv_stage_compute");
        }
        else if(current_stage == rocsparse_v2_spmv_stage_analysis)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_v2_spmv_stage_analysis has already been "
                "executed");
        }
        break;
    }
    case rocsparse_v2_spmv_stage_compute:
    {
        if(current_stage == ((rocsparse_v2_spmv_stage)-1))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_v2_spmv_stage_analysis must be executed before "
                "the stage rocsparse_v2_spmv_stage_compute");
        }
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::v2_spmv(
        handle, descr, alpha, mat, x, beta, y, stage, buffer_size_in_bytes, buffer));

    //
    // Record the stage that has been executed.
    //
    descr->set_stage(stage);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
