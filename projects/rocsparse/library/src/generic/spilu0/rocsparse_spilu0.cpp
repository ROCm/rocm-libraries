/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <map>
#include <sstream>

#include "rocsparse_control.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "../conversion/rocsparse_convert_array.hpp"

#include "../precond/bsrilu0/rocsparse_bsrilu0.hpp"
#include "../precond/csrilu0/rocsparse_csrilu0.hpp"
#include "rocsparse_spilu0_descr.hpp"

#include "internal/generic/rocsparse_spilu0.h"

namespace rocsparse
{
    static rocsparse_status spilu0(rocsparse_handle       handle,
                                   rocsparse_spilu0_descr spilu0_descr,
                                   rocsparse_spmat_descr  A,
                                   rocsparse_spilu0_stage spilu0_stage,
                                   size_t                 buffer_size_in_bytes,
                                   void*                  buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG(2,
                           A,
                           (A->data_type != spilu0_descr->get_compute_datatype()),
                           rocsparse_status_not_implemented);

        const rocsparse_format       format         = A->format;
        const rocsparse_spilu0_stage previous_stage = spilu0_descr->get_stage();
        switch(spilu0_stage)
        {
        case rocsparse_spilu0_stage_analysis:
        {
            switch(previous_stage)
            {
            case rocsparse_spilu0_stage_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spilu0_stage_analysis has already "
                    "been "
                    "executed");
            }

            case rocsparse_spilu0_stage_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spilu0_stage_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spilu0_stage_compute");
            }
            }

            const rocsparse_analysis_policy analysis_policy = spilu0_descr->get_analysis_policy();
            if(rocsparse::enum_utils::is_invalid(analysis_policy))
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                       "invalid analysis_policy");
            }

            //
            // Record the matrix format.
            //
            spilu0_descr->set_format(format);
            spilu0_descr->m_batch_count = A->batch_count;
            switch(format)
            {
            case rocsparse_format_csr:
            {
                rocsparse_csrilu0_info csrilu0_info{};
                switch(analysis_policy)
                {
                case rocsparse_analysis_policy_reuse:
                {
                    spilu0_descr->set_shared_csrilu0_info(A->info->get_shared_csrilu0_info());
                    csrilu0_info = spilu0_descr->get_csrilu0_info();
                    break;
                }
                case rocsparse_analysis_policy_force:
                {
                    csrilu0_info = nullptr;
                    break;
                }
                }

                RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_analysis(handle,
                                                                       A,
                                                                       analysis_policy,
                                                                       rocsparse_solve_policy_auto,
                                                                       &csrilu0_info,
                                                                       buffer)));
                switch(analysis_policy)
                {
                case rocsparse_analysis_policy_reuse:
                {
                    break;
                }
                case rocsparse_analysis_policy_force:
                {
                    spilu0_descr->set_csrilu0_info(csrilu0_info);
                    break;
                }
                }
                spilu0_descr->set_stage(rocsparse_spilu0_stage_analysis);

                return rocsparse_status_success;
            }

            case rocsparse_format_bsr:
            {
                rocsparse_bsrilu0_info bsrilu0_info{};
                switch(analysis_policy)
                {
                case rocsparse_analysis_policy_reuse:
                {
                    spilu0_descr->set_shared_bsrilu0_info(A->info->get_shared_bsrilu0_info());
                    bsrilu0_info = spilu0_descr->get_bsrilu0_info();
                    break;
                }
                case rocsparse_analysis_policy_force:
                {
                    bsrilu0_info = nullptr;
                    break;
                }
                }
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrilu0_analysis(handle,
                                                                       A,
                                                                       analysis_policy,
                                                                       rocsparse_solve_policy_auto,
                                                                       &bsrilu0_info,
                                                                       buffer)));
                switch(analysis_policy)
                {
                case rocsparse_analysis_policy_reuse:
                {
                    break;
                }
                case rocsparse_analysis_policy_force:
                {
                    spilu0_descr->set_bsrilu0_info(bsrilu0_info);
                    break;
                }
                }
                spilu0_descr->set_stage(rocsparse_spilu0_stage_analysis);
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_sell:
            case rocsparse_format_bell:
            case rocsparse_format_coo_aos:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                // LCOV_EXCL_STOP
            }
            }
        }
        case rocsparse_spilu0_stage_compute:
        {

            if(previous_stage == ((rocsparse_spilu0_stage)-1))
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spilu0_stage_analysis must be executed "
                    "before "
                    "the stage rocsparse_spilu0_stage_compute");
            }

            spilu0_descr->m_batch_count = A->batch_count;
            switch(format)
            {
            case rocsparse_format_csr:
            {
                auto csrilu0_info = spilu0_descr->get_csrilu0_info();

                if(spilu0_descr->get_tolerance_pointer() != nullptr)
                {
                    if(csrilu0_info != nullptr)
                    {
                        csrilu0_info->get_singularity_numeric_near()->set_tolerance_pointer(
                            spilu0_descr->get_tolerance_pointer(),
                            spilu0_descr->get_tolerance_pointer_mode(),
                            rocsparse_datatype_f64_r);
                        spilu0_descr->set_tolerance_pointer(nullptr);
                    }
                }

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0(handle,
                                                             A,
                                                             rocsparse_solve_policy_auto,
                                                             csrilu0_info,
                                                             spilu0_descr->get_boost(),
                                                             buffer_size_in_bytes,
                                                             buffer));
                spilu0_descr->set_stage(rocsparse_spilu0_stage_compute);
                return rocsparse_status_success;
            }

            case rocsparse_format_bsr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrilu0(handle,
                                                             spilu0_descr->get_bsrilu0_info(),
                                                             A,
                                                             spilu0_descr->get_boost(),
                                                             buffer_size_in_bytes,
                                                             buffer));
                spilu0_descr->set_stage(rocsparse_spilu0_stage_compute);
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_sell:
            case rocsparse_format_bell:
            case rocsparse_format_coo_aos:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                // LCOV_EXCL_STOP
            }
            }
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spilu0(rocsparse_handle            handle, // 0
                                             rocsparse_spilu0_descr      spilu0_descr, // 1
                                             rocsparse_const_spmat_descr A, // 2
                                             rocsparse_spmat_descr       P, // 3
                                             rocsparse_spilu0_stage      spilu0_stage, // 4
                                             size_t                      buffer_size_in_bytes, // 5
                                             void*                       buffer, // 6
                                             rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, spilu0_descr);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, P);

    ROCSPARSE_CHECKARG(3, P, (A != P), rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ENUM(4, spilu0_stage);

    ROCSPARSE_CHECKARG(5,
                       buffer_size_in_bytes,
                       (buffer_size_in_bytes == 0) && (buffer != nullptr),
                       rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG(6,
                       buffer,
                       (buffer == nullptr) && (buffer_size_in_bytes != 0),
                       rocsparse_status_invalid_pointer);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spilu0(handle, spilu0_descr, P, spilu0_stage, buffer_size_in_bytes, buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
