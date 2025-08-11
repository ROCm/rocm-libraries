/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_mat_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

rocsparse_csrmv_info _rocsparse_mat_info::get_csrmv_info()
{
    return this->csrmv_info;
}

void _rocsparse_mat_info::set_csrmv_info(rocsparse_csrmv_info value)
{
    this->csrmv_info = value;
}

rocsparse_bsrmv_info _rocsparse_mat_info::get_bsrmv_info()
{
    return this->bsrmv_info;
}

void _rocsparse_mat_info::set_bsrmv_info(rocsparse_bsrmv_info value)
{
    this->bsrmv_info = value;
}

void _rocsparse_mat_info::set_sorted_coo2csr_info(rocsparse::sorted_coo2csr_info_t* value)
{
    this->m_sorted_coo2csr_info = value;
}

rocsparse::sorted_coo2csr_info_t* _rocsparse_mat_info::get_sorted_coo2csr_info()
{
    return this->m_sorted_coo2csr_info;
}

_rocsparse_mat_info::~_rocsparse_mat_info()
{
    // Uncouple shared meta data
    if(this->bsrsv_lower_info == this->bsrilu0_info || this->bsrsv_lower_info == this->bsric0_info
       || this->bsrsv_lower_info == this->bsrsm_lower_info)
    {
        this->bsrsv_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->bsrsm_lower_info == this->bsrilu0_info || this->bsrsm_lower_info == this->bsric0_info)
    {
        this->bsrsm_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->bsrilu0_info == this->bsric0_info)
    {
        this->bsrilu0_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->csrsv_lower_info == this->csrilu0_info || this->csrsv_lower_info == this->csric0_info
       || this->csrsv_lower_info == this->csrsm_lower_info)
    {
        this->csrsv_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->csrsm_lower_info == this->csrilu0_info || this->csrsm_lower_info == this->csric0_info)
    {
        this->csrsm_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->csrilu0_info == this->csric0_info)
    {
        this->csrilu0_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->csrsv_upper_info == this->csrsm_upper_info)
    {
        this->csrsv_upper_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->bsrsv_upper_info == this->bsrsm_upper_info)
    {
        this->bsrsv_upper_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->csrsvt_lower_info == this->csrsmt_lower_info)
    {
        this->csrsvt_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->bsrsvt_lower_info == this->bsrsmt_lower_info)
    {
        this->bsrsvt_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->csrsvt_upper_info == this->csrsmt_upper_info)
    {
        this->csrsvt_upper_info = nullptr;
    }

    // Uncouple shared meta data
    if(this->bsrsvt_upper_info == this->bsrsmt_upper_info)
    {
        this->bsrsvt_upper_info = nullptr;
    }

    // Clear bsrsvt upper info struct
    rocsparse::trm_info_t::destroy(this->bsrsvt_upper_info);

    // Clear bsrsvt lower info struct
    rocsparse::trm_info_t::destroy(this->bsrsvt_lower_info);

    // Clear bsric0 info struct
    rocsparse::trm_info_t::destroy(this->bsric0_info);

    // Clear bsrilu0 info struct
    rocsparse::trm_info_t::destroy(this->bsrilu0_info);

    // Clear csrsvt upper info struct
    rocsparse::trm_info_t::destroy(this->csrsvt_upper_info);

    // Clear csrsvt lower info struct
    rocsparse::trm_info_t::destroy(this->csrsvt_lower_info);

    // Clear csrsmt upper info struct
    rocsparse::trm_info_t::destroy(this->csrsmt_upper_info);

    // Clear csrsmt lower info struct
    rocsparse::trm_info_t::destroy(this->csrsmt_lower_info);

    // Clear bsrsmt upper info struct
    rocsparse::trm_info_t::destroy(this->bsrsmt_upper_info);

    // Clear bsrsmt lower info struct
    rocsparse::trm_info_t::destroy(this->bsrsmt_lower_info);

    // Clear csric0 info struct
    rocsparse::trm_info_t::destroy(this->csric0_info);

    // Clear csrilu0 info struct
    rocsparse::trm_info_t::destroy(this->csrilu0_info);

    // Clear bsrsv upper info struct
    rocsparse::trm_info_t::destroy(this->bsrsv_upper_info);

    // Clear bsrsv lower info struct
    rocsparse::trm_info_t::destroy(this->bsrsv_lower_info);

    // Clear csrsv upper info struct
    rocsparse::trm_info_t::destroy(this->csrsv_upper_info);

    // Clear csrsv lower info struct
    rocsparse::trm_info_t::destroy(this->csrsv_lower_info);

    // Clear csrsm upper info struct
    rocsparse::trm_info_t::destroy(this->csrsm_upper_info);

    // Clear csrsm lower info struct
    rocsparse::trm_info_t::destroy(this->csrsm_lower_info);

    // Clear bsrsm upper info struct
    rocsparse::trm_info_t::destroy(this->bsrsm_upper_info);

    // Clear bsrsm lower info struct
    rocsparse::trm_info_t::destroy(this->bsrsm_lower_info);

    // Clear csrgemm info struct
    WARNING_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrgemm_info(this->csrgemm_info));

    // Clear csritsv info struct
    WARNING_IF_ROCSPARSE_ERROR(rocsparse::destroy_csritsv_info(this->csritsv_info));

    // Clear zero pivot
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->zero_pivot));

    // Clear singular pivot
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->singular_pivot));

    if(this->csrmv_info != nullptr)
    {
        delete this->csrmv_info;
    }

    if(this->bsrmv_info != nullptr)
    {
        delete this->bsrmv_info;
    }

    rocsparse::sorted_coo2csr_info_t* sorted_coo2csr_info = this->get_sorted_coo2csr_info();
    if(sorted_coo2csr_info != nullptr)
    {
        hipStream_t default_stream = 0;
        std::ignore                = sorted_coo2csr_info->free_memory(default_stream);

        delete sorted_coo2csr_info;
        this->set_sorted_coo2csr_info(nullptr);
    }
}

/********************************************************************************
 * \brief check_trm_shared checks if the given trm info structure
 * shares its meta data with another trm info structure.
 *******************************************************************************/
bool rocsparse::check_trm_shared(const rocsparse_mat_info info, rocsparse::trm_info_t* trm)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(info == nullptr)
    {
        return false;
    }

    int shared = -1;

    if(trm == info->bsrsv_lower_info)
        ++shared;
    if(trm == info->bsrsv_upper_info)
        ++shared;
    if(trm == info->bsrsvt_lower_info)
        ++shared;
    if(trm == info->bsrsvt_upper_info)
        ++shared;
    if(trm == info->bsrilu0_info)
        ++shared;
    if(trm == info->bsric0_info)
        ++shared;
    if(trm == info->csrilu0_info)
        ++shared;
    if(trm == info->csric0_info)
        ++shared;
    if(trm == info->csrsv_lower_info)
        ++shared;
    if(trm == info->csrsv_upper_info)
        ++shared;
    if(trm == info->csrsvt_lower_info)
        ++shared;
    if(trm == info->csrsvt_upper_info)
        ++shared;
    if(trm == info->csrsm_lower_info)
        ++shared;
    if(trm == info->csrsm_upper_info)
        ++shared;
    if(trm == info->bsrsm_lower_info)
        ++shared;
    if(trm == info->bsrsm_upper_info)
        ++shared;

    return (shared > 0) ? true : false;
}
