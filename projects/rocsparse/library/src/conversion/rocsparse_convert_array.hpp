/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-types.h"

namespace rocsparse
{
    /// @brief Convert a dense vector.
    /// @param handle_ The rocsparse handle
    /// @param target_ The target dense vector
    /// @param source_ The source dense vector
    /// @return The rocsparse status
    rocsparse_status dnvec_transfer_from(rocsparse_handle            handle_,
                                         rocsparse_dnvec_descr       target_,
                                         rocsparse_const_dnvec_descr source_);

    /// @brief Convert an indexing array.
    /// @param handle_ The rocsparse handle
    /// @param nitems_ The number of items to copy
    /// @param target_indextype_ The index type of the target array
    /// @param target_ The target array
    /// @param source_indextype_ The index type of the source array
    /// @param source_ The source array
    /// @return The rocsparse status
    rocsparse_status convert_array(rocsparse_handle    handle_,
                                   size_t              nitems_,
                                   rocsparse_indextype target_indextype_,
                                   void*               target_,
                                   rocsparse_indextype source_indextype_,
                                   const void*         source_);

    /// @brief Convert an indexing array.
    /// @param handle_ The rocsparse handle
    /// @param nitems_ The number of items to copy
    /// @param target_indextype_ The index type of the target array
    /// @param target_ The target array
    /// @param target_inc_ The increment of the target array
    /// @param source_indextype_ The index type of the source array
    /// @param source_ The source array
    /// @param source_inc_ The increment of the target array
    /// @return The rocsparse status
    rocsparse_status convert_array(rocsparse_handle    handle_,
                                   size_t              nitems_,
                                   rocsparse_indextype target_indextype_,
                                   void*               target_,
                                   int64_t             target_inc_,
                                   rocsparse_indextype source_indextype_,
                                   const void*         source_,
                                   int64_t             source_inc_);

    /// @brief Convert a numerical array.
    /// @param handle_ The rocsparse handle
    /// @param nitems_ The number of items to copy
    /// @param target_datatype_ The data type of the target array
    /// @param target_ The target array
    /// @param source_datatype_ The data type of the source array
    /// @param source_ The source array
    /// @return The rocsparse status
    rocsparse_status convert_array(rocsparse_handle   handle_,
                                   size_t             nitems_,
                                   rocsparse_datatype target_datatype_,
                                   void*              target_,
                                   rocsparse_datatype source_datatype_,
                                   const void*        source_);
}
