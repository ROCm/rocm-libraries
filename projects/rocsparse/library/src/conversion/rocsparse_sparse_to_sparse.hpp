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
#include "control.h"
#include "internal/generic/rocsparse_sparse_to_sparse.h"

struct _rocsparse_sparse_to_sparse_descr
{
    typedef enum stage_
    {
        stage_buffer_size_analysis,
        stage_analysis,
        stage_buffer_size_compute,
        stage_compute
    } stage;

    rocsparse_sparse_to_sparse_alg m_alg;
    rocsparse_format               m_source_format;
    rocsparse_format               m_target_format;
    //
    rocsparse_sparse_to_sparse_stage m_stage;
    bool                             m_permissive{};
    rocsparse_spmat_descr            m_intermediate{};
    //
    size_t stage_buffer_sizes[2]{};
    bool   batched{};
    bool   is_batched() const
    {
        return this->batched;
    }
};

namespace rocsparse
{
    rocsparse_status internal_sparse_to_sparse(rocsparse_handle                 handle,
                                               rocsparse_sparse_to_sparse_descr descr,
                                               rocsparse_const_spmat_descr      source,
                                               rocsparse_spmat_descr            target,
                                               rocsparse_sparse_to_sparse_stage stage,
                                               size_t*                          buffer_size,
                                               void*                            buffer_,
                                               bool compute_buffer_size);
}
