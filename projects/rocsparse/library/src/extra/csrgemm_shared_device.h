/*! \file */
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

#pragma once

#include "rocsparse_common.hpp"

namespace rocsparse
{
    // Row classification (both the numeric and symbolic csrgemm paths) buckets rows
    // by their intermediate product count into a shared-memory hash table of
    // HASHSIZE entries, each entry holding one IndexType key. This checks whether
    // that bare keys-only footprint already reaches the device's opt-in dynamic LDS
    // limit, before the numeric/symbolic kernels add their own extra per-entry
    // payload (values, bitmap, ...) on top via exceeding_smem / exceeding_smem_symbolic.
    template <uint32_t HASHSIZE, typename IndexType>
    constexpr bool exceeding_smem_keys(uint32_t shared_mem_optin)
    {
        // Use >= rather than > : a hash that exactly fills the advertised LDS
        // limit still needs to pass through hipFuncSetAttribute, which some
        // runtimes (e.g. under ASAN instrumentation) reject when the request
        // equals rather than stays strictly below the device's optin limit.
        return (sizeof(IndexType) * HASHSIZE) >= shared_mem_optin;
    }
}
