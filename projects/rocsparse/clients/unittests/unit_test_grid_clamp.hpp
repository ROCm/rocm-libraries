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

//
// Test-only helper shared by the large-grid unit tests (AISPARSE-699/700).
//
// The launches hardened by those tickets clamp grid.x against
// handle->properties.maxGridSize[0] and grid-stride over whatever the clamp
// drops. Reaching that clamp for real needs a count in the 1e11 range, so the
// tests shrink the limit instead and run a small problem through the same code
// path. This is the AISPARSE-702 idiom.
//
// A new header rather than an addition to unit_test_utils.hpp: that file is
// shared with the sibling unit-test branches, and a new leaf file cannot
// conflict with them.
//

// Internal handle definition: the complete _rocsparse_handle type is required to
// reach handle->properties.
#include "rocsparse_handle.hpp"

namespace rocsparse_ut
{
    // Temporarily shrink the grid.x limit that the launches clamp against, and
    // restore it on scope exit so a failed assertion cannot leak the override
    // into the next test sharing the fixture's handle.
    struct ScopedMaxGridSizeX
    {
        rocsparse_handle handle;
        int              saved;

        ScopedMaxGridSizeX(rocsparse_handle h, int limit)
            : handle(h)
            , saved(h->properties.maxGridSize[0])
        {
            handle->properties.maxGridSize[0] = limit;
        }

        ~ScopedMaxGridSizeX()
        {
            handle->properties.maxGridSize[0] = saved;
        }

        ScopedMaxGridSizeX(const ScopedMaxGridSizeX&) = delete;

        ScopedMaxGridSizeX& operator=(const ScopedMaxGridSizeX&) = delete;
    };
}
