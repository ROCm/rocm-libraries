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
// Test-only helper for the conversion-directory grid tests
// (AISPARSE-684/685/686).
//
// The launches hardened by those tickets clamp grid.x against
// handle->properties.maxGridSize[0] and grid-stride over whatever the clamp
// drops. Reaching that clamp for real needs more than 2^32 - 1 work-items, e.g.
// 2^27 CSR rows (684) or 2^32 elements (685, 686), so the tests shrink the limit instead
// and run a small problem through the same code path. This is the AISPARSE-702
// idiom.
//
// A new leaf header rather than an addition to unit_test_utils.hpp: that file is
// shared with every sibling unit-test branch, and a new file cannot conflict
// with them. The sibling AISPARSE-699/700 branch adds an equivalent
// unit_test_grid_clamp.hpp; this one is deliberately named differently so the
// two branches do not collide on an add/add. Whichever merges second should
// delete its copy and include the other.
//

// Internal handle definition: the complete _rocsparse_handle type is required to
// reach handle->properties.
#include "rocsparse_handle.hpp"

#include <ostream>
#include <sstream>

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

    // Turn on the debug log and send it to `sink` for the lifetime of the object, for the
    // same reason as ScopedMaxGridSizeX: the handle must not keep pointing at the sink
    // after a failed assertion ends the test.
    struct ScopedDebugLog
    {
        rocsparse_handle     handle;
        rocsparse_layer_mode saved_mode;
        std::ostream*        saved_os;
        std::ostringstream   sink;

        explicit ScopedDebugLog(rocsparse_handle h)
            : handle(h)
            , saved_mode(h->layer_mode)
            , saved_os(h->log_debug_os)
        {
            handle->layer_mode = rocsparse_layer_mode(saved_mode | rocsparse_layer_mode_log_debug);
            handle->log_debug_os = &sink;
        }

        ~ScopedDebugLog()
        {
            handle->layer_mode   = saved_mode;
            handle->log_debug_os = saved_os;
        }

        ScopedDebugLog(const ScopedDebugLog&) = delete;

        ScopedDebugLog& operator=(const ScopedDebugLog&) = delete;
    };
}
