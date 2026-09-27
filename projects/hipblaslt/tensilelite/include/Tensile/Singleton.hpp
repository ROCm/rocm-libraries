/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <mutex>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    template <typename Class>
    class LazySingleton
    {
    public:
        // TENSILELITEHOST_EXPORT is required, not decorative. `Instance()` is an
        // implicitly-inline member of a header-only template, and both
        // libhipblaslt and libtensilelite-host are built with hidden visibility
        // plus VISIBILITY_INLINES_HIDDEN. Without an explicit default-visibility
        // attribute each shared object gets its OWN copy of `instance`, so state
        // written through the singleton in one library is invisible in the other
        // -- silently, with no link error. See ROCM-31245.
        static TENSILELITEHOST_EXPORT Class& Instance()
        {
            static Class instance;

            return instance;
        }

    private:
    };
} // namespace TensileLite

