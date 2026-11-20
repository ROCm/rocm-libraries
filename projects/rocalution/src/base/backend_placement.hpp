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

#ifndef ROCALUTION_BACKEND_PLACEMENT_HPP_
#define ROCALUTION_BACKEND_PLACEMENT_HPP_

#include "../utils/log.hpp"
#include "base_rocalution.hpp"

#include <cassert>

namespace rocalution
{
    // Not installed, so this access to BaseRocalution stays inside the library
    class BackendPlacement
    {
    public:
        // Moves obj to the backend of src. Unlike CloneBackend(), obj keeps its parallel
        // manager, which a GlobalMatrix built by the library owns
        template <typename ValueType>
        static void MoveToBackendOf(BaseRocalution<ValueType>&       obj,
                                    const BaseRocalution<ValueType>& src)
        {
            const void* src_ptr = &src;
            log_debug(&obj, "BackendPlacement::MoveToBackendOf()", src_ptr);

            assert(&obj != &src);

            obj.local_backend_ = src.local_backend_;

            if(src.is_host_())
            {
                obj.MoveToHost();
            }
            else
            {
                assert(src.is_accel_());

                obj.MoveToAccelerator();
            }
        }
    };

} // namespace rocalution

#endif // ROCALUTION_BACKEND_PLACEMENT_HPP_
