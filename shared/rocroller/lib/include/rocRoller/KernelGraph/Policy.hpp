/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

namespace rocRoller
{
    class CacheOnlyPolicy // Use cache to look up order. Caller should ensure the cache is valid.
    {
    };
    class UpdateCachePolicy // < Use cache to look up order. Cache will be re-built if invalid.
    {
    };
    class UseCacheIfAvailablePolicy // < Look up in cache if available, otherwise, use traversal.
    {
    };
    class IgnoreCachePolicy // < Use traversal.
    {
    };

    inline constexpr CacheOnlyPolicy           CacheOnly;
    inline constexpr UpdateCachePolicy         UpdateCache;
    inline constexpr UseCacheIfAvailablePolicy UseCacheIfAvailable;
    inline constexpr IgnoreCachePolicy         IgnoreCache;
}
