/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_DTYPE_DISPATCH_H
#define RPP_TEST_DTYPE_DISPATCH_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include "framework/config_param.hpp"

namespace rpptest {

// Maps a DType to the C++ storage type RPP uses for it.
template <DType D>
struct StorageType;
template <>
struct StorageType<DType::U8> {
    using type = Rpp8u;
};
template <>
struct StorageType<DType::I8> {
    using type = Rpp8s;
};
template <>
struct StorageType<DType::I16> {
    using type = Rpp16s;
};
template <>
struct StorageType<DType::F16> {
    using type = Rpp16f;
};
template <>
struct StorageType<DType::F32> {
    using type = Rpp32f;
};

// The storage type carried by a dispatch tag: `Element<decltype(tag)>`.
template <typename Tag>
using Element = typename Tag::type;

// Turns a runtime DType into the compile-time storage type, invoking fn with a tag whose
// ::type is that storage type:
//
//   dispatch_dtype<DType::U8, DType::F32>(cfg.dtype, [&](auto tag) {
//       run_op<Element<decltype(tag)>>(cfg, op);
//   });
//
// The dtypes are listed per call site rather than baked in, so fn is instantiated only for
// the ones the op actually supports (an op whose reference or API rejects, say, I16 never
// compiles a body for it). A dtype outside the list fails the test: the instantiation lists
// the dtypes an op runs over, so this only fires if the two lists drift apart.
template <DType... Ds, typename Fn>
void dispatch_dtype(DType dtype, Fn&& fn) {
    // Short-circuiting fold over the listed dtypes: fn runs for the first one that matches.
    const bool matched = ((dtype == Ds && (fn(StorageType<Ds>{}), true)) || ...);
    if (!matched) ADD_FAILURE() << "dtype " << dtype_name(dtype) << " not handled by this test";
}

}  // namespace rpptest

#endif  // RPP_TEST_DTYPE_DISPATCH_H
