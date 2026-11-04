// Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TEST_TEST_UTILS_TYPES_HPP_
#define TEST_TEST_UTILS_TYPES_HPP_

#include "test_utils.hpp"

// Global utility defines
#define test_suite_type_def_helper(name, suffix) \
template<class Params> \
class name ## suffix : public ::testing::Test { \
public: \
    using params = Params; \
};

#define test_suite_type_def(name, suffix) test_suite_type_def_helper(name, suffix)

#define typed_test_suite_def_helper(name, suffix, params) TYPED_TEST_SUITE(name ## suffix, params)

#define typed_test_suite_def(name, suffix, params) typed_test_suite_def_helper(name, suffix, params)

#define typed_test_def_helper(suite, suffix, name) TYPED_TEST(suite ## suffix, name)

#define typed_test_def(suite, suffix, name) typed_test_def_helper(suite, suffix, name)

#endif // TEST_TEST_UTILS_TYPES_HPP_
