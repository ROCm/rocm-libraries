// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

block_histo_test_suite_type_def(suite_name_atomic, name_suffix)
block_histo_test_suite_type_def(suite_name_sort, name_suffix)

typed_test_suite_def(suite_name_atomic, name_suffix, block_params_atomic);
typed_test_suite_def(suite_name_sort, name_suffix, block_params_sort);

typed_test_def(suite_name_atomic, name_suffix, Histogram)
{
    using T = typename TestFixture::type;
    using BinType = typename TestFixture::bin_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 4, T, BinType, block_size, rocprim::block_histogram_algorithm::using_atomic>::run();
}

typed_test_def(suite_name_sort, name_suffix, Histogram)
{
    using T = typename TestFixture::type;
    using BinType = typename TestFixture::bin_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 4, T, BinType, block_size, rocprim::block_histogram_algorithm::using_sort>::run();
}
