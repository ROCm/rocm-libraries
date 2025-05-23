// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef _CLANGD
    // When using clangd, to allow the language server to function properly,
    // some context of the template source is required to allow the language
    // server to function properly.
    #include "test_block_discontinuity.cpp.in"
#endif

test_suite_type_def(suite_name, name_suffix)

typed_test_suite_def(RocprimBlockDiscontinuity, name_suffix, warp_params);

typed_test_def(RocprimBlockDiscontinuity, name_suffix, FlagHeads)
{
    using type                  = typename TestFixture::params::input_type;
    using flag_type             = typename TestFixture::params::output_type;
    using flag_op_type_1        = rocprim::less<type>;
    using flag_op_type_2        = rocprim::equal_to<type>;
    using flag_op_type_3        = rocprim::greater<type>;
    using flag_op_type_4        = rocprim::not_equal_to<type>;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 2, type, flag_type, flag_op_type_1, 0, block_size>::run();
    static_for<2, 4, type, flag_type, flag_op_type_2, 0, block_size>::run();
    static_for<4, 6, type, flag_type, flag_op_type_3, 0, block_size>::run();
    static_for<6, n_items, type, flag_type, flag_op_type_4, 0, block_size>::run();
}

typed_test_def(RocprimBlockDiscontinuity, name_suffix, FlagTails)
{
    using type                  = typename TestFixture::params::input_type;
    using flag_type             = typename TestFixture::params::output_type;
    using flag_op_type_1        = rocprim::less<type>;
    using flag_op_type_2        = rocprim::equal_to<type>;
    using flag_op_type_3        = rocprim::greater<type>;
    using flag_op_type_4        = rocprim::not_equal_to<type>;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 2, type, flag_type, flag_op_type_1, 1, block_size>::run();
    static_for<2, 4, type, flag_type, flag_op_type_2, 1, block_size>::run();
    static_for<4, 6, type, flag_type, flag_op_type_3, 1, block_size>::run();
    static_for<6, n_items, type, flag_type, flag_op_type_4, 1, block_size>::run();
}

typed_test_def(RocprimBlockDiscontinuity, name_suffix, FlagHeadsAndTails)
{
    using type                  = typename TestFixture::params::input_type;
    using flag_type             = typename TestFixture::params::output_type;
    using flag_op_type_1        = rocprim::less<type>;
    using flag_op_type_2        = rocprim::equal_to<type>;
    using flag_op_type_3        = rocprim::greater<type>;
    using flag_op_type_4        = rocprim::not_equal_to<type>;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 2, type, flag_type, flag_op_type_1, 2, block_size>::run();
    static_for<2, 4, type, flag_type, flag_op_type_2, 2, block_size>::run();
    static_for<4, 6, type, flag_type, flag_op_type_3, 2, block_size>::run();
    static_for<6, n_items, type, flag_type, flag_op_type_4, 2, block_size>::run();
}
