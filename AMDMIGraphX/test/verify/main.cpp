/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "run_verify.hpp"
#include <migraphx/ranges.hpp>
#include <test.hpp>

#ifdef HAVE_GPU
#include <migraphx/gpu/analyze_streams.hpp>
#include <migraphx/gpu/target.hpp>
#endif
#ifdef HAVE_CPU
#include <migraphx/cpu/target.hpp>
#endif

inline static void check_gpu_streams(const migraphx::program& p)
{
#ifdef HAVE_GPU
    const auto* mm = p.get_main_module();
    auto races     = migraphx::gpu::analyze_streams(*mm);
    for(auto&& race : races)
    {
        std::cout << "FAILED: " << std::endl;
        std::cout << "Race condition detected for: ";
        mm->debug_print(race.ins);
        std::cout << "Should happen after: ";
        mm->debug_print(race.before);
    }
#else
    (void)p;
#endif
}

static void validate_gpu(const migraphx::program& p, const migraphx::parameter_map& m)
{
    check_gpu_streams(p);

    // Ensure the program doesn't modify the context in a dry run
    auto ctx = p.get_context();
    assert(&ctx != &p.get_context());
    EXPECT(is_shared(ctx, p.get_context()));
    p.dry_run(m);
    EXPECT(is_shared(ctx, p.get_context()));
}

int main(int argc, const char* argv[])
{
    run_verify rv;
    rv.add_validation_for("gpu", &validate_gpu);
    rv.disable_test_for(
        "cpu",
        {"test_if_lp",
         "test_if_param",
         "test_if_literal",
         "test_select_module_add",
         "test_select_module_reduce",
         "test_select_module_conv",
         "test_split_single_dyn_dim",
         "test_resize_dyn",
         "test_instancenorm_large_3d<migraphx::shape::float_type>",
         "test_instancenorm_large_3d<migraphx::shape::half_type>",
         "test_isinf<migraphx::generic_float<7, 8> >",
         "test_isinf<migraphx::bf16>",
    // these tests are disabled due issue of lossy downcast, see issue#2517
#if defined(__GNUC__) and !defined(__clang__)
         "test_batch_quant_dot_1<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, true>, "
         "float>",
         "test_quant_dot_3args_4<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, true>, "
         "float>",
         "test_quant_dot_3args_5<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, true>, "
         "float>",
         "test_batch_quant_dot_1<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, false>, "
         "float>",
         "test_quant_dot_3args_4<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, false>, "
         "float>",
         "test_quant_dot_3args_5<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, false>, "
         "float>",
         "test_batch_quant_dot_1<migraphx::fp8::float8<migraphx::fp8::f8_type::bf8, false>, "
         "float>",
         "test_quant_dot_3args_4<migraphx::fp8::float8<migraphx::fp8::f8_type::bf8, false>, "
         "float>",
         "test_quant_dot_3args_5<migraphx::fp8::float8<migraphx::fp8::f8_type::bf8, false>, "
         "float>",
         "test_batch_quant_dot_1<migraphx::fp8::float8<migraphx::fp8::f8_type::bf8, true>, "
         "float>",
         "test_quant_dot_3args_4<migraphx::fp8::float8<migraphx::fp8::f8_type::bf8, true>, "
         "float>",
         "test_quant_dot_3args_5<migraphx::fp8::float8<migraphx::fp8::f8_type::bf8, true>, "
         "float>",
#else
         "test_batch_quant_dot_1<migraphx::fp8::fp8e4m3fnuz, float>",
         "test_quant_dot_3args_4<migraphx::fp8::fp8e4m3fnuz, float>",
         "test_quant_dot_3args_5<migraphx::fp8::fp8e4m3fnuz, float>",
         "test_batch_quant_dot_1<migraphx::fp8::fp8e5m2fnuz, float>",
         "test_quant_dot_3args_4<migraphx::fp8::fp8e5m2fnuz, float>",
         "test_quant_dot_3args_5<migraphx::fp8::fp8e5m2fnuz, float>",
         "test_batch_quant_dot_1<migraphx::fp8::fp8e4m3fn, float>",
         "test_quant_dot_3args_4<migraphx::fp8::fp8e4m3fn, float>",
         "test_quant_dot_3args_5<migraphx::fp8::fp8e4m3fn, float>",
         "test_batch_quant_dot_1<migraphx::fp8::fp8e5m2, float>",
         "test_quant_dot_3args_4<migraphx::fp8::fp8e5m2, float>",
         "test_quant_dot_3args_5<migraphx::fp8::fp8e5m2, float>",
#endif
         "test_block_reduce_small<3, migraphx::shape::int8_type>",
         "test_block_reduce_small<4, migraphx::shape::int8_type>",
         "test_block_reduce_small<8, migraphx::shape::int8_type>",
         "test_block_reduce_small<16, migraphx::shape::int8_type>",
         "test_block_reduce_small<25, migraphx::shape::int8_type>",
         "test_block_reduce_small<32, migraphx::shape::int8_type>",
         "test_block_reduce_small<64, migraphx::shape::int8_type>",
         "test_block_reduce_small<67, migraphx::shape::int8_type>",
         "test_block_reduce_small<128, migraphx::shape::int8_type>",
         "test_block_reduce_small<129, migraphx::shape::int8_type>",

         // disabled because CPU does eliminate_data_type to float for everything
         "test_bitwise_and<migraphx::shape::int32_type>",
         "test_bitwise_and<migraphx::shape::uint8_type>",
         "test_unpack_int4<migraphx::shape::uint8_type>",
         "test_unpack_int4<migraphx::shape::int8_type>",
         "test_unpack_int4<migraphx::shape::uint8_type, 0>",
         "test_unpack_int4<migraphx::shape::int8_type, 0>",
         "test_bit_cast<migraphx::shape::uint8_type, migraphx::shape::int8_type>",
         "test_bit_cast<migraphx::shape::int8_type, migraphx::shape::uint8_type>",
         "test_bit_cast<migraphx::shape::fp8e4m3fn_type, migraphx::shape::fp8e4m3fnuz_type>",
         "test_bit_cast<migraphx::shape::fp8e4m3fnuz_type, migraphx::shape::fp8e4m3fn_type>"});
    rv.disable_test_for("gpu",
                        {
                            // These passes on MI300 but fails on others, same issue as CPU.
                            "test_batch_quant_dot_1<migraphx::fp8::fp8e4m3fnuz, float>",
                            "test_quant_dot_3args_4<migraphx::fp8::fp8e4m3fnuz, float>",
                            "test_quant_dot_3args_5<migraphx::fp8::fp8e4m3fnuz, float>",
                            "test_batch_quant_dot_1<migraphx::fp8::fp8e5m2fnuz, float>",
                            "test_quant_dot_3args_4<migraphx::fp8::fp8e5m2fnuz, float>",
                            "test_quant_dot_3args_5<migraphx::fp8::fp8e5m2fnuz, float>",
                            "test_batch_quant_dot_1<migraphx::fp8::fp8e4m3fn, float>",
                            "test_quant_dot_3args_4<migraphx::fp8::fp8e4m3fn, float>",
                            "test_quant_dot_3args_5<migraphx::fp8::fp8e4m3fn, float>",
                            "test_batch_quant_dot_1<migraphx::fp8::fp8e5m2, float>",
                            "test_quant_dot_3args_4<migraphx::fp8::fp8e5m2, float>",
                            "test_quant_dot_3args_5<migraphx::fp8::fp8e5m2, float>",
                        });
    rv.run(argc, argv);
}
