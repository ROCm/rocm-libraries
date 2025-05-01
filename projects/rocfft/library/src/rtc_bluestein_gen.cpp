// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rtc_bluestein_gen.h"
#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "device/generator/fftgenerator.h"
#include "device/generator/generator.h"
#include "device/kernel-generator-embed.h"
#include "rtc_kernel.h"
#include "rtc_test_harness.h"

std::string bluestein_single_rtc_kernel_name(const BluesteinSingleSpecs& specs)
{
    std::string kernel_name = "bluestein_single";

    if(specs.direction == -1)
        kernel_name += "_fwd";
    else
        kernel_name += "_back";

    kernel_name += "_len";
    kernel_name += std::to_string(specs.length);

    kernel_name += "_dim";
    kernel_name += std::to_string(specs.dim);

    kernel_name += rtc_precision_name(specs.precision);

    if(specs.placement == rocfft_placement_inplace)
    {
        kernel_name += "_ip";
        kernel_name += rtc_array_type_name(specs.inArrayType);
    }
    else
    {
        kernel_name += "_op";
        kernel_name += rtc_array_type_name(specs.inArrayType);
        kernel_name += rtc_array_type_name(specs.outArrayType);
    }

    kernel_name += load_store_name_suffix(specs.loadOps, specs.storeOps);
    kernel_name += rtc_cbtype_name(specs.cbtype);
    return kernel_name;
}

std::string bluestein_single_rtc(const std::string& kernel_name, const BluesteinSingleSpecs& specs)
{
    auto length               = specs.length;
    auto lengthBlue           = product(specs.factors.begin(), specs.factors.end());
    auto transforms_per_block = specs.threads_per_block / specs.threads_per_transform;

    std::string src;

    // includes and declarations
    src += rocfft_complex_h;
    src += common_h;
    src += callback_h;

    src += butterfly_constant_h;
    append_radix_h(src, specs.factors);
    src += rtc_precision_type_decl(specs.precision);

    src += rtc_const_cbtype_decl(specs.cbtype);

    src += "static const unsigned int dim = " + std::to_string(specs.dim) + ";\n";

    Function func{kernel_name};
    func.launch_bounds = specs.threads_per_block;
    func.qualifier     = "extern \"C\" __global__";

    auto ctx        = std::make_shared<Context>();
    auto bluestein  = BluesteinTransform{length,
                                        lengthBlue,
                                        specs.direction,
                                        specs.factors,
                                        specs.threads_per_block,
                                        specs.threads_per_transform,
                                        ctx};
    auto operations = bluestein.generate();

    func.arguments = ctx->get_arguments();
    for(auto& v : ctx->get_locals())
        func.body += Declaration{v};

    Variable lds{"lds", "__shared__ scalar_type", false, false, transforms_per_block * lengthBlue};

    func.body += CallbackLoadDeclaration("scalar_type", "cbtype");
    func.body += CallbackStoreDeclaration("scalar_type", "cbtype");

    func.body += Declaration{lds};
    func.body += Assign{bluestein.a, bluestein.buf_temp};
    func.body += Assign{bluestein.B, bluestein.buf_temp + lengthBlue};
    func.body += Assign{bluestein.A, lds};

    func.body += operations.lower();

    make_load_store_ops(func, specs.loadOps, specs.storeOps);

    if(specs.placement == rocfft_placement_notinplace)
    {
        func = make_outofplace(func, "X", false);

        if(array_type_is_planar(specs.inArrayType))
            func = make_planar(func, "X_in");
        if(array_type_is_planar(specs.outArrayType))
            func = make_planar(func, "X_out");
    }
    else
    {
        if(array_type_is_planar(specs.inArrayType))
            func = make_planar(func, "X");
    }

    func = make_callback_realcomplex(func, specs.cbtype);

    src += func.render();

    write_standalone_test_harness(func, src);
    return src;
}

std::string bluestein_multi_rtc_kernel_name(const BluesteinMultiSpecs& specs)
{
    std::string kernel_name;
    switch(specs.scheme)
    {
    case CS_KERNEL_CHIRP:
        kernel_name += "bluestein_chirp";
        break;
    case CS_KERNEL_PAD_MUL:
        kernel_name += "bluestein_pad_mul";
        break;
    case CS_KERNEL_FFT_MUL:
        kernel_name += "bluestein_fft_mul";
        break;
    case CS_KERNEL_RES_MUL:
        kernel_name += "bluestein_res_mul";
        break;
    default:
        throw std::runtime_error("invalid bluestein rtc scheme");
    }

    kernel_name += rtc_precision_name(specs.precision);
    kernel_name += rtc_array_type_name(specs.inArrayType);
    kernel_name += rtc_array_type_name(specs.outArrayType);
    kernel_name += load_store_name_suffix(specs.loadOps, specs.storeOps);
    kernel_name += rtc_cbtype_name(specs.cbtype);

    return kernel_name;
}

static std::string bluestein_multi_chirp_rtc(const std::string&         kernel_name,
                                             const BluesteinMultiSpecs& specs)
{
    // function arguments
    Variable N{"N", "const size_t"};
    Variable M{"M", "const size_t"};
    Variable output{"output", "scalar_type", true, true};
    Variable twiddles_large{"twiddles_large", "const scalar_type", true, true};
    Variable twl{"twl", "const int"};
    Variable dir{"dir", "const int"};

    Function func{kernel_name};
    func.launch_bounds = LAUNCH_BOUNDS_BLUESTEIN_MULTI_KERNEL;
    func.qualifier     = "extern \"C\" __global__";
    func.arguments.append(N);
    func.arguments.append(M);
    func.arguments.append(output);
    func.arguments.append(twiddles_large);
    func.arguments.append(twl);
    func.arguments.append(dir);

    Variable tx{"tx", "size_t"};
    Variable val{"val", "scalar_type"};

    func.body += Declaration{tx, "threadIdx.x + blockIdx.x * blockDim.x"};
    func.body += Declaration{val, CallExpr{"scalar_type", {Literal{"0.0"}, Literal{"0.0"}}}};

    func.body
        += If{twl == 1, {Assign{val, CallExpr{"TWLstep1", {twiddles_large, (tx * tx) % (2 * N)}}}}};
    func.body += ElseIf{twl == 2,
                        {Assign{val, CallExpr{"TWLstep2", {twiddles_large, (tx * tx) % (2 * N)}}}}};
    func.body += ElseIf{twl == 3,
                        {Assign{val, CallExpr{"TWLstep3", {twiddles_large, (tx * tx) % (2 * N)}}}}};
    func.body += ElseIf{twl == 4,
                        {Assign{val, CallExpr{"TWLstep4", {twiddles_large, (tx * tx) % (2 * N)}}}}};

    func.body += MultiplyAssign(val.y(), CallExpr{"real_type_t<scalar_type>", {dir}});

    func.body += If{tx == 0,
                    {
                        Assign{output[tx], val},
                        Assign{output[tx + M], val},
                    }};
    func.body += ElseIf{tx < N,
                        {Assign{output[tx], val},
                         Assign{output[tx + M], val},

                         Assign{output[M - tx], val},
                         Assign{output[M - tx + M], val}}};
    func.body += ElseIf{
        tx <= (M - N),
        {Assign{output[tx], CallExpr{"scalar_type", {Literal{"0.0"}, Literal{"0.0"}}}},
         Assign{output[tx + M], CallExpr{"scalar_type", {Literal{"0.0"}, Literal{"0.0"}}}}}};

    auto src = func.render();
    write_standalone_test_harness(func, src);
    return src;
}

std::string bluestein_multi_rtc(const std::string& kernel_name, const BluesteinMultiSpecs& specs)
{
    std::string src;
    // includes and declarations
    src += rocfft_complex_h;
    src += common_h;
    src += callback_h;

    src += rtc_precision_type_decl(specs.precision);

    src += rtc_const_cbtype_decl(specs.cbtype);

    // chirp looks different from the other kernels
    if(specs.scheme == CS_KERNEL_CHIRP)
    {
        src += bluestein_multi_chirp_rtc(kernel_name, specs);
        return src;
    }

    // function arguments
    Variable numof{"numof", "const size_t"};
    Variable totalWI{"totalWI", "const size_t"};
    Variable N{"N", "const size_t"};
    Variable M{"M", "const size_t"};
    Variable input{"input", "scalar_type", true, true};
    Variable output{"output", "scalar_type", true, true};
    Variable dim{"dim", "const size_t"};
    Variable lengths{"lengths", "const size_t", true, true};
    Variable stride_in{"stride_in", "const size_t", true, true};
    Variable stride_out{"stride_out", "const size_t", true, true};
    Variable scale_factor{"scale_factor", "const real_type_t<scalar_type>"};

    Function func{kernel_name};
    func.launch_bounds = LAUNCH_BOUNDS_BLUESTEIN_MULTI_KERNEL;
    func.qualifier     = "extern \"C\" __global__";
    func.arguments.append(numof);
    func.arguments.append(totalWI);
    func.arguments.append(N);
    func.arguments.append(M);
    func.arguments.append(input);
    func.arguments.append(output);
    func.arguments.append(dim);
    func.arguments.append(lengths);
    func.arguments.append(stride_in);
    func.arguments.append(stride_out);
    for(const auto& arg : get_callback_args().arguments)
        func.arguments.append(arg);
    func.arguments.append(scale_factor);

    // local variables
    Variable tx{"tx", "size_t"};
    Variable iOffset{"iOffset", "size_t"};
    Variable oOffset{"oOffset", "size_t"};
    Variable counter_mod{"counter_mod", "size_t"};
    Variable currentLength{"currentLength", "size_t"};
    Variable i{"i", "size_t"};
    Variable j{"j", "size_t"};
    Variable iIdx{"iIdx", "size_t"};
    Variable oIdx{"oIdx", "size_t"};
    Variable chirp{"chirp", "scalar_type", true};
    Variable out_elem{"out_elem", "scalar_type"};

    func.body += Declaration{tx, "threadIdx.x + blockIdx.x * blockDim.x"};

    func.body += If{tx >= totalWI, {Return{}}};

    func.body += Declaration{iOffset, 0};
    func.body += Declaration{oOffset, 0};
    func.body += Declaration{counter_mod, tx / numof};

    For iLoop{i, dim, i > 1, -1};

    iLoop.body += Declaration{currentLength, 1};

    For jLoop{j, 1, j < i, 1, {MultiplyAssign(currentLength, lengths[j])}};

    iLoop.body += jLoop;
    iLoop.body += AddAssign(iOffset, (counter_mod / currentLength) * stride_in[i]);
    iLoop.body += AddAssign(oOffset, (counter_mod / currentLength) * stride_out[i]);
    iLoop.body += Assign{counter_mod, counter_mod % currentLength};

    func.body += iLoop;

    func.body += AddAssign(iOffset, counter_mod * stride_in[1]);
    func.body += AddAssign(oOffset, counter_mod * stride_out[1]);

    func.body += Assign{tx, tx % numof};
    func.body += Declaration{iIdx, tx * stride_in[0]};
    func.body += Declaration{oIdx, tx * stride_out[0]};

    func.body += CallbackLoadDeclaration("scalar_type", "cbtype");
    func.body += CallbackStoreDeclaration("scalar_type", "cbtype");

    switch(specs.scheme)
    {
    case CS_KERNEL_PAD_MUL:
    {
        func.body += CommentLines{"PAD_MUL is the first non-chirp step of bluestein and",
                                  "should never be the last kernel to write global memory.",
                                  "So we should never need to run a \"store\" callback."};

        func.body += Declaration{chirp, output};
        func.body += AddAssign(iIdx, iOffset);
        func.body += AddAssign(oIdx, M);
        func.body += AddAssign(oIdx, oOffset);

        Variable in_elem{"in_elem", "scalar_type"};
        If       readBlock{tx < N, {}};
        readBlock.body += Declaration{in_elem};
        readBlock.body += Assign{in_elem, LoadGlobal{input, iIdx}};
        readBlock.body
            += Assign{output[oIdx].x(), in_elem.x() * chirp[tx].x() + in_elem.y() * chirp[tx].y()};
        readBlock.body
            += Assign{output[oIdx].y(), -in_elem.x() * chirp[tx].y() + in_elem.y() * chirp[tx].x()};
        func.body += readBlock;
        func.body += Else{
            {Assign{output[oIdx], CallExpr{"scalar_type", {Literal{"0.0"}, Literal{"0.0"}}}}}};
        break;
    }
    case CS_KERNEL_FFT_MUL:
    {
        func.body += CommentLines{"FFT_MUL is in the middle of bluestein and should never be",
                                  "the first/last kernel to read/write global memory.  So we",
                                  "don't need to run callbacks."};
        func.body += AddAssign(output, oOffset);
        func.body += Declaration{out_elem, output[oIdx]};
        func.body += Assign{output[oIdx].x(),
                            input[iIdx].x() * out_elem.x() - input[iIdx].y() * out_elem.y()};
        func.body += Assign{output[oIdx].y(),
                            input[iIdx].x() * out_elem.y() + input[iIdx].y() * out_elem.x()};
        break;
    }
    case CS_KERNEL_RES_MUL:
    {
        func.body += CommentLines{"RES_MUL is the last step of bluestein and",
                                  "should never be the first kernel to read global memory.",
                                  "So we should never need to run a \"load\" callback."};
        func.body += Declaration{chirp, input};
        func.body += AddAssign(iIdx, 2 * M);
        func.body += AddAssign(iIdx, iOffset);
        func.body += AddAssign(oIdx, oOffset);

        Variable MI{"MI", "real_type_t<scalar_type>"};
        func.body += Declaration{MI, Literal{"1.0"} / CallExpr{"real_type_t<scalar_type>", {M}}};
        func.body += Declaration{out_elem};
        func.body += Assign{
            out_elem.x(), MI * (input[iIdx].x() * chirp[tx].x() + input[iIdx].y() * chirp[tx].y())};
        func.body
            += Assign{out_elem.y(),
                      MI * (-input[iIdx].x() * chirp[tx].y() + input[iIdx].y() * chirp[tx].x())};
        func.body += StoreGlobal{output, oIdx, out_elem};
        break;
    }
    default:
        throw std::runtime_error("invalid bluestein rtc scheme");
    }

    make_load_store_ops(func, specs.loadOps, specs.storeOps);

    if(array_type_is_planar(specs.inArrayType))
        func = make_planar(func, "input");
    if(array_type_is_planar(specs.outArrayType))
        func = make_planar(func, "output");

    src += func.render();
    write_standalone_test_harness(func, src);
    return src;
}
