// Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rtc_realcomplex_gen.h"
#include "../../shared/array_predicate.h"
#include "device/generator/generator.h"
#include "rtc_test_harness.h"

#include "device/kernel-generator-embed.h"

// generate name for RTC realcomplex kernel
std::string realcomplex_rtc_kernel_name(const RealComplexSpecs& specs)
{
    std::string kernel_name;

    switch(specs.scheme)
    {
    case CS_KERNEL_COPY_R_TO_CMPLX:
        kernel_name += "r2c_copy_rtc";
        break;
    case CS_KERNEL_COPY_CMPLX_TO_HERM:
        kernel_name += "c2herm_copy_rtc";
        break;
    case CS_KERNEL_COPY_CMPLX_TO_R:
        kernel_name += "c2r_copy_rtc";
        break;
    case CS_KERNEL_COPY_HERM_TO_CMPLX:
        kernel_name += "herm2c_copy_rtc";
        break;
    default:
        throw std::runtime_error("invalid realcomplex rtc scheme");
    }

    kernel_name += "_dim" + std::to_string(specs.dim);

    kernel_name += rtc_precision_name(specs.precision);
    kernel_name += rtc_array_type_name(specs.inArrayType);
    kernel_name += rtc_array_type_name(specs.outArrayType);

    kernel_name += load_store_name_suffix(specs.loadOps, specs.storeOps);
    kernel_name += rtc_cbtype_name(specs.cbtype);

    return kernel_name;
}

std::string r2c_copy_rtc(const std::string& kernel_name, const RealComplexSpecs& specs)
{
    std::string src;
    // includes and declarations

    src += rocfft_complex_h;
    src += common_h;
    src += device_enum_h;
    src += callback_h;

    src += rtc_precision_type_decl(specs.precision);

    src += rtc_const_cbtype_decl(specs.cbtype);

    src += "static const unsigned int dim = " + std::to_string(specs.dim) + ";\n";

    const char* input_type
        = specs.scheme == CS_KERNEL_COPY_R_TO_CMPLX ? "real_type_t<scalar_type>" : "scalar_type";
    const char* output_type
        = specs.scheme == CS_KERNEL_COPY_CMPLX_TO_R ? "real_type_t<scalar_type>" : "scalar_type";

    // function arguments
    Variable hermitian_size{"hermitian_size", "const unsigned int"};
    Variable lengths0{"lengths0", "unsigned int"};
    Variable lengths1{"lengths1", "unsigned int"};
    Variable lengths2{"lengths2", "unsigned int"};
    Variable nbatch{"nbatch", "unsigned int"};
    Variable stride_in0{"stride_in0", "unsigned int"};
    Variable stride_in1{"stride_in1", "unsigned int"};
    Variable stride_in2{"stride_in2", "unsigned int"};
    Variable stride_in3{"stride_in3", "unsigned int"};
    Variable stride_out0{"stride_out0", "unsigned int"};
    Variable stride_out1{"stride_out1", "unsigned int"};
    Variable stride_out2{"stride_out2", "unsigned int"};
    Variable stride_out3{"stride_out3", "unsigned int"};
    Variable input{"input", input_type, true, true};
    Variable output{"output", output_type, true, true};

    Function func(kernel_name);
    func.launch_bounds = LAUNCH_BOUNDS_R2C_C2R_KERNEL;
    func.qualifier     = "extern \"C\" __global__";

    if(specs.scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
        func.arguments.append(hermitian_size);

    func.arguments.append(lengths0);
    func.arguments.append(lengths1);
    func.arguments.append(lengths2);
    func.arguments.append(nbatch);
    func.arguments.append(stride_in0);
    func.arguments.append(stride_in1);
    func.arguments.append(stride_in2);
    func.arguments.append(stride_in3);
    func.arguments.append(stride_out0);
    func.arguments.append(stride_out1);
    func.arguments.append(stride_out2);
    func.arguments.append(stride_out3);
    func.arguments.append(input);
    func.arguments.append(output);
    for(const auto& arg : get_callback_args().arguments)
        func.arguments.append(arg);

    Variable dim_var{"dim", "const unsigned int"};

    Variable global_idx{"global_idx", "unsigned int"};
    func.body += Declaration{global_idx, "blockIdx.x * blockDim.x + threadIdx.x"};

    Variable idx_0{"idx_0", "const unsigned int"};
    Variable idx_1{"idx_1", "const unsigned int"};
    Variable idx_2{"idx_2", "const unsigned int"};
    Variable idx_batch{"idx_batch", "const unsigned int"};

    // variable to divide by when counting lengths0 - herm2c
    // allocates threads along hermitian length, but other kernels
    // allocate threads along FFT length
    Variable& lengths0_divide
        = specs.scheme == CS_KERNEL_COPY_HERM_TO_CMPLX ? hermitian_size : lengths0;

    func.body += CommentLines{"per-dimension indexes"};
    func.body += Declaration{idx_0, global_idx % lengths0_divide};
    func.body += Assign{global_idx, global_idx / lengths0_divide};
    if(specs.dim > 1)
    {
        func.body += Declaration{idx_1, global_idx % lengths1};
        func.body += Assign{global_idx, global_idx / lengths1};
    }
    else
    {
        func.body += Declaration{idx_1, 0};
    }
    if(specs.dim > 2)
    {
        func.body += Declaration{idx_2, global_idx % lengths2};
        func.body += Assign{global_idx, global_idx / lengths2};
    }
    else
    {
        func.body += Declaration{idx_2, 0};
    }
    func.body += Declaration{idx_batch, global_idx};

    func.body += CommentLines{"any excess threads will be past the end of batch"};
    func.body += If{idx_batch >= nbatch, {Return{}}};

    if(specs.scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
    {
        Variable input_offset{"input_offset", "auto"};
        func.body += Declaration{input_offset,
                                 idx_0 * stride_in0 + idx_1 * stride_in1 + idx_2 * stride_in2
                                     + idx_batch * ("stride_in" + std::to_string(specs.dim))};

        Variable outputs_offset{"outputs_offset", "auto"};
        Variable outputc_offset{"outputc_offset", "auto"};

        func.body += CommentLines{"straight copy indices"};
        Variable is0{"is0", "auto"};
        Variable is1{"is1", "auto"};
        Variable is2{"is2", "auto"};
        func.body += Declaration{is0, idx_0};
        func.body += Declaration{is1, idx_1};
        func.body += Declaration{is2, idx_2};

        func.body += CommentLines{"conjugate copy indices"};
        Variable ic0{"ic0", "auto"};
        Variable ic1{"ic1", "auto"};
        Variable ic2{"ic2", "auto"};
        func.body += Declaration{ic0, Ternary{is0 == 0, 0, lengths0 - is0}};
        func.body += Declaration{ic1, Ternary{is1 == 0, 0, lengths1 - is1}};
        func.body += Declaration{ic2, Ternary{is2 == 0, 0, lengths2 - is2}};

        func.body += Declaration{outputs_offset,
                                 is0 * stride_out0 + is1 * stride_out1 + is2 * stride_out2
                                     + idx_batch * ("stride_out" + std::to_string(specs.dim))};
        func.body += Declaration{outputc_offset,
                                 ic0 * stride_out0 + ic1 * stride_out1 + ic2 * stride_out2
                                     + idx_batch * ("stride_out" + std::to_string(specs.dim))};

        func.body += CallbackLoadDeclaration("scalar_type", "cbtype");
        func.body += CallbackStoreDeclaration("scalar_type", "cbtype");

        func.body += CommentLines{"we would do hermitian2complex at the start of a C2R transform,",
                                  "so it would never be the last kernel to write to global",
                                  "memory.  don't bother going through the store callback to",
                                  "write global memory."};
        Variable outputs{"outputs", "scalar_type", true};
        Variable outputc{"outputc", "scalar_type", true};
        func.body += Declaration{outputs, output + outputs_offset};
        func.body += Declaration{outputc, output + outputc_offset};

        func.body += CommentLines{"simply write the element to output"};
        If write_simple{Or{is0 == 0, is0 * 2 == lengths0}, {}};
        write_simple.body += CommentLines{"simply write the element to output"};
        write_simple.body += Assign{outputs[0], LoadGlobal{input, input_offset}};
        write_simple.body += Return{};
        func.body += write_simple;

        If write_conj{is0 < hermitian_size, {}};

        Variable elem{"elem", "scalar_type"};
        write_conj.body += Declaration{elem};
        write_conj.body += Assign{elem, LoadGlobal{input, input_offset}};
        write_conj.body += Assign{outputs[0], elem};
        write_conj.body += Assign{elem.y(), UnaryMinus{elem.y()}};
        write_conj.body += Assign{outputc[0], elem};
        func.body += write_conj;
    }
    else
    {
        Variable inputIdx{"inputIdx", "auto"};
        Variable outputIdx{"outputIdx", "auto"};
        func.body += Declaration{inputIdx,
                                 idx_0 * stride_in0 + idx_1 * stride_in1 + idx_2 * stride_in2
                                     + idx_batch * ("stride_in" + std::to_string(specs.dim))};
        func.body += Declaration{outputIdx,
                                 idx_0 * stride_out0 + idx_1 * stride_out1 + idx_2 * stride_out2
                                     + idx_batch * ("stride_out" + std::to_string(specs.dim))};

        if(specs.scheme == CS_KERNEL_COPY_R_TO_CMPLX)
        {
            If guard{idx_0 < lengths0, {}};
            guard.body += CommentLines{"we would do real2complex at the beginning of an R2C",
                                       "transform, so it would never be the last kernel to write",
                                       "to global memory.  don't bother going through the store cb",
                                       "to write global memory."};
            guard.body += CallbackLoadDeclaration("real_type_t<scalar_type>", "cbtype");
            guard.body += CallbackStoreDeclaration("real_type_t<scalar_type>", "cbtype");

            ComplexLiteral elem{LoadGlobal{input, inputIdx}, "0.0"};
            guard.body += Assign{output[outputIdx], elem};
            func.body += guard;
        }
        else if(specs.scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
        {
            func.body += CommentLines{"only read and write the first [length0/2+1] elements "
                                      "due to conjugate redundancy"};

            If guard{idx_0 < Parens{1 + lengths0 / 2}, {}};

            guard.body += CommentLines{"we would do complex2hermitian at the end of an R2C",
                                       "transform, so it would never be the first kernel to read",
                                       "from global memory.  don't bother going through the load",
                                       "callback to read global memory."};

            guard.body += CallbackLoadDeclaration("scalar_type", "cbtype");
            guard.body += CallbackStoreDeclaration("scalar_type", "cbtype");

            Variable elem{"elem", "scalar_type"};
            guard.body += Declaration{elem, input[inputIdx]};
            guard.body += StoreGlobal{output, outputIdx, elem};
            func.body += guard;
        }
        else if(specs.scheme == CS_KERNEL_COPY_CMPLX_TO_R)
        {
            func.body += CommentLines{"we would do complex2real at the end of a C2R",
                                      "transform, so it would never be the first kernel to read",
                                      "from global memory.  don't bother going through the load cb",
                                      "to read global memory."};
            func.body += CallbackLoadDeclaration("real_type_t<scalar_type>", "cbtype");
            func.body += CallbackStoreDeclaration("real_type_t<scalar_type>", "cbtype");

            Variable elem{"elem", "auto"};
            func.body += Declaration{elem, input[inputIdx].x()};
            func.body += StoreGlobal{output, outputIdx, elem};
        }
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

// generate source for RTC realcomplex kernel.
std::string realcomplex_rtc(const std::string& kernel_name, const RealComplexSpecs& specs)
{
    switch(specs.scheme)
    {
    case CS_KERNEL_COPY_R_TO_CMPLX:
    case CS_KERNEL_COPY_CMPLX_TO_HERM:
    case CS_KERNEL_COPY_CMPLX_TO_R:
    case CS_KERNEL_COPY_HERM_TO_CMPLX:
        return r2c_copy_rtc(kernel_name, specs);
    default:
        throw std::runtime_error("invalid realcomplex rtc scheme");
    }
}

std::string realcomplex_even_rtc_kernel_name(const RealComplexEvenSpecs& specs)
{
    std::string kernel_name;

    switch(specs.scheme)
    {
    case CS_KERNEL_R_TO_CMPLX:
        kernel_name += "r2c_even_post";
        break;
    case CS_KERNEL_CMPLX_TO_R:
        kernel_name += "c2r_even_pre";
        break;
    default:
        throw std::runtime_error("invalid realcomplex even rtc scheme");
    }

    if(specs.Ndiv4)
    {
        kernel_name += "_Ndiv4";
    }

    kernel_name += "_dim" + std::to_string(specs.dim);

    kernel_name += rtc_precision_name(specs.precision);
    kernel_name += rtc_array_type_name(specs.inArrayType);
    kernel_name += rtc_array_type_name(specs.outArrayType);

    kernel_name += load_store_name_suffix(specs.loadOps, specs.storeOps);
    kernel_name += rtc_cbtype_name(specs.cbtype);

    return kernel_name;
}

std::string realcomplex_even_rtc(const std::string& kernel_name, const RealComplexEvenSpecs& specs)
{
    std::string src;
    // includes and declarations

    src += rocfft_complex_h;
    src += common_h;
    src += device_enum_h;
    src += callback_h;

    src += rtc_precision_type_decl(specs.precision);

    src += rtc_const_cbtype_decl(specs.cbtype);

    src += "static const unsigned int dim = " + std::to_string(specs.dim) + ";\n";

    if(specs.Ndiv4)
        src += "static const bool Ndiv4 = true;\n";
    else
        src += "static const bool Ndiv4 = false;\n";

    src += "// Each thread handles 2 points.\n";
    src += "// When N is divisible by 4, one value is handled separately; this is controlled by "
           "Ndiv4.\n";

    Variable half_N{"half_N", "const unsigned int"};
    Variable idist1D{"idist1D", "const unsigned int"};
    Variable odist1D{"odist1D", "const unsigned int"};
    Variable higherFFTLengths{"higherFFTLengths", "unsigned int"};
    Variable nbatch{"nbatch", "unsigned int"};
    Variable input{"input", "scalar_type", true, true};
    Variable idist{"idist", "const unsigned int"};
    Variable output{"output", "scalar_type", true, true};
    Variable odist{"odist", "const unsigned int"};
    Variable twiddles{"twiddles", "const scalar_type", true, true};

    Function func{kernel_name};
    func.launch_bounds = LAUNCH_BOUNDS_R2C_C2R_KERNEL;
    func.qualifier     = "extern \"C\" __global__";
    func.arguments.append(half_N);
    if(specs.dim > 1)
    {
        func.arguments.append(idist1D);
        func.arguments.append(odist1D);
    }
    func.arguments.append(higherFFTLengths);
    func.arguments.append(nbatch);
    func.arguments.append(input);
    func.arguments.append(idist);
    func.arguments.append(output);
    func.arguments.append(odist);
    func.arguments.append(twiddles);
    for(const auto& arg : get_callback_args().arguments)
        func.arguments.append(arg);

    Variable global_idx{"global_idx", "unsigned int"};
    func.body += Declaration{global_idx, "blockIdx.x * blockDim.x + threadIdx.x"};

    Variable idx_p{"idx_p", "const auto"};
    Variable idx_q{"idx_q", "const auto"};
    Variable idx_batch{"idx_batch", "const unsigned int"};
    Variable idx_batch1D{"idx_batch1D", "const unsigned int"};

    func.body += Declaration{idx_p, global_idx % half_N};
    func.body += Declaration{idx_q, half_N - idx_p};

    func.body += Assign{global_idx, global_idx / half_N};

    Variable quarter_N{"quarter_N", "const auto"};
    func.body += Declaration{quarter_N, Parens{half_N + 1} / 2};

    If guard{idx_p < quarter_N, {}};

    Variable input_offset{"input_offset", "auto"};
    Variable output_offset{"output_offset", "auto"};
    guard.body += Declaration(input_offset, idx_batch * idist);
    guard.body += Declaration(output_offset, idx_batch * odist);
    if(specs.dim > 1)
    {
        guard.body += AddAssign(input_offset, idx_batch1D * idist1D);
        guard.body += AddAssign(output_offset, idx_batch1D * odist1D);
    }

    func.body += CommentLines{
        "this kernel treats all rows as batched 1D (it does not tolerate differing",
        "strides between higher FFT dimensions), but user-provided batch is tracked",
        "separately"};
    func.body += Declaration{idx_batch1D, global_idx % higherFFTLengths};
    func.body += Declaration{idx_batch, global_idx / higherFFTLengths};

    func.body += CommentLines{"any excess threads will be past the end of batch"};
    func.body += If{idx_batch >= nbatch, {Return{}}};

    if(specs.scheme == CS_KERNEL_R_TO_CMPLX)
    {
        guard.body += CommentLines{"post process can't be the first kernel, so don't bother",
                                   "going through the load cb to read global memory"};
    }
    else
    {
        guard.body += CommentLines{"we would do real_pre_process at the beginning of a C2R",
                                   "transform, so it would never be the last kernel to write",
                                   "to global memory.  don't bother going through store",
                                   "callback to write global memory."};
    }
    guard.body += CallbackLoadDeclaration("scalar_type", "cbtype");
    guard.body += CallbackStoreDeclaration("scalar_type", "cbtype");

    Variable outval{"outval", "scalar_type"};
    guard.body += Declaration{outval};

    // p and q can get values from LoadGlobal, which needs to be part
    // of an Assign node for make_planar to work properly.  So p and
    // q can't be const.
    Variable p{"p", "scalar_type"};
    Variable q{"q", "scalar_type"};
    Variable u{"u", "const scalar_type"};
    Variable v{"v", "const scalar_type"};
    Variable twd_p{"twd_p", "const scalar_type"};

    If if_idx_p_zero{idx_p == 0, {}};
    if(specs.scheme == CS_KERNEL_R_TO_CMPLX)
    {
        if_idx_p_zero.body
            += Assign{outval.x(), input[input_offset + 0].x() - input[input_offset + 0].y()};
        if_idx_p_zero.body += Assign{outval.y(), 0};
        if_idx_p_zero.body += StoreGlobal{output, output_offset + half_N, outval};

        if_idx_p_zero.body
            += Assign{outval.x(), input[input_offset + 0].x() + input[input_offset + 0].y()};
        if_idx_p_zero.body += Assign{outval.y(), 0};
        if_idx_p_zero.body += StoreGlobal{output, output_offset + 0, outval};
    }
    else
    {
        if_idx_p_zero.body += Declaration{p};
        if_idx_p_zero.body += Assign{p, LoadGlobal{input, input_offset + idx_p}};
        if_idx_p_zero.body += Declaration{q};
        if_idx_p_zero.body += Assign{q, LoadGlobal{input, input_offset + idx_q}};
        if_idx_p_zero.body += Assign{output[output_offset + idx_p].x(), p.x() + q.x()};
        if_idx_p_zero.body += Assign{output[output_offset + idx_p].y(), p.x() - q.x()};
    }

    If if_Ndiv4{"Ndiv4", {}};
    if(specs.scheme == CS_KERNEL_R_TO_CMPLX)
    {
        if_Ndiv4.body += Assign{outval.x(), input[input_offset + quarter_N].x()};
        if_Ndiv4.body += Assign{outval.y(), -input[input_offset + quarter_N].y()};
        if_Ndiv4.body += StoreGlobal{output, output_offset + quarter_N, outval};
    }
    else
    {
        Variable quarter_elem{"quarter_elem", "scalar_type"};
        if_Ndiv4.body += Declaration{quarter_elem};
        if_Ndiv4.body += Assign{quarter_elem, LoadGlobal{input, input_offset + quarter_N}};
        if_Ndiv4.body
            += Assign{output[output_offset + quarter_N].x(), Literal{"2.0"} * quarter_elem.x()};
        if_Ndiv4.body
            += Assign{output[output_offset + quarter_N].y(), Literal{"-2.0"} * quarter_elem.y()};
    }

    if_idx_p_zero.body += if_Ndiv4;

    guard.body += if_idx_p_zero;

    Else else_idx_p_nonzero{{}};

    if(specs.scheme == CS_KERNEL_R_TO_CMPLX)
    {
        else_idx_p_nonzero.body += Declaration{p, input[input_offset + idx_p]};
        else_idx_p_nonzero.body += Declaration{q, input[input_offset + idx_q]};
        else_idx_p_nonzero.body += Declaration{u, Literal{"0.5"} * (p + q)};
        else_idx_p_nonzero.body += Declaration{v, Literal{"0.5"} * (p - q)};

        else_idx_p_nonzero.body += Declaration{twd_p, twiddles[idx_p]};
        else_idx_p_nonzero.body += CommentLines{"NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);"};

        else_idx_p_nonzero.body
            += Assign{outval.x(), u.x() + v.x() * twd_p.y() + u.y() * twd_p.x()};
        else_idx_p_nonzero.body
            += Assign{outval.y(), v.y() + u.y() * twd_p.y() - v.x() * twd_p.x()};
        else_idx_p_nonzero.body += StoreGlobal{output, output_offset + idx_p, outval};

        else_idx_p_nonzero.body
            += Assign{outval.x(), u.x() - v.x() * twd_p.y() - u.y() * twd_p.x()};
        else_idx_p_nonzero.body
            += Assign{outval.y(), -v.y() + u.y() * twd_p.y() - v.x() * twd_p.x()};
        else_idx_p_nonzero.body += StoreGlobal{output, output_offset + idx_q, outval};
    }
    else
    {
        else_idx_p_nonzero.body += Declaration{p};
        else_idx_p_nonzero.body += Assign{p, LoadGlobal{input, input_offset + idx_p}};
        else_idx_p_nonzero.body += Declaration{q};
        else_idx_p_nonzero.body += Assign{q, LoadGlobal{input, input_offset + idx_q}};
        else_idx_p_nonzero.body += Declaration{u, p + q};
        else_idx_p_nonzero.body += Declaration{v, p - q};

        else_idx_p_nonzero.body += Declaration{twd_p, twiddles[idx_p]};
        else_idx_p_nonzero.body += CommentLines{"NB: twd_q = -conj(twd_p);"};

        else_idx_p_nonzero.body += Assign{output[output_offset + idx_p].x(),
                                          u.x() + v.x() * twd_p.y() - u.y() * twd_p.x()};
        else_idx_p_nonzero.body += Assign{output[output_offset + idx_p].y(),
                                          v.y() + u.y() * twd_p.y() + v.x() * twd_p.x()};

        else_idx_p_nonzero.body += Assign{output[output_offset + idx_q].x(),
                                          u.x() - v.x() * twd_p.y() + u.y() * twd_p.x()};
        else_idx_p_nonzero.body += Assign{output[output_offset + idx_q].y(),
                                          -v.y() + u.y() * twd_p.y() + v.x() * twd_p.x()};
    }

    guard.body += else_idx_p_nonzero;

    func.body += guard;

    make_load_store_ops(func, specs.loadOps, specs.storeOps);

    if(array_type_is_planar(specs.inArrayType))
        func = make_planar(func, "input");
    if(array_type_is_planar(specs.outArrayType))
        func = make_planar(func, "output");

    src += func.render();
    write_standalone_test_harness(func, src);
    return src;
}

std::string realcomplex_even_transpose_rtc_kernel_name(const RealComplexEvenTransposeSpecs& specs)
{
    std::string kernel_name;

    switch(specs.scheme)
    {
    case CS_KERNEL_R_TO_CMPLX_TRANSPOSE:
        kernel_name += "r2c_even_post_transpose";
        break;
    case CS_KERNEL_TRANSPOSE_CMPLX_TO_R:
        kernel_name += "transpose_c2r_even_pre";
        break;
    default:
        throw std::runtime_error("invalid realcomplex even transpose rtc scheme");
    }

    kernel_name += "_tile" + std::to_string(specs.TileX()) + "x" + std::to_string(specs.TileY());

    kernel_name += rtc_precision_name(specs.precision);
    kernel_name += rtc_array_type_name(specs.inArrayType);
    kernel_name += rtc_array_type_name(specs.outArrayType);

    kernel_name += load_store_name_suffix(specs.loadOps, specs.storeOps);
    kernel_name += rtc_cbtype_name(specs.cbtype);

    if(!specs.grid3D)
        kernel_name += "_grid1D";

    return kernel_name;
}

std::string realcomplex_even_transpose_rtc(const std::string&                   kernel_name,
                                           const RealComplexEvenTransposeSpecs& specs)
{
    const bool isR2C = specs.scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE;
    auto       tileX = specs.TileX();
    auto       tileY = specs.TileY();

    std::string src;
    // includes and declarations

    src += rocfft_complex_h;
    src += common_h;
    src += device_enum_h;
    src += callback_h;

    src += rtc_precision_type_decl(specs.precision);

    src += rtc_const_cbtype_decl(specs.cbtype);

    // function arguments
    Variable dim{"dim", "size_t"};
    Variable input{"input", "scalar_type", true, true};
    Variable idist{"idist", "size_t"};
    Variable output{"output", "scalar_type", true, true};
    Variable odist{"odist", "size_t"};
    Variable twiddles{"twiddles", "scalar_type", true, true};
    Variable lengths{"lengths", "size_t", true, true};
    Variable inStride{"inStride", "size_t", true, true};
    Variable outStride{"outStride", "size_t", true, true};
    Variable gridY{"gridY", "const unsigned int"};
    Variable gridZ{"gridZ", "const unsigned int"};

    // r2c uses a device function helper to work out which dimension
    // we're transposing to
    if(isR2C)
    {
        // this helper doesn't need to have its AST transformed or
        // anything, so just add it to source as a string
        src += R"(
	    __device__ size_t output_row_base(size_t        dim,
	                                      size_t        output_batch_start,
	                                      const size_t* outStride,
	                                      const size_t  col)
	    {
	        if(dim == 2)
	            return output_batch_start + outStride[1] * col;
	        else if(dim == 3)
	            return output_batch_start + outStride[2] * col;
	        return 0;
            }
        )";
    }

    Function func{kernel_name};
    func.launch_bounds = tileX * tileY;
    func.qualifier     = "extern \"C\" __global__";

    func.arguments.append(dim);
    func.arguments.append(input);
    func.arguments.append(idist);
    func.arguments.append(output);
    func.arguments.append(odist);
    func.arguments.append(twiddles);
    func.arguments.append(lengths);
    func.arguments.append(inStride);
    func.arguments.append(outStride);

    for(const auto& arg : get_callback_args().arguments)
        func.arguments.append(arg);

    func.arguments.append(gridY);
    func.arguments.append(gridZ);

    func.body += CommentLines{"since gridDim is passed as {gridX, 1, 1}, use the",
                              "following variables to recover block indices in a 3-D fashion:"};

    Variable old_blockIdx_x{"old_blockIdx_x", "unsigned int"};
    Variable old_blockIdx_y{"old_blockIdx_y", "unsigned int"};
    Variable old_blockIdx_z{"old_blockIdx_z", "unsigned int"};
    Variable remaining{"remaining", "unsigned int"};

    // if a 1-D grid was provided because creating a natural 3-D grid exceeded allowed limits, then remap it to a 3-D grid.
    if(!specs.grid3D)
    {
        func.body += Declaration{old_blockIdx_x, Literal{"blockIdx.x"} / (gridY * gridZ)};
        func.body += Declaration{remaining, Literal{"blockIdx.x"} % (gridY * gridZ)};
        func.body += Declaration{old_blockIdx_y, (remaining / gridZ)};
        func.body += Declaration{old_blockIdx_z, Literal{"blockIdx.x"} % gridZ};
    }

    func.body += LineBreak{};

    Variable input_batch_start{"input_batch_start", "size_t"};
    Variable output_batch_start{"output_batch_start", "size_t"};

    if(specs.grid3D)
    {
        func.body += Declaration{input_batch_start, idist * Literal{"blockIdx.z"}};
        func.body += Declaration{output_batch_start, odist * Literal{"blockIdx.z"}};
    }
    else
    {
        func.body += Declaration{input_batch_start, idist * Literal{"old_blockIdx_z"}};
        func.body += Declaration{output_batch_start, odist * Literal{"old_blockIdx_z"}};
    }

    Variable leftTile{"leftTile", "__shared__ scalar_type", false, false, tileX};
    leftTile.size2D = tileY;
    Variable rightTile{"rightTile", "__shared__ scalar_type", false, false, tileX};
    rightTile.size2D = tileY;
    func.body += CommentLines{"post-processing reads rows and transposes them to columns.",
                              "pre-processing reads columns and transposes them to rows."};

    func.body += LineBreak{};

    func.body += CommentLines{"allocate 2 tiles so we can butterfly the values together.",
                              "left tile grabs values from towards the beginnings of the rows",
                              "right tile grabs values from towards the ends"};
    func.body += Declaration{leftTile};
    func.body += Declaration{rightTile};

    // r2c reads fastest dimension as a row, c2r reads higher dims
    //
    // generator code has r2c names for shared variables.  names in
    // generated source are adjusted to suit both r2c and c2r.
    Variable len_row{isR2C ? "len_row" : "len_col", "const size_t"};
    Variable tile_size{"tile_size", "const size_t"};
    Variable left_col_start{isR2C ? "left_col_start" : "left_row_start", "const size_t"};
    Variable middle{"middle", "const size_t"};
    Variable cols_to_read{isR2C ? "cols_to_read" : "rows_to_read", "size_t"};
    Variable row_limit{isR2C ? "row_limit" : "col_limit", "const size_t"};
    Variable row_start{isR2C ? "row_start" : "col_start", "const size_t"};
    Variable row_end{isR2C ? "row_end" : "col_end", "size_t"};

    // initial values for tile accounting variables.  initialize them
    // to Literals, since the variant needs to be something
    Expression len_row_init{""};
    Expression tile_size_init{""};
    Expression left_col_start_init{""};
    Expression row_limit_init{""};
    Expression row_start_init{""};
    Expression row_end_init{""};
    if(isR2C)
    {
        func.body += CommentLines{
            "take fastest dimension and partition it into lengths that will go into each tile"};
        len_row_init   = lengths[0];
        tile_size_init = Ternary{(len_row - 1) / 2 < tileX, (len_row - 1) / 2, tileX};
        row_limit_init = Ternary{dim == 2, lengths[1], lengths[1] * lengths[2]};

        if(specs.grid3D)
        {
            left_col_start_init = Literal{"blockIdx.x"} * tile_size + 1;
            row_start_init      = Literal{"blockIdx.y"} * tileY;
        }
        else
        {
            left_col_start_init = Literal{"old_blockIdx_x"} * tile_size + 1;
            row_start_init      = Literal{"old_blockIdx_y"} * tileY;
        }

        row_end_init = tileY + row_start;
    }
    else
    {
        func.body += CommentLines{
            "take middle dimension and partition it into lengths that will go into each tile",
            "note that last row effectively gets thrown away"};
        len_row_init   = Ternary{dim == 2, lengths[1] - 1, lengths[2] - 1};
        tile_size_init = Ternary{(len_row - 1) / 2 < tileY, (len_row - 1) / 2, tileY};
        row_limit_init = Ternary{dim == 2, lengths[0], lengths[0] * lengths[1]};

        if(specs.grid3D)
        {
            left_col_start_init = Literal{"blockIdx.y"} * tile_size + 1;
            row_start_init      = Literal{"blockIdx.x"} * tileX;
        }
        else
        {
            left_col_start_init = Literal{"old_blockIdx_y"} * tile_size + 1;
            row_start_init      = Literal{"old_blockIdx_x"} * tileX;
        }

        row_end_init = tileX + row_start;
    }

    func.body += Declaration{len_row, len_row_init};
    func.body += CommentLines{"size of a complete tile for this problem - ignore the first",
                              "element and middle element (if there is one).  those are",
                              "treated specially"};
    func.body += Declaration{tile_size, tile_size_init};
    func.body += CommentLines{"first column to read into the left tile, offset by one because",
                              "first element is already handled"};
    func.body += Declaration{left_col_start, left_col_start_init};
    func.body += Declaration{middle, (len_row + 1) / 2};

    func.body += CommentLines{"number of columns to actually read into the tile (can be less",
                              "than tile size if we're out of data)"};
    func.body += Declaration{cols_to_read, tile_size};

    func.body += CommentLines{"maximum number of rows in the problem"};
    func.body += Declaration{row_limit, row_limit_init};

    func.body += CommentLines{"start+end of range this thread will work on"};
    func.body += Declaration{row_start, row_start_init};
    func.body += Declaration{row_end, row_end_init};
    func.body += If{row_end > row_limit, {Assign{row_end, row_limit}}};

    func.body += If{left_col_start + tile_size >= middle,
                    {Assign{cols_to_read, middle - left_col_start}}};

    Variable lds_row{"lds_row", "const size_t"};
    Variable lds_col{"lds_col", "const size_t"};
    Variable val{"val", "scalar_type"};
    Variable first_elem{"first_elem", "scalar_type"};
    Variable middle_elem{"middle_elem", "scalar_type"};
    Variable last_elem{"last_elem", "scalar_type"};

    Expression read_condition{""};
    Expression read_left_idx{""};
    Expression read_right_idx{""};
    Expression read_first_condition{""};
    Expression read_first_idx{""};
    Expression read_middle_idx{""};
    Expression read_last_idx{""};

    Expression    write_condition{""};
    StatementList compute_first_val;
    Expression    write_first_idx{""};
    StatementList compute_middle_val;
    Expression    write_middle_idx{""};
    StatementList compute_last_val;
    Expression    write_last_idx{""};

    // r2c-specific variables
    Variable input_row_idx{"input_row_idx", "const size_t"};
    Variable input_row_base{"input_row_base", "size_t"};

    // c2r-specific variables
    Variable input_col_base{"input_col_base", "const size_t"};
    Variable input_col_stride{"input_col_stride", "const size_t"};
    Variable output_row_base{"output_row_base", "const size_t"};
    Variable output_row_stride{"output_row_stride", "const size_t"};

    func.body += Declaration{lds_row, "threadIdx.y"};
    func.body += Declaration{lds_col, "threadIdx.x"};

    if(isR2C)
    {
        func.body += Declaration{input_row_idx, row_start + lds_row};
        func.body += Declaration{input_row_base, input_row_idx % lengths[1] * inStride[1]};
        func.body
            += If{dim > 2, {AddAssign(input_row_base, input_row_idx / lengths[1] * inStride[2])}};

        read_condition = row_start + lds_row < row_end && lds_col < cols_to_read;
        read_left_idx  = input_batch_start + input_row_base + left_col_start + lds_col;
        read_right_idx = input_batch_start + input_row_base
                         + (len_row - (left_col_start + cols_to_read - 1)) + lds_col;

        if(specs.grid3D)
        {
            read_first_condition = Literal{"blockIdx.x"} == 0 && Literal{"threadIdx.x"} == 0
                                   && row_start + lds_row < row_end;
        }
        else
        {
            read_first_condition = Literal{"old_blockIdx_x"} == 0 && Literal{"threadIdx.x"} == 0
                                   && row_start + lds_row < row_end;
        }

        read_first_idx  = input_batch_start + input_row_base;
        read_middle_idx = input_batch_start + input_row_base + len_row / 2;

        if(specs.grid3D)
        {
            write_condition = Literal{"blockIdx.x"} == 0 && Literal{"threadIdx.x"} == 0
                              && row_start + lds_row < row_end;
        }
        else
        {
            write_condition = Literal{"old_blockIdx_x"} == 0 && Literal{"threadIdx.x"} == 0
                              && row_start + lds_row < row_end;
        }

        compute_first_val += Assign{val.x(), first_elem.x() - first_elem.y()};
        compute_first_val += Assign{val.y(), Literal{"0.0"}};
        write_first_idx = CallExpr{"output_row_base", {dim, output_batch_start, outStride, len_row}}
                          + row_start + lds_row;

        compute_middle_val += Assign{val.x(), middle_elem.x()};
        compute_middle_val += Assign{val.y(), -middle_elem.y()};
        write_middle_idx = CallExpr{"output_row_base", {dim, output_batch_start, outStride, middle}}
                           + row_start + lds_row;

        compute_last_val += Assign{val.x(), first_elem.x() + first_elem.y()};
        compute_last_val += Assign{val.y(), Literal{"0.0"}};
        write_last_idx = CallExpr{"output_row_base", {dim, output_batch_start, outStride, 0}}
                         + row_start + lds_row;
    }
    else
    {
        func.body += Declaration{input_col_base,
                                 (row_start + lds_col) % lengths[0] * inStride[0]
                                     + (row_start + lds_col) / lengths[0] * inStride[1]};
        func.body += Declaration{input_col_stride, Ternary{dim == 2, inStride[1], inStride[2]}};

        func.body += Declaration{output_row_base,
                                 (row_start + lds_col) % lengths[0] * outStride[1]
                                     + (row_start + lds_col) / lengths[0] * outStride[2]};
        func.body += Declaration{output_row_stride, outStride[0]};

        read_condition = row_start + lds_col < row_end && lds_row < cols_to_read;
        read_left_idx
            = input_batch_start + input_col_base + (left_col_start + lds_row) * input_col_stride;
        read_right_idx = input_batch_start + input_col_base
                         + (len_row - (left_col_start + lds_row)) * input_col_stride;

        if(specs.grid3D)
        {
            read_first_condition = Literal{"blockIdx.y"} == 0 && Literal{"threadIdx.y"} == 0
                                   && row_start + lds_col < row_end;
        }
        else
        {
            read_first_condition = Literal{"old_blockIdx_y"} == 0 && Literal{"threadIdx.y"} == 0
                                   && row_start + lds_col < row_end;
        }

        read_first_idx  = input_batch_start + input_col_base;
        read_middle_idx = input_batch_start + input_col_base + middle * input_col_stride;
        read_last_idx   = input_batch_start + input_col_base + len_row * input_col_stride;

        if(specs.grid3D)
        {
            write_condition = Literal{"blockIdx.y"} == 0 && Literal{"threadIdx.y"} == 0
                              && row_start + lds_col < row_end;
        }
        else
        {
            write_condition = Literal{"old_blockIdx_y"} == 0 && Literal{"threadIdx.y"} == 0
                              && row_start + lds_col < row_end;
        }

        compute_first_val += Assign{val.x(), first_elem.x() + last_elem.x()};
        compute_first_val += Assign{val.y(), first_elem.x() - last_elem.x()};
        write_first_idx = output_batch_start + output_row_base;

        compute_middle_val += Assign{val.x(), Literal{"2.0"} * middle_elem.x()};
        compute_middle_val += Assign{val.y(), Literal{"-2.0"} * middle_elem.y()};
        write_middle_idx = output_batch_start + output_row_base + middle * output_row_stride;
    }

    func.body += CallbackLoadDeclaration("scalar_type", "cbtype");
    func.body += CallbackStoreDeclaration("scalar_type", "cbtype");

    func.body += Declaration{val};

    If read_block{read_condition, {}};
    read_block.body += Assign{val, LoadGlobal{input, read_left_idx}};
    read_block.body += Assign{leftTile.at(lds_col, lds_row), val};
    read_block.body += Assign{val, LoadGlobal{input, read_right_idx}};
    read_block.body += Assign{rightTile.at(lds_col, lds_row), val};
    func.body += read_block;

    func.body += Declaration{first_elem};
    func.body += Declaration{middle_elem};
    if(!isR2C)
        func.body += Declaration{last_elem};

    If read_first_block{read_first_condition, {}};
    read_first_block.body += Assign{first_elem, LoadGlobal{input, read_first_idx}};
    read_first_block.body
        += If{len_row % 2 == 0, {Assign{middle_elem, LoadGlobal{input, read_middle_idx}}}};
    if(!isR2C)
        read_first_block.body += Assign{last_elem, LoadGlobal{input, read_last_idx}};

    func.body += CommentLines{"handle first + middle element (if there is a middle),",
                              "and last element (for c2r)"};
    func.body += read_first_block;
    func.body += SyncThreads{};

    func.body += CommentLines{"write first + middle"};
    If write_first_block{write_condition, {}};
    write_first_block.body += compute_first_val;
    write_first_block.body += StoreGlobal{output, write_first_idx, val};
    // only r2c writes the "last" value
    if(isR2C)
    {
        write_first_block.body += compute_last_val;
        write_first_block.body += StoreGlobal{output, write_last_idx, val};
    }

    If write_middle_block{len_row % 2 == 0, {}};

    write_middle_block.body += compute_middle_val;
    write_middle_block.body += StoreGlobal{output, write_middle_idx, val};
    write_first_block.body += write_middle_block;

    func.body += write_first_block;

    func.body += CommentLines{"butterfly the two tiles we've collected (offset col by one",
                              "because first element is special)"};

    Variable p{"p", "const scalar_type"};
    Variable q{"q", "const scalar_type"};
    Variable u{"u", "const scalar_type"};
    Variable v{"v", "const scalar_type"};
    Variable twd_p{"twd_p", "const auto"};
    if(isR2C)
    {
        Variable col{"col", "size_t"};

        If butterfly{row_start + lds_row < row_end && lds_col < cols_to_read, {}};

        if(specs.grid3D)
        {
            butterfly.body
                += Declaration{col, Literal{"blockIdx.x"} * tile_size + 1 + Literal{"threadIdx.x"}};
        }
        else
        {
            butterfly.body += Declaration{
                col, Literal{"old_blockIdx_x"} * tile_size + 1 + Literal{"threadIdx.x"}};
        }

        butterfly.body += Declaration{p, leftTile.at(lds_col, lds_row)};
        butterfly.body += Declaration{q, rightTile.at(cols_to_read - lds_col - 1, lds_row)};
        butterfly.body += Declaration{u, Literal{"0.5"} * (p + q)};
        butterfly.body += Declaration{v, Literal{"0.5"} * (p - q)};

        butterfly.body += Declaration{twd_p, twiddles[col]};

        butterfly.body += CommentLines{"NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y)"};

        butterfly.body += CommentLines{"write left side"};
        butterfly.body += Assign{val.x(), u.x() + v.x() * twd_p.y() + u.y() * twd_p.x()};
        butterfly.body += Assign{val.y(), v.y() + u.y() * twd_p.y() - v.x() * twd_p.x()};
        butterfly.body
            += StoreGlobal{output,
                           CallExpr{"output_row_base", {dim, output_batch_start, outStride, col}}
                               + row_start + lds_row,
                           val};

        butterfly.body += CommentLines{"write right side"};
        butterfly.body += Assign{val.x(), u.x() - v.x() * twd_p.y() - u.y() * twd_p.x()};
        butterfly.body += Assign{val.y(), -v.y() + u.y() * twd_p.y() - v.x() * twd_p.x()};
        butterfly.body += StoreGlobal{
            output,
            CallExpr{"output_row_base", {dim, output_batch_start, outStride, len_row - col}}
                + row_start + lds_row,
            val};

        func.body += butterfly;
    }
    else
    {
        If butterfly{row_start + lds_col < row_end && lds_row < cols_to_read, {}};

        butterfly.body += Declaration{p, leftTile.at(lds_col, lds_row)};
        butterfly.body += Declaration{q, rightTile.at(lds_col, lds_row)};
        butterfly.body += Declaration{u, p + q};
        butterfly.body += Declaration{v, p - q};

        butterfly.body += Declaration{twd_p, twiddles[left_col_start + lds_row]};

        butterfly.body += CommentLines{"write top side"};
        butterfly.body += Assign{val.x(), u.x() + v.x() * twd_p.y() - u.y() * twd_p.x()};
        butterfly.body += Assign{val.y(), v.y() + u.y() * twd_p.y() + v.x() * twd_p.x()};
        butterfly.body += StoreGlobal{output,
                                      output_batch_start + output_row_base
                                          + (left_col_start + lds_row) * output_row_stride,
                                      val};

        butterfly.body += CommentLines{"write bottom side"};
        butterfly.body += Assign{val.x(), u.x() - v.x() * twd_p.y() + u.y() * twd_p.x()};
        butterfly.body += Assign{val.y(), -v.y() + u.y() * twd_p.y() + v.x() * twd_p.x()};
        butterfly.body
            += StoreGlobal{output,
                           output_batch_start + output_row_base
                               + (len_row - (left_col_start + lds_row)) * output_row_stride,
                           val};
        func.body += butterfly;
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
