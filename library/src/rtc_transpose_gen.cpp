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

#include "rtc_transpose_gen.h"
#include "../../shared/array_predicate.h"
#include "device/generator/generator.h"
#include "rtc_test_harness.h"

#include "device/kernel-generator-embed.h"

// generate name for RTC transpose kernel
std::string transpose_rtc_kernel_name(const TransposeSpecs& specs)
{
    std::string kernel_name = "transpose_rtc";

    kernel_name += "_tile";
    kernel_name += std::to_string(specs.tileX);
    kernel_name += "x";
    kernel_name += std::to_string(specs.tileY);

    // 2D + 3D kernels are specialized to omit loops
    switch(specs.dim)
    {
    case 2:
        kernel_name += "_dim2";
        break;
    case 3:
        kernel_name += "_dim3";
        break;
    default:
        break;
    }

    kernel_name += rtc_precision_name(specs.precision);
    kernel_name += rtc_array_type_name(specs.inArrayType);
    kernel_name += rtc_array_type_name(specs.outArrayType);

    if(specs.largeTwdSteps)
    {
        kernel_name += "_twd";
        kernel_name += std::to_string(specs.largeTwdSteps);
        kernel_name += "step";

        if(specs.largeTwdDirection == -1)
            kernel_name += "_fwd";
        else
            kernel_name += "_back";
    }

    if(specs.diagonal)
        kernel_name += "_diag";
    if(specs.tileAligned)
        kernel_name += "_aligned";
    kernel_name += load_store_name_suffix(specs.loadOps, specs.storeOps);
    kernel_name += rtc_cbtype_name(specs.cbtype);

    if(!specs.grid3D)
        kernel_name += "_grid1D";

    return kernel_name;
}

// generate source for RTC transpose kernel.
std::string transpose_rtc(const std::string& kernel_name, const TransposeSpecs& specs)
{
    std::string src;

    // includes and declarations
    src += rocfft_complex_h;
    src += common_h;
    src += device_enum_h;
    src += callback_h;

    src += rtc_precision_type_decl(specs.precision, array_type_is_complex(specs.inArrayType));

    src += rtc_const_cbtype_decl(specs.cbtype);

    // twiddle code assumes scalar type is named T
    src += "typedef scalar_type T;\n";

    // arguments to transpose kernel
    Variable input_var{"input", "scalar_type", true, true};
    Variable output_var{"output", "scalar_type", true, true};
    Variable twiddles_large_var{"twiddles_large", "const scalar_type", true, true};
    Variable dim_var{"dim", "unsigned int"};
    Variable length0_var{"length0", "unsigned int"};
    Variable length1_var{"length1", "unsigned int"};
    Variable length2_var{"length2", "unsigned int"};
    Variable lengths_var{"lengths", "const size_t", true, true};
    Variable stride_in0_var{"stride_in0", "unsigned int"};
    Variable stride_in1_var{"stride_in1", "unsigned int"};
    Variable stride_in2_var{"stride_in2", "unsigned int"};
    Variable stride_in_var{"stride_in", "const size_t", true, true};
    Variable idist_var{"idist", "unsigned int"};
    Variable stride_out0_var{"stride_out0", "unsigned int"};
    Variable stride_out1_var{"stride_out1", "unsigned int"};
    Variable stride_out2_var{"stride_out2", "unsigned int"};
    Variable stride_out_var{"stride_out", "const size_t", true, true};
    Variable odist_var{"odist", "unsigned int"};
    Variable gridX{"gridX", "const unsigned int"};
    Variable gridY{"gridY", "const unsigned int"};
    Variable gridZ{"gridZ", "const unsigned int"};

    Function func(kernel_name);
    func.launch_bounds = specs.tileX * specs.tileY;
    func.qualifier     = "extern \"C\" __global__";

    func.arguments.append(input_var);
    func.arguments.append(output_var);
    func.arguments.append(twiddles_large_var);
    func.arguments.append(dim_var);
    func.arguments.append(length0_var);
    func.arguments.append(length1_var);
    func.arguments.append(length2_var);
    func.arguments.append(lengths_var);
    func.arguments.append(stride_in0_var);
    func.arguments.append(stride_in1_var);
    func.arguments.append(stride_in2_var);
    func.arguments.append(stride_in_var);
    func.arguments.append(idist_var);
    func.arguments.append(stride_out0_var);
    func.arguments.append(stride_out1_var);
    func.arguments.append(stride_out2_var);
    func.arguments.append(stride_out_var);
    func.arguments.append(odist_var);
    func.arguments.append(gridX);
    func.arguments.append(gridY);
    func.arguments.append(gridZ);

    for(const auto& arg : get_callback_args().arguments)
        func.arguments.append(arg);

    // we use tileX*tileX tiles - tileY must evenly divide into
    // tileX, so that elems_per_thread is integral
    if(specs.tileX % specs.tileY != 0)
        throw std::runtime_error("non-integral transpose ELEMS_PER_THREAD");
    auto elems_per_thread = specs.tileX / specs.tileY;
    if(elems_per_thread == 0)
        throw std::runtime_error("zero transpose ELEMS_PER_THREAD");

    // lds is a 2D array
    Variable lds{"lds", "__shared__ scalar_type", false, false, specs.tileX};
    lds.size2D = Literal{specs.tileX};
    func.body += Declaration{lds};

    // recover a 3D GPU grid layout
    func.body += CommentLines{"since gridDim is passed as {gridX, 1, 1}, use the",
                              "following variables to recover block indices in a 3-D fashion:"};

    Variable old_blockIdx_x{"old_blockIdx_x", "unsigned int"};
    Variable old_blockIdx_y{"old_blockIdx_y", "unsigned int"};
    Variable old_blockIdx_z{"old_blockIdx_z", "unsigned int"};

    Variable tileBlockIdx_y{"tileBlockIdx_y", "unsigned int"};
    Variable tileBlockIdx_x{"tileBlockIdx_x", "unsigned int"};

    Variable remaining{"remaining", "unsigned int"};

    // if a 1-D grid was provided because creating a natural 3-D grid exceeded allowed limits, then remap it to a 3-D grid.
    if(!specs.grid3D)
    {
        func.body += Declaration{old_blockIdx_x, Literal{"blockIdx.x"} / (gridY * gridZ)};
        func.body += Declaration{remaining, Literal{"blockIdx.x"} % (gridY * gridZ)};
        func.body += Declaration{old_blockIdx_y, (remaining / gridZ)};
        func.body += Declaration{old_blockIdx_z, Literal{"blockIdx.x"} % gridZ};

        func.body += Declaration{tileBlockIdx_y, "old_blockIdx_y"};
        func.body += Declaration{tileBlockIdx_x, "old_blockIdx_x"};
    }
    else
    {
        func.body += Declaration{tileBlockIdx_y, "blockIdx.y"};
        func.body += Declaration{tileBlockIdx_x, "blockIdx.x"};
    }

    func.body += LineBreak{};

    if(specs.diagonal)
    {
        Variable bid{"bid", "auto"};

        if(specs.grid3D)
        {
            func.body += Declaration{bid, "blockIdx.x + gridDim.x * blockIdx.y"};
            func.body += Assign{tileBlockIdx_y, bid % Variable{"gridDim.y", ""}};
            func.body += Assign{tileBlockIdx_x,
                                (bid / Variable{"gridDim.y", ""} + tileBlockIdx_y)
                                    % Variable{"gridDim.x", ""}};
        }
        else
        {
            func.body += Declaration{bid, "old_blockIdx_x + gridX * old_blockIdx_y"};
            func.body += Assign{tileBlockIdx_y, bid % Variable{"gridY", ""}};
            func.body
                += Assign{tileBlockIdx_x,
                          (bid / Variable{"gridY", ""} + tileBlockIdx_y) % Variable{"gridX", ""}};
        }
    }

    if(specs.dim == 2)
    {
        func.body += CommentLines{"only using 2 dimensions, pretend length2 is 1 so the",
                                  "compiler can optimize out comparisons against it"};
        func.body += Assign{length2_var, 1};
    }

    Variable tile_x_index{"tile_x_index", "unsigned int"};
    Variable tile_y_index{"tile_y_index", "unsigned int"};
    func.body += Declaration{tile_x_index, "threadIdx.x"};
    func.body += Declaration{tile_y_index, "threadIdx.y"};

    func.body += CommentLines{"work out offset for dimensions after the first 3"};
    Variable offset_in{"offset_in", "unsigned int"};
    Variable offset_out{"offset_out", "unsigned int"};
    if(specs.grid3D)
    {
        func.body += Declaration{remaining, "blockIdx.z"};
    }
    else
    {
        func.body += Assign{remaining, "old_blockIdx_z"};
    }
    func.body += Declaration{offset_in, 0};
    func.body += Declaration{offset_out, 0};

    // use specified dim to avoid loops if possible
    if(specs.dim > 3)
    {
        Variable d{"d", "unsigned int"};
        For      offset_loop{
            d,
            3,
            d < dim_var,
            1,
        };

        Variable index_along_d{"index_along_d", "auto"};
        offset_loop.body += Declaration{index_along_d, remaining % lengths_var[d]};
        offset_loop.body += Assign{remaining, remaining / lengths_var[d]};
        offset_loop.body += Assign{offset_in, offset_in + index_along_d * stride_in_var[d]};
        offset_loop.body += Assign{offset_out, offset_out + index_along_d * stride_out_var[d]};
        func.body += offset_loop;
    }

    func.body += CommentLines{"remaining is now the batch"};
    func.body += AddAssign(offset_in, remaining * idist_var);
    func.body += AddAssign(offset_out, remaining * odist_var);
    func.body += CallbackLoadDeclaration("scalar_type", "cbtype");
    func.body += CallbackStoreDeclaration("scalar_type", "cbtype");

    // loop variables for reading/writing
    Variable i{"i", "unsigned int"};
    Variable logical_row{"logical_row", "auto"};
    Variable logical_col{"logical_col", "auto"};
    Variable idx0{"idx0", "auto"};
    Variable idx1{"idx1", "auto"};
    Variable idx2{"idx2", "auto"};
    Variable global_read_idx{"global_read_idx", "auto"};
    Variable global_write_idx{"global_write_idx", "auto"};
    Variable elem{"elem", "scalar_type"};
    Variable twl_idx{"twl_idx", "auto"};

    For read_loop{i, 0, i < elems_per_thread, 1};
    read_loop.pragma_unroll = true;

    read_loop.body
        += Declaration{logical_row, specs.tileX * tileBlockIdx_y + tile_y_index + i * specs.tileY};
    read_loop.body += Declaration{idx0, specs.tileX * tileBlockIdx_x + tile_x_index};
    read_loop.body += Declaration{idx1, logical_row};
    if(specs.dim != 2)
        read_loop.body += ModulusAssign(idx1, length1_var);

    if(specs.dim == 2)
        read_loop.body += Declaration{idx2, 0};
    else
        read_loop.body += Declaration{idx2, logical_row / length1_var};

    if(!specs.tileAligned)
    {
        read_loop.body
            += If{Or{Or{idx0 >= length0_var, idx1 >= length1_var}, idx2 >= length2_var}, {Break{}}};
    }

    read_loop.body += Declaration{global_read_idx,
                                  idx0 * stride_in0_var + idx1 * stride_in1_var
                                      + idx2 * stride_in2_var + offset_in};
    read_loop.body += Declaration{elem};
    read_loop.body += Assign{elem, LoadGlobal{input_var, global_read_idx}};

    if(specs.largeTwdSteps)
    {
        auto twiddle_mul_macro
            = specs.largeTwdDirection == -1 ? "TWIDDLE_STEP_MUL_FWD" : "TWIDDLE_STEP_MUL_INV";
        std::string twiddle_step_func = "TWLstep" + std::to_string(specs.largeTwdSteps);

        read_loop.body += Declaration{twl_idx, idx0 * idx1};
        read_loop.body
            += Call{twiddle_mul_macro, {twiddle_step_func, twiddles_large_var, twl_idx, elem}};
    }
    read_loop.body += Assign{lds.at(tile_x_index, i * specs.tileY + tile_y_index), elem};

    func.body += read_loop;

    func.body += SyncThreads{};

    Variable val{"val", "scalar_type", false, false, elems_per_thread};
    func.body += Declaration{val};

    func.body += CommentLines{"reallocate threads to write along fastest dim (length1) and",
                              "read transposed from LDS"};
    func.body += Assign{tile_x_index, "threadIdx.y"};
    func.body += Assign{tile_y_index, "threadIdx.x"};

    For transpose_loop{i,
                       0,
                       i < elems_per_thread,
                       1,
                       {Assign{val[i], lds.at(tile_x_index + i * specs.tileY, tile_y_index)}}};
    transpose_loop.pragma_unroll = true;
    func.body += transpose_loop;

    For write_loop{i, 0, i < elems_per_thread, 1};
    write_loop.pragma_unroll = true;

    write_loop.body
        += Declaration{logical_col, specs.tileX * tileBlockIdx_x + tile_x_index + i * specs.tileY};
    write_loop.body += Declaration{logical_row, specs.tileX * tileBlockIdx_y + tile_y_index};

    write_loop.body += Declaration{idx0, logical_col};
    write_loop.body += Declaration{idx1, logical_row};
    if(specs.dim != 2)
    {
        write_loop.body += ModulusAssign(idx1, length1_var);
    }
    if(specs.dim == 2)
        write_loop.body += Declaration{idx2, 0};
    else
        write_loop.body += Declaration{idx2, logical_row / length1_var};

    if(!specs.tileAligned)
    {
        write_loop.body
            += If{Or{Or{idx0 >= length0_var, idx1 >= length1_var}, idx2 >= length2_var}, {Break{}}};
    }
    write_loop.body += Declaration{global_write_idx,
                                   idx0 * stride_out0_var + idx1 * stride_out1_var
                                       + idx2 * stride_out2_var + offset_out};
    write_loop.body += StoreGlobal{output_var, global_write_idx, val[i]};

    func.body += write_loop;

    make_load_store_ops(func, specs.loadOps, specs.storeOps);

    if(array_type_is_planar(specs.inArrayType))
        func = make_planar(func, "input");
    if(array_type_is_planar(specs.outArrayType))
        func = make_planar(func, "output");

    func = make_callback_realcomplex(func, specs.cbtype);

    src += func.render();

    write_standalone_test_harness(func, src);

    return src;
}
