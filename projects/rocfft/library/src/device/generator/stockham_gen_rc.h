// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#include "stockham_gen_base.h"

struct StockhamKernelRC : public StockhamKernel
{
    explicit StockhamKernelRC(const StockhamGeneratorSpecs& specs, bool emitGlobalId)
        : StockhamKernel(specs)
        , emitGlobalId(emitGlobalId)
    {
    }

    bool emitGlobalId = false;

    //
    // templates
    //
    Variable sbrc_type{"sbrc_type", "SBRC_TYPE"};
    Variable transpose_type{"transpose_type", "SBRC_TRANSPOSE_TYPE"};

    //
    // locals
    //
    Variable tile{"tile", "unsigned int"};
    Variable offset_in{"offset_in", "unsigned int"};
    Variable offset_out{"offset_out", "unsigned int"};
    Variable stride_in{"stride_in", "const size_t", true};
    Variable stride_out{"stride_out", "const size_t", true};

    Variable stride0_out{"stride0_out", "const size_t"};

    //
    //
    //
    Variable len_along_block{"len_along_block", "const unsigned int"};
    Variable len_along_plane{"len_along_plane", "const unsigned int"};
    Variable stride_load_in{"stride_load_in", "const unsigned int"};
    Variable stride_store_out{"stride_store_out", "const unsigned int"};
    Variable stride_plane_in{"stride_plane_in", "const unsigned int"};
    Variable stride_plane_out{"stride_plane_out", "const unsigned int"};

    //
    // locals
    //
    Variable num_of_tiles_in_plane{"num_of_tiles_in_plane", "unsigned int"};
    Variable num_of_tiles_in_batch{"num_of_tiles_in_batch", "unsigned int"};
    Variable tile_index_in_plane{"tile_index_in_plane", "unsigned int"};

    Variable edge{"edge", "bool"};
    Variable thread{"thread", "unsigned int"}; // replacing tid_ver
    Variable tid_hor{"tid_hor", "unsigned int"};

    std::string tiling_name() override
    {
        return "SBRC";
    }

    // TODO- support embedded Pre/Post
    StatementList set_direct_to_from_registers() override
    {
        // RC: we never do "direct-to-reg", but do "direct-from-reg" and "non-linear"
        if(direct_to_from_reg)
            return {Declaration{direct_load_to_reg, Literal{"false"}},
                    Declaration{direct_store_from_reg,
                                And{directReg_type == "DirectRegType::TRY_ENABLE_IF_SUPPORT",
                                    sbrc_type != "SBRC_3D_FFT_ERC_TRANS_Z_XY"}},
                    Declaration{lds_linear, Not{direct_store_from_reg}}};
        else
            return {Declaration{direct_load_to_reg, Literal{"false"}},
                    Declaration{direct_store_from_reg, Literal{"false"}},
                    Declaration{lds_linear, Literal{"true"}}};
    }

    StatementList set_lds_is_real() override
    {
        // SBRC can't support half-lds
        return {Declaration{lds_is_real, Literal{"false"}}};
    }

    StatementList store_global_generator(unsigned int h,
                                         unsigned int hr,
                                         unsigned int width,
                                         unsigned int dt,
                                         Expression   guard,
                                         unsigned int cumheight)
    {
        if(hr == 0)
            hr = h;
        StatementList work;
        for(unsigned int w = 0; w < width; ++w)
        {
            auto tid = Parens{thread + dt + h * threads_per_transform};
            auto idx
                = Parens{tid / cumheight} * (width * cumheight) + tid % cumheight + w * cumheight;

            if(emitGlobalId)
            {
                work += Assign{global_data_id,
                               global_store_data_offset + tid_hor
                                   + Parens{Expression{idx}} * lengths[1]};
                work += Assign{global_transf_id,
                               global_store_transf_offset + tid_hor
                                   + Parens{Expression{idx}} * lengths[1]};
            }

            work += StoreGlobal{buf,
                                offset + tid_hor * stride0
                                    + Parens{Expression{idx}} * stride_store_out,
                                R[hr * width + w]};
        }
        return work;
    }

    // embedded real-complex needs one extra element per row
    Expression get_lds_padding() override
    {
        return Ternary{sbrc_type != "SBRC_3D_FFT_ERC_TRANS_Z_XY", 0, 1};
    }

    StatementList calculate_offsets() override
    {
        StatementList stmts;

        Variable plane_id{"plane_id", "unsigned int"};
        Variable tile_serial_in_batch{"tile_serial_in_batch", "unsigned int"};
        Variable global_stride_in{"global_stride_in", "const size_t"};
        Variable global_stride_out{"global_stride_out", "const size_t"};

        stmts += Declaration{
            len_along_block,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, lengths[2], lengths[1]}};
        stmts += Declaration{
            stride_load_in,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, stride_in[2], stride_in[1]}};
        stmts += Declaration{
            stride_store_out,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z" || sbrc_type == "SBRC_2D"},
                    stride_out[1],
                    stride_out[2]}};

        stmts += LineBreak{};
        stmts += Declaration{num_of_tiles_in_batch};
        stmts += Declaration{tile_index_in_plane};

        stmts += LineBreak{};
        stmts
            += Declaration{num_of_tiles_in_plane, (len_along_block - 1) / transforms_per_block + 1};

        if(emitGlobalId)
        {
            stmts += Declaration{global_load_data_offset, 0};
            stmts += Declaration{global_load_transf_offset, 0};
            stmts += Declaration{global_store_data_offset, 0};
            stmts += Declaration{global_store_transf_offset, 0};

            stmts += Declaration{global_data_id, 0};
            stmts += Declaration{global_transf_id, 0};
        }

        // --------------------------------------------------
        // SBRC_2D
        // --------------------------------------------------
        StatementList offset_2d;

        Variable d{"d", "unsigned int"};
        Variable index_along_d{"index_along_d", "unsigned int"};
        Variable remaining{"remaining", "unsigned int"};

        // offset_2d += CommentLines{"calculate offset for each tile:",
        //                       "  num_of_tiles_in_plane now means number of tiles along dim1",
        //                       "  tile_index_in_plane now means index of the tile along dim1"};
        // offset_2d += Assign{num_of_tiles_in_plane, (len_along_block - 1) / transforms_per_block + 1};

        offset_2d += Assign{num_of_tiles_in_batch, num_of_tiles_in_plane};
        offset_2d += Assign{tile_index_in_plane, block_id % num_of_tiles_in_plane};

        offset_2d += Declaration{remaining, block_id / num_of_tiles_in_plane};
        offset_2d += Declaration{index_along_d};

        offset_2d += Assign{offset_in, tile_index_in_plane * transforms_per_block * stride_load_in};
        offset_2d += Assign{offset_out, tile_index_in_plane * transforms_per_block * stride0_out};

        if(emitGlobalId)
        {
            stmts += Declaration{Variable{"global_stride_in[3]", "const size_t"},
                                 Literal{"{global_stride_in_0, global_stride_in_1, global_idist}"}};
            stmts
                += Declaration{Variable{"global_stride_out[3]", "const size_t"},
                               Literal{"{global_stride_out_0, global_stride_out_1, global_odist}"}};

            offset_2d += For{
                d,
                2,
                d < dim,
                1,
                {Assign{num_of_tiles_in_batch, num_of_tiles_in_batch * lengths[d]},
                 Assign{index_along_d, remaining % lengths[d]},
                 Assign{remaining, remaining / lengths[d]},
                 Assign{offset, offset + index_along_d * stride[d]},
                 Assign{global_load_data_offset,
                        global_load_data_offset + index_along_d * global_stride_in[d - 2]},
                 Assign{global_store_data_offset,
                        global_store_data_offset + index_along_d * global_stride_out[d - 2]}}};
        }
        else
        {
            offset_2d += For{d,
                             2,
                             d < dim,
                             1,
                             {Assign{num_of_tiles_in_batch, num_of_tiles_in_batch * lengths[d]},
                              Assign{index_along_d, remaining % lengths[d]},
                              Assign{remaining, remaining / lengths[d]},
                              Assign{offset, offset + index_along_d * stride[d]}}};
        }

        //
        // --------------------------------------------------
        // SBRC_3D
        // --------------------------------------------------
        StatementList offset_3d;

        // offset_xy_z
        //     += CommentLines{"calculate offset for each tile:",
        //                     " num_of_tiles_in_plane means number of tiles in a XZ-plane",
        //                     " num_of_tiles_in_batch means the total number of tiles in this batch",
        //                     " tile_index_in_plane means index of the tile in that XZ-plane",
        //                     " plane_id means the index of current XZ-plane (downwards along Y-axis)"};
        // offset_z_xy
        //     += CommentLines{"calculate offset for each tile:",
        //                     " num_of_tiles_in_plane means number of tiles in a xy plane",
        //                     " num_of_tiles_in_batch means the total number of tiles in this batch",
        //                     " tile_index_in_plane means index of the tile in that xy-plane",
        //                     " plane_id means the index of current xy-plane (inwards along Z-axis)"};

        offset_3d += Declaration{
            len_along_plane,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, lengths[1], lengths[2]}};
        offset_3d += Declaration{
            stride_plane_in,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, stride_in[1], stride_in[2]}};
        offset_3d += Declaration{
            stride_plane_out,
            Ternary{Parens{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z"}, stride_out[2], stride_out[1]}};
        stmts += Declaration{plane_id};
        stmts += Declaration{tile_serial_in_batch};

        // offset_3d += Assign{num_of_tiles_in_plane, (len_along_block - 1) / transforms_per_block + 1};
        offset_3d += Assign{num_of_tiles_in_batch, num_of_tiles_in_plane * len_along_plane};
        offset_3d += Assign{tile_serial_in_batch, block_id % num_of_tiles_in_batch};

        // Use DIAGONAL or NOT
        auto xy_z_diagonal = [&]() -> StatementList {
            return {
                Assign{tile_index_in_plane, tile_serial_in_batch % threads_per_transform},
                // looks like lengths[1] is correct? but if we restrict the size to be cubic, then we can simply use length
                Assign{plane_id,
                       (tile_serial_in_batch / threads_per_transform + tile_index_in_plane)
                           % length},
            };
        };
        auto xy_z_regular = [&]() -> StatementList {
            return {
                Assign{tile_index_in_plane, tile_serial_in_batch / lengths[1]},
                Assign{plane_id, block_id % lengths[1]},
                // Assign{tile_index_in_plane, block_id % num_of_tiles_in_plane},
                // Assign{plane_id, (block_id / num_of_tiles_in_plane) % lengths[1]},
            };
        };

        auto z_xy_diagonal = [&]() -> StatementList {
            return {
                Assign{tile_index_in_plane, tile_serial_in_batch % threads_per_transform},
                // looks like lengths[2] is correct? but if we restrict the size to be cubic, then we can simply use length
                Assign{plane_id,
                       (tile_serial_in_batch / threads_per_transform + tile_index_in_plane)
                           % length},
            };
        };
        auto z_xy_regular = [&]() -> StatementList {
            return {
                Assign{plane_id, tile_serial_in_batch / num_of_tiles_in_plane},
                Assign{tile_index_in_plane, block_id % num_of_tiles_in_plane},
            };
        };

        StatementList xy_z_offset;
        xy_z_offset += If{transpose_type == "DIAGONAL", xy_z_diagonal()};
        xy_z_offset += Else{xy_z_regular()};

        StatementList z_xy_offset;
        z_xy_offset += If{transpose_type == "DIAGONAL", z_xy_diagonal()};
        z_xy_offset += Else{z_xy_regular()};

        offset_3d += If{sbrc_type == "SBRC_3D_FFT_TRANS_XY_Z", xy_z_offset};
        offset_3d += Else{z_xy_offset};

        // Offset in/out/lds
        offset_3d += Assign{offset_in,
                            plane_id * stride_plane_in
                                + tile_index_in_plane * transforms_per_block * stride_load_in};
        offset_3d += Assign{offset_out,
                            plane_id * stride_plane_out
                                + tile_index_in_plane * transforms_per_block * stride0_out};

        stmts += If{sbrc_type == "SBRC_2D", offset_2d};
        stmts += Else{offset_3d};

        stmts += LineBreak{};
        stmts += Assign{batch, block_id / num_of_tiles_in_batch};
        stmts += Assign{offset, offset + batch * stride[dim]};

        if(!direct_to_from_reg)
        {
            stmts += Assign{transform,
                            tile_index_in_plane * transforms_per_block
                                + thread_id / threads_per_transform};
            stmts += Assign{stride_lds, (length + get_lds_padding())};
            stmts += Assign{offset_lds, stride_lds * (transform % transforms_per_block)};
        }
        else
        {
            stmts += Assign{
                transform,
                Ternary{
                    lds_linear,
                    tile_index_in_plane * transforms_per_block + thread_id / threads_per_transform,
                    tile_index_in_plane * transforms_per_block + thread_id % transforms_per_block}};
            stmts += Assign{stride_lds,
                            Ternary{lds_linear,
                                    length + get_lds_padding(),
                                    transforms_per_block + get_lds_padding()}};
            stmts += Assign{offset_lds,
                            Ternary{lds_linear,
                                    stride_lds * (transform % transforms_per_block),
                                    thread_id % transforms_per_block}};
        }

        if(emitGlobalId)
        {
            stmts += Assign{global_load_data_offset,
                            global_load_data_offset
                                + tile_index_in_plane * transforms_per_block * length
                                + batch * global_stride_in[2]};
            stmts += Assign{global_load_transf_offset,
                            global_load_transf_offset
                                + tile_index_in_plane * transforms_per_block * length};
            stmts += Assign{global_store_data_offset,
                            global_store_data_offset + tile_index_in_plane * transforms_per_block
                                + batch * global_stride_out[2]};
            stmts
                += Assign{global_store_transf_offset,
                          global_store_transf_offset + tile_index_in_plane * transforms_per_block};
        }

        stmts += Declaration{edge, "false"};
        stmts += Declaration{thread};
        stmts += Declaration{tid_hor};

        stmts += If{transpose_type == "TILE_UNALIGNED",
                    {Assign{edge,
                            Ternary{Parens((tile_index_in_plane + 1) * transforms_per_block
                                           > len_along_block),
                                    "true",
                                    "false"}}}};

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        StatementList stmts;

        if(!load_registers)
        {
            // #-load for each thread to load all element in a tile
            auto num_load_blocks = (length * transforms_per_block) / workgroup_size;
            // #-row for a load block (global mem) = each thread will across these rows
            auto tid0_inc_step = transforms_per_block / num_load_blocks;
            // tpb/num_load_blocks, also = wgs/length, it's possible that they aren't divisible.
            bool divisible = (transforms_per_block % num_load_blocks) == 0;

            // [dim0, dim1] = [tid_ver, tid_hor] :
            // each thread reads position [tid_ver, tid_hor], [tid_ver+step_h*1, tid_hor] , [tid_ver+step_h*2, tid_hor]...
            // tid_ver walks the columns; tid_hor walks the rows
            if(divisible)
            {
                stmts += Assign{thread, thread_id / length};
                stmts += Assign{tid_hor, thread_id % length};
            }

            // we need to take care about two diff cases for offset in buf and lds
            //  divisible: each load leads to a perfect block: update offset much simpler
            //  indivisible: need extra div and mod, otherwise each load will have some elements un-loaded:
            auto offset_tile_rbuf = [&](unsigned int i) {
                if(divisible)
                    return tid_hor * stride0 + (thread + i * tid0_inc_step) * stride_load_in;

                else
                    return ((thread_id + i * workgroup_size) % length) * stride0
                           + ((thread_id + i * workgroup_size) / length) * stride_load_in;
            };
            auto offset_tile_wlds = [&](unsigned int i) {
                if(divisible)
                    return tid_hor * 1 + (thread + i * tid0_inc_step) * stride_lds;
                else
                    return ((thread_id + i * workgroup_size) % length) * 1
                           + ((thread_id + i * workgroup_size) / length) * stride_lds;
            };
            auto offset_tile_wlds_trans = [&](unsigned int i) {
                if(divisible)
                    return tid_hor * stride_lds + (thread + i * tid0_inc_step) * 1;
                else
                    return ((thread_id + i * workgroup_size) % length) * stride_lds
                           + ((thread_id + i * workgroup_size) / length) * 1;
            };
            auto offset_tile_tmp = [&](unsigned int i) {
                if(divisible)
                    return tid_hor + (thread + i * tid0_inc_step) * length;

                else
                    return ((thread_id + i * workgroup_size) % length)
                           + ((thread_id + i * workgroup_size) / length) * length;
            };

            StatementList regular_load;

            for(unsigned int i = 0; i < num_load_blocks; ++i)
            {
                Expression tmp_idx = offset_tile_tmp(i);

                if(emitGlobalId)
                {
                    regular_load += Assign{global_data_id, global_load_data_offset + tmp_idx};
                    regular_load += Assign{global_transf_id, global_load_transf_offset + tmp_idx};
                }

                Expression buf_idx = offset_tile_rbuf(i);
                Expression lds_idx = offset_tile_wlds(i);
                if(direct_to_from_reg)
                    lds_idx = Ternary{lds_linear, offset_tile_wlds(i), offset_tile_wlds_trans(i)};
                regular_load += Assign{lds_complex[lds_idx], LoadGlobal{buf, offset_in + buf_idx}};
            }

            StatementList edge_load;

            Variable t{"t", "unsigned int"};
            if(divisible)
            {
                Expression tmp_idx = tid_hor + (thread + t) * length;
                Expression buf_idx = tid_hor * stride0 + (thread + t) * stride_load_in;
                Expression lds_idx = tid_hor * 1 + (thread + t) * stride_lds;
                Expression pred
                    = (tile_index_in_plane * transforms_per_block + thread + t) < len_along_block;
                if(direct_to_from_reg)
                {
                    lds_idx = Ternary{lds_linear,
                                      tid_hor * 1 + (thread + t) * stride_lds,
                                      tid_hor * stride_lds + (thread + t) * 1};
                }

                if(emitGlobalId)
                {
                    edge_load += For{
                        t,
                        0,
                        pred,
                        tid0_inc_step,
                        {Assign{global_data_id, global_load_data_offset + tmp_idx},
                         Assign{global_transf_id, global_load_transf_offset + tmp_idx},
                         Assign{lds_complex[lds_idx], LoadGlobal{buf, offset_in + buf_idx}}}};
                }
                else
                {
                    edge_load += For{
                        t,
                        0,
                        pred,
                        tid0_inc_step,
                        {Assign{lds_complex[lds_idx], LoadGlobal{buf, offset_in + buf_idx}}}};
                }
            }
            else
            {
                Expression tmp_idx
                    = ((thread_id + t) % length) + ((thread_id + t) / length) * length;
                Expression buf_idx = ((thread_id + t) % length) * stride0
                                     + ((thread_id + t) / length) * stride_load_in;
                Expression lds_idx
                    = ((thread_id + t) % length) * 1 + ((thread_id + t) / length) * stride_lds;
                Expression pred = (thread_id + t) < (length * transforms_per_block);
                if(direct_to_from_reg)
                    lds_idx = Ternary{
                        lds_linear,
                        ((thread_id + t) % length) * 1 + ((thread_id + t) / length) * stride_lds,
                        ((thread_id + t) % length) * stride_lds + ((thread_id + t) / length) * 1};

                if(emitGlobalId)
                {
                    edge_load += For{
                        t,
                        0,
                        pred,
                        workgroup_size,
                        {Assign{global_data_id, global_load_data_offset + tmp_idx},
                         Assign{global_transf_id, global_load_transf_offset + tmp_idx},
                         Assign{lds_complex[lds_idx], LoadGlobal{buf, offset_in + buf_idx}}}};
                }
                else
                {
                    edge_load += For{
                        t,
                        0,
                        pred,
                        workgroup_size,
                        {Assign{lds_complex[lds_idx], LoadGlobal{buf, offset_in + buf_idx}}}};
                }
            }

            stmts += If{Or{transpose_type != "TILE_UNALIGNED", Not{edge}}, regular_load};
            stmts += Else{edge_load};
        }
        else
        {
            // TODO: We don't go this path..
            stmts += CommentLines{"For row-input we don't load to reg"};
        }

        return stmts;
    }

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;
        StatementList non_edge_stmts;
        StatementList edge_stmts;
        Expression    pred{tile_index_in_plane * transforms_per_block + tid_hor < len_along_block};

        if(!store_registers)
        {
            // #-column for a store block (global mem)
            auto store_block_w = transforms_per_block;
            // #-store for each thread to store all element in a tile
            auto num_store_blocks = (length * transforms_per_block) / workgroup_size;
            // #-row for a store block (global mem) = each thread will across these rows
            auto tid0_inc_step = length / num_store_blocks;
            // length / ((length * transforms_per_block) / workgroup_size) = wgs/tpb = blockwidth
            // so divisible should always true. But still put the logic here, since it's a
            // generator-wise if-else test, not inside the kernel code.
            bool divisible = (length % num_store_blocks) == 0;

            // [dim0, dim1] = [tid_ver, tid_hor]:
            // each thread write GLOBAL_POS [tid_ver, tid_hor], [tid_ver+step_h*1, tid_hor] , [tid_ver+step_h*2, tid_hor].
            // NB: This is a transpose from LDS to global, so the pos of lds_read should be remapped
            if(divisible)
            {
                stmts += Assign{thread, thread_id / store_block_w};
                stmts += Assign{tid_hor, thread_id % store_block_w};
            }

            StatementList regular_store;

            // we need to take care about two diff cases for offset in buf and lds
            //  divisible: each store leads to a perfect block: update offset much simpler
            //  indivisible: need extra div and mod, otherwise each store will have some elements un-set:
            auto offset_tile_wbuf = [&](unsigned int i) {
                if(divisible)
                    return tid_hor * stride0 + (thread + i * tid0_inc_step) * stride_store_out;
                else
                    return ((thread_id + i * workgroup_size) % store_block_w) * stride0
                           + (thread + i * tid0_inc_step) * stride_store_out;
            };
            auto offset_tile_rlds = [&](unsigned int i) {
                if(divisible)
                    return tid_hor * stride_lds + (thread + i * tid0_inc_step) * 1;
                else
                    return ((thread_id + i * workgroup_size) % store_block_w) * stride_lds
                           + ((thread_id + i * workgroup_size) / store_block_w) * 1;
            };
            auto offset_tile_tmp = [&](unsigned int i) {
                if(divisible)
                    return tid_hor + (thread + i * tid0_inc_step) * lengths[1];
                else
                    return ((thread_id + i * workgroup_size) % store_block_w)
                           + (thread + i * tid0_inc_step) * lengths[1];
            };

            for(unsigned int i = 0; i < num_store_blocks; ++i)
            {
                if(emitGlobalId)
                {
                    regular_store
                        += Assign{global_data_id, global_store_data_offset + offset_tile_tmp(i)};
                    regular_store += Assign{global_transf_id,
                                            global_store_transf_offset + offset_tile_tmp(i)};
                }

                regular_store += StoreGlobal{
                    buf, offset + offset_tile_wbuf(i), lds_complex[offset_tile_rlds(i)]};
            };

            // ERC_Z_XY
            auto          i = num_store_blocks;
            StatementList stmts_erc_post_no_edge;
            stmts_erc_post_no_edge
                += CommentLines{"extra global write for SBRC_3D_FFT_ERC_TRANS_Z_XY"};
            stmts_erc_post_no_edge += If{
                thread_id < transforms_per_block,
                {StoreGlobal{buf, offset + offset_tile_wbuf(i), lds_complex[offset_tile_rlds(i)]}}};
            non_edge_stmts += regular_store;
            non_edge_stmts += If{sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY", stmts_erc_post_no_edge};

            StatementList stmts_erc_post_edge;
            stmts_erc_post_edge
                += CommentLines{"extra global write for SBRC_3D_FFT_ERC_TRANS_Z_XY"};
            stmts_erc_post_edge += If{thread_id < Parens{len_along_block % transforms_per_block},
                                      {StoreGlobal{buf,
                                                   offset + offset_tile_wbuf(i),
                                                   lds_complex[tid_hor * stride_lds + length]}}};
            edge_stmts += regular_store;
            edge_stmts += If{sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY", stmts_erc_post_edge};
        }
        else
        {
            // Don't forget this.
            stmts += Assign{thread, thread_id / transforms_per_block};
            stmts += Assign{tid_hor, thread_id % transforms_per_block};

            auto width     = factors.back();
            auto cumheight = product(factors.begin(), factors.begin() + (factors.size() - 1));
            auto height    = static_cast<float>(length) / width / threads_per_transform;

            auto store_global = std::mem_fn(&StockhamKernelRC::store_global_generator);
            non_edge_stmts += add_work(std::bind(store_global, this, _1, _2, _3, _4, _5, cumheight),
                                       width,
                                       height,
                                       ThreadGuardMode::GUARD_BY_IF);

            edge_stmts = non_edge_stmts;
        }

        stmts += If{Or{transpose_type != "TILE_UNALIGNED", Not{edge}}, non_edge_stmts};
        stmts += Else{{If{pred, edge_stmts}}};

        return stmts;
    }

    TemplateList device_templates() override
    {
        TemplateList tpls = StockhamKernel::device_templates();
        tpls.arguments.insert(tpls.arguments.begin() + 2, sbrc_type);
        tpls.arguments.insert(tpls.arguments.begin() + 3, transpose_type);
        return tpls;
    }

    TemplateList global_templates() override
    {
        TemplateList tpls = StockhamKernel::global_templates();
        tpls.arguments.insert(tpls.arguments.begin() + 2, sbrc_type);
        tpls.arguments.insert(tpls.arguments.begin() + 3, transpose_type);
        return tpls;
    }

    TemplateList device_call_templates() override
    {
        TemplateList tpls = StockhamKernel::device_call_templates();
        tpls.arguments.insert(tpls.arguments.begin() + 2, sbrc_type);
        tpls.arguments.insert(tpls.arguments.begin() + 3, transpose_type);
        return tpls;
    }

    // POSTPROCESSING SBRC_3D_FFT_ERC_TRANS_Z_XY
    StatementList sbrc_erc_post_process()
    {
        StatementList stmts;
        Variable      null{"nullptr", "nullptr_t"};

        // Todo: We might not have to sync here which depends on the access pattern
        stmts += SyncThreads{};
        stmts += LineBreak{};

        // length is Half_N, remember quarter_N should be is (Half_N + 1) / 2
        // And need to set the Ndiv4 template argument
        Variable Ndiv4{length % 2 == 0 ? "true" : "false", "bool"};
        for(unsigned int h = 0; h < transforms_per_block; ++h)
        {
            stmts += Call{"post_process_interleaved_inplace",
                          {scalar_type, Ndiv4, Variable{"CallbackType::NONE", ""}},
                          {thread_id,
                           length - thread_id,
                           length,
                           (length + 1) / 2,
                           lds_complex + (h * stride_lds),
                           0,
                           twiddles + (length - factors.front()),
                           null,
                           null,
                           0,
                           null,
                           null}};
        }
        return {If{sbrc_type == "SBRC_3D_FFT_ERC_TRANS_Z_XY", stmts}};
    }

    StatementList real_trans_pre_post() override
    {
        if(ebtype != EmbeddedType::Real2C_POST)
            return {};

        // POST is implemented when sbrc_typs is SBRC_3D_FFT_ERC_TRANS_Z_XY
        StatementList stmts;
        stmts += CommentLines{
            "handle post_procession SBRC_3D_FFT_ERC_TRANS_Z_XY in lds after transform"};
        stmts += sbrc_erc_post_process();
        return stmts;
    }
};
