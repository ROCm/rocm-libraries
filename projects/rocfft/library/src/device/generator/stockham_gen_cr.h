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

struct StockhamKernelCR : public StockhamKernel
{
    explicit StockhamKernelCR(const StockhamGeneratorSpecs& specs)
        : StockhamKernel(specs)
    {
    }

    //
    // templates
    //
    Variable intrinsic_mode{"intrinsic_mode", "IntrinsicAccessType"};

    //
    // locals
    //
    Variable tile_index{"tile_index", "size_t"};
    Variable tile_length{"tile_length", "size_t"};
    Variable in_bound{"in_bound", "bool"};
    Variable thread{"thread", "unsigned int"}; // replacing tid_ver
    Variable tid_hor{"tid_hor", "unsigned int"};

    std::string tiling_name() override
    {
        return "SBCR";
    }

    // TODO- support embedded Pre/Post
    StatementList set_direct_to_from_registers() override
    {
        // CR: we do "direct-to-reg" and "non-linear", but never do "direct-from-reg"
        if(direct_to_from_reg)
            return {Declaration{
                        direct_load_to_reg,
                        ebtype == EmbeddedType::NONE
                            ? Expression{directReg_type == "DirectRegType::TRY_ENABLE_IF_SUPPORT"}
                            : Expression{Literal{"false"}}},
                    Declaration{direct_store_from_reg, Literal{"false"}},
                    Declaration{lds_linear, Not{direct_load_to_reg}}};
        else
            return {Declaration{direct_load_to_reg, Literal{"false"}},
                    Declaration{direct_store_from_reg, Literal{"false"}},
                    Declaration{lds_linear, Literal{"true"}}};
    }

    StatementList set_lds_is_real() override
    {
        // SBCR can't support half-lds
        return {Declaration{lds_is_real, Literal{"false"}}};
    }

    StatementList load_global_generator(unsigned int h,
                                        unsigned int hr,
                                        unsigned int width,
                                        unsigned int dt,
                                        Expression   guard,
                                        bool         intrinsic,
                                        Expression   pred) const
    {
        if(hr == 0)
            hr = h;
        StatementList load;
        for(unsigned int w = 0; w < width; ++w)
        {
            auto tid = Parens{thread + dt + h * threads_per_transform};
            auto idx = Parens{tid + w * length / width};
            if(intrinsic)
            {
                // no need to and with trivial "true"
                load += Assign{
                    R[hr * width + w],
                    IntrinsicLoad{
                        {buf,
                         tid_hor * stride[1] + Parens{Expression{idx}} * stride0,
                         offset,
                         std::holds_alternative<Literal>(guard) ? pred : (guard && pred)}}};
            }
            else
            {
                load += Assign{
                    R[hr * width + w],
                    LoadGlobal{buf,
                               offset + tid_hor * stride[1] + Parens{Expression{idx}} * stride0}};
            }
        }
        return load;
    }

    StatementList calculate_offsets() override
    {
        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable plength{"plength", "size_t"};

        StatementList stmts;
        stmts += Declaration{tile_index};
        stmts += Declaration{tile_length};
        stmts += LineBreak{};
        stmts += CommentLines{"calculate offset for each tile:",
                              "  tile_index  now means index of the tile along dim1",
                              "  tile_length now means number of tiles along dim1"};

        stmts += Declaration{plength, 1};
        stmts += Declaration{remaining};
        stmts += Declaration{index_along_d};

        stmts += Assign{tile_length, (lengths[1] - 1) / transforms_per_block + 1};
        stmts += Assign{plength, tile_length};
        stmts += Assign{tile_index, block_id % tile_length};
        stmts += Assign{remaining, block_id / tile_length};
        stmts += Assign{offset, tile_index * transforms_per_block * stride[1]};

        stmts += For{d,
                     2,
                     d < dim,
                     1,
                     {Assign{plength, plength * lengths[d]},
                      Assign{index_along_d, remaining % lengths[d]},
                      Assign{remaining, remaining / lengths[d]},
                      Assign{offset, offset + index_along_d * stride[d]}}};
        stmts += LineBreak{};

        stmts += Assign{batch, block_id / plength};
        stmts += Assign{offset, offset + batch * stride[dim]};

        if(!direct_to_from_reg)
        {
            stmts += Assign{transform,
                            tile_index * transforms_per_block + thread_id / threads_per_transform};
            stmts += Assign{stride_lds, (length + get_lds_padding())};
            stmts += Assign{offset_lds, stride_lds * (transform % transforms_per_block)};
        }
        else
        {
            stmts += Assign{
                transform,
                Ternary{lds_linear,
                        tile_index * transforms_per_block + thread_id / threads_per_transform,
                        tile_index * transforms_per_block + thread_id % transforms_per_block}};
            stmts += Assign{stride_lds,
                            Ternary{lds_linear,
                                    length + get_lds_padding(),
                                    transforms_per_block + get_lds_padding()}};
            stmts += Assign{offset_lds,
                            Ternary{lds_linear,
                                    stride_lds * (transform % transforms_per_block),
                                    thread_id % transforms_per_block}};
        }

        stmts += Declaration{in_bound};
        stmts += Declaration{thread};
        stmts += Declaration{tid_hor};
        stmts += Assign{
            in_bound,
            Ternary{Parens((tile_index + 1) * transforms_per_block > lengths[1]), "false", "true"}};

        // thread walks the columns; tid_hor walks the rows
        stmts += Assign{thread, thread_id / transforms_per_block};
        stmts += Assign{tid_hor, thread_id % transforms_per_block};

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        StatementList stmts;
        StatementList non_edge_stmts;
        StatementList edge_stmts;
        Expression    pred{tile_index * transforms_per_block + tid_hor < lengths[1]};

        if(!load_registers)
        {
            auto stripmine_w = transforms_per_block;
            auto stripmine_h = workgroup_size / stripmine_w;

            auto offset_tile_rbuf = [&](unsigned int i) {
                return tid_hor * stride[1] + (thread + i * stripmine_h) * stride0;
            };
            auto offset_tile_wlds = [&](unsigned int i) {
                return tid_hor * stride_lds + (thread + i * stripmine_h) * 1;
            };

            StatementList regular_load;

            for(unsigned int i = 0; i < length / stripmine_h; ++i)
                regular_load += Assign{lds_complex[offset_tile_wlds(i)],
                                       LoadGlobal{buf, offset + offset_tile_rbuf(i)}};

            StatementList stmts_c2real_pre_no_edge;
            stmts_c2real_pre_no_edge += regular_load;
            stmts_c2real_pre_no_edge += LineBreak{};

            if(ebtype == EmbeddedType::C2Real_PRE)
            {
                stmts_c2real_pre_no_edge
                    += CommentLines{"append extra global loading for C2Real "
                                    "pre-process only, one more element per col."};
                stmts_c2real_pre_no_edge += If{
                    Less{thread_id, transforms_per_block},
                    {Assign{lds_complex[tid_hor * stride_lds + length],
                            LoadGlobal{buf, offset + offset_tile_rbuf(length / stripmine_h)}}}};
            }

            non_edge_stmts += stmts_c2real_pre_no_edge;

            StatementList stmts_c2real_pre_edge;
            stmts_c2real_pre_edge += regular_load;
            stmts_c2real_pre_edge += LineBreak{};

            if(ebtype == EmbeddedType::C2Real_PRE)
            {
                stmts_c2real_pre_edge
                    += CommentLines{"append extra global loading for C2Real "
                                    "pre-process only, one more element per col."};
                stmts_c2real_pre_edge += If{
                    Less{thread_id,
                         Parens{transforms_per_block
                                - (tile_index + 1) * transforms_per_block % lengths[1]}},
                    {Assign{lds_complex[tid_hor * stride_lds + length],
                            LoadGlobal{buf, offset + offset_tile_rbuf(length / stripmine_h)}}}};
            }

            edge_stmts += stmts_c2real_pre_edge;

            stmts += If{in_bound, non_edge_stmts};
            stmts += If{Not{in_bound}, {If{pred, edge_stmts}}};
        }
        else
        {
            StatementList intrinsic_stmts;
            StatementList non_intrinsic_stmts;

            unsigned int width  = factors[0];
            auto         height = static_cast<float>(length) / width / threads_per_transform;

            auto load_global = std::mem_fn(&StockhamKernelCR::load_global_generator);
            intrinsic_stmts += CommentLines{"use intrinsic load"};
            intrinsic_stmts += CommentLines{"evaluate all flags as one rw argument"};
            intrinsic_stmts += add_work(std::bind(load_global,
                                                  this,
                                                  _1,
                                                  _2,
                                                  _3,
                                                  _4,
                                                  _5,
                                                  true,
                                                  Expression{Parens(in_bound || pred)}),
                                        width,
                                        height,
                                        ThreadGuardMode::GURAD_BY_FUNC_ARG,
                                        true);

            non_edge_stmts += add_work(
                std::bind(load_global, this, _1, _2, _3, _4, _5, false, Expression{in_bound}),
                width,
                height,
                ThreadGuardMode::GUARD_BY_IF,
                true);
            non_intrinsic_stmts += CommentLines{"can't use intrinsic load"};
            non_intrinsic_stmts += If{in_bound, non_edge_stmts};
            non_intrinsic_stmts += If{!in_bound, {If{pred, non_edge_stmts}}};

            stmts += If{intrinsic_mode != "IntrinsicAccessType::DISABLE_BOTH", intrinsic_stmts};
            stmts += Else{non_intrinsic_stmts};
        }

        return stmts;
    }

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;

        if(!store_registers)
        {
            // #-store for each thread to load all element in a tile
            auto num_store_blocks = (length * transforms_per_block) / workgroup_size;
            // #-row for a store block (global mem) = each thread will across these rows
            auto tid0_inc_step = transforms_per_block / num_store_blocks;
            // tpb/num_store_blocks, also = wgs/length, it's possible that they aren't divisible.
            bool divisible = (transforms_per_block % num_store_blocks) == 0;

            // [dim0, dim1] = [tid_ver, tid_hor] :
            // each thread reads position [tid_ver, tid_hor], [tid_ver+step_h*1, tid_hor] , [tid_ver+step_h*2, tid_hor]...
            // tid_ver walks the columns; tid_hor walks the rows
            if(divisible)
            {
                stmts += Assign{thread, thread_id / length};
                stmts += Assign{tid_hor, thread_id % length};
            }

            // we need to take care about two diff cases for offset in buf and lds
            //  divisible: each store leads to a perfect block: update offset much simpler
            //  indivisible: need extra div and mod, otherwise each store will have some elements left:
            auto offset_tile_wbuf = [&](unsigned int i) {
                if(divisible)
                    return tid_hor * stride0 + (thread + i * tid0_inc_step) * stride[1];

                else
                    return ((thread_id + i * workgroup_size) % length) * stride0
                           + ((thread_id + i * workgroup_size) / length) * stride[1];
            };
            auto offset_tile_rlds = [&](unsigned int i) {
                if(divisible)
                    return tid_hor * 1 + (thread + i * tid0_inc_step) * stride_lds;
                else
                    return ((thread_id + i * workgroup_size) % length) * 1
                           + ((thread_id + i * workgroup_size) / length) * stride_lds;
            };
            auto offset_tile_rlds_trans = [&](unsigned int i) {
                if(divisible)
                    return tid_hor * stride_lds + (thread + i * tid0_inc_step) * 1;
                else
                    return ((thread_id + i * workgroup_size) % length) * stride_lds
                           + ((thread_id + i * workgroup_size) / length) * 1;
            };

            StatementList regular_store;
            for(unsigned int i = 0; i < num_store_blocks; ++i)
            {
                Expression buf_idx = offset_tile_wbuf(i);
                Expression lds_idx = offset_tile_rlds(i);
                if(direct_to_from_reg)
                    lds_idx = Ternary{lds_linear, offset_tile_rlds(i), offset_tile_rlds_trans(i)};
                regular_store += StoreGlobal{buf, offset + buf_idx, lds_complex[lds_idx]};
            }

            StatementList edge_store;
            Variable      t{"t", "unsigned int"};
            if(divisible)
            {
                Expression buf_idx = tid_hor * stride0 + (thread + t) * stride[1];
                Expression lds_idx = tid_hor * 1 + (thread + t) * stride_lds;
                Expression pred    = (tile_index * transforms_per_block + thread + t) < lengths[1];
                if(direct_to_from_reg)
                    lds_idx = Ternary{lds_linear,
                                      tid_hor * 1 + (thread + t) * stride_lds,
                                      tid_hor * stride_lds + (thread + t) * 1};
                edge_store += For{t,
                                  0,
                                  pred,
                                  tid0_inc_step,
                                  {StoreGlobal{buf, offset + buf_idx, lds_complex[lds_idx]}}};
            }
            else
            {
                Expression buf_idx
                    = ((thread_id + t) % length) * stride0 + ((thread_id + t) / length) * stride[1];
                Expression lds_idx
                    = ((thread_id + t) % length) * 1 + ((thread_id + t) / length) * stride_lds;
                Expression pred = (thread_id + t) < (length * transforms_per_block);
                if(direct_to_from_reg)
                    lds_idx = Ternary{
                        lds_linear,
                        ((thread_id + t) % length) * 1 + ((thread_id + t) / length) * stride_lds,
                        ((thread_id + t) % length) * stride_lds + ((thread_id + t) / length) * 1};
                edge_store += For{t,
                                  0,
                                  pred,
                                  workgroup_size,
                                  {StoreGlobal{buf, offset + buf_idx, lds_complex[lds_idx]}}};
            }

            stmts += If{in_bound, regular_store};
            stmts += If{Not{in_bound}, edge_store};
            // stmts += Else{edge_store};  // FIXME: Need to check with compiler team.
        }
        else
        {
            // TODO: We don't go this path..
            stmts += CommentLines{"For row-output we don't store from reg"};
        }

        return stmts;
    }

    TemplateList global_templates() override
    {
        TemplateList tpls = StockhamKernel::global_templates();
        tpls.append(intrinsic_mode);
        return tpls;
    }

    StatementList real_trans_pre_post() override
    {
        if(ebtype != EmbeddedType::C2Real_PRE)
            return {};

        auto twd_offset = (length - factors.front());

        StatementList stmts;
        stmts += CommentLines{
            "handle even-length real to complex pre-process in lds before transform"};
        stmts += real2cmplx_pre_post(length, threads_per_transform, twd_offset);
        return stmts;
    }
};
