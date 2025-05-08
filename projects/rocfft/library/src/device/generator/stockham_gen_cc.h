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

struct StockhamKernelCC : public StockhamKernel
{
    explicit StockhamKernelCC(const StockhamGeneratorSpecs& specs,
                              bool                          largeTwdBatchIsTransformCount,
                              bool                          emitGlobalId)
        : StockhamKernel(specs)
        , largeTwdBatchIsTransformCount(largeTwdBatchIsTransformCount)
        , emitGlobalId(emitGlobalId)
    {
        large_twiddle_steps.decl_default = 3;
        large_twiddle_base.decl_default  = 8;
    }

    bool largeTwdBatchIsTransformCount = false;
    bool emitGlobalId                  = false;

    //
    // templates
    //
    Variable intrinsic_mode{"intrinsic_mode", "IntrinsicAccessType"};
    Variable apply_large_twiddle{"apply_large_twiddle", "bool"};
    Variable large_twiddle_steps{"large_twiddle_steps", "size_t"};
    Variable large_twiddle_base{"large_twiddle_base", "size_t"};

    //
    // arguments
    //
    Variable large_twiddles{"large_twiddles", "const scalar_type", true};
    Variable trans_local{"trans_local", "size_t"};

    //
    // locals
    //
    Variable tile_index{"tile_index", "size_t"};
    Variable num_of_tiles{"num_of_tiles", "size_t"};
    Variable in_bound{"in_bound", "bool"};
    Variable thread{"thread", "unsigned int"}; // replacing tid_ver
    Variable tid_hor{"tid_hor", "unsigned int"}; // id along row
    Variable stride_in{"stride_in", "const size_t", true};
    Variable stride_out{"stride_out", "const size_t", true};
    Variable length_M_blue{"length_M_blue", "const size_t"};

    // large twiddle support
    Multiply ltwd_entries{Parens{ShiftLeft{1, large_twiddle_base}}, 3};
    And      ltwd_in_lds{apply_large_twiddle, Less{large_twiddle_base, 8}};
    Variable large_twd_lds{"large_twd_lds", "scalar_type", true, true};

    std::string tiling_name() override
    {
        return "SBCC";
    }

    // TODO- support embedded Pre/Post
    StatementList set_direct_to_from_registers() override
    {
        // CC: "direct-to-reg" and non-linear, also "direct-from-reg" at the same time
        if(direct_to_from_reg)
            return {Declaration{direct_load_to_reg,
                                directReg_type == "DirectRegType::TRY_ENABLE_IF_SUPPORT"},
                    Declaration{direct_store_from_reg, direct_load_to_reg},
                    Declaration{lds_linear, Not{direct_load_to_reg}}};
        else
            return {Declaration{direct_load_to_reg, Literal{"false"}},
                    Declaration{direct_store_from_reg, Literal{"false"}},
                    Declaration{lds_linear, Literal{"true"}}};
    }

    StatementList set_lds_is_real() override
    {
        return {Declaration{lds_is_real, Literal{half_lds ? "true" : "false"}}};
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

            if(emitGlobalId)
            {
                load += Assign{global_data_id,
                               global_load_data_offset + tid_hor
                                   + Parens{Expression{idx}} * lengths[1]};
                load += Assign{global_transf_id,
                               global_load_transf_offset + tid_hor
                                   + Parens{Expression{idx}} * lengths[1]};
            }

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

    StatementList store_global_generator(unsigned int h,
                                         unsigned int hr,
                                         unsigned int width,
                                         unsigned int dt,
                                         Expression   guard,
                                         unsigned int cumheight,
                                         bool         intrinsic,
                                         Expression   pred)
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

            if(intrinsic)
            {
                // no need to and with trivial "true"
                work += IntrinsicStore{buf,
                                       tid_hor * stride[1] + Parens{Expression{idx}} * stride0,
                                       offset,
                                       R[hr * width + w],
                                       std::holds_alternative<Literal>(guard) ? pred
                                                                              : (guard && pred)};
            }
            else
            {
                work
                    += StoreGlobal{buf,
                                   offset + tid_hor * stride[1] + Parens{Expression{idx}} * stride0,
                                   R[hr * width + w]};
            }
        }
        return work;
    }

    StatementList calculate_offsets() override
    {
        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable plength{"plength", "size_t"};
        Variable global_stride_in{"global_stride_in", "const size_t"};
        Variable global_stride_out{"global_stride_out", "const size_t"};

        StatementList stmts;
        stmts += Declaration{tile_index};
        stmts += Declaration{num_of_tiles};

        stmts += LineBreak{};
        stmts += CommentLines{"calculate offset for each tile:",
                              "  tile_index  now means index of the tile along dim1",
                              "  num_of_tiles now means number of tiles along dim1"};
        stmts += Declaration{plength, 1};
        stmts += Declaration{remaining};
        stmts += Declaration{index_along_d};
        if(emitGlobalId)
        {
            stmts += Declaration{global_load_data_offset, 0};
            stmts += Declaration{global_load_transf_offset, 0};
            stmts += Declaration{global_store_data_offset, 0};
            stmts += Declaration{global_store_transf_offset, 0};

            stmts += Declaration{global_data_id, 0};
            stmts += Declaration{global_transf_id, 0};
        }
        stmts += Assign{num_of_tiles, (lengths[1] - 1) / transforms_per_block + 1};
        stmts += Assign{plength, num_of_tiles};
        stmts += Assign{tile_index, block_id % num_of_tiles};
        stmts += Assign{remaining, block_id / num_of_tiles};
        stmts += Assign{offset, tile_index * transforms_per_block * stride[1]};

        if(emitGlobalId)
        {
            stmts += Declaration{Variable{"global_stride_in[3]", "const size_t"},
                                 Literal{"{global_stride_in_0, global_stride_in_1, global_idist}"}};
            stmts
                += Declaration{Variable{"global_stride_out[3]", "const size_t"},
                               Literal{"{global_stride_out_0, global_stride_out_1, global_odist}"}};

            stmts += For{
                d,
                2,
                d < dim,
                1,
                {Assign{plength, plength * lengths[d]},
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
            stmts += For{d,
                         2,
                         d < dim,
                         1,
                         {Assign{plength, plength * lengths[d]},
                          Assign{index_along_d, remaining % lengths[d]},
                          Assign{remaining, remaining / lengths[d]},
                          Assign{offset, offset + index_along_d * stride[d]}}};
        }

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

        stmts += Declaration{
            in_bound,
            Ternary{Parens((tile_index + 1) * transforms_per_block > lengths[1]), "false", "true"}};

        // [dim0, dim1] = [tid_ver, tid_hor] :
        // each thread reads position [tid_ver, tid_hor], [tid_ver+step_height*1, tid_hor] , [tid_ver+step_height*2, tid_hor]...
        // tid_ver walks the columns; tid_hor walks the rows
        stmts += Declaration{thread, thread_id / transforms_per_block};
        stmts += Declaration{tid_hor, thread_id % transforms_per_block};

        if(emitGlobalId)
        {
            stmts += Assign{global_load_data_offset,
                            global_load_data_offset + tile_index * transforms_per_block
                                + batch * global_stride_in[2]};
            stmts += Assign{global_load_transf_offset,
                            global_load_transf_offset + tile_index * transforms_per_block};
            stmts += Assign{global_store_data_offset,
                            global_store_data_offset + tile_index * transforms_per_block
                                + batch * global_stride_out[2]};
            stmts += Assign{global_store_transf_offset,
                            global_store_transf_offset + tile_index * transforms_per_block};
        }

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        StatementList stmts;
        StatementList tmp_stmts;
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

            for(unsigned int i = 0; i < length / stripmine_h; ++i)
            {
                if(emitGlobalId)
                {
                    tmp_stmts += Assign{global_data_id,
                                        global_load_data_offset + tid_hor
                                            + (thread + i * stripmine_h) * lengths[1]};
                    tmp_stmts += Assign{global_transf_id,
                                        global_load_transf_offset + tid_hor
                                            + (thread + i * stripmine_h) * lengths[1]};
                }

                tmp_stmts += Assign{lds_complex[offset_tile_wlds(i)],
                                    LoadGlobal{buf, offset + offset_tile_rbuf(i)}};
            }

            stmts += CommentLines{
                "no intrinsic when load to lds. FIXME- check why use nested branch is better"};
            stmts += If{in_bound, tmp_stmts};
            stmts += If{Not{in_bound}, {If{pred, tmp_stmts}}};
            // stmts += Else{{If{pred, tmp_stmts}}}; // FIXME: Need to check with compiler team.
        }
        else
        {
            StatementList intrinsic_stmts;
            StatementList non_intrinsic_stmts;

            unsigned int width  = factors[0];
            auto         height = static_cast<float>(length) / width / threads_per_transform;

            auto load_global = std::mem_fn(&StockhamKernelCC::load_global_generator);
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

            tmp_stmts += add_work(
                std::bind(load_global, this, _1, _2, _3, _4, _5, false, Expression{in_bound}),
                width,
                height,
                ThreadGuardMode::GUARD_BY_IF,
                true);
            non_intrinsic_stmts += CommentLines{"can't use intrinsic load"};
            non_intrinsic_stmts += If{in_bound, tmp_stmts};
            non_intrinsic_stmts += If{!in_bound, {If{pred, tmp_stmts}}};

            stmts += If{intrinsic_mode != "IntrinsicAccessType::DISABLE_BOTH", intrinsic_stmts};
            stmts += Else{non_intrinsic_stmts};
            // stmts += Else{{If{in_bound, tmp_stmts}}};
        }

        return stmts;
    }

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;
        StatementList tmp_stmts;
        Expression    pred{tile_index * transforms_per_block + tid_hor < lengths[1]};

        if(!store_registers)
        {
            auto stripmine_w = transforms_per_block;
            auto stripmine_h = workgroup_size / stripmine_w;

            auto offset_tile_wbuf = [&](unsigned int i) {
                return tid_hor * stride[1] + (thread + i * stripmine_h) * stride0;
            };
            auto offset_tile_rlds = [&](unsigned int i) {
                return tid_hor * stride_lds + (thread + i * stripmine_h) * 1;
            };

            for(unsigned int i = 0; i < length / stripmine_h; ++i)
            {
                if(emitGlobalId)
                {
                    tmp_stmts += Assign{global_data_id,
                                        global_store_data_offset + tid_hor
                                            + (thread + i * stripmine_h) * lengths[1]};
                    tmp_stmts += Assign{global_transf_id,
                                        global_store_transf_offset + tid_hor
                                            + (thread + i * stripmine_h) * lengths[1]};
                }

                tmp_stmts += StoreGlobal{
                    buf, offset + offset_tile_wbuf(i), lds_complex[offset_tile_rlds(i)]};
            }

            stmts += CommentLines{
                "no intrinsic when store from lds. FIXME- check why use nested branch is better"};
            stmts += If{in_bound, tmp_stmts};
            stmts += If{Not{in_bound}, {If{pred, tmp_stmts}}};
            // stmts += Else{{If{pred, tmp_stmts}}}; // FIXME: Need to check with compiler team.
        }
        else
        {
            StatementList intrinsic_stmts;
            StatementList non_intrinsic_stmts;

            auto width     = factors.back();
            auto cumheight = product(factors.begin(), factors.begin() + (factors.size() - 1));
            auto height    = static_cast<float>(length) / width / threads_per_transform;

            auto store_global = std::mem_fn(&StockhamKernelCC::store_global_generator);
            intrinsic_stmts += CommentLines{"use intrinsic store"};
            intrinsic_stmts += add_work(std::bind(store_global,
                                                  this,
                                                  _1,
                                                  _2,
                                                  _3,
                                                  _4,
                                                  _5,
                                                  cumheight,
                                                  true,
                                                  Expression{Parens(in_bound || pred)}),
                                        width,
                                        height,
                                        ThreadGuardMode::GURAD_BY_FUNC_ARG);

            tmp_stmts += add_work(
                std::bind(
                    store_global, this, _1, _2, _3, _4, _5, cumheight, false, Expression{in_bound}),
                width,
                height,
                ThreadGuardMode::GUARD_BY_IF);
            non_intrinsic_stmts += CommentLines{"can't use intrinsic store"};
            non_intrinsic_stmts += If{in_bound, tmp_stmts};
            non_intrinsic_stmts += If{!in_bound, {If{pred, tmp_stmts}}};

            stmts += If{intrinsic_mode == "IntrinsicAccessType::ENABLE_BOTH", intrinsic_stmts};
            stmts += Else{non_intrinsic_stmts};
            // stmts += Else{{If{in_bound, {If{pred, tmp_stmts}}}}};
        }

        return stmts;
    }

    TemplateList device_templates() override
    {
        TemplateList tpls = StockhamKernel::device_templates();
        tpls.append(apply_large_twiddle);
        tpls.append(large_twiddle_steps);
        tpls.append(large_twiddle_base);
        return tpls;
    }

    ArgumentList device_arguments() override
    {
        ArgumentList args = StockhamKernel::device_arguments();
        args.append(large_twiddles);
        args.append(trans_local);
        return args;
    }

    TemplateList global_templates() override
    {
        TemplateList tpls = StockhamKernel::global_templates();
        tpls.append(intrinsic_mode);
        tpls.append(apply_large_twiddle);
        tpls.append(large_twiddle_steps);
        tpls.append(large_twiddle_base);
        return tpls;
    }

    ArgumentList global_arguments() override
    {
        // insert large twiddles
        ArgumentList arglist = StockhamKernel::global_arguments();
        arglist.arguments.insert(arglist.arguments.begin() + 1, large_twiddles);
        return arglist;
    }

    TemplateList device_call_templates() override
    {
        TemplateList tpls = StockhamKernel::device_call_templates();
        tpls.append(apply_large_twiddle);
        tpls.append(large_twiddle_steps);
        tpls.append(large_twiddle_base);
        return tpls;
    }

    std::vector<Expression> device_call_arguments(unsigned int call_iter) override
    {
        std::vector<Expression> args = StockhamKernel::device_call_arguments(call_iter);
        auto which = Ternary{Parens{And{apply_large_twiddle, large_twiddle_base < 8}},
                             Parens{large_twd_lds},
                             Parens{large_twiddles}};
        args.push_back(which);
        args.push_back(largeTwdBatchIsTransformCount ? batch : transform);
        return args;
    }

    StatementList large_twiddles_load() override
    {
        Variable ltwd_id{"ltwd_id", "size_t"};

        StatementList stmts;
        stmts += CommentLines{
            "large_twd_lds starts after the row FFT data in LDS (transforms_per_block * length)"};
        if(half_lds)
            stmts += CommentLines{"data is halved for half_lds"};
        stmts += Declaration{large_twd_lds,
                             lds_complex + transforms_per_block * length / (half_lds ? 2 : 1)};
        stmts += If{ltwd_in_lds,
                    {Declaration{ltwd_id, thread_id},
                     While{Less{ltwd_id, ltwd_entries},
                           {Assign{large_twd_lds[ltwd_id], large_twiddles[ltwd_id]},
                            AddAssign(ltwd_id, workgroup_size)}}}};

        return stmts;
    }

    StatementList large_twiddles_multiply_generator(unsigned int h,
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
            // FIXME- using a .cast('type') would be graceful!
            //        Why casting to ((int)thread % 8) only when passing to TW_NSteps ?
            //        This is completely based on the testing result. We observed that it can
            //        reduce a few vgprs (we don't know why since its behind the compiler)
            //        and avoid the drop of occuapancy (espcially for sbcc_len64_inverse)
            // idx = Parens(Parens(thread % cumheight) + w * cumheight) * trans_local
            auto idx = std::string("(((int)(") + thread.render() + " + " + std::to_string(dt)
                       + " + " + std::to_string(h * threads_per_transform) + ") % "
                       + std::to_string(cumheight) + ") + " + std::to_string(w) + " * "
                       + std::to_string(cumheight) + ") * " + trans_local.render();
            work += Assign{
                W,
                CallExpr{"TW_NSteps",
                         TemplateList{scalar_type, large_twiddle_base, large_twiddle_steps},
                         {large_twiddles, idx}}};
            work += Assign{t, TwiddleMultiply{R[hr * width + w], W}};
            work += Assign{R[hr * width + w], t};
        }
        return work;
    }

    StatementList
        large_twiddles_multiply(unsigned int width, double height, unsigned int cumheight) override
    {
        StatementList stmts;

        stmts += CommentLines{"large twiddle multiplication"};

        auto mf = std::mem_fn(&StockhamKernelCC::large_twiddles_multiply_generator);
        stmts += add_work(std::bind(mf, this, _1, _2, _3, _4, _5, cumheight),
                          width,
                          height,
                          ThreadGuardMode::NO_GUARD);

        return {If{apply_large_twiddle, stmts}};
    }
};
