// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "stockham_gen_cc.h"

// TODO: - Kernel is not getting launched, found out why
//       - Check launch bounds.
//       - Implementation here used a kernel with work_group_size = 256, however, the prototype was using 64.
//         Change kernel_generator.py to use 64 and fix all the issues, comparing again with the prototype.
//       - Start testing with different threads_per_transform once the original configuration works.
//       - Then test with other lengths and direct_from_reg=true, half_lds=true, etc.

struct StockhamPartialPassKernelCC : public StockhamKernelCC
{
    explicit StockhamPartialPassKernelCC(const StockhamGeneratorSpecs& specs,
                                         bool largeTwdBatchIsTransformCount)
        : StockhamKernelCC(specs, largeTwdBatchIsTransformCount, false)
    {
        large_twiddle_steps.decl_default = 3;
        large_twiddle_base.decl_default  = 8;

        // TODO: Address and test all "lds_linear=false" cases

        // TODO: revisit this. Test with factors_pp.size() > 1
        max_factor_pp = *std::max_element(specs.factors_pp.begin(), specs.factors_pp.end());

        // TODO: transforms_per_block_pp or threads_per_transform? Revisit all usages
        transforms_per_block_pp = transforms_per_block / max_factor_pp;
    }

    unsigned int transforms_per_block_pp;
    unsigned int max_factor_pp;

    Variable thread_lds{"thread_lds", "unsigned int"};
    Variable idx_lds{"idx_lds", "unsigned int"};
    Variable stride_lds_pp{"stride_lds_pp", "unsigned int"};
    Variable offset_lds_pp{"offset_lds_pp", "unsigned int"};

    Variable tid_hor_lds{"tid_hor_lds", "unsigned int"};
    Variable offfset_unbatched{"offfset_unbatched", "unsigned int"};
    Variable tid_hor_pp{"tid_hor_pp", "unsigned int"};
    Variable offset_tid_hor{"offset_tid_hor", "unsigned int"};
    Variable offset_pp{"offset_pp", "unsigned int"};
    Variable thread_new{"thread_new", "unsigned int"};
    Variable batch_new{"batch_new", "unsigned int"};

    Variable thread_idx{"thread_idx", "unsigned int"};
    Variable block_idx{"block_idx", "unsigned int"};

    Variable thread_in_device_twd{"thread_in_device_twd", "unsigned int"};

    Variable global_idx{"global_idx", "unsigned int"};
    Variable transpose_idx{"transpose_idx", "unsigned int"};

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

    StatementList store_pp_step_3_4_lds_generator(
        unsigned int h, unsigned int hr, unsigned int width, unsigned int dt, Expression guard)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
            //TODO: lstride not used here, address to have input/output strides working
            work += Assign(lds_complex[offset_lds + (w * stride_lds)], R[w]);

        return work;
    }

    Function generate_lds_from_reg_output_pp_step_3_4_function()
    {
        std::string function_name
            = "lds_from_reg_output_pp_step_3_4_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_templates();
        f.arguments = device_lds_reg_inout_pp_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};

        auto store_lds = std::mem_fn(&StockhamPartialPassKernelCC::store_pp_step_3_4_lds_generator);
        // last pass of store (full)
        unsigned int width  = factors.back();
        float        height = static_cast<float>(length) / width / threads_per_transform;
        body += SyncThreads();
        body += add_work(std::bind(store_lds, this, _1, _2, _3, _4, _5),
                         width,
                         height,
                         ThreadGuardMode::NO_GUARD);
        return f;
    }

    StatementList load_lds_step_3_4_generator(
        unsigned int h, unsigned int hr, unsigned int width, unsigned int dt, Expression guard)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
            //TODO: lstride not used here, address to have input/output strides working
            work += Assign(R[w], lds_complex[offset_lds + (w * stride_lds)]);

        return work;
    }

    ArgumentList device_lds_reg_inout_pp_arguments()
    {
        ArgumentList args{R, lds_complex, stride_lds, offset_lds};
        return args;
    }

    std::vector<Expression> device_lds_reg_inout_pp_device_call_arguments()
    {
        return {R, lds_complex, stride_lds_pp, offset_lds_pp};
    }

    Function generate_lds_to_reg_input_step_3_4_function()
    {
        std::string function_name
            = "lds_to_reg_input_pp_step_3_4_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_templates();
        f.arguments = device_lds_reg_inout_pp_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;

        auto load_lds = std::mem_fn(&StockhamPartialPassKernelCC::load_lds_step_3_4_generator);
        // first pass of load (full)
        unsigned int width  = factors[0];
        float        height = static_cast<float>(length) / width / threads_per_transform;
        body += SyncThreads();
        body += add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5),
                         width,
                         height,
                         ThreadGuardMode::NO_GUARD);

        return f;
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
        stmts += Assign{num_of_tiles, (lengths[1] - 1) / transforms_per_block_pp + 1};
        stmts += Assign{plength, num_of_tiles};
        stmts += Assign{tile_index, block_id % num_of_tiles};
        //TODO figure out mod 128 for other lengths
        // mod 128 required to work with nbatch > 1
        stmts += Assign{remaining, (block_id % 128) / num_of_tiles};
        stmts += Assign{offset, tile_index * transforms_per_block_pp * stride[1]};
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
        //stmts += Assign{offset, offset + batch * stride[dim]};
        if(!direct_to_from_reg)
        {
            // TODO: figure out this branch
            stmts
                += Assign{transform,
                          tile_index * transforms_per_block_pp + thread_id / threads_per_transform};
            stmts += Assign{stride_lds, (length + get_lds_padding())};

            // TODO: figure out factor 4 for other lengths
            stmts += MultiplyAssign(stride_lds, Literal{max_factor_pp});

            //stmts += Assign{offset_lds, stride_lds * (transform % transforms_per_block_pp)};
        }
        else
        {
            stmts += Assign{
                transform,
                Ternary{lds_linear,
                        tile_index * transforms_per_block_pp + thread_id / threads_per_transform,
                        tile_index * transforms_per_block_pp
                            + thread_id % transforms_per_block_pp}};
            stmts += Assign{stride_lds,
                            Ternary{lds_linear,
                                    length + get_lds_padding(),
                                    transforms_per_block_pp + get_lds_padding()}};
            // TODO: figure out factor 4 for other lengths
            stmts += MultiplyAssign(stride_lds, Literal{max_factor_pp});

            // stmts += Assign{offset_lds,
            //                 Ternary{lds_linear,
            //                         stride_lds * (transform % transforms_per_block_pp),
            //                         thread_id % transforms_per_block_pp}};
        }

        stmts += Declaration{
            in_bound,
            Ternary{
                Parens((tile_index + 1) * transforms_per_block_pp > lengths[1]), "false", "true"}};

        // [dim0, dim1] = [tid_ver, tid_hor] :
        // each thread reads position [tid_ver, tid_hor], [tid_ver+step_height*1, tid_hor] , [tid_ver+step_height*2, tid_hor]...
        // tid_ver walks the columns; tid_hor walks the rows
        stmts += Declaration{thread, thread_id / transforms_per_block_pp};
        stmts += Declaration{tid_hor, thread_id % transforms_per_block_pp};

        stmts += Declaration{thread_lds, thread_id / transforms_per_block_pp};
        stmts += Declaration{tid_hor_lds, thread_id % transforms_per_block_pp};

        // TODO: figure out factor 4 here for other lengths
        stmts += Declaration(
            tid_hor_pp, thread_id % transforms_per_block_pp + length * (thread % max_factor_pp));
        stmts += Declaration(thread_new, thread_id / (transforms_per_block_pp * max_factor_pp));
        stmts += Declaration(batch_new, block_id / (plength / max_factor_pp));

        stmts += Declaration(thread_idx, thread_id);
        stmts += Declaration(block_idx, block_id);

        // TODO: figure out factor 192 here for other lengths
        stmts += Declaration(
            offset_pp, offset + Parens(offset / length) * Literal{192} + batch_new * stride[dim]);
        stmts += Declaration(offset_tid_hor, offset_pp + tid_hor_pp * stride[1]);

        // TODO: figure out factor 4 here for other lengths
        if(!direct_to_from_reg)
            stmts += Assign{transform,
                            tile_index * transforms_per_block_pp
                                + thread_id / (threads_per_transform * max_factor_pp)};
        else
            stmts += Assign{transform,
                            Ternary{lds_linear,
                                    tile_index * transforms_per_block_pp
                                        + thread_id / (threads_per_transform * max_factor_pp),
                                    tile_index * transforms_per_block_pp
                                        + thread_id % transforms_per_block_pp}};

        stmts += Assign{offset_lds,
                        Ternary{lds_linear,
                                stride_lds * (transform % transforms_per_block_pp),
                                thread_id % transforms_per_block_pp}};

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        StatementList stmts;
        StatementList tmp_stmts;
        Expression    pred{tile_index * transforms_per_block_pp + tid_hor < lengths[1]};

        if(!load_registers)
        {
            auto stripmine_w = transforms_per_block;
            auto stripmine_h = workgroup_size / stripmine_w;

            auto offset_tile_rbuf
                = [&](unsigned int i) { return (thread_new + i * stripmine_h) * stride0; };
            auto offset_tile_wlds = [&](unsigned int i) {
                return tid_hor_lds * stride_lds
                       + (thread_lds + i * stripmine_h * max_factor_pp) * 1;
            };

            for(unsigned int i = 0; i < length / stripmine_h; ++i)
                tmp_stmts += Assign{lds_complex[offset_tile_wlds(i)],
                                    LoadGlobal{buf, offset_tid_hor + offset_tile_rbuf(i)}};

            stmts += CommentLines{
                "no intrinsic when load to lds. FIXME- check why use nested branch is better"};
            stmts += If{in_bound, tmp_stmts};
            stmts += If{Not{in_bound}, {If{pred, tmp_stmts}}};
            // stmts += Else{{If{pred, tmp_stmts}}}; // FIXME: Need to check with compiler team.
        }
        else
        {
            // TODO: Figure out this branch
            StatementList intrinsic_stmts;
            StatementList non_intrinsic_stmts;

            unsigned int width  = factors[0];
            auto         height = static_cast<float>(length) / width / threads_per_transform;

            auto load_global = std::mem_fn(&StockhamPartialPassKernelCC::load_global_generator);
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

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;
        StatementList tmp_stmts;
        Expression    pred{tile_index * transforms_per_block_pp + tid_hor < lengths[1]};

        if(!store_registers)
        {
            auto stripmine_w = transforms_per_block;
            auto stripmine_h = workgroup_size / stripmine_w;

            // TODO: stride[1] not being handle here, address this to have output strides working
            auto offset_tile_wbuf = [&](unsigned int i) {
                return offset_tid_hor + (thread_new + i * stripmine_h) * stride0;
            };
            auto offset_tile_rlds = [&](unsigned int i) {
                return tid_hor_lds * stride_lds
                       + (thread_lds + i * stripmine_h * max_factor_pp) * 1;
            };

            for(unsigned int i = 0; i < length / stripmine_h; ++i)
                tmp_stmts += StoreGlobal{
                    buf,
                    CallExpr{"local_transpose_pp_length" + std::to_string(length) + "_device",
                             {offset_tile_wbuf(i)}},
                    lds_complex[offset_tile_rlds(i)]};

            stmts += CommentLines{
                "no intrinsic when store from lds. FIXME- check why use nested branch is better"};
            stmts += If{in_bound, tmp_stmts};
            stmts += If{Not{in_bound}, {If{pred, tmp_stmts}}};
            // stmts += Else{{If{pred, tmp_stmts}}}; // FIXME: Need to check with compiler team.
        }
        else
        {
            // TODO: figure out this branch
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

    StatementList load_lds_pp_generator(unsigned int h,
                                        unsigned int hr,
                                        unsigned int width,
                                        unsigned int dt,
                                        Expression   guard,
                                        Component    component)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
        {
            const auto tid = Parens{thread + dt + h * threads_per_transform};
            const auto idx = offset_lds + (tid + w * (length / width) * max_factor_pp) * lstride;
            work += Assign(l_offset, idx);

            switch(component)
            {
            case Component::REAL:
                work += Assign(R[hr * width + w].x(), lds_real[l_offset]);
                break;
            case Component::IMAG:
                work += Assign(R[hr * width + w].y(), lds_real[l_offset]);
                break;
            case Component::BOTH:
                work += Assign(R[hr * width + w], lds_complex[l_offset]);
                break;
            }
        }

        return work;
    }

    StatementList load_lds_generator(unsigned int h,
                                     unsigned int hr,
                                     unsigned int width,
                                     unsigned int dt,
                                     Expression   guard,
                                     Component    component)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
        {
            const auto tid = Parens{thread + dt + h * threads_per_transform};
            const auto idx = offset_lds + (tid + w * (length / width) * max_factor_pp) * lstride;
            work += Assign(l_offset, idx);

            switch(component)
            {
            case Component::REAL:
                work += Assign(R[hr * width + w].x(), lds_real[l_offset]);
                break;
            case Component::IMAG:
                work += Assign(R[hr * width + w].y(), lds_real[l_offset]);
                break;
            case Component::BOTH:
                work += Assign(R[hr * width + w], lds_complex[l_offset]);
                break;
            }
        }

        return work;
    }

    Function generate_lds_to_reg_input_pp_function()
    {
        std::string function_name
            = "lds_to_reg_input_pp_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_templates();
        f.arguments = device_lds_reg_inout_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};

        body += Declaration{l_offset};

        auto load_lds = std::mem_fn(&StockhamPartialPassKernelCC::load_lds_pp_generator);
        // first pass of load (full)
        unsigned int width  = factors[0];
        float        height = static_cast<float>(length) / width / threads_per_transform;
        body += SyncThreads();
        body += add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5, Component::BOTH),
                         width,
                         height,
                         ThreadGuardMode::NO_GUARD);

        return f;
    }

    StatementList store_lds_pp_generator(unsigned int h,
                                         unsigned int hr,
                                         unsigned int width,
                                         unsigned int dt,
                                         Expression   guard,
                                         Component    component,
                                         unsigned int cumheight)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
        {
            const auto tid = thread + dt + h * threads_per_transform;
            const auto idx = offset_lds
                             + (Parens{tid / (max_factor_pp * cumheight)} * (width * cumheight)
                                + tid % (max_factor_pp * cumheight) + w * max_factor_pp * cumheight)
                                   * lstride;
            work += Assign(l_offset, idx);

            switch(component)
            {
            case Component::REAL:
                work += Assign(lds_real[l_offset], R[hr * width + w].x());
                break;
            case Component::IMAG:
                work += Assign(lds_real[l_offset], R[hr * width + w].y());
                break;
            case Component::BOTH:
                work += Assign(lds_complex[l_offset], R[hr * width + w]);
                break;
            }
        }

        return work;
    }

    StatementList store_lds_generator(unsigned int h,
                                      unsigned int hr,
                                      unsigned int width,
                                      unsigned int dt,
                                      Expression   guard,
                                      Component    component,
                                      unsigned int cumheight)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
        {
            const auto tid = thread + dt + h * threads_per_transform;
            const auto idx
                = offset_lds
                  + (Parens{tid / (cumheight * max_factor_pp)} * (width * cumheight * max_factor_pp)
                     + tid % (cumheight * max_factor_pp) + w * cumheight * max_factor_pp)
                        * lstride;
            work += Assign(l_offset, idx);

            switch(component)
            {
            case Component::REAL:
                work += Assign(lds_real[l_offset], R[hr * width + w].x());
                break;
            case Component::IMAG:
                work += Assign(lds_real[l_offset], R[hr * width + w].y());
                break;
            case Component::BOTH:
                work += Assign(lds_complex[l_offset], R[hr * width + w]);
                break;
            }
        }

        return work;
    }

    Function generate_lds_from_reg_output_pp_function()
    {
        std::string function_name
            = "lds_from_reg_output_pp_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_templates();
        f.arguments = device_lds_reg_inout_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};

        body += Declaration{l_offset};

        auto store_lds = std::mem_fn(&StockhamPartialPassKernelCC::store_lds_pp_generator);
        // last pass of store (full)
        unsigned int width     = factors.back();
        float        height    = static_cast<float>(length) / width / threads_per_transform;
        unsigned int cumheight = product(factors.begin(), factors.end() - 1);
        body += SyncThreads();
        body += add_work(std::bind(store_lds, this, _1, _2, _3, _4, _5, Component::BOTH, cumheight),
                         width,
                         height,
                         ThreadGuardMode::GUARD_BY_IF);
        return f;
    }

    Function generate_local_transpose_pp_function()
    {
        std::string function_name
            = "local_transpose_pp_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.arguments   = ArgumentList{global_idx};
        f.return_type = "unsigned int";
        f.qualifier   = "__device__";

        StatementList& body = f.body;

        // TODO: figure out these factors for other lengths
        auto factor_transpose_1 = (length * length) / max_factor_pp;
        auto factor_transpose_2 = length * max_factor_pp;
        auto factor_transpose_3 = length * length;
        auto factor_transpose_4 = length * length - length;
        auto factor_transpose_5 = length * length * length;

        body += Declaration{transpose_idx, global_idx % factor_transpose_5};
        body += Assign{transpose_idx,
                       Parens((transpose_idx % length)
                              + Parens(Parens(transpose_idx % factor_transpose_1) / length)
                                    * factor_transpose_1)
                               % factor_transpose_4
                           + (Parens(transpose_idx / factor_transpose_1) * factor_transpose_2)
                           + Parens(transpose_idx / factor_transpose_3)
                                 * (factor_transpose_3 - factor_transpose_1)};

        // TODO: clean-up this expression: global_idx / factor_transpose_5 * factor_transpose_5
        body += Assign{transpose_idx,
                       transpose_idx + global_idx / factor_transpose_5 * factor_transpose_5};

        body += ReturnExpr(transpose_idx);

        return f;
    }

    // TODO: Move this to a device function
    StatementList perform_partial_pass_step_3_4()
    {
        StatementList stmts;

        // TODO: figure out factor 1 here (what happens with different in/out strides and lengths)
        stmts += Declaration{stride_lds_pp, Literal{1}};
        stmts += Declaration{offset_lds_pp, thread_id * transforms_per_block_pp};

        auto pre_post_lds_tmpl = device_lds_reg_inout_device_call_templates();
        auto pre_post_lds_args = device_lds_reg_inout_pp_device_call_arguments();
        pre_post_lds_tmpl.set_value(stride_type.name, "lds_linear ? SB_UNIT : SB_NONUNIT");

        // TODO: handle direct_to_from_reg
        StatementList preLoad;
        preLoad += Call{"lds_to_reg_input_pp_step_3_4_length" + std::to_string(length) + "_device",
                        pre_post_lds_tmpl,
                        pre_post_lds_args};
        stmts += preLoad;

        for(unsigned int npass = 0; npass < factors_pp.size(); ++npass)
        {
            unsigned int width = factors_pp[npass];
            // TODO: revisit this. Different from same function in stockham_pp_gen_rr.h
            unsigned int height = threads_per_transform / max_factor_pp;

            auto butterfly = std::mem_fn(&StockhamKernel::butterfly_generator);
            stmts += add_work(std::bind(butterfly, this, _1, _2, _3, _4, _5),
                              width,
                              height,
                              ThreadGuardMode::NO_GUARD);
        }

        StatementList postStore;
        postStore
            += Call{"lds_from_reg_output_pp_step_3_4_length" + std::to_string(length) + "_device",
                    pre_post_lds_tmpl,
                    pre_post_lds_args};
        stmts += postStore;

        return stmts;
    }

    // The "stacked" twiddle table starts at the second factor, since
    // the first factor's values are not actually needed for
    // anything.  It still counts towards cumulative height, but we
    // subtract it from the twiddle table offset when computing an
    // index.
    StatementList apply_twiddle_generator(unsigned int h,
                                          unsigned int hr,
                                          unsigned int width,
                                          unsigned int dt,
                                          Expression   guard,
                                          unsigned int cumheight,
                                          unsigned int firstFactor)
    {
        if(hr == 0)
            hr = h;
        StatementList work;
        Expression    loadFlag{thread < length / width};
        for(unsigned int w = 1; w < width; ++w)
        {
            auto tid  = thread_in_device_twd + dt + h * threads_per_transform;
            auto tidx = cumheight - firstFactor + w - 1 + (width - 1) * (tid % cumheight);
            auto ridx = hr * width + w;

            // TODO- Can try IntrinsicLoadToDest, but should not be a bottleneck
            work += Assign(W, twiddles[tidx]);
            work += Assign(t, TwiddleMultiply(R[ridx], W));
            work += Assign(R[ridx], t);
        }
        return work;
    }

    ArgumentList device_arguments() override
    {
        ArgumentList args = StockhamKernel::device_arguments();
        args.append(large_twiddles);
        args.append(trans_local);
        args.append(thread_in_device_twd);
        return args;
    }

    std::vector<Expression> device_call_arguments(unsigned int call_iter) override
    {
        std::vector<Expression> args = StockhamKernel::device_call_arguments(call_iter);
        auto which = Ternary{Parens{And{apply_large_twiddle, large_twiddle_base < 8}},
                             Parens{large_twd_lds},
                             Parens{large_twiddles}};
        args.push_back(which);
        args.push_back(largeTwdBatchIsTransformCount ? batch : transform);
        args.push_back(thread_in_device_twd);
        return args;
    }

    Function generate_device_function()
    {
        std::string function_name
            = "forward_pp_length" + std::to_string(length) + "_" + tiling_name() + "_device";

        Function f{function_name};
        f.arguments = device_arguments();
        f.templates = device_templates();
        f.qualifier = "__device__";
        if(length == 1)
        {
            return f;
        }

        StatementList& body = f.body;
        body += Declaration{W};
        body += Declaration{t};
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};
        body += Declaration{l_offset};

        for(unsigned int npass = 0; npass < factors.size(); ++npass)
        {
            // width is the butterfly width, Radix-n.
            unsigned int width = factors[npass];
            // height is how many butterflies per thread will do on average
            float height = static_cast<float>(length) / width / threads_per_transform;

            unsigned int cumheight
                = product(factors.begin(),
                          factors.begin() + npass); // cumheight is irrelevant to the above height,
            // is used for twiddle multiplication and lds writing.

            body += LineBreak{};
            body += CommentLines{
                "pass " + std::to_string(npass) + ", width " + std::to_string(width),
                "using " + std::to_string(threads_per_transform) + " threads we need to do "
                    + std::to_string(length / width) + " radix-" + std::to_string(width)
                    + " butterflies",
                "therefore each thread will do " + std::to_string(height) + " butterflies"};

            auto load_lds  = std::mem_fn(&StockhamPartialPassKernelCC::load_lds_generator);
            auto store_lds = std::mem_fn(&StockhamPartialPassKernelCC::store_lds_generator);

            if(npass > 0)
            {
                // internal full lds2reg (both linear/nonlinear variants)
                StatementList lds2reg_full;
                lds2reg_full += SyncThreads();
                lds2reg_full
                    += add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5, Component::BOTH),
                                width,
                                height,
                                ThreadGuardMode::GUARD_BY_IF);
                body += If{Not{lds_is_real}, lds2reg_full};

                auto apply_twiddle
                    = std::mem_fn(&StockhamPartialPassKernelCC::apply_twiddle_generator);
                body += add_work(
                    std::bind(apply_twiddle, this, _1, _2, _3, _4, _5, cumheight, factors.front()),
                    width,
                    height,
                    ThreadGuardMode::NO_GUARD);
            }

            auto butterfly = std::mem_fn(&StockhamKernel::butterfly_generator);
            body += add_work(std::bind(butterfly, this, _1, _2, _3, _4, _5),
                             width,
                             height,
                             ThreadGuardMode::NO_GUARD);

            if(npass == factors.size() - 1)
                body += large_twiddles_multiply(width, height, cumheight);

            // internal lds store (half-with-linear and full-with-linear/nonlinear)
            StatementList reg2lds_full;
            StatementList reg2lds_half;
            if(npass < factors.size() - 1)
            {
                // linear variant store (half) and load (half)
                for(auto component : {Component::REAL, Component::IMAG})
                {
                    bool isFirstStore = (npass == 0) && (component == Component::REAL);
                    auto half_width   = factors[npass];
                    auto half_height
                        = static_cast<float>(length) / half_width / threads_per_transform;
                    // minimize sync as possible
                    if(!isFirstStore)
                        reg2lds_half += SyncThreads();
                    reg2lds_half += add_work(
                        std::bind(store_lds, this, _1, _2, _3, _4, _5, component, cumheight),
                        half_width,
                        half_height,
                        ThreadGuardMode::GUARD_BY_IF);

                    half_width  = factors[npass + 1];
                    half_height = static_cast<float>(length) / half_width / threads_per_transform;
                    reg2lds_half += SyncThreads();
                    reg2lds_half
                        += add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5, component),
                                    half_width,
                                    half_height,
                                    ThreadGuardMode::GUARD_BY_IF);
                }

                // internal full lds store (both linear/nonlinear variants)
                if(npass == 0)
                    reg2lds_full += If{!direct_load_to_reg, {SyncThreads()}};
                else
                    reg2lds_full += SyncThreads();
                reg2lds_full += add_work(
                    std::bind(store_lds, this, _1, _2, _3, _4, _5, Component::BOTH, cumheight),
                    width,
                    height,
                    ThreadGuardMode::GUARD_BY_IF);

                body += If{Not{lds_is_real}, reg2lds_full};
                body += Else{reg2lds_half};
            }
        }
        return f;
    }

    Function generate_global_function() override
    {
        Function f("forward_pp_length" + std::to_string(length) + "_" + tiling_name());
        f.qualifier     = "__global__";
        f.launch_bounds = workgroup_size;

        StatementList& body = f.body;
        body += CommentLines{
            "this kernel:",
            "  uses " + std::to_string(threads_per_transform) + " threads per transform",
            "  does " + std::to_string(transforms_per_block) + " transforms per thread block",
            "therefore it should be called with " + std::to_string(workgroup_size)
                + " threads per thread block"};
        body += Declaration{R};
        body += LDSDeclaration{scalar_type.name};
        body += Declaration{offset, 0};
        body += Declaration{offset_lds};
        body += Declaration{stride_lds};
        body += Declaration{batch};
        body += Declaration{transform};

        // TODO- don't override, unify them
        body += set_direct_to_from_registers();

        body += Declaration{lds_is_real, Literal{"false"}};

        body += CallbackLoadDeclaration{scalar_type.name, callback_type.name};
        body += CallbackStoreDeclaration{scalar_type.name, callback_type.name};

        body += LineBreak{};
        body += CommentLines{"large twiddles"};
        body += large_twiddles_load();

        body += LineBreak{};
        body += CommentLines{"offsets"};
        collect_length_stride(body);
        body += calculate_offsets();
        body += LineBreak{};

        StatementList loadlds;
        loadlds += CommentLines{"load global into lds"};
        loadlds += load_from_global(false);
        loadlds += LineBreak{};

        if(!direct_to_from_reg)
        {
            body += loadlds;
        }
        else
        {
            StatementList loadr;
            loadr += CommentLines{"load global into registers"};
            loadr += load_from_global(true);

            body += If{direct_load_to_reg, loadr};
            body += Else{loadlds};
        }

        // partial pass here
        body += perform_partial_pass_step_3_4();

        body += LineBreak{};
        body += CommentLines{"calc the thread_in_device value once and for all device funcs"};
        body += Declaration{thread_in_device,
                            Ternary{lds_linear,
                                    thread_id % (threads_per_transform * max_factor_pp),
                                    thread_id / transforms_per_block}};
        body += Declaration{thread_in_device_twd,
                            Parens(thread_id / max_factor_pp) % threads_per_transform};

        // before starting the transform job (core device function)
        // we call a re-load lds-to-reg function here, but it's not always doing things.
        // If we're doing direct-to-reg, this function simply returns.
        body += LineBreak{};
        body += CommentLines{"call a pre-load from lds to registers (if necessary)"};
        auto pre_post_lds_tmpl = device_lds_reg_inout_device_call_templates();
        auto pre_post_lds_args = device_lds_reg_inout_device_call_arguments();
        pre_post_lds_tmpl.set_value(stride_type.name, "lds_linear ? SB_UNIT : SB_NONUNIT");
        StatementList preLoad;
        preLoad += Call{"lds_to_reg_input_pp_length" + std::to_string(length) + "_device",
                        pre_post_lds_tmpl,
                        pre_post_lds_args};
        if(!direct_to_from_reg)
            body += preLoad;
        else
            body += If{!direct_load_to_reg, preLoad};

        body += LineBreak{};
        body += CommentLines{"transform"};
        for(unsigned int c = 0; c < n_device_calls; ++c)
        {
            auto templates = device_call_templates();
            auto arguments = device_call_arguments(c);

            templates.set_value(stride_type.name, "lds_linear ? SB_UNIT : SB_NONUNIT");

            body += Call{"forward_pp_length" + std::to_string(length) + "_" + tiling_name()
                             + "_device",
                         templates,
                         arguments};
            body += LineBreak{};
        }

        // after finishing the transform job (core device function)
        // we call a post-store reg-to-lds function here, but it's not always doing things.
        // If we're doing direct-from-reg, this function simply returns.
        body += LineBreak{};
        body += CommentLines{"call a post-store from registers to lds (if necessary)"};
        StatementList postStore;
        postStore += Call{"lds_from_reg_output_pp_length" + std::to_string(length) + "_device",
                          pre_post_lds_tmpl,
                          pre_post_lds_args};
        if(!direct_to_from_reg)
            body += postStore;
        else
            body += If{!direct_store_from_reg, postStore};

        body += LineBreak{};
        StatementList storelds;
        storelds += LineBreak{};

        storelds += LineBreak{};
        storelds += CommentLines{"store global"};
        storelds += SyncThreads{};
        storelds += store_to_global(false);

        if(!direct_to_from_reg)
        {
            body += storelds;
        }
        else
        {
            StatementList storer;
            storer += CommentLines{"store registers into global"};
            storer += store_to_global(true);

            body += If{direct_store_from_reg, storer};
            body += Else{storelds};
        }

        f.templates = global_templates();
        f.arguments = global_arguments();
        return f;
    }
};
