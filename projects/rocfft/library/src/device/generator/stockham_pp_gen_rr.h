// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "stockham_pp_gen.h"

// TODO: Support non-unit input/output strides.

// Variation of StockhamKernelRR that implements the partial pass
// method. Similarities of StockhamPartialPassKernelRR with
// StockhamKernelRR include the overall kernel structure with the
// main operations: (1) global-to-LDS, (2) LDS-to-register,
// (3) full forward/backward pass in the designated direction,
// (4) register-to-LDS and (5) LDS-to-global. The main difference
// with the base function is the added steps between operations
// (4) and (5). These steps perform partial FFT work in the
// off-direction. The full FFT work in the off-direction can be
// performed when used in conjunction with another partial-pass kernel,
// thus eliminating the need for a separate kernel with a full pass
// in the off-direction. Another difference is how data is accessed
// in global memory, and this is reflected in how calculate_offsets()
// is implemented.
struct StockhamPartialPassKernelRR : public StockhamPartialPassKernel
{
    explicit StockhamPartialPassKernelRR(const StockhamGeneratorSpecs&    specs,
                                         const StockhamPartialPassParams& params)
        : StockhamPartialPassKernel(specs, params)
    {
        length_off_dim = params.parent_length[params.off_dim];

        R.size = Expression{std::max(
            nregisters, compute_nregisters(pp_factors_prod, factors_pp, threads_per_transform_pp))};
    }

    unsigned int length_off_dim;

    std::string tiling_name() override
    {
        return "SBRR";
    }

    std::vector<unsigned int> launcher_lengths() override
    {
        return params.parent_length;
    }

    void collect_length_stride(StatementList& body)
    {
        if(static_dim)
        {
            body += Declaration{dim, static_dim};
        }
        body += Declaration{stride0, Parens{stride[0]}};
    }

    StatementList set_lds_is_real() override
    {
        // Half-LDS always disabled in partial-pass.
        // To make this option work, steps_1_2 would
        // need to implement half LDS usage in the
        // off-direction pass.
        return {Declaration{lds_is_real, Literal{"false"}}};
    }

    StatementList calculate_offsets() override
    {
        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable remaining_pp{"remaining_pp", "size_t"};

        StatementList stmts;
        stmts += Declaration{thread};
        stmts += Declaration(remaining);
        stmts += Declaration(index_along_d);
        stmts += Declaration(remaining_pp, Literal{0});
        stmts += Declaration(offset_pp, Literal{0});
        stmts += Assign{transform,
                        block_id * transforms_per_block + thread_id / threads_per_transform};
        stmts += Assign{remaining, transform};
        stmts += Assign{remaining_pp,
                        length_off_dim * Parens(transform / length_off_dim)
                            + Parens(transform % length_off_dim) / transforms_per_block
                            + Parens(transform * (length_off_dim / transforms_per_block))
                                  % length_off_dim};

        stmts += For{d,
                     1,
                     d < dim,
                     1,
                     {
                         Assign{remaining, remaining / lengths[d]},
                         Assign{index_along_d, remaining_pp % lengths[d]},
                         Assign{remaining_pp, remaining_pp / lengths[d]},
                         Assign{offset_pp, offset_pp + index_along_d * stride[d]},
                     }};

        stmts += Assign{batch, remaining};
        stmts += Assign{offset_pp, offset_pp + batch * stride[dim]};
        stmts += Assign{stride_lds, (length + get_lds_padding())};
        stmts += Assign{offset_lds, stride_lds * Parens{transform % transforms_per_block}};

        stmts += Declaration{in_bound, batch < nbatch};

        return stmts;
    }

    // Call generator as many times as needed.
    // generator accepts h, hr, width, dt, guard_pred parameters
    StatementList add_pp_work(
        std::function<StatementList(
            unsigned int, unsigned int, unsigned int, unsigned int, Expression)> generator,
        unsigned int                                                             width,
        double                                                                   height,
        ThreadGuardMode                                                          guard,
        bool trans_dir = false) const
    {
        StatementList stmts;
        unsigned int  iheight = std::floor(height);
        if(height > iheight && threads_per_transform_pp > length / width)
            iheight += 1;

        Expression guard_expr = Expression{Literal{"true"}};

        const auto work_length       = pp_factors_prod;
        const auto thread_guard_cond = work_length / width;

        // do thread guard when guard_by_if or guard_by_arg
        if(guard != ThreadGuardMode::NO_GUARD)
        {
            // using ">" : no need to test "if(thread < XXX)"" if it is always true
            if((!trans_dir && threads_per_transform_pp > (work_length / width))
               || (trans_dir && workgroup_size / transforms_per_block_pp > (work_length / width)))
            {
                if(writeGuard)
                    guard_expr = Expression{write && (thread < thread_guard_cond)};
                else
                    guard_expr = Expression{thread < thread_guard_cond};
            }
            else
            {
                if(writeGuard)
                    guard_expr = Expression{write};
            }
        }

        StatementList work;
        for(unsigned int h = 0; h < iheight; ++h)
            work += generator(h, 0, width, 0, guard_expr);

        // guard_expr is not a trivial value "true"
        if(guard == ThreadGuardMode::GUARD_BY_IF && !std::holds_alternative<Literal>(guard_expr))
        {
            stmts += CommentLines{"more than enough threads, some do nothing"};
            stmts += If{guard_expr, work};
        }
        else
        {
            stmts += work;
        }

        if(height > iheight && threads_per_transform_pp < work_length / width)
        {
            stmts += CommentLines{"not enough threads, some threads do extra work"};
            unsigned int dt = iheight * threads_per_transform_pp;

            // always do thread guard
            if(writeGuard)
                guard_expr = Expression{write && (thread + dt < thread_guard_cond)};
            else
                guard_expr = Expression{thread + dt < thread_guard_cond};

            work = generator(0, iheight, width, dt, guard_expr);

            // put in if only if guard_by_if
            if(guard == ThreadGuardMode::GUARD_BY_IF)
                stmts += If{guard_expr, work};
            else
                stmts += work;
        }

        return stmts;
    }

    StatementList load_global_generator(unsigned int h,
                                        unsigned int hr,
                                        unsigned int width,
                                        unsigned int dt,
                                        Expression   guard) const
    {
        if(hr == 0)
            hr = h;
        StatementList load;
        for(unsigned int w = 0; w < width; ++w)
        {
            auto tid = Parens{thread + dt + h * threads_per_transform};
            auto idx = Parens{tid + w * length / width};
            load += Assign{R[hr * width + w],
                           LoadGlobal{buf, offset_pp + Parens{Expression{idx}} * stride0}};
        }
        return load;
    }

    StatementList load_from_global(bool load_registers) override
    {
        StatementList stmts;
        stmts += Assign{thread, thread_id % threads_per_transform};

        if(!load_registers)
        {
            unsigned int width  = threads_per_transform;
            unsigned int height = length / width;

            for(unsigned int h = 0; h < height; ++h)
            {
                auto idx = thread + h * width;
                stmts += Assign{lds_complex[offset_lds + idx],
                                LoadGlobal{buf, offset_pp + idx * stride0}};
            }

            StatementList stmts_c2real_pre;
            stmts_c2real_pre += CommentLines{
                "use the last thread of each transform to load one more element per row"};
            stmts_c2real_pre += If{
                thread == threads_per_transform - 1,
                {Assign{lds_complex[offset_lds + thread + (height - 1) * width + 1],
                        LoadGlobal{buf, offset + (thread + (height - 1) * width + 1) * stride0}}}};
            if(ebtype == EmbeddedType::C2Real_PRE)
                stmts += stmts_c2real_pre;
        }
        else
        {
            unsigned int width  = factors[0];
            auto         height = static_cast<float>(length) / width / threads_per_transform;

            auto load_global = std::mem_fn(&StockhamPartialPassKernelRR::load_global_generator);
            stmts += add_work(std::bind(load_global, this, _1, _2, _3, _4, _5),
                              width,
                              height,
                              ThreadGuardMode::GUARD_BY_IF);
        }

        return {If{in_bound, stmts}};
    }

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;

        if(!store_registers)
        {
            auto width  = threads_per_transform;
            auto height = length / width;
            for(unsigned int h = 0; h < height; ++h)
            {
                auto idx = thread + h * width;
                stmts += StoreGlobal{buf, offset_pp + idx * stride0, lds_complex[offset_lds + idx]};
            }

            stmts += LineBreak{};
            StatementList stmts_real2c_post;
            stmts_real2c_post += CommentLines{
                "use the last thread of each transform to write one more element per row"};
            stmts_real2c_post
                += If{Equal{thread, threads_per_transform - 1},
                      {StoreGlobal{buf,
                                   offset_pp + (thread + (height - 1) * width + 1) * stride0,
                                   lds_complex[offset_lds + thread + (height - 1) * width + 1]}}};
            if(ebtype == EmbeddedType::Real2C_POST)
            {
                stmts += CommentLines{"append extra global write for Real2C post-process only"};
                stmts += stmts_real2c_post;
            }
        }
        else
            throw std::runtime_error(
                "Direct store from registers not allowed in partial pass SBRR kernels");

        return {If{in_bound, stmts}};
    }

    StatementList real_trans_pre_post() override
    {
        if(ebtype == EmbeddedType::NONE)
            return {};

        std::string   pre_post   = (ebtype == EmbeddedType::C2Real_PRE) ? " before " : " after ";
        auto          twd_offset = (length - factors.front());
        StatementList stmts;
        stmts += CommentLines{"handle even-length real to complex pre-process in lds" + pre_post
                              + "transform"};
        stmts += real2cmplx_pre_post(length, threads_per_transform, twd_offset);
        return stmts;
    }

    ArgumentList global_arguments() override
    {
        auto arguments
            = static_dim
                  ? ArgumentList{twiddles_pp, twiddles_off_dim, twiddles, lengths, stride, nbatch}
                  : ArgumentList{
                      twiddles_pp, twiddles_off_dim, twiddles, dim, lengths, stride, nbatch};
        for(const auto& arg : get_callback_args().arguments)
            arguments.append(arg);
        arguments.append(buf);
        return arguments;
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

        body += set_direct_to_from_registers();

        // half-lds
        body += set_lds_is_real();

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
        // handle even-length real to complex pre-process in lds before transform
        if(ebtype == EmbeddedType::C2Real_PRE)
            loadlds += real_trans_pre_post();

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

        body += LineBreak{};
        body += CommentLines{"calc the thread_in_device value once and for all device funcs"};
        body += Declaration{thread_in_device, thread_id % threads_per_transform};

        // before starting the transform job (core device function)
        // we call a re-load lds-to-reg function here, but it's not always doing things.
        // If we're doing direct-to-reg, this function simply returns.
        body += LineBreak{};
        body += CommentLines{"call a pre-load from lds to registers (if necessary)"};
        auto pre_post_lds_tmpl = device_lds_reg_inout_device_call_templates();
        auto pre_post_lds_args = device_lds_reg_inout_device_call_arguments();
        pre_post_lds_tmpl.set_value(stride_type.name, "SB_UNIT");
        StatementList preLoad;
        preLoad += Call{"lds_to_reg_input_length" + std::to_string(length) + "_device",
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

            templates.set_value(stride_type.name, "SB_UNIT");

            body += Call{"forward_full_pass_length" + std::to_string(length) + "_" + tiling_name()
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

        // Post stores must be in LDS since partial steps 1/2 are always in LDS.
        StatementList postStore;
        postStore += Call{"lds_from_reg_output_length" + std::to_string(length) + "_device",
                          pre_post_lds_tmpl,
                          pre_post_lds_args};
        body += postStore;

        // handle even-length real to complex post-process in lds after full pass
        if(ebtype == EmbeddedType::Real2C_POST)
        {
            body += LineBreak{};
            body += real_trans_pre_post();
        }

        body += generate_partial_pass_steps_1_2();

        body += LineBreak{};
        StatementList storelds;
        storelds += LineBreak{};
        storelds += CommentLines{"store global"};
        storelds += SyncThreads{};

        // Cannot have direct from register stores
        // to global mem since partial steps 1/2
        // are always in LDS
        storelds += store_to_global(false);
        body += storelds;

        f.templates = global_templates();
        f.arguments = global_arguments();
        return f;
    }
};
