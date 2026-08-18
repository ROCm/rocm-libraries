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
#include "stockham_gen_base.h"

// Base class for stockham partial pass kernels.
// Subclasses are responsible for different tiling types.
struct StockhamPartialPassKernel : public StockhamKernel
{
    explicit StockhamPartialPassKernel(const StockhamGeneratorSpecs&    specs,
                                       const StockhamPartialPassParams& params)
        : StockhamKernel(specs)
        , params(params)
    {
        length_pp                = params.parent_length[params.off_dim];
        factors_pp               = params.pp_factors_curr;
        max_factor_pp            = *std::max_element(factors_pp.begin(), factors_pp.end());
        factors_pp_other         = params.pp_factors_other;
        pp_factors_prod          = product(factors_pp.begin(), factors_pp.end());
        pp_factors_other_prod    = product(factors_pp_other.begin(), factors_pp_other.end());
        threads_per_transform_pp = params.pp_threads_per_transform;
        transforms_per_block_pp  = workgroup_size / threads_per_transform_pp;
    }
    virtual ~StockhamPartialPassKernel(){};

    StockhamPartialPassParams params;

    unsigned int              max_factor_pp;
    unsigned int              pp_factors_prod;
    unsigned int              pp_factors_other_prod;
    std::vector<unsigned int> factors_pp_other;

    Variable tile_index{"tile_index", "size_t"};
    Variable num_of_tiles{"num_of_tiles", "size_t"};
    Variable in_bound{"in_bound", "bool"};
    Variable thread{"thread", "unsigned int"}; // replacing tid_ver
    Variable tid_hor{"tid_hor", "unsigned int"}; // id along row
    Variable stride_in{"stride_in", "const size_t", true};
    Variable stride_out{"stride_out", "const size_t", true};

    Variable intrinsic_mode{"intrinsic_mode", "IntrinsicAccessType"};
    Variable apply_large_twiddle{"apply_large_twiddle", "bool"};
    Variable large_twiddle_steps{"large_twiddle_steps", "size_t"};
    Variable large_twiddle_base{"large_twiddle_base", "size_t"};

    Variable large_twiddles{"large_twiddles", "const scalar_type", true};

    Variable stride_lds_pp{"stride_lds_pp", "unsigned int"};
    Variable offset_lds_pp{"offset_lds_pp", "unsigned int"};
    Variable offset_pp{"offset_pp", "unsigned int"};
    Variable thread_pp{"thread_pp", "unsigned int"};
    Variable twiddles_pp{"twiddles_pp", "const scalar_type", true, true};
    Variable twiddles_off_dim{"twiddles_off_dim", "const scalar_type", true, true};
    Variable global_idx{"global_idx", "unsigned int"};
    Variable transpose_idx{"transpose_idx", "unsigned int"};

    ArgumentList device_lds_reg_inout_pp_steps_1_2_arguments()
    {
        ArgumentList args{R, lds_complex, stride_lds, offset_lds, thread};
        return args;
    }

    ArgumentList device_lds_reg_inout_pp_steps_3_4_arguments()
    {
        ArgumentList args{R, lds_complex, stride_lds, offset_lds};
        return args;
    }

    TemplateList device_lds_reg_inout_pp_steps_1_2_templates()
    {
        TemplateList tpls;
        tpls.append(scalar_type);
        return tpls;
    }

    TemplateList device_lds_reg_inout_pp_steps_3_4_templates()
    {
        return device_lds_reg_inout_templates();
    }

    std::vector<Expression> device_lds_reg_inout_pp_steps_1_2_device_call_arguments()
    {
        return {R, lds_complex, stride_lds_pp, offset_lds_pp, thread_in_device_pp};
    }

    std::vector<Expression> device_lds_reg_inout_pp_steps_3_4_device_call_arguments()
    {
        return {R, lds_complex, stride_lds_pp, offset_lds_pp};
    }

    TemplateList device_pp_call_templates()
    {
        return {scalar_type, lds_is_real, lds_linear, direct_load_to_reg};
    }

    TemplateList device_pp_steps_1_2_call_templates()
    {
        return device_pp_call_templates();
    }

    TemplateList device_pp_steps_3_4_call_templates()
    {
        return device_pp_call_templates();
    }

    TemplateList device_pp_templates()
    {
        TemplateList tpls;
        tpls.append(scalar_type);
        tpls.append(lds_is_real);
        tpls.append(lds_linear);
        tpls.append(direct_load_to_reg);
        return tpls;
    }

    TemplateList device_pp_steps_1_2_templates()
    {
        return device_pp_templates();
    }

    TemplateList device_pp_steps_3_4_templates()
    {
        return device_pp_templates();
    }

    ArgumentList device_pp_steps_1_2_arguments()
    {
        ArgumentList args{R,
                          lds_real,
                          lds_complex,
                          twiddles_pp,
                          twiddles,
                          stride_lds,
                          offset_lds,
                          thread,
                          thread_pp,
                          write};
        return args;
    }

    ArgumentList device_pp_steps_3_4_arguments()
    {
        ArgumentList args{R, lds_real, lds_complex, stride_lds, offset_lds, write};
        return args;
    }

    std::vector<Expression> device_pp_steps_1_2_call_arguments(unsigned int call_iter)
    {
        return {R,
                lds_real,
                lds_complex,
                twiddles_pp,
                twiddles_off_dim,
                stride_lds_pp,
                call_iter ? Expression{offset_lds_pp
                                       + call_iter * stride_lds_pp * transforms_per_block_pp}
                          : Expression{offset_lds_pp},
                thread_in_device_pp,
                thread_in_device_pp_twiddles,
                Literal{"true"}};
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

    StatementList load_pp_steps_1_2_lds_generator(
        unsigned int h, unsigned int hr, unsigned int width, unsigned int dt, Expression guard)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
        {
            const auto tid = Parens{thread + dt + h * threads_per_transform_pp};
            work += Assign(
                R[hr * width + w],
                lds_complex[offset_lds + (tid + w * pp_factors_prod / width) * stride_lds]);
        }

        return work;
    }

    StatementList load_pp_steps_3_4_lds_generator(
        unsigned int h, unsigned int hr, unsigned int width, unsigned int dt, Expression guard)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
            work += Assign(R[hr * width + w],
                           lds_complex[offset_lds + (hr * width + w) * stride_lds]);

        return work;
    }

    StatementList store_pp_steps_1_2_lds_generator(unsigned int h,
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
            const auto tid = thread + dt + h * threads_per_transform_pp;
            const auto idx = offset_lds
                             + (Parens{tid / cumheight} * (width * cumheight) + tid % cumheight
                                + w * cumheight)
                                   * stride_lds;

            work += Assign(lds_complex[idx], R[hr * width + w]);
        }

        return work;
    }

    StatementList store_pp_steps_3_4_lds_generator(
        unsigned int h, unsigned int hr, unsigned int width, unsigned int dt, Expression guard)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        for(unsigned int w = 0; w < width; ++w)
            work += Assign(lds_complex[offset_lds + (hr * width + w) * stride_lds],
                           R[hr * width + w]);

        return work;
    }

    std::vector<Expression> device_pp_steps_3_4_call_arguments(unsigned int call_iter)
    {
        return {R,
                lds_real,
                lds_complex,
                stride_lds_pp,
                call_iter ? Expression{offset_lds_pp
                                       + call_iter * stride_lds_pp * transforms_per_block_pp}
                          : Expression{offset_lds_pp},
                Literal{"true"}};
    }

    Function generate_lds_to_reg_partial_pass_steps_1_2_input_function()
    {
        std::string function_name = "lds_to_reg_steps_1_2_input_partial_pass_length"
                                    + std::to_string(pp_factors_prod) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_pp_steps_1_2_templates();
        f.arguments = device_lds_reg_inout_pp_steps_1_2_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;

        auto load_lds = std::mem_fn(&StockhamPartialPassKernel::load_pp_steps_1_2_lds_generator);
        // first pass of load (full)
        unsigned int width = factors_pp[0];
        float height       = static_cast<float>(pp_factors_prod) / width / threads_per_transform_pp;
        body += SyncThreads();
        body += add_pp_work(std::bind(load_lds, this, _1, _2, _3, _4, _5),
                            width,
                            height,
                            ThreadGuardMode::GUARD_BY_IF,
                            false);

        return f;
    }

    Function generate_lds_to_reg_partial_pass_steps_3_4_input_function()
    {
        std::string function_name
            = "lds_to_reg_steps_3_4_input_partial_pass_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_pp_steps_3_4_templates();
        f.arguments = device_lds_reg_inout_pp_steps_3_4_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;

        auto load_lds = std::mem_fn(&StockhamPartialPassKernel::load_pp_steps_3_4_lds_generator);
        // first pass of load (partial-pass)
        unsigned int width  = factors_pp[0];
        float        height = static_cast<float>(length) / width / threads_per_transform;
        body += SyncThreads();
        body += add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5),
                         width,
                         height,
                         ThreadGuardMode::NO_GUARD);

        return f;
    }

    Function generate_lds_from_reg_partial_pass_steps_1_2_output_function()
    {
        std::string function_name = "lds_from_reg_steps_1_2_output_partial_pass_length"
                                    + std::to_string(pp_factors_prod) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_pp_steps_1_2_templates();
        f.arguments = device_lds_reg_inout_pp_steps_1_2_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;

        auto store_lds = std::mem_fn(&StockhamPartialPassKernel::store_pp_steps_1_2_lds_generator);
        // last pass of store (full)
        unsigned int width = factors_pp.back();
        float height       = static_cast<float>(pp_factors_prod) / width / threads_per_transform_pp;
        unsigned int cumheight = product(factors_pp.begin(), factors_pp.end() - 1);
        body += SyncThreads();
        body += add_pp_work(std::bind(store_lds, this, _1, _2, _3, _4, _5, cumheight),
                            width,
                            height,
                            ThreadGuardMode::GUARD_BY_IF,
                            false);
        return f;
    }

    Function generate_lds_from_reg_partial_pass_steps_3_4_output_function()
    {
        std::string function_name = "lds_from_reg_steps_3_4_output_partial_pass_length"
                                    + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_pp_steps_3_4_templates();
        f.arguments = device_lds_reg_inout_pp_steps_3_4_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};

        auto store_lds = std::mem_fn(&StockhamPartialPassKernel::store_pp_steps_3_4_lds_generator);
        // last pass of store (partial-pass)
        unsigned int width  = factors_pp.back();
        float        height = static_cast<float>(length) / width / threads_per_transform;
        body += SyncThreads();
        body += add_work(std::bind(store_lds, this, _1, _2, _3, _4, _5),
                         width,
                         height,
                         ThreadGuardMode::NO_GUARD);
        return f;
    }

    // The "stacked" twiddle table starts at the second factor, since
    // the first factor's values are not actually needed for
    // anything.  It still counts towards cumulative height, but we
    // subtract it from the twiddle table offset when computing an
    // index.
    StatementList apply_twiddle_off_dim_generator(unsigned int h,
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
        Expression    loadFlag{thread < pp_factors_prod / width};
        for(unsigned int w = 1; w < width; ++w)
        {
            auto tid  = thread + dt + h * threads_per_transform_pp;
            auto tidx = cumheight - firstFactor + w - 1 + (width - 1) * (tid % cumheight);
            auto ridx = hr * width + w;

            // TODO- Can try IntrinsicLoadToDest, but should not be a bottleneck
            work += Assign(W, twiddles[tidx]);
            work += Assign(t, TwiddleMultiply(R[ridx], W));
            work += Assign(R[ridx], t);
        }
        return work;
    }

    StatementList apply_twiddle_pp_generator(unsigned int h,
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
        for(unsigned int w = 0; w < width; ++w)
        {
            auto tid  = thread + dt + h * threads_per_transform_pp;
            auto tidx = thread_pp * Literal(length_pp)
                        + (Parens{tid / cumheight} * (width * cumheight) + tid % cumheight
                           + w * cumheight);
            auto ridx = hr * width + w;

            work += Assign(W, twiddles_pp[tidx]);
            work += Assign(t, TwiddleMultiply(R[ridx], W));
            work += Assign(R[ridx], t);
        }
        return work;
    }

    Function generate_pp_steps_1_2_device_function()
    {
        std::string function_name = "forward_partial_pass_steps_1_2_length"
                                    + std::to_string(pp_factors_prod) + "_" + tiling_name()
                                    + "_device";

        Function f{function_name};
        f.arguments = device_pp_steps_1_2_arguments();
        f.templates = device_pp_steps_1_2_templates();
        f.qualifier = "__device__";
        if(pp_factors_prod == 1)
            return f;

        unsigned int cumheight = 0;
        unsigned int width     = 0;
        float        height    = 0.0f;

        StatementList& body = f.body;
        body += Declaration{W};
        body += Declaration{t};

        for(unsigned int npass = 0; npass < factors_pp.size(); ++npass)
        {
            // width is the butterfly width, Radix-n.
            width = factors_pp[npass];
            // height is how many butterflies per thread will do on average
            height = static_cast<float>(pp_factors_prod) / width / threads_per_transform_pp;

            cumheight = product(factors_pp.begin(),
                                factors_pp.begin()
                                    + npass); // cumheight is irrelevant to the above height,
            // is used for twiddle multiplication and lds writing.

            body += LineBreak{};
            body += CommentLines{
                "pass " + std::to_string(npass) + ", width " + std::to_string(width),
                "using " + std::to_string(threads_per_transform_pp) + " threads we need to do "
                    + std::to_string(pp_factors_prod / width) + " radix-" + std::to_string(width)
                    + " butterflies",
                "therefore each thread will do " + std::to_string(height) + " butterflies"};

            auto load_lds
                = std::mem_fn(&StockhamPartialPassKernel::load_pp_steps_1_2_lds_generator);
            auto store_lds
                = std::mem_fn(&StockhamPartialPassKernel::store_pp_steps_1_2_lds_generator);

            if(npass > 0)
            {
                // internal full lds2reg (both linear/nonlinear variants)
                StatementList lds2reg_full;
                lds2reg_full += SyncThreads();
                lds2reg_full += add_pp_work(std::bind(load_lds, this, _1, _2, _3, _4, _5),
                                            width,
                                            height,
                                            ThreadGuardMode::GUARD_BY_IF,
                                            true);
                body += If{Not{lds_is_real}, lds2reg_full};

                auto apply_twiddle
                    = std::mem_fn(&StockhamPartialPassKernel::apply_twiddle_off_dim_generator);
                body += add_work(
                    std::bind(
                        apply_twiddle, this, _1, _2, _3, _4, _5, cumheight, factors_pp.front()),
                    width,
                    height,
                    ThreadGuardMode::NO_GUARD);
            }

            auto butterfly = std::mem_fn(&StockhamKernel::butterfly_generator);
            body += add_work(std::bind(butterfly, this, _1, _2, _3, _4, _5),
                             width,
                             height,
                             ThreadGuardMode::NO_GUARD);

            if(npass == factors_pp.size() - 1)
                body += large_twiddles_multiply(width, height, cumheight);

            // internal lds store
            StatementList reg2lds_full;
            if(npass < factors_pp.size() - 1)
            {
                // internal full lds store (both linear/nonlinear variants)
                if(npass == 0)
                    reg2lds_full += If{!direct_load_to_reg, {SyncThreads()}};
                else
                    reg2lds_full += SyncThreads();
                reg2lds_full
                    += add_pp_work(std::bind(store_lds, this, _1, _2, _3, _4, _5, cumheight),
                                   width,
                                   height,
                                   ThreadGuardMode::GUARD_BY_IF,
                                   false);

                body += reg2lds_full;
            }
        }

        body += LineBreak{};
        body += CommentLines{"extra twiddle multiplication step for partial transform"};
        auto apply_twiddle_pp = std::mem_fn(&StockhamPartialPassKernel::apply_twiddle_pp_generator);
        body += add_work(
            std::bind(apply_twiddle_pp, this, _1, _2, _3, _4, _5, cumheight, factors_pp.front()),
            width,
            height,
            ThreadGuardMode::NO_GUARD);

        return f;
    }

    Function generate_pp_steps_3_4_device_function()
    {
        std::string function_name = "forward_partial_pass_steps_3_4_length"
                                    + std::to_string(pp_factors_prod) + "_" + tiling_name()
                                    + "_device";

        Function f{function_name};
        f.arguments = device_pp_steps_3_4_arguments();
        f.templates = device_pp_steps_3_4_templates();
        f.qualifier = "__device__";
        if(pp_factors_prod == 1)
            return f;

        StatementList& body = f.body;

        for(unsigned int npass = 0; npass < factors_pp.size(); ++npass)
        {
            unsigned int pass_width = factors_pp[npass];
            float pass_height = static_cast<float>(length) / pass_width / threads_per_transform;

            auto butterfly = std::mem_fn(&StockhamKernel::butterfly_generator);
            body += add_work(std::bind(butterfly, this, _1, _2, _3, _4, _5),
                             pass_width,
                             pass_height,
                             ThreadGuardMode::NO_GUARD);
        }

        return f;
    }

    TemplateList device_lds_reg_inout_pp_steps_1_2_device_call_templates()
    {
        return {scalar_type};
    }

    TemplateList device_lds_reg_inout_pp_steps_3_4_device_call_templates(bool syncthreads = true)
    {
        Variable sync_var{syncthreads ? "true" : "false", "bool"};
        return {scalar_type, stride_type, sync_var};
    }

    StatementList generate_partial_pass_steps_1_2()
    {
        StatementList stmts;

        stmts += LineBreak{};
        stmts += CommentLines{
            "calc the thread_in_device value once and for all partial-pass device funcs"};
        stmts += Declaration{thread_in_device_pp, thread_id % threads_per_transform_pp};
        stmts
            += Declaration{thread_in_device_pp_twiddles, block_id % (length_pp / pp_factors_prod)};

        stmts += LineBreak{};
        stmts += CommentLines{"partial-pass offsets"};
        stmts += Declaration{stride_lds_pp, (length + get_lds_padding())};
        stmts += Declaration{offset_lds_pp,
                             Parens(block_id * transforms_per_block + thread_id)
                                 % (length + get_lds_padding())};

        auto pre_post_lds_tmpl = device_lds_reg_inout_pp_steps_1_2_device_call_templates();
        auto pre_post_lds_args = device_lds_reg_inout_pp_steps_1_2_device_call_arguments();

        StatementList preLoad;
        stmts += LineBreak{};
        stmts += CommentLines{"call a pre-load from lds to registers"};
        preLoad += Call{"lds_to_reg_steps_1_2_input_partial_pass_length"
                            + std::to_string(pp_factors_prod) + "_device",
                        pre_post_lds_tmpl,
                        pre_post_lds_args};
        stmts += preLoad;

        auto device_tmpl = device_pp_steps_1_2_call_templates();
        auto device_args = device_pp_steps_1_2_call_arguments(0);

        StatementList device;
        stmts += LineBreak{};
        stmts += CommentLines{"partial transform in off-dimension"};
        device += Call{"forward_partial_pass_steps_1_2_length" + std::to_string(pp_factors_prod)
                           + "_" + tiling_name() + "_device",
                       device_tmpl,
                       device_args};
        device += LineBreak{};
        stmts += device;

        StatementList postStore;
        stmts += LineBreak{};
        stmts += CommentLines{"call a post-store from registers to lds"};
        postStore += Call{"lds_from_reg_steps_1_2_output_partial_pass_length"
                              + std::to_string(pp_factors_prod) + "_device",
                          pre_post_lds_tmpl,
                          pre_post_lds_args};
        stmts += postStore;

        return stmts;
    }

    StatementList generate_partial_pass_steps_3_4()
    {
        StatementList stmts;

        unsigned int width  = factors_pp[0];
        unsigned int height = length / width / threads_per_transform;

        stmts += LineBreak{};
        stmts += CommentLines{"partial-pass offsets"};
        stmts += Declaration{stride_lds_pp, Literal{1}};
        stmts += Declaration{offset_lds_pp, thread_id * Literal{width * height}};

        auto pre_post_lds_tmpl = device_lds_reg_inout_pp_steps_3_4_device_call_templates();
        auto pre_post_lds_args = device_lds_reg_inout_pp_steps_3_4_device_call_arguments();
        pre_post_lds_tmpl.set_value(stride_type.name, "lds_linear ? SB_UNIT : SB_NONUNIT");

        StatementList preLoad;
        stmts += LineBreak{};
        stmts += CommentLines{"call a pre-load from lds to registers"};
        preLoad += Call{"lds_to_reg_steps_3_4_input_partial_pass_length" + std::to_string(length)
                            + "_device",
                        pre_post_lds_tmpl,
                        pre_post_lds_args};
        stmts += preLoad;

        auto device_tmpl = device_pp_steps_3_4_call_templates();
        auto device_args = device_pp_steps_3_4_call_arguments(0);

        StatementList device;
        stmts += LineBreak{};
        stmts += CommentLines{"partial transform in off-dimension"};
        device += Call{"forward_partial_pass_steps_3_4_length" + std::to_string(pp_factors_prod)
                           + "_" + tiling_name() + "_device",
                       device_tmpl,
                       device_args};
        device += LineBreak{};
        stmts += device;

        width  = factors_pp.back();
        height = length / width / threads_per_transform;
        stmts += Assign{offset_lds_pp, thread_id * Literal{width * height}};

        StatementList postStore;
        stmts += LineBreak{};
        stmts += CommentLines{"call a post-store from registers to lds"};
        postStore += Call{"lds_from_reg_steps_3_4_output_partial_pass_length"
                              + std::to_string(length) + "_device",
                          pre_post_lds_tmpl,
                          pre_post_lds_args};
        stmts += postStore;

        return stmts;
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

        auto len_1 = params.parent_length[2];
        auto len_2 = params.parent_length[1];
        auto len_3 = params.parent_length[0];

        auto len_1_2_3 = len_1 * len_2 * len_3;
        auto len_1_2   = len_1 * len_2;

        auto len_pp_factors_prod       = pp_factors_prod * len_2;
        auto len_pp_factors_other_prod = pp_factors_other_prod * len_2;

        body += Declaration{transpose_idx, global_idx % len_1_2_3};

        body += Assign{
            transpose_idx,
            Parens{transpose_idx % len_2}
                + Parens{Parens{Parens{transpose_idx % (len_pp_factors_prod)} / len_2}
                         * len_pp_factors_other_prod}
                + Parens{Parens{Parens{transpose_idx % len_1_2} / len_pp_factors_prod} * len_2}
                + Parens{Parens{transpose_idx / len_1_2} * len_1_2}};

        body += Assign{transpose_idx, transpose_idx + Parens{global_idx / len_1_2_3} * len_1_2_3};

        body += ReturnExpr(transpose_idx);

        return f;
    }
};