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
#include "stockham_gen_rr.h"

// TODO: Once partial pass is fully configurable in kernel-generator.py:
//       - Revisit all usages of transform_per_block and max_factor_pp.
//       - Test with factors_pp.size() > 1
//       - Revisit lstride usage and input/output strides

// Variation of StockhamKernelRR that implements the partial pass
// method. Similarities of StockhamPartialPassKernelRR with
// StockhamKernelRR include the overal kernel structure with the
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
struct StockhamPartialPassKernelRR : public StockhamKernelRR
{
    explicit StockhamPartialPassKernelRR(const StockhamGeneratorSpecs&    specs,
                                         const StockhamPartialPassParams& params)
        : StockhamKernelRR(specs)
        , params(params)
    {
        length_pp  = params.parent_length[params.off_dim];
        factors_pp = params.pp_factors_curr;

        max_factor_pp = *std::max_element(factors_pp.begin(), factors_pp.end());

        length_off_dim = params.parent_length[params.off_dim];

        R.size = Expression{std::max(nregisters, max_factor_pp)};

        // nregister must not be larger than max_factor_pp.
        // If that were to be true,  work in the off-dimension
        // in perform_partial_pass_step_1_2() would require to
        // to be applied to (nregisters-max_factor_pp) elements
        // in the off-dimension, but this data is not available
        // in the LDS, and the number of additional elements
        // would need to be at least a multiple of max_factor_pp.
        if(nregisters > max_factor_pp)
            throw std::runtime_error(
                "StockhamPartialPassKernelRR: nregisters cannot be larger than max_factor_pp");
    }

    StockhamPartialPassParams params;

    unsigned int              length_off_dim;
    unsigned int              max_factor_pp;
    std::vector<unsigned int> factors_pp;
    unsigned int              length_pp;

    Variable offset_pp{"offset_pp", "size_t"};
    Variable stride_lds_pp{"stride_lds_pp", "size_t"};
    Variable offset_lds_pp{"offset_lds_pp", "size_t"};
    Variable twiddles_pp{"twiddles_pp", "const scalar_type", true, true};

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

        stmts += Declaration{inbound, batch < nbatch};

        return stmts;
    }

    std::vector<unsigned int> launcher_lengths() override
    {
        return params.parent_length;
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
            stmts += LineBreak();
            stmts += CommentLines{"append extra global loading for C2Real pre-process only"};

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

        return {If{inbound, stmts}};
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
            stmts += CommentLines{"append extra global write for Real2C post-process only"};
            StatementList stmts_real2c_post;
            stmts_real2c_post += CommentLines{
                "use the last thread of each transform to write one more element per row"};
            stmts_real2c_post
                += If{Equal{thread, threads_per_transform - 1},
                      {StoreGlobal{buf,
                                   offset + (thread + (height - 1) * width + 1) * stride0,
                                   lds_complex[offset_lds + thread + (height - 1) * width + 1]}}};
            if(ebtype == EmbeddedType::Real2C_POST)
                stmts += stmts_real2c_post;
        }
        else
            throw std::runtime_error(
                "Direct store from registers not allowed in partial pass SBRR kernels");

        return {If{inbound, stmts}};
    }

    StatementList load_lds_step_1_2_generator(
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

    ArgumentList device_lds_reg_inout_pp_arguments()
    {
        ArgumentList args{R, lds_complex, stride_lds, offset_lds, thread};
        return args;
    }

    std::vector<Expression> device_lds_reg_inout_pp_device_call_arguments()
    {
        return {R, lds_complex, stride_lds_pp, offset_lds_pp, thread_id / max_factor_pp};
    }

    TemplateList device_lds_reg_inout_pp_templates()
    {
        TemplateList tpls;
        tpls.append(scalar_type);
        return tpls;
    }

    Function generate_lds_to_reg_input_step_1_2_function()
    {
        std::string function_name
            = "lds_to_reg_input_pp_step_1_2_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_pp_templates();
        f.arguments = device_lds_reg_inout_pp_arguments();
        f.qualifier = "__device__";

        auto effective_length = std::max(length, length_pp);

        StatementList& body = f.body;

        auto load_lds = std::mem_fn(&StockhamPartialPassKernelRR::load_lds_step_1_2_generator);
        // first pass of load (full)
        unsigned int width  = factors_pp[0];
        float        height = static_cast<float>(effective_length) / width / threads_per_transform;
        body += SyncThreads();
        body += add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5),
                         width,
                         height,
                         ThreadGuardMode::GUARD_BY_IF,
                         false,
                         std::nullopt,
                         effective_length);

        return f;
    }

    StatementList store_pp_step_1_2_lds_generator(
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

    Function generate_lds_from_reg_output_pp_step_1_2_function()
    {
        std::string function_name
            = "lds_from_reg_output_pp_step_1_2_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_pp_templates();
        f.arguments = device_lds_reg_inout_pp_arguments();
        f.qualifier = "__device__";

        auto effective_length = std::max(length_pp, length);

        StatementList& body = f.body;

        auto store_lds = std::mem_fn(&StockhamPartialPassKernelRR::store_pp_step_1_2_lds_generator);
        // last pass of store (full)
        unsigned int width  = factors_pp.back();
        float        height = static_cast<float>(effective_length) / width / threads_per_transform;
        body += SyncThreads();
        body += add_work(std::bind(store_lds, this, _1, _2, _3, _4, _5),
                         width,
                         height,
                         ThreadGuardMode::GUARD_BY_IF,
                         false,
                         std::nullopt,
                         effective_length);
        return f;
    }

    Function generate_twiddle_multiply_pp_function(int direction)
    {
        std::string function_name
            = "twiddle_multiply_pp_length" + std::to_string(length_pp) + "_device";

        Function f{function_name};

        TemplateList tpls = {scalar_type};
        f.templates       = tpls;

        f.arguments = ArgumentList{R, thread, twiddles_pp};

        f.return_type = "void";
        f.qualifier   = "__device__";

        StatementList& body = f.body;

        body += Declaration{t};
        body += Declaration{W};

        for(unsigned int w = 0; w < max_factor_pp; ++w)
        {
            body += Assign{W, twiddles_pp[thread * Literal(length_pp) + w]};

            if(direction == -1)
                body += Assign{t, TwiddleMultiply{R[w], W}};
            else if(direction == 1)
                body += Assign{t, TwiddleMultiplyConjugate{R[w], W}};
            else
                throw std::runtime_error("Invalid FFT direction for twiddle multiply");

            body += Assign{R[w], t};
        }

        return f;
    }

    TemplateList device_lds_reg_inout_pp_step_1_2_device_call_templates()
    {
        return {scalar_type};
    }

    StatementList perform_partial_pass_step_1_2()
    {
        StatementList stmts;

        stmts += Declaration{stride_lds_pp, length};
        stmts += Declaration{offset_lds_pp,
                             Parens(block_id * transforms_per_block + thread_id) % length};

        auto pre_post_lds_tmpl = device_lds_reg_inout_pp_step_1_2_device_call_templates();
        auto pre_post_lds_args = device_lds_reg_inout_pp_device_call_arguments();

        StatementList preLoad;
        preLoad += Call{"lds_to_reg_input_pp_step_1_2_length" + std::to_string(length) + "_device",
                        pre_post_lds_tmpl,
                        pre_post_lds_args};
        stmts += preLoad;

        for(unsigned int npass = 0; npass < factors_pp.size(); ++npass)
        {
            unsigned int width  = factors_pp[npass];
            unsigned int height = transforms_per_block / max_factor_pp;

            auto butterfly = std::mem_fn(&StockhamKernel::butterfly_generator);
            stmts += add_work(std::bind(butterfly, this, _1, _2, _3, _4, _5),
                              width,
                              height,
                              ThreadGuardMode::NO_GUARD);
        }

        TemplateList            pre_twd_mul_tmpl = TemplateList{scalar_type};
        std::vector<Expression> pre_twd_mul_args
            = {R, block_id % (length_pp / max_factor_pp), twiddles_pp};
        StatementList twdMul;
        twdMul += Call{"twiddle_multiply_pp_length" + std::to_string(length_pp) + "_device",
                       pre_twd_mul_tmpl,
                       pre_twd_mul_args};

        stmts += twdMul;

        StatementList postStore;
        postStore
            += Call{"lds_from_reg_output_pp_step_1_2_length" + std::to_string(length) + "_device",
                    pre_post_lds_tmpl,
                    pre_post_lds_args};
        stmts += postStore;

        return stmts;

        return stmts;
    }

    ArgumentList global_arguments() override
    {
        auto arguments = static_dim
                             ? ArgumentList{twiddles_pp, twiddles, lengths, stride, nbatch}
                             : ArgumentList{twiddles_pp, twiddles, dim, lengths, stride, nbatch};
        for(const auto& arg : get_callback_args().arguments)
            arguments.append(arg);
        arguments.append(buf);
        return arguments;
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
        // To make this option work, step_1_2 here
        // would need to implement half LDS usage in
        // the off-direction pass.
        return {Declaration{lds_is_real, Literal{"false"}}};
    }

    Function generate_global_function() override
    {
        Function f("forward_length" + std::to_string(length) + "_" + tiling_name());
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

            body
                += Call{"forward_length" + std::to_string(length) + "_" + tiling_name() + "_device",
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

        body += perform_partial_pass_step_1_2();

        body += LineBreak{};
        StatementList storelds;
        storelds += LineBreak{};
        // handle even-length complex to real post-process in lds after transform
        if(ebtype == EmbeddedType::Real2C_POST)
            storelds += real_trans_pre_post();
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
