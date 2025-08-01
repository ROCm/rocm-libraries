// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "stockham_gen.h"
#include <cmath>

// Base class for stockham kernels.  Subclasses are responsible for
// different tiling types.
//
// This inherits from specs as a shortcut to avoid having to look
// inside a class member for variables we need all the time
struct StockhamKernel : public StockhamGeneratorSpecs
{
    StockhamKernel(const StockhamGeneratorSpecs& specs)
        : StockhamGeneratorSpecs(specs)
    {
        // RTC-ing kernels for tuning always goes this way
        if(wgs_is_derived)
        {
            transforms_per_block = workgroup_size / threads_per_transform;
        }
        else
        {
            auto bytes_per_batch = length * bytes_per_element;

            if(half_lds)
                bytes_per_batch /= 2;

            if(threads_per_transform == 0)
            {
                threads_per_transform = 1;
                for(unsigned int t = 2; t < length; ++t)
                {
                    if(t > workgroup_size)
                        continue;
                    if(length % t == 0)
                    {
                        if(std::all_of(factors.begin(), factors.end(), [=](unsigned int f) {
                               return (length / t) % f == 0;
                           }))
                            threads_per_transform = t;
                    }
                }
            }

            transforms_per_block = lds_byte_limit / bytes_per_batch;
            while(threads_per_transform * transforms_per_block > workgroup_size)
                --transforms_per_block;
            if(!factors2d.empty())
                transforms_per_block = std::min(transforms_per_block, length2d);

            workgroup_size = threads_per_transform * transforms_per_block;
        }

        nregisters                = compute_nregisters(length, factors, threads_per_transform);
        R.size                    = Expression{nregisters};
        lds_reg_sync.decl_default = Literal{"true"};
    }
    virtual ~StockhamKernel(){};

    unsigned int nregisters;
    unsigned int transforms_per_block;

    // data that may be overridden by subclasses (different tiling types)
    unsigned int n_device_calls = 1;
    bool         writeGuard     = false;

    static unsigned int compute_nregisters(unsigned int                     length,
                                           const std::vector<unsigned int>& factors,
                                           unsigned int                     threads_per_transform)
    {
        unsigned int max_registers = 0;
        for(auto width : factors)
        {
            unsigned int n = std::ceil(double(length) / width / threads_per_transform) * width;
            if(n > max_registers)
                max_registers = n;
        }
        return max_registers;
    }

    //
    // templates
    //
    Variable scalar_type{"scalar_type", "typename"};
    Variable callback_type{"cbtype", "CallbackType"};
    Variable stride_type{"sb", "StrideBin"};
    Variable directReg_type{"drtype", "DirectRegType"};

    //
    // arguments
    //
    // global input/ouput buffer
    Variable buf{"buf", "scalar_type", true, true};

    // global twiddle table (stacked)
    Variable twiddles{"twiddles", "const scalar_type", true, true};

    // rank/dimension of transform
    Variable dim{"dim", "const size_t"};

    // transform lengths
    Variable lengths{"lengths", "const size_t", true, true};

    // input/output array strides
    Variable stride{"stride", "const size_t", true, true};

    // number of transforms/batches
    Variable nbatch{"nbatch", "const size_t"};

    // should the device function write to lds?
    // only used for 2D
    Variable write{"write", "bool"};

    // is LDS real-only?
    Variable lds_is_real = Variable{"lds_is_real", "const bool"};

    // The LDS access pattern is linear both on R/W
    Variable lds_linear = Variable{"lds_linear", "const bool"};

    // Enable directly load to registers and directly store from registers?
    // This variable affects the lds_linear (internally)
    Variable direct_load_to_reg    = Variable{"direct_load_to_reg", "const bool"};
    Variable direct_store_from_reg = Variable{"direct_store_from_reg", "const bool"};

    //
    // locals
    //
    // lds storage buffer
    Variable lds_real{"lds_real", "real_type_t<scalar_type>", true, true};
    Variable lds_complex{"lds_complex", "scalar_type", true, true};
    Variable lds_row_padding{"lds_row_padding", "unsigned int"};

    // hip thread grid dim
    Variable grid_dim{"gridDim.x", "unsigned int"};

    // hip thread block id
    Variable block_id{"blockIdx.x", "unsigned int"};

    // hip thread id
    Variable thread_id{"threadIdx.x", "unsigned int"};

    // thread within transform
    // Variable thread{"thread", "size_t"};
    Variable thread{"thread", "unsigned int"};

    // The "pre-cal" thread that we're passing into device function,
    // Since it is calculated either mod or div (depends on linear/nonlinear)
    // So we'd like to do that expensive mod or div once and for all
    // Variable thread_in_device{"thread_in_device", "size_t"};
    Variable thread_in_device{"thread_in_device", "unsigned int"};

    // global input/output buffer offset to current transform
    Variable offset{"offset", "size_t"};

    // lds buffer offset to current transform
    Variable offset_lds{"offset_lds", "unsigned int"};

    // current batch
    Variable batch{"batch", "size_t"};

    // current transform index in a batch
    Variable transform{"transform", "size_t"};

    // data index and offsets (for contiguous read/write)
    Variable global_data_id{"global_data_id", "size_t"};
    Variable global_load_data_offset{"global_load_data_offset", "size_t"};
    Variable global_store_data_offset{"global_store_data_offset", "size_t"};

    // transform index and offsets
    Variable global_transf_id{"global_transf_id", "size_t"};
    Variable global_load_transf_offset{"global_load_transf_offset", "size_t"};
    Variable global_store_transf_offset{"global_store_transf_offset", "size_t"};

    // stride between consecutive indexes
    Variable stride0{"stride0", "const size_t"};

    // stride between consecutive indexes in lds
    // Variable stride_lds{"stride_lds", "size_t"};
    Variable stride_lds{"stride_lds", "unsigned int"};

    // usually in device: const size_t lstride = (sb == SB_UNIT) ? 1 : stride_lds;
    // with this definition, the compiler knows that "index * lstride" is trivial under SB_UNIT
    // Variable lstride{"lstride", "const size_t"};
    Variable lstride{"lstride", "const unsigned int"};

    // local temp variable in device function
    Variable l_offset{"l_offset", "unsigned int"};

    // twiddle value during twiddle application
    Variable W{"W", "scalar_type"};

    // temporary register during twiddle application
    Variable t{"t", "scalar_type"};

    // butterfly registers
    Variable R{"R", "scalar_type", false, false};

    // do syncthreads in lds_reg device functions
    Variable lds_reg_sync{"lds_reg_sync", "bool"};

    virtual unsigned int launcher_workgroup_size()
    {
        return workgroup_size;
    }

    virtual unsigned int launcher_transforms_per_block()
    {
        return transforms_per_block;
    }

    virtual std::vector<unsigned int> launcher_lengths()
    {
        return {length};
    }
    virtual std::vector<unsigned int> launcher_factors()
    {
        return factors;
    }

    virtual TemplateList device_lds_reg_inout_templates()
    {
        TemplateList tpls;
        tpls.append(scalar_type);
        tpls.append(stride_type);
        tpls.append(lds_reg_sync);
        return tpls;
    }

    virtual TemplateList device_templates()
    {
        TemplateList tpls;
        tpls.append(scalar_type);
        tpls.append(lds_is_real);
        tpls.append(stride_type);
        tpls.append(lds_linear);
        tpls.append(direct_load_to_reg);
        return tpls;
    }

    virtual TemplateList global_templates()
    {
        return {scalar_type, stride_type, callback_type, directReg_type};
    }

    virtual ArgumentList device_lds_reg_inout_arguments()
    {
        ArgumentList args{R, lds_complex, stride_lds, offset_lds, thread, write};
        return args;
    }

    virtual ArgumentList device_arguments()
    {
        ArgumentList args{
            R, lds_real, lds_complex, twiddles, stride_lds, offset_lds, thread, write};
        return args;
    }

    virtual ArgumentList global_arguments()
    {
        auto arguments = static_dim ? ArgumentList{twiddles, lengths, stride, nbatch}
                                    : ArgumentList{twiddles, dim, lengths, stride, nbatch};
        for(const auto& arg : get_callback_args().arguments)
            arguments.append(arg);
        arguments.append(buf);
        return arguments;
    }

    // virtual functions implemented by different tiling implementations
    virtual std::string tiling_name() = 0;

    // TODO- support embedded Pre/Post
    virtual StatementList set_direct_to_from_registers()
    {
        // by default (RR): "direct-to-reg" and "direct-from-reg" at the same time
        if(direct_to_from_reg)
            return {Declaration{
                        direct_load_to_reg,
                        ebtype == EmbeddedType::NONE
                            ? Expression{directReg_type == "DirectRegType::TRY_ENABLE_IF_SUPPORT"}
                            : Expression{Literal{"false"}}},
                    Declaration{direct_store_from_reg, direct_load_to_reg},
                    Declaration{lds_linear, Literal{"true"}}};
        else
            return {Declaration{direct_load_to_reg, Literal{"false"}},
                    Declaration{direct_store_from_reg, Literal{"false"}},
                    Declaration{lds_linear, Literal{"true"}}};
    }

    virtual StatementList set_lds_is_real()
    {
        if(half_lds)
            return {Declaration{lds_is_real,
                                ebtype == EmbeddedType::NONE ? Literal{"true"} : Literal{"false"}}};
        else
            return {Declaration{lds_is_real, Literal{"false"}}};
    }

    virtual StatementList large_twiddles_load()
    {
        return {CommentLines{"- no large twiddles"}};
    }

    virtual StatementList
        large_twiddles_multiply(unsigned int width, double height, unsigned int cumheight)
    {
        return {};
    }

    enum class ThreadGuardMode
    {
        NO_GUARD,
        GUARD_BY_IF,
        GURAD_BY_FUNC_ARG,
    };

    virtual StatementList real_trans_pre_post()
    {
        return {};
    }

    // add one extra element per row for embedded real-complex
    // processing
    virtual Expression get_lds_padding()
    {
        return ebtype == EmbeddedType::NONE ? Literal{0} : Literal{1};
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
            const auto idx = offset_lds + (tid + w * length / width) * lstride;
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
            const auto idx = offset_lds
                             + (Parens{tid / cumheight} * (width * cumheight) + tid % cumheight
                                + w * cumheight)
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
            auto tid  = thread + dt + h * threads_per_transform;
            auto tidx = cumheight - firstFactor + w - 1 + (width - 1) * (tid % cumheight);
            auto ridx = hr * width + w;

            // TODO- Can try IntrinsicLoadToDest, but should not be a bottleneck
            work += Assign(W, twiddles[tidx]);
            work += Assign(t, TwiddleMultiply(R[ridx], W));
            work += Assign(R[ridx], t);
        }
        return work;
    }

    StatementList butterfly_generator(
        unsigned int h, unsigned int hr, unsigned int width, unsigned int dt, Expression guard)
    {
        if(hr == 0)
            hr = h;
        std::vector<Expression> args;
        for(unsigned int w = 0; w < width; ++w)
            args.push_back(R + (hr * width + w));
        return {Butterfly{true, args}};
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
                           LoadGlobal{buf, offset + Parens{Expression{idx}} * stride0}};
        }
        return load;
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
            auto tid = thread + dt + h * threads_per_transform;
            auto idx = offset
                       + (Parens{tid / cumheight} * (width * cumheight) + tid % cumheight
                          + w * cumheight)
                             * stride0;
            work += StoreGlobal(buf, idx, R[hr * width + w]);
        }
        return work;
    }

    // Call generator as many times as needed.
    // generator accepts h, hr, width, dt, guard_pred parameters
    StatementList
        add_work(std::function<StatementList(
                     unsigned int, unsigned int, unsigned int, unsigned int, Expression)> generator,
                 unsigned int                                                             width,
                 double                                                                   height,
                 ThreadGuardMode                                                          guard,
                 bool                               trans_dir    = false,
                 const std::optional<unsigned int>& guard_factor = std::nullopt,
                 const std::optional<unsigned int>& work_length  = std::nullopt) const
    {
        StatementList stmts;
        unsigned int  iheight = std::floor(height);
        if(height > iheight && threads_per_transform > length / width)
            iheight += 1;

        Expression guard_expr = Expression{Literal{"true"}};

        const auto effective_length = work_length ? *work_length : length;
        const auto thread_guard_cond
            = (effective_length / width) * (guard_factor ? *guard_factor : 1);

        // do thread gurad when guard_by_if or guard_by_arg
        if(guard != ThreadGuardMode::NO_GUARD)
        {
            // using ">" : no need to test "if(thread < XXX)"" if it is always true
            if((!trans_dir && threads_per_transform > (effective_length / width))
               || (trans_dir && workgroup_size / transforms_per_block > (effective_length / width)))
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

        if(height > iheight && threads_per_transform < effective_length / width)
        {
            stmts += CommentLines{"not enough threads, some threads do extra work"};
            unsigned int dt = iheight * threads_per_transform;

            // always do thread gurad
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

    Function generate_lds_to_reg_input_function()
    {
        std::string function_name = "lds_to_reg_input_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_templates();
        f.arguments = device_lds_reg_inout_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};

        body += Declaration{l_offset};

        auto load_lds = std::mem_fn(&StockhamKernel::load_lds_generator);
        // first pass of load (full)
        unsigned int width  = factors[0];
        float        height = static_cast<float>(length) / width / threads_per_transform;
        body += If{lds_reg_sync, {SyncThreads()}};
        body += add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5, Component::BOTH),
                         width,
                         height,
                         ThreadGuardMode::NO_GUARD);

        return f;
    }

    Function generate_lds_from_reg_output_function()
    {
        std::string function_name
            = "lds_from_reg_output_length" + std::to_string(length) + "_device";

        Function f{function_name};
        f.templates = device_lds_reg_inout_templates();
        f.arguments = device_lds_reg_inout_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};

        body += Declaration{l_offset};

        auto store_lds = std::mem_fn(&StockhamKernel::store_lds_generator);
        // last pass of store (full)
        unsigned int width     = factors.back();
        float        height    = static_cast<float>(length) / width / threads_per_transform;
        unsigned int cumheight = product(factors.begin(), factors.end() - 1);
        body += If{lds_reg_sync, {SyncThreads()}};
        body += add_work(std::bind(store_lds, this, _1, _2, _3, _4, _5, Component::BOTH, cumheight),
                         width,
                         height,
                         ThreadGuardMode::GUARD_BY_IF);
        return f;
    }

    Function generate_device_function()
    {
        std::string function_name
            = "forward_length" + std::to_string(length) + "_" + tiling_name() + "_device";

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

            auto load_lds  = std::mem_fn(&StockhamKernel::load_lds_generator);
            auto store_lds = std::mem_fn(&StockhamKernel::store_lds_generator);

            if(npass > 0)
            {
                // internal full lds2reg (both linear/nonlinear variants)
                StatementList lds2reg_full;
                lds2reg_full += SyncThreads();
                lds2reg_full
                    += add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5, Component::BOTH),
                                width,
                                height,
                                ThreadGuardMode::GUARD_BY_IF,
                                true);
                body += If{Not{lds_is_real}, lds2reg_full};

                auto apply_twiddle = std::mem_fn(&StockhamKernel::apply_twiddle_generator);
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

    void collect_length_stride(StatementList& body)
    {
        if(static_dim)
        {
            body += Declaration{dim, static_dim};
        }
        body += Declaration{
            stride0, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride[0]}}};
    }

    virtual Function generate_global_function()
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

        // TODO- don't override, unify them
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
        body += Declaration{thread_in_device,
                            Ternary{lds_linear,
                                    thread_id % threads_per_transform,
                                    thread_id / transforms_per_block}};

        // before starting the transform job (core device function)
        // we call a re-load lds-to-reg function here, but it's not always doing things.
        // If we're doing direct-to-reg, this function simply returns.
        body += LineBreak{};
        body += CommentLines{"call a pre-load from lds to registers (if necessary)"};
        auto pre_post_lds_tmpl = device_lds_reg_inout_device_call_templates();
        auto pre_post_lds_args = device_lds_reg_inout_device_call_arguments();
        pre_post_lds_tmpl.set_value(stride_type.name, "lds_linear ? SB_UNIT : SB_NONUNIT");
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

            templates.set_value(stride_type.name, "lds_linear ? SB_UNIT : SB_NONUNIT");

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
        StatementList postStore;
        postStore += Call{"lds_from_reg_output_length" + std::to_string(length) + "_device",
                          pre_post_lds_tmpl,
                          pre_post_lds_args};
        if(!direct_to_from_reg)
            body += postStore;
        else
            body += If{!direct_store_from_reg, postStore};

        body += LineBreak{};
        StatementList storelds;
        storelds += LineBreak{};
        // handle even-length complex to real post-process in lds after transform
        if(ebtype == EmbeddedType::Real2C_POST)
            storelds += real_trans_pre_post();
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

    // void update_kernel_settings();

    virtual StatementList calculate_offsets() = 0;

    virtual StatementList load_from_global(bool load_registers) = 0;

    virtual StatementList store_to_global(bool store_registers) = 0;

    virtual TemplateList device_lds_reg_inout_device_call_templates(bool syncthreads = true)
    {
        Variable sync_var{syncthreads ? "true" : "false", "bool"};
        return {scalar_type, stride_type, sync_var};
    }

    virtual std::vector<Expression> device_lds_reg_inout_device_call_arguments()
    {
        return {R, lds_complex, stride_lds, offset_lds, thread_in_device, Literal{"true"}};
    }

    virtual TemplateList device_call_templates()
    {
        return {scalar_type, lds_is_real, stride_type, lds_linear, direct_load_to_reg};
    }

    virtual std::vector<Expression> device_call_arguments(unsigned int call_iter)
    {
        return {R,
                lds_real,
                lds_complex,
                twiddles,
                stride_lds,
                call_iter ? Expression{offset_lds + call_iter * stride_lds * transforms_per_block}
                          : Expression{offset_lds},
                thread_in_device,
                Literal{"true"}};
    }

    StatementList
        real2cmplx_pre_post(unsigned int half_N, unsigned int tpt, unsigned int twd_offset)
    {
        if(ebtype == EmbeddedType::NONE)
            return {};

        std::string function_name = ebtype == EmbeddedType::C2Real_PRE
                                        ? "real_pre_process_kernel_inplace"
                                        : "real_post_process_kernel_inplace";
        Variable    Ndiv4{half_N % 2 == 0 ? "true" : "false", "bool"};
        auto        quarter_N = half_N / 2;
        if(half_N % 2 == 1)
            quarter_N += 1;

        StatementList stmts;
        // Todo: We might not have to sync here which depends on the access pattern
        stmts += SyncThreads{};
        stmts += LineBreak{};

        // Todo: For case threads_per_transform == quarter_N, we
        // could save one more "if" in the c2r/r2r kernels

        // if we have fewer threads per transform than quarter_N,
        // we need to call the pre/post function multiple times
        auto r2c_calls_per_transform = quarter_N / tpt;
        if(quarter_N % tpt > 0)
            r2c_calls_per_transform += 1;
        for(unsigned int i = 0; i < r2c_calls_per_transform; ++i)
        {
            TemplateList tpls;
            tpls.append(scalar_type);
            tpls.append(Ndiv4);
            std::vector<Expression> args{thread_id % tpt + i * tpt,
                                         half_N - thread_id % tpt - i * tpt,
                                         quarter_N,
                                         lds_complex + offset_lds,
                                         0,
                                         twiddles + twd_offset};
            stmts += Call{function_name, tpls, args};
        }
        if(ebtype == EmbeddedType::C2Real_PRE)
        {
            stmts += SyncThreads();
            stmts += LineBreak();
        }

        return stmts;
    }
};
