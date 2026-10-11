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
#include "rocfft/rocfft.h"
#include "stockham_gen_base.h"

// How a partial-pass kernel lays out the off-dimension points it owns in LDS.
// The SBRR gives each transform its own LDS column and strides between columns
// to reach the off-dimension; the SBCC interleaves the off-dimension into the
// column, so a thread's points are contiguous.
enum class LDSColumnPattern
{
    NON_INTERLEAVED,
    OFF_DIM_INTERLEAVED
};

// The two partial-pass kernels split the off-dimension pass between them.
// r2c runs SBRR then SBCC and c2r runs SBCC then SBRR, so whichever kernel
// the plan runs first performs steps 1/2 of the four-step decomposition.
enum class PartialPassSteps
{
    STEPS_1_2,
    STEPS_3_4
};

// Base class for stockham partial pass kernels.
// Subclasses are responsible for different tiling types.
struct StockhamPartialPassKernel : public StockhamKernel
{
    explicit StockhamPartialPassKernel(const StockhamGeneratorSpecs&    specs,
                                       const StockhamPartialPassParams& params,
                                       const LDSColumnPattern&          lds_column_pattern)
        : StockhamKernel(specs)
        , params(params)
        , lds_column_pattern(lds_column_pattern)
    {
        factors_pp               = params.pp_factors_curr;
        factors_pp_other         = params.pp_factors_other;
        pp_factors_prod          = product(factors_pp.begin(), factors_pp.end());
        pp_factors_other_prod    = product(factors_pp_other.begin(), factors_pp_other.end());
        threads_per_transform_pp = params.pp_threads_per_transform;
        transforms_per_block_pp  = workgroup_size / threads_per_transform_pp;

        // the two kernels split the off-dimension between them, so their
        // factors multiply out to its length
        length_pp = pp_factors_prod * pp_factors_other_prod;

        if(params.node_length.empty())
            throw std::runtime_error("partial pass node_length is not set");

        // off_dim is a plan dimension, and node_length is in this kernel's node
        // ordering: plan order for the SBRR, rotated one slot to the right for
        // the SBCC so its own transform dimension comes first.  If the
        // off-dimension length is not where that puts it, the caller built
        // node_length in the wrong order.
        const auto off_dim_index = lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                                       ? params.off_dim
                                       : (params.off_dim + 1) % params.node_length.size();
        if(params.node_length[off_dim_index] != length_pp)
            throw std::runtime_error(
                "partial pass node_length is not in the ordering this kernel expects");

        // the interleaved layout addresses LDS per transform point, so a thread
        // has to own a whole number of off-dimension transforms
        if(lds_column_pattern == LDSColumnPattern::OFF_DIM_INTERLEAVED && factors_pp.size() > 1
           && length % (threads_per_transform * pp_factors_prod) != 0)
            throw std::runtime_error(
                "interleaved partial pass with multiple factors needs length divisible by "
                "threads_per_transform * pp_factors_prod");

        if(!transform_type.has_value())
            throw std::runtime_error("transform_type is not set");
        transform_type_pp = static_cast<rocfft_transform_type>(transform_type.value());
    }
    virtual ~StockhamPartialPassKernel(){};

    StockhamPartialPassParams params;

    unsigned int              pp_factors_prod;
    unsigned int              pp_factors_other_prod;
    std::vector<unsigned int> factors_pp_other;
    rocfft_transform_type     transform_type_pp;

    PartialPassSteps partial_pass_steps = PartialPassSteps::STEPS_1_2;

    // Number of off-dimension butterflies each thread performs.  With one LDS
    // column per transform a thread owns pp_factors_prod off-dimension points;
    // with the off-dimension interleaved into the column it owns a whole
    // transform's worth of them.
    float pp_height(unsigned int width) const
    {
        return lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                   ? static_cast<float>(pp_factors_prod) / width / threads_per_transform_pp
                   : static_cast<float>(length) / width / threads_per_transform;
    }

    // Which of the pp_factors_other_prod groups along the off-dimension this
    // block works on, i.e. the row of the four-step twiddle table.
    virtual Expression pp_twiddle_row_index()
    {
        return block_id % (length_pp / pp_factors_prod);
    }

    Variable tile_index{"tile_index", "integer_type"};
    Variable num_of_tiles{"num_of_tiles", "integer_type"};
    Variable in_bound{"in_bound", "bool"};
    Variable thread{"thread", rtc_kint_type(KIntType::U32)}; // replacing tid_ver
    Variable tid_hor{"tid_hor", rtc_kint_type(KIntType::U32)}; // id along row
    Variable stride_in{"stride_in", "const integer_type", true};
    Variable stride_out{"stride_out", "const integer_type", true};

    Variable intrinsic_mode{"intrinsic_mode", "IntrinsicAccessType"};
    Variable apply_large_twiddle{"apply_large_twiddle", "bool"};
    Variable large_twiddle_steps{"large_twiddle_steps", rtc_kint_type(KIntType::U32)};
    Variable large_twiddle_base{"large_twiddle_base", rtc_kint_type(KIntType::U32)};

    Variable large_twiddles{"large_twiddles", "const scalar_type", true};

    Variable stride_lds_pp{"stride_lds_pp", rtc_kint_type(KIntType::U32)};
    Variable offset_lds_pp{"offset_lds_pp", rtc_kint_type(KIntType::U32)};
    Variable offset_pp{"offset_pp", "integer_type"};
    Variable thread_pp{"thread_pp", rtc_kint_type(KIntType::U32)};
    Variable twiddles_pp{"twiddles_pp", "const scalar_type", true, true};
    Variable twiddles_off_dim{"twiddles_off_dim", "const scalar_type", true, true};
    Variable global_idx{"global_idx", "integer_type"};
    Variable transpose_idx{"transpose_idx", "integer_type"};

    LDSColumnPattern lds_column_pattern = LDSColumnPattern::NON_INTERLEAVED;

    ArgumentList device_lds_reg_inout_non_interleaved_arguments()
    {
        ArgumentList args{R, lds_complex, stride_lds, offset_lds, thread};
        return args;
    }

    ArgumentList device_lds_reg_inout_off_dim_interleaved_arguments()
    {
        ArgumentList args{R, lds_complex, stride_lds, offset_lds};
        return args;
    }

    TemplateList device_lds_reg_pp_inout_templates()
    {
        TemplateList tpls;
        tpls.append(scalar_type);
        return tpls;
    }

    std::vector<Expression> device_lds_reg_inout_non_interleaved_device_call_arguments()
    {
        return {R, lds_complex, stride_lds_pp, offset_lds_pp, thread_in_device_pp};
    }

    std::vector<Expression> device_lds_reg_inout_off_dim_interleaved_device_call_arguments()
    {
        return {R, lds_complex, stride_lds_pp, offset_lds_pp};
    }

    TemplateList device_pp_steps_call_templates()
    {
        return {scalar_type, lds_is_real, lds_linear, direct_load_to_reg};
    }

    StatementList load_non_interleaved_lds_generator(
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

    // A thread's slice of the interleaved column holds its transform points back
    // to back, each point's pp_factors_prod off-dimension values contiguous, so
    // butterfly hr belongs to point hr / nbutterfly and gathers its inputs from
    // within that point.  Collapses to a straight copy for a single-factor
    // partial pass, where one butterfly covers the whole off-dimension.
    StatementList load_off_dim_interleaved_lds_generator(
        unsigned int h, unsigned int hr, unsigned int width, unsigned int dt, Expression guard)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        const auto nbutterfly = pp_factors_prod / width;
        const auto base       = (hr / nbutterfly) * pp_factors_prod;
        const auto butterfly  = hr % nbutterfly;

        for(unsigned int w = 0; w < width; ++w)
            work += Assign(
                R[hr * width + w],
                lds_complex[offset_lds + (base + butterfly + w * nbutterfly) * stride_lds]);

        return work;
    }

    StatementList store_non_interleaved_lds_generator(unsigned int h,
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

    StatementList store_off_dim_interleaved_lds_generator(unsigned int h,
                                                          unsigned int hr,
                                                          unsigned int width,
                                                          unsigned int dt,
                                                          Expression   guard,
                                                          unsigned int cumheight)
    {
        if(hr == 0)
            hr = h;
        StatementList work;

        const auto nbutterfly = pp_factors_prod / width;
        const auto base       = (hr / nbutterfly) * pp_factors_prod;
        const auto butterfly  = hr % nbutterfly;

        for(unsigned int w = 0; w < width; ++w)
        {
            const auto idx = base + (butterfly / cumheight) * (width * cumheight)
                             + butterfly % cumheight + w * cumheight;

            work += Assign(lds_complex[offset_lds + idx * stride_lds], R[hr * width + w]);
        }

        return work;
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

    Function generate_non_interleaved_lds_to_reg_input_function(const std::string& function_name)
    {
        Function f{function_name};
        f.templates = device_lds_reg_pp_inout_templates();
        f.arguments = device_lds_reg_inout_non_interleaved_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;

        auto load_lds = std::mem_fn(&StockhamPartialPassKernel::load_non_interleaved_lds_generator);
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

    Function
        generate_off_dim_interleaved_lds_to_reg_input_function(const std::string& function_name)
    {
        Function f{function_name};
        f.templates = device_lds_reg_pp_inout_templates();
        f.arguments = device_lds_reg_inout_off_dim_interleaved_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;

        auto load_lds
            = std::mem_fn(&StockhamPartialPassKernel::load_off_dim_interleaved_lds_generator);
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

    Function generate_lds_to_reg_partial_pass_steps_1_2_input_function()
    {
        std::string function_name = "lds_to_reg_steps_1_2_input_partial_pass_length"
                                    + std::to_string(pp_factors_prod) + "_device";

        return lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                   ? generate_non_interleaved_lds_to_reg_input_function(function_name)
                   : generate_off_dim_interleaved_lds_to_reg_input_function(function_name);
    }

    Function generate_lds_to_reg_partial_pass_steps_3_4_input_function()
    {
        std::string function_name = "lds_to_reg_steps_3_4_input_partial_pass_length"
                                    + std::to_string(pp_factors_prod) + "_device";

        return lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                   ? generate_non_interleaved_lds_to_reg_input_function(function_name)
                   : generate_off_dim_interleaved_lds_to_reg_input_function(function_name);
    }

    Function generate_non_interleaved_lds_from_reg_output_function(const std::string& function_name)
    {
        Function f{function_name};
        f.templates = device_lds_reg_pp_inout_templates();
        f.arguments = device_lds_reg_inout_non_interleaved_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;

        auto store_lds
            = std::mem_fn(&StockhamPartialPassKernel::store_non_interleaved_lds_generator);
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

    Function
        generate_off_dim_interleaved_lds_from_reg_output_function(const std::string& function_name)
    {
        Function f{function_name};
        f.templates = device_lds_reg_pp_inout_templates();
        f.arguments = device_lds_reg_inout_off_dim_interleaved_arguments();
        f.qualifier = "__device__";

        StatementList& body = f.body;

        auto store_lds
            = std::mem_fn(&StockhamPartialPassKernel::store_off_dim_interleaved_lds_generator);
        // last pass of store (partial-pass)
        unsigned int width     = factors_pp.back();
        float        height    = static_cast<float>(length) / width / threads_per_transform;
        unsigned int cumheight = product(factors_pp.begin(), factors_pp.end() - 1);
        body += SyncThreads();
        body += add_work(std::bind(store_lds, this, _1, _2, _3, _4, _5, cumheight),
                         width,
                         height,
                         ThreadGuardMode::NO_GUARD);
        return f;
    }

    Function generate_lds_from_reg_partial_pass_steps_1_2_output_function()
    {
        std::string function_name = "lds_from_reg_steps_1_2_output_partial_pass_length"
                                    + std::to_string(pp_factors_prod) + "_device";

        return lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                   ? generate_non_interleaved_lds_from_reg_output_function(function_name)
                   : generate_off_dim_interleaved_lds_from_reg_output_function(function_name);
    }

    Function generate_lds_from_reg_partial_pass_steps_3_4_output_function()
    {
        std::string function_name = "lds_from_reg_steps_3_4_output_partial_pass_length"
                                    + std::to_string(pp_factors_prod) + "_device";

        return lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                   ? generate_non_interleaved_lds_from_reg_output_function(function_name)
                   : generate_off_dim_interleaved_lds_from_reg_output_function(function_name);
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

        // when the off-dimension is interleaved into the LDS column, hr walks
        // the main transform and the off-dimension index is known at generation
        // time from the butterfly this register belongs to
        const auto butterfly = hr % (pp_factors_prod / width);
        const auto off_dim = (butterfly / cumheight) * (width * cumheight) + butterfly % cumheight;

        for(unsigned int w = 0; w < width; ++w)
        {
            auto tid = thread + dt + h * threads_per_transform_pp;
            auto tidx
                = lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                      ? Expression{thread_pp * Literal(length_pp)
                                   + (Parens{tid / cumheight} * (width * cumheight)
                                      + tid % cumheight + w * cumheight)}
                      : Expression{thread_pp * Literal(length_pp) + (off_dim + w * cumheight)};
            auto ridx = hr * width + w;

            work += Assign(W, twiddles_pp[tidx]);
            work += Assign(t, TwiddleMultiply(R[ridx], W));
            work += Assign(R[ridx], t);
        }
        return work;
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

    // The Stockham shuffle between two partial-pass radix passes goes through
    // LDS in both layouts.  Only the addressing and the work decomposition
    // differ: one LDS column per transform splits the off-dimension across
    // threads_per_transform_pp threads, while interleaving it into the column
    // gives each thread a private slice of its own transform points.
    StatementList add_pp_lds2reg_work(unsigned int width, float height)
    {
        if(lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED)
        {
            auto load_lds
                = std::mem_fn(&StockhamPartialPassKernel::load_non_interleaved_lds_generator);
            return add_pp_work(std::bind(load_lds, this, _1, _2, _3, _4, _5),
                               width,
                               height,
                               ThreadGuardMode::GUARD_BY_IF,
                               true);
        }

        auto load_lds
            = std::mem_fn(&StockhamPartialPassKernel::load_off_dim_interleaved_lds_generator);
        return add_work(std::bind(load_lds, this, _1, _2, _3, _4, _5),
                        width,
                        height,
                        ThreadGuardMode::NO_GUARD);
    }

    StatementList add_pp_reg2lds_work(unsigned int width, float height, unsigned int cumheight)
    {
        if(lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED)
        {
            auto store_lds
                = std::mem_fn(&StockhamPartialPassKernel::store_non_interleaved_lds_generator);
            return add_pp_work(std::bind(store_lds, this, _1, _2, _3, _4, _5, cumheight),
                               width,
                               height,
                               ThreadGuardMode::GUARD_BY_IF,
                               false);
        }

        auto store_lds
            = std::mem_fn(&StockhamPartialPassKernel::store_off_dim_interleaved_lds_generator);
        return add_work(std::bind(store_lds, this, _1, _2, _3, _4, _5, cumheight),
                        width,
                        height,
                        ThreadGuardMode::NO_GUARD);
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

    Function generate_pp_steps_1_2_device_function()
    {
        std::string function_name = "forward_partial_pass_steps_1_2_length"
                                    + std::to_string(pp_factors_prod) + "_" + tiling_name()
                                    + "_device";

        Function f{function_name};
        f.arguments = device_pp_steps_1_2_arguments();
        f.templates = device_pp_templates();
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
            height = pp_height(width);

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

            if(npass > 0)
            {
                // internal full lds2reg (both linear/nonlinear variants)
                StatementList lds2reg_full;
                lds2reg_full += SyncThreads();
                lds2reg_full += add_pp_lds2reg_work(width, height);
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
                reg2lds_full += add_pp_reg2lds_work(width, height, cumheight);

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

    ArgumentList device_pp_steps_3_4_arguments()
    {
        ArgumentList args{
            R, lds_real, lds_complex, twiddles, stride_lds, offset_lds, thread, write};
        return args;
    }

    Function generate_pp_steps_3_4_device_function()
    {
        std::string function_name = "forward_partial_pass_steps_3_4_length"
                                    + std::to_string(pp_factors_prod) + "_" + tiling_name()
                                    + "_device";

        Function f{function_name};
        f.arguments = device_pp_steps_3_4_arguments();
        f.templates = device_pp_templates();
        f.qualifier = "__device__";
        if(pp_factors_prod == 1)
            return f;

        StatementList& body = f.body;
        body += Declaration{W};
        body += Declaration{t};

        for(unsigned int npass = 0; npass < factors_pp.size(); ++npass)
        {
            // width is the butterfly width, Radix-n.
            unsigned int width = factors_pp[npass];
            // height is how many butterflies per thread will do on average
            float height = pp_height(width);

            unsigned int cumheight = product(factors_pp.begin(), factors_pp.begin() + npass);

            body += LineBreak{};
            body += CommentLines{
                "pass " + std::to_string(npass) + ", width " + std::to_string(width),
                "using " + std::to_string(threads_per_transform_pp) + " threads we need to do "
                    + std::to_string(pp_factors_prod / width) + " radix-" + std::to_string(width)
                    + " butterflies",
                "therefore each thread will do " + std::to_string(height) + " butterflies"};

            if(npass > 0)
            {
                // internal full lds2reg (both linear/nonlinear variants)
                StatementList lds2reg_full;
                lds2reg_full += SyncThreads();
                lds2reg_full += add_pp_lds2reg_work(width, height);
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

            // internal lds store
            if(npass < factors_pp.size() - 1)
            {
                StatementList reg2lds_full;
                reg2lds_full += SyncThreads();
                reg2lds_full += add_pp_reg2lds_work(width, height, cumheight);

                body += reg2lds_full;
            }
        }

        return f;
    }

    TemplateList device_lds_reg_inout_pp_device_call_templates()
    {
        return {scalar_type};
    }

    StatementList generate_partial_pass_offsets()
    {
        StatementList stmts;

        stmts += LineBreak{};
        stmts += CommentLines{"partial-pass offsets"};
        switch(lds_column_pattern)
        {
        case LDSColumnPattern::NON_INTERLEAVED:
            stmts += Declaration{stride_lds_pp, (length + get_lds_padding())};
            stmts += Declaration{offset_lds_pp,
                                 Parens(block_id * transforms_per_block + thread_id)
                                     % (length + get_lds_padding())};
            break;
        case LDSColumnPattern::OFF_DIM_INTERLEAVED:
            unsigned int width  = factors_pp[0];
            unsigned int height = length / width / threads_per_transform;

            stmts += Declaration{stride_lds_pp, Literal{1}};
            stmts += Declaration{offset_lds_pp, thread_id * Literal{width * height}};
            break;
        }

        return stmts;
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

    StatementList generate_partial_pass_steps_1_2()
    {
        StatementList stmts;

        stmts += LineBreak{};
        stmts += CommentLines{
            "calc the thread_in_device value once and for all partial-pass device funcs"};
        stmts += Declaration{thread_in_device_pp, thread_id % threads_per_transform_pp};
        stmts += Declaration{thread_in_device_pp_twiddles, pp_twiddle_row_index()};

        stmts += generate_partial_pass_offsets();

        auto pre_post_lds_tmpl = device_lds_reg_inout_pp_device_call_templates();
        auto pre_post_lds_args
            = lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                  ? device_lds_reg_inout_non_interleaved_device_call_arguments()
                  : device_lds_reg_inout_off_dim_interleaved_device_call_arguments();

        StatementList preLoad;
        stmts += LineBreak{};
        stmts += CommentLines{"call a pre-load from lds to registers"};
        preLoad += Call{"lds_to_reg_steps_1_2_input_partial_pass_length"
                            + std::to_string(pp_factors_prod) + "_device",
                        pre_post_lds_tmpl,
                        pre_post_lds_args};
        stmts += preLoad;

        auto          device_tmpl = device_pp_steps_call_templates();
        auto          device_args = device_pp_steps_1_2_call_arguments(0);
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

    std::vector<Expression> device_pp_steps_3_4_call_arguments(unsigned int call_iter)
    {
        return {R,
                lds_real,
                lds_complex,
                twiddles_off_dim,
                stride_lds_pp,
                call_iter ? Expression{offset_lds_pp
                                       + call_iter * stride_lds_pp * transforms_per_block_pp}
                          : Expression{offset_lds_pp},
                thread_in_device_pp,
                Literal{"true"}};
    }

    StatementList generate_partial_pass_steps_3_4()
    {
        StatementList stmts;

        stmts += LineBreak{};
        stmts += CommentLines{
            "calc the thread_in_device value once and for all partial-pass device funcs"};
        stmts += Declaration{thread_in_device_pp, thread_id % threads_per_transform_pp};

        stmts += generate_partial_pass_offsets();

        auto pre_post_lds_tmpl = device_lds_reg_inout_pp_device_call_templates();
        auto pre_post_lds_args
            = lds_column_pattern == LDSColumnPattern::NON_INTERLEAVED
                  ? device_lds_reg_inout_non_interleaved_device_call_arguments()
                  : device_lds_reg_inout_off_dim_interleaved_device_call_arguments();

        StatementList preLoad;
        stmts += LineBreak{};
        stmts += CommentLines{"call a pre-load from lds to registers"};
        preLoad += Call{"lds_to_reg_steps_3_4_input_partial_pass_length"
                            + std::to_string(pp_factors_prod) + "_device",
                        pre_post_lds_tmpl,
                        pre_post_lds_args};
        stmts += preLoad;

        auto device_tmpl = device_pp_steps_call_templates();
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

        if(lds_column_pattern == LDSColumnPattern::OFF_DIM_INTERLEAVED)
        {
            unsigned int width  = factors_pp.back();
            unsigned int height = length / width / threads_per_transform;
            stmts += Assign{offset_lds_pp, thread_id * Literal{width * height}};
        }

        StatementList postStore;
        stmts += LineBreak{};
        stmts += CommentLines{"call a post-store from registers to lds"};
        postStore += Call{"lds_from_reg_steps_3_4_output_partial_pass_length"
                              + std::to_string(pp_factors_prod) + "_device",
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
        f.return_type = "integer_type";
        f.qualifier   = "__device__";

        StatementList& body = f.body;

        auto len_1 = params.node_length[2];
        auto len_2 = params.node_length[1];
        auto len_3 = params.node_length[0];

        auto len_1_2_3 = len_1 * len_2 * len_3;
        auto len_1_2   = len_1 * len_2;

        // off-dimension index i = lo + radix_lo * hi becomes hi + radix_hi * lo.
        // Steps 1/2 splits off the low digit and steps 3/4 the high one, so the
        // two halves of the pass swap the digit they gather over.
        auto radix_lo = partial_pass_steps == PartialPassSteps::STEPS_1_2 ? pp_factors_other_prod
                                                                          : pp_factors_prod;
        auto radix_hi = partial_pass_steps == PartialPassSteps::STEPS_1_2 ? pp_factors_prod
                                                                          : pp_factors_other_prod;

        auto len_radix_lo = radix_lo * len_2;
        auto len_radix_hi = radix_hi * len_2;

        body += Declaration{transpose_idx, global_idx % len_1_2_3};

        body += Assign{
            transpose_idx,
            Parens{transpose_idx % len_2}
                + Parens{Parens{Parens{transpose_idx % (len_radix_lo)} / len_2} * len_radix_hi}
                + Parens{Parens{Parens{transpose_idx % len_1_2} / len_radix_lo} * len_2}
                + Parens{Parens{transpose_idx / len_1_2} * len_1_2}};

        body += Assign{transpose_idx, transpose_idx + Parens{global_idx / len_1_2_3} * len_1_2_3};

        body += ReturnExpr(transpose_idx);

        return f;
    }
};