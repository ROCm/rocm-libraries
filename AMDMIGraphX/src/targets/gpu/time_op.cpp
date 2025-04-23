/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/program.hpp>
#include <migraphx/gpu/time_op.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/context.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/time.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_BENCHMARKING_BUNDLE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_BENCHMARKING_NRUNS);

static std::vector<argument> generate_arguments(const std::vector<shape>& shapes,
                                                unsigned long seed = 0,
                                                random_mode rm     = random_mode::random)
{
    std::vector<argument> args;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(args), [&](const auto& s) {
        return to_gpu(generate_argument(s, seed++, rm));
    });
    return args;
}

double
time_loop(migraphx::gpu::context& gctx, int bundle, int nruns, const std::function<void()>& f)
{
    // check for manual overrides
    bundle = value_of(MIGRAPHX_BENCHMARKING_BUNDLE{}, bundle);
    nruns  = value_of(MIGRAPHX_BENCHMARKING_NRUNS{}, nruns);

    std::vector<std::pair<hip_event_ptr, hip_event_ptr>> events(nruns);
    std::generate(events.begin(), events.end(), [] {
        return std::make_pair(context::create_event_for_timing(),
                              context::create_event_for_timing());
    });
    std::vector<double> times;
    // Warmup
    f();
    for(auto i : range(nruns))
    {
        gctx.get_stream().record(events[i].first.get());
        for(auto j : range(bundle))
        {
            (void)j;
            f();
        }
        gctx.get_stream().record(events[i].second.get());
    }
    gctx.finish();
    std::transform(events.begin(), events.end(), std::back_inserter(times), [&](const auto& p) {
        return context::get_elapsed_ms(p.first.get(), p.second.get()) / bundle;
    });
    std::sort(times.begin(), times.end());

    // compute common average by removing top and bottom 25% of values
    std::size_t quarters = times.size() / 4;
    double total         = std::accumulate(times.begin() + quarters, times.end() - quarters, 0.0);
    return total / std::distance(times.begin() + quarters, times.end() - quarters);
}

double
time_op(const context& ictx, operation op, const std::vector<shape>& inputs, int bundle, int nruns)
{
    // TODO: Use std::ref
    migraphx::context ctx = ictx;
    auto& gctx            = any_cast<migraphx::gpu::context>(ctx);
    auto output           = op.compute_shape(inputs);
    op.finalize(ctx, output, inputs);
    auto args = generate_arguments(inputs);
    auto run  = [&] { op.compute(ctx, output, args); };
    return time_loop(gctx, bundle, nruns, run);
}

double time_op(const context& ictx, operation op, int bundle, int nruns)
{
    auto inputs = any_cast<migraphx::gpu::code_object_op>(op).expected_inputs;
    return time_op(ictx, op, inputs, bundle, nruns);
}

double time_program(const context& ictx, program p, int bundle, int nruns)
{
    std::vector<migraphx::context> ctx_vec = {ictx};
    auto& gctx                             = any_cast<migraphx::gpu::context>(ctx_vec.front());
    auto* mm                               = p.get_main_module();
    mm->finalize(ctx_vec);
    auto in_shapes = p.get_parameter_shapes();
    std::unordered_map<std::string, migraphx::argument> param_map;
    unsigned long seed = 0;
    for(const auto& [name, shape] : in_shapes)
    {
        param_map[name] = to_gpu(generate_argument(shape, seed++, random_mode::random));
    }
    auto run = [&] { p.eval_with_context(ctx_vec, param_map); };
    return time_loop(gctx, bundle, nruns, run);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
