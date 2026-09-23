// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/generic_search.hpp>
#include <miopen/generic_search_controls.hpp>

#include <cstddef>
#include <chrono>

namespace miopen {
namespace solver {
namespace debug {

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static std::optional<std::size_t> tuning_iterations_limit;

TuningIterationScopedLimiter::TuningIterationScopedLimiter(std::size_t new_limit)
    : old_limit(tuning_iterations_limit)
{
    tuning_iterations_limit = new_limit;
}

TuningIterationScopedLimiter::~TuningIterationScopedLimiter()
{
    tuning_iterations_limit = old_limit;
}
} // namespace debug

std::size_t GetTuningIterationsMax()
{
    if(debug::tuning_iterations_limit)
        return *debug::tuning_iterations_limit;
    return env::value(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX);
}

std::chrono::milliseconds GetTuningTimeMax()
{
    return std::chrono::milliseconds{env::value(MIOPEN_TUNING_TIME_MS_MAX)};
}

std::size_t GetTuningThreadsMax() { return env::value(MIOPEN_COMPILE_PARALLEL_LEVEL); }

std::size_t GetTuningPatience() { return env::value(MIOPEN_TUNING_PATIENCE); }

} // namespace solver
} // namespace miopen
