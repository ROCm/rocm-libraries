#include "hipconv/hipconv.hpp"
#include "conv_kernel.h"
#include "algorithm.h"
#include "arch_registry.h"
#include "hip_util.h"
#include "launch_params.h"
#include "tolerance.h"

#include <array>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipconv
{

namespace
{

// Match an arch name against a bare gfx name, ignoring feature qualifiers
// (e.g. "gfx950:sramecc+:xnack-" matches "gfx950").
bool arch_name_matches(std::string_view requested, std::string_view registered)
{
    return requested.starts_with(registered) &&
           (requested.size() == registered.size() || requested[registered.size()] == ':');
}

// Drop the scores the merge ranked on; the public list is ordered, not annotated.
std::vector<ConvKernelHandle> handles(const std::vector<ScoredKernel>& scored)
{
    std::vector<ConvKernelHandle> out;
    out.reserve(scored.size());
    for(const auto& entry : scored)
        out.push_back(entry.kernel);
    return out;
}

// Never launched; only something for hipFuncGetAttributes to ask about.
__global__ void device_code_probe() {}

// Whether this binary carries device code for the current GPU.
//
// The registry covers every arch in the tree, so a hit there does not mean the
// kernels were built here. Cached per device; architectures can differ per GPU.
bool device_code_available()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return false;

    const auto probe = [] {
        hipFuncAttributes attr{};
        return hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&device_code_probe)) ==
               hipSuccess;
    };

    enum : signed char
    {
        unknown = 0,
        present = 1,
        absent  = -1
    };
    static constexpr int max_cached_devices = 32;
    static std::array<std::atomic<signed char>, max_cached_devices> cache{};

    if(device < 0 || device >= max_cached_devices)
        return probe();

    auto& slot       = cache[static_cast<std::size_t>(device)];
    const auto known = slot.load(std::memory_order_relaxed);
    if(known != unknown)
        return known == present;

    const bool available = probe();
    slot.store(available ? present : absent, std::memory_order_relaxed);
    return available;
}

} // anonymous namespace

std::optional<ArchHandle> resolve_arch(std::string_view name)
{
    if(!device_code_available())
        return std::nullopt;
    for(std::size_t i = 0; i < hipconv_arch_registry_size; ++i)
    {
        if(arch_name_matches(name, hipconv_arch_registry[i].name))
            return &hipconv_arch_registry[i];
    }
    return std::nullopt;
}

std::vector<ConvKernelHandle>
get_valid_configs(ArchHandle arch, const Conv2dParams& par, Algorithm algo, std::size_t max_ranked)
{
    if(!arch)
        throw std::invalid_argument("null arch handle");
    for(const auto& entry : arch->algorithms)
    {
        if(entry.algorithm == algo)
            return handles(entry.impl->get_valid_configs(par, max_ranked));
    }
    return {};
}

std::vector<ConvKernelHandle>
get_valid_configs(ArchHandle arch, const Conv2dParams& par, std::size_t max_ranked)
{
    if(!arch)
        throw std::invalid_argument("null arch handle");
    std::vector<ScoredKernel> scored;
    for(const auto& entry : arch->algorithms)
    {
        auto cfgs = entry.impl->get_valid_configs(par, max_ranked);
        scored.insert(scored.end(), cfgs.begin(), cfgs.end());
    }
    keep_top_ranked(scored, max_ranked);
    return handles(scored);
}

std::optional<ConvKernelHandle> find_config(ArchHandle arch, const Conv2dParams& par)
{
    auto cfgs = get_valid_configs(arch, par);
    if(cfgs.empty())
        return std::nullopt;
    return cfgs.front();
}

bool is_applicable(ConvKernelHandle kernel, const Conv2dParams& par)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->is_applicable(par) && kernel->is_valid_config(par);
}

std::string_view name(ConvKernelHandle kernel)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->name();
}

Algorithm algorithm(ConvKernelHandle kernel)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->algorithm();
}

std::string describe_config(ConvKernelHandle kernel)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->describe_config();
}

bool matches_descriptor(ConvKernelHandle kernel, std::string_view spec, std::string* error)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->matches_descriptor(spec, error);
}

size_t get_workspace_size(ConvKernelHandle kernel, const Conv2dParams& par)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->get_workspace_size(par);
}

float get_weighted_throughput_index(ConvKernelHandle kernel, const Conv2dParams& par)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->get_weighted_throughput_index(par);
}

hipconvError_t launch(ConvKernelHandle kernel,
                      const Conv2dParams& par,
                      const void* in,
                      const void* wei,
                      void* out,
                      void* workspace,
                      hipStream_t stream)
{
    if(!kernel)
        return hipErrorInvalidValue;
    auto lp = kernel->get_launch_params(par);
    // Internally hipconv throws on error; the public API translates that into
    // a return code. Execution faults are asynchronous and surface at a later
    // synchronization, not here.
    try
    {
        kernel->launch(lp, par, in, wei, out, workspace, stream);
    }
    catch(const HipError& e)
    {
        return e.code;
    }
    return hipSuccess;
}

void get_tolerance(ConvKernelHandle kernel, const Conv2dParams& par, float& atol, float& rtol)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    kernel->get_tolerance(par, atol, rtol);
}

void get_recursive_summation_tolerance(const Conv2dParams& par, float& atol, float& rtol)
{
    get_mixed_precision_tolerance(par, atol, rtol);
}

struct ConvLaunch::State
{
    ConvKernel* kernel;
    Conv2dParams par;
    LaunchParams lp;
    size_t workspace_size;
};

std::optional<ConvLaunch> ConvLaunch::make(ConvKernelHandle kernel, Conv2dParams par)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    if(!kernel->is_applicable(par) || !kernel->is_valid_config(par))
        return std::nullopt;

    auto lp = kernel->get_launch_params(par);
    auto ws = kernel->get_workspace_size(par);
    auto st = std::make_unique<State>(State{kernel, std::move(par), lp, ws});
    return ConvLaunch{std::move(st)};
}

ConvLaunch::ConvLaunch(std::unique_ptr<State> state) noexcept : state_(std::move(state)) {}

ConvLaunch::ConvLaunch(ConvLaunch&&) noexcept            = default;
ConvLaunch& ConvLaunch::operator=(ConvLaunch&&) noexcept = default;
ConvLaunch::~ConvLaunch()                                = default;

size_t ConvLaunch::workspace_size() const noexcept
{
    return state_->workspace_size;
}

void ConvLaunch::get_tolerance(float& atol, float& rtol) const
{
    state_->kernel->get_tolerance(state_->par, atol, rtol);
}

const Conv2dParams& ConvLaunch::params() const noexcept
{
    return state_->par;
}

ConvKernelHandle ConvLaunch::kernel() const noexcept
{
    return state_->kernel;
}

hipconvError_t ConvLaunch::launch(const void* in,
                                  const void* wei,
                                  void* out,
                                  void* workspace,
                                  hipStream_t stream) const
{
    // Internally hipconv throws on error; the public API translates that into
    // a return code. Execution faults are asynchronous and surface at a later
    // synchronization, not here.
    try
    {
        state_->kernel->launch(state_->lp, state_->par, in, wei, out, workspace, stream);
    }
    catch(const HipError& e)
    {
        return e.code;
    }
    return hipSuccess;
}

} // namespace hipconv
