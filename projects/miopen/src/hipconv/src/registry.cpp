#include "hipconv/hipconv.hpp"
#include "conv_kernel.h"
#include "device_runtime.h"
#include "algorithm.h"
#include "arch_registry.h"
#include "hip_util.h"
#include "launch_params.h"
#include "registry.h"
#include "tolerance.h"

#include <array>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipconv
{

// Never launched: gives each GPU of the build a code object (docs/multi-arch-convergence.md),
// which has_device_code queries.
__global__ void device_code_anchor() {}

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

// A selection or launch precondition; see hipconv.hpp.
void check_device(const Device& device)
{
    if(!device.arch)
        throw std::invalid_argument("null arch handle");
    if(device.cu_count <= 0)
        throw std::invalid_argument("device.cu_count must be positive");
}

} // anonymous namespace

std::optional<ArchHandle> find_arch(std::string_view name)
{
    for(std::size_t i = 0; i < hipconv_arch_registry_size; ++i)
    {
        if(arch_name_matches(name, hipconv_arch_registry[i].name))
            return &hipconv_arch_registry[i];
    }
    return std::nullopt;
}

std::vector<std::string_view> arch_names()
{
    std::vector<std::string_view> names;
    names.reserve(hipconv_arch_registry_size);
    for(std::size_t i = 0; i < hipconv_arch_registry_size; ++i)
        names.push_back(hipconv_arch_registry[i].name);
    return names;
}

std::optional<ArchHandle> resolve_arch(std::string_view name)
{
    return find_arch(name);
}

bool has_device_code(int hip_device)
{
    enum : signed char
    {
        unknown = 0,
        present = 1,
        absent  = -1
    };
    static constexpr int max_cached_devices = 64;
    static std::array<std::atomic<signed char>, max_cached_devices> cache{};

    if(hip_device < 0)
        return false;
    const void* anchor = reinterpret_cast<const void*>(&device_code_anchor);
    if(hip_device >= max_cached_devices)
    {
        const auto found = kernel_loads_on(anchor, hip_device);
        return found.has_value() && *found;
    }
    auto& slot = cache[static_cast<std::size_t>(hip_device)];
    if(const auto known = slot.load(std::memory_order_relaxed); known != unknown)
        return known == present;
    // A failed HIP call says nothing about the code object, so it is not cached.
    const auto found = kernel_loads_on(anchor, hip_device);
    if(!found.has_value())
        return false;
    signed char known = unknown;
    if(slot.compare_exchange_strong(known, *found ? present : absent, std::memory_order_relaxed))
        return *found;
    return known == present;
}

std::vector<ConvKernelHandle> get_valid_configs(const Device& device,
                                                const ConvParams& par,
                                                Algorithm algo,
                                                std::size_t max_ranked)
{
    check_device(device);
    for(const auto& entry : device.arch->algorithms)
    {
        if(entry.algorithm == algo)
            return handles(entry.impl->get_valid_configs(device, par, max_ranked));
    }
    return {};
}

std::vector<ConvKernelHandle>
get_valid_configs(const Device& device, const ConvParams& par, std::size_t max_ranked)
{
    check_device(device);
    std::vector<ScoredKernel> scored;
    for(const auto& entry : device.arch->algorithms)
    {
        auto cfgs = entry.impl->get_valid_configs(device, par, max_ranked);
        scored.insert(scored.end(), cfgs.begin(), cfgs.end());
    }
    keep_top_ranked(scored, max_ranked);
    return handles(scored);
}

std::optional<ConvKernelHandle> find_config(const Device& device, const ConvParams& par)
{
    auto cfgs = get_valid_configs(device, par);
    if(cfgs.empty())
        return std::nullopt;
    return cfgs.front();
}

namespace
{
// Applies get_valid_configs' filter to one kernel, for is_applicable and ConvLaunch::make.
bool kernel_supports(const ConvKernel* kernel, const Device& device, const ConvParams& par)
{
    const auto algorithm_supports = [&] {
        for(const auto& entry : device.arch->algorithms)
            if(entry.algorithm == kernel->algorithm())
                return entry.impl->is_applicable(par);
        return false;
    };
    return algorithm_supports() && kernel->supports_dims(par.dims) &&
           kernel->is_applicable(device, par) && kernel->is_valid_config(device, par);
}

void check_kernel(ConvKernelHandle kernel, const Device& device)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    check_device(device);
}
} // namespace

bool is_applicable(ConvKernelHandle kernel, const Device& device, const ConvParams& par)
{
    check_kernel(kernel, device);
    return kernel_supports(kernel, device, par);
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

std::string describe_config(ConvKernelHandle kernel, bool full)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->describe_config(full);
}

std::string describe_kernel(ConvKernelHandle kernel)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    std::string description(kernel->name());
    const std::string config = kernel->describe_config(true);
    if(!config.empty())
        description += "[" + config + "]";
    return description;
}

auto split_kernel_description(std::string_view desc)
    -> std::pair<std::string_view, std::string_view>
{
    char const* begin  = desc.data();
    char const* cursor = begin;
    char const* lim    = desc.data() + desc.size();

    while(cursor < lim && *cursor != '[' && *cursor != ']')
        ++cursor;
    if(cursor == lim)
        return {desc, ""};
    char const* lbr_pos = cursor;
    char const* rbr_pos = cursor + (lim - cursor - 1);
    if(lbr_pos > begin && *lbr_pos == '[' && *rbr_pos == ']' && rbr_pos - lbr_pos > 1)
        return {std::string_view{begin, lbr_pos}, std::string_view{lbr_pos + 1, rbr_pos}};

    return {"", ""};
}

std::optional<ConvKernelHandle> find_kernel(ArchHandle arch, std::string_view description)
{
    if(!arch)
        throw std::invalid_argument("null arch handle");
    const std::less<> before;
    if(before(arch, hipconv_arch_registry) ||
       !before(arch, hipconv_arch_registry + hipconv_arch_registry_size))
        throw std::invalid_argument("arch handle not from find_arch");

    const auto [family_name, config] = split_kernel_description(description);
    if(family_name.empty())
        return std::nullopt;

    const auto reprhash = RepresentationHashPair(config);
    for(const auto& entry : arch->algorithms)
    {
        for(const ConvKernelGroup* group : entry.impl->kernel_groups())
        {
            if(!group->kernels.empty() && group->kernels.front()->name() == family_name)
            {
                auto kernel = group->get_kernel_by_config(reprhash);
                if(kernel)
                    return kernel;
            }
        }
    }
    return std::nullopt;
}

bool matches_descriptor(ConvKernelHandle kernel, std::string_view spec, std::string* error)
{
    if(!kernel)
        throw std::invalid_argument("null kernel");
    return kernel->matches_descriptor(spec, error);
}

size_t get_workspace_size(ConvKernelHandle kernel, const Device& device, const ConvParams& par)
{
    check_kernel(kernel, device);
    return kernel->get_workspace_size(device, par);
}

bool is_deterministic(ConvKernelHandle kernel, const Device& device, const ConvParams& par)
{
    check_kernel(kernel, device);
    return kernel->is_deterministic(device, par);
}

float get_weighted_throughput_index(ConvKernelHandle kernel,
                                    const Device& device,
                                    const ConvParams& par)
{
    check_kernel(kernel, device);
    return kernel->get_weighted_throughput_index(device, par);
}

hipconvError_t launch(ConvKernelHandle kernel,
                      const Device& device,
                      const ConvParams& par,
                      const void* in,
                      const void* wei,
                      void* out,
                      void* workspace,
                      hipStream_t stream)
{
    if(!kernel || !device.arch || device.cu_count <= 0)
        return hipErrorInvalidValue;
    // Internally hipconv throws on error; the public API translates that into
    // a return code. Execution faults are asynchronous and surface at a later
    // synchronization, not here.
    try
    {
        kernel->launch(
            kernel->get_launch_params(device, par), device, par, in, wei, out, workspace, stream);
    }
    catch(const HipError& e)
    {
        return e.code;
    }
    return hipSuccess;
}

void get_tolerance(ConvKernelHandle kernel,
                   const Device& device,
                   const ConvParams& par,
                   float& atol,
                   float& rtol)
{
    check_kernel(kernel, device);
    kernel->get_tolerance(device, par, atol, rtol);
}

void get_recursive_summation_tolerance(const ConvParams& par, float& atol, float& rtol)
{
    get_mixed_precision_tolerance(par, atol, rtol);
}

struct ConvLaunch::State
{
    ConvKernel* kernel;
    Device device;
    ConvParams par;
    LaunchParams lp;
    size_t workspace_size;
};

std::optional<ConvLaunch>
ConvLaunch::make(ConvKernelHandle kernel, const Device& device, ConvParams par)
{
    check_kernel(kernel, device);
    if(!kernel_supports(kernel, device, par))
        return std::nullopt;

    auto lp = kernel->get_launch_params(device, par);
    auto ws = kernel->get_workspace_size(device, par);
    auto st = std::make_unique<State>(State{kernel, device, std::move(par), lp, ws});
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
    state_->kernel->get_tolerance(state_->device, state_->par, atol, rtol);
}

const Device& ConvLaunch::device() const noexcept
{
    return state_->device;
}

const ConvParams& ConvLaunch::params() const noexcept
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
        state_->kernel->launch(
            state_->lp, state_->device, state_->par, in, wei, out, workspace, stream);
    }
    catch(const HipError& e)
    {
        return e.code;
    }
    return hipSuccess;
}

} // namespace hipconv
