// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/DefaultPersistentLaunch.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <origami/hardware.hpp>
#include <rocisa/include/enum.hpp>
#include <tilewright/model.hpp>
#include <tilewright/types.hpp>

namespace TensileLite
{
    inline tilewright::DataType datatypeToTilewrightDatatype(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return tilewright::DataType::Float;
        case rocisa::DataType::Double:
            return tilewright::DataType::Double;
        case rocisa::DataType::ComplexFloat:
            return tilewright::DataType::ComplexFloat;
        case rocisa::DataType::ComplexDouble:
            return tilewright::DataType::ComplexDouble;
        case rocisa::DataType::Half:
            return tilewright::DataType::Half;
        case rocisa::DataType::Int8x4:
            return tilewright::DataType::Int8x4;
        case rocisa::DataType::Int32:
            return tilewright::DataType::Int32;
        case rocisa::DataType::BFloat16:
            return tilewright::DataType::BFloat16;
        case rocisa::DataType::Int8:
            return tilewright::DataType::Int8;
        case rocisa::DataType::Int64:
            return tilewright::DataType::Int64;
        case rocisa::DataType::XFloat32:
            return tilewright::DataType::XFloat32;
        case rocisa::DataType::Float8_fnuz:
            return tilewright::DataType::Float8_fnuz;
        case rocisa::DataType::BFloat8_fnuz:
            return tilewright::DataType::BFloat8_fnuz;
        case rocisa::DataType::Float8BFloat8_fnuz:
            return tilewright::DataType::Float8BFloat8_fnuz;
        case rocisa::DataType::BFloat8Float8_fnuz:
            return tilewright::DataType::BFloat8Float8_fnuz;
        case rocisa::DataType::Float8:
            return tilewright::DataType::Float8;
        case rocisa::DataType::BFloat8:
            return tilewright::DataType::BFloat8;
        case rocisa::DataType::Float8BFloat8:
            return tilewright::DataType::Float8BFloat8;
        case rocisa::DataType::BFloat8Float8:
            return tilewright::DataType::BFloat8Float8;
        case rocisa::DataType::Float6:
            return tilewright::DataType::Float6;
        case rocisa::DataType::BFloat6:
            return tilewright::DataType::BFloat6;
        case rocisa::DataType::Float4:
            return tilewright::DataType::Float4;
        default:
            return tilewright::DataType::None;
        }
    }

    inline rocisa::DataType tilewrightToRocisaDatatype(tilewright::DataType type)
    {
        switch(type)
        {
        case tilewright::DataType::Float:
            return rocisa::DataType::Float;
        case tilewright::DataType::Double:
            return rocisa::DataType::Double;
        case tilewright::DataType::ComplexFloat:
            return rocisa::DataType::ComplexFloat;
        case tilewright::DataType::ComplexDouble:
            return rocisa::DataType::ComplexDouble;
        case tilewright::DataType::Half:
            return rocisa::DataType::Half;
        case tilewright::DataType::Int8x4:
            return rocisa::DataType::Int8x4;
        case tilewright::DataType::Int32:
            return rocisa::DataType::Int32;
        case tilewright::DataType::BFloat16:
            return rocisa::DataType::BFloat16;
        case tilewright::DataType::Int8:
            return rocisa::DataType::Int8;
        case tilewright::DataType::Int64:
            return rocisa::DataType::Int64;
        case tilewright::DataType::XFloat32:
            return rocisa::DataType::XFloat32;
        case tilewright::DataType::Float8_fnuz:
            return rocisa::DataType::Float8_fnuz;
        case tilewright::DataType::BFloat8_fnuz:
            return rocisa::DataType::BFloat8_fnuz;
        case tilewright::DataType::Float8BFloat8_fnuz:
            return rocisa::DataType::Float8BFloat8_fnuz;
        case tilewright::DataType::BFloat8Float8_fnuz:
            return rocisa::DataType::BFloat8Float8_fnuz;
        case tilewright::DataType::Float8:
            return rocisa::DataType::Float8;
        case tilewright::DataType::BFloat8:
            return rocisa::DataType::BFloat8;
        case tilewright::DataType::Float8BFloat8:
            return rocisa::DataType::Float8BFloat8;
        case tilewright::DataType::BFloat8Float8:
            return rocisa::DataType::BFloat8Float8;
        case tilewright::DataType::Float6:
            return rocisa::DataType::Float6;
        case tilewright::DataType::BFloat6:
            return rocisa::DataType::BFloat6;
        case tilewright::DataType::Float4:
            return rocisa::DataType::Float4;
        default:
            return rocisa::DataType::None;
        }
    }

    inline std::int64_t tilewrightAttributeValue(TileProcessingStrategy strategy)
    {
        switch(strategy)
        {
        case TileProcessingStrategy::None:
            return 0;
        case TileProcessingStrategy::DataParallel:
            return 1;
        case TileProcessingStrategy::StreamK:
            return 2;
        }
        throw std::runtime_error("Invalid TileProcessingStrategy");
    }

    inline std::int64_t tilewrightAttributeValue(WorkAssignment assignment)
    {
        switch(assignment)
        {
        case WorkAssignment::StaticGrid:
            return 0;
        case WorkAssignment::DynamicWorkQueue:
            return 1;
        case WorkAssignment::Hybrid:
            return 2;
        }
        throw std::runtime_error("Invalid WorkAssignment");
    }

    // Asks for every cache line of the `bytes` bytes at `p` without waiting
    // for any. A prefetch never faults and changes no value.
    inline void prefetchLines(void const* p, size_t bytes) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        if(p == nullptr || bytes == 0)
            return;
        auto const end = reinterpret_cast<std::uintptr_t>(p) + bytes;
        for(auto at = reinterpret_cast<std::uintptr_t>(p) & ~std::uintptr_t{63}; at < end; at += 64)
            __builtin_prefetch(reinterpret_cast<void const*>(at));
#else
        (void)p;
        (void)bytes;
#endif
    }

    /**
     * Storage from tilewright::allocate_huge_pages, on transparent huge pages
     * where the system has them: for what a ranking reads on every call.
     */
    template <typename T>
    struct HugePageAllocator
    {
        using value_type = T;

        HugePageAllocator() = default;
        template <typename U>
        HugePageAllocator(HugePageAllocator<U> const&) noexcept
        {
        }

        T* allocate(size_t n)
        {
            return static_cast<T*>(tilewright::allocate_huge_pages(n * sizeof(T)));
        }

        void deallocate(T* p, size_t n) noexcept
        {
            tilewright::free_huge_pages(p, n * sizeof(T));
        }

        template <typename U>
        bool operator==(HugePageAllocator<U> const&) const noexcept
        {
            return true;
        }

        template <typename U>
        bool operator!=(HugePageAllocator<U> const&) const noexcept
        {
            return false;
        }
    };

    /**
     * The problem and device of the ranking in progress on this thread. The
     * launch plans that ranking asks for are those of this problem on this
     * device.
     */
    template <typename MyProblem>
    class TilewrightLaunchScope
    {
    public:
        TilewrightLaunchScope(MyProblem const& problem, Hardware const& hardware)
            : m_problem(problem)
            , m_hardware(hardware)
            , m_id(++counter())
            , m_previous(current())
        {
            current() = this;
        }

        ~TilewrightLaunchScope()
        {
            current() = m_previous;
        }

        TilewrightLaunchScope(TilewrightLaunchScope const&)            = delete;
        TilewrightLaunchScope& operator=(TilewrightLaunchScope const&) = delete;

        static TilewrightLaunchScope const* active() noexcept
        {
            return current();
        }

        MyProblem const& problem() const noexcept
        {
            return m_problem;
        }

        Hardware const& hardware() const noexcept
        {
            return m_hardware;
        }

        // Distinct for every scope of the thread.
        std::uint64_t id() const noexcept
        {
            return m_id;
        }

    private:
        static TilewrightLaunchScope*& current() noexcept
        {
            static thread_local TilewrightLaunchScope* scope = nullptr;
            return scope;
        }

        static std::uint64_t& counter() noexcept
        {
            static thread_local std::uint64_t count = 0;
            return count;
        }

        MyProblem const&       m_problem;
        Hardware const&        m_hardware;
        std::uint64_t          m_id;
        TilewrightLaunchScope* m_previous;
    };

    /**
     * TensileLite's launch of the kernels of a tilewright pool (pool position i
     * is solutions[i]), as the plans that MLREC_v3 models read: the grid, the
     * work assignment and the partial-tile reduction resolvePersistentSettings()
     * launches with, for the problem under the execution context, the default
     * persistent settings of the device, no uniform summation order and a
     * workspace that holds every partial tile. The device is the one of the
     * TilewrightLaunchScope of the thread, else the one given here.
     *
     * When followsRule() holds for the kernel and the device, plan() hands
     * what it recorded of each kernel to resolveDefaultPersistentLaunch(), the
     * function resolvePersistentSettings() decides that launch with; otherwise
     * it calls resolvePersistentSettings(), as resolvedPlan() does.
     */
    template <typename MySolution>
    class TilewrightLaunch : public tilewright::LaunchModel
    {
    public:
        using MyProblem = typename MySolution::Problem;
        using Scope     = TilewrightLaunchScope<MyProblem>;

        // RULE_VERSION of training_pipeline/lib/hipblaslt_launch.py, whose
        // launch_plan() returns the plans this class returns but in one
        // class: a static Stream-K launch of batch 1 with one K iteration per
        // tile and more tiles than CUs takes the grid Origami's
        // select_grid_size() chooses (rulePlan() through
        // resolveDefaultPersistentLaunch()), which can differ from rule
        // version 1's; the pipeline's scripts/check_launch_plans.py counts
        // such plans as a known class.
        static constexpr int RuleVersion = 1;

        TilewrightLaunch(std::string                              arch,
                         std::vector<std::shared_ptr<MySolution>> solutions,
                         std::shared_ptr<Hardware const>          device = nullptr)
            : m_arch(std::move(arch))
            , m_solutions(std::move(solutions))
            , m_device(std::move(device))
        {
            m_kernels.reserve(m_solutions.size());
            for(auto const& solution : m_solutions)
                m_kernels.push_back(kernelLaunch(*solution));
        }

        std::string id() const override
        {
            return "hipblaslt:" + m_arch + ":" + std::to_string(RuleVersion);
        }

        std::uint64_t launch_key(tilewright::Config const& config) const override
        {
            checkIndex(config.index);
            return m_kernels[config.index].key;
        }

        // launch_key() of training_pipeline/lib/hipblaslt_launch.py: FNV-1a 64
        // over the 8-byte little-endian launch inputs, followed by a sixth
        // value, 1, for a tree-only kernel (treeOnly()).
        static std::uint64_t launchKey(std::array<std::int64_t, 5> const& inputs,
                                       bool                               treeOnly = false)
        {
            std::uint64_t hash = 0xcbf29ce484222325ull;
            auto const    add  = [&hash](std::int64_t value) {
                auto const bits = static_cast<std::uint64_t>(value);
                for(int byte = 0; byte < 8; ++byte)
                {
                    hash ^= (bits >> (8 * byte)) & 0xffu;
                    hash *= 0x100000001b3ull;
                }
            };
            for(std::int64_t const value : inputs)
                add(value);
            if(treeOnly)
                add(1);
            return hash;
        }

        // Whether `solution` is a hand-written Stream-K kernel that can run on
        // a static grid (any work assignment but DynamicWorkQueue): its
        // customKernel names a kernel that was not generated, as kernelLaunch()
        // and ContractionSolution::handwrittenCustomKernel() tell it.
        // resolvePersistentSettings() reduces its partial tiles in a tree only,
        // so its plans can differ from those of a generated kernel with equal
        // launch inputs; launch_inputs() of hipblaslt_launch.py follows its
        // five values with a sixth, 1, that makes it a launch class of its own.
        static bool treeOnly(MySolution const& solution)
        {
            auto const& sm = solution.sizeMapping;
            return !solution.customKernel.name.empty() && !solution.customKernel.generated
                   && sm.isStreamK() && sm.workAssignment != WorkAssignment::DynamicWorkQueue;
        }

        // launch_inputs() of hipblaslt_launch.py without its sixth value:
        // (strategy, assignment, atomic, threads, occupancy), with the values
        // plan() does not tell apart made equal.
        static std::array<std::int64_t, 5> launchInputs(SizeMapping const& sizeMapping)
        {
            std::int64_t const strategy
                = tilewrightAttributeValue(sizeMapping.tileProcessingStrategy);
            if(strategy != 1 && strategy != 2)
                return {0, 0, 0, 0, 0};
            std::int64_t threads = 1;
            for(auto const size : {sizeMapping.workGroupSize.x,
                                   sizeMapping.workGroupSize.y,
                                   sizeMapping.workGroupSize.z})
                threads *= std::max<std::int64_t>(size, 0);
            bool const atomic = sizeMapping.streamKAtomic != 0 && strategy == 2;
            return {strategy,
                    tilewrightAttributeValue(sizeMapping.workAssignment),
                    atomic ? 1 : 0,
                    threads,
                    std::max<std::int64_t>(sizeMapping.CUOccupancy, 1)};
        }

        // tilewright's re-rank memo (TW_LEVER_MEMO) may answer a ranking with
        // the scores of the thread's last one when both rank this pool for
        // equal tilewright inputs (problem, hardware, scoring context), which
        // assumes that plan() returns equal plans for both. plan() also reads,
        // of the scope's device, its persistent settings, its CU count and its
        // whole analytical hardware, of which the tilewright hardware holds
        // only the CU count and the LDS and L2 sizes, and, for a kernel it does
        // not plan from its record, the scope's whole problem. TensileLite gives
        // every device it builds the persistent settings it reads once per
        // process, so only devices whose CU count or analytical hardware
        // differ beyond what the tilewright hardware holds can plan
        // differently for equal inputs. A remembered ranking only orders
        // candidates that TensileLite's predicates still check against the
        // call's problem: at worst a call serves another valid kernel than a
        // fresh ranking would. Where the memo's plan check (TW_LEVER_MEMO_CHECK)
        // is compiled, by default in builds with assertions, it asks for the
        // plans again and ranks afresh when one differs.
        tilewright::LaunchPlan plan(tilewright::Config const&           config,
                                    tilewright::Problem const&          problem,
                                    tilewright::ExecutionContext const& context) const override
        {
            checkIndex(config.index);
            Scope const*    scope    = Scope::active();
            Hardware const& hardware = deviceOf(scope);
            auto const&     kernel   = m_kernels[config.index];
            if(kernel.rule)
            {
                if(!kernel.persistent)
                    return {.dynamic_queue = false, .grid_wgs = outputTiles(kernel, problem)};
                DeviceLaunch const device
                    = scope ? scopeDeviceLaunch(*scope) : deviceLaunch(hardware);
                if(device.rule)
                    return rulePlan(kernel, device, problem, context);
            }
            return resolvedPlan(config.index, scope, hardware, problem, context);
        }

        // Every line of the record plan() reads of the config's kernel
        // (LaunchModel::prefetch); nothing for a pool position plan() would
        // refuse.
        void prefetch(tilewright::Config const& config) const noexcept override
        {
            if(config.index < m_kernels.size())
                prefetchLines(&m_kernels[config.index], sizeof(KernelLaunch));
        }

        // The plan resolvePersistentSettings() launches with, which plan()
        // returns whatever followsRule() says.
        tilewright::LaunchPlan resolvedPlan(tilewright::Config const&           config,
                                            tilewright::Problem const&          problem,
                                            tilewright::ExecutionContext const& context) const
        {
            checkIndex(config.index);
            Scope const* scope = Scope::active();
            return resolvedPlan(config.index, scope, deviceOf(scope), problem, context);
        }

        // Whether plan() evaluates the launch of the kernel at pool position
        // `index` on `hardware` from its record rather than through
        // resolvePersistentSettings(). It does unless the kernel is laid out
        // in a way the record does not hold (handwritten, packed batch, a
        // persistent kernel with a cluster, an execution policy TensileLite
        // rejects) or the launch reads a setting other than its default: a
        // fixed grid, a grid selection other than k_split_aware or a CU cap
        // among the device's persistent settings, or a debug setting of
        // defaultPersistentLaunchDebug().
        bool followsRule(size_t index, Hardware const& hardware) const
        {
            checkIndex(index);
            auto const& kernel = m_kernels[index];
            return kernel.rule && (!kernel.persistent || deviceLaunch(hardware).rule);
        }

    private:
        // What plan() and launch_key() read of a kernel. The fields from
        // config to name hold only when `rule` does, and config only for a
        // persistent kernel.
        struct KernelLaunch
        {
            std::uint64_t     key         = 0;
            origami::config_t config      = {};
            std::uint32_t     tileM       = 0;
            std::uint32_t     tileN       = 0;
            std::uint32_t     depthU      = 0;
            std::uint32_t     threads     = 1;
            int               occupancy   = 0;
            WorkAssignment    assignment  = WorkAssignment::StaticGrid;
            char const*       name        = "";
            bool              persistent  = false;
            bool              batchInGrid = false;
            bool              streamK     = false;
            bool              atomic      = false;
            bool              rule        = false;
        };

        // What plan() reads of the device and of the debug settings.
        struct DeviceLaunch
        {
            bool                       rule         = false;
            size_t                     computeUnits = 0;
            origami::hardware_t const* analytical   = nullptr;
        };

        static KernelLaunch kernelLaunch(MySolution const& solution)
        {
            auto const&  sm = solution.sizeMapping;
            KernelLaunch kernel{.key         = launchKey(launchInputs(sm), treeOnly(solution)),
                                .assignment  = sm.workAssignment,
                                .persistent  = sm.isPersistent(),
                                .batchInGrid = !sm.persistentKernelAlongBatch && !sm.packBatchDims,
                                .streamK     = sm.isStreamK(),
                                .atomic      = sm.streamKAtomic != 0};

            size_t const threads = sm.workGroupSize.x * sm.workGroupSize.y * sm.workGroupSize.z;
            size_t const limit   = std::numeric_limits<std::uint32_t>::max();
            if(sm.macroTile.x == 0 || sm.macroTile.y == 0 || sm.depthU == 0
               || sm.macroTile.x > limit || sm.macroTile.y > limit || sm.depthU > limit
               || threads > limit || sm.packBatchDims != 0
               || (!solution.customKernel.name.empty() && !solution.customKernel.generated))
                return kernel;
            if(kernel.persistent)
            {
                if(!solution.followsDefaultPersistentLaunch())
                    return kernel;
                try
                {
                    solution.validatePersistentLoopArgsVersion();
                    kernel.config = solution.persistentLaunchConfig();
                }
                catch(std::exception const&)
                {
                    return kernel;
                }
            }
            kernel.tileM     = static_cast<std::uint32_t>(sm.macroTile.x);
            kernel.tileN     = static_cast<std::uint32_t>(sm.macroTile.y);
            kernel.depthU    = static_cast<std::uint32_t>(sm.depthU);
            kernel.threads   = static_cast<std::uint32_t>(threads);
            kernel.occupancy = sm.CUOccupancy;
            kernel.name      = solution.kernelName.c_str();
            kernel.rule      = true;
            return kernel;
        }

        // rulePlan() passes the default debug settings, and the CU budget as
        // ContractionProblem::setSmCountTarget() keeps it.
        static DeviceLaunch deviceLaunch(Hardware const& hardware)
        {
            if(!hasDefaultPersistentSettings(hardware) || !defaultPersistentLaunchDebug().unset())
                return {};
            auto const& device = static_cast<hip::HipAMDGPU const&>(hardware);
            return {.rule         = true,
                    .computeUnits = static_cast<size_t>(device.computeUnitCount),
                    .analytical   = device.analyticalHardware.get()};
        }

        // deviceLaunch() of the scope's device, reused by the plans of one
        // ranking.
        static DeviceLaunch scopeDeviceLaunch(Scope const& scope)
        {
            static thread_local std::uint64_t id = 0;
            static thread_local DeviceLaunch  device;
            if(id != scope.id())
            {
                device = deviceLaunch(scope.hardware());
                id     = scope.id();
            }
            return device;
        }

        Hardware const& deviceOf(Scope const* scope) const
        {
            if(scope)
                return scope->hardware();
            if(m_device == nullptr)
                throw std::logic_error("tilewright launch: no device to plan the launch for");
            return *m_device;
        }

        // getNumTiles() of a kernel without packed batch dimensions, with the
        // batch.
        static size_t outputTiles(KernelLaunch const& kernel, tilewright::Problem const& problem)
        {
            return CeilDivide(problem.size.m, size_t{kernel.tileM})
                   * CeilDivide(problem.size.n, size_t{kernel.tileN}) * problem.batch;
        }

        static tilewright::LaunchPlan launchPlan(KernelLaunch const&  kernel,
                                                 WorkAssignment       effectiveWorkAssignment,
                                                 size_t               grid,
                                                 size_t               tiles,
                                                 origami::reduction_t reduction)
        {
            bool const dynamic = effectiveWorkAssignment == WorkAssignment::DynamicWorkQueue;
            tilewright::LaunchPlan plan{.dynamic_queue = dynamic, .grid_wgs = grid};
            if(kernel.streamK && !dynamic && grid > 0 && tiles % grid != 0)
                plan.partial_reduction = kernel.atomic ? tilewright::PartialReduction::Atomic
                                         : reduction == origami::reduction_t::parallel
                                             ? tilewright::PartialReduction::Parallel
                                             : tilewright::PartialReduction::Tree;
            return plan;
        }

        // resolvePersistentSettings() of a persistent kernel whose `rule` holds,
        // on a device whose `rule` holds.
        static tilewright::LaunchPlan rulePlan(KernelLaunch const&                 kernel,
                                               DeviceLaunch const&                 device,
                                               tilewright::Problem const&          problem,
                                               tilewright::ExecutionContext const& context)
        {
            size_t const tiles            = outputTiles(kernel, problem);
            auto const   computeInputType = problem.mi_dtype == tilewright::DataType::XFloat32
                                                ? problem.a_dtype
                                                : problem.mi_dtype;

            DefaultPersistentLaunchInputs inputs;
            inputs.m                = problem.size.m;
            inputs.n                = problem.size.n;
            inputs.k                = problem.size.k;
            inputs.batch            = problem.batch;
            inputs.transA           = problem.a_transpose == tilewright::Transpose::T;
            inputs.transB           = problem.b_transpose == tilewright::Transpose::T;
            inputs.aType            = tilewrightToRocisaDatatype(problem.a_dtype);
            inputs.bType            = tilewrightToRocisaDatatype(problem.b_dtype);
            inputs.computeInputType = tilewrightToRocisaDatatype(computeInputType);
            inputs.tiles            = tiles;
            inputs.itersPerTile     = CeilDivide(problem.size.k, size_t{kernel.depthU});
            inputs.workspaceSize    = std::numeric_limits<size_t>::max();
            inputs.smCountTarget    = static_cast<int>(
                std::min<size_t>(context.cu_budget, std::numeric_limits<int>::max()));
            inputs.schedulingMode = schedulingMode(context.schedule);
            inputs.kernel         = &kernel.config;
            inputs.threads        = kernel.threads;
            inputs.occupancy      = kernel.occupancy;
            inputs.workAssignment = kernel.assignment;
            inputs.streamK        = kernel.streamK;
            inputs.atomic         = kernel.atomic;
            inputs.kernelName     = kernel.name;
            inputs.hardware       = device.analytical;
            inputs.computeUnits   = device.computeUnits;

            auto const launch = resolveDefaultPersistentLaunch(inputs);
            return launchPlan(
                kernel, launch.effectiveWorkAssignment, launch.grid, tiles, launch.reduction);
        }

        tilewright::LaunchPlan resolvedPlan(size_t                              index,
                                            Scope const*                        scope,
                                            Hardware const&                     hardware,
                                            tilewright::Problem const&          problem,
                                            tilewright::ExecutionContext const& context) const
        {
            auto const& solution      = *m_solutions[index];
            auto const& kernel        = m_kernels[index];
            auto const& launchProblem = launchProblemOf(scope, problem, context);
            size_t      tiles         = launchProblem.getNumTiles(solution.sizeMapping, 1);
            if(!kernel.persistent)
            {
                // getNumTiles leaves the batch out of a non-persistent launch,
                // which takes it as the grid's z dimension.
                if(kernel.batchInGrid)
                    for(size_t i = 0; i < launchProblem.batchIndices().size(); ++i)
                        tiles *= launchProblem.batchSize(i);
                return {.dynamic_queue = false, .grid_wgs = tiles};
            }

            auto const launch = solution.resolvePersistentSettings(launchProblem, hardware);
            return launchPlan(kernel,
                              launch.effectiveWorkAssignment,
                              launch.grid,
                              launch.totalTiles,
                              launch.reduction);
        }

        void checkIndex(size_t index) const
        {
            if(index >= m_kernels.size())
                throw std::out_of_range("tilewright launch: no kernel at pool position "
                                        + std::to_string(index));
        }

        static bool sameProblem(tilewright::Problem const& a, tilewright::Problem const& b)
        {
            return a.size.m == b.size.m && a.size.n == b.size.n && a.size.k == b.size.k
                   && a.batch == b.batch && a.a_transpose == b.a_transpose
                   && a.b_transpose == b.b_transpose && a.a_dtype == b.a_dtype
                   && a.b_dtype == b.b_dtype && a.c_dtype == b.c_dtype && a.d_dtype == b.d_dtype;
        }

        static MyProblem makeProblem(tilewright::Problem const& p)
        {
            bool const   transA = p.a_transpose == tilewright::Transpose::T;
            bool const   transB = p.b_transpose == tilewright::Transpose::T;
            size_t const m = p.size.m, n = p.size.n, k = p.size.k;
            size_t const lda         = transA ? k : m;
            size_t const ldb         = transB ? n : k;
            auto const   computeType = tilewrightToRocisaDatatype(
                p.mi_dtype == tilewright::DataType::XFloat32 ? p.a_dtype : p.mi_dtype);
            auto problem = MyProblem::GEMM_Strides(transA,
                                                   transB,
                                                   tilewrightToRocisaDatatype(p.a_dtype),
                                                   tilewrightToRocisaDatatype(p.b_dtype),
                                                   tilewrightToRocisaDatatype(p.c_dtype),
                                                   tilewrightToRocisaDatatype(p.d_dtype),
                                                   m,
                                                   n,
                                                   k,
                                                   p.batch,
                                                   lda,
                                                   lda * (transA ? m : k),
                                                   ldb,
                                                   ldb * (transB ? k : n),
                                                   m,
                                                   m * n,
                                                   m,
                                                   m * n,
                                                   0.0);
            problem.setComputeInputTypeA(computeType);
            problem.setComputeInputTypeB(computeType);
            return problem;
        }

        static int schedulingMode(tilewright::Schedule schedule)
        {
            switch(schedule)
            {
            case tilewright::Schedule::Dynamic:
                return 1;
            case tilewright::Schedule::Auto:
                return 2;
            default:
                return 0;
            }
        }

        // The scope's problem, or one made from `problem`, under `context`;
        // reused by the plans of one ranking.
        static MyProblem const& launchProblemOf(Scope const*                        scope,
                                                tilewright::Problem const&          problem,
                                                tilewright::ExecutionContext const& context)
        {
            struct Cached
            {
                std::uint64_t            scope = 0;
                tilewright::Problem      problem{};
                size_t                   cuBudget = 0;
                tilewright::Schedule     schedule = tilewright::Schedule::Default;
                std::optional<MyProblem> launch;
            };
            static thread_local Cached cached;

            bool const reuse
                = cached.launch && cached.cuBudget == context.cu_budget
                  && cached.schedule == context.schedule
                  && (scope ? cached.scope == scope->id()
                            : cached.scope == 0 && sameProblem(cached.problem, problem));
            if(!reuse)
            {
                if(scope)
                    cached.launch = scope->problem();
                else
                    cached.launch = makeProblem(problem);
                cached.scope    = scope ? scope->id() : 0;
                cached.problem  = problem;
                cached.cuBudget = context.cu_budget;
                cached.schedule = context.schedule;

                auto& launch = *cached.launch;
                launch.setWorkspaceSize(std::numeric_limits<size_t>::max());
                auto& params = launch.setParams();
                params.setUniformSummationOrder(false);
                params.setSmCountTarget(static_cast<int>(
                    std::min<size_t>(context.cu_budget, std::numeric_limits<int>::max())));
                params.setStreamKTileSchedulingMode(schedulingMode(context.schedule));
            }
            return *cached.launch;
        }

        std::string                              m_arch;
        std::vector<std::shared_ptr<MySolution>> m_solutions;
        // On huge pages: plan() reads the records of the tier's kernels on
        // every ranking, scattered over the pool.
        std::vector<KernelLaunch, HugePageAllocator<KernelLaunch>> m_kernels;
        std::shared_ptr<Hardware const>                            m_device;
    };

    /**
     * Ranks the kernels of a Prediction library with the tilewright model that
     * ships for the library. Each config's index is its pool position.
     *
     * A library stem can ship a blocklist next to its tilewright_index,
     * `<stem>.tilewright.blocklist`: the names (ContractionSolution::name()) of
     * kernels that compute wrong results, one per line. The ranker leaves them
     * out of its pool, so a group of configs the model scores as one candidate
     * is served by its first member that is not blocked, and a group whose
     * members are all blocked is no candidate. A name blocks every kernel
     * whose name has its blocklistKey(): builds of TensileLite can spell the
     * name of one kernel differently, and a blocklist written from one build
     * still names the kernel in another.
     */
    class TilewrightRanker
    {
    public:
        // The kernel names a blocklist holds; null for a library without one.
        using Blocklist = std::shared_ptr<const std::unordered_set<std::string>>;

        // Throws std::invalid_argument for a null model, an invalid attribute
        // name, or a launch model other than the one the model was trained with.
        // `blocked` flags the pool positions `blocklist` left out of `configs`,
        // `unresolved` lists its names that match no kernel of the pool, and
        // `library` names the library for warnings.
        TilewrightRanker(tilewright::ModelPtr                           model,
                         std::vector<tilewright::Config>                configs,
                         std::shared_ptr<const tilewright::LaunchModel> launch     = nullptr,
                         std::string                                    arch       = {},
                         Blocklist                                      blocklist  = nullptr,
                         std::vector<bool>                              blocked    = {},
                         std::vector<std::string>                       unresolved = {},
                         std::string                                    library    = {})
            : m_candidates(std::move(model),
                           std::move(configs),
                           tilewright::PoolOptions{.tie_break = tilewright::TieBreak::PoolOrder,
                                                   .launch    = std::move(launch)})
            , m_arch(std::move(arch))
            , m_blocklist(std::move(blocklist))
            , m_blocked(std::move(blocked))
            , m_unresolved(std::move(unresolved))
            , m_library(std::move(library))
        {
        }

        // Looks up the stem of `logicFile` in the tilewright_index of its
        // directory, for the PCI chip ID of indexChipId(). Returns nullptr
        // when no model is listed for the stem, or, with a warning, when the
        // listed model cannot be loaded.
        static tilewright::ModelPtr loadModel(std::string const& logicFile)
        {
            return loadModel(logicFile, indexChipId(logicArch(logicFile)));
        }

        // As above for a device of PCI chip ID `pciChipId` (0: unknown): the
        // stem's row naming that chip ID wins over its row without one.
        static tilewright::ModelPtr loadModel(std::string const& logicFile, std::uint32_t pciChipId)
        {
            if(logicFile.empty())
                return nullptr;

            std::string dir  = ".";
            std::string stem = logicStem(logicFile, &dir);

            std::string error;
            auto        model = tilewright::load_model_by_index(stem, dir, pciChipId, &error);
            if(!model && !error.empty())
                warnUnusable(stem, error);
            if(model && pciChipId != 0 && diagnostics())
                std::cerr << "[TILEWRIGHT_DIAG TENSILELITE] " << stem << " pci_chip_id=0x"
                          << std::hex << pciChipId << std::dec << std::endl;
            return model;
        }

        // The PCI chip ID that chooses among a tilewright_index's rows for
        // the libraries of `arch`. Parts of gfx1250 that load the same
        // library file can still get different models, by chip ID; a
        // CU-count class with a file of its own gets its own stem. On gfx1250
        // it is the chip ID of the calling thread's current device, whose
        // call loads the library; 0, the stem's row without a chip ID, on
        // other architectures (gfx950's chip-specific libraries are files of
        // their own) and whenever HIP cannot tell it.
        //
        // hipBLASLt loads a library file once per process and builds one
        // Hardware per architecture, from the first device of it, so a
        // process whose devices of one architecture differ in chip ID, or in
        // CU count (compute partitions in different modes), ranks every one
        // of them with the model of the device that loaded the library and
        // the CU count of the first device.
        static std::uint32_t indexChipId(std::string const& arch)
        {
            if(arch != "gfx1250")
                return 0;
            int device = -1;
            int chipId = 0;
            if(hipGetDevice(&device) != hipSuccess
               || hipDeviceGetAttribute(&chipId, hipDeviceAttributePciChipId, device) != hipSuccess
               || chipId <= 0 || chipId > std::numeric_limits<std::uint16_t>::max())
                return 0;
            return static_cast<std::uint32_t>(chipId);
        }

        // The blocklist of the stem of `logicFile` in the directory of its
        // tilewright_index: every line but blank ones and those that start with
        // '#', without surrounding blanks. Null when there is no such file;
        // throws std::runtime_error when it cannot be read.
        static Blocklist loadBlocklist(std::string const& logicFile)
        {
            std::string           dir  = ".";
            std::string const     stem = logicStem(logicFile, &dir);
            std::filesystem::path path = std::filesystem::path(dir) / (stem + BlocklistSuffix);
            std::error_code       ec;
            if(!std::filesystem::exists(path, ec))
                return nullptr;
            std::ifstream in(path);
            if(!std::filesystem::is_regular_file(path, ec) || !in)
                throw std::runtime_error("cannot read the blocklist " + path.string());
            auto        names = std::make_shared<std::unordered_set<std::string>>();
            std::string line;
            while(std::getline(in, line))
            {
                size_t const first = line.find_first_not_of(" \t\r");
                if(first == std::string::npos || line[first] == '#')
                    continue;
                names->insert(line.substr(first, line.find_last_not_of(" \t\r") + 1 - first));
            }
            if(in.bad())
                throw std::runtime_error("cannot read the blocklist " + path.string());
            return names;
        }

        // A ranker over `poolSize` kernels of `arch` whose kernel i is
        // solutionAt(i), without the kernels `blocklist` names (blocklistKey()),
        // with its per-cell data built now when `prepare` is set and
        // TENSILE_TILEWRIGHT_PREPARE does not turn that off. `library` names the
        // library in warnings. Throws what the CandidateSet throws.
        template <typename SolutionAt>
        static std::shared_ptr<const TilewrightRanker> make(tilewright::ModelPtr model,
                                                            size_t               poolSize,
                                                            SolutionAt&&         solutionAt,
                                                            std::string const&   arch,
                                                            bool                 prepare = true,
                                                            Blocklist          blocklist = nullptr,
                                                            std::string const& library   = {})
        {
            using MySolution = typename std::decay_t<decltype(*solutionAt(0))>;

            auto const                      names = tilewright::attribute_names(*model);
            std::vector<tilewright::Config> configs;
            std::vector<bool>               blocked;
            // The keys of the blocklist's names, each with whether it matched a
            // kernel of the pool.
            std::unordered_map<std::string, bool> keys;
            if(blocklist)
            {
                for(auto const& name : *blocklist)
                    keys.emplace(blocklistKey(name), false);
            }
            configs.reserve(poolSize);
            for(size_t i = 0; i < poolSize; i++)
            {
                if(!keys.empty())
                {
                    auto const key = keys.find(blocklistKey(solutionAt(i)->name()));
                    if(key != keys.end())
                    {
                        key->second = true;
                        blocked.resize(poolSize);
                        blocked[i] = true;
                        continue;
                    }
                }
                auto const& solution = solutionAt(i);
                configs.push_back(makeConfig(solution->sizeMapping, solution->name(), i, &names));
            }
            std::vector<std::string> unresolved;
            if(blocklist)
            {
                for(auto const& name : *blocklist)
                {
                    if(!keys.at(blocklistKey(name)))
                        unresolved.push_back(name);
                }
                std::sort(unresolved.begin(), unresolved.end());
            }

            std::shared_ptr<const tilewright::LaunchModel> launch;
            if(tilewright::describe(*model).format_version >= 3)
            {
                std::vector<std::shared_ptr<MySolution>> solutions;
                solutions.reserve(poolSize);
                for(size_t i = 0; i < poolSize; i++)
                    solutions.push_back(solutionAt(i));
                launch = std::make_shared<const TilewrightLaunch<MySolution>>(arch,
                                                                              std::move(solutions));
            }

            auto ranker = std::make_shared<const TilewrightRanker>(std::move(model),
                                                                   std::move(configs),
                                                                   std::move(launch),
                                                                   arch,
                                                                   std::move(blocklist),
                                                                   std::move(blocked),
                                                                   std::move(unresolved),
                                                                   library);
            if(prepare && Debug::Instance().tilewrightPrepare())
            {
                auto const start = std::chrono::steady_clock::now();
                ranker->m_candidates.prepare();
                if(Debug::Instance().printDataInit())
                    std::cout << "tilewright: prepared " << poolSize << " kernels in "
                              << std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - start)
                                     .count()
                              << " ms" << std::endl;
            }
            return ranker;
        }

        // Looks up the stem of `logicFile` in the tilewright_index of its
        // directory; pool position i is solutions[i], unless the stem's
        // blocklist names it. Returns nullptr when no model is listed for the
        // stem, or, with a warning, when the listed model or the blocklist
        // cannot be used.
        template <typename MySolution>
        static std::shared_ptr<const TilewrightRanker>
            load(std::string const&                                              logicFile,
                 std::vector<std::pair<int, std::shared_ptr<MySolution>>> const& solutions)
        {
            return load(loadModel(logicFile), logicFile, solutions);
        }

        // As above, for the model loadModel(logicFile) returned, prepared as
        // make() prepares.
        template <typename MySolution>
        static std::shared_ptr<const TilewrightRanker>
            load(tilewright::ModelPtr                                            model,
                 std::string const&                                              logicFile,
                 std::vector<std::pair<int, std::shared_ptr<MySolution>>> const& solutions,
                 bool                                                            prepare = true)
        {
            if(!model)
                return nullptr;

            std::string error;
            try
            {
                return make(
                    std::move(model),
                    solutions.size(),
                    [&](size_t i) -> std::shared_ptr<MySolution> const& {
                        return solutions[i].second;
                    },
                    logicArch(logicFile),
                    prepare,
                    loadBlocklist(logicFile),
                    logicStem(logicFile));
            }
            catch(std::exception const& e)
            {
                error = e.what();
            }
            catch(...)
            {
                error = "unknown exception";
            }
            warnUnusable(logicStem(logicFile), error);
            return nullptr;
        }

        // Whether a library ranked with `model` also ranks its problem type's
        // Equality kernels: as TENSILE_TILEWRIGHT_BANK says, or, when it is
        // unset, when the model was trained on them or reads kernel
        // attributes, which the Equality variants of one tile need to be told
        // apart.
        static bool ranksBank(tilewright::Model const& model)
        {
            int const setting = Debug::Instance().tilewrightBank();
            if(setting >= 0)
                return setting == 1;
            return tilewright::describe(model).bank || !tilewright::attribute_names(model).empty();
        }

        // The TDMInst a solution name spells, its `_TDMI<n>_` token (decimal
        // digits followed by '_' or the end of the name): the bit mask of the
        // operands the tensor data mover loads, bit 0 A and bit 1 B. 0 when
        // the name spells none, or several different values. The training
        // pipeline reads it alike (training_pipeline/lib/dat.py).
        static int tdmInst(std::string_view solutionName)
        {
            constexpr std::string_view token = "_TDMI";
            int                        found = -1;
            for(size_t at = solutionName.find(token); at != std::string_view::npos;
                at        = solutionName.find(token, at + 1))
            {
                size_t end   = at + token.size();
                int    value = 0;
                while(end < solutionName.size() && solutionName[end] >= '0'
                      && solutionName[end] <= '9')
                {
                    // Values past 4 bits name no operand; keep them bounded.
                    value = std::min(value * 10 + (solutionName[end] - '0'), 1 << 16);
                    ++end;
                }
                if(end == at + token.size()
                   || (end < solutionName.size() && solutionName[end] != '_'))
                    continue;
                if(found >= 0 && found != value)
                    return 0;
                found = value;
            }
            return std::max(found, 0);
        }

        // Attaches the attributes listed in `attributeNames`, or all of them
        // when it is null. `solutionName` gives TDMInst (tdmInst()).
        static tilewright::Config makeConfig(SizeMapping const&              sizeMapping,
                                             std::string_view                solutionName,
                                             size_t                          index,
                                             std::vector<std::string> const* attributeNames
                                             = nullptr)
        {
            tilewright::Dim3 const mtSize{
                sizeMapping.macroTile.x, sizeMapping.macroTile.y, sizeMapping.depthU};

            auto const& mi = sizeMapping.matrixInstruction;

            // Same stand-in for dot2 kernels as the origami config.
            tilewright::Dim3 miSize{1, 1, 64};
            if(mi[0] != 0 || mi[1] != 0 || mi[2] != 0)
                miSize = {static_cast<size_t>(mi[0]),
                          static_cast<size_t>(mi[1]),
                          static_cast<size_t>(mi[2])};

            // TensileLite's cache hint ignores TDMInst, which SizeMapping does
            // not hold. A cache hint does not reach the code of an operand the
            // tensor data mover loads, so its cache hint is 0, as the training
            // pipeline reads it (training_pipeline/lib/dat.py); the attributes
            // keep the hints as the logic file gives them.
            int const tdm = tdmInst(solutionName);

            return {
                .mt            = mtSize,
                .mi            = miSize,
                .occupancy     = std::max(sizeMapping.CUOccupancy, 1),
                .cache_hints_a = (tdm & 1) != 0 ? 0 : sizeMapping.cacheHintA(),
                .cache_hints_b = (tdm & 2) != 0 ? 0 : sizeMapping.cacheHintB(),
                .grvw_a        = sizeMapping.grvwA,
                .grvw_b        = sizeMapping.grvwB,
                .gwvw_d        = sizeMapping.gwvwD,
                .index         = index,
                .attributes    = makeAttributes(sizeMapping, attributeNames),
            };
        }

        template <typename MyProblem>
        static tilewright::ExecutionContext executionContext(MyProblem const& problem)
        {
            int const smCountTarget = problem.getParams().smCountTarget();
            return {
                .cu_budget = smCountTarget > 0 ? static_cast<size_t>(smCountTarget) : 0,
                .schedule  = schedule(problem),
            };
        }

        // Pool positions of the numSolutions kernels (every kernel when
        // negative) tilewright ranks first for `problem` on `device`, best
        // first; with `everyScored`, every kernel the same ranking scores. A
        // call whose CU budget is outside the model's range is ranked at the
        // nearest budget in it (tilewright::scoring_context) and still runs
        // with its own budget; a call with a schedule the model does not list
        // gets none.
        template <typename MyProblem>
        std::vector<size_t> rank(MyProblem const&      problem,
                                 hip::HipAMDGPU const& device,
                                 int                   numSolutions,
                                 bool                  everyScored = false) const
        {
            size_t m     = 1;
            size_t n     = 1;
            size_t k     = 1;
            size_t batch = 1;
            for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
                m *= problem.freeSizeA(i);
            for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
                n *= problem.freeSizeB(i);
            for(size_t i = 0; i < problem.boundIndices().size(); ++i)
                k *= problem.boundSize(i);
            for(size_t i = 0; i < problem.batchIndices().size(); ++i)
                batch *= problem.batchSize(i);

            tilewright::Problem const tilewrightProblem = {
                .size  = {m, n, k},
                .batch = batch,
                .a_transpose
                = problem.transA() ? tilewright::Transpose::T : tilewright::Transpose::N,
                .b_transpose
                = problem.transB() ? tilewright::Transpose::T : tilewright::Transpose::N,
                .a_dtype  = datatypeToTilewrightDatatype(problem.a().dataType()),
                .b_dtype  = datatypeToTilewrightDatatype(problem.b().dataType()),
                .c_dtype  = datatypeToTilewrightDatatype(problem.c().dataType()),
                .d_dtype  = datatypeToTilewrightDatatype(problem.d().dataType()),
                .mi_dtype = problem.f32XdlMathOp() == rocisa::DataType::XFloat32
                                ? tilewright::DataType::XFloat32
                                : datatypeToTilewrightDatatype(problem.computeInputTypeA()),
            };
            auto const&                hardware           = *device.analyticalHardware;
            // All of it, once per ranking: the launch plans read it (origami's
            // grid and reduction selection in rulePlan()), after the records
            // the engine asks for ahead of them (TilewrightLaunch::prefetch).
            prefetchLines(&hardware, sizeof(hardware));
            tilewright::Hardware const tilewrightHardware = {
                .N_CU         = hardware.N_CU,
                .lds_capacity = hardware.lds_capacity,
                .L2_capacity  = hardware.L2_capacity,
            };
            size_t const depth   = numSolutions < 0 ? std::numeric_limits<size_t>::max()
                                                    : static_cast<size_t>(numSolutions);
            auto const   context = executionContext(problem);

            static thread_local std::vector<tilewright::Result> ranked;
            std::vector<size_t>                                 positions;
            try
            {
                TilewrightLaunchScope<MyProblem> const scope(problem, device);
                if(everyScored)
                    ranked
                        = m_candidates.rank(tilewrightProblem, tilewrightHardware, context, depth);
                else
                    m_candidates.rank_top(
                        tilewrightProblem, tilewrightHardware, context, depth, ranked);
                auto const& configs = m_candidates.configs();
                positions.reserve(ranked.size());
                for(auto const& r : ranked)
                {
                    if(!r.scored)
                        break;
                    positions.push_back(configs[r.config_index].index);
                }
            }
            catch(std::exception const& e)
            {
                // Ranking fails only to allocate or to plan a launch; origami
                // ranks every kernel then.
                positions.clear();
                if(diagnostics())
                    std::cerr << "[TILEWRIGHT_DIAG TENSILELITE] ranking failed (" << e.what()
                              << "); ranking with origami instead" << std::endl;
            }
            if(diagnostics())
            {
                auto const note = positions.empty()
                                      ? unsupportedContextNote(context, tilewrightHardware)
                                      : nearestContextNote(context, tilewrightHardware);
                if(note)
                    std::cerr << *note << std::endl;
            }
            return positions;
        }

        // The TILEWRIGHT_DIAG line for a context the model scores nothing in
        // (a schedule it does not list), the first time the ranker meets that
        // context; nullopt otherwise.
        std::optional<std::string>
            unsupportedContextNote(tilewright::ExecutionContext const& context,
                                   tilewright::Hardware const&         hardware) const
        {
            if(tilewright::scoring_context(m_candidates.model(), context, hardware))
                return std::nullopt;
            std::lock_guard<std::mutex> lock(m_reportedMutex);
            if(!m_reported.emplace(context.cu_budget, static_cast<int>(context.schedule)).second)
                return std::nullopt;
            size_t const floor = tilewright::describe(m_candidates.model()).min_effective_cus;
            return "[TILEWRIGHT_DIAG TENSILELITE] arch=" + m_arch
                   + " cu_budget=" + std::to_string(context.cu_budget)
                   + " schedule=" + std::to_string(static_cast<int>(context.schedule))
                   + " n_cu=" + std::to_string(hardware.N_CU)
                   + (floor != 0 ? " min_effective_cus=" + std::to_string(floor) : std::string())
                   + ": the model does not support this context; ranking with origami";
        }

        // The TILEWRIGHT_DIAG line for a context outside the model's range
        // only by its CU budget, which it ranks at the nearest budget in it
        // (tilewright::scoring_context), the first time the ranker meets that
        // context; nullopt otherwise.
        std::optional<std::string> nearestContextNote(tilewright::ExecutionContext const& context,
                                                      tilewright::Hardware const& hardware) const
        {
            auto const& model = m_candidates.model();
            if(tilewright::supports(model, context, hardware))
                return std::nullopt;
            auto const at = tilewright::scoring_context(model, context, hardware);
            if(!at)
                return std::nullopt;
            std::lock_guard<std::mutex> lock(m_reportedMutex);
            if(!m_reported.emplace(context.cu_budget, static_cast<int>(context.schedule)).second)
                return std::nullopt;
            size_t const floor = tilewright::describe(model).min_effective_cus;
            return "[TILEWRIGHT_DIAG TENSILELITE] arch=" + m_arch
                   + " cu_budget=" + std::to_string(context.cu_budget)
                   + " schedule=" + std::to_string(static_cast<int>(context.schedule))
                   + " n_cu=" + std::to_string(hardware.N_CU)
                   + (floor != 0 ? " min_effective_cus=" + std::to_string(floor) : std::string())
                   + ": outside the model's CU range; ranked as cu_budget="
                   + std::to_string(at->cu_budget)
                   + ", the kernels launch with cu_budget=" + std::to_string(context.cu_budget);
        }

        // The architecture of the library's kernels.
        std::string const& arch() const noexcept
        {
            return m_arch;
        }

        tilewright::CandidateSet const& candidates() const noexcept
        {
            return m_candidates;
        }

        // The blocklist the ranker's pool was built with; null without one.
        Blocklist const& blocklist() const noexcept
        {
            return m_blocklist;
        }

        // Whether the blocklist left pool position `position` out of the pool.
        bool blocked(size_t position) const noexcept
        {
            return position < m_blocked.size() && m_blocked[position];
        }

        bool blocksAny() const noexcept
        {
            return !m_blocked.empty();
        }

        // The blocklist names that match no kernel of the pool, sorted.
        std::vector<std::string> const& unresolvedBlocklistNames() const noexcept
        {
            return m_unresolved;
        }

        // The library the ranker ranks, as warnings name it.
        std::string const& library() const noexcept
        {
            return m_library;
        }

        // Warns when some blocklist names match no kernel of the pool: they
        // leave nothing out, so a kernel one was written for is offered when
        // its name differs in more than the tokens blocklistKey() drops.
        void warnUnresolvedBlocklistNames() const
        {
            if(m_unresolved.empty())
                return;
            std::cerr << "hipBLASLt Warning: " << m_unresolved.size() << " of the "
                      << m_blocklist->size() << " names in the tilewright blocklist of "
                      << m_library << " match no kernel tilewright ranks for it, even without";
            char const* separator = " their ";
            for(std::string_view const token : BlocklistIgnoredTokens)
            {
                std::cerr << separator << token;
                separator = "/";
            }
            std::cerr << " tokens (the first: " << m_unresolved.front()
                      << "); they leave nothing out.\n";
        }

        static constexpr char const* BlocklistSuffix = ".tilewright.blocklist";

        // The name tokens, an abbreviated parameter and its value, that builds
        // of TensileLite spell differently for one kernel: IA (InterleaveAlpha),
        // which older builds leave out of names, and PKA (PreloadKernArgs),
        // which a build turns off when its compiler cannot preload kernel
        // arguments. The training pipeline's blocklist reader drops the same
        // tokens.
        static constexpr std::array<std::string_view, 2> BlocklistIgnoredTokens{"IA", "PKA"};

        // `name` without its BlocklistIgnoredTokens: every '_'-separated part
        // that is one of them followed by its value (decimal digits, after an
        // 'n' when negative) is dropped with its separator. A blocklist name
        // names every kernel whose name has its key.
        static std::string blocklistKey(std::string_view name)
        {
            std::string key;
            key.reserve(name.size());
            bool first = true;
            for(size_t start = 0;;)
            {
                size_t const           end  = std::min(name.find('_', start), name.size());
                std::string_view const part = name.substr(start, end - start);
                if(!ignoredToken(part))
                {
                    if(!first)
                        key += '_';
                    key += part;
                    first = false;
                }
                if(end == name.size())
                    return key;
                start = end + 1;
            }
        }

    private:
        // Whether `part` is a BlocklistIgnoredTokens token with its value.
        static bool ignoredToken(std::string_view part)
        {
            for(std::string_view const token : BlocklistIgnoredTokens)
            {
                if(part.substr(0, token.size()) != token)
                    continue;
                std::string_view value = part.substr(token.size());
                if(!value.empty() && value.front() == 'n')
                    value.remove_prefix(1);
                if(!value.empty() && std::all_of(value.begin(), value.end(), [](char c) {
                       return c >= '0' && c <= '9';
                   }))
                    return true;
            }
            return false;
        }

        // TILEWRIGHT_DIAG, read as tilewright reads it.
        static bool diagnostics()
        {
            static bool const on = [] {
                char const* value = std::getenv("TILEWRIGHT_DIAG");
                return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
            }();
            return on;
        }

        // The architecture a library file holds kernels for: the compiler
        // target of the last '_' part of its stem, without a stepping
        // ("-strict") or an ASIC revision ("v0"), as the training pipeline's
        // launch_arch() names it, so every library one model ships for names
        // the same launch model.
        static std::string logicArch(std::string const& logicFile)
        {
            std::string const stem = logicStem(logicFile);
            size_t const      pos  = stem.rfind('_');
            if(pos == std::string::npos)
                return {};
            std::string arch = stem.substr(pos + 1);
            arch             = arch.substr(0, arch.find(':'));
            if(arch.rfind("gfx", 0) == 0)
                arch = arch.substr(0, arch.find('-'));
            size_t const revision = arch.find_last_not_of("0123456789");
            if(revision != std::string::npos && revision > 3 && revision + 1 < arch.size()
               && arch[revision] == 'v')
                arch.erase(revision);
            return arch;
        }

        // Logic stems contain no '.', and the file may carry two extensions.
        static std::string logicStem(std::string const& logicFile, std::string* dir = nullptr)
        {
            std::string stem         = logicFile;
            size_t      directoryPos = stem.rfind('/');
#ifdef _WIN32
            if(directoryPos == std::string::npos)
                directoryPos = stem.rfind('\\');
#endif
            if(directoryPos != std::string::npos)
            {
                if(dir != nullptr)
                    *dir = stem.substr(0, directoryPos);
                stem = stem.substr(directoryPos + 1);
            }
            return stem.substr(0, stem.find('.'));
        }

        static void warnUnusable(std::string const& stem, std::string const& error)
        {
            std::cerr << "hipBLASLt Warning: TENSILE_USE_TILEWRIGHT is set but the tilewright "
                         "model for "
                      << stem << " could not be used (" << error
                      << "); ranking with origami instead.\n";
        }

        // Default exactly when ExactLogicLibrary::findTopSolutions computes
        // effectiveDynamic as false.
        template <typename MyProblem>
        static tilewright::Schedule schedule(MyProblem const& problem)
        {
            switch(Debug::Instance().streamK5ForceMode())
            {
            case 0:
                return tilewright::Schedule::Default;
            case 1:
                return tilewright::Schedule::Dynamic;
            default:
                break;
            }
            switch(problem.getParams().streamKTileSchedulingMode())
            {
            case 0:
                return tilewright::Schedule::Default;
            case 2:
                return tilewright::Schedule::Auto;
            default:
                return tilewright::Schedule::Dynamic;
            }
        }

        // The training pipeline derives the same names and values from the
        // sizeMapping keys of the library .dat; both sides change together.
        static std::vector<tilewright::Attribute>
            makeAttributes(SizeMapping const& s, std::vector<std::string> const* names)
        {
            std::vector<tilewright::Attribute> attributes;
            if(names != nullptr && names->empty())
                return attributes;

            auto add = [&attributes, names](char const* name, auto value) {
                if(names == nullptr
                   || std::find(names->begin(), names->end(), name) != names->end())
                    attributes.push_back({name, static_cast<std::int64_t>(value)});
            };
            add("wave_num", s.waveNum);
            add("work_group_x", s.workGroupSize.x);
            add("work_group_y", s.workGroupSize.y);
            add("work_group_z", s.workGroupSize.z);
            add("thread_tile_x", s.threadTile.x);
            add("thread_tile_y", s.threadTile.y);
            add("thread_tile_z", s.threadTile.z);
            add("gwvw_c", s.gwvwC);
            add("stagger_u", s.staggerU);
            add("stagger_u_mapping", s.staggerUMapping);
            add("global_split_u_pgr", s.globalSplitUPGR);
            add("global_split_u", s.globalSplitU);
            add("stagger_stride_shift", s.staggerStrideShift);
            add("workgroup_mapping", s.workGroupMapping);
            add("workgroup_mapping_xcc", s.workGroupMappingXCC);
            add("workgroup_mapping_xcc_group", s.workGroupMappingXCCGroup);
            add("global_split_u_coalesced", s.globalSplitUCoalesced);
            add("global_split_u_wgm_round_robin", s.globalSplitUWorkGroupMappingRoundRobin);
            add("pack_batch_dims", s.packBatchDims);
            add("pack_summation_dims", s.packSummationDims);
            add("magic_div_alg", s.magicDivAlg);
            add("stream_k_atomic", s.streamKAtomic);
            add("tile_processing_strategy", tilewrightAttributeValue(s.tileProcessingStrategy));
            add("work_assignment", tilewrightAttributeValue(s.workAssignment));
            add("prefetch_across_persistent", s.prefetchAcrossPersistent);
            add("persistent_kernel", s.persistentKernel);
            add("persistent_kernel_along_batch", s.persistentKernelAlongBatch);
            add("source_kernel", s.sourceKernel);
            add("global_accumulation", s.globalAccumulation);
            add("adaptive_gemm_gsua", s.adaptiveGemmGSUA);
            add("activation_fused", s.activationFused);
            add("prefetch_global_read", s.PrefetchGlobalRead);
            add("math_clocks_unrolled_loop", s.MathClocksUnrolledLoop);
            add("non_temporal_a", s.nonTemporalA);
            add("non_temporal_b", s.nonTemporalB);
            add("temporal_hint_a", s.temporalHintA);
            add("temporal_hint_b", s.temporalHintB);
            add("has_temporal_hint", s.hasTemporalHint);
            add("adaptive_gemm_ntab", s.adaptiveGemmNTAB);
            add("custom_main_loop_scheduling", s.customMainLoopScheduling);
            add("use_subtile_impl", s.useSubtileImpl);
            add("source_swap", s.SourceSwap);
            add("non_temporal_d", s.NonTemporalD);
            add("wave_separate_global_read_a", s.WaveSeparateGlobalReadA);
            add("wave_separate_global_read_b", s.WaveSeparateGlobalReadB);
            add("unroll_loop_swap_global_read_order", s.UnrollLoopSwapGlobalReadOrder);
            add("direct_to_vgpr_a", s.DirectToVgprA);
            add("direct_to_vgpr_b", s.DirectToVgprB);
            add("num_loads_coalesced_a", s.NumLoadsCoalescedA);
            add("num_loads_coalesced_b", s.NumLoadsCoalescedB);
            add("wave_group_0", s.waveGroup[0]);
            add("wave_group_1", s.waveGroup[1]);
            add("vector_width_a", s.VectorWidthA);
            add("vector_width_b", s.VectorWidthB);
            add("local_split_u", s.LocalSplitU);
            add("direct_to_lds_a", s.DirectToLdsA);
            add("direct_to_lds_b", s.DirectToLdsB);
            add("expert_scheduling_mode", s.expertSchedulingMode);
            add("cluster_dim_x", s.clusterDim.x);
            add("cluster_dim_y", s.clusterDim.y);
            add("cluster_dim_z", s.clusterDim.z);
            return attributes;
        }

        tilewright::CandidateSet                 m_candidates;
        std::string                              m_arch;
        Blocklist                                m_blocklist;
        std::vector<bool>                        m_blocked;
        std::vector<std::string>                 m_unresolved;
        std::string                              m_library;
        mutable std::mutex                       m_reportedMutex;
        mutable std::set<std::pair<size_t, int>> m_reported;
    };

    /**
     * The pools a library ranks when tilewright also ranks its problem type's
     * Equality kernels (the bank): one per device, because a device runs only
     * the bank kernels whose hardware predicate accepts it.
     *
     * Pool position i < P (P kernels in the library's own pool) is the library's
     * kernel i; position P + j is bank[j]. The bank holds, in ascending solution
     * index, the Equality-row solutions whose hardware predicate accepts the
     * device, of every library file that holds the same problem type for the
     * same architecture (libraryProblemTypeKey; the library's own file
     * included), less the solutions of the library's own pool. The training
     * pipeline builds the same pool for its benchmark device.
     */
    template <typename MySolution>
    class TilewrightBank
    {
    public:
        struct Pool
        {
            std::vector<std::shared_ptr<MySolution>> bank;
            // Null when the bank is empty or could not be ranked: the library's
            // own ranker ranks its own pool then.
            std::shared_ptr<const TilewrightRanker> ranker;
        };

        explicit TilewrightBank(tilewright::ModelPtr model)
            : m_model(std::move(model))
        {
        }

        tilewright::ModelPtr const& model() const noexcept
        {
            return m_model;
        }

        // The pool of `device`, made by build() the first time the device asks.
        template <typename Build>
        Pool const& pool(AMDGPU const& device, Build&& build) const
        {
            Device const key{device.processor, device.computeUnitCount, device.pciChipId()};
            if(Entry const* last = m_last.load(std::memory_order_acquire);
               last && last->device == key)
                return last->pool;

            std::lock_guard<std::mutex> lock(m_mutex);
            for(auto const& entry : m_entries)
            {
                if(entry.device == key)
                {
                    m_last.store(&entry, std::memory_order_release);
                    return entry.pool;
                }
            }
            m_entries.push_back(Entry{key, build()});
            m_last.store(&m_entries.back(), std::memory_order_release);
            return m_entries.back().pool;
        }

    private:
        // The fields AMDGPU::operator== compares.
        struct Device
        {
            AMDGPU::Processor  processor;
            int                computeUnitCount;
            std::optional<int> pciChipId;

            bool operator==(Device const& other) const
            {
                return processor == other.processor && computeUnitCount == other.computeUnitCount
                       && pciChipId == other.pciChipId;
            }
        };

        struct Entry
        {
            Device device;
            Pool   pool;
        };

        tilewright::ModelPtr              m_model;
        mutable std::mutex                m_mutex;
        mutable std::list<Entry>          m_entries;
        mutable std::atomic<Entry const*> m_last{nullptr};
    };
} // namespace TensileLite
