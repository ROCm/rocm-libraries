// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace hip_kernel_provider::core
{

/// CPUs this process may run on: the scheduler affinity mask on Linux, otherwise
/// std::thread::hardware_concurrency(). Never less than 1.
std::size_t affinityCpuCount() noexcept;

/// A count of worker slots shared by concurrent callers. Acquisition never blocks: it grants
/// what is free, possibly nothing.
class WorkerBudget
{
public:
    explicit WorkerBudget(std::size_t capacity) noexcept;

    WorkerBudget(const WorkerBudget&) = delete;
    WorkerBudget& operator=(const WorkerBudget&) = delete;
    WorkerBudget(WorkerBudget&&) = delete;
    WorkerBudget& operator=(WorkerBudget&&) = delete;

    /// The process-wide budget, max(1, affinityCpuCount() / 2) slots.
    static WorkerBudget& process() noexcept;

    /// Grants min(wanted, available) slots and returns how many.
    std::size_t tryAcquire(std::size_t wanted) noexcept;

    void release(std::size_t count) noexcept;

    std::size_t available() const noexcept;

private:
    std::atomic<std::size_t> _available;
};

/// Worker threads that share a batch of independent tasks with the calling thread.
///
/// Threads start on the first batch that needs them and are joined in the destructor, so the
/// owner bounds their lifetime; never hold one in a static. Tasks must not submit tasks.
class ApplicabilityPool
{
public:
    explicit ApplicabilityPool(WorkerBudget& budget = WorkerBudget::process());
    ~ApplicabilityPool();

    ApplicabilityPool(const ApplicabilityPool&) = delete;
    ApplicabilityPool& operator=(const ApplicabilityPool&) = delete;
    ApplicabilityPool(ApplicabilityPool&&) = delete;
    ApplicabilityPool& operator=(ApplicabilityPool&&) = delete;

    /// Runs task(0) .. task(taskCount - 1) with up to maxWorkers pool threads helping the
    /// caller. task(0) runs on the caller, first; the rest go to whichever thread claims them.
    /// Every task has finished when this returns true. Returns false, having run nothing, when
    /// no worker can be had: fewer than 2 tasks, no budget, or thread creation failed.
    /// task must not throw.
    bool tryRun(std::size_t taskCount,
                std::size_t maxWorkers,
                const std::function<void(std::size_t)>& task);

    /// Threads this pool owns.
    std::size_t threadCount() const;

    /// Pool worker threads alive in this process, across every pool.
    static std::size_t liveThreads() noexcept;

private:
    struct Batch;

    void workerLoop();
    static void drain(Batch& batch);

    WorkerBudget& _budget;
    mutable std::mutex _mutex;
    std::condition_variable _wake;
    std::deque<Batch*> _queue;
    std::vector<std::thread> _threads;
    bool _stop = false;
};

} // namespace hip_kernel_provider::core
