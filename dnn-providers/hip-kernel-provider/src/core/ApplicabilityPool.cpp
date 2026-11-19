// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ApplicabilityPool.hpp"

#include <algorithm>
#include <new>
#include <system_error>

#ifdef __linux__
#include <sched.h>
#endif

namespace hip_kernel_provider::core
{

namespace
{

// Constant-initialized and trivially destructible: safe to touch from a worker at any point.
std::atomic<std::size_t> liveThreadCount{0};

} // namespace

std::size_t affinityCpuCount() noexcept
{
#ifdef __linux__
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    if(sched_getaffinity(0, sizeof(cpus), &cpus) == 0)
    {
        const int count = CPU_COUNT(&cpus);
        if(count > 0)
        {
            return static_cast<std::size_t>(count);
        }
    }
#endif
    const auto hardware = std::thread::hardware_concurrency();
    return hardware == 0 ? 1 : static_cast<std::size_t>(hardware);
}

WorkerBudget::WorkerBudget(std::size_t capacity) noexcept
    : _available(capacity)
{
}

WorkerBudget& WorkerBudget::process() noexcept
{
    static WorkerBudget s_budget(std::max<std::size_t>(1, affinityCpuCount() / 2));
    return s_budget;
}

std::size_t WorkerBudget::tryAcquire(std::size_t wanted) noexcept
{
    auto available = _available.load(std::memory_order_relaxed);
    std::size_t granted = 0;
    do
    {
        granted = std::min(wanted, available);
        if(granted == 0)
        {
            return 0;
        }
    } while(!_available.compare_exchange_weak(
        available, available - granted, std::memory_order_relaxed));
    return granted;
}

void WorkerBudget::release(std::size_t count) noexcept
{
    _available.fetch_add(count, std::memory_order_relaxed);
}

std::size_t WorkerBudget::available() const noexcept
{
    return _available.load(std::memory_order_relaxed);
}

/// One tryRun call's tasks. Lives on the caller's stack; the caller does not leave tryRun
/// until no worker can reach it.
struct ApplicabilityPool::Batch
{
    Batch(const std::function<void(std::size_t)>& batchTask, std::size_t taskCount)
        : task(batchTask)
        , count(taskCount)
    {
    }

    const std::function<void(std::size_t)>& task;
    const std::size_t count;
    // Index 0 belongs to the caller.
    std::atomic<std::size_t> next{1};
    // Workers inside drain(); guarded by the pool mutex.
    std::size_t running = 0;
    std::condition_variable finished;
};

ApplicabilityPool::ApplicabilityPool(WorkerBudget& budget)
    : _budget(budget)
{
}

ApplicabilityPool::~ApplicabilityPool()
{
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _stop = true;
    }
    _wake.notify_all();
    for(auto& thread : _threads)
    {
        thread.join();
    }
}

bool ApplicabilityPool::tryRun(std::size_t taskCount,
                               std::size_t maxWorkers,
                               const std::function<void(std::size_t)>& task)
{
    if(taskCount < 2 || maxWorkers == 0)
    {
        return false;
    }

    const auto granted = _budget.tryAcquire(std::min(maxWorkers, taskCount - 1));
    if(granted == 0)
    {
        return false;
    }

    struct BudgetLease
    {
        WorkerBudget& budget;
        std::size_t count;
        ~BudgetLease()
        {
            budget.release(count);
        }
    };
    const BudgetLease lease{_budget, granted};

    Batch batch(task, taskCount);
    std::size_t helpers = 0;
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        while(_threads.size() < granted)
        {
            try
            {
                _threads.emplace_back([this] { workerLoop(); });
            }
            catch(const std::system_error&)
            {
                // The threads already running still serve; with none, the caller runs inline.
                break;
            }
            catch(const std::bad_alloc&)
            {
                break;
            }
            // Counted only once the thread exists. The worker's own decrement cannot run
            // before this: its first act is to take _mutex, which this scope holds.
            liveThreadCount.fetch_add(1, std::memory_order_relaxed);
        }
        helpers = std::min(granted, _threads.size());
        if(helpers == 0)
        {
            return false;
        }
        _queue.insert(_queue.end(), helpers, &batch);
    }

    // Declared after the batch, so it runs first on every exit: withdraws the batch's
    // unclaimed queue entries and waits out the workers already inside it.
    struct JoinGuard
    {
        ApplicabilityPool& pool;
        Batch& joined;
        ~JoinGuard()
        {
            std::unique_lock<std::mutex> lock(pool._mutex);
            pool._queue.erase(std::remove(pool._queue.begin(), pool._queue.end(), &joined),
                              pool._queue.end());
            joined.finished.wait(lock, [this] { return joined.running == 0; });
        }
    };
    const JoinGuard joinGuard{*this, batch};

    if(helpers == 1)
    {
        _wake.notify_one();
    }
    else
    {
        _wake.notify_all();
    }

    task(0);
    drain(batch);
    return true;
}

std::size_t ApplicabilityPool::threadCount() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _threads.size();
}

std::size_t ApplicabilityPool::liveThreads() noexcept
{
    return liveThreadCount.load(std::memory_order_relaxed);
}

void ApplicabilityPool::workerLoop()
{
    std::unique_lock<std::mutex> lock(_mutex);
    while(true)
    {
        _wake.wait(lock, [this] { return _stop || !_queue.empty(); });
        if(_stop)
        {
            break;
        }

        Batch* batch = _queue.front();
        _queue.pop_front();
        ++batch->running;

        lock.unlock();
        drain(*batch);
        lock.lock();

        // Notified under the lock: the caller cannot destroy the batch before it is released.
        if(--batch->running == 0)
        {
            batch->finished.notify_all();
        }
    }
    lock.unlock();
    liveThreadCount.fetch_sub(1, std::memory_order_relaxed);
}

void ApplicabilityPool::drain(Batch& batch)
{
    for(auto index = batch.next.fetch_add(1, std::memory_order_relaxed); index < batch.count;
        index = batch.next.fetch_add(1, std::memory_order_relaxed))
    {
        batch.task(index);
    }
}

} // namespace hip_kernel_provider::core
