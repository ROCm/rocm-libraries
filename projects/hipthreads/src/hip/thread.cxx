// -*- C++ -*-

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "hip/thread"

#include "hip/__support/misuse.h"
#include <cstdlib>
#include <hip/atomic>
// For cuda::std::__cccl_thread_sleep_for / cuda::std::__libcpp_thread_sleep_for(__ns)
#include <hip/std/atomic>

// Workaround: On Windows, the HIP runtime does not dispatch kernels to the GPU
// until hipStreamSynchronize is called on the stream. We spawn a detached
// std::thread ("kicker thread") in prepDeviceForWork() to call
// hipStreamSynchronize(mainStream) without blocking the calling thread.
// See prepDeviceForWork() below for the full explanation.
// If this workaround is ever removed, this include can be removed too.
#ifdef _WIN32
    #include <thread>
#endif

#ifndef HIPTHREADS_DEFAULT_VCORES_PER_WGP
#define HIPTHREADS_DEFAULT_VCORES_PER_WGP 16
#endif

namespace cuda {

enum {
    CPU_WORK_QUEUE_SIZE = 4096U, // TODO: Make this depend on hardware_concurrency or something.
    MAIN_WORK_QUEUE_SIZE = 4096U, // TODO: Make this depend on hardware_concurrency or something.
};

namespace internal {

template <size_t queueSize>
struct WorkQueue {
    uint32_t popCount = 0;
    uint32_t pushCount = 0;
    WorkNode_Header *queue[queueSize] = {};

    // Post-condition: worknode is likely unlocked and may get invalidated
    __device__ void push(WorkNode_Header *worknode);
    // Tries to fetch and simultaneously lock the next waiting worknode. If none exists, or it's currently locked, or
    // someone else got to it first, returns nullptr.
    [[nodiscard]] __device__ WorkNode_Header *tryPop(bool shouldIncrementActiveCount) {
        return tryPop(shouldIncrementActiveCount, atomicAdd(&popCount, 0) % queueSize);
    }
    // Tries to fetch and simultaneously lock the next waiting worknode. If none exists, or it's currently locked, or
    // someone else got to it first, returns nullptr. Safe to use with a queue which is written to by the CPU using
    // hipMemcpy.
    [[nodiscard]] __device__ WorkNode_Header *tryPop_cpuSafe(bool shouldIncrementActiveCount) {
        return tryPop_cpuSafe(shouldIncrementActiveCount, atomicAdd(&popCount, 0) % queueSize);
    }
    // Fetches and simultaneously lock the next waiting worknode. If the queue is empty, returns nullptr.
    [[nodiscard]] __device__ WorkNode_Header *pop(bool shouldIncrementActiveCount);

  private:
    __device__ void waitForSpace(uint32_t myPushCount);
    // Tries to fetch and simultaneously lock the worknode at index. If none exists, or it's currently locked, or
    // someone else got to it first, returns nullptr.
    [[nodiscard]] __device__ WorkNode_Header *tryPop(bool shouldIncrementActiveCount, uint32_t index);
    // Tries to fetch and simultaneously lock the worknode at index. If none exists, or it's currently locked, or
    // someone else got to it first, returns nullptr. Safe to use with a queue which is written to by the CPU using
    // hipMemcpy.
    [[nodiscard]] __device__ WorkNode_Header *tryPop_cpuSafe(bool shouldIncrementActiveCount, uint32_t index);
};

// TODO: should these be static members of WorkNode_Header?

// Initializing cpuWorkQueue.pushCount to the maximum value of a uint32_t forces the GPU to keep polling until we update
// cpuWorkQueue.pushCount with the value of cpuWorkQueuePushCount from the CPU. We do this update once there are no
// more hip::wthread objects in the current scope.
__device__ WorkQueue<CPU_WORK_QUEUE_SIZE> cpuWorkQueue = {0, -1U};
::std::atomic<uint32_t> cpuWorkQueuePushCount = 0;

__device__ WorkQueue<MAIN_WORK_QUEUE_SIZE> mainWorkQueue;

__device__ uint32_t numVcores = 0;
__device__ uint32_t activeVcoreCount = 0;
// Server GPUs can have CU/WGP counts in the hundreds, and we spawn multiple vcores per WGP.
__device__ WorkNode_Header *currentWorkNode[8192] = {};

hipStream_t mainStream;
::std::atomic<uint32_t> gpuThreadFromHost_counter = 0;

__host__ __thread_id::underlying_type ThreadData::nextTid() {
    static ::std::atomic<__thread_id::underlying_type> threadIdCounter = 1;
    // Increment by 2 so host and device can use alternating ids.
    return (threadIdCounter += 2);
}
__device__ __thread_id::underlying_type ThreadData::nextTid() {
    static __device__ hip::atomic<__thread_id::underlying_type, hip::thread_scope_device> threadIdCounter = 2;
    // Increment by 2 so host and device can use alternating ids.
    return (threadIdCounter += 2);
}

__host__ ThreadData::ThreadData(uint32_t w) : width(w), vthread_id(nextTid()) {
}
__device__ ThreadData::ThreadData(uint32_t w) : width(w), vthread_id(nextTid()) {
}


// Attempts to lock the WorkNode at location, and if successful, returns the WorkNode. If unsuccessful (i.e. it's
// already locked) returns nullptr.
[[nodiscard]] __device__ WorkNode_Header *WorkNode_Header::tryLockAndFetch(WorkNode_Header **location) {
    // Note: I think access through this cast might be a strict aliasing violation, but it's kind of unavoidable.
    // atomicExch only accepts arithmetic types, so we have to cast location to an arithmetic type like uintptr_t,
    // resulting in a strict aliasing violation.
    return reinterpret_cast<WorkNode_Header*>(atomicExch(reinterpret_cast<uintptr_t*>(location), 0));
}

// Locks and returns the WorkNode at location. If *location == nullptr (i.e. it's currently locked), spins until the
// WorkNode is unlocked.
[[nodiscard]] __device__ WorkNode_Header *WorkNode_Header::lockAndFetch(WorkNode_Header **location) {
    WorkNode_Header *worknode;
    do {
        worknode = tryLockAndFetch(location);
    } while (worknode == nullptr);
    return worknode;
}

// Lock a WorkNode we already have a pointer to. For convenience, returns this->link_to_self (which might be nullptr if
// the node has finished executing). This function is not safe to call from the scheduler, and is only meant for use in
// join and detach, where we can be sure that the WorkNode pointer isn't going to be invalidated unexpectedly while we
// are trying to acquire the lock for it.
__device__ WorkNode_Header **WorkNode_Header::lock() {
    // Even if we know *this will remain valid, this->link_to_self might change/be invalidated while we try to
    // acquire the lock. That's OK though, because the CAS will only succeed if we have an up-to-date value for
    // link_to_self AND nobody is currently holding a lock on *this.
    WorkNode_Header **copy_of_lts;
    for (copy_of_lts = reinterpret_cast<WorkNode_Header **>(atomicAdd(reinterpret_cast<uintptr_t*>(&this->link_to_self), 0));
         copy_of_lts != nullptr /* While *this has not finished executing and ... */ &&
         atomicCAS(reinterpret_cast<uintptr_t *>(copy_of_lts), reinterpret_cast<uintptr_t>(this), 0) !=
             reinterpret_cast<uintptr_t>(this) /* ... we failed to acquire the lock, keep trying. */;
         copy_of_lts = reinterpret_cast<WorkNode_Header **>(atomicAdd(reinterpret_cast<uintptr_t*>(&this->link_to_self), 0))) {
        //__builtin_amdgcn_s_sleep(1);
    }
    return copy_of_lts;
}

// For use in functions like get_width where the WorkNode hasn't moved anywhere, and we know *link_to_self == nullptr.
__device__ void WorkNode_Header::unlockActive() {
    __threadfence();
    // I think this might be a strict aliasing violation, but it's kind of unavoidable
    atomicExch(reinterpret_cast<uintptr_t*>(&currentWorkNode[blockIdx.x]), reinterpret_cast<uintptr_t>(this));
    // assert(oldValue == nullptr);
}
__device__ void WorkNode_Header::moveAndUnlock(WorkNode_Header **new_location) {
    // assert(new_location != nullptr);
    // TODO: do we need a __threadfence() here to establish a synchronizes-with relationship on link_to_self and not just new_location?
    // See https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence.
    // __threadfence();
    // Update link_to_self = new_location; (unless link_to_self == nullptr)
    if (atomicAdd(reinterpret_cast<uintptr_t*>(&this->link_to_self), 0) != 0)
        atomicExch(reinterpret_cast<uintptr_t*>(&this->link_to_self), reinterpret_cast<uintptr_t>(new_location));

    __threadfence();
    // Complete the unlock process by setting *new_location = this;
    atomicExch(reinterpret_cast<uintptr_t*>(new_location), reinterpret_cast<uintptr_t>(this));
}
// Signal that the caller is abdicating any responsability for freeing the WorkNode, unless the caller is the last one
// to do so, in which case, in which case, we free ourself. Note that it is legal to call `delete this`:
// https://isocpp.org/wiki/faq/freestore-mgmt#delete-this, so presumably free(this) is also legal for trivially
// destructible types (as long as we're careful).
// Returns true if we're the last one with any responsability for the WorkNode (i.e. the WorkNode has already been
// detached, or the WorkNode has finished execution).
__device__ bool WorkNode_Header::release() {
    // Use a threadfence to establishing a synchronizes-with relationship between WorkNode_Header::release and itself,
    // and between WorkNode_Header::lock and WorkNode_Header::release. See
    // https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence.
    __threadfence();
    if (atomicExch(reinterpret_cast<uintptr_t*>(&this->link_to_self), 0) == 0) {
        ::free(this);
        return true;
    }
    return false;
}
[[nodiscard]] __device__ bool WorkNode_Header::isSchedulerDoneWith() {
    return atomicAdd(reinterpret_cast<uintptr_t*>(&this->link_to_self), 0) == 0;
}

// Postcondition: unlocks node
__device__ void WorkNode_Header::makeCurrent(bool yielding) {
    // assert(node->link_to_self == nullptr || *(node->link_to_self) == nullptr);
    if (yielding) {
        this->hasWaiting = true;
        WorkNode_Header *yieldingWorkNode = lockAndFetch(&currentWorkNode[blockIdx.x]);
        yieldingWorkNode->moveAndUnlock(&this->next);
    }
    this->moveAndUnlock(&currentWorkNode[blockIdx.x]);
}

// Post-condition: worknode is likely unlocked and may get invalidated
template <size_t queueSize>
__device__ void WorkQueue<queueSize>::push(WorkNode_Header *worknode) {
    // TODO: is it possible to allow multiple lanes to call WorkQueue.push at once?

    const uint32_t myPushCount = atomicAdd(&pushCount, 1);
    waitForSpace(myPushCount);
    worknode->moveAndUnlock(&queue[myPushCount % queueSize]);
}
__device__ void WorkNode_Header::insertIntoMainQueue() {
    mainWorkQueue.push(this);
}

template <size_t queueSize>
__device__ void WorkQueue<queueSize>::waitForSpace(uint32_t myPushCount) {
    if (myPushCount < queueSize) {
        return;
    }
    for (uint32_t curPopCount = atomicAdd(&popCount, 0); myPushCount - curPopCount >= queueSize;
                  curPopCount = atomicAdd(&popCount, 0)) {
        assert(atomicAdd(&activeVcoreCount, 0) != 0 && "Probable deadlock: No space left in main work queue and no threads are currently being executed!");
        __builtin_amdgcn_s_sleep(32);
    }
}

// Tries to fetch and simultaneously lock the worknode at index. If none exists, or it's currently locked, or someone
// else got to it first, returns nullptr.
template <size_t queueSize>
[[nodiscard]] __device__ WorkNode_Header *WorkQueue<queueSize>::tryPop(bool shouldIncrementActiveCount, uint32_t index) {
    WorkNode_Header *work = WorkNode_Header::tryLockAndFetch(&queue[index]);
    if (work == nullptr) {
        return nullptr;
    }
    if (shouldIncrementActiveCount) {
        atomicAdd(&activeVcoreCount, 1);
    }
    __threadfence();
    atomicAdd(&popCount, 1);
    return work;
}

// Tries to fetch and simultaneously lock the worknode at index. If none exists, or it's currently locked, or someone
// else got to it first, returns nullptr. Safe to use with a queue which is written to by the CPU using hipMemcpy.
template <size_t queueSize>
__device__ WorkNode_Header *WorkQueue<queueSize>::tryPop_cpuSafe(bool shouldIncrementActiveCount, uint32_t index) {
    // Atomic ops on device memory are only atomic with respect to the actions of other GPU cores, not the
    // asyncEngine/copy engine. In other words, the copy engine can execute a write in the middle of an atomic op. Thus,
    // if we don't have this 'if' guarding the tryPop call, it's possible for the atomicExch in tryPop to load nullptr
    // from cpuWorkQueue[i], the copy engine writes a non-null value to cpuWorkQueue[i], then the atomicExch over-writes
    // it with null, and finally atomicExch returns null as if nothing happened.
    if (atomicAdd(reinterpret_cast<uintptr_t *>(&queue[index]), 0) == 0) {
        return nullptr;
    }
    return tryPop(shouldIncrementActiveCount, index);

    // TODO: Are we certain that we won't see individual bytes as they get written?
    // If we do see them, then we can use the least significant bit of the pointer as a flag to indicate the write is
    // complete. See sendToGPU for more details.
    // if (atomicAdd(reinterpret_cast<uintptr_t *>(&queue[index]), 0) & 0x1ULL == 0) {
    //     return nullptr;
    // }
    // WorkNode_Header *res = tryPop(shouldIncrementActiveCount, index);
    // return reinterpret_cast<WorkNode_Header *>(reinterpret_cast<uintptr_t>(res) & ~0x1ULL);
}

// Fetches and simultaneously lock the next waiting worknode. If the queue is empty, returns nullptr.
template <size_t queueSize>
[[nodiscard]] __device__ WorkNode_Header *WorkQueue<queueSize>::pop(bool shouldIncrementActiveCount) {
    for (uint32_t curPopCount = atomicAdd(&popCount, 0); curPopCount < atomicAdd(&pushCount, 0);
                  curPopCount = atomicAdd(&popCount, 0)) {
        WorkNode_Header *work = tryPop(shouldIncrementActiveCount, curPopCount % queueSize);
        if (work != nullptr) {
            return work;
        }
    }
    return nullptr;
}

static inline __device__ WorkNode_Header *getWork(bool yielding) {
    WorkNode_Header *workFromCpu = cpuWorkQueue.tryPop_cpuSafe(!yielding);
    WorkNode_Header *workFromMainQueue = mainWorkQueue.tryPop(!yielding && workFromCpu == nullptr);

    if (workFromMainQueue == nullptr && workFromCpu == nullptr)
        return nullptr;

    if (workFromMainQueue != nullptr) {
        if (workFromCpu != nullptr) {
            mainWorkQueue.push(workFromCpu);
        }
        return workFromMainQueue;
    }
    else {
        return workFromCpu;
    }
}

// Returns true if there was work waiting
static inline __device__ bool invokeNext(bool yielding = false) {
    __shared__ WorkNode_Header *worknode_s;
    if (threadIdx.x == 0) {
        worknode_s = getWork(yielding);
    }
    __syncthreads();
    if (worknode_s == nullptr) {
        return false;
    }

    // Invoke the user-provided function.
    // So we don't have to keep locking and re-loading worknode from currentWorkNode[], wrapper_fn is responsible for unlock
    worknode_s->wrapper_fn(worknode_s, yielding);

    __syncthreads();

    // Now we have to re-lock it and re-fetch worknode in case it was detached while the user function was running.
    if (threadIdx.x == 0) {
        WorkNode_Header *worknode = WorkNode_Header::lockAndFetch(&currentWorkNode[blockIdx.x]);
        if (!yielding) {
            atomicSub(&activeVcoreCount, 1);
        } else {
            // If we were called from pseudo_yield, restore the original worknode.
            WorkNode_Header *waiting = WorkNode_Header::lockAndFetch(&worknode->next);
            waiting->makeCurrent(false);
        }

        worknode->release();
    }
    return true;
}

static __host__ WorkNode_Header **getCPUWorkQueueAddr() {
    static WorkNode_Header ** const cpuWorkQueueAddr = [](){
        void *temp;
        __LIBHIPTHREADS_HIP_CHECK__(hipGetSymbolAddress(&temp, HIP_SYMBOL(cpuWorkQueue)));
        auto workQueuePtr = static_cast<decltype(cpuWorkQueue) *>(temp);
        return &workQueuePtr->queue[0];
    }();
    return cpuWorkQueueAddr;
}

static __host__ void waitForSpaceInCPUQueue(const uint32_t myPushCount) {
    for (uint32_t curPopCount = 0; myPushCount - curPopCount >= CPU_WORK_QUEUE_SIZE; ) {
        // TODO: should we put this in a different stream so the copy from Device to Host can happen at the same time as other copies from Host to Device?
        __LIBHIPTHREADS_HIP_CHECK__(hipMemcpyFromSymbolAsync(&curPopCount, HIP_SYMBOL(cpuWorkQueue), sizeof(curPopCount), offsetof(decltype(cpuWorkQueue), popCount), hipMemcpyDeviceToHost, getEnqueingStream()));
        __LIBHIPTHREADS_HIP_CHECK__(hipStreamSynchronize(getEnqueingStream()));
        // Maybe sleep or yield here? On the other hand, hipStreamSynchronize is a blocking call that is likely to take a while
    }
}

static __global__ void threading_main();
static __host__ void prepDeviceForWork() {
    if (gpuThreadFromHost_counter++ != 0) {
        return;
    }
    // 1 zero followed by 511 ones
    static constexpr uint32_t cuMask[] = {
        0x7FFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU,
        0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
    };

    bool isFirstTime = false;
    // This bit serves 2 purposes: The lambda is a trick to run a code snippet once, then, since we have to allocate a
    // variable for the trick anyways, and we also need some persistent memory to copy from when setting
    // cpuWorkQueue.pushCount = -1U, we'll use temp for that purpose too.
    static uint32_t temp = ([&isFirstTime]() {
        // TODO: investigate using hipExtStreamCreateWithCUMask for this
        __LIBHIPTHREADS_HIP_CHECK__(hipExtStreamCreateWithCUMask(&mainStream, sizeof(cuMask)/sizeof(cuMask[0]), cuMask));
        isFirstTime = true;
    }(), -1U);

    if (isFirstTime) {
        // Initalize numVcores, so device code can call hip::wthread::hardware_concurrency
        static uint32_t temp2 = hip::wthread::hardware_concurrency();
        __LIBHIPTHREADS_HIP_CHECK__(hipMemcpyToSymbolAsync(HIP_SYMBOL(numVcores), &temp2, sizeof(temp2), 0, hipMemcpyHostToDevice, getEnqueingStream()));
    } else {
        // Tell the device there's work coming, so threading_main doesn't immediately return. This has to happen on the
        // EnqueingStream because notifyDeviceThereMightNotBeAnyMoreWork does its copy in that stream, and we need to make
        // sure the change made by the last wthread's destructor doesn't get over-written.
        // cpuWorkQueue.pushCount is initialized to -1U, so we don't need to do this the first time through.
        __LIBHIPTHREADS_HIP_CHECK__(hipMemcpyToSymbolAsync(HIP_SYMBOL(cpuWorkQueue), &temp, sizeof(temp), offsetof(decltype(cpuWorkQueue), pushCount), hipMemcpyHostToDevice, getEnqueingStream()));
        __LIBHIPTHREADS_HIP_CHECK__(hipStreamSynchronize(getEnqueingStream()));
    }

    hipLaunchKernelGGL(threading_main, dim3(wthread::hardware_concurrency()), dim3(hip::wthread::max_width()), 0, mainStream);

    // Workaround for Windows lazy kernel dispatch:
    // On Windows, kernel dispatch is delayed until something triggers a flush of the stream (eg. hipStreamSynchronize).
    // Without this workaround, threading_main is queued but never actually dispatched to the GPU, causing later
    // hip::wthread operations to hang (work is submitted to cpuWorkQueue but the kernel that processes it never starts).
    //
    // We can't call hipStreamSynchronize(mainStream) on the calling thread because threading_main is a persistent
    // kernel that will loop forever until it's told to stop - the sync would block until the kernel finishes, which
    // only happens after all work is enqueued and notifyDeviceThereMightNotBeAnyMoreWork() is called. That creates a
    // deadlock: the calling thread can't finish submitting work because it's blocked, and the kernel can't stop because
    // it is still waiting to receive work.
    //
    // Solution: spawn a detached thread that calls hipStreamSynchronize. This kick-starts threading_main, while the
    // original calling thread continues and can submit all of its work. The detached thread blocks in
    // hipStreamSynchronize until the kernel eventually exits, at which point the thread silently terminates.
    #ifdef _WIN32
        ::std::thread([](){
            __LIBHIPTHREADS_HIP_CHECK__(hipStreamSynchronize(mainStream));
        }).detach();
    #endif
}

static __host__ void notifyDeviceThereMightNotBeAnyMoreWork [[maybe_unused]] (bool blocking = false) {
    // Needs to be static because the copy might not be done before we return
    static uint32_t temp;
    temp = cpuWorkQueuePushCount;
    __LIBHIPTHREADS_HIP_CHECK__(hipMemcpyToSymbolAsync(HIP_SYMBOL(cpuWorkQueue), &temp, sizeof(temp), offsetof(decltype(cpuWorkQueue), pushCount), hipMemcpyHostToDevice, getEnqueingStream()));
    if (blocking) {
        // We could wait for the memcpy to finish (with hipStreamSynchronize(enqueingStream)) before
        // synchronizing on mainStream, but there's no need.
        __LIBHIPTHREADS_HIP_CHECK__(hipStreamSynchronize(mainStream));
    }
}

// TODO: Should we return a thrust::unique_ptr instead?
__host__ ::std::unique_ptr<WorkNode_Header, WorkNodeDeleter> WorkNode_Header::sendToGPU(WorkNode_Header **new_location) {
    prepDeviceForWork();

    this->link_to_self = new_location;

    ::std::unique_ptr<WorkNode_Header, WorkNodeDeleter> worknode_d(static_cast<WorkNode_Header *>(hip::malloc(this->worknodeSize)));
    __LIBHIPTHREADS_HIP_CHECK__(hipMemcpyAsync(worknode_d.get(), this, this->worknodeSize, hipMemcpyHostToDevice, getEnqueingStream()));
    hipEvent_t copyFinished;
    __LIBHIPTHREADS_HIP_CHECK__(hipEventCreate(&copyFinished));
    __LIBHIPTHREADS_HIP_CHECK__(hipEventRecord(copyFinished, getEnqueingStream()));

    // *new_location = worknode_d.get();
    __LIBHIPTHREADS_HIP_CHECK__(hipStreamWriteValue64(getEnqueingStream(), new_location, reinterpret_cast<uintptr_t>(worknode_d.get()), 0));

    // TODO: If the GPU is able to see individual bytes as they get written, we need a way to signal when the whole
    // write is complete. It seems that on Navi4 cards, if the host uses hipMemcpyAsync to update queue[index], the
    // device will see individual bytes as they get written. Switching to hipStreamWriteValue64 seems to have fixed the
    // issue, but we should confirm that it's a proper fix. If it's not enough, uncomment this code and update
    // tryPop_cpuSafe as well.
    // Since the least significant bits of the pointer should always be 0, we can use one of them as
    // a flag to signal the write is complete. Note: We are relying on stream operation ordering to guarantee that flag
    // is only set AFTER the write completes. Futher note: If we were guaranteed that the least significant byte is
    // always written last, we could combine the two operations into a single write. GPU is little-endian, therefore
    // reinterpret_cast<char *>(new_location) + 7 is the MSB and new_location is the LSB

    // __LIBHIPTHREADS_HIP_CHECK__(hipMemsetAsync(new_location, (reinterpret_cast<uintptr_t>(worknode_d.get()) & 0xFF) | 0x1, 1, getEnqueingStream()));

    __LIBHIPTHREADS_HIP_CHECK__(hipEventSynchronize(copyFinished));
    return worknode_d;
}
__host__ ::std::unique_ptr<WorkNode_Header, WorkNodeDeleter> WorkNode_Header::sendToGPU() {
    const uint32_t myPushCount = cpuWorkQueuePushCount++;
    const size_t myPushIndex = myPushCount % CPU_WORK_QUEUE_SIZE;

    waitForSpaceInCPUQueue(myPushCount);

    return sendToGPU(getCPUWorkQueueAddr() + myPushIndex);
}

static __device__ bool shouldKeepPollingForWork() {
    // Don't bother with any of the other atomic loads if the CPU hasn't told us it's done sending work.
    if (atomicAdd(&cpuWorkQueue.pushCount, 0) == -1U) {
        return true;
    }
    // Pre-fetch the pop counts so that the load of activeVcoreCount happens after the load of the pop count, but
    // before the load of the push count. This ensures that if mainWorkQueue.tryPop and/or cpuWorkQueue.tryPop_cpuSafe
    // returned null only because someone beat us to the punch, we're guaranteed shouldKeepPollingForWork will return
    // true.
    uint32_t cached_cpuPopCount = atomicAdd(&cpuWorkQueue.popCount, 0);
    uint32_t cached_mainPopCount = atomicAdd(&mainWorkQueue.popCount, 0);
    return atomicAdd(&activeVcoreCount, 0) != 0 ||
            cached_mainPopCount         < atomicAdd(&mainWorkQueue.pushCount, 0) ||
            cached_cpuPopCount          < atomicAdd(&cpuWorkQueue.pushCount, 0);
}

//====================================================================================================================//
//      KERNELS
//====================================================================================================================//

static __global__ void threading_main() {
    for (bool workFound = true; workFound || shouldKeepPollingForWork();) {
        // TODO: why do we need this when blockDim.x == hip::wthread::max_width() == warpSize?
        __syncthreads();
        workFound = invokeNext();
        if (!workFound)
            __builtin_amdgcn_s_sleep(8);
    }
}

static __global__ void detachWorkNode(WorkNode_Header *oldWorkNode) {
    WorkNode_Header *newWorkNode = static_cast<WorkNode_Header *>(::malloc(oldWorkNode->worknodeSize));

    WorkNode_Header **link_to_self = oldWorkNode->lock();
    if (link_to_self == nullptr) {
        // workitem has already finished, so the scheduler has no way of finding oldWorkNode. Thus, we don't have to
        // worry about updating its state or copying it. Just return so the hip::wthread destructor can free oldWorkNode.
        ::free(newWorkNode);
        return;
    }

    oldWorkNode->link_to_self = nullptr;
    // TODO: Technically this is not standards compliant. Even though WorkNode<T> is TriviallyCopyable, it is not a
    // StandardLayoutType, so WorkNode_Header and WorkNode<T> pointers are not interchangeable.
    hip::memcpy(newWorkNode, oldWorkNode, oldWorkNode->worknodeSize);

    // If there is a waiting worknode, update its link_to_self value so it points at newWorkNode->next.
    if (oldWorkNode->hasWaiting) {
        // Lock next in case it's also in the middle of being detached.
        WorkNode_Header *next = WorkNode_Header::lockAndFetch(&oldWorkNode->next);
        next->moveAndUnlock(&newWorkNode->next);
    }

    // This does an extra unnecessary load of newWorkNode->link_to_self to check if it's null, but that's unlikely to
    // have a major performance impact.
    newWorkNode->moveAndUnlock(link_to_self);
}

} // namespace internal

//====================================================================================================================//
//      USER FACING API
//====================================================================================================================//

namespace this_thread {
_LIBHIPTHREADS_EXPORTED_FROM_ABI __host__ __device__ void sleep_for(cuda::std::chrono::nanoseconds __ns) {
#if CCCL_VERSION >= 3000000
    cuda::std::__cccl_thread_sleep_for(__ns);
#else
    cuda::std::__libcpp_thread_sleep_for(__ns);
#endif
}

__device__ hip::wthread::id get_id() noexcept {
    using namespace internal;
    __shared__ hip::__thread_id::underlying_type base_vtid;
    if (threadIdx.x == 0) {
        WorkNode_Header *current = WorkNode_Header::lockAndFetch(&currentWorkNode[blockIdx.x]);
        base_vtid = current->tdata.vthread_id;
        current->unlockActive();
    }
    // TODO: What if only some of the fibers call this_thread::get_id()? This could deadlock. Also, what if the fiber
    // with threadIdx.x == 0 isn't one of the calling fibers?
    __syncthreads();
    return hip::__thread_id(base_vtid * wthread::max_width() + threadIdx.x);

    // Something along these lines might work for when threadIdx.x == 0 isn't among the calling fibers.
    // __shared__ hip::__thread_id::underlying_type base_vtid;
    // base_vtid = 0;
    // // TODO: this could still deadlock though...
    // __syncthreads();
    // do {
    //     WorkNode_Header *current = WorkNode_Header::tryLockAndFetch(&currentWorkNode[blockIdx.x]);
    //     if (current != nullptr) {
    //         base_vtid = current->tdata.vthread_id;
    //         current->unlockActive();
    //     }
    //     __syncthreads();
    // } while (base_vtid == 0);
    // return hip::__thread_id(base_vtid * wthread::max_width() + threadIdx.x);
}

__device__ void pseudo_yield() {
    using namespace internal;

    // TODO: This won't work if the new wthread has a width greater than the current one.
    // What happens if we just force the Exec mask to all 1s using inline asm?

    // TODO: what kind of synchronization do we need here? Is this good enough?
    __threadfence();

    invokeNext(true);
}
__device__ unsigned int get_width() noexcept {
    using namespace internal;
    __shared__ unsigned int width;
    if (threadIdx.x == 0) {
        WorkNode_Header *current = WorkNode_Header::lockAndFetch(&currentWorkNode[blockIdx.x]);
        width = current->tdata.width;
        current->unlockActive();
    }
    // TODO: What if only some of the fibers call this_thread::get_width()? This could deadlock. Also, what if the fiber
    // with threadIdx.x == 0 isn't one of the calling fibers?
    __syncthreads();
    return width;
}
__device__ unsigned int get_fiber_id() noexcept {
    using namespace internal;
    return threadIdx.x;
}

} // namespace this_thread

__host__ wthread::wthread() noexcept {
    prepDeviceForWork();
}

__host__ __device__ wthread &wthread::operator=(wthread &&other) noexcept {
#ifdef __HIP_DEVICE_COMPILE__
    if (joinable()) [[unlikely]] {
        __HIPTHREADS_REPORT_MISUSE("move-assigned onto a still-joinable hip::wthread. A hip::wthread must be "
                                   "join()ed or detach()ed before it is destroyed or move-assigned onto.");
        // Nonfatal mode: detection recorded; leave *this owning its wave.
        return *this;
    }

    worknode_d = other.worknode_d;
    other.worknode_d = nullptr;
#else // __HIP_DEVICE_COMPILE__
    if (joinable()) {
        ::std::terminate();
    }

    worknode_d = ::std::move(other.worknode_d);
#endif // !__HIP_DEVICE_COMPILE__
    cached_tdata = ::std::move(other.cached_tdata);
    return *this;
}

__host__ __device__ wthread::~wthread() {
#ifdef __HIP_DEVICE_COMPILE__
    if (joinable()) [[unlikely]] {
        __HIPTHREADS_REPORT_MISUSE("destroyed a still-joinable hip::wthread. A hip::wthread must be join()ed "
                                   "or detach()ed before it is destroyed or move-assigned onto.");
        // Nonfatal mode: detection recorded; detach so the still-running work
        // node is cleaned up and the test can continue to a clean exit.
        detach();
        return;
    }
#else
    if (joinable()) {
        ::std::terminate();
    }
    if (--gpuThreadFromHost_counter == 0) {
        notifyDeviceThereMightNotBeAnyMoreWork();
    }
#endif
}

// TODO: Maybe instead of returning different ids for each fiber based on a user-provided index, a single wthread::id
// should hold info for all the fibers? Do we even want to differentiate IDs between fibers? wthread::id is supposed to
// be a pretty opaque type anyways, and is generally only used as a key for storing/sorting threads in containers.
__host__ __device__ wthread::id wthread::get_id(uint32_t index) const {
    if (!joinable()) {
        return {};
    }

#ifdef __HIP_DEVICE_COMPILE__
    // Don't need to lock because it's illegal for hip::wthread::detach to be called at the same time as any other
    // hip::wthread method, so we know worknode_d won't change while we're in this function.
    // Also, since vthread_id doesn't change, we don't need to force a fetch from memory, a cached value is fine.
    assert(index < cached_tdata.width);
#else // __HIP_DEVICE_COMPILE__
    if (index >= cached_tdata.width) {
        throw ::std::out_of_range("wthread::get_id: index is greater than thread width");
    }
#endif // !__HIP_DEVICE_COMPILE__
    return hip::__thread_id(cached_tdata.vthread_id * wthread::max_width() + index);
}

__host__ __device__ void wthread::join() {
#ifdef __HIP_DEVICE_COMPILE__
    // TODO: check that the user has called hip::start(), in case they use a hip kernel launch to get here
    // Nonfatal mode: nothing to join; detection recorded, __HIPTHREADS_ASSERT causes early return.
    __HIPTHREADS_ASSERT(joinable(),
                        "join() called on a hip::wthread that has no associated thread (not joinable). "
                        "Only a joinable hip::wthread (one that has not been join()ed or detach()ed) "
                        "may be join()ed.");
    // A cached value is ok here because if we did call join on ourselves, then we would have been the ones to write to
    // worknode_d->link_to_self when we popped worknode_d off the work queue. It's also not possible for
    // worknode_d->link_to_self == nullptr if we called join on ourselves, because calling join implies nobody will call
    // detach, and the actively executing thread is by definition, not finished.
    // Nonfatal mode: a self-join can never complete; detection recorded, __HIPTHREADS_ASSERT causes early return.
    __HIPTHREADS_ASSERT(worknode_d->link_to_self != &currentWorkNode[blockIdx.x],
                        "join() called on the hip::wthread associated with the currently-running thread "
                        "(self-join). A thread cannot join itself; this would spin forever.");
    // We don't need to lock here because we know nobody is going to call detach on worknode_d.
    while (!worknode_d->isSchedulerDoneWith()) {
        // spin while we wait for it to finish.
        __builtin_amdgcn_s_sleep(8);
    }

    // WorkNode<T> is trivially destructible (checked in hip::wthread constructor), so we can safely use free instead
    // of delete
    ::free(worknode_d);
    worknode_d = nullptr;
#else // __HIP_DEVICE_COMPILE__
    if (!joinable()) {
        throw ::std::system_error(::std::error_code(EINVAL, ::std::system_category()), "wthread::join failed");
    }

    // We don't have to worry about worknode_d getting invalidated by detach because we would have to be the one calling detach
    for (WorkNode_Header **link_to_self = reinterpret_cast<WorkNode_Header **>(1); link_to_self != nullptr;) {
        __LIBHIPTHREADS_HIP_CHECK__(hipMemcpyAsync(&link_to_self, &(worknode_d->link_to_self), sizeof(worknode_d->link_to_self), hipMemcpyDeviceToHost, getEnqueingStream()));
        __LIBHIPTHREADS_HIP_CHECK__(hipStreamSynchronize(getEnqueingStream()));
        // Maybe sleep or yield here? On the other hand, hipStreamSynchronize is a blocking call that is likely to take a while
    }
    worknode_d = nullptr;
#endif // !__HIP_DEVICE_COMPILE__
}

__host__ __device__ void wthread::detach() {
#ifdef __HIP_DEVICE_COMPILE__
    // Nonfatal mode: nothing to detach; detection recorded, __HIPTHREADS_ASSERT causes early return.
    __HIPTHREADS_ASSERT(joinable(),
                        "detach() called on a hip::wthread that has no associated thread (not joinable). "
                        "Only a joinable hip::wthread (one that has not been join()ed or detach()ed) "
                        "may be detach()ed.");

    worknode_d->release();

    worknode_d = nullptr;
#else // __HIP_DEVICE_COMPILE__
    if (!joinable()) {
        throw ::std::system_error(::std::error_code(EINVAL, ::std::system_category()), "wthread::detach failed");
    }
    hipLaunchKernelGGL(detachWorkNode, dim3(1), dim3(1), 0, getEnqueingStream(), worknode_d.get());
    // No sync needed because hip::free does the hipFreeAsync in EnqueingStream.
    worknode_d = nullptr;
#endif // !__HIP_DEVICE_COMPILE__
}

[[gnu::const]]
__host__ unsigned int wthread::hardware_concurrency() noexcept {
    try {
        static uint32_t multiprocessorCount = [](){
            int temp;
            __LIBHIPTHREADS_HIP_CHECK__(hipDeviceGetAttribute(&temp, hipDeviceAttributeMultiprocessorCount, 0));
            // __LIBHIPTHREADS_HIP_CHECK__(hipDeviceGetAttribute(&physicalMultiProcessorCount, hipDeviceAttributePhysicalMultiProcessorCount, 0));
            return temp;
        }();
        // Number of scheduler vcores launched per WGP/multiprocessor. Overridable at runtime via the
        // HIPTHREADS_VCORES_PER_WGP environment variable; defaults to 16 if unset or unparseable.
        static uint32_t vcoresPerWgp = [](){
            const char *env = ::std::getenv("HIPTHREADS_VCORES_PER_WGP");
            if (env != nullptr) {
                char *end;
                unsigned long val = ::std::strtoul(env, &end, 10);
                if (end != env && *end == '\0' && val > 0) {
                    return static_cast<uint32_t>(val);
                }
            }
            return static_cast<uint32_t>(HIPTHREADS_DEFAULT_VCORES_PER_WGP);
        }();
        return multiprocessorCount * vcoresPerWgp;
    }
    catch (...) {
        ::std::cerr << "Exception while fetching multiprocessorCount\n";
        return 1;
    }
}
__device__ unsigned int wthread::hardware_concurrency() noexcept {
    return numVcores;
}
}
