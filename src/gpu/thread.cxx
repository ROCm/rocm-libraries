#include "gpu/thread"

namespace gpu {

namespace internal {


__device__ uint32_t numVcores = NUM_VCORES;

// TODO: should these be static members of WorkNode_Header?
__device__ WorkNode_Header *cpuWorkQueue[CPU_WORK_QUEUE_SIZE] = {};
std::atomic<uint32_t> cpuWorkQueueIndex_push = 0;
__device__ uint32_t cpuWorkQueueIndex_pop = 0;

__device__ WorkNode_Header *mainWorkQueue[MAIN_WORK_QUEUE_SIZE] = {};
__device__ uint32_t mainWorkQueueIndex_pop = 0;
__device__ uint32_t mainWorkQueueIndex_push = 0;

__device__ uint32_t activeVcoreCount = 0;
__device__ WorkNode_Header *currentWorkNode[NUM_VCORES] = {};

hipStream_t mainStream;
bool started = false;
__device__ volatile bool finishing = false;


// Locks and returns the worknode at location. If *location == nullptr (i.e. it's locked), waits for the node to be unlocked.
[[nodiscard]] static inline __device__ WorkNode_Header *lockAndFetchWorkNode(WorkNode_Header **location) {
    WorkNode_Header *worknode;
    do {
        // Note: I think access through this cast might be a strict aliasing violation, but it's kind of unavoidable.
        // atomicExch only accepts arithmetic types, so we have to cast currentWorkNode[blockIdx.x] to an arithmetic type like uintptr_t,
        // resulting in a strict aliasing violation.
        worknode = reinterpret_cast<WorkNode_Header*>(atomicExch(reinterpret_cast<uintptr_t*>(location), 0));
    } while (worknode == nullptr);
    return worknode;

    // If we decide to insert a sleep call, switch to this implementation:
    // uintptr_t worknode;
    // // Roughly equivalent to, but faster than:
    // // uintptr_t old = *reinterpret_cast<uintptr_t*>(location), assumed;
    // // do {
    // //     assumed = old;
    // //     old = atomicCAS(reinterpret_cast<uintptr_t*>(location), assumed, 0);
    // // } while (assumed != old || assumed == 0);
    // // Note: I think access through this cast might be a strict aliasing violation, but it's kind of unavoidable.
    // // atomicExch only accepts arithmetic types, so we have to cast *location to an arithmetic type like uintptr_t,
    // // resulting in a strict aliasing violation.
    // while((worknode = atomicExch(reinterpret_cast<uintptr_t*>(location), 0)) == 0) {
    //     //__builtin_amdgcn_s_sleep(1);
    // }
    // return reinterpret_cast<WorkNode_Header *>(worknode);
}

// For use in functions like get_width where worknode hasn't moved anywhere, and we know *link_to_self == nullptr.
static inline __device__ void unlockActiveWorkNode(WorkNode_Header *worknode) {
    __threadfence();
    // I think this might be a strict aliasing violation, but it's kind of unavoidable
    atomicExch(reinterpret_cast<uintptr_t*>(&currentWorkNode[blockIdx.x]), reinterpret_cast<uintptr_t>(worknode));
    // assert(oldValue == nullptr);
}
static inline __device__ void moveAndUnlockWorkNode(WorkNode_Header *worknode, WorkNode_Header **new_location) {
    // assert(new_location != nullptr);
    // TODO: do we need a __threadfence() here to establish a synchronizes-with relationship on link_to_self and not just new_location?
    // See https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence.
    // __threadfence();
    // Update link_to_self = new_location; (unless link_to_self == nullptr) - worknode isn't unlocked just yet!
    if (atomicAdd(reinterpret_cast<uintptr_t*>(&worknode->link_to_self), 0) != 0)
        atomicExch(reinterpret_cast<uintptr_t*>(&worknode->link_to_self), reinterpret_cast<uintptr_t>(new_location));

    __threadfence();
    // Complete the unlock process for worknode by setting *new_location = worknode;
    atomicExch(reinterpret_cast<uintptr_t*>(new_location), reinterpret_cast<uintptr_t>(worknode));
}

// Postcondition: unlocks node
__device__ void setCurrentWorkNode(WorkNode_Header *node, bool yielding) {
    // assert(node->link_to_self == nullptr || *(node->link_to_self) == nullptr);
    if (yielding) {
        node->hasWaiting = true;
        WorkNode_Header *yieldingWorkNode = lockAndFetchWorkNode(&currentWorkNode[blockIdx.x]);
        moveAndUnlockWorkNode(yieldingWorkNode, &node->next);
    }
    moveAndUnlockWorkNode(node, const_cast<WorkNode_Header **>(&currentWorkNode[blockIdx.x]));
}

// TODO: many of these might be better as member functions of WorkNode or WorkNode_Header, and we could potentially make
// a bunch of WorkNode_Header's member variables private

// Post-condition: worknode is likely unlocked and may get invalidated
__device__ void insertWorkNodeIntoMainQueue(WorkNode_Header *worknode) {
    // TODO: is it possible to allow multiple lanes to call insertWorkNode at once?

    const uint32_t myCount = atomicAdd(&mainWorkQueueIndex_push, 1);
    const size_t myIndex = myCount % MAIN_WORK_QUEUE_SIZE;

    if (myCount >= MAIN_WORK_QUEUE_SIZE) {
        for (uint32_t popIndex = atomicAdd(&mainWorkQueueIndex_pop, 0); myCount - popIndex >= MAIN_WORK_QUEUE_SIZE;
             popIndex = atomicAdd(&mainWorkQueueIndex_pop, 0)) {
            assert(atomicAdd(&activeVcoreCount, 0) != 0 && "Probable deadlock: No space left in main work queue and no threads are currently being executed!");
            __builtin_amdgcn_s_sleep(32);
        }
    }

    moveAndUnlockWorkNode(worknode, &mainWorkQueue[myIndex]);
}

// Tries to fetch and simultaneously lock the next waiting worknode in mainWorkQueue. If none exists, or the next one is
// currently locked, or someone else got to it first, returns nullptr.
[[nodiscard]] static inline __device__ WorkNode_Header *popWorkNodeFromMainQueue() {
    // Don't increment until we've successfully fetched a WorkNode.
    uint32_t index = atomicAdd(&mainWorkQueueIndex_pop, 0) % MAIN_WORK_QUEUE_SIZE;
    // Note: I think access through this cast might be a strict aliasing violation, but it's kind of unavoidable.
    // atomicExch only accepts arithmetic types, so we have to cast mainWorkQueue[index] to an arithmetic type like uintptr_t,
    // resulting in a strict aliasing violation.
    WorkNode_Header *work = reinterpret_cast<WorkNode_Header*>(atomicExch(reinterpret_cast<uintptr_t*>(&mainWorkQueue[index]), 0));
    if (work != nullptr) {
        atomicAdd(&mainWorkQueueIndex_pop, 1);
    }
    return work;
}

// Tries to fetch and simultaneously lock the next waiting worknode in cpuWorkQueue. If none exists, or the next one is
// currently locked, or someone else got to it first, returns nullptr.
static inline __device__ WorkNode_Header *popWorkNodeFromCpuQueue() {
    uint32_t index = atomicAdd(&cpuWorkQueueIndex_pop, 0) % CPU_WORK_QUEUE_SIZE;
    WorkNode_Header *work = nullptr;
    // Atomic ops are only atomic with respect to the actions of other GPU cores, not the copy engine. In other words,
    // the copy engine can execute a write in the middle of an atomic op. Thus, if we don't have this 'if' guarding the
    // atomicExch, it's possible for the atomicExch to load nullptr from cpuWorkQueue[i], the copy engine writes a
    // non-null value to cpuWorkQueue[i], then the atomicExch over-writes it with null, and finally atomicExch returns
    // null as if nothing happened.
    if (reinterpret_cast<WorkNode_Header *>(atomicAdd(reinterpret_cast<uintptr_t *>(&cpuWorkQueue[index]), 0)) != nullptr) {
        work = reinterpret_cast<WorkNode_Header*>(atomicExch(reinterpret_cast<uintptr_t*>(&cpuWorkQueue[index]), 0));
    }
    if (work != nullptr) {
        atomicAdd(&cpuWorkQueueIndex_pop, 1);
    }
    return work;
}

// Returns true if there was work waiting
static inline __device__ bool invokeNext(bool yielding = false) {
    __shared__ WorkNode_Header *worknode_s;
    if (threadIdx.x == 0) {
        WorkNode_Header *workFromCpu = popWorkNodeFromCpuQueue();
        WorkNode_Header *workFromMainQueue = popWorkNodeFromMainQueue();

        if (workFromMainQueue != nullptr) {
            if (workFromCpu != nullptr) {
                insertWorkNodeIntoMainQueue(workFromCpu);
            }
            worknode_s = workFromMainQueue;
        }
        else {
            worknode_s = workFromCpu;
        }
    }
    __syncthreads();
    if (worknode_s == nullptr) {
        return false;
    }

    if (threadIdx.x == 0 && !yielding)
        atomicAdd(&activeVcoreCount, 1);

    // Invoke the user-provided function.
    // So we don't have to keep locking and re-loading worknode from currentWorkNode[], wrapper_fn is responsible for unlock
    worknode_s->wrapper_fn(worknode_s, yielding);

    __syncthreads();

    // Now we have to re-lock it and re-fetch worknode in case it was detached while the user function was running.
    if (threadIdx.x == 0) {
        WorkNode_Header *worknode = lockAndFetchWorkNode(&currentWorkNode[blockIdx.x]);
        if (!yielding) {
            atomicSub(&activeVcoreCount, 1);
        } else {
            // If we were called from pseudo_yield, restore the original worknode.
            WorkNode_Header *waiting = lockAndFetchWorkNode(&worknode->next);
            setCurrentWorkNode(waiting, false);
        }

        // Update link_to_self to indicate to detach that the worknode has finished executing and detach needs to free
        // the worknode itself.
        if (atomicExch(reinterpret_cast<uintptr_t*>(&worknode->link_to_self), 0) == 0) {
            // Detach already has been called
            ::free(worknode);
        }
        // Don't need to "unlock" worknode because the scheduler is about to loose any way of accessing worknode, and
        // detach knows what to do when link_to_self == nullptr.
        // TODO: do we need a threadfence and/or some way to establishing a synchronizes-with relationship?
        // See https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence.
    }
    return true;
}

__host__ WorkNode_Header **getCPUWorkQueueAddr() {
    static WorkNode_Header ** const cpuWorkQueueAddr = static_cast<WorkNode_Header **>([](){
        void *temp;
        __LIBGPU_HIP_CHECK__(hipGetSymbolAddress(&temp, HIP_SYMBOL(cpuWorkQueue)));
        return temp;
    }());
    return cpuWorkQueueAddr;
}

static __host__ void waitForSpaceInCPUQueue(const uint32_t myPushCount) {
    for (uint32_t curPopCount = 0; myPushCount - curPopCount >= CPU_WORK_QUEUE_SIZE; ) {
        // TODO: should we put this in a different stream so the copy from Device to Host can happen at the same time as other copies from Host to Device?
        __LIBGPU_HIP_CHECK__(hipMemcpyFromSymbolAsync(&curPopCount, HIP_SYMBOL(cpuWorkQueueIndex_pop), sizeof(curPopCount), 0, hipMemcpyDeviceToHost, getEnqueingStream()));
        __LIBGPU_HIP_CHECK__(hipStreamSynchronize(getEnqueingStream()));
        // Maybe sleep or yield here? On the other hand, hipStreamSynchronize is a blocking call that is likely to take a while
    }
}

__host__ void insertWorkNodeIntoCPUQueue(WorkNode_Header *worknode_d, const uint32_t myPushCount) {
    waitForSpaceInCPUQueue(myPushCount);
    const size_t myIndex = myPushCount % CPU_WORK_QUEUE_SIZE;
    static WorkNode_Header *raw_ptrs[CPU_WORK_QUEUE_SIZE] = {};
    raw_ptrs[myIndex] = worknode_d;
    // Set cpuWorkQueue[myIndex] = worknode_d;
    __LIBGPU_HIP_CHECK__(hipMemcpyToSymbolAsync(HIP_SYMBOL(cpuWorkQueue), &raw_ptrs[myIndex], sizeof(void*), myIndex * sizeof(void*), hipMemcpyHostToDevice, getEnqueingStream()));
}

//====================================================================================================================//
//      KERNELS
//====================================================================================================================//

static __global__ void threading_main() {
    // TODO: Because invokeNext can return false even if there's work waiting (because somebody else snagged the job
    // before we could - see popWorkNodeFromMainQueue and popWorkNodeFromCpuQueue), and we don't increment
    // activeVcoreCount until AFTER we pop work from the queues, it's theoretically possible for a vcore to exit before
    // all the work is done. I don't know if this is a big enough concern to be worth fixing or not.
    for (bool workFound = true; workFound || !finishing || atomicAdd(&activeVcoreCount, 0) != 0;) {
        // TODO: why do we need this when blockDim.x == MAX_THREAD_WIDTH == warpSize?
        __syncthreads();
        workFound = invokeNext();
        if (!workFound)
            __builtin_amdgcn_s_sleep(8);
    }
}

static __global__ void detachWorkNode(WorkNode_Header *oldWorkNode, uint32_t worknodeSize) {
    WorkNode_Header *newWorkNode = static_cast<WorkNode_Header *>(::malloc(worknodeSize));

    // Lock the worknode (unless the workitem finishes before we can get the lock).
    // Note that detachWorkNode is the only function that can potentially invalidate a worknode pointer, so we're
    // guaranteed that oldWorkNode will remain valid. However, oldWorkNode->link_to_self might change/be invalidated
    // while we try to acquire the lock, but that's OK, the CAS will only succeed if we have an up-to-date value for
    // link_to_self AND nobody is currently holding a lock on oldWorkNode.
    WorkNode_Header **link_to_self;
    for (link_to_self = reinterpret_cast<WorkNode_Header **>(atomicAdd(reinterpret_cast<uintptr_t*>(&oldWorkNode->link_to_self), 0));
         link_to_self != nullptr /* while workitem is not finished executing */ &&
         atomicCAS(reinterpret_cast<uintptr_t *>(link_to_self), reinterpret_cast<uintptr_t>(oldWorkNode), 0) !=
             reinterpret_cast<uintptr_t>(oldWorkNode) /* and we failed to acquire the lock, keep trying */;
         link_to_self = reinterpret_cast<WorkNode_Header **>(atomicAdd(reinterpret_cast<uintptr_t*>(&oldWorkNode->link_to_self), 0))) {
        //__builtin_amdgcn_s_sleep(1);
    }
    if (link_to_self == nullptr) {
        // workitem has already finished, so the scheduler has no way of finding oldWorkNode. Thus, we don't have to
        // worry about updating its state or copying it. Just return so the gpu::thread destructor can free oldWorkNode.
        ::free(newWorkNode);
        return;
    }

    oldWorkNode->link_to_self = nullptr;
    // TODO: Technically this is not standards compliant. Even though WorkNode<T> is TriviallyCopyable, it is not a
    // StandardLayoutType, so WorkNode_Header and WorkNode<T> pointers are not interchangeable.
    gpu::memcpy(newWorkNode, oldWorkNode, worknodeSize);

    // If there is a waiting worknode, update its link_to_self value so it points at newWorkNode->next.
    if (oldWorkNode->hasWaiting) {
        // Lock next in case it's also in the middle of being detached.
        WorkNode_Header *next = lockAndFetchWorkNode(&oldWorkNode->next);
        moveAndUnlockWorkNode(next, &newWorkNode->next);
    }

    // Unlock
    __threadfence();
    atomicExch(reinterpret_cast<uintptr_t*>(link_to_self), reinterpret_cast<uintptr_t>(newWorkNode));
}

} // namespace internal

//====================================================================================================================//
//      USER FACING API
//====================================================================================================================//

namespace this_thread {
__device__ gpu::thread::id get_id() noexcept {
    using namespace internal;
    __shared__ WorkNode_Header *current;
    if (threadIdx.x == 0)
        current = lockAndFetchWorkNode(&currentWorkNode[blockIdx.x]);
    gpu::thread::id tid = current->vthread_id + threadIdx.x;
    if (threadIdx.x == 0)
        unlockActiveWorkNode(current);
    return tid;
}

__device__ void pseudo_yield() {
    using namespace internal;

    // TODO: This won't work if the new thread has a width greater than the current one.
    // What happens if we just force the Exec mask to all 1s using inline asm?

    // TODO: what kind of synchronization do we need here? Is this good enough?
    __threadfence();

    invokeNext(true);
}
__device__ unsigned int get_width() noexcept {
    using namespace internal;
    __shared__ WorkNode_Header *current;
    if (threadIdx.x == 0)
        current = lockAndFetchWorkNode(&currentWorkNode[blockIdx.x]);
    __threadfence();
    unsigned int width = current->width;
    if (threadIdx.x == 0)
        unlockActiveWorkNode(current);
    return width;
}
__device__ unsigned int get_lane_id() noexcept {
    using namespace internal;
    return threadIdx.x;
}

} // namespace this_thread

__host__ __device__ thread &thread::operator=(thread &&other) noexcept {
#ifdef __HIP_DEVICE_COMPILE__
    if (joinable()) {
        assert(!joinable() && "Attempted to assign to a gpu::thread object that still has an associated thread");
    }

    worknode_d = other.worknode_d;
    other.worknode_d = nullptr;
    // We can skip setting worknodeSize b/c it's only needed for thread::detach in host code.
#else // __HIP_DEVICE_COMPILE__
    if (joinable()) {
        std::terminate();
    }

    worknode_d = std::move(other.worknode_d);
    worknodeSize = other.worknodeSize;
#endif // !__HIP_DEVICE_COMPILE__
    return *this;
}

__host__ __device__ thread::~thread() {
    if (joinable()) {
#ifdef __HIP_DEVICE_COMPILE__
        assert(!joinable() && "Attempted to destroy a gpu::thread object that still has an associated thread");
#else
        std::terminate();
#endif
    }
}

__host__ __device__ thread::id thread::get_id(uint32_t index) const {
    if (!joinable()) {
        return {};
    }

#ifdef __HIP_DEVICE_COMPILE__
    // Don't need to lock because it's illegal for gpu::thread::detach to be called at the same time as any other
    // gpu::thread method, so we know worknode_d won't change while we're in this function.
    // Also, since vthread_id doesn't change, we don't need to force a fetch from memory, a cached value is fine.
    assert(index < worknode_d->width);
    return worknode_d->vthread_id + index;
#else // __HIP_DEVICE_COMPILE__
    // Don't need to lock because it's illegal for gpu::thread::detach to be called at the same time as any other
    // gpu::thread method, so we know worknode_d won't change while we're in this function.
    // Also, since the memcpy happens in the same stream as the thread constructor launches insertWorkNodeFromHost, we
    // know the memcpy happens after the vthread_id has been assigned.
    // TODO: Is there any way for insertWorkNodeFromHost to be finished and for the write to worknode_d->vthread_id to
    // still be waiting in a cache somewhere? I'm pretty sure the answer is no.
    WorkNode_Header hdr;
    // Copy just the parts we need. Almost guaranteed to copy 2 uint32_ts of data starting at &(worknode_d->width).
    // TODO: Do we want to store a copy of the vthread_id in gpu::thread (and only fetch from worknode_d->vthread_id if the cached copy is invalid)?
    // TODO: fix this
    // if constexpr (offsetof(WorkNode_Header, width) < offsetof(WorkNode_Header, vthread_id)) {
    //     constexpr size_t size = (offsetof(WorkNode_Header, vthread_id) - offsetof(WorkNode_Header, width)) + sizeof(worknode_d->vthread_id);
    //     __LIBGPU_HIP_CHECK__(hipMemcpyAsync(&(hdr.width), &(worknode_d->width), size, hipMemcpyDeviceToHost, getEnqueingStream()));
    // } else {
    //     constexpr size_t size = (offsetof(WorkNode_Header, width) - offsetof(WorkNode_Header, vthread_id)) + sizeof(worknode_d->width);
    //     __LIBGPU_HIP_CHECK__(hipMemcpyAsync(&(hdr.vthread_id), &(worknode_d->vthread_id), size, hipMemcpyDeviceToHost, getEnqueingStream()));
    // }
    // __LIBGPU_HIP_CHECK__(hipStreamSynchronize(getEnqueingStream()));

    if (index >= hdr.width) {
        throw std::out_of_range("thread::get_id: index is greater than thread width");
    }
    return hdr.vthread_id + index;
#endif // !__HIP_DEVICE_COMPILE__
}

__host__ __device__ void thread::join() {
#ifdef __HIP_DEVICE_COMPILE__
    // TODO: check that the user has called gpu::start(), in case they use a hip kernel launch to get here
    if (!joinable()) {
        assert(joinable() && "Attempted to join a gpu::thread object that doesn't have an associated thread");
    }
    // A cached value is ok here because if we did call join on ourselves, then we would have been the ones to write to
    // worknode_d->link_to_self when we popped worknode_d off the work queue. It's also not possible for
    // worknode_d->link_to_self == nullptr if we called join on ourselves, because calling join implies nobody will call
    // detach, and the actively executing thread is by definition, not finished.
    if (worknode_d->link_to_self == &currentWorkNode[blockIdx.x]) {
        assert(false && "Attempted to join the gpu::thread object associated with the active thread");
    }
    // We don't need to lock here because we know nobody is going to call detach on worknode_d, but we do need to make
    // sure that this does a load from memory and not a cache, so we use an atomicAdd(..., 0).
    //
    // Note that link_to_self == nullptr implies the thread has finished, since we can't possibly have called detach
    while (atomicAdd(reinterpret_cast<uintptr_t *>(&(worknode_d->link_to_self)), 0) != 0) {
        // spin while we wait for it to finish.
        __builtin_amdgcn_s_sleep(8);
    }

    // WorkNode<T> is trivially destructible (checked in gpu::thread constructor), so we can safely use free instead
    // of delete
    ::free(worknode_d);
    worknode_d = nullptr;
#else // __HIP_DEVICE_COMPILE__
    if (!started) {
        throw std::logic_error("called gpu::thread::join() before calling gpu::start()");
    }
    if (!joinable()) {
        throw std::system_error(std::error_code(EINVAL, std::system_category()), "thread::join failed");
    }

    // We don't have to worry about worknode_d getting invalidated by detach because we would have to be the one calling detach
    for (WorkNode_Header **link_to_self = reinterpret_cast<WorkNode_Header **>(1); link_to_self != nullptr;) {
        __LIBGPU_HIP_CHECK__(hipMemcpyAsync(&link_to_self, &(worknode_d->link_to_self), sizeof(worknode_d->link_to_self), hipMemcpyDeviceToHost, getEnqueingStream()));
        __LIBGPU_HIP_CHECK__(hipStreamSynchronize(getEnqueingStream()));
        // Maybe sleep or yield here? On the other hand, hipStreamSynchronize is a blocking call that is likely to take a while
    }
    worknode_d = nullptr;
#endif // !__HIP_DEVICE_COMPILE__
}

__host__ __device__ void thread::detach() {
#ifdef __HIP_DEVICE_COMPILE__
    if (!joinable()) {
        assert(joinable() && "Attempted to detach a gpu::thread object that doesn't have an associated thread");
    }

    // We don't need to lock worknode_d before accessing it because the only way for a WorkNode pointer to be
    // invalidated is for a host-side detach to occur.
    // If worknode_d->link_to_self == nullptr, free worknode_d, otherwise set worknode_d->link_to_self = nullptr.
    if (atomicExch(reinterpret_cast<uintptr_t*>(&worknode_d->link_to_self), 0) == 0) {
        // It's already finished, so we have the only pointer to worknode_d. Thus, we can just free the memory.
        ::free(worknode_d);
    }
    // Since the gpu::thread instance is in device memory, we know the gpu did the allocation for worknode_d, so the
    // scheduler can already perform the free when the worknode is finished. No need to copy the data over like in the
    // host version of detach.

    worknode_d = nullptr;
#else // __HIP_DEVICE_COMPILE__
    if (!joinable()) {
        throw std::system_error(std::error_code(EINVAL, std::system_category()), "thread::detach failed");
    }
    hipLaunchKernelGGL(detachWorkNode, dim3(1), dim3(1), 0, getEnqueingStream(), worknode_d.get(), worknodeSize);
    worknode_d = nullptr;
#endif // !__HIP_DEVICE_COMPILE__
}

[[gnu::const]]
__host__ unsigned int thread::hardware_concurrency() noexcept {
    try {
        uint32_t temp;
        __LIBGPU_HIP_CHECK__(hipMemcpyFromSymbol(&temp, HIP_SYMBOL(numVcores), sizeof(temp), 0, hipMemcpyDeviceToHost));
        return temp;
    }
    catch (...) {
        std::cerr << "Exception while fetching numVcores\n";
        return 1;
    }
}

__host__ void thread::start() {
    if (started) {
        throw std::logic_error("gpu::start() called twice");
    }
    started = true;
    // TODO: investigate using hipExtStreamCreateWithCUMask for this
    __LIBGPU_HIP_CHECK__(hipStreamCreateWithFlags(&mainStream, hipStreamNonBlocking));
    hipLaunchKernelGGL(threading_main, dim3(thread::hardware_concurrency()), dim3(MAX_THREAD_WIDTH), 0, mainStream);
}

__host__ void thread::finish(bool blocking) {
    if (!started) {
        throw std::logic_error("called gpu::finish() before calling gpu::start()");
    }
    bool temp = true;
    __LIBGPU_HIP_CHECK__(hipMemcpyToSymbolAsync(HIP_SYMBOL(finishing), &temp, sizeof(bool), 0, hipMemcpyHostToDevice, getEnqueingStream()));
    if (blocking) {
        // We could wait for the memcpy to finish (with hipStreamSynchronize(enqueingStream)) before
        // synchronizing on mainStream, but there's no need.
        __LIBGPU_HIP_CHECK__(hipStreamSynchronize(mainStream));
        __LIBGPU_HIP_CHECK__(hipStreamDestroy(mainStream));
        // Don't destroy enqueingStream because unlike mainStream we won't re-create it if the user decides to 're-start the GPU'
    }
}

}
