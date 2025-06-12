#ifndef __GPU___THREAD_WORKITEM_H__
#define __GPU___THREAD_WORKITEM_H__

#include <memory>

#include "hip/hip_runtime.h"

#include "gpu/__functional/invoke.h"

namespace gpu::internal {

struct WorkNode_Header;

typedef void (*WrappedFnPointer)(WorkNode_Header *, bool);

struct WorkNode_Header {
    WrappedFnPointer wrapper_fn = nullptr;

    // link_to_self enables thread::detach() to copy a worknode into memory the gpu can free on its own.
    // It can be either &mainWorkQueue[i], &cpuWorkQueue[i], &currentWorkNode[blockIdx.x], &(prev->next) or nullptr
    //
    // In the scheduler, link_to_self == nullptr indicates we've been detached.
    // In detach and join, link_to_self == nullptr indicates we've finished executing.
    //
    // When holding a pointer to a WorkNode, it must always be in a "locked" state to ensure the pointer doesn't get
    // invalidated by detach. A WorkNode is "locked" if *link_to_self != self (or, if the worknode has already been
    // detached, whatever link_to_self would have been).
    //
    // Most of the time if a WorkNode is locked, then *link_to_self == nullptr, but if a node is in the middle of being
    // moved (e.g. from cpuQueue to mainQueue or mainQueue to currentWorkNode), then the old location might get re-used
    // to store a different WorkNode before link_to_self is updated with the new location.
    //
    // If I'm not mistaken, lock contention cannot occur within the scheduler proper. I think it only occur between
    // detachWorkNode and a scheduler function? In other words, if the scheduler is unable to immediately acquire the
    // lock, I think that implies detachWorkNode is holding the lock.
    WorkNode_Header **link_to_self = reinterpret_cast<WorkNode_Header **>(1);

    WorkNode_Header *next = nullptr; // Must always be device-accessible. Either zero-copy pinned host memory, or device memory
    // Since *link_to_self == nullptr is used as a per-workitem lock, we need some way to differentiate between
    // next == nullptr because there's nobody waiting and next == nullptr because next has been locked.
    bool hasWaiting = false;
    uint32_t width = 1; // How many threads per block/vthread are active
    // Stores the sizeof(WorkNode<T>), so we can do a memcpy without knowing T.
    // Not initialized when a WorkNode is constructed from device code because we don't need it.
    size_t worknodeSize;
    // TODO: fix vthread_id - it's currently non-functional
    uint32_t vthread_id = {}; // TODO: should this be a gpu::thread::max_width() array? For now we just store the lowest id.
};
static_assert(std::is_standard_layout_v<WorkNode_Header>);

template <class Callable_t>
struct WorkNode : WorkNode_Header {
    using Callable = Callable_t;
    WorkNode(WorkNode &&other) = default;
    inline __device__ WorkNode(uint32_t width, Callable &&callable);
    inline __host__ WorkNode(uint32_t width, Callable &&callable);

    Callable fn;
};

template <class Fn_t, class... Args_t>
__host__ auto make_worknode(uint32_t width, Fn_t &&typed_fn, Args_t &&...args) {
    // Ideally, we would also forward args in the capture (...args = std::forward<Args_t>(args)) to avoid an extra copy,
    // but that requires C++20
    auto lambda = [typed_fn = std::forward<Fn_t>(typed_fn), args...] __device__() {
        gpu::__invoke(std::move(typed_fn), std::move(args)...);
    };
    using WorkNode_t = WorkNode<decltype(lambda)>;
    WorkNode_t *worknode_ptr = new WorkNode_t(width, std::move(lambda));
    // Sadly, hipHostUnregister performs an implicit device-wide synchronization. Thus, in order to use pinned host
    // memory for the async copy, we would either end up with a gradually growing amount of pinned memory, or need to
    // re-use the same pinned memory every time.
    // __LIBGPU_HIP_CHECK__(hipHostRegister(worknode_ptr, sizeof(WorkNode_t), hipHostRegisterDefault));

    return std::unique_ptr<WorkNode_t>(worknode_ptr);
}
template <class Fn_t, class... Args_t>
__device__ auto make_worknode(uint32_t width, Fn_t &&typed_fn, Args_t &&...args) {
    // These will give a more user-friendly error message when the lambda is not move-constructible.
    static_assert(std::is_move_constructible_v<Fn_t>);
    static_assert((std::is_move_constructible_v<Args_t> && ...));

    // Ideally, we would also forward args in the capture (...args = std::forward<Args_t>(args)) to avoid an extra copy,
    // but that requires C++20
    auto lambda = [typed_fn = std::forward<Fn_t>(typed_fn), args...] __device__() {
        gpu::__invoke(std::move(typed_fn), std::move(args)...);
    };

    // Allocate memory using malloc instead of new, to guaranteed that ::free(worknode) is valid.
    // The C++ standard doesn't guarantee that new and malloc allocate from the same pool of memory.
    void *buf = ::malloc(sizeof(WorkNode<decltype(lambda)>));
    return new(buf) WorkNode<decltype(lambda)>(width, std::move(lambda));
}

//====================================================================================================================//
//      INTERNAL/HELPER FUNCTIONS
//====================================================================================================================//

// Postcondition: unlocks node
__device__ void setCurrentWorkNode(WorkNode_Header *node, bool yielding);

// Precondition: We're still holding the lock acquired in invokeNext. Needed to make sure detach doesn't make worknode
// an invalid pointer before we load typed_node_ptr and width.
template <class WorkNode_t>
__device__ void wrapper(WorkNode_Header *worknode, bool yielding) {
    WorkNode_t *typed_node_ptr = static_cast<WorkNode_t *>(worknode);
    typename WorkNode_t::Callable fn = std::move(typed_node_ptr->fn);
    uint32_t width = typed_node_ptr->width;
    __syncthreads();
    // Include a threadfence for all threads just to be safe.
    __threadfence();
    // Also unlocks worknode
    if (threadIdx.x == 0)
        setCurrentWorkNode(worknode, yielding);
    if (threadIdx.x < width) {
        fn();
    }
    __threadfence();
    // TODO: figure out how to make all the threads with idx > width 'catch up' on missed __syncthreads() calls.
    // Shouldn't be a concern as long as blockDim.x == gpu::thread::max_width() == warpSize
}

template <class Callable_t>
__device__ WorkNode<Callable_t>::WorkNode(uint32_t w, Callable_t &&callable) : fn(std::move(callable)) {
    wrapper_fn = wrapper<WorkNode<Callable_t>>;
    width = w;
}

template <class WorkNode_t>
__global__ void getWrapperFn(WrappedFnPointer *ptr) {
    *ptr = wrapper<WorkNode_t>;
}

template <class Callable_t>
__host__ WorkNode<Callable_t>::WorkNode(uint32_t w, Callable_t &&callable) : fn(std::move(callable)) {
    width = w;
    worknodeSize = sizeof(WorkNode<Callable_t>);
    // Only way to pass the device the information about how to invoke WorkNode<Callable_t>.fn is by launching a kernel.
    // Why do we NEED a kernel?
    // We can't reference __device__ functions from __host__ functions, so this is illegal:
    // header = {nullptr, nullptr, false, {}, 0, wrapper<WorkNode<Callable_t>> };
    //
    // Extended lambda's don't define a pointer-to-function conversion operator, so wrapping the invokation of the
    // __device__ function in an extended lambda without captures doesn't work:
    // header = {nullptr, nullptr, false, {}, 0, [] __device__ () { wrapper<WorkNode<Callable_t>>() } };
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 14.7.2.16 Extended Lambda Restrictions
    //
    // __device__ template variables cannot be instantiated using a type defined in host code, so this doesn't work:
    // template <class Fn_t> __device__ WrappedFnPointer wrapper_ptr = wrapper<WorkNode<Fn_t>>;
    // __LIBGPU_HIP_CHECK__(hipMemcpyFromSymbol(&temp, HIP_SYMBOL(wrapper_ptr<Callable_t>), sizeof(temp), 0,
    // hipMemcpyDeviceToHost)); https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 14.5.12 Restrictions ->
    // Templates
    //
    // If there's some other way of bundling the arguments up, and passing them to the device along with a function that
    // accepts those arguments, we could use that and avoid a kernel launch from the host.

    // Note that we only do this once for a given set of Fn_t and Args_t types
    static WrappedFnPointer saved_wrapper_fn = []() {
        WrappedFnPointer *tmp, *tmp_d;
        __LIBGPU_HIP_CHECK__(hipHostMalloc(reinterpret_cast<void **>(&tmp), sizeof(tmp), hipHostRegisterMapped));
        __LIBGPU_HIP_CHECK__(hipHostGetDevicePointer(reinterpret_cast<void **>(&tmp_d), tmp, 0));
        hipLaunchKernelGGL(getWrapperFn<WorkNode<Callable_t>>, dim3(1), dim3(1), 0, getEnqueingStream(), tmp_d);
        __LIBGPU_HIP_CHECK__(hipStreamSynchronize(getEnqueingStream()));
        // TODO: Memory Leak! We can't un-register or free tmp because of the implicit hipDeviceSynchronize() that would
        // cause. However, this should only be a small amount of memory, and because this code only runs once per
        // specialization of the WorkNode class, it cannot grow indefinitely.
        return *tmp;
    }();
    wrapper_fn = saved_wrapper_fn;
}

// TODO: many of these might be better as member functions of WorkNode or WorkNode_Header, and we could potentially make
// a bunch of WorkNode_Header's member variables private

// Post-condition: worknode is likely unlocked and may get invalidated
__device__ void insertWorkNodeIntoMainQueue(WorkNode_Header *worknode);
__host__ WorkNode_Header *sendWorkNodeToGPU(WorkNode_Header *worknode_h);

} // namespace gpu::internal

#endif // __GPU___THREAD_WORKITEM_H__
