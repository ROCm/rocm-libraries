#ifndef __LIBHIPTHREADS___THREAD_WORKITEM_H__
#define __LIBHIPTHREADS___THREAD_WORKITEM_H__

/**
 * @file
 * @brief Internal work node structures for GPU wthread scheduling.
 * @ingroup thread
 *
 *
 * Defines the low-level work node (WorkNode_Header, WorkNode<Callable_t>)
 * that backs hip::wthread.
 */

#include <memory>

#include "hip/hip_runtime.h"

#include <hip/std/__functional/invoke.h>

#include "hip/__clib/malloc.h"
#include "hip/__thread/id.h"

namespace cuda::internal {

struct WorkNode_Header;

/// Type-erased function pointer for invoking a work node's callable.
typedef void (*WrappedFnPointer)(WorkNode_Header *, bool);

// Info about the wthread itself that the user might query. (As opposed to info the scheduler uses behind the scenes)
/**
 * @struct ThreadData
 * @brief User-visible metadata about a logical wthread.
 *
 *
 * Stores width (active lane count) and a base thread id. The actual
 * per-lane id is derived from base + lane_index (see this_thread::get_id).
 */
struct ThreadData {
    // How many threads per block/vthread are active.
    uint32_t width = 0;
    // TODO: should this be a hip::wthread::max_width() array? For now we just store a common "base" id. See
    // this_thread::get_id for details on how the base id is converted to a full thread id.
    __thread_id::underlying_type vthread_id = {};

    __device__ ThreadData() = default;
    __host__ ThreadData() = default;

    __device__ ThreadData(uint32_t width);
    __host__ ThreadData(uint32_t width);
  private:
    static __device__ __thread_id::underlying_type nextTid();
    static __host__ __thread_id::underlying_type nextTid();
};

/**
 * @struct WorkNodeDeleter
 * @brief Custom deleter for unique_ptr<WorkNode_Header>.
 *
 *
 * Calls `hip::free` (work nodes are allocated via `hip::malloc`).
 */
struct WorkNodeDeleter {
    // WorkNode<T> is trivially destructible (implied by ::std::is_trivially_copyable).
    // Note: Technically, we should do a static_cast of ptr back to WorkNode<T> before freeing. If we really want to fix
    // it, we could give unique_ptr a function pointer instead of a functor (i.e. the type of worknode_d would be
    // ::std::unique_ptr<WorkNode_Header, void (*)(WorkNode_Header *)>), and at construction time pass a pointer to a
    // function that will do the cast before the free.
    void operator()(WorkNode_Header* ptr) { hip::free(ptr); }
};

// TODO: Can we make a bunch of these members private?
/**
 * @struct WorkNode_Header
 * @brief Type-erased base for a scheduled unit of GPU work.
 */
struct WorkNode_Header {
    /// Points to instantiation of wrapper<WorkNode<Callable_t>>.
    const WrappedFnPointer wrapper_fn;
    /// Data the user might query (width, base id).
    const ThreadData tdata;

    /**
     * @brief Stores sizeof(WorkNode<T>) for memcpy.
     *
     * Stores the sizeof(WorkNode<T>), so we can do a memcpy without knowing T
     * (e.g. in wthread::detach()).
     *
     * Not initialized when a WorkNode is constructed from device code because
     * we don't need it.
     */
    const size_t worknodeSize = 0;

    /**
     * @brief Double pointer for locking and wthread::detach().
     *
     * Enables wthread::detach() to copy a worknode into memory the gpu can free on its own.
     *
     * It can be either &mainWorkQueue[i], &cpuWorkQueue[i], &currentWorkNode[blockIdx.x], &(prev->next) or nullptr.
     *
     * Locking semantics:
     *  - *link_to_self == this: Unlocked.
     *  - *link_to_self == nullptr (or another node during move): Locked.
     *  - link_to_self == nullptr: In wthread::detach() and wthread::join() - execution is complete.
     *                             In the scheduler - WorkNode has been detached.
     */
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

    /// Pointer to the next waiting/yielded node the scheduler will resume on completion.
    WorkNode_Header *next = nullptr;
    /// Flag distinguishing null next (no waiters vs next WorkNode is currently locked).
    bool hasWaiting = false;

    /// Factory (host): allocates + constructs WorkNode with lambda wrapping callable + args.
    template <class Fn_t, class... Args_t>
    static __host__ auto make_worknode(uint32_t width, Fn_t &&typed_fn, Args_t &&...args);

    /// Factory (device): placement new into malloc buffer.
    template <class Fn_t, class... Args_t>
    static __device__ auto make_worknode(uint32_t width, Fn_t &&typed_fn, Args_t &&...args);

    /**
     * @brief Attempts to lock the WorkNode at location.
     * @return A pointer to the WorkNode if ownership obtained; nullptr if already locked.
     */
    [[nodiscard]] static __device__ WorkNode_Header *tryLockAndFetch(WorkNode_Header **location);

    /**
     * @brief Locks and returns the WorkNode at location.
     *
     * If *location == nullptr (i.e. it's currently locked), spins until the WorkNode is unlocked.
     */
    [[nodiscard]] static __device__ WorkNode_Header *lockAndFetch(WorkNode_Header **location);

    /**
     * @brief Lock a worknode we already have a pointer to (for join/detach).
     * @return worknode->link_to_self (which might be nullptr if the node has finished executing).
     *
     * This function is not safe to call from the scheduler, and is only meant
     * for use in join and detach, where we can be sure that the worknode pointer
     * isn't going to be invalidated unexpectedly while we are trying to acquire
     * the lock for it.
     */
    __device__ WorkNode_Header **lock();

    /// Unlock after inline query (e.g. get_width), where node hasn't moved and we know *link_to_self == nullptr.
    __device__ void unlockActive();

    /// Move the worknode to *new_location and unlock.
    __device__ void moveAndUnlock(WorkNode_Header **new_location);

    /**
     * @brief Either free ourself, or tell someone else to.
     * @return true if we free'd ourselves
     *
     * Signal that the caller is abdicating any responsability for freeing the
     * WorkNode, unless the caller is the last one to do so, in which case, we
     * free ourself. I.e. If the WorkNode has already been detached, or the
     * WorkNode has already finished execution, we free ourselves.
     */
    __device__ bool release();

    /// Check if scheduler no longer references this node.
    __device__ bool isSchedulerDoneWith();

    /// Mark as current work node and unlock.
    __device__ void makeCurrent(bool yielding);

    /**
     * @brief Insert into main scheduler queue.
     * @post worknode is likely unlocked and may get invalidated
     */
    __device__ void insertIntoMainQueue();

    /// Transfer node to device memory for execution (host utility).
    __host__ ::std::unique_ptr<WorkNode_Header, WorkNodeDeleter> sendToGPU();
    __host__ ::std::unique_ptr<WorkNode_Header, WorkNodeDeleter> sendToGPU(WorkNode_Header **new_location);
};
static_assert(::std::is_standard_layout_v<WorkNode_Header>);

/**
 * @struct WorkNode
 * @brief Templated work node holding a concrete Callable.
 * @tparam Callable_t Move-constructible callable type.
 *
 *
 * Extends WorkNode_Header with typed member `fn` (the callable).
 * Constructors capture callable; wrapper<WorkNode<Callable_t>> extracts
 * and invokes it.
 */
template <class Callable_t>
struct WorkNode : WorkNode_Header {
    using Callable = Callable_t;
    WorkNode(WorkNode &&other) = default;
    inline __device__ WorkNode(uint32_t width, Callable &&callable);
    inline __host__ WorkNode(uint32_t width, Callable &&callable);

    Callable fn;
};

template <class Fn_t, class... Args_t>
__host__ auto WorkNode_Header::make_worknode(uint32_t width, Fn_t &&typed_fn, Args_t &&...args) {
    // Ideally, we would also forward args in the capture (...args = ::std::forward<Args_t>(args)) to avoid an extra copy,
    // but that requires C++20
    auto lambda = [typed_fn = ::std::forward<Fn_t>(typed_fn), args...] __device__() mutable -> void {
        cuda::std::invoke(::std::move(typed_fn), ::std::move(args)...);
    };
    using WorkNode_t = WorkNode<decltype(lambda)>;
    WorkNode_t *worknode_ptr = new WorkNode_t(width, ::std::move(lambda));
    // Sadly, hipHostUnregister performs an implicit device-wide synchronization. Thus, in order to use pinned host
    // memory for the async copy, we would either end up with a gradually growing amount of pinned memory, or need to
    // re-use the same pinned memory every time.
    // __LIBHIPTHREADS_HIP_CHECK__(hipHostRegister(worknode_ptr, sizeof(WorkNode_t), hipHostRegisterDefault));

    return ::std::unique_ptr<WorkNode_t>(worknode_ptr);
}
template <class Fn_t, class... Args_t>
__device__ auto WorkNode_Header::make_worknode(uint32_t width, Fn_t &&typed_fn, Args_t &&...args) {
    // These will give a more user-friendly error message when the lambda is not move-constructible.
    static_assert(::std::is_move_constructible_v<Fn_t>);
    static_assert((::std::is_move_constructible_v<Args_t> && ...));

    // Ideally, we would also forward args in the capture (...args = ::std::forward<Args_t>(args)) to avoid an extra copy,
    // but that requires C++20
    auto lambda = [typed_fn = ::std::forward<Fn_t>(typed_fn), args...] () mutable {
        cuda::std::invoke(::std::move(typed_fn), ::std::move(args)...);
    };

    // Allocate memory using malloc instead of new, to guaranteed that ::free(worknode) is valid.
    // The C++ standard doesn't guarantee that new and malloc allocate from the same pool of memory.
    void *buf = ::malloc(sizeof(WorkNode<decltype(lambda)>));
    return new(buf) WorkNode<decltype(lambda)>(width, ::std::move(lambda));
}

//====================================================================================================================//
//      INTERNAL/HELPER FUNCTIONS
//====================================================================================================================//

// Precondition: We're still holding the lock acquired in invokeNext. Needed to make sure detach doesn't make worknode
// an invalid pointer before we load typed_node_ptr and width.
/**
 * @brief Type-erased wrapper function (device kernel invoked by scheduler).
 * @tparam WorkNode_t Concrete WorkNode<Callable> type.
 * @param worknode Pointer to work node.
 * @param yielding True if invocation is part of a yield path.
 *
 *
 * Extracts callable, moves it out, unlocks node (makeCurrent), then invokes
 * with active lanes (threadIdx.x < width).
 */
template <class WorkNode_t>
__device__ void wrapper(WorkNode_Header *worknode, bool yielding) {
    WorkNode_t *typed_node_ptr = static_cast<WorkNode_t *>(worknode);
    uint32_t width = typed_node_ptr->tdata.width;

    using Callable_t = typename WorkNode_t::Callable;
    __shared__ Callable_t *fn_ptr;

    if (threadIdx.x == 0) {
        fn_ptr = new Callable_t(::std::move(typed_node_ptr->fn));
        typed_node_ptr->fn.~Callable_t();
    }

    __syncthreads();
    // Include a threadfence for all threads just to be safe.
    __threadfence();
    // Also unlocks worknode
    if (threadIdx.x == 0)
        worknode->makeCurrent(yielding);
    if (threadIdx.x < width) {
        (*fn_ptr)();
    }
    __threadfence();

    if (threadIdx.x == 0) {
        delete fn_ptr;
    }

    // TODO: figure out how to make all the threads with idx > width 'catch up' on missed __syncthreads() calls.
    // Shouldn't be a concern as long as blockDim.x == hip::wthread::max_width() == warpSize
}

template <class Callable_t>
__device__ WorkNode<Callable_t>::WorkNode(uint32_t w, Callable_t &&callable)
    : WorkNode_Header{wrapper<WorkNode<Callable_t>>, ThreadData(w)}, fn(::std::move(callable)) {}

/**
 * @brief Host helper kernel: stores device pointer to wrapper<WorkNode_t> in output.
 *
 */
template <class WorkNode_t>
__global__ void getWrapperFn(WrappedFnPointer *ptr) {
    *ptr = wrapper<WorkNode_t>;
}

/**
 * @brief Host utility: obtain device-side wrapper function pointer for a given Callable_t.
 *
 *
 * Launches getWrapperFn kernel once per specialization, caches result.
 */
template <class Callable_t>
__host__ WrappedFnPointer getWrapperFn() {
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
    // __LIBHIPTHREADS_HIP_CHECK__(hipMemcpyFromSymbol(&temp, HIP_SYMBOL(wrapper_ptr<Callable_t>), sizeof(temp), 0,
    // hipMemcpyDeviceToHost)); https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 14.5.12 Restrictions ->
    // Templates
    //
    // If there's some other way of bundling the arguments up, and passing them to the device along with a function that
    // accepts those arguments, we could use that and avoid a kernel launch from the host.

    // Note that we only do this once for a given set of Fn_t and Args_t types
    static WrappedFnPointer saved_wrapper_fn = []() {
        WrappedFnPointer *tmp, *tmp_d;
        __LIBHIPTHREADS_HIP_CHECK__(hipHostMalloc(reinterpret_cast<void **>(&tmp), sizeof(tmp), hipHostRegisterMapped));
        __LIBHIPTHREADS_HIP_CHECK__(hipHostGetDevicePointer(reinterpret_cast<void **>(&tmp_d), tmp, 0));
        hipLaunchKernelGGL(getWrapperFn<WorkNode<Callable_t>>, dim3(1), dim3(1), 0, getEnqueingStream(), tmp_d);
        __LIBHIPTHREADS_HIP_CHECK__(hipStreamSynchronize(getEnqueingStream()));
        // TODO: Memory Leak! We can't un-register or free tmp because of the implicit hipDeviceSynchronize() that would
        // cause. However, this should only be a small amount of memory, and because this code only runs once per
        // specialization of the WorkNode class, it cannot grow indefinitely.
        return *tmp;
    }();
    return saved_wrapper_fn;
}

template <class Callable_t>
__host__ WorkNode<Callable_t>::WorkNode(uint32_t w, Callable_t &&callable)
    : WorkNode_Header{getWrapperFn<Callable_t>(), ThreadData(w), sizeof(WorkNode<Callable_t>)},
      fn(::std::move(callable)) {}

} // namespace cuda::internal

#endif // __LIBHIPTHREADS___THREAD_WORKITEM_H__
