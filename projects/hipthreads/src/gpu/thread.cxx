#include "gpu/thread"

namespace gpu::internal {

__device__ uint32_t numVcores = NUM_VCORES;

__device__ volatile uint32_t nextVthreadId_stored = 0;
__device__ volatile uint32_t nextVthreadId_launched = 0;

__device__ volatile WrappedFnPointer wrapper_table[MAX_VTHREADS];
__device__ void *volatile fn_table[MAX_VTHREADS];

// TODO: once we have a proper thread::id class, the default constructor for it will correspond to an invalid vthreadId
__shared__ gpu::internal::thread::id currentVthreadId; // TODO: then we can remove this initializer

hipStream_t mainStream;
bool started = false;

} // namespace gpu::internal
