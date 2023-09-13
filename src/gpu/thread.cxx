#include "gpu/thread"

namespace gpu::internal {

__device__ uint32_t numVcores = NUM_VCORES;

__device__ WorkNode_Header *workListHead = nullptr;
__device__ WorkNode_Header **workListTail = &workListHead;
__device__ WorkNode_Header *volatile currentWorkNode[NUM_VCORES] = {};

hipStream_t mainStream;
bool started = false;
__device__ volatile bool finishing = false;

} // namespace gpu::internal
