#include "gpu/thread"

namespace gpu::internal {

__device__ uint32_t numVcores = NUM_VCORES;

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

} // namespace gpu::internal
