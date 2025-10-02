// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef PLAN_H
#define PLAN_H

#include <array>
#include <complex>
#include <cstring>
#include <list>
#include <vector>

#include "../../../shared/array_predicate.h"
#include "function_pool.h"
#include "load_store_ops.h"
#include "rocfft_mpi.h"
#include "tree_node.h"

// Calculate the maximum pow number with the given base number
template <int base>
constexpr size_t PowMax()
{
    size_t u = base;
    while(u < std::numeric_limits<size_t>::max() / base)
    {
        u = u * base;
    }
    return u;
}

// types of grid layouts for global transpositions
enum class grid_layout
{
    invalid = 0,
    slab    = 1,
    pencil  = 2,
    brick   = 3
};
using transpose_type = std::pair<grid_layout, grid_layout>;

// Generic function to check is pow of a given base number or not
template <int base>
static inline bool IsPow(size_t u)
{
    constexpr size_t max = PowMax<base>(); //Practically, we could save this by using 3486784401
    return (u > 0 && max % u == 0);
}

struct rocfft_brick_t
{
    // all vectors here are column-major, with same length as FFT
    // dimension + 1 (for batch dimension)

    // inclusive lower bound of brick
    std::vector<size_t> lower;
    // exclusive upper bound of brick
    std::vector<size_t> upper;
    // stride of brick in memory
    std::vector<size_t> stride;

    // Location of the brick
    rocfft_location_t location;

    // Compute the length of this brick
    std::vector<size_t> length() const
    {
        std::vector<size_t> ret;
        for(size_t i = 0; i < lower.size(); ++i)
            ret.push_back(upper[i] > lower[i] ? upper[i] - lower[i] : 0);
        return ret;
    }

    // Functions and operators

    // check if brick is empty
    bool empty() const;
    // return intersection of *this and another brick.  note that
    // strides and device are not set on the returned brick, as this
    // method can't know if the caller wants to look at the result in
    // *this or in other.
    rocfft_brick_t intersect(const rocfft_brick_t& other) const;

    // test whether this brick covers same coordinates as another
    // brick.  strides are not considered.
    bool equal_coords(const rocfft_brick_t& other) const;

    // compute the number of elements in this brick
    size_t count_elems() const;
    bool   is_contiguous() const;
    // Return strides for this brick, if it were transposed to be
    // contiguous.
    std::vector<size_t> contiguous_strides() const;

    // return true if this brick is contiguous in the specified field
    bool is_contiguous_in_field(const std::vector<size_t>& field_length,
                                const std::vector<size_t>& field_stride) const;

    // compute offset of this brick, given the field's stride
    size_t offset_in_field(const std::vector<size_t>& fieldStride) const;

    bool operator==(const rocfft_brick_t& other) const
    {
        return lower == other.lower && upper == other.upper && stride == other.stride
               && location == other.location;
    }

    std::string str() const;
};

struct rocfft_field_t
{
    std::vector<rocfft_brick_t> bricks;
};

struct rocfft_plan_description_t
{
    rocfft_array_type inArrayType  = rocfft_array_type_unset;
    rocfft_array_type outArrayType = rocfft_array_type_unset;

    std::vector<size_t> inStrides;
    std::vector<size_t> outStrides;

    size_t inDist  = 0;
    size_t outDist = 0;

    std::array<size_t, 2> inOffset  = {0, 0};
    std::array<size_t, 2> outOffset = {0, 0};

    std::vector<rocfft_field_t> inFields;
    std::vector<rocfft_field_t> outFields;

    // Multi-process communicator info:
    rocfft_comm_type comm_type = rocfft_comm_none;
#ifdef ROCFFT_MPI_ENABLE
    // this is the communicator that was directly provided by the user
    // - we don't own it so it doesn't need to be wrapped or freed
    MPI_Comm user_mpi_comm = MPI_COMM_NULL;
#endif

    LoadOps  loadOps;
    StoreOps storeOps;

    rocfft_plan_description_t()  = default;
    ~rocfft_plan_description_t() = default;

    // A plan description is created in a vacuum and does not know what
    // type of transform it will be for.  Once that's known, we can
    // initialize default values for in/out type, stride, dist if they're
    // unspecified.
    void init_defaults(rocfft_transform_type      transformType,
                       rocfft_result_placement    placement,
                       const std::vector<size_t>& lengths,
                       const std::vector<size_t>& outputLengths);

    // Count the number of pointers required for either input or output
    // - planar data requires two pointers, real + complex require one.
    // But if fields are declared then the number of pointers is the
    // number of bricks in the fields.
    static size_t count_pointers(const std::vector<rocfft_field_t>& fields,
                                 rocfft_array_type                  arrayType,
                                 int                                comm_rank)
    {
        if(fields.empty())
            return array_type_is_planar(arrayType) ? 2 : 1;
        size_t fieldPtrs = 0;
        for(auto& f : fields)
        {
            fieldPtrs += std::count_if(
                f.bricks.begin(), f.bricks.end(), [comm_rank](const rocfft_brick_t& b) {
                    return b.location.comm_rank == comm_rank;
                });
        }
        return fieldPtrs;
    }

    // returns true if a field has bricks such that any rank has
    // bricks on more than one device
    static bool multiple_devices_in_rank(const rocfft_field_t& field);
};

struct rocfft_plan_t
{
#ifdef ROCFFT_MPI_ENABLE
    MPI_Comm_wrapper_t mpi_comm;
#endif
    size_t rank = 0;
    // input lengths
    std::vector<size_t> lengths;
    // output lengths, which differ from input lengths for real-complex
    // transforms
    std::vector<size_t> outputLengths;
    size_t              batch = 1;

    rocfft_result_placement placement     = rocfft_placement_inplace;
    rocfft_transform_type   transformType = rocfft_transform_type_complex_forward;
    rocfft_precision        precision     = rocfft_precision_single;

    rocfft_plan_description_t desc;

    rocfft_plan_t() = default;

    // Users can provide lengths+strides in any order, but we'll
    // construct the most sensible plans if they're in row-major order.
    // Sort the FFT dimensions.
    //
    // This should be done when the plan parameters are known, but
    // before we start creating any child nodes from the root plan.
    void sort();

    static bool is_contiguous(const std::vector<size_t>& length,
                              const std::vector<size_t>& stride,
                              size_t                     dist);
    bool        is_contiguous_input();
    bool        is_contiguous_output();

    // Add a multi-plan item for execution.  Returns the index of the
    // new item in the overall multi-GPU plan.  Also provide a
    // vector of indexes of other items that must complete before this
    // item can run.
    size_t AddMultiPlanItem(std::unique_ptr<MultiPlanItem>&& item,
                            const std::vector<size_t>&       antecedents);

    // Add a new antecedent for an existing item index
    void AddAntecedent(size_t itemIdx, size_t antecedentIdx);

    // Execute the multi-GPU plan.
    void Execute(void* in_buffer[], void* out_buffer[], rocfft_execution_info info);

    size_t WorkBufBytes() const;

    // Insert core execPlan into multi-item plan, surrounding it with
    // sufficient items to gather/scatter to/from a single device if
    // the plan needs it.  Gathering all the data to a single device is
    // suboptimal but is a first step towards proper multi-device
    // logic.
    void GatherScatterSingleDevicePlan(std::unique_ptr<ExecPlan>&& execPlan);

    // Construct an optimized multi-device plan for the FFT
    // parameters in *this.  Returns false if:
    // - multiple devices are not requested for this FFT, or
    // - we have no particular optimization for this FFT and we'll need
    //   to fall back to a single-device plan
    bool BuildOptMultiDevicePlan();

    // check log level, log the topologically sorted plan if plan
    // logging is enabled
    void LogSortedPlan(const std::vector<size_t>& sortedIdx) const;

    // log field layout at plan level
    static void LogFields(const char* description, const std::vector<rocfft_field_t>& fields);

    // throw exception if input/output fields are not valid (e.g. they
    // don't cover the whole index space, or bricks overlap)
    void ValidateFields() const;

    // Get the local communication rank
    int get_local_comm_rank() const;
    // Get number of ranks in the local communicator
    int get_local_comm_size() const;

    // During plan creation, InternalTempBuffer remembers how much
    // space will be needed but doesn't allocate.  Allocate the buffers
    // after the space requirements are finalized.
    void AllocateInternalTempBuffers();

private:
    // Multi-node or multi-GPU plan is built up from a vector of plan
    // items.  Items can launch kernels on a device, or move
    // data between devices.
    std::vector<std::unique_ptr<MultiPlanItem>> multiPlan;

    // Communicate bricks on all ranks to all other ranks
    rocfft_status allgather_brick_params_mpi(rocfft_plan& plan);

    // Adjacency list describing dependencies between multiPlan items.
    // Size of this vector == multiPlan.size().
    //
    // The size_t's at multiPlanAntecedents[i] are the indexes in
    // multiPlan that need to complete before multiPlan[i] can run
    // (i.e. its antecedents).
    std::vector<std::vector<size_t>> multiPlanAntecedents;

    // Return a stack of multiPlan indexes that are in topological
    // order.  Traverse this vector in reverse order to follow the
    // sorting.
    std::vector<size_t> MultiPlanTopologicalSort() const;

    // Recursive utility function to do depth-first search.  tracks
    // visited indexes as it goes along.
    void TopologicalSortDFS(size_t               idx,
                            std::vector<bool>&   visited,
                            std::vector<size_t>& sorted) const;

    // Temp buffers allocated during plan creation for multi-device
    // plans are remembered here.  Mapped per-location.  Individual
    // plan items can have void*'s that point to these buffers.
    std::multimap<rocfft_location_t, std::shared_ptr<InternalTempBuffer>> tempBuffers;

    // gather a set of bricks to a field on the current device
    std::vector<size_t> GatherBricksToField(rocfft_location_t                  destLocation,
                                            const std::vector<rocfft_brick_t>& bricks,
                                            rocfft_precision                   precision,
                                            rocfft_array_type                  arrayType,
                                            const std::vector<size_t>&         field_length,
                                            const std::vector<size_t>&         field_stride,
                                            BufferPtr                          output,
                                            const std::vector<size_t>&         antecedents,
                                            size_t                             elem_size);

    // scatter a field on the current device to a set of bricks
    std::vector<size_t> ScatterFieldToBricks(rocfft_location_t                  srcLocation,
                                             BufferPtr                          input,
                                             rocfft_precision                   precision,
                                             rocfft_array_type                  arrayType,
                                             const std::vector<size_t>&         field_length,
                                             const std::vector<size_t>&         field_stride,
                                             const std::vector<rocfft_brick_t>& bricks,
                                             const std::vector<size_t>&         antecedents,
                                             size_t                             elem_size);

    // Transpose the input field to the output field by adding work items
    // to the plan.  Antecedents are provided as a vector of item
    // indexes, one per brick.  Final work item per brick (that future
    // per-brick operations can depend on) is returned in outputItems.
    //
    // transposeNumber identifies this particular transpose in the
    // plan, for debugging.
    void GlobalTranspose(size_t                     elem_size,
                         const rocfft_field_t&      inField,
                         const rocfft_field_t&      outField,
                         std::vector<BufferPtr>&    input,
                         std::vector<BufferPtr>&    output,
                         const std::vector<size_t>& inputAntecedents,
                         std::vector<size_t>&       outputItems,
                         size_t                     transposeNumber);

    // default global all-to-all transpose
    void GlobalTransposeA2A(size_t                     elem_size,
                            const rocfft_field_t&      inField,
                            const rocfft_field_t&      outField,
                            std::vector<BufferPtr>&    input,
                            std::vector<BufferPtr>&    output,
                            const std::vector<size_t>& inputAntecedents,
                            std::vector<size_t>&       outputItems,
                            const std::string&         itemGroup);

    // fallback case for global transpose that uses point-to-point
    // communications, for when all-to-all isn't possible.
    void GlobalTransposeP2P(size_t                     elem_size,
                            const rocfft_field_t&      inField,
                            const rocfft_field_t&      outField,
                            std::vector<BufferPtr>&    input,
                            std::vector<BufferPtr>&    output,
                            const std::vector<size_t>& inputAntecedents,
                            std::vector<size_t>&       outputItems,
                            const std::string&         itemGroup);

    // Transform (complex-complex FFT) a whole field along specified
    // dimensions.  Input and output ptrs are provided as a vector of
    // BufferPtrs, one per brick in the field.
    //
    // Input antecedents, if provided, are the last items from the
    // previous global operation (e.g. a global transpose).  Operations
    // in this transform will depend on those antecedents that touch the
    // same buffers.
    //
    // Work items are added to the plan.  Final work item per brick (that
    // future per-brick operations can depend on) is returned in
    // outputItems.
    void C2CField(const rocfft_field_t&      field,
                  const std::vector<size_t>& fftDims,
                  std::vector<BufferPtr>&    input,
                  std::vector<BufferPtr>&    output,
                  const std::vector<size_t>& inputAntecedents,
                  std::vector<size_t>&       outputItems);
};

bool PlanPowX(ExecPlan& execPlan);
bool GetTuningKernelInfo(ExecPlan& execPlan);
void RuntimeCompilePlan(ExecPlan& execPlan);

#endif // PLAN_H
