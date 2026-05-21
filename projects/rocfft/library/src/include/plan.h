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
#include <optional>
#include <set>
#include <vector>

#include "../../../shared/array_predicate.h"
#include "data_layout.h"
#include "exec_info.h"
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
    // no default constructor
    rocfft_brick_t() = delete;
    // default move and copy constructors
    rocfft_brick_t(const rocfft_brick_t&) = default;
    rocfft_brick_t& operator=(const rocfft_brick_t&) = default;
    rocfft_brick_t(rocfft_brick_t&&)                 = default;
    rocfft_brick_t& operator=(rocfft_brick_t&&) = default;

    // all vectors here are column-major, with same size as FFT
    // rank + 1 (batch axis last)
    rocfft_brick_t(const std::vector<size_t>& field_lower,
                   const std::vector<size_t>& field_upper,
                   const std::vector<size_t>& brick_stride,
                   const rocfft_location_t&   location)
        : layout(field_lower, field_upper, brick_stride)
        , location(location)
    {
    }
    rocfft_brick_t(const data_layout_t& brick_layout, const rocfft_location_t& location)
        : layout(brick_layout)
        , location(location)
    {
    }

    // Data layout of the brick
    data_layout_t layout;
    // Location of the brick
    rocfft_location_t location;

    bool operator==(const rocfft_brick_t& other) const
    {
        return layout == other.layout && location == other.location;
    }

    std::string str() const;
};

struct rocfft_field_t
{
    std::vector<rocfft_brick_t> bricks;

    /**
     * @return the minimal full (i.e., non-partial) data layout that logically includes
     * all bricks' partial layouts. The returned layout is set with contiguous in-buffer
     * strides.
     * 
     * @throw An `std::logic_error` is thrown if the field has no brick or some
     * dimensionally inconsistent bricks. 
     */
    data_layout_t get_full_data_range() const;

    /**
     * @brief Finalizes all the field's bricks, i.e., sorts them by increasing rank (stable
     * sort) and sets the `is_partial` flags for all axes of their layouts.
     */
    void finalize();

    /**
     * @return true iff all parts of the field's full range of logical indices are covered
     * once and only once by the field's bricks.
     */
    bool has_valid_tessellation() const;

    /**
     * @param[in] dim dimension index of interest (flattened: length axes
     * followed by batch axes)
     * @return `true` iff the corresponding length dimension is not partial
     * in any of the field's bricks.
     * 
     * @throw An `std::logic_error` is thrown if the field has no bricks. An
     * `std::invalid_argument` is thrown if `dim` is out of range.
     */
    bool has_undistributed_axis(size_t dim) const;

    /**
     * @return An `std::set<size_t>` of the length dimensions that are not
     * partial in any of the field's bricks.
     * 
     * @throw An `std::logic_error` is thrown if the field has no bricks.
     */
    std::set<size_t> get_undistributed_length_dimensions() const;

    /**
     * @brief Verifies whether this field is consistent as input (resp. output)
     * for embarrassingly-parallel computations of specific types of Fourier
     * Transforms and, if so, returns the corresponding output (resp. input)
     * field. All of this object's bricks must have full (i.e., non-partial)
     * length axes in their layouts.
     * 
     * @param[in] other_io I/O label for the field to be returned. Explicitly,
     * the calling object's layout is considered an input (resp. output) layout
     * if the argument value is `io_data_label::OUTPUT` (resp. `io_data_label::INPUT`)
     * @param[in] fft_type intended type of Fourier Transform
     * @param[in] placement intended placement of each brick's results for the
     * desired transform. For out-of-place operations, each brick in the
     * returned field is given a default, contiguous data layout.
     * @param[in] other_innermost_length_is_odd flag indicating whether the
     * logical span of the innermost length axis in the corresponding field's
     * full data range is odd (if `true`) or not (if `false`). This is ignored
     * (and can be safely omitted in calls) *unless* the field to be returned
     * corresponds to the input of a real forward transform or the output of a
     * real inverse transform.
     * @return An `std::optional<rocfft_field_t>` object which has a value set
     * iff a corresponding field for the desired operation does actually exist.
     * The returned value may have no value set for in-place operations. When a
     * value is set, the returned field has as many bricks as the current object
     * (sharing the same locations, brick-wise).
     * 
     * @throw An `std::logic_error` is thrown if the current object has no brick,
     * or if any of its bricks has some partial length axis in its data layout.
     */
    std::optional<rocfft_field_t>
        get_other_embarrassingly_parallel_io_field(io_data_label           other_io,
                                                   rocfft_transform_type   fft_type,
                                                   rocfft_result_placement placement,
                                                   bool other_innermost_length_is_odd
                                                   = false) const;

    inline bool operator==(const rocfft_field_t& other) const
    {
        return bricks == other.bricks;
    }
};

struct rocfft_plan_description_t
{
    rocfft_array_type inArrayType  = rocfft_array_type_unset;
    rocfft_array_type outArrayType = rocfft_array_type_unset;

    data_layout_t input_layout;
    data_layout_t output_layout;

    std::array<size_t, 2> inOffset  = {0, 0};
    std::array<size_t, 2> outOffset = {0, 0};

    std::vector<rocfft_field_t> inFields;
    std::vector<rocfft_field_t> outFields;

    // Multi-process communicator info:
    rocfft_comm_type comm_type = rocfft_comm_none;
#ifdef ROCFFT_MPI_ENABLE
    // Wrapper to a unique mpi communicator, duplicated from the user-provided
    // one (when one is given) and set with our own error policy handling
    MPI_Comm_wrapper_t mpi_comm;
#endif

    LoadOps  loadOps;
    StoreOps storeOps;

    rocfft_plan_description_t()  = default;
    ~rocfft_plan_description_t() = default;

    /**
     * @return The number of length dimensions.
     */
    size_t rank() const;
    /**
     * @return The batch size.
     */
    size_t batch() const;

    // Get the local communication rank
    int get_local_comm_rank() const;
    // Get number of ranks in the local communicator
    int get_local_comm_size() const;
    // Returns the current rocfft_location_t (process rank + current device ID)
    // seen by this object
    rocfft_location_t get_current_location() const;

    /**
     * @brief Finalizes the plan description by
     * 
     * - assigning default values for structure members that have not been explicitly set yet;
     * 
     * - gathering all the fields' bricks on all processes involved in the description's
     *   communicator, if any;
     * 
     * - finalizing bricks for all member fields (see `rocfft_field_t::finalize()`);
     * 
     * - removing trivial unit-span axes from all layouts (including bricks' if any)
     * and sorting length axes by increasing strides if that can be done consistently
     * across all of them.
     * 
     * The description is also validated w.r.t. the following criteria:
     * 
     * - the number of user-defined strides matches the rank of the planned transform;
     * 
     * - I/O fields are dimensionally consistent with the planned transform, cover the
     *   expected full ranges of logical indices and have valid tessellations;
     * 
     * - Input and output fields are both set for multi-process usage;
     * 
     * - Usage of multiple I/O fields in descriptors is reported as unsupported;
     * 
     * - I/O array types are not planar if the corresponding field(s) are set;
     * 
     * - I/O array types are consistent with the expected data types;
     * 
     * - I/O array types are consistent for in-place transforms (if in-place is requested);
     * 
     * - locations of I/O buffers are consistent for in-place transforms (if in-place is requested);
     * 
     * - I/O data layouts and offsets (unless ignored) are consistent with in-place usage
     *   for single-device transforms (if in-place is requested);
     *
     * @param[in] dft_type user-provided type of transform for the owning plan.
     * @param[in] placement user-provided placement of transform results for the owning plan.
     * @param[in] user_lengths user-provided lengths of the transform for the owning plan.
     * @param[in] len_rank user-provided number of length dimensions for the owning plan.
     * @param[in] number_of_transforms user-provided batch size for the owning plan.
     * @return A `rocfft_status` value is returned, which should be escalated back to the user
     * as is if different from `rocfft_status_success`. An `std::runtime_error` with
     * insightful information (for logging purposes) is thrown instead if no dedicated error
     * code exists yet.
     */
    rocfft_status finalize_and_validate_for(rocfft_transform_type   dft_type,
                                            rocfft_result_placement placement,
                                            const size_t*           user_lengths,
                                            const size_t            len_rank,
                                            const size_t            number_of_transforms);

    /**
     * @brief Verifies if the description has undistributed input/output data and,
     * if so, returns the location thereof.
     * 
     * @param[in] io input (resp. output) data sets are considered for argument value
     * `io_data_label::INPUT` (resp. `io_data_label::OUTPUT`)
     * @return An `std::optional<rocfft_location_t>` object which has a value set to
     * the expected location of the input (resp. output) data, if undistributed. 
     * @throw An `std::invalid_argument` exception is thrown if `io` is not an
     * expected value.
     */
    std::optional<rocfft_location_t> expected_undistributed_location_for(io_data_label io) const;

    /**
     * @param[in] io input (resp. output) fields are considered for argument value
     * `io_data_label::INPUT` (resp. `io_data_label::OUTPUT`).
     * @param[in] field_idx index of the desired field.
     * @return A constant reference to the requested input (resp. output) field.
     * 
     * @throw An `std::invalid_argument` exception is thrown if `io` is not an
     * expected value, or if `field_idx` is out of bounds.
     */
    const rocfft_field_t& get_field_for(io_data_label io, size_t field_idx = 0) const;

    /**
     * @return `true` if the description is consistent with single-device
     * operations on the current location.
     */
    bool has_undistributed_io_on_current_location() const;

    /**
     * @brief Returns a read accessor for the input (resp. output) data layout that must take
     * precedence. If a lone-brick input (resp. output) field is used, this returns a (const)
     * reference to that brick's layout. Otherwise, a (const) reference to `input_layout`
     * (resp. `output_layout`) is returned.
     * 
     * @param[in] io input (resp. output) data set(s) are considered for argument
     * value `io_data_label::INPUT` (resp. `io_data_label::OUTPUT`)
     * 
     * @throw An `std::logic_error` is thrown if this description involves more than one brick
     * on input (resp. output), i.e., if this description does not have an undistributed data
     * layout on input (resp. output). An `std::invalid_argument` exception is thrown if `io`
     * is not an expected value.
     */
    const data_layout_t& undistributed_layout_for(io_data_label io) const;

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

    /**
     * @return true iff the input (resp. output) innermost length is odd for
     * argument value `io_data_label::INPUT` (resp. `io_data_label::OUTPUT`).
     * @throw An `std::invalid_argument` is thrown if the argument is neither
     * `io_data_label::INPUT` nor `io_data_label::OUTPUT`.
     */
    inline bool innermost_length_is_odd(io_data_label io) const
    {
        switch(io)
        {
        case io_data_label::INPUT:
            return input_layout.lengths().front() % 2 == 1;
        case io_data_label::OUTPUT:
            return output_layout.lengths().front() % 2 == 1;
        default:
            throw std::invalid_argument("Unknown io data label given to "
                                        + ROCFFT_CURRENT_FUNCTION);
        }
    }

    // returns true if a field has bricks such that any rank has
    // bricks on more than one device
    static bool multiple_devices_in_rank(const rocfft_field_t& field);

private:
#ifdef ROCFFT_MPI_ENABLE
    // Communicate bricks on all ranks to all other ranks
    rocfft_status allgather_brick_params_mpi();
    rocfft_status allgather_brick_params_lus_mpi(rocfft_field_t& field,
                                                 const size_t    global_brick_length);
#endif
    // I/O fields may not be set explicitly for single-device plan's input/output on
    // current device, yet it may be convenient to capture such cases via a lone-brick
    // field representation, internally (simplifying and unifying logic).
    std::optional<rocfft_field_t> single_dev_ifield, single_dev_ofield;
};

struct rocfft_plan_t
{
    rocfft_result_placement placement     = rocfft_placement_inplace;
    rocfft_transform_type   transformType = rocfft_transform_type_complex_forward;
    rocfft_precision        precision     = rocfft_precision_single;

    rocfft_plan_description_t desc;

    rocfft_plan_t() = default;

    // Add a multi-plan item for execution.  Returns the index of the
    // new item in the overall multi-GPU plan.  Also provide a
    // vector of indexes of other items that must complete before this
    // item can run.
    size_t AddMultiPlanItem(std::unique_ptr<MultiPlanItem>&& item,
                            const std::vector<size_t>&       antecedents);

    // Add a new antecedent for an existing item index
    void AddAntecedent(size_t itemIdx, size_t antecedentIdx);

    // Execute the multi-GPU plan.
    void Execute(void* in_buffer[], void* out_buffer[], const rocfft_execution_info_internal& info);

    std::vector<size_t> WorkBufBytesPerDevice() const;

    // Construct an optimized multi-device plan for the FFT
    // parameters in *this.  Returns false if:
    // - multiple devices are not requested for this FFT, or
    // - we have no particular optimization for this FFT and we'll need
    //   to fall back to a single-device plan
    bool BuildOptMultiDevicePlan();

    // Alternative path to BuildOptMultiDevicePlan (will eventually substitute it)
    // returns `true` iff a successful multi-device computing plan configuration
    // was set up
    bool BuildMultiDevicePlan();

    // check log level, log the topologically sorted plan if plan
    // logging is enabled
    void LogSortedPlan(const std::vector<size_t>& sortedIdx) const;

    // log field layout at plan level
    static void LogFields(const char* description, const std::vector<rocfft_field_t>& fields);

    // During plan creation, InternalTempBuffer remembers how much
    // space will be needed but doesn't allocate.  Allocate the buffers
    // after the space requirements are finalized.
    void AllocateInternalTempBuffers();

    /**
     * @return The lengths provided by the user at creation of this object
     * @note Trivial unit-length dimensions are erased at plan creation,
     * therefore *NOT* returned by this function.
     */
    std::vector<size_t> get_user_facing_lengths() const;
    /**
     * @brief Creates a plan execution item capable of tackling the plan's task
     * via a single-device execution (the current-location device, implicitly). If
     * the plan was not configured for single-device executions, the required
     * input-gathering and output-scattering execution items are also created.
     */
    void MakeSingleDevPlanWithGatherScatterIfNeeded();

private:
    // Multi-node or multi-GPU plan is built up from a vector of plan
    // items.  Items can launch kernels on a device, or move
    // data between devices.
    std::vector<std::unique_ptr<MultiPlanItem>> multiPlan;

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

    /**
     * @brief Creates the plan items required to gather the input data buffer(s) of a
     * multi-device transform into the input buffer of the (single-device) execution
     * plan (observing the input data layout set for that execution plan). If the input
     * data is undistributed and can be read directly by the (single-device) execution
     * plan, no input-gathering plan items are created.
     * 
     * @param[in] exec_plan_metadata single-device execution plan's metadata.
     * @param[in] exec_plan_location location of the single-device execution plan.
     * @param[in] antecedents indices of the plan items that must complete before
     * the gather steps may be initiated.
     * @return An `std::vector<size_t>` of indices of the created plan items. When these
     * items complete, the execution plan's input buffer is set for computing the desired
     * transform.
     */
    std::vector<size_t>
        CreateInputGatheringItemsIfNeeded(const NodeMetaData&        exec_plan_metadata,
                                          const rocfft_location_t&   exec_plan_location,
                                          const std::vector<size_t>& antecedents = {});

    /**
     * @brief Creates the plan items required to scatter the output buffer of the (single-device)
     * execution plan (observing the output data layout set for that execution plan) into the
     * output data buffer(s) of a multi-device transform. If the output data is undistributed
     * and can be written directly by the (single-device) execution plan, no output-scattering
     * plan items are created.
     * 
     * @param[in] exec_plan_metadata single-device execution plan's metadata.
     * @param[in] exec_plan_location location of the single-device execution plan.
     * @param[in] antecedents indices of the plan items that must complete before
     * the scatter steps may be initiated. 
     * @return An `std::vector<size_t>` of indices of the created plan items. When these items
     * complete, the user's output buffers are set with the corresponding portions of the
     * transform's results.
     */
    std::vector<size_t>
        CreateOutputScatteringItemsIfNeeded(const NodeMetaData&        exec_plan_metadata,
                                            const rocfft_location_t&   exec_plan_location,
                                            const std::vector<size_t>& antecedents);

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
    void C2CField(const rocfft_field_t&          field,
                  const std::vector<size_t>&     fftDims,
                  std::vector<BufferPtr>&        input,
                  std::vector<BufferPtr>&        output,
                  const std::optional<LoadOps>&  loadOps,
                  const std::optional<StoreOps>& storeOps,
                  const std::vector<size_t>&     inputAntecedents,
                  std::vector<size_t>&           outputItems);

    // RAII struct to 'lease' a temp buffer from a multimap of per-device
    // buffers.  When this struct is destroyed, the buffer is returned to
    // the map for reuse.
    struct TempBufferLease
    {
        TempBufferLease(
            std::multimap<rocfft_location_t, std::shared_ptr<InternalTempBuffer>>& _tempBuffers,
            int                                                                    local_comm_rank,
            rocfft_location_t                                                      _location,
            size_t                                                                 byte_size)
            : location(_location)
            , tempBuffers(&_tempBuffers)
        {
            // no need to allocate anything for non-local ranks
            if(local_comm_rank != location.comm_rank)
            {
                // instead allocate a placeholder that remembers which
                // rank this was for, to aid debugging
                buf = std::make_shared<InternalTempBuffer>(location.comm_rank);
                return;
            }

            // return an existing buffer that's big enough, if one exists
            auto i = tempBuffers->lower_bound(location);
            if(i != tempBuffers->upper_bound(location))
            {
                // found a buffer, ensure it's big enough
                i->second->set_size_bytes(byte_size);

                // leasing out this temp buffer, remove it from the map
                buf = i->second;
                tempBuffers->erase(i);
                return;
            }
            // no buffer was found, allocate a new one
            buf = std::make_shared<InternalTempBuffer>(local_comm_rank);
            buf->set_size_bytes(byte_size);
        }
        ~TempBufferLease()
        {
            // return the buffer to the map
            if(buf)
                tempBuffers->emplace(std::make_pair(location, std::move(buf)));
        }
        // allow moves, disallow copies
        TempBufferLease(TempBufferLease&& other)
            : location(other.location)
            , tempBuffers(other.tempBuffers)
            , buf(std::move(other.buf))
        {
        }
        TempBufferLease& operator=(TempBufferLease&& other)
        {
            location    = other.location;
            tempBuffers = other.tempBuffers;
            buf         = std::move(other.buf);
            return *this;
        }
        TempBufferLease(const TempBufferLease& other) = delete;
        TempBufferLease& operator=(const TempBufferLease& other) = delete;

        std::shared_ptr<InternalTempBuffer> data()
        {
            return buf;
        }

    private:
        rocfft_location_t                                                      location;
        std::multimap<rocfft_location_t, std::shared_ptr<InternalTempBuffer>>* tempBuffers;
        std::shared_ptr<InternalTempBuffer>                                    buf;
    };

    /**
     * @brief Creates and returns the metadata configuring the single-device execution
     * plan item used for tackling the plan's task. If the plan's task cannot be handled
     * by a single-device execution plan on the given location, the returned object
     * parameterizes an execution plan configured to
     * 
     * - read (resp. write) directly from (resp. into) the user's input (resp. output)
     *   data buffer, if that data is undistributed and expected on the given location at
     *   execution. If not, default packed layouts for in-place operations are set on
     *   input and output (possibly requiring temporary, leased I/O);
     * 
     * - operate in-place, unless the above made that impossible or if the plan's
     *   configuration does not guarantee that it is safe to do so.
     * 
     * If the returned configuration is such that reading (resp. writing) directly from
     * (resp. to) a user-provided input (resp. output) data buffer cannot or should not
     * be done, leased temporary buffers of appropriate sizes are created instead and
     * appended to the given `leased_io`.
     * 
     * @param[inout] leased_io vector of leased temporary buffers. If a temporary buffer
     * needs to be leased for the input and/or output of the single-device execution plan
     * that the returned object configures, that leased buffer is created and appended to
     * this vector.
     * @param[in] exec_plan_location location where the single-device execution plan
     * configured by the returned object is meant to reside.
     */
    NodeMetaData get_single_dev_exec_plan_metadata(std::vector<TempBufferLease>& leased_io,
                                                   const rocfft_location_t& exec_plan_location);

    struct field_view_t
    {
        field_view_t(const rocfft_field_t&         field_,
                     const std::vector<BufferPtr>& buffers_,
                     rocfft_array_type             array_type_,
                     rocfft_precision              precision_,
                     const std::string&            group_name_ = "");

        field_view_t get_view_for_lengths(const std::set<size_t>& len_dims) const;
        field_view_t get_embedding_view() const;

        inline bool operator==(const field_view_t& other) const
        {
            // group_name is irrelevant for logical comparisons of field views
            // Do not use a strict equality of `array_type` as that would prevent simple
            // re-interpretation of `rocfft_array_type_hermitian_interleaved` into
            // `rocfft_array_type_complex_interleaved` and vice-versa.
            return field == other.field && precision == other.precision
                   && array_type_is_complex(array_type) == array_type_is_complex(other.array_type)
                   && buffers == other.buffers;
        }
        inline bool operator!=(const field_view_t& other) const
        {
            return !(*this == other);
        }

        const rocfft_field_t         field;
        const std::vector<BufferPtr> buffers;
        const rocfft_array_type      array_type;
        const rocfft_precision       precision;
        const std::string            group_name;
    };

    template <io_data_label io>
    field_view_t make_user_field_view() const;

    struct embarrassingly_parallel_fft
    {
        embarrassingly_parallel_fft(rocfft_transform_type   fft_type_,
                                    rocfft_result_placement placement_,
                                    rocfft_precision        precision_,
                                    const field_view_t&     input_view_,
                                    const field_view_t&     output_view_);
        inline void set_load_ops(const LoadOps& load_operations)
        {
            load_ops = load_operations;
        }
        inline void set_store_ops(const StoreOps& store_operations)
        {
            store_ops = store_operations;
        }

        inline const std::optional<LoadOps>& get_load_ops() const
        {
            return load_ops;
        }
        inline const std::optional<StoreOps>& get_store_ops() const
        {
            return store_ops;
        }

        const rocfft_transform_type   fft_type;
        const rocfft_result_placement placement;
        const rocfft_precision        precision;
        const field_view_t            input, output;

        std::string get_description() const;
        std::string get_group() const;

    private:
        inline std::vector<size_t> get_axes_in_embedding() const
        {
            // validation checks at object's construction guarantee
            // it's identical for all I/O bricks
            return input.field.bricks[0].layout.corresponding_axes_in_embedding();
        }
        std::optional<LoadOps>  load_ops  = std::nullopt;
        std::optional<StoreOps> store_ops = std::nullopt;
    };

    template <io_data_label field_view_label>
    embarrassingly_parallel_fft make_embarrassingly_parallel_fft_from_user_field(
        const std::set<size_t>        fft_len_dims,
        std::vector<TempBufferLease>& leased_buffers,
        bool                          prefer_in_place_if_possible);

    std::vector<size_t> create_plan_items_for(const embarrassingly_parallel_fft& parallel_fft,
                                              const std::vector<size_t>&         antecedents);

    // Transpose the input field to the output field by adding work items
    // to the plan.  Antecedents are provided as a vector of item
    // indexes, one per brick.  Final work item per brick (that future
    // per-brick operations can depend on) is returned in outputItems.
    std::vector<size_t> GlobalTranspose(const field_view_t&        input,
                                        const field_view_t&        output,
                                        const std::vector<size_t>& antecedents);

    // default global all-to-all transpose
    std::vector<size_t> GlobalTransposeA2A(const field_view_t&        input,
                                           const field_view_t&        output,
                                           const std::vector<size_t>& antecedents);

    // fallback case for global transpose that uses point-to-point
    // communications, for when all-to-all isn't possible.
    std::vector<size_t> GlobalTransposeP2P(const field_view_t&        input,
                                           const field_view_t&        output,
                                           const std::vector<size_t>& antecedents);

    // Pre-existing signature, to be used only for complex fields only! (Note: this will be removed eventually)
    void GlobalTranspose(size_t                     elem_size,
                         const rocfft_field_t&      inField,
                         const rocfft_field_t&      outField,
                         std::vector<BufferPtr>&    input,
                         std::vector<BufferPtr>&    output,
                         const std::vector<size_t>& inputAntecedents,
                         std::vector<size_t>&       outputItems,
                         size_t                     transposeNumber);
};

bool PlanPowX(ExecPlan& execPlan);
bool GetTuningKernelInfo(ExecPlan& execPlan);
void RuntimeCompilePlan(ExecPlan& execPlan);

#endif // PLAN_H
