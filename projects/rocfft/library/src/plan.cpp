// Copyright (C) 2016 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "plan.h"
#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "../../shared/device_properties.h"
#include "../../shared/environment.h"
#include "../../shared/precision_type.h"
#include "../../shared/ptrdiff.h"
#include "../../shared/rocfft_params.h"
#include "assignment_policy.h"
#include "enum_printer.h"
#include "function_pool.h"
#include "hip/hip_runtime_api.h"
#include "logging.h"
#include "node_factory.h"
#include "rocfft/rocfft-version.h"
#include "rocfft/rocfft.h"
#include "rocfft_current_function.h"
#include "rocfft_exception.h"
#include "rocfft_mpi.h"
#include "rocfft_ostream.hpp"
#include "rtc_kernel.h"
#include "solution_map.h"
#include "tree_node.h"
#include "tuning_helper.h"
#include "tuning_plan_tuner.h"

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef ROCFFT_MPI_ENABLE
#include <type_traits>
#include <typeinfo>
#endif

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

// clang-format off
const char* ROCFFT_VERSION_STRING = (TO_STR(rocfft_version_major) "." \
                                     TO_STR(rocfft_version_minor) "." \
                                     TO_STR(rocfft_version_patch) "." \
                                     TO_STR(rocfft_version_tweak) );
// clang-format on

rocfft_status rocfft_plan_description_set_scale_factor(rocfft_plan_description description,
                                                       const double            scale_factor)
try
{
    log_trace(__func__, "description", description, "scale", scale_factor);
    if(!std::isfinite(scale_factor))
        return rocfft_status_invalid_arg_value;
    description->storeOps.scale_factor = scale_factor;
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

static size_t offset_count(rocfft_array_type type)
{
    // planar data has 2 sets of offsets, otherwise we have one
    return type == rocfft_array_type_complex_planar || type == rocfft_array_type_hermitian_planar
               ? 2
               : 1;
}

namespace
{
    // functor to search for bricks on a comm rank, in a container of bricks
    // sorted by comm rank
    struct match_comm_rank
    {
        bool operator()(const rocfft_brick_t& b, int comm_rank) const
        {
            return b.location.comm_rank < comm_rank;
        }
        bool operator()(int comm_rank, const rocfft_brick_t& b) const
        {
            return comm_rank < b.location.comm_rank;
        }
    };

}

std::optional<rocfft_location_t>
    rocfft_plan_description_t::expected_undistributed_location_for(io_data_label io) const
{
    if(io != io_data_label::INPUT && io != io_data_label::OUTPUT)
        throw std::invalid_argument("Invalid argument value given to " + ROCFFT_CURRENT_FUNCTION);
    const auto& io_fields = io == io_data_label::INPUT ? inFields : outFields;
    if(io_fields.empty())
        return get_current_location();
    // Undistributed I/O data iff only one brick per field.
    std::optional<rocfft_location_t> ret;
    if(io_fields.size() == 1 && io_fields[0].bricks.size() == 1)
        ret = std::make_optional<rocfft_location_t>(io_fields[0].bricks[0].location);
    return ret;
}

bool rocfft_plan_description_t::has_undistributed_io_on_current_location() const
{
    const auto input_loc = expected_undistributed_location_for(io_data_label::INPUT);
    if(!input_loc)
        return false;
    const auto output_loc = expected_undistributed_location_for(io_data_label::OUTPUT);
    if(!output_loc || *output_loc != *input_loc)
        return false;
    return *input_loc == get_current_location();
}

const data_layout_t& rocfft_plan_description_t::undistributed_layout_for(io_data_label io) const
{
    if(io != io_data_label::INPUT && io != io_data_label::OUTPUT)
        throw std::invalid_argument("Invalid argument value given to " + ROCFFT_CURRENT_FUNCTION);

    const auto& io_field = io == io_data_label::INPUT ? inFields : outFields;
    if(io_field.size() > 1 || (io_field.size() == 1 && io_field[0].bricks.size() > 1))
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " cannot be used with multi-field or multi-brick descriptions.");

    if(io_field.size() == 1 && io_field[0].bricks.size() == 1)
        return io_field[0].bricks[0].layout;

    return io == io_data_label::INPUT ? input_layout : output_layout;
}

bool rocfft_plan_description_t::multiple_devices_in_rank(const rocfft_field_t& field)
{
    // map ranks to a set of distinct devices
    std::map<int, std::set<int>> rank_devices;

    // collect information on bricks in the field
    for(const auto& brick : field.bricks)
    {
        auto& devices = rank_devices[brick.location.comm_rank];
        devices.insert(brick.location.device);
    }

    // any rank needs to have multiple devices to return true
    for(const auto& rank_device : rank_devices)
    {
        if(rank_device.second.size() > 1)
            return true;
    }
    return false;
}

size_t rocfft_plan_t::AddMultiPlanItem(std::unique_ptr<MultiPlanItem>&& item,
                                       const std::vector<size_t>&       antecedents)
{
    // ensure antecedents all exist
    if(std::any_of(antecedents.begin(), antecedents.end(), [=, this](size_t i) {
           return i >= multiPlan.size();
       }))
        throw std::runtime_error("antecedent does not exist");

    item->local_comm_rank = desc.get_local_comm_rank();

    multiPlan.emplace_back(std::move(item));
    multiPlanAntecedents.emplace_back(antecedents);

    // return index of new item
    return multiPlan.size() - 1;
}

void rocfft_plan_t::AddAntecedent(size_t itemIdx, size_t antecedentIdx)
{
    // We're not implementing full dependency cycle checks but at least
    // we can check for obvious errors.
    if(itemIdx >= multiPlan.size() || antecedentIdx >= multiPlan.size() || itemIdx == antecedentIdx)
        throw std::runtime_error("invalid antecedent during plan creation");

    auto& antecedents = multiPlanAntecedents[itemIdx];
    if(std::find(antecedents.begin(), antecedents.end(), antecedentIdx) == antecedents.end())
        antecedents.push_back(antecedentIdx);
}

std::vector<size_t> rocfft_plan_t::WorkBufBytesPerDevice() const
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess)
        throw std::runtime_error("failed to get device count");
    std::vector<size_t> workBufBytes(deviceCount);

    const auto base_type_size = real_type_size(precision);

    for(const auto& i : multiPlan)
    {
        if(!i)
            continue;
        if(i->ExecutesOnLocalRank())
            i->WorkBufBytesPerDevice(base_type_size, workBufBytes);
    }
    return workBufBytes;
}

rocfft_status rocfft_plan_description_set_comm(rocfft_plan_description description,
                                               rocfft_comm_type        comm_type,
                                               void*                   comm_handle)
try
{
    log_trace(
        __func__, "description", description, "comm_type", comm_type, "comm_handle", comm_handle);

    // If you say that you're going to give us a communicator, please don't give us a null
    // pointer.
    if(comm_type != rocfft_comm_none && comm_handle == nullptr)
    {
        return rocfft_status_invalid_arg_value;
    }

    description->comm_type = comm_type;

    switch(description->comm_type)
    {
    case rocfft_comm_none:
    {
#ifdef ROCFFT_MPI_ENABLE
        description->mpi_comm.free();
#endif
        break;
    }
#ifdef ROCFFT_MPI_ENABLE
    case rocfft_comm_mpi:
    {
        // duplicate the user-provided MPI communicator and set
        // our own error policy on it
        description->mpi_comm.duplicate(*static_cast<MPI_Comm*>(comm_handle));
        auto mpiret = MPI_Comm_set_errhandler(description->mpi_comm, MPI_ERRORS_RETURN);
        if(mpiret != MPI_SUCCESS)
            return rocfft_status_failure;
        break;
    }
#endif
    default:
        return rocfft_status_failure;
    }
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description description,
                                                      const rocfft_array_type in_array_type,
                                                      const rocfft_array_type out_array_type,
                                                      const size_t*           in_offsets,
                                                      const size_t*           out_offsets,
                                                      const size_t            in_strides_size,
                                                      const size_t*           in_strides,
                                                      const size_t            in_distance,
                                                      const size_t            out_strides_size,
                                                      const size_t*           out_strides,
                                                      const size_t            out_distance)
try
{
    log_trace(__func__,
              "description",
              description,
              "in_array_type",
              in_array_type,
              "out_array_type",
              out_array_type,
              "in_offsets",
              std::make_pair(in_offsets, offset_count(in_array_type)),
              "out_offsets",
              std::make_pair(out_offsets, offset_count(out_array_type)),
              "in_strides",
              std::make_pair(in_strides, in_strides_size),
              "in_distance",
              in_distance,
              "out_strides",
              std::make_pair(out_strides, out_strides_size),
              "out_distance",
              out_distance);

    for(auto io : {io_data_label::INPUT, io_data_label::OUTPUT})
    {
        // object's I/O members to (re)set
        auto& desc_io_array_type
            = io == io_data_label::INPUT ? description->inArrayType : description->outArrayType;
        auto& desc_io_offsets
            = io == io_data_label::INPUT ? description->inOffset : description->outOffset;
        auto& desc_io_layout
            = io == io_data_label::INPUT ? description->input_layout : description->output_layout;
        // user's setting
        const auto& user_io_array_type
            = io == io_data_label::INPUT ? in_array_type : out_array_type;
        const auto& user_io_offsets = io == io_data_label::INPUT ? in_offsets : out_offsets;
        const auto& user_io_strides_sz
            = io == io_data_label::INPUT ? in_strides_size : out_strides_size;
        const auto& user_io_strides = io == io_data_label::INPUT ? in_strides : out_strides;
        const auto& user_io_dist    = io == io_data_label::INPUT ? in_distance : out_distance;
        // -------------------------------
        // Set/Reset description's members
        // -------------------------------
        // array type
        desc_io_array_type = user_io_array_type;
        // offsets
        desc_io_offsets[0] = desc_io_offsets[1] = 0; // default if unspecified
        if(user_io_offsets != nullptr)
        {
            desc_io_offsets[0] = user_io_offsets[0];
            if(array_type_is_planar(desc_io_array_type))
                desc_io_offsets[1] = user_io_offsets[1];
        }
        // layout parameters: implicit default strides (resp. distances) to be determined
        // at plan creation if strides are unspecified here (resp. distances are 0)
        desc_io_layout.clear();
        if(user_io_strides)
        {
            desc_io_layout.len_axes.resize(user_io_strides_sz);
            for(size_t dim = 0; dim < user_io_strides_sz; dim++)
                desc_io_layout.len_axes[dim].inbuffer_stride = user_io_strides[dim];
        }
        if(user_io_dist != 0)
        {
            desc_io_layout.batch_axes.resize(1);
            desc_io_layout.batch_axes[0].inbuffer_stride = user_io_dist;
        }
    }
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_plan_description_create(rocfft_plan_description* description)
try
{
    rocfft_plan_description desc = new rocfft_plan_description_t;
    *description                 = desc;
    log_trace(__func__, "description", *description);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description)
try
{
    log_trace(__func__, "description", description);
    if(description != nullptr)
    {
        delete description;
    }
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_field_create(rocfft_field* field)
try
{
    *field = new rocfft_field_t;
    log_trace(__func__, "field", *field);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_field_destroy(rocfft_field field)
try
{
    log_trace(__func__, "field", field);
    delete field;
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

std::string rocfft_brick_t::str() const
{
    return layout.str() + " " + location.str();
}

// internal brick-adding API
static void rocfft_field_add_brick_internal(rocfft_field_t& field, const rocfft_brick_t& brick)
{
    if(!field.bricks.empty()
       && !brick.layout.is_dimensionally_consistent_with(field.bricks[0].layout))
    {
        // could motivate a new "rocfft_status_invalid_brick{_dimensions}" error code
        throw std::runtime_error(
            "Brick to be added is not dimensionally consistent with other brick(s) in field.");
    }

    field.bricks.emplace_back(brick);
}

// public brick-adding API that logs
rocfft_status rocfft_field_add_brick(rocfft_field field, rocfft_brick brick)
try
{
    log_trace(__func__, "field", field, "brick", brick);
    if(!field || !brick)
        return rocfft_status_invalid_arg_value;
    rocfft_field_add_brick_internal(*field, *brick);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

// public brick-creating API that logs calls
rocfft_status rocfft_brick_create(rocfft_brick* brick,
                                  const size_t* field_lower,
                                  const size_t* field_upper,
                                  const size_t* brick_stride,
                                  size_t        dim,
                                  int           deviceID)
try
{
    log_trace(__func__,
              "brick",
              brick,
              "field_lower",
              std::make_pair(field_lower, dim),
              "field_upper",
              std::make_pair(field_upper, dim),
              "brick_stride",
              std::make_pair(brick_stride, dim),
              "dim",
              dim,
              "deviceID",
              deviceID);
    if(!brick)
        return rocfft_status_invalid_arg_value;

    if(dim < 2 || dim > 4) // {1,2,3} length dimensions + 1 batch dimension
        return rocfft_status_invalid_dimensions;

    if(!field_lower || !field_upper || !brick_stride)
        return rocfft_status_invalid_arg_value;

    // Assume comm rank 0, as we don't have a communicator at
    // this point.  If this is actually an MPI transform, these
    // bricks will be recreated with correct rank when we gather all
    // of the brick data at plan creation time.
    auto brick_ptr
        = std::make_unique<rocfft_brick_t>(std::vector<size_t>{field_lower, field_lower + dim},
                                           std::vector<size_t>{field_upper, field_upper + dim},
                                           std::vector<size_t>{brick_stride, brick_stride + dim},
                                           rocfft_location_t{0, deviceID});
    *brick = brick_ptr.release();
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_brick_destroy(rocfft_brick brick)
try
{
    log_trace(__func__, "brick", brick);
    delete brick;
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_plan_description_add_infield(rocfft_plan_description description,
                                                  rocfft_field            field)
try
{
    log_trace(__func__, "description", description, "field", field);
    if(!description || !field)
        return rocfft_status_invalid_arg_value;
    description->inFields.push_back(*field);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_plan_description_add_outfield(rocfft_plan_description description,
                                                   rocfft_field            field)
try
{
    log_trace(__func__, "description", description, "field", field);
    if(!description || !field)
        return rocfft_status_invalid_arg_value;
    description->outFields.push_back(*field);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

std::vector<size_t> rocfft_plan_t::get_user_facing_lengths() const
{
    if(transformType == rocfft_transform_type_complex_forward
       || transformType == rocfft_transform_type_real_forward)
    {
        return desc.input_layout.lengths();
    }
    return desc.output_layout.lengths();
}

std::string rocfft_bench_command(const rocfft_plan& plan)
{
    rocfft_params params;

    params.nbatch         = plan->desc.batch();
    params.placement      = fft_result_placement_from_rocfft_result_placement(plan->placement);
    params.transform_type = fft_transform_type_from_rocfft_transform_type(plan->transformType);
    params.precision      = fft_precision_from_rocfft_precision(plan->precision);
    params.itype          = fft_array_type_from_rocfft_array_type(plan->desc.inArrayType);
    params.otype          = fft_array_type_from_rocfft_array_type(plan->desc.outArrayType);
    params.idist          = plan->desc.input_layout.distance();
    params.odist          = plan->desc.output_layout.distance();
    params.scale_factor   = plan->desc.storeOps.scale_factor;

    // reverse-copy lengths and strides (col-major to row-major)
    const auto lengths  = plan->get_user_facing_lengths();
    const auto istrides = plan->desc.input_layout.strides();
    const auto ostrides = plan->desc.output_layout.strides();
    params.length.assign(lengths.rbegin(), lengths.rend());
    params.istride.assign(istrides.rbegin(), istrides.rend());
    params.ostride.assign(ostrides.rbegin(), ostrides.rend());
    // copy offsets
    params.ioffset.assign(plan->desc.inOffset.begin(), plan->desc.inOffset.end());
    params.ooffset.assign(plan->desc.outOffset.begin(), plan->desc.outOffset.end());
    // fields:
    auto copy_fields_to_params
        = [&](std::vector<fft_params::fft_field>& dest, const std::vector<rocfft_field_t>& src) {
              dest.resize(src.size());
              for(size_t fidx = 0; fidx < src.size(); ++fidx)
              {
                  dest[fidx].bricks.resize(src[fidx].bricks.size());
                  for(size_t bidx = 0; bidx < src[fidx].bricks.size(); ++bidx)
                  {
                      fft_params::fft_brick& dest_brick = dest[fidx].bricks[bidx];
                      const rocfft_brick_t&  src_brick  = src[fidx].bricks[bidx];
                      // reverse-copy bricks' coordinates and strides (col-major to row-major)
                      const auto& src_layout = src_brick.layout;
                      const auto  lower_cm   = src_layout.lower();
                      const auto  upper_cm   = src_layout.upper();
                      const auto  strides_cm = src_layout.strides_and_distances();
                      dest_brick.lower.assign(lower_cm.rbegin(), lower_cm.rend());
                      dest_brick.upper.assign(upper_cm.rbegin(), upper_cm.rend());
                      dest_brick.stride.assign(strides_cm.rbegin(), strides_cm.rend());
                      dest_brick.rank   = src_brick.location.comm_rank;
                      dest_brick.device = src_brick.location.device;
                  }
              }
          };
    copy_fields_to_params(params.ifields, plan->desc.inFields);
    copy_fields_to_params(params.ofields, plan->desc.outFields);

    std::stringstream bench;
    bench << "rocfft-bench --token ";
    bench << params.token();

    return bench.str();
}

rocfft_location_t rocfft_plan_description_t::get_current_location() const
{
    rocfft_location_t current_loc;
    current_loc.comm_rank = get_local_comm_rank();
    current_loc.device    = hipInvalidDeviceId;
    if(hipGetDevice(&current_loc.device) != hipSuccess || current_loc.device == hipInvalidDeviceId)
        throw std::runtime_error("hipGetDevice failed");
    return current_loc;
}

static void set_bluestein_strides(const rocfft_plan_t* plan, NodeMetaData& planData)
{
    std::array<size_t, 3> inStridesBlue  = {0, 0, 0};
    std::array<size_t, 3> outStridesBlue = {0, 0, 0};
    std::array<size_t, 3> lengthsBlue    = {0, 0, 0};
    size_t                inDistBlue     = 0;
    size_t                outDistBlue    = 0;

    function_pool pool{planData.deviceProp};

    const auto precision     = plan->precision;
    const auto transformType = plan->transformType;
    const auto rank          = plan->desc.rank();
    const auto fftLength     = plan->get_user_facing_lengths();
    const auto placement     = plan->placement;
    const auto dimension     = planData.dimension;

    assert(rank == dimension);

    lengthsBlue[0] = NodeFactory::SupportedLength(pool, precision, fftLength[0])
                         ? fftLength[0]
                         : NodeFactory::GetBluesteinLength(pool, precision, fftLength[0]);
    for(size_t i = 1; i < dimension; i++)
        lengthsBlue[i] = fftLength[i];

    // =================================
    // inStrides
    // =================================
    inStridesBlue[0] = 1;

    if((transformType == rocfft_transform_type_real_forward)
       && (placement == rocfft_placement_inplace))
    {
        // real-to-complex in-place
        size_t dist = 2 * (1 + (lengthsBlue[0]) / 2);

        for(size_t i = 1; i < rank; i++)
        {
            inStridesBlue[i] = dist;
            dist *= lengthsBlue[i];
        }

        inDistBlue = dist;
    }
    else if(transformType == rocfft_transform_type_real_inverse)
    {
        // complex-to-real
        size_t dist = 1 + (lengthsBlue[0]) / 2;

        for(size_t i = 1; i < rank; i++)
        {
            inStridesBlue[i] = dist;
            dist *= lengthsBlue[i];
        }

        inDistBlue = dist;
    }
    else
    {
        // Set the inStrides to deal with contiguous data
        for(size_t i = 1; i < rank; i++)
            inStridesBlue[i] = lengthsBlue[i - 1] * inStridesBlue[i - 1];

        inDistBlue = lengthsBlue[rank - 1] * inStridesBlue[rank - 1];
    }

    // =================================
    // outStrides
    // =================================
    outStridesBlue[0] = 1;

    if((transformType == rocfft_transform_type_real_forward)
       && (placement == rocfft_placement_inplace))
    {
        // real-to-complex in-place
        size_t dist = 2 * (1 + (lengthsBlue[0]) / 2);

        for(size_t i = 1; i < rank; i++)
        {
            outStridesBlue[i] = dist;
            dist *= lengthsBlue[i];
        }

        outDistBlue = dist;
    }
    else if(transformType == rocfft_transform_type_real_inverse)
    {
        // complex-to-real
        size_t dist = 1 + (lengthsBlue[0]) / 2;

        for(size_t i = 1; i < rank; i++)
        {
            outStridesBlue[i] = dist;
            dist *= lengthsBlue[i];
        }

        outDistBlue = dist;
    }
    else
    {
        // Set the inStrides to deal with contiguous data
        for(size_t i = 1; i < rank; i++)
            outStridesBlue[i] = lengthsBlue[i - 1] * outStridesBlue[i - 1];

        outDistBlue = lengthsBlue[rank - 1] * outStridesBlue[rank - 1];
    }

    for(size_t i = 0; i < dimension; i++)
    {
        planData.inStrideBlue.push_back(inStridesBlue[i]);
        planData.outStrideBlue.push_back(outStridesBlue[i]);
    }
    planData.iDistBlue = inDistBlue;
    planData.oDistBlue = outDistBlue;
}

NodeMetaData
    rocfft_plan_t::get_single_dev_exec_plan_metadata(std::vector<TempBufferLease>& leased_io,
                                                     const rocfft_location_t& exec_plan_location)
{
    NodeMetaData root_plan(nullptr);
    root_plan.dimension         = desc.rank();
    root_plan.batch             = desc.batch();
    root_plan.precision         = precision;
    root_plan.direction         = ((transformType == rocfft_transform_type_complex_forward)
                           || (transformType == rocfft_transform_type_real_forward))
                                      ? -1
                                      : 1;
    root_plan.inArrayType       = desc.inArrayType;
    root_plan.outArrayType      = desc.outArrayType;
    root_plan.rootTransformType = transformType;
    // root plan's data layouts and placement may be different than the calling plan's
    std::optional<data_layout_t> root_plan_input_layout, root_plan_output_layout;
    if(desc.has_undistributed_io_on_current_location()
       && desc.get_current_location() == exec_plan_location)
    {
        // "true" single-device usage (no I/O field used or glorified version using
        // lone bricks for some reason): use the calling plan's own parameters
        root_plan_input_layout  = desc.undistributed_layout_for(io_data_label::INPUT);
        root_plan_output_layout = desc.undistributed_layout_for(io_data_label::OUTPUT);
        root_plan.placement     = placement;
        root_plan.input_buffer  = BufferPtr::user_input(0, desc.get_local_comm_rank());
        if(root_plan.placement == rocfft_placement_inplace)
            root_plan.output_buffer = root_plan.input_buffer;
        else
            root_plan.output_buffer = BufferPtr::user_output(0, desc.get_local_comm_rank());
    }
    else
    {
        // Single-device plan can always operate out-of-place, but that may trigger
        // undesirable additional costs such as extra temporary buffer required for
        // output results (and/or unfriendly gather/scatter extra steps in case of
        // real transforms). If possible (and allowed) or likely preferable,
        // in-place operations are used instead.
        for(auto io : {io_data_label::INPUT, io_data_label::OUTPUT})
        {
            auto& root_plan_io_buffer
                = io == io_data_label::INPUT ? root_plan.input_buffer : root_plan.output_buffer;
            auto& root_plan_io_layout
                = io == io_data_label::INPUT ? root_plan_input_layout : root_plan_output_layout;
            if(root_plan_io_layout)
                continue; // layout already set
            const auto io_loc = desc.expected_undistributed_location_for(io);
            if(io_loc && *io_loc == exec_plan_location)
            {
                // The execution plan can access the user's undistributed data directly
                root_plan_io_buffer = io == io_data_label::INPUT
                                          ? BufferPtr::user_input(0, desc.get_local_comm_rank())
                                          : BufferPtr::user_output(0, desc.get_local_comm_rank());
                root_plan_io_layout = desc.undistributed_layout_for(io);
                // other io layout to be set for the execution plan:
                auto&      root_plan_other_io_layout = other(io) == io_data_label::INPUT
                                                           ? root_plan_input_layout
                                                           : root_plan_output_layout;
                auto&      root_plan_other_io_buffer = other(io) == io_data_label::INPUT
                                                           ? root_plan.input_buffer
                                                           : root_plan.output_buffer;
                const auto other_lengths             = other(io) == io_data_label::INPUT
                                                           ? desc.input_layout.lengths()
                                                           : desc.output_layout.lengths();
                auto       tmp = root_plan_io_layout->get_other_inplace_layout_for(
                    other(io), transformType, other_lengths[0] % 2 == 1);
                // If tmp has a value set, in-place operations can be done using that layout
                // - if the plan was configured in-place by the user
                // - or if the execution plan can operate entirely in the user's (undistributed)
                //   output data buffer without risking overwriting anything it shouldn't touch
                //   (i.e., if contiguous data is expected therein), *and* without risking writing
                //   past the buffer's bounds.
                if(tmp
                   && (placement == rocfft_placement_inplace
                       || (io == io_data_label::OUTPUT && root_plan_io_layout->is_contiguous()
                           && transformType != rocfft_transform_type_real_inverse)))
                {
                    // execution plan may and can operate in-place.
                    root_plan_other_io_layout = std::move(*tmp);
                    root_plan.placement       = rocfft_placement_inplace;
                    root_plan_other_io_buffer = root_plan_io_buffer;
                }
                else
                {
                    // in-place is not allowed or it can't be done or it would require a larger buffer
                    // than what's safe to expect from user. Pack results contiguously in a temporary
                    // buffer and work out of place.
                    root_plan_other_io_layout
                        = data_layout_t::default_full_layout(other_lengths, desc.batch());
                    root_plan.placement = rocfft_placement_notinplace;
                    const auto tmp_size
                        = root_plan_other_io_layout->buffer_element_count()
                          * element_size(precision,
                                         other(io) == io_data_label::INPUT ? desc.inArrayType
                                                                           : desc.outArrayType);
                    leased_io.emplace_back(tempBuffers,
                                           desc.get_local_comm_rank(),
                                           desc.get_current_location(),
                                           tmp_size);
                    root_plan_other_io_buffer = BufferPtr::temp(leased_io.back().data());
                }
            }
        }
    }

    if(!root_plan_input_layout && !root_plan_output_layout)
    {
        // Input and output data are distributed or none of them are directly accessible
        // from the current location. Temporary buffers on current device are used for input
        // and output of the execution plan. In-place default layout operations are
        // used to minimize the library's memory footprint.
        root_plan.placement    = rocfft_placement_inplace;
        root_plan_input_layout = data_layout_t::default_full_layout(
            desc.input_layout.lengths(),
            desc.input_layout.batch(),
            transformType == rocfft_transform_type_real_forward);
        root_plan_output_layout = data_layout_t::default_full_layout(
            desc.output_layout.lengths(),
            desc.output_layout.batch(),
            transformType == rocfft_transform_type_real_inverse);

        const auto tmp_size = std::max(root_plan_input_layout->buffer_element_count()
                                           * element_size(precision, desc.inArrayType),
                                       root_plan_output_layout->buffer_element_count()
                                           * element_size(precision, desc.outArrayType));
        leased_io.emplace_back(
            tempBuffers, desc.get_local_comm_rank(), desc.get_current_location(), tmp_size);
        root_plan.input_buffer = root_plan.output_buffer = BufferPtr::temp(leased_io.back().data());
    }
    // I/O should both be fully defined at this point
    for(auto io : {io_data_label::INPUT, io_data_label::OUTPUT})
    {
        const auto& io_is_defined = io == io_data_label::INPUT
                                        ? (root_plan_input_layout && root_plan.input_buffer)
                                        : (root_plan_output_layout && root_plan.output_buffer);
        if(!io_is_defined)
            throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " failed to define the " + to_str(io)
                                   + " layout and/or the corresponding buffer for the "
                                     "single-device execution plan item");
    }

    root_plan.length       = root_plan_input_layout->lengths();
    root_plan.inStride     = root_plan_input_layout->strides();
    root_plan.iDist        = root_plan_input_layout->distance();
    root_plan.outputLength = root_plan_output_layout->lengths();
    root_plan.outStride    = root_plan_output_layout->strides();
    root_plan.oDist        = root_plan_output_layout->distance();

    root_plan.deviceProp = get_curr_device_prop();
    set_bluestein_strides(this, root_plan);

    return root_plan;
}

// return an ExecPlan that transposes a brick
std::unique_ptr<ExecPlan> transpose_brick(int                        local_comm_rank,
                                          rocfft_location_t          location,
                                          const std::vector<size_t>& length,
                                          rocfft_precision           precision,
                                          rocfft_array_type          arrayType,
                                          BufferPtr                  inputPtr,
                                          size_t                     offsetIn,
                                          const std::vector<size_t>& strideIn,
                                          BufferPtr                  outputPtr,
                                          size_t                     offsetOut,
                                          const std::vector<size_t>& strideOut,
                                          std::string&&              description)
{
    auto      execPlanMultiItem = std::make_unique<ExecPlan>(local_comm_rank, true, location);
    ExecPlan& execPlan          = *execPlanMultiItem;
    execPlan.deviceProp         = get_curr_device_prop();

    // add input buffers provided by users
    execPlan.inputPtr  = inputPtr;
    execPlan.outputPtr = outputPtr;

    // transpose 2D
    switch(length.size())
    {
    case 2:
    {
        execPlan.rootPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, nullptr);

        execPlan.rootPlan->length    = length;
        execPlan.rootPlan->dimension = 2;

        execPlan.rootPlan->outStride = strideOut;
        break;
    }
    case 3:
    {
        execPlan.rootPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, nullptr);

        execPlan.rootPlan->length    = length;
        execPlan.rootPlan->dimension = 3;

        execPlan.rootPlan->outStride = strideOut;
        break;
    }
        // 4D is required if we have a 3D problem + batch
    case 4:
    {
        execPlan.rootPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, nullptr);

        execPlan.rootPlan->length    = length;
        execPlan.rootPlan->dimension = 4;

        execPlan.rootPlan->outStride = strideOut;
        break;
    }
    default:
        throw std::runtime_error("unsupported transpose_brick dimension");
    }

    // Set input/output buffers - these will either be actual user
    // input/output (when packing/unpacking for communication), or we
    // want the kernel to use overridden pointers that we allocate
    // during plan creation, which are also passed to look like user
    // input/output pointers.
    execPlan.rootPlan->obIn  = OB_USER_IN;
    execPlan.rootPlan->obOut = OB_USER_OUT;

    execPlan.oLength            = execPlan.rootPlan->length;
    execPlan.rootPlan->inStride = strideIn;

    execPlan.rootPlan->precision = precision;
    execPlan.rootPlan->placement = rocfft_placement_notinplace;
    execPlan.rootPlan->iOffset   = offsetIn;
    execPlan.rootPlan->oOffset   = offsetOut;

    execPlan.rootPlan->inArrayType  = arrayType;
    execPlan.rootPlan->outArrayType = arrayType;

    execPlan.execSeq.push_back(execPlan.rootPlan.get());

    // only initialize the execPlan with kernels, twiddles, etcn if it
    // will run on this rank
    if(local_comm_rank != location.comm_rank)
        return execPlanMultiItem;

    rocfft_scoped_device dev(location.device);
    execPlan.rootPlan->CreateDevKernelArgs();

    execPlan.rootPlan->comments.emplace_back(std::move(description));

    // TODO: on multi-rank plans, we should only compile for the current rank
    RuntimeCompilePlan(execPlan);

    // grid params are set during runtime compilation, put them on
    // the execPlan so they're known at exec time
    auto& gp       = execPlan.gridParam.emplace_back();
    dim3  gridDim  = execPlan.execSeq.front()->compiledKernel.get()->gridDim;
    dim3  blockDim = execPlan.execSeq.front()->compiledKernel.get()->blockDim;
    gp.b_x         = gridDim.x;
    gp.b_y         = gridDim.y;
    gp.b_z         = gridDim.z;
    gp.wgs_x       = blockDim.x;
    gp.wgs_y       = blockDim.y;
    gp.wgs_z       = blockDim.z;

    return execPlanMultiItem;
}

size_t rocfft_plan_description_t::rank() const
{
    const auto ret = input_layout.get_len_rank();
    if(ret != output_layout.get_len_rank())
        throw std::logic_error("Inconsistent ranks between input and output layouts detected by "
                               + ROCFFT_CURRENT_FUNCTION);

    return ret;
}

size_t rocfft_plan_description_t::batch() const
{
    if(input_layout.get_batch_rank() != 1 || output_layout.get_batch_rank() != 1)
        throw std::logic_error(
            "Unexpected number of batch dimensions detected in input or output layout by "
            + ROCFFT_CURRENT_FUNCTION);
    const auto ret = input_layout.batch();
    if(ret != output_layout.batch())
        throw std::logic_error(
            "Inconsistent batch values between input and output layouts detected by "
            + ROCFFT_CURRENT_FUNCTION);

    return ret;
}

data_layout_t rocfft_field_t::get_full_data_range() const
{
    if(bricks.empty())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " cannot operate on empty fields");
    data_layout_t ret;
    ret.len_axes.resize(bricks[0].layout.get_len_rank());
    ret.batch_axes.resize(bricks[0].layout.get_batch_rank());
    for(const auto& brick : bricks)
    {
        if(!brick.layout.is_dimensionally_consistent_with(ret))
        {
            throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                                   + " detected a pair of dimensionally-inconsistent bricks");
        }
        for(size_t dim = 0; dim < ret.get_full_rank(); dim++)
            ret[dim].upper = std::max(ret[dim].upper, brick.layout[dim].upper);
    }
    // finalize: set contiguous strides and mark all layout's axes as full
    for(size_t dim = 0; dim < ret.get_full_rank(); dim++)
    {
        ret[dim].is_partial = false;
        if(dim == 0)
            ret[dim].inbuffer_stride = 1;
        else
            ret[dim].inbuffer_stride = ret[dim - 1].inbuffer_stride * ret[dim - 1].logical_span();
    }
    return ret;
}

void rocfft_field_t::finalize()
{
    const auto full_data_range = get_full_data_range();

    // make sure bricks are sorted by increasing ranks (without modifying
    // brick ordering within ranks)
    std::stable_sort(
        bricks.begin(), bricks.end(), [](const rocfft_brick_t& a, const rocfft_brick_t& b) {
            return a.location.comm_rank < b.location.comm_rank;
        });

    // Set the `is_partial` flags for all dimensions of the bricks' layouts, according to
    // the given full range of logical indices
    std::for_each(bricks.begin(), bricks.end(), [&full_data_range](auto& brick) {
        for(size_t dim = 0; dim < brick.layout.get_full_rank(); dim++)
        {
            brick.layout[dim].is_partial
                = !brick.layout[dim].has_same_logical_range_as(full_data_range[dim]);
        }
    });
}

bool rocfft_field_t::has_valid_tessellation() const
{
    size_t total_brick_logical_count = 0;
    for(auto brickI = bricks.begin(); brickI != bricks.end(); ++brickI)
    {
        for(auto brickJ = brickI + 1; brickJ != bricks.end(); ++brickJ)
        {
            const auto ij_intersection
                = data_layout_t::make_contiguous_intersection_of(brickI->layout, brickJ->layout);
            if(!ij_intersection.is_empty())
                return false;
        }
        // Add up the brick's logical count
        total_brick_logical_count += brickI->layout.logical_count();
    }

    const auto full_data_range = get_full_data_range();
    // The bricks cover the whole index space iff their total number of
    // logical elements is the same as the full data layout
    return full_data_range.logical_count() == total_brick_logical_count;
}

rocfft_status
    rocfft_plan_description_t::finalize_and_validate_for(rocfft_transform_type   dft_type,
                                                         rocfft_result_placement placement,
                                                         const size_t*           user_lengths,
                                                         const size_t            len_rank,
                                                         const size_t number_of_transforms)
{
    // -------------------------------------
    //   Finalize and validate I/O layouts
    // -------------------------------------
    for(auto io : {io_data_label::INPUT, io_data_label::OUTPUT})
    {
        const bool is_real_domain
            = (dft_type == rocfft_transform_type_real_forward && io == io_data_label::INPUT)
              || (dft_type == rocfft_transform_type_real_inverse && io == io_data_label::OUTPUT);
        const bool is_hermitian_domain
            = (dft_type == rocfft_transform_type_real_forward && io == io_data_label::OUTPUT)
              || (dft_type == rocfft_transform_type_real_inverse && io == io_data_label::INPUT);
        // Finalize the I/O layouts with the user-provided lengths
        auto&               io_layout = io == io_data_label::INPUT ? input_layout : output_layout;
        std::vector<size_t> io_lengths(user_lengths, user_lengths + len_rank);
        if(is_hermitian_domain)
            io_lengths[0] = user_lengths[0] / 2 + 1;
        else
            io_lengths[0] = user_lengths[0];
        const auto current_strides = io_layout.strides();
        // Current strides may be empty (if unset by user for implicit default)
        // If not, users must have set len_rank of them
        if(!current_strides.empty() && current_strides.size() != len_rank)
        {
            return rocfft_status_invalid_strides;
        }
        const auto current_distances = io_layout.distances();
        // Current distances may be empty (if 0-initialized by the user for implicit default)
        // If not, users cannot possibly have set more than 1 (multi-dimensional batches not
        // exposed to users)
        if(!current_distances.empty() && current_distances.size() != 1)
        {
            // cannot happen, theoretically
            throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                                   + " detected an unexpected batch rank in a plan description");
        }
        io_layout.full_range_reset(io_lengths,
                                   current_strides /* default strides set if empty */,
                                   {number_of_transforms},
                                   current_distances /* default distances if empty */,
                                   is_real_domain && placement == rocfft_placement_inplace);
    }
    // -------------------------------------------
    //   Finalize and validate I/O fields if any
    // -------------------------------------------
#ifdef ROCFFT_MPI_ENABLE
    const auto rc_mpi = allgather_brick_params_mpi();
    if(rc_mpi != rocfft_status_success)
        return rc_mpi;
#endif
    for(auto io : {io_data_label::INPUT, io_data_label::OUTPUT})
    {
        const auto& expected_full_range = io == io_data_label::INPUT ? input_layout : output_layout;
        auto&       io_fields           = io == io_data_label::INPUT ? inFields : outFields;
        std::for_each(
            io_fields.begin(), io_fields.end(), [](auto& io_field) { io_field.finalize(); });
        if(std::any_of(io_fields.begin(), io_fields.end(), [&expected_full_range](auto& io_field) {
               const auto field_full_range = io_field.get_full_data_range();
               return !expected_full_range.is_dimensionally_consistent_with(field_full_range)
                      || !expected_full_range.has_same_logical_range_as(field_full_range);
           }))
        {
            // This would warrant a dedicated error code. For now, throw an std::runtime_error so
            // that ROCFFT_LAYER reports something insightful
            throw std::runtime_error("Inconsistent full data range detected for an " + to_str(io)
                                     + " field");
        }
        if(io_fields.empty())
        {
            if(comm_type != rocfft_comm_none)
            {
                // This would warrant a dedicated error code. For now, throw an std::runtime_error so
                // that ROCFFT_LAYER reports something insightful
                throw std::runtime_error("Multi-process transforms require both input and output "
                                         "fields to be specified");
            }
            else
            {
                // Single-proc: users may not set a field with a single brick in case of
                // undistributed input/output data. Internally, it may be convenient to
                // assimilate such usage to a lone-brick field usage: the internally-created
                // single_dev_ifield and/or single_dev_ofield serve that purpose.
                auto& single_dev_io_field
                    = io == io_data_label::INPUT ? single_dev_ifield : single_dev_ofield;
                single_dev_io_field = std::make_optional<rocfft_field_t>();
                single_dev_io_field->bricks.reserve(1);
                single_dev_io_field->bricks.emplace_back(expected_full_range,
                                                         get_current_location());
            }
        }
        if(std::any_of(io_fields.begin(), io_fields.end(), [](auto& field) {
               return !field.has_valid_tessellation();
           }))
        {
            throw std::runtime_error("Invalid tessellation detected for an " + to_str(io)
                                     + " field");
        }
    }

    if(inFields.size() > 1 || outFields.size() > 1)
    {
        // This would warrant a dedicated error code, e.g., rocfft_status_unsupported. For now,
        // throw an std::runtime_error so that ROCFFT_LAYER reports something insightful
        throw std::runtime_error("Multi-field transforms are not supported yet");
    }

    // -------------------------------------------------------------------------------
    // Remove trivial axes of unit lengths from layouts (and from all bricks' layouts,
    // too) and sort length axes by increasing strides (in corresponding bricks, too)
    // if that can be done consistently across ALL (full AND partial) data layouts at
    // play.
    // -------------------------------------------------------------------------------

    if(rank() > 1)
    {
        auto may_erase_length_dimension = [this](size_t len_dim, io_data_label io) -> bool {
            const auto& io_full_layout = io == io_data_label::INPUT ? input_layout : output_layout;
            const auto& io_fields      = io == io_data_label::INPUT ? inFields : outFields;
            return len_dim < io_full_layout.get_len_rank()
                   && io_full_layout[len_dim].logical_span() == 1
                   && std::all_of(
                       io_fields.begin(), io_fields.end(), [&len_dim](const auto& field) {
                           return std::all_of(field.bricks.begin(),
                                              field.bricks.end(),
                                              [&len_dim](const auto& brick) {
                                                  return len_dim < brick.layout.get_len_rank()
                                                         && brick.layout[len_dim].logical_span()
                                                                == 1;
                                              });
                       });
        };

        // 0th length dimension must not be modified for real DFTs
        const bool is_real_dft = dft_type == rocfft_transform_type_real_forward
                                 || dft_type == rocfft_transform_type_real_inverse;
        size_t len_dim = is_real_dft ? 1 : 0;
        while(len_dim < rank() && 1 < rank())
        {
            if(may_erase_length_dimension(len_dim, io_data_label::INPUT)
               && may_erase_length_dimension(len_dim, io_data_label::OUTPUT))
            {
                for(auto io : {io_data_label::INPUT, io_data_label::OUTPUT})
                {
                    auto& io_layout = io == io_data_label::INPUT ? input_layout : output_layout;
                    auto& io_fields = io == io_data_label::INPUT ? inFields : outFields;
                    io_layout.len_axes.erase(io_layout.len_axes.begin() + len_dim);
                    std::for_each(io_fields.begin(), io_fields.end(), [&len_dim](auto& field) {
                        std::for_each(
                            field.bricks.begin(), field.bricks.end(), [&len_dim](auto& brick) {
                                brick.layout.len_axes.erase(brick.layout.len_axes.begin()
                                                            + len_dim);
                            });
                    });
                }
                // no incrementing of len_dim
            }
            else
                len_dim++;
        }
        if(rank() > (is_real_dft ? 2 : 1))
        {
            // Sort qualifying dimensions by increasing strides, if that can be done
            // consistently across all relevant layouts at play
            auto bricks_have_consistent_stride_ordering
                = [this, is_real_dft](io_data_label io) -> bool {
                const auto& io_full_layout
                    = io == io_data_label::INPUT ? input_layout : output_layout;
                const auto& io_fields = io == io_data_label::INPUT ? inFields : outFields;
                return std::all_of(
                    io_fields.begin(),
                    io_fields.end(),
                    [&io_full_layout, is_real_dft](const auto& field) {
                        return std::all_of(
                            field.bricks.begin(),
                            field.bricks.end(),
                            [&io_full_layout, is_real_dft](const auto& brick) {
                                return io_full_layout.length_axes_by_increasing_strides(is_real_dft)
                                       == brick.layout.length_axes_by_increasing_strides(
                                           is_real_dft);
                            });
                    });
            };

            if(input_layout.length_axes_by_increasing_strides(is_real_dft)
                   == output_layout.length_axes_by_increasing_strides(is_real_dft)
               && bricks_have_consistent_stride_ordering(io_data_label::INPUT)
               && bricks_have_consistent_stride_ordering(io_data_label::OUTPUT))
            {
                const auto reordered_length_axes
                    = input_layout.length_axes_by_increasing_strides(is_real_dft);
                for(auto io : {io_data_label::INPUT, io_data_label::OUTPUT})
                {
                    auto& io_layout = io == io_data_label::INPUT ? input_layout : output_layout;
                    auto& io_fields = io == io_data_label::INPUT ? inFields : outFields;
                    io_layout.reorder_length_axes(reordered_length_axes);
                    std::for_each(
                        io_fields.begin(), io_fields.end(), [&reordered_length_axes](auto& field) {
                            std::for_each(field.bricks.begin(),
                                          field.bricks.end(),
                                          [&reordered_length_axes](auto& brick) {
                                              brick.layout.reorder_length_axes(
                                                  reordered_length_axes);
                                          });
                        });
                }
            }
        }
    }

    // -----------------------------------------
    //   Finalize and validate I/O array types
    // -----------------------------------------
    for(auto io : {io_data_label::INPUT, io_data_label::OUTPUT})
    {
        auto&      io_array_type = io == io_data_label::INPUT ? inArrayType : outArrayType;
        const bool is_real_domain
            = (dft_type == rocfft_transform_type_real_forward && io == io_data_label::INPUT)
              || (dft_type == rocfft_transform_type_real_inverse && io == io_data_label::OUTPUT);
        const bool is_hermitian_domain
            = (dft_type == rocfft_transform_type_real_forward && io == io_data_label::OUTPUT)
              || (dft_type == rocfft_transform_type_real_inverse && io == io_data_label::INPUT);

        if(io_array_type == rocfft_array_type_unset)
        {
            switch(dft_type)
            {
            case rocfft_transform_type_complex_forward:
            case rocfft_transform_type_complex_inverse:
                io_array_type = rocfft_array_type_complex_interleaved;
                break;
            case rocfft_transform_type_real_forward:
            case rocfft_transform_type_real_inverse:
                io_array_type = is_real_domain ? rocfft_array_type_real
                                               : rocfft_array_type_hermitian_interleaved;
                break;
            default:
                throw std::invalid_argument("Unexpected type of transform given to "
                                            + ROCFFT_CURRENT_FUNCTION);
            }
        }

        // planar array types only supported with empty fields
        const auto& io_fields = io == io_data_label::INPUT ? inFields : outFields;
        if(array_type_is_planar(io_array_type) && !io_fields.empty())
            return rocfft_status_invalid_array_type;

        switch(dft_type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            if(!array_type_is_complex_but_not_hermitian(io_array_type))
                return rocfft_status_invalid_array_type;
            break;
        case rocfft_transform_type_real_forward:
        case rocfft_transform_type_real_inverse:
            if((is_real_domain && !array_type_is_real(io_array_type))
               || (is_hermitian_domain && !array_type_is_hermitian(io_array_type)))
                return rocfft_status_invalid_array_type;
            break;
        default:
            throw std::invalid_argument("Unexpected type of transform given to "
                                        + ROCFFT_CURRENT_FUNCTION);
        }
    }
    // -----------------------------------------
    //        In-place specific validations
    // -----------------------------------------
    if(placement == rocfft_placement_inplace)
    {
        // ----------------------------------------------------
        //      Validate array types for in-place operations
        // ----------------------------------------------------
        switch(dft_type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            // We need same array type for input and output
            if(inArrayType != outArrayType)
                return rocfft_status_invalid_array_type;
            break;
        case rocfft_transform_type_real_forward:
        case rocfft_transform_type_real_inverse:
        {
            // Hermitian array must be interleaved
            const auto hermitian_array_type
                = dft_type == rocfft_transform_type_real_forward ? outArrayType : inArrayType;
            if(!array_type_is_interleaved(hermitian_array_type))
                return rocfft_status_invalid_array_type;
        }
        break;
        default:
            throw std::invalid_argument("Unexpected type of transform given to "
                                        + ROCFFT_CURRENT_FUNCTION);
        }
        // ----------------------------------------------------------------------
        //   Consistency of data locations for multi-device in-place operations
        // ----------------------------------------------------------------------
        if(!inFields.empty() || !outFields.empty())
        {
            // Verify that input and output locations match
            if(inFields.empty() || outFields.empty())
            {
                const auto& first_corresponding_brick
                    = inFields.empty() ? outFields[0].bricks[0] : inFields[0].bricks[0];
                if(first_corresponding_brick.location != get_current_location())
                {
                    // This would warrant a dedicated error code, e.g.,
                    // "rocfft_status_invalid_description". For now, throw an std::runtime_error
                    // so that ROCFFT_LAYER reports something insightful
                    throw std::runtime_error(
                        "If only one field is used and in-place operations are required, the first "
                        "brick on the first field must be located on the current device");
                }
            }
            else
            {
                // Fields are used for both input and output, parse all bricks in
                // parallel and verify per-rank consistency of locations
                const auto local_rank = get_local_comm_rank();
                for(size_t field_idx = 0; field_idx < std::min(inFields.size(), outFields.size());
                    field_idx++)
                {
                    const auto local_ibricks = std::equal_range(inFields[field_idx].bricks.begin(),
                                                                inFields[field_idx].bricks.end(),
                                                                local_rank,
                                                                match_comm_rank());
                    const auto local_obricks = std::equal_range(outFields[field_idx].bricks.begin(),
                                                                outFields[field_idx].bricks.end(),
                                                                local_rank,
                                                                match_comm_rank());
                    auto       ibrick        = local_ibricks.first;
                    auto       obrick        = local_obricks.first;
                    while(ibrick != local_ibricks.second && obrick != local_obricks.second)
                    {
                        if(ibrick->location.device != obrick->location.device)
                        {
                            throw std::runtime_error(
                                "Inconsistent brick locations for in-place operations: input brick "
                                + ibrick->str() + " and output brick " + obrick->str()
                                + " were expected on the same device");
                        }
                        ibrick++;
                        obrick++;
                    }
                }
            }
        }
        // -------------------------------------------------------------------
        //          Consistency of data layouts for in-place operations
        // -------------------------------------------------------------------
        if(has_undistributed_io_on_current_location())
        {
            // Actual single-device usage (possibly with lone bricks, though).
            // Lone bricks' layouts take precedence, if used
            const auto& in_layout           = undistributed_layout_for(io_data_label::INPUT);
            const auto& out_layout          = undistributed_layout_for(io_data_label::OUTPUT);
            const auto  out_lengths         = output_layout.lengths();
            const auto  expected_out_layout = in_layout.get_other_inplace_layout_for(
                io_data_label::OUTPUT, dft_type, out_lengths[0] % 2 == 1);
            if(!expected_out_layout)
                throw std::runtime_error(
                    "Undistributed input layout is incompatible with in-place operations.");
            if(*expected_out_layout != out_layout)
                throw std::runtime_error("Undistributed input and output layout are not compatible "
                                         "for single-device in-place transforms.");
            // Check offsets if they're not to be ignored
            if(inFields.empty() && outFields.empty())
            {
                if(dft_type == rocfft_transform_type_complex_forward
                   || dft_type == rocfft_transform_type_complex_inverse)
                {
                    if(inOffset[0] != outOffset[0]
                       || (array_type_is_planar(inArrayType) && inOffset[1] != outOffset[1]))
                        throw std::runtime_error("Identical offsets are required for single-device "
                                                 "in-place complex transforms.");
                }
                else
                {
                    const auto& real_offset
                        = dft_type == rocfft_transform_type_real_forward ? inOffset : outOffset;
                    const auto& hermitian_offset
                        = dft_type == rocfft_transform_type_real_forward ? outOffset : inOffset;
                    if(real_offset[0] != 2 * hermitian_offset[0])
                        throw std::runtime_error(
                            "Inconsistent offsets for single-device in-place real transforms.");
                }
            }
            else if((inFields.empty() && inOffset[0] != 0 && !outFields.empty())
                    || (outFields.empty() && outOffset[0] != 0 && !inFields.empty()))
            {
                // If input (resp. output) is a lone-brick field while output (resp. input) has
                // no field set, the input (resp. output) offset values will be ignored and nonzero
                // offset values on output (resp. input) cannot be accepted (note: planar array
                // types not supported if any I/O field is set)
                throw std::runtime_error("Nonzero input (resp. output) offset values cannot be "
                                         "used for in-place operations if the output (resp. input) "
                                         "layout is dictated by a lone-brick field.");
            }
        }
        //else
        //{
        //    // TBD
        //    // "in-place" operations with non-trivial fields used on both inputs and outputs, so
        //    // offsets are ignored since brick layouts take precedence in that case.
        //    // TODO: parse input and output fields' bricks: for any brick that covers a non-partial
        //    // (i.e., full) length dimension that may be processed in-place, the corresponding
        //    // layout's requirements must be satisfied.
        //    // More developments required in data_layout_t for those purposes, e.g., sthg like
        //    // -----------------------------------------------------------------------------------------------
        //    // data_layout_t data_layout_t::extract_full_length_axes_for(rocfft_transform_type dft_type);
        //    // -----------------------------------------------------------------------------------------------
        //    // then continue if empty or query the returned object's get_other_inplace_layout_for and throw if
        //    // the returned optional has no value...
        //    //
        //}
    }
    return rocfft_status_success;
}

void rocfft_plan_t::AllocateInternalTempBuffers()
{
    for(auto& t : tempBuffers)
    {
        if(t.first.comm_rank != desc.get_local_comm_rank())
            continue;

        t.second->alloc(t.first.device);
        if(LOG_PLAN_ENABLED())
            *LogSingleton::GetInstance().GetPlanOS()
                << "temp buffer " << t.second->data() << ", device " << t.first.device
                << ", size_bytes " << t.second->get_size_bytes() << std::endl;
    }
}

// Given user-specified brick layout, return a vector of BufferPtrs
// that point to those bricks.  Assign comm_rank and rank-specific
// index on those BufferPtrs appropriately.  Constructor specifies
// whether user input or user output BufferPtrs are returned.
//
// For example, if the input brick layout says rank-0 has 2 bricks
// and rank-1 has 3, this will return 5 BufferPtrs:
//
// - rank-0 user input index 0
// - rank-0 user input index 1
// - rank-1 user input index 0
// - rank-1 user input index 1
// - rank-1 user input index 2
template <typename BufferPtrConstruct>
static std::vector<BufferPtr> GatherUserBuffers(BufferPtrConstruct                 ctor,
                                                const std::vector<rocfft_brick_t>& bricks)
{
    std::vector<BufferPtr> ret;

    auto rank_sorter = [](const rocfft_brick_t& a, const rocfft_brick_t& b) {
        return a.location.comm_rank < b.location.comm_rank;
    };

    // In a multi-process environment, we've gathered the bricks on
    // multiple ranks into the field.  Bricks from the same rank should
    // appear together in the field.  Assert that this is the case,
    // since the code below depends on it.
    if(!std::is_sorted(bricks.begin(), bricks.end(), rank_sorter))
    {
        throw std::runtime_error("bricks not sorted after gather");
    }

    // go over the range of bricks on each rank
    for(auto range = std::equal_range(
            bricks.begin(), bricks.end(), bricks.front().location.comm_rank, match_comm_rank());
        range.first != range.second;
        range = std::equal_range(
            range.second, bricks.end(), range.second->location.comm_rank, match_comm_rank()))
    {
        // track the index of the user-provided brick
        for(size_t userIdx = 0; range.first != range.second; ++range.first, ++userIdx)
        {
            ret.emplace_back(ctor(userIdx, range.first->location.comm_rank));
        }
        if(range.second == bricks.end())
            break;
    }
    return ret;
}

// test if the specified dimension is split up across separate bricks
// in the field
static bool DimensionSplitInField(size_t length, size_t dimIdx, const rocfft_field_t& field)
{
    for(const auto& b : field.bricks)
        if(b.layout.lengths_and_batches()[dimIdx] != length)
            return true;
    return false;
}

// Construct a single-device execPlan - fill out the provided
// execPlan with nodes to implement the FFT.
static std::unique_ptr<ExecPlan> BuildSingleDevicePlan(NodeMetaData&         rootPlanData,
                                                       int                   local_comm_rank,
                                                       rocfft_location_t     location,
                                                       rocfft_transform_type transformType,
                                                       const std::optional<LoadOps>&  loadOps,
                                                       const std::optional<StoreOps>& storeOps,
                                                       bool partOfMultiPlan)
{
    rocfft_scoped_device dev(location.device);

    auto execPlanMultiItem = std::make_unique<ExecPlan>(local_comm_rank, partOfMultiPlan, location);
    ExecPlan& execPlan     = *execPlanMultiItem;
    try
    {
        execPlan.deviceProp = rootPlanData.deviceProp;
        execPlan.rootPlan   = NodeFactory::CreateExplicitNode(rootPlanData, nullptr);

        // If we are doing tuning initializing now, we shouldn't apply any solution,
        // since we are trying enumerating solutions now
        if(TuningBenchmarker::GetSingleton().IsInitializingTuning() == false)
        {
            // Solutions do not consider strides.  Even-length real
            // transforms need special consideration, since the
            // even-length optimization is only valid for stride-1 on
            // the fastest dimension.  So only apply a solution if
            // we're not doing even-length real, or if fastest dim
            // stride is 1.
            const auto& realLength     = transformType == rocfft_transform_type_real_forward
                                             ? execPlan.rootPlan->length
                                             : execPlan.rootPlan->outputLength;
            const bool  evenLengthReal = (transformType == rocfft_transform_type_real_forward
                                         || transformType == rocfft_transform_type_real_inverse)
                                        && realLength.front() % 2 == 0;
            const bool stride1 = execPlan.rootPlan->inStride.front() == 1
                                 && execPlan.rootPlan->outStride.front() == 1;
            if(evenLengthReal && !stride1)
            {
                if(LOG_TRACE_ENABLED())
                {
                    (*LogSingleton::GetInstance().GetTraceOS())
                        << "transform is even-length real but not stride-1, not applying solution "
                           "map"
                        << std::endl;
                }
            }
            else
            {
                execPlan.rootScheme = ApplySolution(execPlan);
                if(execPlan.rootScheme)
                {
                    execPlan.rootPlan = nullptr;
                    execPlan.rootPlan = NodeFactory::CreateExplicitNode(
                        rootPlanData, nullptr, execPlan.rootScheme->curScheme);
                }
            }
        }

        execPlan.iLength = rootPlanData.length;
        execPlan.oLength
            = rootPlanData.outputLength.empty() ? rootPlanData.length : rootPlanData.outputLength;

        // setup isUnitStride values
        execPlan.rootPlan->inStrideUnit  = BufferIsUnitStride(execPlan, OB_USER_IN);
        execPlan.rootPlan->outStrideUnit = BufferIsUnitStride(execPlan, OB_USER_OUT);

        // set load/store ops on the root plan
        if(loadOps)
            execPlan.rootPlan->loadOps = loadOps;
        if(storeOps)
            execPlan.rootPlan->storeOps = storeOps;
        execPlan.inputPtr  = rootPlanData.input_buffer;
        execPlan.outputPtr = rootPlanData.output_buffer;

        // only allocate kernels, twiddles, etc if plan will run on this rank
        if(local_comm_rank != location.comm_rank)
            return execPlanMultiItem;

        // check if we are doing tuning init now. If yes, we just return
        // since we are not going to do the execution
        if(TuningBenchmarker::GetSingleton().IsInitializingTuning())
        {
            EnumerateTrees(execPlan);
            TuningBenchmarker::GetSingleton().GetPacket()->init_step = false;
            TuningBenchmarker::GetSingleton().GetPacket()->is_tuning = true;
            return execPlanMultiItem;
        }

        // TODO: more descriptions are needed
        ProcessNode(execPlan);

        // Plan is compiled, no need to alloc twiddles + kargs etc
        if(rocfft_getenv("ROCFFT_INTERNAL_COMPILE_ONLY") == "1")
            return execPlanMultiItem;

        if(!PlanPowX(execPlan)) // PlanPowX enqueues the GPU kernels by function
        {
            throw std::runtime_error("Unable to create execution plan.");
        }

        // When running each solution during tuning, get the information to packet,
        // then we can dump the information to a table for analysis
        if(TuningBenchmarker::GetSingleton().IsProcessingTuning())
        {
            if(!GetTuningKernelInfo(execPlan))
                throw std::runtime_error("Unable to get the solution info.");
        }

        return execPlanMultiItem;
    }
    catch(std::exception&)
    {
        if(LOG_PLAN_ENABLED())
            PrintNode(*LogSingleton::GetInstance().GetPlanOS(), execPlan);
        throw;
    }
}

const rocfft_field_t& rocfft_plan_description_t::get_field_for(io_data_label io,
                                                               size_t        field_idx) const
{
    if(io != io_data_label::INPUT && io != io_data_label::OUTPUT)
        throw std::invalid_argument("Invalid io value given to " + ROCFFT_CURRENT_FUNCTION);
    const auto& io_fields = io == io_data_label::INPUT ? inFields : outFields;
    if(field_idx >= std::max(io_fields.size(), (size_t)1))
        throw std::invalid_argument("Invalid field index value given to "
                                    + ROCFFT_CURRENT_FUNCTION);
    if(!io_fields.empty())
        return io_fields[field_idx];
    // no field was actually set by the user, i.e., I/O data is undistributed and on the
    // current device (implicitly)
    const auto& single_dev_io_field
        = io == io_data_label::INPUT ? single_dev_ifield : single_dev_ofield;
    if(!single_dev_io_field)
        throw std::logic_error("Incomplete/unfinalized plan description encountered by "
                               + ROCFFT_CURRENT_FUNCTION);
    return *single_dev_io_field;
}

std::vector<size_t>
    rocfft_plan_t::CreateInputGatheringItemsIfNeeded(const NodeMetaData&        exec_plan_metadata,
                                                     const rocfft_location_t&   exec_plan_location,
                                                     const std::vector<size_t>& antecedents)
{
    std::vector<size_t> gather_plan_items; // to be returned;

    const auto in_loc = desc.expected_undistributed_location_for(io_data_label::INPUT);
    if(in_loc && *in_loc == exec_plan_location
       && (exec_plan_metadata.input_buffer.ptr_type() == BufferPtr::PtrType::PTR_USER_IN
           || (placement == rocfft_placement_inplace
               && exec_plan_metadata.input_buffer.ptr_type() == BufferPtr::PTR_USER_OUT)))
    {
        // no need to gather, input data is readily available to the single-device execution plan
        return gather_plan_items;
    }

    const auto& input_bricks    = desc.get_field_for(io_data_label::INPUT).bricks;
    const auto  local_comm_rank = desc.get_local_comm_rank();
    const auto  input_buffers   = GatherUserBuffers(BufferPtr::user_input, input_bricks);
    const auto  input_elem_size = element_size(precision, desc.inArrayType);
    // data layout to be observed by the final results of the data-gathering steps
    const auto single_dev_input_layout = exec_plan_metadata.layout_for(io_data_label::INPUT);

    // Create node that captures data-gathering steps
    std::unique_ptr<CommGather> gather_node;
    if(std::all_of(input_bricks.begin(), input_bricks.end(), [&](const rocfft_brick_t& ibrick) {
           return ibrick.layout.is_continuous_in(single_dev_input_layout)
                  && (exec_plan_metadata.input_buffer.ptr_type() == BufferPtr::PtrType::PTR_TEMP
                      || ibrick.layout.is_contiguous());
       }))
    {
        // All inputs may and can be copied directly into the execution plan's input buffer
        gather_node              = std::make_unique<CommGather>(local_comm_rank,
                                                   precision,
                                                   desc.inArrayType,
                                                   exec_plan_location,
                                                   exec_plan_metadata.input_buffer);
        gather_node->description = "Gather input data of " + std::to_string(input_bricks.size())
                                   + " bricks directly into single-device plan's input buffer";
        for(size_t b_idx = 0; b_idx < input_bricks.size(); ++b_idx)
        {
            const auto& ibrick = input_bricks[b_idx];
            gather_node->AddOperation(local_comm_rank,
                                      {ibrick.location,
                                       input_buffers[b_idx],
                                       0 /* : srcOffset */,
                                       ibrick.layout.offset_in(single_dev_input_layout),
                                       ibrick.layout.buffer_element_count()});
        }
        gather_plan_items.push_back(AddMultiPlanItem(std::move(gather_node), antecedents));
    }
    else
    {
        // At least one of the inputs cannot be copied directly into the execution plan's
        // input buffer. A temporary buffer is used to pack input buffers locally before
        // transposing it to match the data layout that's expected by the execution plan.
        const data_layout_t packed_layout = data_layout_t::default_full_layout(
            single_dev_input_layout.lengths(), single_dev_input_layout.batch());
        TempBufferLease packing_temp_buffer(tempBuffers,
                                            local_comm_rank,
                                            exec_plan_location,
                                            packed_layout.logical_count() * input_elem_size);
        gather_node = std::make_unique<CommGather>(local_comm_rank,
                                                   precision,
                                                   desc.inArrayType,
                                                   exec_plan_location,
                                                   BufferPtr::temp(packing_temp_buffer.data()));

        gather_node->description = "Gather contiguously-packed input data chunks for "
                                   + std::to_string(input_bricks.size()) + " bricks";
        // save the raw pointer as gather_node will be moved below
        auto gather_node_raw_ptr = gather_node.get();
        // Add gather_node to the plan, operations are added to it below
        const auto gatherIdx = AddMultiPlanItem(std::move(gather_node), antecedents);
        // Devices may need to pack their data (locally) before having it transferred
        // to packing_temp_buffer
        std::vector<TempBufferLease> local_packed_chunk;
        size_t                       packed_offset = 0;
        for(size_t b_idx = 0; b_idx < input_bricks.size(); ++b_idx)
        {
            const auto& ibrick = input_bricks[b_idx];

            if(ibrick.layout.is_contiguous())
            {
                // a direct copy into packing_temp_buffer may be done
                gather_node_raw_ptr->AddOperation(local_comm_rank,
                                                  {ibrick.location,
                                                   input_buffers[b_idx],
                                                   0 /* : srcOffset */,
                                                   packed_offset,
                                                   ibrick.layout.logical_count()});
            }
            else
            {
                // Pre-pack data locally on the brick's device before transferring it
                // to packing_temp_buffer
                std::string local_pack_description
                    = "pack brick " + std::to_string(b_idx) + "'s input data on device "
                      + std::to_string(ibrick.location.device) + " (rank "
                      + std::to_string(ibrick.location.comm_rank) + ") before gather";
                local_packed_chunk.emplace_back(tempBuffers,
                                                local_comm_rank,
                                                ibrick.location,
                                                ibrick.layout.logical_count() * input_elem_size);
                const auto packIdx = AddMultiPlanItem(
                    transpose_brick(local_comm_rank,
                                    ibrick.location,
                                    ibrick.layout.lengths_and_batches(),
                                    precision,
                                    desc.inArrayType,
                                    input_buffers[b_idx],
                                    0 /* : offsetIn */,
                                    ibrick.layout.strides_and_distances(),
                                    BufferPtr::temp(local_packed_chunk.back().data()),
                                    0 /* : offsetOut */,
                                    ibrick.layout.contiguous_strides_and_distances(),
                                    std::move(local_pack_description)),
                    antecedents);
                AddAntecedent(gatherIdx, packIdx);

                gather_node_raw_ptr->AddOperation(
                    local_comm_rank,
                    {ibrick.location,
                     BufferPtr::temp(local_packed_chunk.back().data()),
                     0 /* : srcOffset */,
                     packed_offset,
                     ibrick.layout.logical_count()});
            }
            std::string description
                = "unpack brick " + std::to_string(b_idx)
                  + "'s contiguous data chunk into single-device plan's input buffer";
            gather_plan_items.push_back(
                AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                 exec_plan_location,
                                                 ibrick.layout.lengths_and_batches(),
                                                 precision,
                                                 desc.inArrayType,
                                                 BufferPtr::temp(packing_temp_buffer.data()),
                                                 packed_offset,
                                                 ibrick.layout.contiguous_strides_and_distances(),
                                                 exec_plan_metadata.input_buffer,
                                                 ibrick.layout.offset_in(single_dev_input_layout),
                                                 single_dev_input_layout.strides_and_distances(),
                                                 std::move(description)),
                                 {gatherIdx}));
            packed_offset += ibrick.layout.logical_count();
        }
    }
    return gather_plan_items;
}

std::vector<size_t>
    rocfft_plan_t::CreateOutputScatteringItemsIfNeeded(const NodeMetaData&      exec_plan_metadata,
                                                       const rocfft_location_t& exec_plan_location,
                                                       const std::vector<size_t>& antecedents)
{
    std::vector<size_t> scatter_plan_items; // to be returned;

    const auto out_loc = desc.expected_undistributed_location_for(io_data_label::OUTPUT);
    if(out_loc && *out_loc == exec_plan_location
       && (exec_plan_metadata.output_buffer.ptr_type() == BufferPtr::PtrType::PTR_USER_OUT
           || (placement == rocfft_placement_inplace
               && exec_plan_metadata.output_buffer.ptr_type() == BufferPtr::PTR_USER_IN)))
    {
        // no need to scatter, output data is written directly by the single-device execution plan
        return scatter_plan_items;
    }

    const auto& output_bricks    = desc.get_field_for(io_data_label::OUTPUT).bricks;
    const auto  local_comm_rank  = desc.get_local_comm_rank();
    const auto  output_buffers   = GatherUserBuffers(BufferPtr::user_output, output_bricks);
    const auto  output_elem_size = element_size(precision, desc.outArrayType);
    // data layout of the results to be scattered
    const auto single_dev_output_layout = exec_plan_metadata.layout_for(io_data_label::OUTPUT);

    // Create node that captures data-scattering steps
    std::unique_ptr<CommScatter> scatter_node;
    if(std::all_of(output_bricks.begin(), output_bricks.end(), [&](const rocfft_brick_t& obrick) {
           return obrick.layout.is_continuous_in(single_dev_output_layout)
                  && obrick.layout.is_contiguous();
       }))
    {
        // All outputs are continuous chunks of the execution plan's output buffer and the
        // chunks may be transferred directly
        scatter_node = std::make_unique<CommScatter>(
            precision, desc.outArrayType, exec_plan_location, exec_plan_metadata.output_buffer);
        scatter_node->description = "Scatter single-device plan's output buffer directly to "
                                    + std::to_string(output_bricks.size()) + " bricks";
        for(size_t b_idx = 0; b_idx < output_bricks.size(); ++b_idx)
        {
            const auto& obrick = output_bricks[b_idx];
            scatter_node->AddOperation(local_comm_rank,
                                       {obrick.location,
                                        output_buffers[b_idx],
                                        obrick.layout.offset_in(single_dev_output_layout),
                                        0 /* : destOffset*/,
                                        obrick.layout.buffer_element_count()});
        }
        scatter_plan_items.push_back(AddMultiPlanItem(std::move(scatter_node), antecedents));
    }
    else
    {
        // At least one of the outputs cannot be copied directly from the execution plan's
        // output buffer. A temporary buffer is used to pack the successive output chunks
        // contiguously. These (contiguous) chunks are then scattered and unpacked into
        // their respective destinations.
        const data_layout_t packed_layout = data_layout_t::default_full_layout(
            single_dev_output_layout.lengths(), single_dev_output_layout.batch());
        TempBufferLease packing_temp_buffer(tempBuffers,
                                            local_comm_rank,
                                            exec_plan_location,
                                            packed_layout.logical_count() * output_elem_size);
        scatter_node              = std::make_unique<CommScatter>(precision,
                                                     desc.outArrayType,
                                                     exec_plan_location,
                                                     BufferPtr::temp(packing_temp_buffer.data()));
        scatter_node->description = "Scatter contiguously-packed output data chunks for "
                                    + std::to_string(output_bricks.size()) + " bricks";
        // save the raw pointer as scatter_node will be moved below
        auto scatter_node_raw_ptr = scatter_node.get();
        // Add scatter_node to the plan, operations are added to it below
        const auto scatter_idx = AddMultiPlanItem(std::move(scatter_node), antecedents);
        // Devices may need to unpack their data (locally) after having received it from
        // packing_temp_buffer
        std::vector<TempBufferLease> local_packed_chunk;
        size_t                       packed_offset = 0;
        for(size_t b_idx = 0; b_idx < output_bricks.size(); b_idx++)
        {
            const auto& obrick = output_bricks[b_idx];
            std::string description
                = "pack brick " + std::to_string(b_idx) + "'s output data on device "
                  + std::to_string(obrick.location.device) + " (rank "
                  + std::to_string(obrick.location.comm_rank) + ") before scatter";

            const auto packIdx = AddMultiPlanItem(
                transpose_brick(local_comm_rank,
                                exec_plan_location,
                                obrick.layout.lengths_and_batches(),
                                precision,
                                desc.outArrayType,
                                exec_plan_metadata.output_buffer,
                                obrick.layout.offset_in(single_dev_output_layout),
                                single_dev_output_layout.strides_and_distances(),
                                BufferPtr::temp(packing_temp_buffer.data()),
                                packed_offset,
                                obrick.layout.contiguous_strides_and_distances(),
                                std::move(description)),
                antecedents);
            AddAntecedent(scatter_idx, packIdx);
            if(obrick.layout.is_contiguous())
            {
                // Bricks are packed to be contiguous - if output is the
                // same shape, then there's no need for unpacking
                scatter_node_raw_ptr->AddOperation(local_comm_rank,
                                                   {obrick.location,
                                                    output_buffers[b_idx],
                                                    packed_offset,
                                                    0 /* : destOffset */,
                                                    obrick.layout.logical_count()});
            }
            else
            {
                // allocate memory for packed data chunk local to destination device
                local_packed_chunk.emplace_back(tempBuffers,
                                                local_comm_rank,
                                                obrick.location,
                                                obrick.layout.logical_count() * output_elem_size);

                // send the data
                scatter_node_raw_ptr->AddOperation(
                    local_comm_rank,
                    {obrick.location,
                     BufferPtr::temp(local_packed_chunk.back().data()),
                     packed_offset,
                     0 /* : destOffset */,
                     obrick.layout.logical_count()});

                // unpack data after sending
                description = "unpack brick " + std::to_string(b_idx)
                              + "'s contiguous data chunk into user's output buffer";

                scatter_plan_items.push_back(AddMultiPlanItem(
                    transpose_brick(local_comm_rank,
                                    obrick.location,
                                    obrick.layout.lengths_and_batches(),
                                    precision,
                                    desc.outArrayType,
                                    BufferPtr::temp(local_packed_chunk.back().data()),
                                    0 /* : offsetIn */,
                                    obrick.layout.contiguous_strides_and_distances(),
                                    output_buffers[b_idx],
                                    0 /* : offsetOut */,
                                    obrick.layout.strides_and_distances(),
                                    std::move(description)),
                    {scatter_idx}));
            }
            packed_offset += obrick.layout.logical_count();
        }
    }
    return scatter_plan_items;
}

void rocfft_plan_t::MakeSingleDevPlanWithGatherScatterIfNeeded()
{
    // The single-device execution plan item always operates on the current device.
    // NOTE: until we have public API allowing users to set work buffers for
    // more than one device (the current one, implicitly), the execution plan
    // should not be instructed to operate on another device, as it may be
    // incapable of accessing any user-provided work buffer if peer-to-peer
    // device access is not enabled.
    const auto exec_plan_location = rocfft_location_t::rank0_current_device();
    const bool plan_is_single_dev = desc.has_undistributed_io_on_current_location();
    std::vector<TempBufferLease> leased_exec_plan_io;
    auto                         exec_plan_metadata
        = get_single_dev_exec_plan_metadata(leased_exec_plan_io, exec_plan_location);
    if(plan_is_single_dev && !leased_exec_plan_io.empty())
    {
        throw std::logic_error("Leased temporary I/O buffers were unexpectedly created for a "
                               "single-device plan configuration.");
    }

    std::vector<size_t> gather_items;
    if(!plan_is_single_dev)
        gather_items = CreateInputGatheringItemsIfNeeded(exec_plan_metadata, exec_plan_location);
    auto exec_item             = BuildSingleDevicePlan(exec_plan_metadata,
                                           desc.get_local_comm_rank(),
                                           exec_plan_location,
                                           transformType,
                                           desc.loadOps,
                                           desc.storeOps,
                                           !plan_is_single_dev);
    exec_item->description     = "Single-device FFT execution plan";
    const auto exec_item_index = AddMultiPlanItem(std::move(exec_item), gather_items);
    if(!plan_is_single_dev)
        CreateOutputScatteringItemsIfNeeded(
            exec_plan_metadata, exec_plan_location, {exec_item_index});
}

// Transform (complex-complex FFT) one dimension of a brick, by
// adding a multi-plan item to the rocfft_plan_t, and return the new
// item's index.  A brick is on a single device and has the specified
// length and stride.  Input and output may point to the same buffer.
//
// The specified dimension is assumed to be contiguous on the brick.
// Other dimensions (including batch) may have any length (including
// length 1).
//
// Specified antecedent items are required to complete before this
// new item will begin execution.
//
// NOTE: lengths and stride include batch dimension
static size_t C2CBrickOneDimension(rocfft_plan_t&                 plan,
                                   size_t                         dimIdx,
                                   rocfft_location_t              location,
                                   const std::vector<size_t>&     lengths,
                                   const std::vector<size_t>&     stride,
                                   BufferPtr                      input,
                                   BufferPtr                      output,
                                   const std::optional<LoadOps>&  loadOps,
                                   const std::optional<StoreOps>& storeOps,
                                   const std::vector<size_t>&     antecedents)
{
    auto transformLengths = lengths;
    auto transformStride  = stride;

    // move the dimension-we-want-to-transform to the front
    std::swap(transformLengths.front(), transformLengths[dimIdx]);
    std::swap(transformStride.front(), transformStride[dimIdx]);

    NodeMetaData rootPlanData(nullptr);

    rootPlanData.batch = transformLengths.back();
    rootPlanData.iDist = transformStride.back();
    rootPlanData.oDist = transformStride.back();
    transformLengths.pop_back();
    transformStride.pop_back();

    rootPlanData.dimension = 1;
    rootPlanData.length    = transformLengths;
    rootPlanData.inStride  = transformStride;
    rootPlanData.outStride = transformStride;
    rootPlanData.direction = plan.transformType == rocfft_transform_type_complex_forward
                                     || plan.transformType == rocfft_transform_type_real_forward
                                 ? -1
                                 : 1;
    rootPlanData.placement
        = input == output ? rocfft_placement_inplace : rocfft_placement_notinplace;
    rootPlanData.precision         = plan.precision;
    rootPlanData.inArrayType       = rocfft_array_type_complex_interleaved;
    rootPlanData.outArrayType      = rocfft_array_type_complex_interleaved;
    rootPlanData.rootTransformType = plan.transformType;
    rootPlanData.deviceProp        = get_curr_device_prop();
    rootPlanData.input_buffer      = input;
    rootPlanData.output_buffer     = output;

    auto singlePlan = BuildSingleDevicePlan(rootPlanData,
                                            plan.desc.get_local_comm_rank(),
                                            location,
                                            plan.transformType,
                                            loadOps,
                                            storeOps,
                                            true);
    return plan.AddMultiPlanItem(std::move(singlePlan), antecedents);
}

void rocfft_plan_t::C2CField(const rocfft_field_t&          field,
                             const std::vector<size_t>&     fftDims,
                             std::vector<BufferPtr>&        input,
                             std::vector<BufferPtr>&        output,
                             const std::optional<LoadOps>&  loadOps,
                             const std::optional<StoreOps>& storeOps,
                             const std::vector<size_t>&     inputAntecedents,
                             std::vector<size_t>&           outputItems)
{
    outputItems.resize(field.bricks.size());

    for(size_t i = 0; i < field.bricks.size(); ++i)
    {
        const auto& inBrick = field.bricks[i];

        std::vector<size_t> antecedents;
        BufferPtr           fftInput = input[i];
        for(auto item : inputAntecedents)
        {
            if(multiPlan[item]->WritesToBuffer(fftInput))
                antecedents.push_back(item);
        }

        for(auto dimIdx : fftDims)
        {
            // apply load ops to first dimension we transform and store ops to the last
            std::optional<LoadOps> appliedLoadOps
                = dimIdx == fftDims.front() ? loadOps : std::nullopt;
            std::optional<StoreOps> appliedStoreOps
                = dimIdx == fftDims.back() ? storeOps : std::nullopt;

            auto transformItem              = C2CBrickOneDimension(*this,
                                                      dimIdx,
                                                      inBrick.location,
                                                      inBrick.layout.lengths_and_batches(),
                                                      inBrick.layout.strides_and_distances(),
                                                      fftInput,
                                                      output[i],
                                                      appliedLoadOps,
                                                      appliedStoreOps,
                                                      antecedents);
            multiPlan[transformItem]->group = "fft_dim_" + std::to_string(dimIdx);
            multiPlan[transformItem]->description
                = "FFT dim " + std::to_string(dimIdx) + " brick " + std::to_string(i);

            antecedents    = {transformItem};
            outputItems[i] = transformItem;
            fftInput       = output[i];
        }
    }
}

// Return a transposed field layout that makes the specified
// dimension contiguous on all bricks.  Length covers the whole field
// and includes batch dimension.  Input field is provided so we can
// distribute output bricks among the same devices that the input
// bricks are distributed to.
static rocfft_field_t MakeFieldDimContiguous(const rocfft_field_t&      field,
                                             const std::vector<size_t>& length,
                                             size_t                     dimIdx)
{
    rocfft_field_t out = field;
    // find first dim that's not the one we're making contiguous and
    // is at least as big as the number of bricks - we can split on
    // that dimension
    std::optional<size_t> splitDim;
    for(size_t dim = 0; dim < length.size(); ++dim)
    {
        if(dim != dimIdx && length[dim] >= field.bricks.size())
            splitDim = dim;
    }
    if(!splitDim)
        throw std::runtime_error("not enough lengths to split to make dim contiguous");

    std::vector<size_t> brick_lower(length.size()), brick_upper(length.size()),
        brick_strides(length.size());
    for(size_t i = 0; i < out.bricks.size(); ++i)
    {
        // reset lower and upper to origin and max
        std::fill(brick_lower.begin(), brick_lower.end(), 0);
        std::copy(length.begin(), length.end(), brick_upper.begin());

        // divide up the split dim
        brick_lower[*splitDim] = length[*splitDim] / out.bricks.size() * i;
        // last brick needs to include the whole length
        if(i == out.bricks.size() - 1)
            brick_upper[*splitDim] = length[*splitDim];
        else
            brick_upper[*splitDim] = length[*splitDim] / out.bricks.size() * (i + 1);
        // set strides - contiguous dim has stride 1
        size_t dist           = 1;
        brick_strides[dimIdx] = dist;
        dist *= (brick_upper[dimIdx] - brick_lower[dimIdx]);
        // split dim is contiguous after that
        brick_strides[*splitDim] = dist;
        dist *= (brick_upper[*splitDim] - brick_lower[*splitDim]);
        // fill in remaining strides
        for(size_t s = 0; s < brick_strides.size(); ++s)
        {
            if(s == dimIdx || s == *splitDim)
                continue;
            brick_strides[s] = dist;
            dist *= (brick_upper[s] - brick_lower[s]);
        }

        out.bricks[i].layout = data_layout_t{brick_lower, brick_upper, brick_strides};
    }
    out.finalize();
    return out;
}

void rocfft_plan_t::GlobalTranspose(size_t                     elem_size,
                                    const rocfft_field_t&      inField,
                                    const rocfft_field_t&      outField,
                                    std::vector<BufferPtr>&    input,
                                    std::vector<BufferPtr>&    output,
                                    const std::vector<size_t>& inputAntecedents,
                                    std::vector<size_t>&       outputItems,
                                    size_t                     transposeNumber)
{
    // All-to-all transpose is preferred as it's faster. This requires
    // that each rank have a single base pointer to send/receive with
    // offsets for every other rank.
    // That's only feasible if each rank has data on just one device
    // (since we can hipMalloc a single buffer per device and have
    // offsets into it).

    // Fall back to point-to-point transfers if all-to-all is not
    // possible.
    std::string itemGroup = "transpose_" + std::to_string(transposeNumber);
    if(rocfft_plan_description_t::multiple_devices_in_rank(inField)
       || rocfft_plan_description_t::multiple_devices_in_rank(outField))
    {
        GlobalTransposeP2P(
            elem_size, inField, outField, input, output, inputAntecedents, outputItems, itemGroup);
    }
    else
    {
        // GlobalTransposeA2A will use MPI_Ialltoall when possible,
        // falling back to MPI_Ialltoallv otherwise
        GlobalTransposeA2A(
            elem_size, inField, outField, input, output, inputAntecedents, outputItems, itemGroup);
    }
}

void rocfft_plan_t::GlobalTransposeP2P(size_t                     elem_size,
                                       const rocfft_field_t&      inField,
                                       const rocfft_field_t&      outField,
                                       std::vector<BufferPtr>&    input,
                                       std::vector<BufferPtr>&    output,
                                       const std::vector<size_t>& inputAntecedents,
                                       std::vector<size_t>&       outputItems,
                                       const std::string&         itemGroup)
{
    std::vector<TempBufferLease> packBufs;

    const auto local_comm_rank = desc.get_local_comm_rank();

    // loop over each input brick, finding the intersection of it with
    // every output brick
    for(size_t inBrickIdx = 0; inBrickIdx < inField.bricks.size(); ++inBrickIdx)
    {
        const auto& inBrick = inField.bricks[inBrickIdx];
        for(size_t outBrickIdx = 0; outBrickIdx < outField.bricks.size(); ++outBrickIdx)
        {
            const auto& outBrick = outField.bricks[outBrickIdx];

            const auto intersection
                = data_layout_t::make_contiguous_intersection_of(inBrick.layout, outBrick.layout);
            if(intersection.is_empty())
                continue;

            // pack data for communication
            packBufs.reserve(packBufs.size() + 2);
            TempBufferLease& pack     = packBufs.emplace_back(tempBuffers,
                                                          local_comm_rank,
                                                          inBrick.location,
                                                          intersection.logical_count() * elem_size);
            TempBufferLease& recv     = packBufs.emplace_back(tempBuffers,
                                                          local_comm_rank,
                                                          outBrick.location,
                                                          intersection.logical_count() * elem_size);
            auto             packIdx  = AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                            inBrick.location,
                                                            intersection.lengths_and_batches(),
                                                            precision,
                                                            desc.inArrayType,
                                                            input[inBrickIdx],
                                                            intersection.offset_in(inBrick.layout),
                                                            inBrick.layout.strides_and_distances(),
                                                            BufferPtr::temp(pack.data()),
                                                            0,
                                                            intersection.strides_and_distances(),
                                                            "pack brick for global transpose"),
                                            {inputAntecedents[inBrickIdx]});
            multiPlan[packIdx]->group = itemGroup;
            multiPlan[packIdx]->description
                = "pack " + std::to_string(inBrickIdx) + " + " + std::to_string(outBrickIdx);

            // send packed data
            auto sendOp = std::make_unique<CommPointToPoint>(local_comm_rank,
                                                             precision,
                                                             desc.inArrayType,
                                                             intersection.logical_count(),
                                                             inBrick.location,
                                                             BufferPtr::temp(pack.data()),
                                                             0,
                                                             outBrick.location,
                                                             BufferPtr::temp(recv.data()),
                                                             0);

            auto sendIdx              = AddMultiPlanItem(std::move(sendOp), {packIdx});
            multiPlan[sendIdx]->group = itemGroup;
            multiPlan[sendIdx]->description
                = "send " + std::to_string(inBrickIdx) + " + " + std::to_string(outBrickIdx);

            // unpack data on destination to output
            auto unpackIdx
                = AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                   outBrick.location,
                                                   intersection.lengths_and_batches(),
                                                   precision,
                                                   desc.inArrayType,
                                                   BufferPtr::temp(recv.data()),
                                                   0,
                                                   intersection.strides_and_distances(),
                                                   output[outBrickIdx],
                                                   intersection.offset_in(outBrick.layout),
                                                   outBrick.layout.strides_and_distances(),
                                                   "unpack brick for global transpose"),
                                   {sendIdx});
            multiPlan[unpackIdx]->group = itemGroup;
            multiPlan[unpackIdx]->description
                = "unpack " + std::to_string(inBrickIdx) + " + " + std::to_string(outBrickIdx);
            outputItems.push_back(unpackIdx);
        }
    }
}

// Helper to compute disjoint sets of ranks that need to collectively
// communicate (e.g. during an all-to-all operation).  This
// information can be used to partition the global communicator into
// sub-communicators.
struct ConnectedRanks
{
    // Initialize parent array - each rank begins as its own parent
    ConnectedRanks(int comm_size)
        : parent(comm_size)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }
    std::vector<int> parent;

    // Get the root parent of a rank
    int get_root(int rank)
    {
        // If rank is its own parent, that's the root.
        // Otherwise, follow the links until we get to the root
        if(parent[rank] != rank)
            parent[rank] = get_root(parent[rank]);
        return parent[rank];
    }

    // Add a connection between rankA and rankB
    void add_connection(int rankA, int rankB)
    {
        // A and B are now part of the same set, so give them the same
        // root
        auto rootA = get_root(rankA);
        auto rootB = get_root(rankB);
        if(rootA != rootB)
            parent[rootA] = parent[rootB];
    }

    // Get the set of ranks that's connected to a specified rank.
    // MPI requires that the ranks in a sub-communicator be listed in
    // the same order on all ranks in that sub-communicator, so a
    // std::set helps ensure this ordering.
    std::set<int> get_connected(int rank)
    {
        // Compress the paths in the parent array so that each rank
        // points to its root parent
        for(int i = 0; i < static_cast<int>(parent.size()); ++i)
        {
            parent[i] = get_root(i);
        }

        // Return ranks that have the same parent
        std::set<int> ret;
        for(int i = 0; i < static_cast<int>(parent.size()); ++i)
        {
            if(parent[i] == parent[rank])
                ret.insert(i);
        }
        return ret;
    }
};

void rocfft_plan_t::GlobalTransposeA2A(size_t                     elem_size,
                                       const rocfft_field_t&      inField,
                                       const rocfft_field_t&      outField,
                                       std::vector<BufferPtr>&    input,
                                       std::vector<BufferPtr>&    output,
                                       const std::vector<size_t>& inputAntecedents,
                                       std::vector<size_t>&       outputItems,
                                       const std::string&         itemGroup)
{
    const auto local_comm_rank = desc.get_local_comm_rank();
    const auto local_comm_size = desc.get_local_comm_size();

    // for us to be attempting an AlltoAll, bricks send/received
    // from/to the local rank should be on only one device
    std::optional<int> local_send_device;
    std::optional<int> local_recv_device;

    // keep track of the things this rank will need to send/receive,
    // as we will be allocating a single buffer for each of
    // send+receive, and then sending/receiving to/from multiple
    // ranks into offsets of that buffer.
    size_t              cumulative_send_elems = 0;
    std::vector<size_t> send_offsets(local_comm_size);
    std::vector<size_t> send_counts(local_comm_size);
    size_t              cumulative_recv_elems = 0;
    std::vector<size_t> recv_offsets(local_comm_size);
    std::vector<size_t> recv_counts(local_comm_size);

    // pack operations before the all-to-all communication
    std::vector<size_t> pack_ops;
    // unpack operations after the all-to-all communication
    std::vector<size_t> unpack_ops;

    ConnectedRanks conn(local_comm_size);

    // loop over each input brick, finding the intersection of it with
    // every output brick
    for(size_t inBrickIdx = 0; inBrickIdx < inField.bricks.size(); ++inBrickIdx)
    {
        const auto& inBrick = inField.bricks[inBrickIdx];
        const auto  inRank  = inBrick.location.comm_rank;
        for(size_t outBrickIdx = 0; outBrickIdx < outField.bricks.size(); ++outBrickIdx)
        {
            const auto& outBrick = outField.bricks[outBrickIdx];
            const auto  outRank  = outBrick.location.comm_rank;

            const auto intersection
                = data_layout_t::make_contiguous_intersection_of(inBrick.layout, outBrick.layout);
            if(intersection.is_empty())
                continue;

            conn.add_connection(inRank, outRank);

            const auto elems = intersection.logical_count();

            if(inRank == local_comm_rank)
            {
                // assert that all send bricks on this rank use the
                // same device
                if(local_send_device && *local_send_device != inBrick.location.device)
                    throw std::runtime_error("multiple devices for send during AlltoAll");
                local_send_device     = inBrick.location.device;
                send_offsets[outRank] = cumulative_send_elems;
                send_counts[outRank]  = elems;
                cumulative_send_elems += elems;
            }
            if(outRank == local_comm_rank)
            {
                if(local_recv_device && *local_recv_device != outBrick.location.device)
                    throw std::runtime_error("multiple devices for recv during AlltoAll");
                local_recv_device    = outBrick.location.device;
                recv_offsets[inRank] = cumulative_recv_elems;
                recv_counts[inRank]  = elems;
                cumulative_recv_elems += elems;
            }
        }
    }

    // ensure that we actually found send/recv devices
    if(!local_send_device)
        throw std::runtime_error("no local send device for all-to-all communication");
    if(!local_recv_device)
        throw std::runtime_error("no local recv device for all-to-all communication");

    // now we know how much we'll need to send/recv in total on this
    // rank.  allocate send/receive buffers.
    TempBufferLease send_buf(tempBuffers,
                             local_comm_rank,
                             {local_comm_rank, *local_send_device},
                             cumulative_send_elems * elem_size);
    TempBufferLease recv_buf(tempBuffers,
                             local_comm_rank,
                             {local_comm_rank, *local_recv_device},
                             cumulative_recv_elems * elem_size);

    // go over all the intersections of in/out bricks again, this
    // time packing/unpacking the data to/from the newly-allocated
    // send/recv buffers
    for(size_t inBrickIdx = 0; inBrickIdx < inField.bricks.size(); ++inBrickIdx)
    {
        const auto& inBrick = inField.bricks[inBrickIdx];
        const auto  inRank  = inBrick.location.comm_rank;
        for(size_t outBrickIdx = 0; outBrickIdx < outField.bricks.size(); ++outBrickIdx)
        {
            const auto& outBrick = outField.bricks[outBrickIdx];
            const auto  outRank  = outBrick.location.comm_rank;

            const auto intersection
                = data_layout_t::make_contiguous_intersection_of(inBrick.layout, outBrick.layout);
            if(intersection.is_empty())
                continue;

            // this transpose will only actually run if inRank ==
            // local_comm_rank, but adding it to the plan so all
            // ranks see a consistent plan.
            //
            // if(inRank == local_comm_rank)
            {
                auto pack_op
                    = AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                       inBrick.location,
                                                       intersection.lengths_and_batches(),
                                                       precision,
                                                       desc.inArrayType,
                                                       input[inBrickIdx],
                                                       intersection.offset_in(inBrick.layout),
                                                       inBrick.layout.strides_and_distances(),
                                                       BufferPtr::temp(send_buf.data()),
                                                       send_offsets[outRank],
                                                       intersection.strides_and_distances(),
                                                       "pack brick for global transpose"),
                                       inputAntecedents);
                multiPlan[pack_op]->group = itemGroup;
                multiPlan[pack_op]->description
                    = "pack " + std::to_string(inBrickIdx) + " + " + std::to_string(outBrickIdx);

                pack_ops.push_back(pack_op);
            }
            // this transpose will only actually run if outRank ==
            // local_comm_rank, but adding it to the plan so all
            // ranks see a consistent plan
            //
            // if(outRank == local_comm_rank)
            {
                auto unpack_op
                    = AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                       outBrick.location,
                                                       intersection.lengths_and_batches(),
                                                       precision,
                                                       desc.inArrayType,
                                                       BufferPtr::temp(recv_buf.data()),
                                                       recv_offsets[inRank],
                                                       intersection.strides_and_distances(),
                                                       output[outBrickIdx],
                                                       intersection.offset_in(outBrick.layout),
                                                       outBrick.layout.strides_and_distances(),
                                                       "unpack brick for global transpose"),
                                       {});
                multiPlan[unpack_op]->group = itemGroup;
                multiPlan[unpack_op]->description
                    = "unpack " + std::to_string(inBrickIdx) + " + " + std::to_string(outBrickIdx);
                unpack_ops.push_back(unpack_op);
            }
        }
    }

    // get the set of ranks that this rank needs to communicate with
    std::set<int> comm_ranks = conn.get_connected(local_comm_rank);

    // if the set of ranks we need to communicate with is a subset,
    // narrow the AllToAll to just that set
    std::optional<MPI_Comm_wrapper_t> subcomm;
#ifdef ROCFFT_MPI_ENABLE
    if(comm_ranks.size() < static_cast<size_t>(local_comm_size))
    {
        // go backwards through the ranks
        for(int i = local_comm_size - 1; i >= 0; --i)
        {
            if(comm_ranks.find(i) == comm_ranks.end())
            {
                send_counts.erase(send_counts.begin() + i);
                send_offsets.erase(send_offsets.begin() + i);
                recv_counts.erase(recv_counts.begin() + i);
                recv_offsets.erase(recv_offsets.begin() + i);
            }
        }

        // create a subcommunicator with this subset - preserve order
        // enforced by the std::set so that all ranks see the same
        // ordering
        std::vector<int> comm_ranks_vec;
        std::copy(comm_ranks.begin(), comm_ranks.end(), std::back_inserter(comm_ranks_vec));
        subcomm = make_subcommunicator(desc.mpi_comm, comm_ranks_vec);
    }
#endif

    // add the all-to-all op itself, which depends on pack ops
    auto alltoall_ptr = std::make_unique<CommAllToAll>(precision,
                                                       desc.inArrayType,
                                                       send_offsets,
                                                       send_counts,
                                                       recv_offsets,
                                                       recv_counts,
                                                       BufferPtr::temp(send_buf.data()),
                                                       BufferPtr::temp(recv_buf.data()),
                                                       std::move(subcomm));

    auto alltoall_op                    = AddMultiPlanItem(std::move(alltoall_ptr), pack_ops);
    multiPlan[alltoall_op]->group       = itemGroup;
    multiPlan[alltoall_op]->description = "all-to-all communication";

    // update the unpack ops to depend on the all-to-all
    for(auto op : unpack_ops)
    {
        AddAntecedent(op, alltoall_op);
    }

    // subsequent operations can depend on the unpack ops
    outputItems = unpack_ops;
}

// geometry handling helpers for brick and field manipulation
inline std::array<int, 3> infer_grid_from_bricks(const std::vector<rocfft_brick_t>& bricks)
{
    std::set<size_t> xset, yset, zset;
    for(const auto& b : bricks)
    {
        const auto brick_lower = b.layout.lower();
        if(brick_lower.size() >= 1)
            xset.insert(brick_lower[0]);
        if(brick_lower.size() >= 2)
            yset.insert(brick_lower[1]);
        if(brick_lower.size() >= 3)
            zset.insert(brick_lower[2]);
    }
    int nx = std::max(1, static_cast<int>(xset.size()));
    int ny = std::max(1, static_cast<int>(yset.size()));
    int nz = std::max(1, static_cast<int>(zset.size()));
    return {nx, ny, nz};
}

// construct a pencil-decomposed field using a base shape
rocfft_field_t MakeFieldWithPencilSplit(const rocfft_field_t&      currentField,
                                        const std::vector<size_t>& lengthsWithBatch,
                                        const std::vector<int>&    split_axes,
                                        const std::vector<int>&    split_sizes)
{
    assert(split_axes.size() == 2 && split_sizes.size() == 2);

    int P = split_sizes[0], Q = split_sizes[1];
    int n_bricks = P * Q;

    rocfft_field_t out;
    out.bricks.reserve(n_bricks);

    const int           ndim = static_cast<int>(lengthsWithBatch.size());
    std::vector<size_t> brick_lower(ndim), brick_upper(ndim), brick_strides(ndim);
    for(int i = 0; i < 2; ++i)
    {
        int ax = split_axes[i];
        int sz = split_sizes[i];
        if(ax < 0 || ax >= ndim)
            throw std::out_of_range("split_axes[" + std::to_string(i) + "] value "
                                    + std::to_string(ax)
                                    + " is out of range for ndim=" + std::to_string(ndim));
        if(sz <= 0 || static_cast<size_t>(sz) > lengthsWithBatch[ax])
            throw std::invalid_argument("split_sizes[" + std::to_string(i) + "] ("
                                        + std::to_string(sz)
                                        + ") must be positive and not exceed axis length ("
                                        + std::to_string(lengthsWithBatch[ax]) + ")");
    }

    // distribute location/device assignments round-robin
    for(int i = 0; i < n_bricks; ++i)
    {
        // compute 2D (p, q) indices for this brick
        int p = i / Q;
        int q = i % Q;

        // set bounds for each axis
        std::fill(brick_lower.begin(), brick_lower.end(), 0);
        std::copy(lengthsWithBatch.begin(), lengthsWithBatch.end(), brick_upper.begin());

        int axis_p = split_axes[0];
        int axis_q = split_axes[1];

        int len_p           = lengthsWithBatch[axis_p];
        int len_q           = lengthsWithBatch[axis_q];
        brick_lower[axis_p] = len_p * p / P;
        brick_upper[axis_p] = len_p * (p + 1) / P;
        brick_lower[axis_q] = len_q * q / Q;
        brick_upper[axis_q] = len_q * (q + 1) / Q;

        // contiguous strides
        int dist = 1;
        for(size_t s = 0; s < brick_strides.size(); ++s)
        {
            brick_strides[s] = dist;
            dist *= brick_upper[s] - brick_lower[s];
        }

        // assign device/rank in a round-robin fashion
        const auto& ref_brick = currentField.bricks[i % currentField.bricks.size()];
        out.bricks.emplace_back(brick_lower, brick_upper, brick_strides, ref_brick.location);
    }
    out.finalize();

    return out;
}

// get transpose plan structure
inline grid_layout grid_kind(const std::array<int, 3>& grid)
{
    const int n_ones = std::count(grid.begin(), grid.end(), 1);
    if(n_ones == 2)
        return grid_layout::slab;
    if(n_ones == 1)
        return grid_layout::pencil;
    if(n_ones == 0)
        return grid_layout::brick;
    return grid_layout::invalid;
}

inline transpose_type get_transpose_type(const std::array<int, 3>& from,
                                         const std::array<int, 3>& to)
{
    return std::make_pair(grid_kind(from), grid_kind(to));
}

inline std::string grid_layout_str(grid_layout l)
{
    switch(l)
    {
    case grid_layout::slab:
        return "slab";
    case grid_layout::pencil:
        return "pencil";
    case grid_layout::brick:
        return "brick";
    default:
        return "invalid";
    }
}

inline std::string transpose_type_str(transpose_type t)
{
    return grid_layout_str(t.first) + "_to_" + grid_layout_str(t.second);
}

// heuristic method to create a processor grid
std::pair<int, int> get_most_balanced_proc_pair(int prod, int limit_a, int limit_b)
{
    int best_a = -1, best_b = -1, min_diff = prod;
    for(int a = 1; a <= prod; ++a)
    {
        if(prod % a == 0)
        {
            int b = prod / a;
            if(a <= limit_a && b <= limit_b)
            {
                int diff = std::abs(a - b);
                if(diff < min_diff)
                {
                    best_a   = a;
                    best_b   = b;
                    min_diff = diff;
                }
            }
        }
    }
    if(best_a == -1 || best_b == -1)
    {
        throw std::runtime_error(
            "get_most_balanced_proc_pair: Cannot decompose prod=" + std::to_string(prod)
            + " within given limits: limit_a=" + std::to_string(limit_a)
            + ", limit_b=" + std::to_string(limit_b)
            + " (limit_a * limit_b = " + std::to_string(limit_a * limit_b) + ")");
    }
    return {best_a, best_b};
}

void get_transpose_plan(const std::array<int, 3>&        input_grid,
                        const std::array<int, 3>&        output_grid,
                        const std::array<size_t, 3>&     global_lengths,
                        std::vector<std::array<int, 3>>& transpose_plan,
                        std::vector<transpose_type>&     transpose_types)
{
    transpose_plan.clear();
    transpose_types.clear();

    int                          prod = input_grid[0] * input_grid[1] * input_grid[2];
    std::set<std::array<int, 3>> pencils;

    // for each axis, generate the pencil with 1 in that axis, as balanced as possible
    for(int pos = 0; pos < 3; ++pos)
    {
        // the two axes to split (not the pencil axis)
        int split_a = (pos + 1) % 3;
        int split_b = (pos + 2) % 3;

        // find the best factorization that does not exceed axis lengths
        auto [best_a, best_b]
            = get_most_balanced_proc_pair(prod,
                                          static_cast<int>(global_lengths[split_a]),
                                          static_cast<int>(global_lengths[split_b]));

        // assign axes explicitly to avoid confusion
        std::array<int, 3> grid = {0, 0, 0};
        grid[pos]               = 1; // Pencil axis
        grid[split_a]           = best_a; // First split axis
        grid[split_b]           = best_b; // Second split axis

        if((grid != input_grid) && (grid != output_grid))
            pencils.insert(grid); // direct insert, no duplicate check needed
    }

    // full transpose plan is: input -> [all pencils] -> output
    transpose_plan.push_back(input_grid);
    for(const auto& g : pencils)
        transpose_plan.push_back(g);
    transpose_plan.push_back(output_grid);

    // generate transpose type sequence as pairs of grid_layouts
    for(size_t i = 1; i < transpose_plan.size(); ++i)
        transpose_types.push_back(get_transpose_type(transpose_plan[i - 1], transpose_plan[i]));
}

bool rocfft_plan_t::BuildOptMultiDevicePlan()
{
    const auto local_comm_rank = desc.get_local_comm_rank();

    // keep track of how many transposes we've done so we can log
    // distinct messages about each one
    size_t transposeNumber = 0;

    // currently, can only optimize c2c
    if(transformType != rocfft_transform_type_complex_forward
       && transformType != rocfft_transform_type_complex_inverse)
        return false;

    // must be out-of-place so that we don't have to worry about
    // overwriting an input before everything's done reading
    if(placement == rocfft_placement_inplace)
        return false;

    // input/output Fields must not be empty
    if(desc.inFields.empty() || desc.outFields.empty())
        return false;

    // work out what FFT dimensions are already contiguous in the fields
    std::vector<size_t> contiguousInputDims;
    std::vector<size_t> contiguousOutputDims;
    std::vector<size_t> nonContiguousDims;
    const auto          input_lengths = desc.input_layout.lengths();
    for(size_t dimIdx = 0; dimIdx < desc.rank(); ++dimIdx)
    {
        if(!DimensionSplitInField(input_lengths[dimIdx], dimIdx, desc.inFields.front()))
            contiguousInputDims.push_back(dimIdx);
        else if(!DimensionSplitInField(input_lengths[dimIdx], dimIdx, desc.outFields.front()))
            contiguousOutputDims.push_back(dimIdx);
        else
            nonContiguousDims.push_back(dimIdx);
    }

    // can optimize if at least one FFT dim is contiguous in input and output
    if(contiguousInputDims.empty() || contiguousOutputDims.empty())
        return false;

    const auto elem_size = element_size(precision, desc.inArrayType);

    // transform contiguous input dims

    // gather up input pointers and allocate temp storage for
    // FFTed contiguous input dims (since we don't want to
    // overwrite input)
    std::vector<BufferPtr> inputBufs
        = GatherUserBuffers(BufferPtr::user_input, desc.inFields.front().bricks);
    std::vector<BufferPtr> inputFFTBufs;
    inputBufs.reserve(desc.inFields.front().bricks.size());
    inputFFTBufs.reserve(desc.inFields.front().bricks.size());
    std::vector<TempBufferLease> inputTemp;
    inputTemp.reserve(desc.inFields.front().bricks.size());
    for(size_t inBrickIdx = 0; inBrickIdx < desc.inFields.front().bricks.size(); ++inBrickIdx)
    {
        const auto& inBrick = desc.inFields.front().bricks[inBrickIdx];
        inputTemp.emplace_back(tempBuffers,
                               local_comm_rank,
                               inBrick.location,
                               inBrick.layout.logical_count() * elem_size);
        inputFFTBufs.emplace_back(BufferPtr::temp(inputTemp.back().data()));
    }

    std::vector<BufferPtr> outputBufs
        = GatherUserBuffers(BufferPtr::user_output, desc.outFields.front().bricks);

    // plan FFTs along already contiguous dimensions
    std::vector<size_t> inputFFTItems;
    // first FFT needs to apply user-specified load callback
    C2CField(desc.inFields.front(),
             contiguousInputDims,
             inputBufs,
             inputFFTBufs,
             desc.loadOps,
             std::nullopt,
             {},
             inputFFTItems);

    auto lengthsWithBatch = input_lengths;
    lengthsWithBatch.push_back(desc.batch());

    // track which dimensions have already been FFTed
    std::vector<int> fft_done(desc.rank(), 0);
    for(auto d : contiguousInputDims)
        fft_done[d] = 1;

    // get processor grid from bricks for input and output
    std::array<int, 3> in_grid  = infer_grid_from_bricks(desc.inFields[0].bricks);
    std::array<int, 3> out_grid = infer_grid_from_bricks(desc.outFields[0].bricks);

    // count number of split dims in input and output grids
    const int num_split_dims_in
        = std::count_if(in_grid.begin(), in_grid.end(), [](int n) { return n > 1; });
    const int num_split_dims_out
        = std::count_if(out_grid.begin(), out_grid.end(), [](int n) { return n > 1; });

    // get transpose grids sequence for pencil and brick decompositions, from input to output
    std::vector<std::array<int, 3>> grids_sequence;
    std::vector<transpose_type>     transpose_sequence;

    bool pencil_to_pencil = false;
    // plan transposition steps
    if(num_split_dims_in >= 2 && num_split_dims_out >= 2 && desc.rank() == 3)
    {
        get_transpose_plan(in_grid,
                           out_grid,
                           {input_lengths[0], input_lengths[1], input_lengths[2]},
                           grids_sequence,
                           transpose_sequence);

        pencil_to_pencil = std::all_of(
            transpose_sequence.begin(), transpose_sequence.end(), [](transpose_type t) {
                return t == std::make_pair(grid_layout::pencil, grid_layout::pencil);
            });
    }

    rocfft_field_t         currentField       = desc.inFields.front();
    std::vector<BufferPtr> currentBufs        = inputFFTBufs;
    std::vector<size_t>    currentAntecedents = inputFFTItems;

    // using MPI sub-communicators for optimized pencil-to-pencil
    if(pencil_to_pencil)
    {
        // This vector holds leases from an earlier iteration of the
        // loop below, and is cleared when we are sure the leases can
        // be reused.
        std::vector<TempBufferLease> prevTempLeases;

        // plan global transposes and local FFTs
        for(size_t i = 0; i < transpose_sequence.size(); ++i)
        {
            // get next grid, note that transpose_sequence size is one less than grids_sequence
            std::array<int, 3> grid = grids_sequence[i + 1];

            // find pencil_axis (where grid==1), and split axes (where grid > 1)
            int              pencil_axis;
            std::vector<int> split_axes;
            std::vector<int> split_sizes;
            for(size_t d = 0; d < grid.size(); ++d)
            {
                if(grid[d] == 1)
                    pencil_axis = d;
                else
                {
                    split_axes.push_back(d);
                    split_sizes.push_back(grid[d]);
                }
            }

            // if this axis is already done, move to the next one
            if(fft_done[pencil_axis])
                continue;

            // create the next field by splitting using a heuristic approach
            rocfft_field_t nextField;
            bool           writeToUserOutput = i == transpose_sequence.size() - 1;
            if(writeToUserOutput)
                nextField = desc.outFields.front();
            else
                nextField = MakeFieldWithPencilSplit(
                    currentField, lengthsWithBatch, split_axes, split_sizes);

            // allocate temp buffers for nextField, though we only
            // need to do this if we're in the middle of the
            // transpose sequence (last one will write to output, not
            // temp)
            std::vector<TempBufferLease> tempLeases;
            std::vector<BufferPtr>       tempBufs(nextField.bricks.size());

            if(!writeToUserOutput)
            {
                for(size_t b = 0; b < nextField.bricks.size(); ++b)
                {
                    // Allocate a buffer only if this global rank owns the brick.
                    if(nextField.bricks[b].location.comm_rank == local_comm_rank)
                    {
                        tempLeases.emplace_back(tempBuffers,
                                                local_comm_rank,
                                                nextField.bricks[b].location,
                                                nextField.bricks[b].layout.logical_count()
                                                    * elem_size);
                        tempBufs[b] = BufferPtr::temp(tempLeases.back().data());
                    }
                    else
                    {
                        tempBufs[b] = BufferPtr();
                    }
                }
            }

            // plan transpose from currentField to nextField
            std::vector<size_t> transposeItems;
            GlobalTranspose(elem_size,
                            currentField,
                            nextField,
                            currentBufs,
                            writeToUserOutput ? outputBufs : tempBufs,
                            currentAntecedents,
                            transposeItems,
                            transposeNumber++);

            currentField       = nextField;
            currentBufs        = tempBufs;
            currentAntecedents = transposeItems;

            // leases allocated in this iteration need to live
            // through next loop iteration, since they will be passed
            // as input to the next GlobalTranspose.
            prevTempLeases.swap(tempLeases);

            // once data is transposed, plan intermediate FFT

            // user output needs to apply store operations
            std::vector<size_t> fftItems;
            C2CField(currentField,
                     {static_cast<size_t>(pencil_axis)},
                     writeToUserOutput ? outputBufs : currentBufs,
                     writeToUserOutput ? outputBufs : currentBufs,
                     std::nullopt,
                     writeToUserOutput ? std::optional<StoreOps>{desc.storeOps} : std::nullopt,
                     currentAntecedents,
                     fftItems);
            fft_done[pencil_axis] = 1;
            currentAntecedents    = fftItems;
        }
    }
    // default general decomposition without sub-communicators
    else
    {
        // transpose non-contiguous dims to be contiguous and
        // transform them too
        std::vector<BufferPtr>       transposeInputBufs = inputFFTBufs;
        std::vector<TempBufferLease> transposeOutputTemp;
        std::vector<BufferPtr>       transposeOutputBufs;
        auto                         transposeInputAntecedents = inputFFTItems;
        std::vector<size_t>          midFFTItems               = inputFFTItems;
        rocfft_field_t               transposedField;

        for(auto dimIdx : nonContiguousDims)
        {
            // transpose so this dim is contiguous
            transposedField
                = MakeFieldDimContiguous(desc.inFields.front(), lengthsWithBatch, dimIdx);

            // allocate bricks to store the transposed data
            for(auto& b : transposedField.bricks)
            {
                transposeOutputTemp.emplace_back(
                    tempBuffers, local_comm_rank, b.location, b.layout.logical_count() * elem_size);
                transposeOutputBufs.emplace_back(
                    BufferPtr::temp(transposeOutputTemp.back().data()));
            }

            std::vector<size_t> transposeItems;
            GlobalTranspose(elem_size,
                            desc.inFields.front(),
                            transposedField,
                            transposeInputBufs,
                            transposeOutputBufs,
                            transposeInputAntecedents,
                            transposeItems,
                            transposeNumber++);

            // now dimIdx dimension is contiguous on all bricks
            midFFTItems.clear();
            // first transform needs to apply load operations
            const std::optional<LoadOps> loadOps = dimIdx == nonContiguousDims.front()
                                                       ? std::optional<LoadOps>{desc.loadOps}
                                                       : std::nullopt;
            C2CField(transposedField,
                     {dimIdx},
                     transposeOutputBufs,
                     transposeOutputBufs,
                     loadOps,
                     std::nullopt,
                     transposeItems,
                     midFFTItems);

            // next iteration of loop will depend on these fft items and
            // work on the output we just produced
            transposeInputAntecedents = midFFTItems;
            transposeInputBufs        = transposeOutputBufs;
            std::swap(transposeOutputTemp, inputTemp);
            transposeOutputTemp.clear();
            transposeOutputBufs.clear();
        }

        // transpose data to output layout and transform along remaining dimensions
        std::vector<size_t> finalTransposeItems;
        std::vector<size_t> finalFFTItems;
        GlobalTranspose(elem_size,
                        transposedField.bricks.empty() ? desc.inFields.front() : transposedField,
                        desc.outFields.front(),
                        transposeInputBufs,
                        outputBufs,
                        midFFTItems,
                        finalTransposeItems,
                        transposeNumber++);
        // apply store operations to last dimension
        C2CField(desc.outFields.front(),
                 contiguousOutputDims,
                 outputBufs,
                 outputBufs,
                 std::nullopt,
                 desc.storeOps,
                 finalTransposeItems,
                 finalFFTItems);
    }

    return true;
}

// All-gather all of the brick parameters for a given field.
#ifdef ROCFFT_MPI_ENABLE
rocfft_status
    rocfft_plan_description_t::allgather_brick_params_lus_mpi(rocfft_field_t& field,
                                                              const size_t    global_brick_length)
{
    if(comm_type == rocfft_comm_none)
        return rocfft_status_success;
    if(!mpi_comm)
    {
        // If the comm type is MPI, but the comm pointer is null, then return an error.
        return rocfft_status_failure;
    }

    auto rcmpi = MPI_SUCCESS;

    // TODO: make async.

    // First, compute the number of bricks per field.
    std::vector<int>                                 global_brick_count(get_local_comm_size());
    typeof(decltype(global_brick_count)::value_type) local_brick_count = field.bricks.size();
    {
        // OpenMPI has a runtime error if this is const, so make sure it isn't.
        static_assert(!std::is_const_v<typeof(local_brick_count)>);
        rcmpi = MPI_Allgather(&local_brick_count,
                              1,
                              type_to_mpi_type<typeof(local_brick_count)>(),
                              global_brick_count.data(),
                              1,
                              type_to_mpi_type<typeof(decltype(global_brick_count)::value_type)>(),
                              mpi_comm);
        if(rcmpi != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Allgather failed: " + std::to_string(rcmpi));
        }
    }

    // We need the displacments (as an int[]) for MPI_Allgatherv:
    typeof(global_brick_count) displacements(get_local_comm_size());
    displacements[0] = 0;
    std::partial_sum(
        global_brick_count.begin(), global_brick_count.end() - 1, displacements.begin() + 1);
    const auto scalar_displacements = displacements;
    for(auto& val : displacements)
    {
        val *= global_brick_length;
    }

    // Total number of bricks on all ranks for this field:
    const auto num_global_bricks = std::accumulate(
        global_brick_count.begin(), global_brick_count.end(), static_cast<size_t>(0));

    std::vector<size_t> global_lowers(num_global_bricks * global_brick_length);
    std::vector<size_t> global_uppers(num_global_bricks * global_brick_length);
    std::vector<size_t> global_strides(num_global_bricks * global_brick_length);
    std::vector<decltype(rocfft_location_t::device)>    global_devices(num_global_bricks);
    std::vector<decltype(rocfft_location_t::comm_rank)> global_comm_ranks(num_global_bricks);

    // Make a contiguous copy for the MPI routine:
    decltype(global_lowers)     local_lowers;
    decltype(global_uppers)     local_uppers;
    decltype(global_strides)    local_strides;
    decltype(global_devices)    local_devices;
    decltype(global_comm_ranks) local_comm_ranks;
    local_lowers.reserve(local_brick_count * global_brick_length);
    local_uppers.reserve(local_brick_count * global_brick_length);
    local_strides.reserve(local_brick_count * global_brick_length);
    for(const auto& brick : field.bricks)
    {
        static_assert(std::is_same_v<decltype(brick.layout.lower())::value_type,
                                     decltype(local_lowers)::value_type>);
        static_assert(std::is_same_v<decltype(brick.layout.upper())::value_type,
                                     decltype(local_uppers)::value_type>);
        static_assert(std::is_same_v<decltype(brick.layout.strides_and_distances())::value_type,
                                     decltype(local_strides)::value_type>);
        for(auto v : brick.layout.lower())
        {
            local_lowers.push_back(v);
        }
        for(auto v : brick.layout.upper())
        {
            local_uppers.push_back(v);
        }
        for(auto v : brick.layout.strides_and_distances())
        {
            local_strides.push_back(v);
        }
        local_devices.push_back(brick.location.device);
        local_comm_ranks.push_back(brick.location.comm_rank);
    }

    auto recvcount = global_brick_count;
    for(auto& val : recvcount)
    {
        val *= global_brick_length;
    }

    // The receive count must be an integer type:
    static_assert(std::is_same_v<std::underlying_type<typeof(decltype(recvcount)::value_type)>,
                                 std::underlying_type<int>>);

    // We must send and receive the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_lowers)::value_type)>,
                       std::underlying_type<typeof(decltype(global_lowers)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_lowers.data(),
                           recvcount[get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_lowers)::value_type)>(),
                           global_lowers.data(),
                           recvcount.data(),
                           displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_lowers)::value_type)>(),
                           mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    // We must send and receive the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_uppers)::value_type)>,
                       std::underlying_type<typeof(decltype(global_uppers)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_uppers.data(),
                           recvcount[get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_uppers)::value_type)>(),
                           global_uppers.data(),
                           recvcount.data(),
                           displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_uppers)::value_type)>(),
                           mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    // We must send and receive the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_strides)::value_type)>,
                       std::underlying_type<typeof(decltype(global_strides)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_strides.data(),
                           recvcount[get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_strides)::value_type)>(),
                           global_strides.data(),
                           recvcount.data(),
                           displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_strides)::value_type)>(),
                           mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    // We must send and receive the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_devices)::value_type)>,
                       std::underlying_type<typeof(decltype(global_devices)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_devices.data(),
                           global_brick_count[get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_devices)::value_type)>(),
                           global_devices.data(),
                           global_brick_count.data(),
                           scalar_displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_devices)::value_type)>(),
                           mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    // We must send and receive the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_comm_ranks)::value_type)>,
                       std::underlying_type<typeof(decltype(global_comm_ranks)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_comm_ranks.data(),
                           global_brick_count[get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_comm_ranks)::value_type)>(),
                           global_comm_ranks.data(),
                           global_brick_count.data(),
                           scalar_displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_comm_ranks)::value_type)>(),
                           mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    field.bricks.clear();
    field.bricks.reserve(num_global_bricks);
    std::vector<size_t> brick_lower(global_brick_length);
    std::vector<size_t> brick_upper(global_brick_length);
    std::vector<size_t> brick_strides(global_brick_length);
    for(std::remove_const_t<decltype(num_global_bricks)> ibrick = 0; ibrick < num_global_bricks;
        ++ibrick)
    {
        brick_lower.assign(global_lowers.data() + ibrick * global_brick_length,
                           global_lowers.data() + (ibrick + 1) * global_brick_length);
        brick_upper.assign(global_uppers.data() + ibrick * global_brick_length,
                           global_uppers.data() + (ibrick + 1) * global_brick_length);
        brick_strides.assign(global_strides.data() + ibrick * global_brick_length,
                             global_strides.data() + (ibrick + 1) * global_brick_length);
        // Add bricks one by one.
        rocfft_brick_t brick{brick_lower,
                             brick_upper,
                             brick_strides,
                             {global_comm_ranks[ibrick], global_devices[ibrick]}};
        rocfft_field_add_brick_internal(field, brick);
    }

    return rocfft_status_success;
}
#endif

#if defined ROCFFT_MPI_ENABLE
// Helper function to check that all of the values of a scalar are identical on all MPI ranks.
// Relies on MPI_MAX and MPI_MIN.
template <typename Tval>
bool all_mpi_ranks_same_scalar(const Tval val, MPI_Comm comm)
{
    // NB: these can be made async if we note that plan generation performance needs improvement.

    Tval valmax;
    auto rcmpi = MPI_Allreduce(&val, &valmax, 1, type_to_mpi_type<Tval>(), MPI_MAX, comm);
    if(rcmpi != MPI_SUCCESS)
        throw std::runtime_error("MPI_Allreduce failed in all_mpi_ranks_same_scalar");

    Tval valmin;
    rcmpi = MPI_Allreduce(&val, &valmin, 1, type_to_mpi_type<Tval>(), MPI_MIN, comm);
    if(rcmpi != MPI_SUCCESS)
        throw std::runtime_error("MPI_Allreduce failed in all_mpi_ranks_same_scalar");

    return valmax == valmin;
}
#endif

#ifdef ROCFFT_MPI_ENABLE
rocfft_status rocfft_plan_description_t::allgather_brick_params_mpi()
{
    if(comm_type == rocfft_comm_none)
        return rocfft_status_success;
    if(!mpi_comm)
    {
        // If the comm type is MPI, but the comm pointer is null, then return an error.
        return rocfft_status_failure;
    }
    auto rcmpi = MPI_SUCCESS;

    // Associate the bricks with the local communicator rank, and while we're at it, check that
    // all of the bricks have the same number of lengths.

    // TODO: add tests for various brick valid/invalid combinations.

    // Get the local brick array length.  If there are no bricks, this is zero.
    size_t local_bricklength = 0;
    for(auto& ifield : inFields)
    {
        for(auto& ibrick : ifield.bricks)
        {
            ibrick.location.comm_rank = get_local_comm_rank();
            if(local_bricklength == 0)
            {
                local_bricklength = ibrick.layout.get_full_rank();
            }
            else
            {
                if(local_bricklength != ibrick.layout.get_full_rank())
                {
                    return rocfft_status_failure;
                }
            }
        }
    }
    for(auto& ofield : outFields)
    {
        for(auto& obrick : ofield.bricks)
        {
            obrick.location.comm_rank = get_local_comm_rank();
            if(local_bricklength == 0)
            {
                local_bricklength = obrick.layout.get_full_rank();
            }
            else
            {
                if(local_bricklength != obrick.layout.get_full_rank())
                {
                    return rocfft_status_failure;
                }
            }
        }
    }
    // Communicate bricks between ranks.

    // Step 1: get the local brick dimension and then verify that this is the same on all ranks.
    // If a rank has a brick dimension of zero, that means that this rank didn't provide bricks,
    // which is ok.  However, it should not build its house out of straw.
    size_t global_brick_length = 0;
    {
        std::vector<decltype(local_bricklength)> all_brick_lengths(get_local_comm_size());
        rcmpi = MPI_Allgather(&local_bricklength,
                              1,
                              type_to_mpi_type<typeof(local_bricklength)>(),
                              all_brick_lengths.data(),
                              1,
                              type_to_mpi_type<typeof(decltype(all_brick_lengths)::value_type)>(),
                              mpi_comm);
        if(rcmpi != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Allgather failed: " + std::to_string(rcmpi));
        }

        // Find first non-zero element:
        auto offset
            = distance(all_brick_lengths.begin(),
                       find_if(all_brick_lengths.begin(), all_brick_lengths.end(), [](auto x) {
                           return x != 0;
                       }));
        global_brick_length = all_brick_lengths[offset];
        if(!std::all_of(all_brick_lengths.begin() + offset,
                        all_brick_lengths.end(),
                        [global_brick_length](decltype(all_brick_lengths)::value_type i) {
                            return i == 0 || i == global_brick_length;
                        }))
        {
            // There are different non-zero brick lengths.
            return rocfft_status_failure;
        }
    }

    // Verify that all ranks have the same number of input and output fields.
    try
    {
        if(!all_mpi_ranks_same_scalar(inFields.size(), mpi_comm))
            return rocfft_status_failure;
        if(!all_mpi_ranks_same_scalar(outFields.size(), mpi_comm))
            return rocfft_status_failure;
    }
    catch(std::exception& e)
    {
        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS()) << e.what() << std::endl;
        }
        return rocfft_status_failure;
    }

    // Communicate the brick info so that all ranks know about all the bricks.
    for(auto& field : inFields)
    {
        const auto rcfft = allgather_brick_params_lus_mpi(field, global_brick_length);
        if(rcfft != rocfft_status_success)
        {
            return rcfft;
        }
    }
    for(auto& field : outFields)
    {
        const auto rcfft = allgather_brick_params_lus_mpi(field, global_brick_length);
        if(rcfft != rocfft_status_success)
        {
            return rcfft;
        }
    }

    // TODO: if some ranks do not have any input (or output), then we should make a
    // sub-communicator.  Need to figure out how to deal with completion though.

    return rocfft_status_success;
}
#endif

int rocfft_plan_description_t::get_local_comm_rank() const
{
#ifdef ROCFFT_MPI_ENABLE
    if(comm_type == rocfft_comm_none || !mpi_comm)
        return 0;

    int  mpi_rank = 0;
    auto rcmpi    = MPI_Comm_rank(mpi_comm, &mpi_rank);
    if(rcmpi != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed: " + std::to_string(rcmpi));
    return mpi_rank;
#else
    return 0;
#endif
}

int rocfft_plan_description_t::get_local_comm_size() const
{
#ifdef ROCFFT_MPI_ENABLE
    if(comm_type == rocfft_comm_none || !mpi_comm)
        return 1;

    int  mpi_size = 0;
    auto rcmpi    = MPI_Comm_size(mpi_comm, &mpi_size);
    if(rcmpi != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed: " + std::to_string(rcmpi));
    return mpi_size;
#else
    return 1;
#endif
}

rocfft_status rocfft_plan_create_internal(rocfft_plan                   plan,
                                          const rocfft_result_placement placement,
                                          const rocfft_transform_type   transform_type,
                                          const rocfft_precision        precision,
                                          const size_t                  dimensions,
                                          const size_t*                 lengths,
                                          const size_t                  number_of_transforms,
                                          const rocfft_plan_description description)
{
#ifdef ROCFFT_MPI_ENABLE
// TODO allgather for placement, transform_type, precision, dimensions, {lengths}, and
// number_of_transforms and validate that all ranks' args are identical, before proceeding
#endif
    if(dimensions > 3)
        return rocfft_status_invalid_dimensions;
    try
    {
        plan->placement     = placement;
        plan->precision     = precision;
        plan->transformType = transform_type;

        if(description)
        {
            plan->desc = *description;
        }
        const auto rcfft = plan->desc.finalize_and_validate_for(
            transform_type, placement, lengths, dimensions, number_of_transforms);
        if(rcfft != rocfft_status_success)
            return rcfft;

        log_bench(rocfft_bench_command(plan));

        // Construct the plan
        if(plan->desc.has_undistributed_io_on_current_location()
           || !plan->BuildOptMultiDevicePlan())
        {
            // Plan is configured for single-device operations or the multi-device plan
            // creation failed. Either way, a single-device execution is used, possibly
            // with gather/scatter steps.
            plan->MakeSingleDevPlanWithGatherScatterIfNeeded();
        }

        plan->AllocateInternalTempBuffers();
        return rocfft_status_success;
    }
    catch(std::exception& e)
    {
        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS()) << e.what() << std::endl;
        }
        return rocfft_status_failure;
    }
}

rocfft_status rocfft_plan_allocate(rocfft_plan* plan)
{
    *plan = new rocfft_plan_t;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_create(rocfft_plan*                  plan,
                                 const rocfft_result_placement placement,
                                 const rocfft_transform_type   transform_type,
                                 const rocfft_precision        precision,
                                 const size_t                  dimensions,
                                 const size_t*                 lengths,
                                 const size_t                  number_of_transforms,
                                 const rocfft_plan_description description)
try
{
    rocfft_plan_allocate(plan);

    size_t log_len[3] = {1, 1, 1};
    if(dimensions > 0)
        log_len[0] = lengths[0];
    if(dimensions > 1)
        log_len[1] = lengths[1];
    if(dimensions > 2)
        log_len[2] = lengths[2];

    log_trace(__func__,
              "plan",
              *plan,
              "placement",
              placement,
              "transform_type",
              transform_type,
              "precision",
              precision,
              "dimensions",
              dimensions,
              "lengths",
              std::make_pair(lengths, dimensions),
              "number_of_transforms",
              number_of_transforms,
              "description",
              description);

    return rocfft_plan_create_internal(*plan,
                                       placement,
                                       transform_type,
                                       precision,
                                       dimensions,
                                       lengths,
                                       number_of_transforms,
                                       description);
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_plan_destroy(rocfft_plan plan)
try
{
    log_trace(__func__, "plan", plan);
    delete plan;
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan, size_t* size_in_bytes)
try
{
    if(!plan)
        return rocfft_status_failure;

    if(!size_in_bytes)
        return rocfft_status_invalid_arg_value;

    auto sizes_per_device = plan->WorkBufBytesPerDevice();
    int  currentDevice    = hipInvalidDeviceId;
    if(hipGetDevice(&currentDevice) != hipSuccess)
        return rocfft_status_failure;
    *size_in_bytes = sizes_per_device[currentDevice];
    log_trace(__func__, "plan", plan, "size_in_bytes ptr", size_in_bytes, "val", *size_in_bytes);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_plan_get_print(const rocfft_plan plan)
try
{
    log_trace(__func__, "plan", plan);
    rocfft_cout << std::endl;
    rocfft_cout << "precision: " << precision_name(plan->precision) << std::endl;

    rocfft_cout << "transform type: ";
    switch(plan->transformType)
    {
    case rocfft_transform_type_complex_forward:
        rocfft_cout << "complex forward";
        break;
    case rocfft_transform_type_complex_inverse:
        rocfft_cout << "complex inverse";
        break;
    case rocfft_transform_type_real_forward:
        rocfft_cout << "real forward";
        break;
    case rocfft_transform_type_real_inverse:
        rocfft_cout << "real inverse";
        break;
    }
    rocfft_cout << std::endl;

    rocfft_cout << "result placement: ";
    switch(plan->placement)
    {
    case rocfft_placement_inplace:
        rocfft_cout << "in-place";
        break;
    case rocfft_placement_notinplace:
        rocfft_cout << "not in-place";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input array type: ";
    switch(plan->desc.inArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        rocfft_cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        rocfft_cout << "complex planar";
        break;
    case rocfft_array_type_real:
        rocfft_cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        rocfft_cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        rocfft_cout << "hermitian planar";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;

    rocfft_cout << "output array type: ";
    switch(plan->desc.outArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        rocfft_cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        rocfft_cout << "comple planar";
        break;
    case rocfft_array_type_real:
        rocfft_cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        rocfft_cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        rocfft_cout << "hermitian planar";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "dimensions: " << plan->desc.rank() << std::endl;

    const auto lengths = plan->get_user_facing_lengths();
    rocfft_cout << "lengths: " << lengths[0];
    for(size_t i = 1; i < plan->desc.rank(); i++)
        rocfft_cout << ", " << lengths[i];
    rocfft_cout << std::endl;
    rocfft_cout << "batch size: " << plan->desc.batch() << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input offset: " << plan->desc.inOffset[0];
    if((plan->desc.inArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.inArrayType == rocfft_array_type_hermitian_planar))
        rocfft_cout << ", " << plan->desc.inOffset[1];
    rocfft_cout << std::endl;

    rocfft_cout << "output offset: " << plan->desc.outOffset[0];
    if((plan->desc.outArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.outArrayType == rocfft_array_type_hermitian_planar))
        rocfft_cout << ", " << plan->desc.outOffset[1];
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    const auto input_strides = plan->desc.input_layout.strides();
    if(!input_strides.empty())
        rocfft_cout << "input strides: " << input_strides[0];
    for(size_t i = 1; i < input_strides.size(); i++)
        rocfft_cout << ", " << input_strides[i];
    rocfft_cout << std::endl;

    const auto output_strides = plan->desc.output_layout.strides();
    if(!output_strides.empty())
        rocfft_cout << "output strides: " << output_strides[0];
    for(size_t i = 1; i < output_strides.size(); i++)
        rocfft_cout << ", " << output_strides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "input distance: " << plan->desc.input_layout.distance() << std::endl;
    rocfft_cout << "output distance: " << plan->desc.output_layout.distance() << std::endl;
    rocfft_cout << std::endl;

    plan->desc.loadOps.print(rocfft_cout, {});
    plan->desc.storeOps.print(rocfft_cout, {});
    rocfft_cout << std::endl;

    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

ROCFFT_EXPORT rocfft_status rocfft_get_version_string(char* buf, const size_t len)
try
{
    log_trace(__func__, "buf", static_cast<void*>(buf), "len", len);
    // include nul terminator
    const auto version_len = std::strlen(ROCFFT_VERSION_STRING) + 1;
    if(!buf)
        return rocfft_status_failure;
    if(len < version_len)
        return rocfft_status_invalid_arg_value;
    std::memcpy(buf, ROCFFT_VERSION_STRING, version_len);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

// Compute the large twd decomposition base
// 2-Steps:
//  e.g., ( CeilPo2(10000)+ 1 ) / 2 , returns 7 : (2^7)*(2^7) = 16384 >= 10000
// 3-Steps:
//  e.g., ( CeilPo2(10000)+ 2 ) / 3 , returns 5 : (2^5)*(2^5)*(2^5) = 32768 >= 10000
void get_large_twd_base_steps(size_t large1DLen, bool use3steps, size_t& base, size_t& steps)
{
    // use3steps, then 16^3 ~ 64^3, basically enough for 262144
    // else, base is 8 (2^8 = 256), could be 2-steps 256^2 = 65536, if exceed, then is 256^3, and so on..
    base = use3steps ? std::min((size_t)6, std::max((size_t)4, (CeilPo2(large1DLen) + 2) / 3)) : 8;

    // but we still want to know the exact steps we will loop
    steps                  = 0;
    size_t lenLargeTwdBase = pow(2, base);
    while(pow(lenLargeTwdBase, steps) < large1DLen)
        steps++;

    if(base == 8 && steps > 3)
        throw std::runtime_error(
            "large-twd-base 8 could be 2,3 steps, but not supported for 4-steps yet");
    if(base < 8 && steps != 3)
        throw std::runtime_error("large-twd-base for 4,5,6 must be 3-steps");
}

bool BufferIsUnitStride(ExecPlan& execPlan, OperatingBuffer buf)
{
    // temp buffers are unit stride
    if(buf != OB_USER_IN && buf != OB_USER_OUT)
        return true;

    if(execPlan.isUnitStride.find(buf) != execPlan.isUnitStride.end())
        return execPlan.isUnitStride.at(buf);

    auto stride = (buf == OB_USER_IN) ? execPlan.rootPlan->inStride : execPlan.rootPlan->outStride;
    auto length = (buf == OB_USER_IN) ? execPlan.iLength : execPlan.oLength;
    auto dist   = (buf == OB_USER_IN) ? execPlan.rootPlan->iDist : execPlan.rootPlan->oDist;
    size_t curStride = 1;
    do
    {
        if(stride.front() != curStride)
            return false;
        curStride *= length.front();
        stride.erase(stride.begin());
        length.erase(length.begin());
    } while(!stride.empty());

    // NB: users may input incorrect i/o-dist value for inplace transform
    //     however, when the batch-size is 1, we can simply make it permissive
    //     since the dist is not used in single batch. But note that we still need
    //     to pass the above do-while to ensure all the previous strides are valid.
    bool result = (execPlan.rootPlan->batch == 1) || (curStride == dist);

    execPlan.isUnitStride[buf] = result;
    return result;
}

void TreeNode::CopyNodeData(const TreeNode& srcNode)
{
    dimension = srcNode.dimension;
    batch     = srcNode.batch;
    length    = srcNode.length;
    if(!srcNode.outputLength.empty())
        outputLength = srcNode.outputLength;
    inStride        = srcNode.inStride;
    inStrideBlue    = srcNode.inStrideBlue;
    outStride       = srcNode.outStride;
    outStrideBlue   = srcNode.outStrideBlue;
    iDist           = srcNode.iDist;
    iDistBlue       = srcNode.iDistBlue;
    oDist           = srcNode.oDist;
    oDistBlue       = srcNode.oDistBlue;
    iOffset         = srcNode.iOffset;
    oOffset         = srcNode.oOffset;
    placement       = srcNode.placement;
    precision       = srcNode.precision;
    direction       = srcNode.direction;
    inArrayType     = srcNode.inArrayType;
    outArrayType    = srcNode.outArrayType;
    allowInplace    = srcNode.allowInplace;
    allowOutofplace = srcNode.allowOutofplace;

    // conditional
    large1D        = srcNode.large1D;
    largeTwd3Steps = srcNode.largeTwd3Steps;
    largeTwdBase   = srcNode.largeTwdBase;
    lengthBlue     = srcNode.lengthBlue;
    lengthBlueN    = srcNode.lengthBlueN;
    typeBlue       = srcNode.typeBlue;
    fuseBlue       = srcNode.fuseBlue;

    //
    obIn  = srcNode.obIn;
    obOut = srcNode.obOut;

    // NB:
    //   we don't copy this since it's possible we're copying
    //   a node to another one that is different scheme/derived class
    //   (for example, when doing fusion).
    //   The src ebtype could be incorrect in the new node
    //   so we don't copy this value, the target node already sets its value
    // ebtype      = srcNode.ebtype;
}

void TreeNode::CopyNodeData(const NodeMetaData& data)
{
    dimension = data.dimension;
    batch     = data.batch;
    length    = data.length;
    if(!data.outputLength.empty())
        outputLength = data.outputLength;
    inStride      = data.inStride;
    inStrideBlue  = data.inStrideBlue;
    outStride     = data.outStride;
    outStrideBlue = data.outStrideBlue;
    iDist         = data.iDist;
    iDistBlue     = data.iDistBlue;
    oDist         = data.oDist;
    oDistBlue     = data.oDistBlue;
    iOffset       = data.iOffset;
    oOffset       = data.oOffset;
    placement     = data.placement;
    precision     = data.precision;
    direction     = data.direction;
    inArrayType   = data.inArrayType;
    outArrayType  = data.outArrayType;
}

bool TreeNode::isPlacementAllowed(rocfft_result_placement test_placement) const
{
    return (test_placement == rocfft_placement_inplace) ? allowInplace : allowOutofplace;
}

bool TreeNode::isOutBufAllowed(OperatingBuffer oB) const
{
    return (oB & allowedOutBuf) != 0;
}

bool TreeNode::isOutArrayTypeAllowed(rocfft_array_type oArrayType) const
{
    return allowedOutArrayTypes.count(oArrayType) > 0;
}

bool TreeNode::isRootNode() const
{
    return parent == nullptr;
}

bool TreeNode::isLeafNode() const
{
    return nodeType == NT_LEAF;
}

LeafNode& TreeNode::getLeafNode()
{
    return dynamic_cast<LeafNode&>(*this);
}

const LeafNode& TreeNode::getLeafNode() const
{
    return dynamic_cast<const LeafNode&>(*this);
}

// Tree node builders

// NB:
// Don't assign inArrayType and outArrayType when building any tree node.
// That should be done in buffer assignment stage or
// TraverseTreeAssignPlacementsLogicA().

void TreeNode::RecursiveBuildTree(SchemeTree* solution_scheme)
{
    // Some-Common-Work...
    // We must follow the placement of RootPlan, so needs to make it explicit
    if(isRootNode())
    {
        allowInplace    = (placement == rocfft_placement_inplace);
        allowOutofplace = !allowInplace;
    }

    SchemeTreeVec& child_scheme
        = (solution_scheme) ? solution_scheme->children : EmptySchemeTreeVec;

    // overridden by each derived class
    BuildTree_internal(child_scheme);
}

void TreeNode::SanityCheck(SchemeTree* solution_scheme, std::vector<FMKey>& kernel_keys)
{
    // no un-defined node is allowed in the tree
    if(nodeType == NT_UNDEFINED)
        throw std::runtime_error("NT_UNDEFINED node");

    // Check buffer: all operating buffers have been assigned
    if(obIn == OB_UNINIT)
        throw std::runtime_error("obIn un-init");
    if(obOut == OB_UNINIT)
        throw std::runtime_error("obOut un-init");
    if((obIn == obOut) && (placement != rocfft_placement_inplace))
        throw std::runtime_error("[obIn,obOut] mismatch placement inplace");
    if((obIn != obOut) && (placement != rocfft_placement_notinplace))
        throw std::runtime_error("[obIn,obOut] mismatch placement out-of-place");

    // Check length and stride and dimension:
    if(length.size() != inStride.size())
        throw std::runtime_error("length.size() mismatch inStride.size()");
    if(length.size() != outStride.size())
        throw std::runtime_error("length.size() mismatch outStride.size()");
    if(length.size() < dimension)
        throw std::runtime_error("not enough length[] for dimension");

    // make sure the tree has the same decomposition way as in solution map
    if(solution_scheme)
    {
        if(childNodes.size() != solution_scheme->children.size())
            throw std::runtime_error("scheme-decomposition error: plan-tree != scheme-tree");
        if(scheme != solution_scheme->curScheme)
            throw std::runtime_error("scheme-decomposition error: node-scheme != solution-scheme");
    }

    OperatingBuffer previousOut = obIn;
    for(size_t id = 0; id < childNodes.size(); ++id)
    {
        auto&       child = childNodes[id];
        SchemeTree* child_scheme
            = (solution_scheme) ? solution_scheme->children[id].get() : nullptr;

        // 1. Recursively check child
        child->SanityCheck(child_scheme, kernel_keys);

        // 2. Assert that the kernel chain is connected
        // Note: The Bluestein algorithm uses setup nodes that aren't
        // connected in the chain.

        if(child->IsBluesteinChirpSetup())
            continue;
        if(child->obIn != previousOut)
            throw std::runtime_error("Sanity Check failed: " + PrintScheme(child->scheme)
                                     + " input " + PrintOperatingBuffer(child->obIn)
                                     + " does not match previous output "
                                     + PrintOperatingBuffer(previousOut));
        previousOut = child->obOut;
    }
}

bool TreeNode::fuse_CS_KERNEL_TRANSPOSE_Z_XY()
{
    if(pool.has_SBRC_kernel(length[0], precision))
    {
        auto kernel = pool.get_kernel(
            FMKey(length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC, TILE_ALIGNED));
        size_t bwd = kernel.transforms_per_block;
        if((length[1] >= bwd) && (length[2] >= bwd) && (length[1] * length[2] % bwd == 0))
            return true;
    }

    return false;
}

bool TreeNode::fuse_CS_KERNEL_TRANSPOSE_XY_Z()
{
    if(pool.has_SBRC_kernel(length[0], precision))
    {
        if((length[0] == length[2]) // limit to original "cubic" case
           && (length[0] / 2 + 1 == length[1])
           && !IsPo2(length[0]) // Need more investigation for diagonal transpose
        )
            return true;
    }
    return false;
}

bool TreeNode::fuse_CS_KERNEL_STK_R2C_TRANSPOSE()
{
    if(pool.has_SBRC_kernel(length[0], precision)) // kernel available
    {
        if((length[0] * 2 == length[1]) // limit to original "cubic" case
           && (length.size() == 2 || length[1] == length[2]) // 2D or 3D
        )
            return true;
    }
    return false;
}

void TreeNode::ApplyFusion()
{
    // Do the final fusion after the buffer assign is completed
    for(auto& fuse : fuseShims)
    {
        // the flag was overwritten by execPlan (according to the arch for some specical cases)
        if(!fuse->IsSchemeFusable())
            continue;

        auto fused = fuse->FuseKernels();
        if(fused)
        {
            auto firstFusedNode = fuse->FirstFuseNode();
            this->RecursiveInsertNode(firstFusedNode, fused);

            // iterate from first to last to remove old nodes
            fuse->ForEachNode([=, this](TreeNode* node) { this->RecursiveRemoveNode(node); });
        }
    }

    for(auto& child : childNodes)
        child->ApplyFusion();
}

void TreeNode::RefreshTree()
{
    if(childNodes.empty())
        return;

    for(auto& child : childNodes)
        child->RefreshTree();

    // only modify nodes that work with user data, and skip Bluestein
    // nodes that only set up the chirp buffer
    auto firstIt = std::find_if_not(
        childNodes.begin(), childNodes.end(), [](const std::unique_ptr<TreeNode>& n) {
            return n->IsBluesteinChirpSetup();
        });
    // if these children are all setup nodes, there's nothing further to refresh
    if(firstIt == childNodes.end())
        return;

    auto first = firstIt->get();
    auto last  = childNodes.back().get();

    // Skip first node in multi-kernel fused Bluestein
    // since it is not connected to the buffer chain
    if(fuseBlue != BFT_FWD_CHIRP)
    {
        this->obIn      = first->obIn;
        this->obOut     = last->obOut;
        this->placement = (obIn == obOut) ? rocfft_placement_inplace : rocfft_placement_notinplace;

        // even-length real transform nodes need to have real
        // input/output even if their first/last child treats the
        // real data as complex
        const bool isRealEvenNode = scheme == CS_REAL_TRANSFORM_EVEN || scheme == CS_REAL_2D_EVEN
                                    || scheme == CS_REAL_3D_EVEN || scheme == CS_REAL_3D_PP;
        if(isRealEvenNode && direction == -1)
            this->inArrayType = rocfft_array_type_real;
        else
            this->inArrayType = first->inArrayType;

        if(isRealEvenNode && direction == 1)
            this->outArrayType = rocfft_array_type_real;
        else
            this->outArrayType = last->outArrayType;
    }
}

void TreeNode::AssignParams()
{
    if((length.size() != inStride.size()) || (length.size() != outStride.size()))
        throw std::runtime_error("length size mismatches stride size");

    for(auto& child : childNodes)
    {
        child->inStride.clear();
        child->inStrideBlue.clear();
        child->outStride.clear();
        child->outStrideBlue.clear();
    }

    AssignParams_internal();
}

///////////////////////////////////////////////////////////////////////////////
/// Collect leaf node
void TreeNode::CollectLeaves(std::vector<TreeNode*>& seq, std::vector<FuseShim*>& fuseSeq)
{
    // re-collect after kernel fusion, so clear the previous collected elements
    if(isRootNode())
    {
        seq.clear();
        fuseSeq.clear();
    }

    if(nodeType == NT_LEAF)
    {
        seq.push_back(this);
    }
    else
    {
        for(auto& child : childNodes)
            child->CollectLeaves(seq, fuseSeq);

        for(auto& fuse : fuseShims)
            fuseSeq.push_back(fuse.get());
    }
}

// Important: Make sure the order of the fuse-shim is consistent with the execSeq
// This is essential for BackTracking in BufferAssignment
void OrderFuseShims(std::vector<TreeNode*>& seq, std::vector<FuseShim*>& fuseSeq)
{
    std::vector<FuseShim*> reordered;
    for(auto node : seq)
    {
        for(size_t fuseID = 0; fuseID < fuseSeq.size(); ++fuseID)
        {
            if(node == fuseSeq[fuseID]->FirstFuseNode())
            {
                reordered.emplace_back(fuseSeq[fuseID]);
                break;
            }
        }
    }

    if(reordered.size() != fuseSeq.size())
        throw std::runtime_error("reorder fuse shim list error");

    fuseSeq.swap(reordered);
}

void CheckFuseShimForArch(ExecPlan& execPlan)
{
    // for gfx906...
    if(is_device_gcn_arch(execPlan.deviceProp, "gfx906"))
    {
        auto& fusions = execPlan.fuseShims;
        for(auto& fusion : fusions)
        {
            if(fusion->fuseType == FT_STOCKHAM_WITH_TRANS
               && fusion->FirstFuseNode()->length[0] == 168)
            {
                fusion->OverwriteFusableFlag(false);

                // remove it from the execPlan list
                fusions.erase(std::remove(fusions.begin(), fusions.end(), fusion), fusions.end());
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Calculate work memory requirements,
/// note this should be done after buffer assignment and deciding oDist
void TreeNode::DetermineBufferMemory(size_t& tmpBufSize,
                                     size_t& cmplxForRealSize,
                                     size_t& blueSize,
                                     size_t& chirpSize)
{
    if(nodeType == NT_LEAF)
    {
        auto outputPtrDiff
            = compute_ptrdiff(UseOutputLengthForPadding() ? GetOutputLength() : length,
                              (typeBlue == BT_MULTI_KERNEL_FUSED) ? outStrideBlue : outStride,
                              batch,
                              (typeBlue == BT_MULTI_KERNEL_FUSED) ? oDistBlue : oDist);

        if(scheme == CS_KERNEL_CHIRP)
            chirpSize = std::max(lengthBlue, chirpSize);

        if(obOut == OB_TEMP_BLUESTEIN)
            blueSize = std::max(typeBlue == BT_MULTI_KERNEL_FUSED ? outputPtrDiff + lengthBlue
                                                                  : outputPtrDiff,
                                blueSize);

        if(obOut == OB_TEMP_CMPLX_FOR_REAL)
            cmplxForRealSize = std::max(outputPtrDiff, cmplxForRealSize);

        if(obOut == OB_TEMP)
            tmpBufSize = std::max(outputPtrDiff, tmpBufSize);
    }

    for(auto& child : childNodes)
        child->DetermineBufferMemory(tmpBufSize, cmplxForRealSize, blueSize, chirpSize);
}

void TreeNode::Print(rocfft_ostream& os, const int indent) const
{

    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << "\n" << indentStr << "scheme: " << PrintScheme(scheme);
    os << "\n" << indentStr << "dimension: " << dimension;
    os << "\n" << indentStr << "batch: " << batch;
    os << "\n" << indentStr << "length: ";
    for(const auto val : length)
    {
        os << " " << val;
    }
    if(!outputLength.empty() && outputLength != length)
    {
        os << "\n" << indentStr << "outputLength:";
        for(const auto val : outputLength)
        {
            os << " " << val;
        }
    }

    os << "\n" << indentStr << "iStrides: ";
    for(size_t i = 0; i < inStride.size(); i++)
        os << inStride[i] << " ";

    if(typeBlue == BT_MULTI_KERNEL_FUSED)
    {
        os << "\n" << indentStr << "iStridesBlue: ";
        for(size_t i = 0; i < inStrideBlue.size(); i++)
            os << inStrideBlue[i] << " ";
    }

    os << "\n" << indentStr << "oStrides: ";
    for(size_t i = 0; i < outStride.size(); i++)
        os << outStride[i] << " ";

    if(typeBlue == BT_MULTI_KERNEL_FUSED)
    {
        os << "\n" << indentStr << "oStridesBlue: ";
        for(size_t i = 0; i < outStrideBlue.size(); i++)
            os << outStrideBlue[i] << " ";
    }

    if(iOffset)
    {
        os << "\n" << indentStr;
        os << "iOffset: " << iOffset;
    }
    if(oOffset)
    {
        os << "\n" << indentStr;
        os << "oOffset: " << oOffset;
    }

    os << "\n" << indentStr;
    os << "iDist: " << iDist;
    if(typeBlue == BT_MULTI_KERNEL_FUSED)
    {
        os << "\n" << indentStr;
        os << "iDistBlue: " << iDistBlue;
    }
    os << "\n" << indentStr;
    os << "oDist: " << oDist;
    if(typeBlue == BT_MULTI_KERNEL_FUSED)
    {
        os << "\n" << indentStr;
        os << "oDistBlue: " << oDistBlue;
    }

    os << "\n" << indentStr;
    os << "direction: " << direction;

    os << "\n" << indentStr;
    os << "placement: " << PrintPlacement(placement);

    os << "\n" << indentStr;
    os << precision_name(precision) << "-precision";

    os << std::endl << indentStr;
    os << "array type: ";
    os << PrintArrayType(inArrayType);
    os << " -> ";
    os << PrintArrayType(outArrayType);

    if(large1D)
    {
        os << "\n" << indentStr << "large1D: " << large1D;
        os << "\n" << indentStr << "largeTwdBase: " << largeTwdBase;
        os << "\n" << indentStr << "largeTwdSteps: " << ltwdSteps;
    }
    if(twiddles)
    {
        os << "\n"
           << indentStr << "twiddle table length: " << twiddles_size / complex_type_size(precision);
    }
    if(twiddles_large)
    {
        os << "\n"
           << indentStr
           << "large twiddle table length: " << twiddles_large_size / complex_type_size(precision);
    }
    if(lengthBlue)
        os << "\n" << indentStr << "lengthBlue: " << lengthBlue;
    os << "\n";
    switch(ebtype)
    {
    case EmbeddedType::NONE:
        break;
    case EmbeddedType::C2Real_PRE:
        os << indentStr << "EmbeddedType: C2Real_PRE\n";
        break;
    case EmbeddedType::Real2C_POST:
        os << indentStr << "EmbeddedType: Real2C_POST\n";
        break;
    }

    os << indentStr << "SBRC_Trans_Type: " << PrintSBRCTransposeType(sbrcTranstype);
    os << "\n";

    switch(intrinsicMode)
    {
    case IntrinsicAccessType::DISABLE_BOTH:
        break;
    case IntrinsicAccessType::ENABLE_LOAD_ONLY:
        os << indentStr << "Intrinsic Mode: LOAD_ONLY\n";
        break;
    case IntrinsicAccessType::ENABLE_BOTH:
        os << indentStr << "Intrinsic Mode: LOAD_AND_STORE\n";
        break;
    }

    os << indentStr << "Direct_to_from_Reg: " << PrintDirectToFromRegMode(dir2regMode);
    os << "\n";
    if(loadOps)
        loadOps->print(os, indentStr);
    if(storeOps)
        storeOps->print(os, indentStr);

    os << indentStr << PrintOperatingBuffer(obIn) << " -> " << PrintOperatingBuffer(obOut) << "\n";
    os << indentStr << PrintOperatingBufferCode(obIn) << " -> " << PrintOperatingBufferCode(obOut)
       << "\n";

    for(const auto& c : comments)
    {
        os << "\n" << indentStr << "comment: " << c;
    }

    if(childNodes.size())
    {
        for(auto& children_p : childNodes)
        {
            children_p->Print(os, indent + 1);
        }
    }

    os << std::flush;
}

void TreeNode::RecursiveFindChildNodes(const ComputeScheme&    findScheme,
                                       std::vector<TreeNode*>& nodes)
{
    if(scheme == findScheme)
        nodes.emplace_back(this);

    for(auto& child : childNodes)
        child->RecursiveFindChildNodes(findScheme, nodes);
}

void TreeNode::RecursiveCopyNodeData(const TreeNode& srcNode)
{
    CopyNodeData(srcNode);

    if(childNodes.size() != srcNode.childNodes.size())
        throw std::runtime_error("Invalid copy of source tree data");

    std::size_t i = 0;
    for(auto& child : childNodes)
    {
        child->CopyNodeData(*srcNode.childNodes[i]);
        ++i;
    }
}

void TreeNode::RecursiveRemoveNode(TreeNode* node)
{
    for(auto& child : childNodes)
        child->RecursiveRemoveNode(node);
    childNodes.erase(std::remove_if(childNodes.begin(),
                                    childNodes.end(),
                                    [node](const std::unique_ptr<TreeNode>& child) {
                                        return child.get() == node;
                                    }),
                     childNodes.end());
}

void TreeNode::RecursiveInsertNode(TreeNode* pos, std::unique_ptr<TreeNode>& newNode)
{
    auto found = std::find_if(
        childNodes.begin(), childNodes.end(), [pos](const std::unique_ptr<TreeNode>& child) {
            return child.get() == pos;
        });
    if(found != childNodes.end())
    {
        childNodes.insert(found, std::move(newNode));
    }
    else
    {
        for(auto& child : childNodes)
            child->RecursiveInsertNode(pos, newNode);
    }
}

const TreeNode* TreeNode::GetPlanRoot() const
{
    if(isRootNode())
        return this;

    return parent->GetPlanRoot();
}

TreeNode* TreeNode::GetFirstLeaf()
{
    return (nodeType == NT_LEAF) ? this : childNodes.front()->GetFirstLeaf();
}

TreeNode* TreeNode::GetLastLeaf()
{
    return (nodeType == NT_LEAF) ? this : childNodes.back()->GetLastLeaf();
}

TreeNode* TreeNode::GetRealEvenAncestor()
{
    // If no ancestor, stop
    if(!parent)
        return nullptr;

    // If parent is directly an even-length plan, then that's what
    // we're looking for
    if(parent->scheme == CS_REAL_TRANSFORM_EVEN || parent->scheme == CS_REAL_2D_EVEN
       || parent->scheme == CS_REAL_3D_EVEN || parent->scheme == CS_REAL_3D_PP)
        return parent;

    // Otherwise keep looking up the tree
    return parent->GetRealEvenAncestor();
}

TreeNode* TreeNode::GetPartialPassAncestor() const
{
    if(!parent)
        return nullptr;

    if(parent->scheme == CS_3D_PP || parent->scheme == CS_REAL_3D_PP)
        return parent;

    return parent->GetPartialPassAncestor();
}

bool TreeNode::IsRootPlanC2CTransform() const
{
    auto root = GetPlanRoot();
    return (root->inArrayType != rocfft_array_type_real)
           && (root->outArrayType != rocfft_array_type_real);
}

bool TreeNode::IsRootPlanR2CTransform() const
{
    auto root = GetPlanRoot();
    return (root->inArrayType == rocfft_array_type_real)
           && (root->outArrayType != rocfft_array_type_real);
}

bool TreeNode::IsRootPlanC2RTransform() const
{
    auto root = GetPlanRoot();
    return (root->inArrayType != rocfft_array_type_real)
           && (root->outArrayType == rocfft_array_type_real);
}

rocfft_transform_type TreeNode::GetRootPlanTransformType() const
{
    const auto root = GetPlanRoot();

    if(IsRootPlanC2CTransform() && root->direction == -1)
        return rocfft_transform_type_complex_forward;
    else if(IsRootPlanC2CTransform() && root->direction == 1)
        return rocfft_transform_type_complex_inverse;
    else if(IsRootPlanR2CTransform())
        return rocfft_transform_type_real_forward;
    else if(IsRootPlanC2RTransform())
        return rocfft_transform_type_real_inverse;
    else
        throw std::runtime_error("Unknown root plan transform type");
}

// remove a leaf node from the plan completely - plan optimization
// can remove unnecessary nodes to skip unnecessary work.
void RemoveNode(ExecPlan& execPlan, TreeNode* node)
{
    auto& execSeq = execPlan.execSeq;
    // remove it from the non-owning leaf nodes
    execSeq.erase(std::remove(execSeq.begin(), execSeq.end(), node), execSeq.end());

    // remove it from the tree structure
    execPlan.rootPlan->RecursiveRemoveNode(node);
}

// insert a leaf node to the plan, bot execSeq and tree - plan optimization
void InsertNode(ExecPlan& execPlan, TreeNode* pos, std::unique_ptr<TreeNode>& newNode)
{
    auto& execSeq = execPlan.execSeq;
    // insert it to execSeq, before pos
    execSeq.insert(std::find(execSeq.begin(), execSeq.end(), pos), newNode.get());

    // insert it before pos in the tree structure
    execPlan.rootPlan->RecursiveInsertNode(pos, newNode);
}

std::pair<TreeNode*, TreeNode*> ExecPlan::get_load_store_nodes() const
{
    const auto& seq = execSeq;

    // look forward for the first node that reads from input
    auto load_it = std::find_if(
        seq.begin(), seq.end(), [&](const TreeNode* n) { return n->obIn == rootPlan->obIn; });
    TreeNode* load = load_it == seq.end() ? nullptr : *load_it;

    // look backward for the last node that writes to output
    auto store_it = std::find_if(
        seq.rbegin(), seq.rend(), [&](const TreeNode* n) { return n->obOut == rootPlan->obOut; });
    TreeNode* store = store_it == seq.rend() ? nullptr : *store_it;

    assert(load && store);
    return std::make_pair(load, store);
}

void RuntimeCompilePlan(ExecPlan& execPlan)
{
    std::string kernel_name;
    bool        is_tuning = TuningBenchmarker::GetSingleton().IsProcessingTuning();

    for(size_t i = 0; i < execPlan.execSeq.size(); ++i)
    {
        auto& node = execPlan.execSeq[i];

        // If this isn't for the local rank, don't compile.

        node->compiledKernel = RTCKernel::runtime_compile(
            node->getLeafNode(), execPlan.deviceProp.gcnArchName, kernel_name);

        // Log kernel name when tuning
        if(is_tuning)
        {
            TuningBenchmarker::GetSingleton().GetPacket()->kernel_names[i] = kernel_name;
            if(LOG_TUNING_ENABLED())
                (*LogSingleton::GetInstance().GetTuningOS())
                    << "kernel: " << kernel_name << std::endl;
        }
    }

    TreeNode* load_node             = nullptr;
    TreeNode* store_node            = nullptr;
    std::tie(load_node, store_node) = execPlan.get_load_store_nodes();

    // callbacks are only possible on plans that don't use planar format for input or output
    bool need_callbacks = !array_type_is_planar(load_node->inArrayType)
                          && !array_type_is_planar(store_node->outArrayType);

    // don't spend time compiling callback
    if(need_callbacks && !is_tuning)
    {
        load_node->compiledKernelWithCallbacks = RTCKernel::runtime_compile(
            load_node->getLeafNode(), execPlan.deviceProp.gcnArchName, kernel_name, true);

        if(store_node != load_node)
        {
            store_node->compiledKernelWithCallbacks = RTCKernel::runtime_compile(
                store_node->getLeafNode(), execPlan.deviceProp.gcnArchName, kernel_name, true);
        }
    }

    // All of the compilations are started in parallel (via futures),
    // so resolve the futures now.  That ensures that the plan is
    // ready to run as soon as the caller gets the plan back.
    for(auto& node : execPlan.execSeq)
    {
        if(node->compiledKernel.valid())
            node->compiledKernel.get();
        if(node->compiledKernelWithCallbacks.valid())
            node->compiledKernelWithCallbacks.get();
    }
}

// Input a node, get the representative prob-token as the key of solution-map
void GetNodeToken(const TreeNode& probNode, std::string& min_token, std::string& full_token)
{
    // min_token: consider only length, precision, placement, complex/real,
    //             and direction for real-trans (R2C/C2R)
    // full_token: consider batch, dist, stride, offset, direction for complex
    // When searching solution, looking for full-match first, and then min-match

    // if this is a leaf-node TRANSPOSE, call_back or others with external-kernel = false
    // currently we don't tune it, but still need to put an entry in the map. So we
    // set a pre-defined token
    if(probNode.isLeafNode() && probNode.GetKernelKey() == FMKey::EmptyFMKey())
    {
        min_token = full_token = solution_map::LEAFNODE_TOKEN_BUILTIN_KERNEL;
        return;
    }

    std::string token = ComputeSchemeIsAProblem(probNode.scheme)
                            ? ("")
                            : (PrintKernelSchemeAbbr(probNode.scheme) + "_");

    // Solutions are keyed on complex length.  So for C2R transforms,
    // use the output length to search for solutions.
    auto& probLength = (array_type_is_complex(probNode.inArrayType)
                        && probNode.outArrayType == rocfft_array_type_real)
                           ? probNode.outputLength
                           : probNode.length;
    for(size_t i = 0; i < probNode.dimension; ++i)
        token += std::to_string(probLength[i]) + "_";

    std::string precision_str;
    if(probNode.precision == rocfft_precision_single)
        precision_str = "sp_";
    else if(probNode.precision == rocfft_precision_double)
        precision_str = "dp_";
    else if(probNode.precision == rocfft_precision_half)
        precision_str = "half_";
    else
        throw std::runtime_error("tree node has invalid precision");

    token += precision_str;
    token += (probNode.placement == rocfft_placement_inplace) ? "ip_" : "op_";

    bool is_real_trans = ((probNode.inArrayType == rocfft_array_type_real)
                          || (probNode.outArrayType == rocfft_array_type_real));
    bool is_fwd        = (probNode.direction == -1);

    if(is_real_trans)
    {
        token += "real_";
        token += (is_fwd) ? "fwd" : "bwd";
        min_token = token;
    }
    else
    {
        token += "complex";
        min_token = token;
        token += (is_fwd) ? "_fwd" : "_bwd";
    }

    token += "_batch_" + std::to_string(probNode.batch);

    token += "_istride";
    for(size_t i = 0; i < probNode.inStride.size(); ++i)
        token += "_" + std::to_string(probNode.inStride[i]);

    token += "_ostride";
    for(size_t i = 0; i < probNode.outStride.size(); ++i)
        token += "_" + std::to_string(probNode.outStride[i]);

    token += "_idist_" + std::to_string(probNode.iDist);
    token += "_odist_" + std::to_string(probNode.oDist);
    token += "_ioffset_" + std::to_string(probNode.iOffset);
    token += "_ooffset_" + std::to_string(probNode.oOffset);

    full_token = token;
}

// generate all possible keys from a root problem, try them all to find a solution.
void GenerateProbKeys(const TreeNode& probNode, std::vector<ProblemKey>& possibleKeys)
{
    possibleKeys.clear();

    std::string min_token;
    std::string full_token;
    std::string archName = get_arch_name(probNode.deviceProp);
    GetNodeToken(probNode, min_token, full_token);

    for(auto arch : {archName, std::string("any")})
    {
        for(auto prob_token : {full_token, min_token})
        {
            ProblemKey problemKey(arch, prob_token);
            possibleKeys.push_back(problemKey);
        }
    }
}

// recursively apply the solutions (breadth-first)
// return: A pointer of a sub-scheme-tree
// If solution is a kernel, append the kernel_key to the output vector
std::unique_ptr<SchemeTree>
    RecursivelyApplySol(const ProblemKey& problemKey, ExecPlan& execPlan, size_t sol_option)
{
    auto& sol_map_single = solution_map::get_solution_map();
    if(!sol_map_single.has_solution_node(problemKey, sol_option))
        return nullptr;

    std::string  arch     = problemKey.arch;
    SolutionNode sol_node = sol_map_single.get_solution_node(problemKey, sol_option);

    // it is a dummy solution.
    if(sol_node.using_scheme == CS_NONE)
    {
        if(LOG_TRACE_ENABLED())
            (*LogSingleton::GetInstance().GetTraceOS())
                << "found a dummy root-solution(" << arch << ", " << problemKey.probToken << ")"
                << std::endl;
        return nullptr;
    }

    std::unique_ptr<SchemeTree> curScheme
        = std::make_unique<SchemeTree>(SchemeTree(sol_node.using_scheme));

    if(sol_node.sol_node_type == SOL_INTERNAL_NODE)
    {
        if(sol_node.solution_childnodes.empty())
            return nullptr;

        // we stick to the current arch same as the root's problemkey
        // e.g even we are in gfx908, but if the found root solution is in "any" map,
        // then we should keep looking-up the "any" map
        for(auto& child_node : sol_node.solution_childnodes)
        {
            ProblemKey probKey(arch, child_node.child_token);
            auto childScheme = RecursivelyApplySol(probKey, execPlan, child_node.child_option);
            if(!childScheme)
                return nullptr;

            curScheme->numKernels += childScheme->numKernels;
            curScheme->children.emplace_back(std::move(childScheme));
        }
    }
    // SOL_LEAF_NODE
    else if(sol_node.sol_node_type == SOL_LEAF_NODE)
    {
        // a leaf node should have exactly one child sol-node (SOL_KERNEL_ONLY or SOL_BUILTIN_KERNEL)
        if(sol_node.solution_childnodes.size() != 1)
            return nullptr;

        std::string& kernel_token   = sol_node.solution_childnodes[0].child_token;
        size_t       kernel_option  = sol_node.solution_childnodes[0].child_option;
        bool         tunable_kernel = (kernel_token != solution_map::KERNEL_TOKEN_BUILTIN_KERNEL);

        // When tuning, we're running through each bench
        // so we use the elaborated token (_leafnode_id_phase_id)
        if(TuningBenchmarker::GetSingleton().IsProcessingTuning() && tunable_kernel)
        {
            auto tuningPacket          = TuningBenchmarker::GetSingleton().GetPacket();
            int  curr_tuning_node_id   = tuningPacket->tuning_node_id;
            int  curr_tuning_phase     = tuningPacket->tuning_phase;
            int  curr_tuning_config_id = tuningPacket->current_ssn;

            // replacing the tuning target kernel_token to the candidate version
            size_t cur_leaf_node_id = execPlan.solution_kernels.size();
            kernel_token += "_leafnode_" + std::to_string(cur_leaf_node_id);

            if(cur_leaf_node_id == (size_t)curr_tuning_node_id)
            {
                // if this kernel is the one we're tuning, then we set the testing phase and
                // config_id
                kernel_token += "_phase_" + std::to_string(curr_tuning_phase);
                kernel_option = curr_tuning_config_id;
            }
            else
            {
                // if the kernel is not the tuning target: we should fix the kernel to the current
                // winner
                int curWinnerPhase = tuningPacket->winner_phases[cur_leaf_node_id];
                int curWinnerID    = tuningPacket->winner_ids[cur_leaf_node_id];

                kernel_token += "_phase_" + std::to_string(curWinnerPhase);
                kernel_option = curWinnerID;
            }
        }

        ProblemKey probKey_kernel(arch, kernel_token);
        if(!sol_map_single.has_solution_node(probKey_kernel, kernel_option))
            return nullptr;

        // get the kernel of this leaf node, be sure to pick the right kernel option
        SolutionNode& kernel_node = sol_map_single.get_solution_node(probKey_kernel, kernel_option);
        execPlan.solution_kernels.push_back(kernel_node.kernel_key);
        curScheme->numKernels = 1;

        // Keep the references, and after buffer-assignment and collapse-batch-dim,
        // we can save some info back to the kernel-configurations
        if(TuningBenchmarker::GetSingleton().IsProcessingTuning())
        {
            execPlan.sol_kernel_configs.push_back(&(kernel_node.kernel_key.kernel_config));
        }

        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS())
                << "found the kernel solution(" << arch << ", " << kernel_token
                << ") with option: " << kernel_option << std::endl;
        }
    }
    // we shouldn't handle any SOL_KERNEL_ONLY directly
    else
    {
        throw std::runtime_error("Tree-Decomposition in solution map is invalid");
        return nullptr;
    }

    // if here, means we've found valid solutions of all sub-probs
    if(LOG_TRACE_ENABLED())
    {
        (*LogSingleton::GetInstance().GetTraceOS())
            << "found solution for problemKey(" << problemKey.arch << ", " << problemKey.probToken
            << ") with option: " << sol_option << std::endl;
    }
    if(LOG_TUNING_ENABLED())
    {
        (*LogSingleton::GetInstance().GetTuningOS())
            << "[SolToken]: " << problemKey.probToken << std::endl;
    }

    return curScheme;
}

std::unique_ptr<SchemeTree> ApplySolution(ExecPlan& execPlan)
{
    std::vector<ProblemKey>     possibleKeys;
    std::unique_ptr<SchemeTree> rootNodeScheme = nullptr;
    GenerateProbKeys(*(execPlan.rootPlan), possibleKeys);

    for(const auto& probKey : possibleKeys)
    {
        // found a valid solution-tree-decomposition
        rootNodeScheme = RecursivelyApplySol(probKey, execPlan, 0);
        if(rootNodeScheme)
            break;

        execPlan.solution_kernels = EmptyFMKeyVec;
    }

    return rootNodeScheme;
}

void ProcessNode(ExecPlan& execPlan)
{
    SchemeTree* rootScheme = (execPlan.rootScheme) ? execPlan.rootScheme.get() : nullptr;
    bool        noSolution = (rootScheme == nullptr);

    execPlan.rootPlan->RecursiveBuildTree(rootScheme);

    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->inStride.size());
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->outStride.size());

    // collect leaf-nodes to execSeq and fuseShims
    execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);

    if(noSolution)
    {
        CheckFuseShimForArch(execPlan);
        OrderFuseShims(execPlan.execSeq, execPlan.fuseShims);
    }

    // initialize root plan input/output location if not already done
    if(execPlan.rootPlan->obOut == OB_UNINIT)
        execPlan.rootPlan->obOut = OB_USER_OUT;
    if(execPlan.rootPlan->obIn == OB_UNINIT)
        execPlan.rootPlan->obIn
            = execPlan.rootPlan->placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;

    // guarantee min buffers but possible less fusions
    // execPlan.assignOptStrategy = rocfft_optimize_min_buffer;
    // starting from ABT
    execPlan.assignOptStrategy = rocfft_optimize_balance;
    // try to use all buffer to get most fusion
    //execPlan.assignOptStrategy = rocfft_optimize_max_fusion;
    AssignmentPolicy policy;
    policy.AssignBuffers(execPlan);

    if(TuningBenchmarker::GetSingleton().IsProcessingTuning() == false)
    {
        // Apply the fusion after buffer, strides are assigned
        execPlan.rootPlan->ApplyFusion();

        // collect the execSeq since we've fused some kernels
        execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);
    }

    // So we also need to update the whole tree including internal nodes
    // NB: The order matters: assign param -> fusion -> refresh internal node param
    execPlan.rootPlan->RefreshTree();

    // add padding if necessary
    policy.PadPlan(execPlan);

    // Collapse high dims on leaf nodes where possible
    execPlan.rootPlan->CollapseContiguousDims();

    // Check the buffer, param and tree integrity, Note we do this after fusion
    try
    {
        // rootScheme might be nullptr and solution_kernels might be empty (when no solution)
        // if has solution, will also check if it's valid
        execPlan.rootPlan->SanityCheck(rootScheme, execPlan.solution_kernels);
    }
    catch(const std::exception& e)
    {
        // When SanityCheck fails,
        // if solution_kernels is empty or rootScheme is nullptr,
        // means this has nothing to do with solution map. Throw to terminate
        if(execPlan.solution_kernels.empty() || rootScheme == nullptr)
            throw;
        else
        {
            // data from solution map are invalid, then we're not able to use them
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "input solution are invalid, try replacing kernels" << std::endl;
            execPlan.rootPlan->SanityCheck();
        }
    }

    // get workBufSize..
    size_t tmpBufSize       = 0;
    size_t cmplxForRealSize = 0;
    size_t blueSize         = 0;
    size_t chirpSize        = 0;
    execPlan.rootPlan->DetermineBufferMemory(tmpBufSize, cmplxForRealSize, blueSize, chirpSize);

    if(execPlan.rootPlan->loadOps && execPlan.rootPlan->loadOps->enabled())
    {
        // Load ops happen on first node that reads input
        auto load_node = std::find_if(
            execPlan.execSeq.begin(), execPlan.execSeq.end(), [&execPlan](TreeNode* node) {
                return node->obIn == execPlan.rootPlan->obIn;
            });
        (*load_node)->loadOps = execPlan.rootPlan->loadOps;
    }

    if(execPlan.rootPlan->storeOps && execPlan.rootPlan->storeOps->enabled())
    {
        // Store ops happen on last node of the plan that writes
        // output
        auto store_node = std::find_if(
            execPlan.execSeq.rbegin(), execPlan.execSeq.rend(), [&execPlan](TreeNode* node) {
                return node->obOut == execPlan.rootPlan->obOut;
            });
        (*store_node)->storeOps = execPlan.rootPlan->storeOps;
    }

    // compile kernels for applicable nodes
    RuntimeCompilePlan(execPlan);

    execPlan.workBufSize      = tmpBufSize + cmplxForRealSize + blueSize + chirpSize;
    execPlan.tmpWorkBufSize   = tmpBufSize;
    execPlan.copyWorkBufSize  = cmplxForRealSize;
    execPlan.blueWorkBufSize  = blueSize;
    execPlan.chirpWorkBufSize = chirpSize;
}

void PrintNode(rocfft_ostream& os, const ExecPlan& execPlan, const int indent)
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << indentStr
       << "**********************************************************************"
          "*********"
       << std::endl;

    const size_t N = product(execPlan.rootPlan->length.begin(), execPlan.rootPlan->length.end())
                     * execPlan.rootPlan->batch;
    os << indentStr << "Work buffer size: " << execPlan.workBufSize << std::endl;
    os << indentStr << "Work buffer ratio: " << (double)execPlan.workBufSize / (double)N
       << std::endl;
    os << indentStr << "Assignment strategy: " << PrintOptimizeStrategy(execPlan.assignOptStrategy)
       << std::endl;

    execPlan.rootPlan->Print(os, indent);

    os << indentStr << "GridParams\n";
    for(const auto& gp : execPlan.gridParam)
    {
        os << indentStr << "  b[" << gp.b_x << "," << gp.b_y << "," << gp.b_z << "] wgs["
           << gp.wgs_x << "," << gp.wgs_y << "," << gp.wgs_z << "], dy_lds bytes " << gp.lds_bytes
           << "\n";
    }
    os << indentStr << "End GridParams\n";

    os << indentStr
       << "======================================================================"
          "========="
       << std::endl
       << std::endl;
}
