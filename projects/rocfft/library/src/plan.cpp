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
#include "rocfft_exception.h"
#include "rocfft_mpi.h"
#include "rocfft_ostream.hpp"
#include "rtc_kernel.h"
#include "solution_map.h"
#include "tuning_helper.h"
#include "tuning_plan_tuner.h"

#include <algorithm>
#include <assert.h>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#ifdef ROCFFT_MPI_ENABLE
#include <type_traits>
#include <typeinfo>
#endif

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

// clang-format off
#define ROCFFT_VERSION_STRING (TO_STR(rocfft_version_major) "." \
                               TO_STR(rocfft_version_minor) "." \
                               TO_STR(rocfft_version_patch) "." \
                               TO_STR(rocfft_version_tweak) )
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

void rocfft_plan_description_t::init_defaults(rocfft_transform_type      transformType,
                                              rocfft_result_placement    placement,
                                              const std::vector<size_t>& lengths,
                                              const std::vector<size_t>& outputLengths)
{
    const size_t rank = lengths.size();

    // assume interleaved data
    if(inArrayType == rocfft_array_type_unset)
    {
        switch(transformType)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            inArrayType = rocfft_array_type_complex_interleaved;
            break;
        case rocfft_transform_type_real_inverse:
            inArrayType = rocfft_array_type_hermitian_interleaved;
            break;
        case rocfft_transform_type_real_forward:
            inArrayType = rocfft_array_type_real;
            break;
        }
    }
    if(outArrayType == rocfft_array_type_unset)
    {
        switch(transformType)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            outArrayType = rocfft_array_type_complex_interleaved;
            break;
        case rocfft_transform_type_real_forward:
            outArrayType = rocfft_array_type_hermitian_interleaved;
            break;
        case rocfft_transform_type_real_inverse:
            outArrayType = rocfft_array_type_real;
            break;
        }
    }

    // Set inStrides, if not specified
    if(inStrides.empty())
    {
        inStrides.push_back(1);

        if((transformType == rocfft_transform_type_real_forward)
           && (placement == rocfft_placement_inplace))
        {
            // real-to-complex in-place
            size_t dist = 2 * outputLengths[0];

            for(size_t i = 1; i < rank; i++)
            {
                inStrides.push_back(dist);
                dist *= lengths[i];
            }
        }
        else
        {
            // Set the inStrides to deal with contiguous data
            for(size_t i = 1; i < rank; i++)
                inStrides.push_back(lengths[i - 1] * inStrides[i - 1]);
        }
    }

    // Set outStrides, if not specified
    if(outStrides.empty())
    {
        outStrides.push_back(1);

        if((transformType == rocfft_transform_type_real_inverse)
           && (placement == rocfft_placement_inplace))
        {
            // complex-to-real in-place
            size_t dist = 2 * lengths[0];

            for(size_t i = 1; i < rank; i++)
            {
                outStrides.push_back(dist);
                dist *= lengths[i];
            }
        }
        else
        {
            // Set the outStrides to deal with contiguous data
            for(size_t i = 1; i < rank; i++)
                outStrides.push_back(outputLengths[i - 1] * outStrides[i - 1]);
        }
    }

    // Set in and out Distances, if not specified
    if(inDist == 0)
    {
        // In-place 1D transforms need extra dist.
        if(transformType == rocfft_transform_type_real_forward && lengths.size() == 1
           && placement == rocfft_placement_inplace)
            inDist = 2 * (lengths[0] / 2 + 1) * inStrides[0];
        else
            inDist = lengths[rank - 1] * inStrides[rank - 1];
    }
    if(outDist == 0)
    {
        // In-place 1D transforms need extra dist.
        if(transformType == rocfft_transform_type_real_inverse && lengths.size() == 1
           && placement == rocfft_placement_inplace)
            outDist = 2 * lengths[0] * outStrides[0];
        else
            outDist = outputLengths[rank - 1] * outStrides[rank - 1];
    }
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

void rocfft_plan_t::sort()
{
    // copy the lengths + strides separately, and then sort them
    // fastest to slowest.
    struct rocfft_iodim
    {
        size_t length;
        size_t olength;
        size_t istride;
        size_t ostride;
    };

    // complex-complex transforms can be freely reordered starting from
    // the fastest dimension.  real-complex has to leave the fastest
    // dimension alone
    const size_t start_dim = (transformType == rocfft_transform_type_complex_forward
                              || transformType == rocfft_transform_type_complex_inverse)
                                 ? 0
                                 : 1;

    std::vector<rocfft_iodim> iodims;
    for(size_t dim = start_dim; dim < rank; ++dim)
        iodims.push_back(rocfft_iodim{
            lengths[dim], outputLengths[dim], desc.inStrides[dim], desc.outStrides[dim]});
    if(iodims.empty())
        return;

    bool sort_on_istride = true;
    auto sorter          = [sort_on_istride](const rocfft_iodim& a, const rocfft_iodim& b) {
        // move any lengths of 1 to the end
        if(a.length == 1 && b.length != 1)
            return false;
        if(b.length == 1 && a.length != 1)
            return true;
        return sort_on_istride ? (a.istride < b.istride) : (a.ostride < b.ostride);
    };

    // sort on istride first
    std::sort(iodims.begin(), iodims.end(), sorter);

    // if that means ostride is no longer sorted, then don't bother
    // changing anything - the user is asking for some kind of
    // transposed FFT so let's just assume they know what they're doing
    sort_on_istride = false;
    if(!std::is_sorted(iodims.begin(), iodims.end(), sorter))
        return;

    // chop off any lengths of 1 from the end
    while(iodims.size() > 1 && iodims.back().length == 1)
    {
        --rank;
        iodims.pop_back();
    }
    // copy back the sorted lengths + strides
    for(size_t dim = start_dim; dim < rank; ++dim)
    {
        lengths[dim]         = iodims[dim - start_dim].length;
        outputLengths[dim]   = iodims[dim - start_dim].olength;
        desc.inStrides[dim]  = iodims[dim - start_dim].istride;
        desc.outStrides[dim] = iodims[dim - start_dim].ostride;
    }
}

bool rocfft_plan_t::is_contiguous(const std::vector<size_t>& length,
                                  const std::vector<size_t>& stride,
                                  size_t                     dist)
{
    size_t expected_stride = 1;
    auto   stride_it       = stride.begin();
    auto   length_it       = length.begin();
    for(; stride_it != stride.end() && length_it != length.end(); ++stride_it, ++length_it)
    {
        if(*stride_it != expected_stride)
            return false;
        expected_stride *= *length_it;
    }
    return expected_stride == dist;
}

bool rocfft_plan_t::is_contiguous_input()
{
    return is_contiguous(lengths, desc.inStrides, desc.inDist);
}
bool rocfft_plan_t::is_contiguous_output()
{
    return is_contiguous(lengths, desc.outStrides, desc.outDist);
}

size_t rocfft_plan_t::AddMultiPlanItem(std::unique_ptr<MultiPlanItem>&& item,
                                       const std::vector<size_t>&       antecedents)
{
    // ensure antecedents all exist
    if(std::any_of(
           antecedents.begin(), antecedents.end(), [=](size_t i) { return i >= multiPlan.size(); }))
        throw std::runtime_error("antecedent does not exist");

    item->local_comm_rank = get_local_comm_rank();

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

size_t rocfft_plan_t::WorkBufBytes() const
{
    auto base_type_size = real_type_size(precision);

    // Something wants to know how much work buffer to allocate, but
    // our plan could span multiple ranks and devices.  For now, just
    // get the largest work buf size of any item in the
    // multi-rank/device plan.
    size_t workBufBytes = 0;
    for(const auto& i : multiPlan)
    {
        if(i)
            workBufBytes = std::max(workBufBytes, i->WorkBufBytes(base_type_size));
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

    // If you give say that you're going to give is a communicator, please don't give us a null
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
        description->user_mpi_comm = MPI_COMM_NULL;
#endif
        break;
    }
#ifdef ROCFFT_MPI_ENABLE
    case rocfft_comm_mpi:
    {
        description->user_mpi_comm = *static_cast<MPI_Comm*>(comm_handle);
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

    description->inArrayType  = in_array_type;
    description->outArrayType = out_array_type;

    if(in_offsets != nullptr)
    {
        description->inOffset[0] = in_offsets[0];
        if((in_array_type == rocfft_array_type_complex_planar)
           || (in_array_type == rocfft_array_type_hermitian_planar))
            description->inOffset[1] = in_offsets[1];
    }

    if(out_offsets != nullptr)
    {
        description->outOffset[0] = out_offsets[0];
        if((out_array_type == rocfft_array_type_complex_planar)
           || (out_array_type == rocfft_array_type_hermitian_planar))
            description->outOffset[1] = out_offsets[1];
    }

    if(in_strides != nullptr)
    {
        description->inStrides.clear();
        std::copy(
            in_strides, in_strides + in_strides_size, std::back_inserter(description->inStrides));
    }

    if(in_distance != 0)
        description->inDist = in_distance;

    if(out_strides != nullptr)
    {
        description->outStrides.clear();
        std::copy(out_strides,
                  out_strides + out_strides_size,
                  std::back_inserter(description->outStrides));
    }

    if(out_distance != 0)
        description->outDist = out_distance;

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

bool rocfft_brick_t::empty() const
{
    auto len = length();
    return std::any_of(len.begin(), len.end(), [](int l) { return l == 0; });
}

size_t rocfft_brick_t::count_elems() const
{
    auto len = length();
    return product(len.begin(), len.end());
}

std::vector<size_t> rocfft_brick_t::contiguous_strides() const
{
    std::vector<size_t> ret;
    size_t              dist = 1;

    auto len = length();
    for(size_t i = 0; i < len.size(); ++i)
    {
        ret.push_back(dist);
        dist *= len[i];
    }
    return ret;
}

bool rocfft_brick_t::is_contiguous() const
{
    auto contiguous = contiguous_strides();

    auto len = length();
    // only care about dimensions whose length is greater than 1
    for(size_t i = 1; i < len.size(); ++i)
    {
        if(len[i] > 1)
        {
            if(contiguous[i] != stride[i])
                return false;
        }
    }
    return true;
}

bool rocfft_brick_t::is_contiguous_in_field(const std::vector<size_t>& field_length,
                                            const std::vector<size_t>& field_stride) const
{
    const auto brick_len = length();

    // a contiguous brick in a field is shorter than field length
    // only on the highest dimension, ignoring dimensions of
    // length-1, and strides must match the field for all those
    // dimensions.
    size_t shortDim       = std::numeric_limits<size_t>::max();
    size_t highestNot1Dim = std::numeric_limits<size_t>::max();
    for(size_t i = 0; i < brick_len.size(); ++i)
    {
        if(field_length[i] == 1)
            continue;

        highestNot1Dim = i;
        if(brick_len[i] < field_length[i])
        {
            // another dim is already short?  this isn't contiguous
            if(shortDim != std::numeric_limits<size_t>::max())
                return false;
            shortDim = i;
        }
    }

    return shortDim == highestNot1Dim;
}

rocfft_brick_t rocfft_brick_t::intersect(const rocfft_brick_t& other) const
{
    rocfft_brick_t ret;

    for(size_t i = 0; i < lower.size(); ++i)
        ret.lower.push_back(std::max(lower[i], other.lower[i]));

    for(size_t i = 0; i < upper.size(); ++i)
        ret.upper.push_back(std::min(upper[i], other.upper[i]));

    return ret;
}

bool rocfft_brick_t::equal_coords(const rocfft_brick_t& other) const
{
    return this->lower == other.lower && this->upper == other.upper;
}

size_t rocfft_brick_t::offset_in_field(const std::vector<size_t>& fieldStride) const
{
    return std::inner_product(lower.begin(), lower.end(), fieldStride.begin(), 0);
}

std::string rocfft_brick_t::str() const
{
    std::string ret;
    ret += "lower ";
    for(auto i : lower)
    {
        ret += " ";
        ret += std::to_string(i);
    }

    ret += " upper ";
    for(auto i : upper)
    {
        ret += " ";
        ret += std::to_string(i);
    }

    ret += " stride ";
    for(auto i : stride)
    {
        ret += " ";
        ret += std::to_string(i);
    }
    ret += " comm rank ";
    ret += std::to_string(location.comm_rank);
    ret += " device ";
    ret += std::to_string(location.device);
    return ret;
}

rocfft_status rocfft_field_add_brick(rocfft_field field, rocfft_brick brick)
try
{
    log_trace(__func__, "field", field, "brick", brick);
    if(!field || !brick)
        return rocfft_status_invalid_arg_value;
    field->bricks.emplace_back(*brick);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

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

    auto brick_ptr = std::make_unique<rocfft_brick_t>();
    std::copy_n(field_lower, dim, std::back_inserter(brick_ptr->lower));
    std::copy_n(field_upper, dim, std::back_inserter(brick_ptr->upper));
    std::copy_n(brick_stride, dim, std::back_inserter(brick_ptr->stride));

    brick_ptr->location.device = deviceID;
    *brick                     = brick_ptr.release();
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

std::string rocfft_bench_command(const rocfft_plan& plan)
{
    rocfft_params params;

    params.nbatch         = plan->batch;
    params.placement      = fft_result_placement_from_rocfft_result_placement(plan->placement);
    params.transform_type = fft_transform_type_from_rocfft_transform_type(plan->transformType);
    params.precision      = fft_precision_from_rocfft_precision(plan->precision);
    params.itype          = fft_array_type_from_rocfft_array_type(plan->desc.inArrayType);
    params.otype          = fft_array_type_from_rocfft_array_type(plan->desc.outArrayType);
    params.idist          = plan->desc.inDist;
    params.odist          = plan->desc.outDist;
    params.scale_factor   = plan->desc.storeOps.scale_factor;

    // reverse-copy lengths and strides (col-major to row-major)
    params.length.assign(plan->lengths.rbegin(), plan->lengths.rend());
    params.istride.assign(plan->desc.inStrides.rbegin(), plan->desc.inStrides.rend());
    params.ostride.assign(plan->desc.outStrides.rbegin(), plan->desc.outStrides.rend());
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
                  for(size_t bidx = 0; bidx < src.size(); ++bidx)
                  {
                      fft_params::fft_brick& dest_brick = dest[fidx].bricks[bidx];
                      const rocfft_brick_t&  src_brick  = src[fidx].bricks[bidx];
                      // reverse-copy bricks' coordinates and strides (col-major to row-major)
                      dest_brick.lower.assign(src_brick.lower.rbegin(), src_brick.lower.rend());
                      dest_brick.upper.assign(src_brick.upper.rbegin(), src_brick.upper.rend());
                      dest_brick.stride.assign(src_brick.stride.rbegin(), src_brick.stride.rend());
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

// Verify that the transform type, array type, and placement are coherent.
rocfft_status check_array_type_validity(const rocfft_plan plan)
{
    switch(plan->transformType)
    {
    case rocfft_transform_type_complex_forward:
    case rocfft_transform_type_complex_inverse:
        // We need complex input data
        if(!((plan->desc.inArrayType == rocfft_array_type_complex_interleaved)
             || (plan->desc.inArrayType == rocfft_array_type_complex_planar)))
            return rocfft_status_invalid_array_type;
        // We need complex output data
        if(!((plan->desc.outArrayType == rocfft_array_type_complex_interleaved)
             || (plan->desc.outArrayType == rocfft_array_type_complex_planar)))
            return rocfft_status_invalid_array_type;
        // In-place transform requires that the input and output
        // format be identical
        if(plan->placement == rocfft_placement_inplace)
        {
            if(plan->desc.inArrayType != plan->desc.outArrayType)
                return rocfft_status_invalid_array_type;
        }
        break;
    case rocfft_transform_type_real_forward:
        // Input must be real
        if(plan->desc.inArrayType != rocfft_array_type_real)
            return rocfft_status_invalid_array_type;
        // Output must be Hermitian
        if(!((plan->desc.outArrayType == rocfft_array_type_hermitian_interleaved)
             || (plan->desc.outArrayType == rocfft_array_type_hermitian_planar)))
            return rocfft_status_invalid_array_type;
        // In-place transform must output to interleaved format
        if((plan->placement == rocfft_placement_inplace)
           && (plan->desc.outArrayType != rocfft_array_type_hermitian_interleaved))
            return rocfft_status_invalid_array_type;
        break;
    case rocfft_transform_type_real_inverse:
        // Output must be real
        if(plan->desc.outArrayType != rocfft_array_type_real)
            return rocfft_status_invalid_array_type;
        // Input must be Hermitian
        if(!((plan->desc.inArrayType == rocfft_array_type_hermitian_interleaved)
             || (plan->desc.inArrayType == rocfft_array_type_hermitian_planar)))
            return rocfft_status_invalid_array_type;
        // In-place transform must have interleaved input
        if((plan->placement == rocfft_placement_inplace)
           && (plan->desc.inArrayType != rocfft_array_type_hermitian_interleaved))
            return rocfft_status_invalid_array_type;
        break;
    }
    return rocfft_status_success;
}

// Given a rocfft_plan with validated parameters, set the transform parameters for the root of the
// tree plan.
void set_rootplan_params(const rocfft_plan plan, NodeMetaData& planData)
{
    planData.dimension = plan->rank;
    planData.batch     = plan->batch;
    for(size_t i = 0; i < plan->rank; i++)
    {
        planData.length.push_back(plan->lengths[i]);
        planData.outputLength.push_back(plan->outputLengths[i]);

        planData.inStride.push_back(plan->desc.inStrides[i]);
        planData.outStride.push_back(plan->desc.outStrides[i]);
    }
    planData.iDist     = plan->desc.inDist;
    planData.oDist     = plan->desc.outDist;
    planData.placement = plan->placement;

    // If in+out fields are specified, currently that means we're
    // gathering the data to one device and doing FFT there.  So
    // the FFT becomes in-place.
    // TODO: if we have no bricks, or just one brick which is the entire FFT, then we can
    // just do a normal FFT.
    if(!plan->desc.inFields.empty() && !plan->desc.outFields.empty())
    {
        // c2c can be inplace so both input/output can be
        // contiguous without needing extra buffers
        if(plan->transformType == rocfft_transform_type_complex_forward
           || plan->transformType == rocfft_transform_type_complex_inverse)
        {
            planData.placement = rocfft_placement_inplace;
        }
        // real-complex would require non-contiguous data to be
        // inplace (which likely means more packing/unpacking and
        // extra temp buffer usage anyway), so make it
        // not-in-place instead
        else
        {
            planData.placement = rocfft_placement_notinplace;
        }
    }
    // If we only have outfield, then there's no need to gather.  But we can't assume we can
    // overwrite input, so the FFT will be outplace to a temp buf
    else if(plan->desc.inFields.empty() && !plan->desc.outFields.empty())
    {
        planData.placement = rocfft_placement_notinplace;
    }
    // If we only have infield, then we must gather.
    else if(!plan->desc.inFields.empty() && plan->desc.outFields.empty())
    {
        // If output is contiguous and this is c2c then we can
        // gather to the output buf and do an inplace FFT.
        if(plan->is_contiguous_output()
           && (plan->transformType == rocfft_transform_type_complex_forward
               || plan->transformType == rocfft_transform_type_complex_inverse))
            planData.placement = rocfft_placement_inplace;
        // Otherwise we must gather to a temp buf and do outplace FFT to output
        else
            planData.placement = rocfft_placement_notinplace;
    }

    planData.precision = plan->precision;
    planData.direction = ((plan->transformType == rocfft_transform_type_complex_forward)
                          || (plan->transformType == rocfft_transform_type_real_forward))
                             ? -1
                             : 1;

    planData.inArrayType  = plan->desc.inArrayType;
    planData.outArrayType = plan->desc.outArrayType;
    planData.rootIsC2C    = (planData.inArrayType != rocfft_array_type_real)
                         && (planData.outArrayType != rocfft_array_type_real);
}

void set_bluestein_strides(const rocfft_plan plan, NodeMetaData& planData)
{
    std::array<size_t, 3> inStridesBlue  = {0, 0, 0};
    std::array<size_t, 3> outStridesBlue = {0, 0, 0};
    std::array<size_t, 3> lengthsBlue    = {0, 0, 0};
    size_t                inDistBlue     = 0;
    size_t                outDistBlue    = 0;

    function_pool pool{planData.deviceProp};

    const auto precision     = plan->precision;
    const auto transformType = plan->transformType;
    const auto rank          = plan->rank;
    const auto lengths       = plan->lengths;
    const auto placement     = plan->placement;
    const auto dimension     = planData.dimension;

    assert(rank == dimension);

    // for real inverse transforms we need to look at the complex length
    auto fftLength
        = transformType == rocfft_transform_type_real_inverse ? plan->outputLengths : plan->lengths;

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

// RAII struct to 'lease' a temp buffer from a multimap of per-device
// buffers.  When this struct is destroyed, the buffer is returned to
// the map for reuse.
struct TempBufferLease
{
    TempBufferLease(
        std::multimap<rocfft_location_t, std::shared_ptr<InternalTempBuffer>>& _tempBuffers,
        int                                                                    local_comm_rank,
        rocfft_location_t                                                      _location,
        size_t                                                                 _elems,
        size_t                                                                 _elem_size)
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
        const size_t alloc_size = _elems * _elem_size;
        auto         i          = tempBuffers->lower_bound(location);
        if(i != tempBuffers->upper_bound(location))
        {
            // found a buffer, ensure it's big enough
            i->second->set_size_bytes(alloc_size);

            // leasing out this temp buffer, remove it from the map
            buf = i->second;
            tempBuffers->erase(i);
            return;
        }
        // no buffer was found, allocate a new one
        buf = std::make_shared<InternalTempBuffer>(local_comm_rank);
        buf->set_size_bytes(alloc_size);
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

void rocfft_plan_t::AllocateInternalTempBuffers()
{
    for(auto& t : tempBuffers)
    {
        if(t.first.comm_rank != get_local_comm_rank())
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

    // In a multi-process enviroment, we've gathered the bricks on
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

std::vector<size_t> rocfft_plan_t::GatherBricksToField(rocfft_location_t currentLocation,
                                                       const std::vector<rocfft_brick_t>& bricks,
                                                       rocfft_precision                   precision,
                                                       rocfft_array_type                  arrayType,
                                                       const std::vector<size_t>& field_length,
                                                       const std::vector<size_t>& field_stride,
                                                       BufferPtr                  output,
                                                       const std::vector<size_t>& antecedents,
                                                       size_t                     elem_size)
{
    std::vector<size_t>            outputPlanItems;
    std::vector<TempBufferLease>   gatherPackBufs;
    std::optional<TempBufferLease> gatherDestBuf;

    std::vector<BufferPtr> outputBufs = GatherUserBuffers(BufferPtr::user_input, bricks);

    const auto local_comm_rank = get_local_comm_rank();

    BufferPtr  gatherDest;
    const bool gatherToTemp
        = std::any_of(bricks.begin(), bricks.end(), [&](const rocfft_brick_t& brick) {
              return !brick.is_contiguous()
                     || !brick.is_contiguous_in_field(field_length, field_stride);
          });
    if(gatherToTemp)
    {
        // this is from source-rank to source-rank, before MPI comm.
        gatherDestBuf
            = std::make_optional<TempBufferLease>(tempBuffers,
                                                  local_comm_rank,
                                                  currentLocation,
                                                  product(field_length.begin(), field_length.end()),
                                                  elem_size);
        gatherDest = BufferPtr::temp(gatherDestBuf->data());
    }
    else
    {
        gatherDest = output;
    }

    // Create gather operation
    auto gatherPtr = std::make_unique<CommGather>(
        local_comm_rank, precision, arrayType, currentLocation, gatherDest);
    auto gather         = gatherPtr.get();
    gather->description = "Gather " + std::to_string(bricks.size()) + " bricks";

    // Add gather to the plan first - we will add operations to it later
    size_t gatherIdx = AddMultiPlanItem(std::move(gatherPtr), antecedents);

    // We'll be packing the brick data contiguously into the output,
    // so keep track of how much of the output we've filled up.
    // Track offset per location since we have one temp buffer per
    // location.
    std::map<rocfft_location_t, size_t> packOffsets;
    size_t                              gatherOffset = 0;
    for(size_t brickIdx = 0; brickIdx < bricks.size(); ++brickIdx)
    {
        const auto& brick = bricks[brickIdx];

        // init offset for this location if it's not already
        // initialized, and get a reference to this location's
        // offset.
        size_t& packOffset
            = packOffsets.emplace(brick.location, static_cast<size_t>(0)).first->second;

        if(brick.is_contiguous())
        {
            // Contiguous brick, just copy the data
            gather->AddOperation(
                local_comm_rank,
                {brick.location, outputBufs[brickIdx], 0, gatherOffset, brick.count_elems()});
        }
        else
        {
            // Allocate temp memory for the pack
            gatherPackBufs.emplace_back(
                tempBuffers, local_comm_rank, brick.location, brick.count_elems(), elem_size);

            // Brick is not contiguous, insert a pack node on the brick's device
            std::string description = "pack brick " + std::to_string(brickIdx) + " before gather";
            auto        packIdx
                = AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                   brick.location,
                                                   brick.length(),
                                                   precision,
                                                   arrayType,
                                                   outputBufs[brickIdx],
                                                   0,
                                                   brick.stride,
                                                   BufferPtr::temp(gatherPackBufs.back().data()),
                                                   packOffset,
                                                   brick.contiguous_strides(),
                                                   std::move(description)),
                                   antecedents);
            AddAntecedent(gatherIdx, packIdx);

            gather->AddOperation(local_comm_rank,
                                 {brick.location,
                                  BufferPtr::temp(gatherPackBufs.back().data()),
                                  0,
                                  gatherOffset,
                                  brick.count_elems()});
        }

        // if brick is not contiguous in the field, it was packed to
        // be contiguous for communication.  unpack it so it's
        // contiguous in the field.
        if(!brick.is_contiguous() || !brick.is_contiguous_in_field(field_length, field_stride))
        {
            std::string description = "unpack brick " + std::to_string(brickIdx) + " after gather";

            outputPlanItems.push_back(
                AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                 currentLocation,
                                                 brick.length(),
                                                 precision,
                                                 arrayType,
                                                 BufferPtr::temp(gatherDestBuf->data()),
                                                 gatherOffset,
                                                 brick.contiguous_strides(),
                                                 output,
                                                 brick.offset_in_field(field_stride),
                                                 field_stride,
                                                 std::move(description)),
                                 {gatherIdx}));
        }

        packOffset += brick.count_elems();
        gatherOffset += brick.count_elems();
    }

    if(outputPlanItems.empty())
    {
        // following items in the plan just depend on the gather completing
        outputPlanItems.push_back(gatherIdx);
    }
    return outputPlanItems;
}

std::vector<size_t> rocfft_plan_t::ScatterFieldToBricks(rocfft_location_t          currentLocation,
                                                        BufferPtr                  input,
                                                        rocfft_precision           precision,
                                                        rocfft_array_type          arrayType,
                                                        const std::vector<size_t>& field_length,
                                                        const std::vector<size_t>& field_stride,
                                                        const std::vector<rocfft_brick_t>& bricks,
                                                        const std::vector<size_t>& antecedents,
                                                        size_t                     elem_size)
{
    std::vector<size_t>            outputPlanItems;
    std::vector<TempBufferLease>   scatterPackBufs;
    std::optional<TempBufferLease> scatterSrcBuf;

    std::vector<BufferPtr> outputBufs = GatherUserBuffers(BufferPtr::user_output, bricks);

    const auto local_comm_rank = get_local_comm_rank();

    BufferPtr  scatterSrc;
    const bool scatterFromTemp
        = std::any_of(bricks.begin(), bricks.end(), [&](const rocfft_brick_t& b) {
              return !b.is_contiguous_in_field(field_length, field_stride);
          });
    if(scatterFromTemp)
    {
        scatterSrcBuf
            = std::make_optional<TempBufferLease>(tempBuffers,
                                                  local_comm_rank,
                                                  currentLocation,
                                                  product(field_length.begin(), field_length.end()),
                                                  elem_size);
        scatterSrc = BufferPtr::temp(scatterSrcBuf->data());
    }
    else
    {
        scatterSrc = input;
    }

    // create scatter operation
    auto scatterPtr
        = std::make_unique<CommScatter>(precision, arrayType, currentLocation, scatterSrc);
    auto scatter         = scatterPtr.get();
    scatter->description = "Scatter " + std::to_string(bricks.size()) + " bricks";

    // add scatter to the multi-plan first, add operations afterwards
    auto scatterIdx = AddMultiPlanItem(std::move(scatterPtr), antecedents);

    // we'll be packing the brick data contiguously into the output,
    // so keep track of how much of the output we've filled up
    size_t scatterOffset = 0;

    for(size_t brickIdx = 0; brickIdx < bricks.size(); ++brickIdx)
    {
        const auto& brick = bricks[brickIdx];

        if(brick.is_contiguous_in_field(field_length, field_stride))
        {
            // contiguous brick, just copy the data
            scatter->AddOperation(
                local_comm_rank,
                {brick.location, outputBufs[brickIdx], scatterOffset, 0, brick.count_elems()});
        }
        else
        {
            // pack data to be contiguous
            std::string description = "pack brick " + std::to_string(brickIdx) + " before scatter";

            const auto brickLen = brick.length();
            auto       packIdx  = AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                            currentLocation,
                                                            brick.length(),
                                                            precision,
                                                            arrayType,
                                                            input,
                                                            brick.offset_in_field(field_stride),
                                                            field_stride,
                                                            BufferPtr::temp(scatterSrcBuf->data()),
                                                            scatterOffset,
                                                            brick.contiguous_strides(),
                                                            std::move(description)),
                                            antecedents);
            AddAntecedent(scatterIdx, packIdx);

            // Bricks are packed to be contiguous - if output is the
            // same shape, then there's no need for unpacking
            if(brick.is_contiguous())
            {
                scatter->AddOperation(
                    local_comm_rank,
                    {brick.location, outputBufs[brickIdx], scatterOffset, 0, brick.count_elems()});
            }
            else
            {
                // allocate memory for packed data
                scatterPackBufs.emplace_back(
                    tempBuffers, local_comm_rank, brick.location, brick.count_elems(), elem_size);

                // send the data
                scatter->AddOperation(local_comm_rank,
                                      {brick.location,
                                       BufferPtr::temp(scatterPackBufs.back().data()),
                                       scatterOffset,
                                       0,
                                       brick.count_elems()});

                // unpack data after sending
                description = "unpack brick " + std::to_string(brickIdx) + " after scatter";

                outputPlanItems.push_back(
                    AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                     brick.location,
                                                     brick.length(),
                                                     precision,
                                                     arrayType,
                                                     BufferPtr::temp(scatterPackBufs.back().data()),
                                                     0,
                                                     brick.contiguous_strides(),
                                                     outputBufs[brickIdx],
                                                     0,
                                                     brick.stride,
                                                     std::move(description)),
                                     {scatterIdx}));
            }
        }
        scatterOffset += brick.count_elems();
    }

    // following items in the plan just depend on the scatter
    // completing if no unpacking was required
    if(outputPlanItems.empty())
    {
        outputPlanItems.push_back(scatterIdx);
    }

    return outputPlanItems;
}

// test if the specified dimension is split up across separate bricks
// in the field
static bool DimensionSplitInField(size_t length, size_t dimIdx, const rocfft_field_t& field)
{
    for(const auto& b : field.bricks)
        if(b.length()[dimIdx] != length)
            return true;
    return false;
}

// Construct a single-device execPlan - fill out the provided
// execPlan with nodes to implement the FFT.
void rocfft_plan_t::GatherScatterSingleDevicePlan(std::unique_ptr<ExecPlan>&& execPlanPtr)
{
    // The smart pointer will be moved into the multi-plan during this
    // function, so keep a plain non-owning pointer
    auto execPlan = execPlanPtr.get();

    // If we have no input/output fields, then the single ExecPlan is
    // exactly what we need to do
    if(desc.inFields.empty() && desc.outFields.empty())
    {
        AddMultiPlanItem(std::move(execPlanPtr), {});
        return;
    }

    // Code below this line is only required for multi-device plans
    execPlan->mgpuPlan    = true;
    execPlan->description = "FFT gathered data";

    // Ensure fields are real or interleaved - planar is not supported
    if((!desc.inFields.empty() && array_type_is_planar(desc.inArrayType))
       || (!desc.outFields.empty() && array_type_is_planar(desc.outArrayType)))
    {
        throw std::runtime_error("fields must be not be planar");
    }

    const auto in_elem_size = element_size(precision, desc.inArrayType);
    // We do in-place transforms here, so allocate a buffer for the
    // complex side of real-complex transforms, since it's bigger.
    const size_t in_elem_count = compute_ptrdiff(lengths, desc.inStrides, batch, desc.inDist);

    const auto local_comm_rank = get_local_comm_rank();

    // May need buffer(s) to do the actual FFT
    std::shared_ptr<TempBufferLease> fftBuf;
    std::shared_ptr<TempBufferLease> fftOutBuf;

    // Gather to temp buf if infields were specified and output is not contiguous or is also a field
    BufferPtr gatherBuf;
    {
        if(!desc.inFields.empty() && (!is_contiguous_output() || !desc.outFields.empty()))
        {
            fftBuf = std::make_shared<TempBufferLease>(
                tempBuffers, local_comm_rank, execPlanPtr->location, in_elem_count, in_elem_size);
            gatherBuf = BufferPtr::temp(fftBuf->data());
        }
        else if(desc.inFields.empty())
        {
            gatherBuf = BufferPtr::user_input(0, 0);
        }
        else
        {
            // Else, we can gather directly to the output buffer
            gatherBuf = BufferPtr::user_output(0, 0);
        }
    }

    // Allocate another temp buf if FFT is not-in-place and outfield was specified
    // TODO: this only needs to be done if there are multiple bricks in the output field.
    if(placement == rocfft_placement_notinplace && !desc.outFields.empty())
    {
        const auto   out_elem_size  = element_size(precision, desc.outArrayType);
        const size_t out_elem_count = compute_ptrdiff(
            execPlan->rootPlan->GetOutputLength(), desc.outStrides, batch, desc.outDist);
        fftOutBuf = std::make_shared<TempBufferLease>(
            tempBuffers, local_comm_rank, execPlanPtr->location, out_elem_count, out_elem_size);
    }

    std::vector<size_t> gatherIndexes;
    {
        for(const auto& inField : desc.inFields)
        {
            // brick indexes include batch so create a set of comparable field strides
            auto fieldStrideWithBatch = desc.inStrides;
            fieldStrideWithBatch.push_back(desc.inDist);

            auto fieldLengthWithBatch = lengths;
            fieldLengthWithBatch.push_back(batch);

            auto curIndexes = GatherBricksToField(execPlan->location,
                                                  inField.bricks,
                                                  precision,
                                                  desc.inArrayType,
                                                  fieldLengthWithBatch,
                                                  fieldStrideWithBatch,
                                                  gatherBuf,
                                                  {},
                                                  element_size(precision, desc.inArrayType));
            std::copy(curIndexes.begin(), curIndexes.end(), std::back_inserter(gatherIndexes));
        }

        // Data is gathered and unpacked (if necessary), run the core FFT plan we started with
        if(gatherBuf)
            execPlan->inputPtr = gatherBuf;
        else
            execPlan->inputPtr = BufferPtr::user_input(0, 0);

        if(execPlan->rootPlan->placement == rocfft_placement_inplace)
            execPlan->outputPtr = execPlan->inputPtr;
        else if(fftOutBuf)
            execPlan->outputPtr = BufferPtr::temp(fftOutBuf->data());
    }
    auto fftIdx = AddMultiPlanItem(std::move(execPlanPtr), gatherIndexes);

    // Scatter data back out
    for(const auto& outField : desc.outFields)
    {
        // brick indexes include batch so create a set of comparable field strides
        auto fieldStrideWithBatch = desc.outStrides;
        fieldStrideWithBatch.push_back(desc.outDist);

        auto fieldLengthWithBatch = lengths;
        fieldLengthWithBatch.push_back(batch);

        auto scatterSrcBuf = execPlan->rootPlan->placement == rocfft_placement_notinplace
                                 ? fftOutBuf->data()
                                 : fftBuf->data();

        ScatterFieldToBricks(execPlan->location,
                             BufferPtr::temp(scatterSrcBuf),
                             execPlan->rootPlan->precision,
                             execPlan->rootPlan->outArrayType,
                             fieldLengthWithBatch,
                             fieldStrideWithBatch,
                             outField.bricks,
                             {fftIdx},
                             element_size(precision, execPlan->rootPlan->outArrayType));
    }
}

// Construct a single-device execPlan - fill out the provided
// execPlan with nodes to implement the FFT.
static std::unique_ptr<ExecPlan> BuildSingleDevicePlan(NodeMetaData&         rootPlanData,
                                                       int                   local_comm_rank,
                                                       rocfft_location_t     location,
                                                       rocfft_transform_type transformType,
                                                       LoadOps&              loadOps,
                                                       StoreOps&             storeOps,
                                                       bool                  partOfMultiPlan)
{
    rocfft_scoped_device dev(location.device);

    auto execPlanMultiItem = std::make_unique<ExecPlan>(local_comm_rank, partOfMultiPlan, location);
    ExecPlan& execPlan     = *execPlanMultiItem;
    try
    {
        execPlan.deviceProp = rootPlanData.deviceProp;
        execPlan.rootPlan   = NodeFactory::CreateExplicitNode(rootPlanData, nullptr);

        // TODO: some solutions require the problems to be unit_stride, otherwise the
        //   scheme-tree may not be applicable. In this case, we can't apply the solutions.
        //   Currently, it happens on Real3DEven with REAL_2D_SINGLE kernels. This needs to
        //   be detected.

        // If we are doing tuning initialzing now, we shouldn't apply any solution,
        // since we are trying enumerating solutions now
        if(TuningBenchmarker::GetSingleton().IsInitializingTuning() == false)
        {
            execPlan.rootScheme = ApplySolution(execPlan);
            if(execPlan.rootScheme)
            {
                execPlan.rootPlan = nullptr;
                execPlan.rootPlan = NodeFactory::CreateExplicitNode(
                    rootPlanData, nullptr, execPlan.rootScheme->curScheme);
            }
        }

        execPlan.iLength = rootPlanData.length;
        execPlan.oLength
            = rootPlanData.outputLength.empty() ? rootPlanData.length : rootPlanData.outputLength;

        // setup isUnitStride values
        execPlan.rootPlan->inStrideUnit  = BufferIsUnitStride(execPlan, OB_USER_IN);
        execPlan.rootPlan->outStrideUnit = BufferIsUnitStride(execPlan, OB_USER_OUT);

        // set load/store ops on the root plan
        if(loadOps.enabled())
            execPlan.rootPlan->loadOps = loadOps;
        if(storeOps.enabled())
            execPlan.rootPlan->storeOps = storeOps;

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
static size_t C2CBrickOneDimension(rocfft_plan_t&             plan,
                                   size_t                     dimIdx,
                                   rocfft_location_t          location,
                                   const std::vector<size_t>& lengths,
                                   const std::vector<size_t>& stride,
                                   BufferPtr                  input,
                                   BufferPtr                  output,
                                   const std::vector<size_t>& antecedents)
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
    rootPlanData.precision    = plan.precision;
    rootPlanData.inArrayType  = rocfft_array_type_complex_interleaved;
    rootPlanData.outArrayType = rocfft_array_type_complex_interleaved;
    rootPlanData.deviceProp   = get_curr_device_prop();

    auto singlePlan       = BuildSingleDevicePlan(rootPlanData,
                                            plan.get_local_comm_rank(),
                                            location,
                                            plan.transformType,
                                            plan.desc.loadOps,
                                            plan.desc.storeOps,
                                            true);
    singlePlan->mgpuPlan  = true;
    singlePlan->inputPtr  = input;
    singlePlan->outputPtr = output;
    return plan.AddMultiPlanItem(std::move(singlePlan), antecedents);
}

void rocfft_plan_t::C2CField(const rocfft_field_t&      field,
                             const std::vector<size_t>& fftDims,
                             std::vector<BufferPtr>&    input,
                             std::vector<BufferPtr>&    output,
                             const std::vector<size_t>& inputAntecedents,
                             std::vector<size_t>&       outputItems)
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
            auto transformItem              = C2CBrickOneDimension(*this,
                                                      dimIdx,
                                                      inBrick.location,
                                                      inBrick.length(),
                                                      inBrick.stride,
                                                      fftInput,
                                                      output[i],
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

    for(size_t i = 0; i < out.bricks.size(); ++i)
    {
        auto& outBrick = out.bricks[i];

        // start lower and upper at origin and max, respectively
        std::fill(outBrick.lower.begin(), outBrick.lower.end(), 0);
        outBrick.upper = length;

        // divide up the split dim
        outBrick.lower[*splitDim] = length[*splitDim] / out.bricks.size() * i;
        // last brick needs to include the whole length
        if(i == out.bricks.size() - 1)
            outBrick.upper[*splitDim] = length[*splitDim];
        else
            outBrick.upper[*splitDim] = length[*splitDim] / out.bricks.size() * (i + 1);

        auto brickLength = outBrick.length();

        // set strides - contiguous dim has stride 1
        size_t dist             = 1;
        outBrick.stride[dimIdx] = dist;
        dist *= brickLength[dimIdx];
        // split dim is contiguous after that
        outBrick.stride[*splitDim] = dist;
        dist *= brickLength[*splitDim];
        // fill in remaining strides
        for(size_t s = 0; s < outBrick.stride.size(); ++s)
        {
            if(s == dimIdx || s == *splitDim)
                continue;
            outBrick.stride[s] = dist;
            dist *= brickLength[s];
        }
    }
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

    const auto local_comm_rank = get_local_comm_rank();

    // loop over each input brick, finding the intersection of it with
    // every output brick
    for(size_t inBrickIdx = 0; inBrickIdx < inField.bricks.size(); ++inBrickIdx)
    {
        const auto& inBrick = inField.bricks[inBrickIdx];
        for(size_t outBrickIdx = 0; outBrickIdx < outField.bricks.size(); ++outBrickIdx)
        {
            const auto& outBrick = outField.bricks[outBrickIdx];

            auto intersection = inBrick.intersect(outBrick);
            if(intersection.empty())
                continue;
            intersection.stride = intersection.contiguous_strides();

            // pack data for communication
            packBufs.reserve(packBufs.size() + 2);
            TempBufferLease& pack = packBufs.emplace_back(tempBuffers,
                                                          local_comm_rank,
                                                          inBrick.location,
                                                          intersection.count_elems(),
                                                          elem_size);
            TempBufferLease& recv = packBufs.emplace_back(tempBuffers,
                                                          local_comm_rank,
                                                          outBrick.location,
                                                          intersection.count_elems(),
                                                          elem_size);
            auto             packIdx
                = AddMultiPlanItem(transpose_brick(local_comm_rank,
                                                   inBrick.location,
                                                   intersection.length(),
                                                   precision,
                                                   desc.inArrayType,
                                                   input[inBrickIdx],
                                                   intersection.offset_in_field(inBrick.stride)
                                                       - inBrick.offset_in_field(inBrick.stride),
                                                   inBrick.stride,
                                                   BufferPtr::temp(pack.data()),
                                                   0,
                                                   intersection.stride,
                                                   "pack brick for global transpose"),
                                   {inputAntecedents[inBrickIdx]});
            multiPlan[packIdx]->group = itemGroup;
            multiPlan[packIdx]->description
                = "pack " + std::to_string(inBrickIdx) + " + " + std::to_string(outBrickIdx);

            // send packed data
            auto sendOp = std::make_unique<CommPointToPoint>(local_comm_rank,
                                                             precision,
                                                             desc.inArrayType,
                                                             intersection.count_elems(),
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
                                                   intersection.length(),
                                                   precision,
                                                   desc.inArrayType,
                                                   BufferPtr::temp(recv.data()),
                                                   0,
                                                   intersection.stride,
                                                   output[outBrickIdx],
                                                   intersection.offset_in_field(outBrick.stride)
                                                       - outBrick.offset_in_field(outBrick.stride),
                                                   outBrick.stride,
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
    const auto local_comm_rank = get_local_comm_rank();
    const auto local_comm_size = get_local_comm_size();

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

            auto intersection = inBrick.intersect(outBrick);
            if(intersection.empty())
                continue;

            conn.add_connection(inRank, outRank);

            const auto elems = intersection.count_elems();

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
                             cumulative_send_elems,
                             elem_size);
    TempBufferLease recv_buf(tempBuffers,
                             local_comm_rank,
                             {local_comm_rank, *local_recv_device},
                             cumulative_recv_elems,
                             elem_size);

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

            auto intersection = inBrick.intersect(outBrick);
            if(intersection.empty())
                continue;
            intersection.stride = intersection.contiguous_strides();

            // this transpose will only actually run if inRank ==
            // local_comm_rank, but adding it to the plan so all
            // ranks see a consistent plan.
            //
            // if(inRank == local_comm_rank)
            {
                auto pack_op = AddMultiPlanItem(
                    transpose_brick(local_comm_rank,
                                    inBrick.location,
                                    intersection.length(),
                                    precision,
                                    desc.inArrayType,
                                    input[inBrickIdx],
                                    intersection.offset_in_field(inBrick.stride)
                                        - inBrick.offset_in_field(inBrick.stride),
                                    inBrick.stride,
                                    BufferPtr::temp(send_buf.data()),
                                    send_offsets[outRank],
                                    intersection.stride,
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
                auto unpack_op = AddMultiPlanItem(
                    transpose_brick(local_comm_rank,
                                    outBrick.location,
                                    intersection.length(),
                                    precision,
                                    desc.inArrayType,
                                    BufferPtr::temp(recv_buf.data()),
                                    recv_offsets[inRank],
                                    intersection.stride,
                                    output[outBrickIdx],
                                    intersection.offset_in_field(outBrick.stride)
                                        - outBrick.offset_in_field(outBrick.stride),
                                    outBrick.stride,
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
        subcomm = make_subcommunicator(mpi_comm, comm_ranks_vec);
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
        if(b.lower.size() >= 1)
            xset.insert(b.lower[0]);
        if(b.lower.size() >= 2)
            yset.insert(b.lower[1]);
        if(b.lower.size() >= 3)
            zset.insert(b.lower[2]);
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

    rocfft_field_t out = currentField;
    out.bricks.resize(n_bricks);

    const int ndim = static_cast<int>(lengthsWithBatch.size());
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
        auto& brick = out.bricks[i];

        // compute 2D (p, q) indices for this brick
        int p = i / Q;
        int q = i % Q;

        // set bounds for each axis
        for(int d = 0; d < ndim; ++d)
        {
            brick.lower[d] = 0;
            brick.upper[d] = lengthsWithBatch[d];
        }

        int axis_p = split_axes[0];
        int axis_q = split_axes[1];

        int len_p           = lengthsWithBatch[axis_p];
        int len_q           = lengthsWithBatch[axis_q];
        brick.lower[axis_p] = len_p * p / P;
        brick.upper[axis_p] = len_p * (p + 1) / P;
        brick.lower[axis_q] = len_q * q / Q;
        brick.upper[axis_q] = len_q * (q + 1) / Q;

        // contiguous strides
        const auto brickLength = brick.length();
        int        dist        = 1;
        for(size_t s = 0; s < brick.stride.size(); ++s)
        {
            brick.stride[s] = dist;
            dist *= brickLength[s];
        }

        // assign device/rank in a round-robin fashion
        const auto& ref_brick = currentField.bricks[i % currentField.bricks.size()];
        brick.location        = ref_brick.location;
    }

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
    const auto local_comm_rank = get_local_comm_rank();

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
    for(size_t dimIdx = 0; dimIdx < rank; ++dimIdx)
    {
        if(!DimensionSplitInField(lengths[dimIdx], dimIdx, desc.inFields.front()))
            contiguousInputDims.push_back(dimIdx);
        else if(!DimensionSplitInField(lengths[dimIdx], dimIdx, desc.outFields.front()))
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
        inputTemp.emplace_back(
            tempBuffers, local_comm_rank, inBrick.location, inBrick.count_elems(), elem_size);
        inputFFTBufs.emplace_back(BufferPtr::temp(inputTemp.back().data()));
    }

    std::vector<BufferPtr> outputBufs
        = GatherUserBuffers(BufferPtr::user_output, desc.outFields.front().bricks);

    // plan FFTs along already contiguous dimensions
    std::vector<size_t> inputFFTItems;
    C2CField(
        desc.inFields.front(), contiguousInputDims, inputBufs, inputFFTBufs, {}, inputFFTItems);

    auto lengthsWithBatch = lengths;
    lengthsWithBatch.push_back(batch);

    // track which dimensions have already been FFTed
    std::vector<int> fft_done(rank, 0);
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
    if(num_split_dims_in >= 2 && num_split_dims_out >= 2 && rank == 3)
    {
        get_transpose_plan(in_grid,
                           out_grid,
                           {lengths[0], lengths[1], lengths[2]},
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
                                                nextField.bricks[b].count_elems(),
                                                elem_size);
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
            std::vector<size_t> fftItems;
            C2CField(currentField,
                     {static_cast<size_t>(pencil_axis)},
                     writeToUserOutput ? outputBufs : currentBufs,
                     writeToUserOutput ? outputBufs : currentBufs,
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
                    tempBuffers, local_comm_rank, b.location, b.count_elems(), elem_size);
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
            C2CField(transposedField,
                     {dimIdx},
                     transposeOutputBufs,
                     transposeOutputBufs,
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
        C2CField(desc.outFields.front(),
                 contiguousOutputDims,
                 outputBufs,
                 outputBufs,
                 finalTransposeItems,
                 finalFFTItems);
    }

    return true;
}

// All-gather all of the brick parameters for a given field.
rocfft_status allgather_brick_params_lus_mpi(rocfft_plan&    plan,
                                             rocfft_field_t& field,
                                             const size_t    global_brick_length)
{
#if !defined ROCFFT_MPI_ENABLE
    return rocfft_status_failure;
#else
    auto rcmpi = MPI_SUCCESS;

    // TODO: make async.

    // First, compute the number of bricks per field.
    std::vector<int> global_brick_count(plan->get_local_comm_size());
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
                              plan->mpi_comm);
        if(rcmpi != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Allgather failed: " + std::to_string(rcmpi));
        }
    }

    // We need the displacments (as an int[]) for MPI_Allgatherv:
    typeof(global_brick_count) displacements(plan->get_local_comm_size());
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

    std::vector<typeof(decltype(rocfft_brick_t::lower)::value_type)> global_lowers(
        num_global_bricks * global_brick_length);
    std::vector<typeof(decltype(rocfft_brick_t::upper)::value_type)> global_uppers(
        num_global_bricks * global_brick_length);
    std::vector<typeof(decltype(rocfft_brick_t::stride)::value_type)> global_strides(
        num_global_bricks * global_brick_length);
    std::vector<decltype(rocfft_location_t::device)>    global_devices(num_global_bricks);
    std::vector<decltype(rocfft_location_t::comm_rank)> global_comm_ranks(num_global_bricks);

    // Make a contiguous copy for the MPI routine:
    std::vector<typeof(decltype(rocfft_brick_t::lower)::value_type)>  local_lowers;
    std::vector<typeof(decltype(rocfft_brick_t::upper)::value_type)>  local_uppers;
    std::vector<typeof(decltype(rocfft_brick_t::stride)::value_type)> local_strides;
    std::vector<int>                                                  local_devices;
    std::vector<int>                                                  local_comm_ranks;
    local_lowers.reserve(local_brick_count * global_brick_length);
    local_uppers.reserve(local_brick_count * global_brick_length);
    local_strides.reserve(local_brick_count * global_brick_length);
    for(const auto& brick : field.bricks)
    {
        for(auto v : brick.lower)
        {
            local_lowers.push_back(v);
        }
        for(auto v : brick.upper)
        {
            local_uppers.push_back(v);
        }
        for(auto v : brick.stride)
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

    // We must send and recieve the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_lowers)::value_type)>,
                       std::underlying_type<typeof(decltype(global_lowers)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_lowers.data(),
                           recvcount[plan->get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_lowers)::value_type)>(),
                           global_lowers.data(),
                           recvcount.data(),
                           displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_lowers)::value_type)>(),
                           plan->mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    // We must send and recieve the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_uppers)::value_type)>,
                       std::underlying_type<typeof(decltype(global_uppers)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_uppers.data(),
                           recvcount[plan->get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_uppers)::value_type)>(),
                           global_uppers.data(),
                           recvcount.data(),
                           displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_uppers)::value_type)>(),
                           plan->mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    // We must send and recieve the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_strides)::value_type)>,
                       std::underlying_type<typeof(decltype(global_strides)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_strides.data(),
                           recvcount[plan->get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_strides)::value_type)>(),
                           global_strides.data(),
                           recvcount.data(),
                           displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_strides)::value_type)>(),
                           plan->mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    // We must send and recieve the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_devices)::value_type)>,
                       std::underlying_type<typeof(decltype(global_devices)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_devices.data(),
                           global_brick_count[plan->get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_devices)::value_type)>(),
                           global_devices.data(),
                           global_brick_count.data(),
                           scalar_displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_devices)::value_type)>(),
                           plan->mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    // We must send and recieve the same type:
    static_assert(
        std::is_same_v<std::underlying_type<typeof(decltype(local_comm_ranks)::value_type)>,
                       std::underlying_type<typeof(decltype(global_comm_ranks)::value_type)>>);
    rcmpi = MPI_Allgatherv(local_comm_ranks.data(),
                           global_brick_count[plan->get_local_comm_rank()],
                           type_to_mpi_type<typeof(decltype(local_comm_ranks)::value_type)>(),
                           global_comm_ranks.data(),
                           global_brick_count.data(),
                           scalar_displacements.data(),
                           type_to_mpi_type<typeof(decltype(global_comm_ranks)::value_type)>(),
                           plan->mpi_comm);
    if(rcmpi != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Allgatherv failed: " + std::to_string(rcmpi));
    }

    field.bricks.clear();
    field.bricks.reserve(num_global_bricks);
    for(std::remove_const_t<decltype(num_global_bricks)> ibrick = 0; ibrick < num_global_bricks;
        ++ibrick)
    {
        // Add bricks one by one.
        rocfft_brick brick = nullptr;
        auto         rcfft = rocfft_brick_create(&brick,
                                         global_lowers.data() + ibrick * global_brick_length,
                                         global_uppers.data() + ibrick * global_brick_length,
                                         global_strides.data() + ibrick * global_brick_length,
                                         global_brick_length,
                                         global_devices[ibrick]); // device id
        if(rcfft != rocfft_status_success)
        {
            return rcfft;
        }
        // It should be possible to get the brick comm rank from global_brick_count, but for now we
        // can just communicate it and set it.
        brick->location.comm_rank = global_comm_ranks[ibrick];
        rocfft_field_add_brick(&field, brick);
        rocfft_brick_destroy(brick);
        brick = nullptr;
    }

    return rocfft_status_success;
#endif
}

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

rocfft_status allgather_brick_params_mpi(rocfft_plan& plan)
{
#if !defined ROCFFT_MPI_ENABLE
    return rocfft_status_failure;
#else
    if(!plan->mpi_comm)
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
    for(auto& ifield : plan->desc.inFields)
    {
        for(auto& ibrick : ifield.bricks)
        {
            ibrick.location.comm_rank = plan->get_local_comm_rank();
            if(local_bricklength == 0)
            {
                local_bricklength = ibrick.lower.size();
            }
            else
            {
                if(local_bricklength != ibrick.lower.size())
                {
                    return rocfft_status_failure;
                }
            }
        }
    }
    for(auto& ofield : plan->desc.outFields)
    {
        for(auto& obrick : ofield.bricks)
        {
            obrick.location.comm_rank = plan->get_local_comm_rank();
            if(local_bricklength == 0)
            {
                local_bricklength = obrick.lower.size();
            }
            else
            {
                if(local_bricklength != obrick.lower.size())
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
        std::vector<decltype(local_bricklength)> all_brick_lengths(plan->get_local_comm_size());
        rcmpi = MPI_Allgather(&local_bricklength,
                              1,
                              type_to_mpi_type<typeof(local_bricklength)>(),
                              all_brick_lengths.data(),
                              1,
                              type_to_mpi_type<typeof(decltype(all_brick_lengths)::value_type)>(),
                              plan->mpi_comm);
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
        if(!all_mpi_ranks_same_scalar(plan->desc.inFields.size(), plan->mpi_comm))
            return rocfft_status_failure;
        if(!all_mpi_ranks_same_scalar(plan->desc.outFields.size(), plan->mpi_comm))
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
    for(auto& field : plan->desc.inFields)
    {
        const auto rcfft = allgather_brick_params_lus_mpi(plan, field, global_brick_length);
        if(rcfft != rocfft_status_success)
        {
            return rcfft;
        }
    }
    for(auto& field : plan->desc.outFields)
    {
        const auto rcfft = allgather_brick_params_lus_mpi(plan, field, global_brick_length);
        if(rcfft != rocfft_status_success)
        {
            return rcfft;
        }
    }

    // TODO: if some ranks do not have any input (or output), then we should make a
    // sub-communicator.  Need to figure out how to deal with completion though.

    return rocfft_status_success;
#endif
}

void rocfft_plan_t::ValidateFields() const
{
    auto validateField = [](const char*                type,
                            const std::vector<size_t>& lengths,
                            size_t                     batch,
                            const rocfft_field_t&      field) {
        // make sure all bricks have enough dimensions (FFT + batch dimensions)
        for(const auto& b : field.bricks)
        {
            if(b.lower.size() != lengths.size() + 1 || b.upper.size() != lengths.size() + 1
               || b.stride.size() != lengths.size() + 1)
                throw std::runtime_error(std::string(type)
                                         + " brick has wrong number of dimensions, expected "
                                         + std::to_string(lengths.size() + 1));
        }

        // construct a brick that covers the whole field
        rocfft_brick_t whole_field;
        whole_field.lower.resize(lengths.size() + 1);
        whole_field.upper = lengths;
        whole_field.upper.push_back(batch);

        // ensure no bricks in the field overlap - check each brick
        // against each other brick
        for(auto brickI = field.bricks.begin(); brickI != field.bricks.end(); ++brickI)
        {
            for(auto brickJ = brickI + 1; brickJ != field.bricks.end(); ++brickJ)
            {
                if(!brickI->intersect(*brickJ).empty())
                {
                    throw std::runtime_error(std::string(type) + " brick " + brickI->str()
                                             + " overlaps with brick " + brickJ->str());
                }
            }

            // ensure each brick is within the field
            if(!brickI->equal_coords(brickI->intersect(whole_field)))
                throw std::runtime_error(std::string(type) + " brick " + brickI->str()
                                         + " is not within field");
        }

        // check that the bricks cover the whole index space
        const size_t whole_field_elems = whole_field.count_elems();

        // add up total number of elements in all bricks,
        // should be the same since we know there are no
        // overlaps
        size_t total_brick_elems = 0;
        for(const auto& b : field.bricks)
            total_brick_elems += b.count_elems();
        if(whole_field_elems != total_brick_elems)
            throw std::runtime_error(std::string(type) + " field has "
                                     + std::to_string(whole_field_elems) + " elems but bricks have "
                                     + std::to_string(total_brick_elems) + " elems");
    };

    if(!desc.inFields.empty())
        validateField("input", lengths, batch, desc.inFields.front());
    if(!desc.outFields.empty())
        validateField("output", outputLengths, batch, desc.outFields.front());

    // multi-process transforms must use fields for both input and output
    if(desc.comm_type != rocfft_comm_none && (desc.inFields.empty() || desc.outFields.empty()))
        throw std::runtime_error(
            "multi-process transforms require both input and output fields to be specified");
}

int rocfft_plan_t::get_local_comm_rank() const
{
#ifdef ROCFFT_MPI_ENABLE
    if(desc.comm_type == rocfft_comm_none || !mpi_comm)
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

int rocfft_plan_t::get_local_comm_size() const
{
#ifdef ROCFFT_MPI_ENABLE
    if(desc.comm_type == rocfft_comm_none || !mpi_comm)
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
    if(dimensions > 3)
        return rocfft_status_invalid_dimensions;

    try
    {
        plan->rank = dimensions;
        std::copy(lengths, lengths + dimensions, std::back_inserter(plan->lengths));
        plan->batch         = number_of_transforms;
        plan->placement     = placement;
        plan->precision     = precision;
        plan->transformType = transform_type;

        plan->outputLengths = plan->lengths;
        if(transform_type == rocfft_transform_type_real_forward
           || transform_type == rocfft_transform_type_real_inverse)
        {
            plan->outputLengths.front() = plan->outputLengths.front() / 2 + 1;
        }
        if(transform_type == rocfft_transform_type_real_inverse)
            std::swap(plan->outputLengths, plan->lengths);

        if(description != nullptr)
        {
            plan->desc = *description;
        }
        plan->desc.init_defaults(
            plan->transformType, plan->placement, plan->lengths, plan->outputLengths);

        auto rcfft = rocfft_status_success;

#ifdef ROCFFT_MPI_ENABLE
        if(plan->desc.comm_type == rocfft_comm_mpi)
        {
            // duplicate the user-provided MPI communicator and set
            // our own error policy on it
            if(plan->desc.user_mpi_comm)
            {
                plan->mpi_comm.duplicate(plan->desc.user_mpi_comm);
                auto mpiret = MPI_Comm_set_errhandler(plan->mpi_comm, MPI_ERRORS_RETURN);
                if(mpiret != MPI_SUCCESS)
                    return rocfft_status_failure;
            }

            // Each rank needs to know the global data distribution for plan generation, so we
            // gather all the brick information here.
            rcfft = allgather_brick_params_mpi(plan);
            if(rcfft != rocfft_status_success)
                throw std::runtime_error("gather brick params failed");
        }
#endif

        // Sort the parameters to be row major, in case they're not
        plan->sort();

        rcfft = check_array_type_validity(plan);
        if(rcfft != rocfft_status_success)
            return rcfft;

        log_bench(rocfft_bench_command(plan));

        // Construct the plan

        // Build an optimized multi-device plan, if possible
        plan->ValidateFields();

        // If we have no input/output fields, then the single ExecPlan is
        // exactly what we need to do.
        // FIXME: this check should actually be single-brick/no bricks.
        if(plan->desc.inFields.empty() && plan->desc.outFields.empty())
        {
            NodeMetaData rootPlanData(nullptr);
            set_rootplan_params(plan, rootPlanData);
            rootPlanData.deviceProp = get_curr_device_prop();
            set_bluestein_strides(plan, rootPlanData);

            auto singleDevicePlan = BuildSingleDevicePlan(rootPlanData,
                                                          0,
                                                          rocfft_location_t::rank0_current_device(),
                                                          plan->transformType,
                                                          plan->desc.loadOps,
                                                          plan->desc.storeOps,
                                                          false);
            plan->AddMultiPlanItem(std::move(singleDevicePlan), {});
        }
        else
        {
            if(!plan->BuildOptMultiDevicePlan())
            {
                // If optimized multi-device was not possible (either because
                // multi-device was not requested, or we can't optimize for
                // that case), fall back to single-device plan

                NodeMetaData rootPlanData(nullptr);
                set_rootplan_params(plan, rootPlanData);
                rootPlanData.deviceProp = get_curr_device_prop();
                set_bluestein_strides(plan, rootPlanData);

                auto singleDevicePlan
                    = BuildSingleDevicePlan(rootPlanData,
                                            0,
                                            rocfft_location_t::rank0_current_device(),
                                            plan->transformType,
                                            plan->desc.loadOps,
                                            plan->desc.storeOps,
                                            true);

                plan->GatherScatterSingleDevicePlan(std::move(singleDevicePlan));
            }
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

    *size_in_bytes = plan->WorkBufBytes();
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

    rocfft_cout << "dimensions: " << plan->rank << std::endl;

    rocfft_cout << "lengths: " << plan->lengths[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->lengths[i];
    rocfft_cout << std::endl;
    rocfft_cout << "batch size: " << plan->batch << std::endl;
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

    rocfft_cout << "input strides: " << plan->desc.inStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->desc.inStrides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "output strides: " << plan->desc.outStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->desc.outStrides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "input distance: " << plan->desc.inDist << std::endl;
    rocfft_cout << "output distance: " << plan->desc.outDist << std::endl;
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
    static constexpr char v[] = ROCFFT_VERSION_STRING;
    if(!buf)
        return rocfft_status_failure;
    if(len < sizeof(v))
        return rocfft_status_invalid_arg_value;
    memcpy(buf, v, sizeof(v));
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

    // overriden by each derived class
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
            fuse->ForEachNode([=](TreeNode* node) { this->RecursiveRemoveNode(node); });
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
                                    || scheme == CS_REAL_3D_EVEN;
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
    loadOps.print(os, indentStr);
    storeOps.print(os, indentStr);

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

TreeNode* TreeNode::GetPlanRoot()
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
       || parent->scheme == CS_REAL_3D_EVEN)
        return parent;

    // Otherwise keep looking up the tree
    return parent->GetRealEvenAncestor();
}

TreeNode* TreeNode::GetPartialPassAncestor() const
{
    if(!parent)
        return nullptr;

    if(parent->scheme == CS_3D_PP)
        return parent;

    return parent->GetPartialPassAncestor();
}

bool TreeNode::IsRootPlanC2CTransform()
{
    auto root = GetPlanRoot();
    return (root->inArrayType != rocfft_array_type_real)
           && (root->outArrayType != rocfft_array_type_real);
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

        // When tuning, we're runing through each bench
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

        // Keep the references, and after buffer-assignment and colapse-batch-dim,
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
        // means this is nothing to do with solution map. Throw to terminate
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

    if(execPlan.rootPlan->loadOps.enabled())
    {
        // Load ops happen on first node that reads input
        auto load_node = std::find_if(
            execPlan.execSeq.begin(), execPlan.execSeq.end(), [&execPlan](TreeNode* node) {
                return node->obIn == execPlan.rootPlan->obIn;
            });
        (*load_node)->loadOps = execPlan.rootPlan->loadOps;
    }

    if(execPlan.rootPlan->storeOps.enabled())
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
