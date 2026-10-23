// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

std::string rocfft_brick_t::str() const
{
    return layout.str() + " " + location.str();
}

// internal brick-adding API
void rocfft_field_t::add_brick_internal(rocfft_field_t& field, const rocfft_brick_t& brick)
{
    if(!field.empty() && !brick.layout.is_dimensionally_consistent_with(field.get_brick(0).layout))
    {
        // could motivate a new "rocfft_status_invalid_brick{_dimensions}" error code
        throw std::runtime_error(
            "Brick to be added is not dimensionally consistent with other brick(s) in field.");
    }

    field.bricks.emplace_back(brick);
}

rocfft_field_t::rocfft_field_t(std::vector<rocfft_brick_t> field_bricks)
    : bricks(std::move(field_bricks))
{
    if(bricks.empty())
    {
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + " cannot operate on empty fields.");
    }
    const auto array_type = bricks[0].layout.get_array_type();
    if(array_type == rocfft_array_type_unset
       || std::any_of(bricks.begin(), bricks.end(), [this, array_type](const rocfft_brick_t& b) {
              return !b.layout.has_precision() || b.layout.get_array_type() != array_type
                     || b.layout.get_precision() != bricks[0].layout.get_precision();
          }))
    {
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires bricks with set and consistent array types "
                                      "and precisions");
    }
    // compute field's global range: set global lengths and batch for the field based
    // on the bricks's supremum
    const auto          len_rank   = bricks[0].layout.get_len_rank();
    const auto          batch_rank = bricks[0].layout.get_batch_rank();
    std::vector<size_t> global_lengths(len_rank);
    std::vector<size_t> global_batches(batch_rank);
    for(const auto& brick : bricks)
    {
        if(brick.layout.get_len_rank() != len_rank || brick.layout.get_batch_rank() != batch_rank)
        {
            throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                        + " requires dimensionally-consistent bricks");
        }
        const auto brick_upper = brick.layout.upper();
        for(size_t idx = 0; idx < brick_upper.size(); idx++)
        {
            if(idx < len_rank)
                global_lengths[idx] = std::max(global_lengths[idx], brick_upper[idx]);
            else
                global_batches[idx - len_rank]
                    = std::max(global_batches[idx - len_rank], brick_upper[idx]);
        }
    }
    // note: strides are irrelevant for the global data range.
    // Default contiguous strides are used here for simplicity.
    global_data_range = data_descriptor_t::default_full_layout(global_lengths, global_batches);

    // verify consistency and validity of the bricks as a field tessellation
    if(!has_valid_tessellation())
    {
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " detected an invalid list of bricks (incomplete or "
                                      "overlapping coverage of the field's global data range)");
    }
    for(size_t dim = 0; dim < global_data_range.get_len_rank(); dim++)
    {
        if(std::all_of(bricks.begin(), bricks.end(), [&dim, this](const rocfft_brick_t& b) {
               return b.layout[dim].has_same_logical_range_as(global_data_range[dim]);
           }))
        {
            undistributed_length_dimensions.insert(dim);
        }
    }

    // make sure bricks are sorted by increasing ranks (without modifying
    // brick ordering within ranks)
    std::stable_sort(
        bricks.begin(), bricks.end(), [](const rocfft_brick_t& a, const rocfft_brick_t& b) {
            return a.location.comm_rank < b.location.comm_rank;
        });
}

bool rocfft_field_t::has_valid_tessellation() const
{
    size_t total_brick_logical_count = 0;
    for(auto brickI = bricks.begin(); brickI != bricks.end(); ++brickI)
    {
        for(auto brickJ = brickI + 1; brickJ != bricks.end(); ++brickJ)
        {
            const auto ij_intersection = data_descriptor_t::make_contiguous_intersection_of(
                brickI->layout, brickJ->layout);
            if(!ij_intersection.is_empty())
                return false;
        }
        // Add up the brick's logical count
        total_brick_logical_count += brickI->layout.logical_count();
    }

    // The bricks cover the whole index space iff their total number of
    // logical elements is the same as the full data layout
    return global_data_range.logical_count() == total_brick_logical_count;
}

void rocfft_field_t::erase_unit_length_axis(size_t len_dim)
{
    if(len_dim >= global_data_range.get_len_rank())
        throw std::out_of_range(ROCFFT_CURRENT_FUNCTION
                                + " cannot erase a length dimension that is out of bounds");
    if(global_data_range[len_dim].logical_span() != 1)
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " cannot erase a length dimension that is not of length 1");
    if(!undistributed_length_dimensions.contains(len_dim))
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " cannot erase a length dimension that is distributed");

    for(auto& brick : bricks)
    {
        brick.layout.len_axes.erase(brick.layout.len_axes.begin() + len_dim);
    }
    global_data_range.len_axes.erase(global_data_range.len_axes.begin() + len_dim);
    std::set<size_t> new_undistributed_length_dimensions;
    for(auto& dim : undistributed_length_dimensions)
    {
        if(dim < len_dim)
            new_undistributed_length_dimensions.insert(dim);
        else if(dim > len_dim)
            new_undistributed_length_dimensions.insert(dim - 1);
    }
    std::swap(undistributed_length_dimensions, new_undistributed_length_dimensions);
}

void rocfft_field_t::reorder_length_axes(const std::vector<size_t>& new_order)
{
    std::vector<size_t> current_order(global_data_range.get_len_rank());
    std::iota(current_order.begin(), current_order.end(), 0);
    if(!std::is_permutation(
           new_order.begin(), new_order.end(), current_order.begin(), current_order.end()))
    {
        throw std::invalid_argument("Invalid re-ordering of length axes requested to "
                                    + ROCFFT_CURRENT_FUNCTION);
    }
    // re-order lengths as requested across all members
    for(auto& brick : bricks)
    {
        brick.layout.reorder_length_axes(new_order);
    }
    global_data_range.reorder_length_axes(new_order);
    std::set<size_t> new_undistributed_length_dimensions;
    for(auto len_dim : undistributed_length_dimensions)
    {
        auto new_len_dim = std::distance(new_order.begin(),
                                         std::find(new_order.begin(), new_order.end(), len_dim));
        new_undistributed_length_dimensions.insert(new_len_dim);
    }
    std::swap(undistributed_length_dimensions, new_undistributed_length_dimensions);
}

std::optional<rocfft_field_t> rocfft_field_t::get_other_embarrassingly_parallel_io_field(
    io_data_label           other_io,
    rocfft_transform_type   fft_type,
    rocfft_result_placement placement,
    bool                    other_innermost_length_is_odd) const
{
    if(bricks.empty())
    {
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " cannot operate on empty fields");
    }

    if(!has_undistributed_length_axes())
    {
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " requires all bricks in the field to have undistributed lengths");
    }
    // copy bricks
    std::vector<rocfft_brick_t> ret_bricks = bricks;
    // modify/set brick layouts as needed
    for(size_t brick_idx = 0; brick_idx < bricks.size(); brick_idx++)
    {
        const auto& brick = bricks[brick_idx];
        if(placement == rocfft_placement_inplace)
        {
            const auto tmp = brick.layout.get_other_inplace_layout_for(
                other_io, fft_type, other_innermost_length_is_odd);
            if(!tmp)
                return std::nullopt; // in-place cannot be done on that brick
            ret_bricks[brick_idx].layout = *tmp;
        }
        else
        {
            // use default, packed contiguous layout for the other I/O
            if(is_hermitian_domain(fft_type, other_io))
            {
                ret_bricks[brick_idx].layout[0].upper   = brick.layout[0].upper / 2 + 1;
                ret_bricks[brick_idx].layout.array_type = rocfft_array_type_hermitian_interleaved;
            }
            else if(is_real_domain(fft_type, other_io))
            {
                ret_bricks[brick_idx].layout[0].upper
                    = 2 * (brick.layout[0].upper - 1) + (other_innermost_length_is_odd ? 1 : 0);
                ret_bricks[brick_idx].layout.array_type = rocfft_array_type_real;
            }
            ret_bricks[brick_idx].layout[0].inbuffer_stride = 1;
            for(size_t dim = 1; dim < brick.layout.get_full_rank(); dim++)
            {
                ret_bricks[brick_idx].layout[dim].inbuffer_stride
                    = ret_bricks[brick_idx].layout[dim - 1].inbuffer_stride
                      * ret_bricks[brick_idx].layout[dim - 1].logical_span();
            }
        }
    }
    return rocfft_field_t{std::move(ret_bricks)};
}

rocfft_array_type rocfft_field_t::get_array_type() const
{
    return get_brick(0).layout.get_array_type();
}

rocfft_precision rocfft_field_t::get_precision() const
{
    return get_brick(0).layout.get_precision();
}

void rocfft_field_t::reinterpret_as(rocfft_array_type new_array_type)
{
    // relabel copies first so that a failure leaves this field unchanged
    std::vector<rocfft_brick_t> relabeled_bricks;
    relabeled_bricks.reserve(bricks.size());
    for(const auto& brick : bricks)
        relabeled_bricks.emplace_back(brick.layout.reinterpret_as(new_array_type), brick.location);
    std::swap(bricks, relabeled_bricks);
}

rocfft_field_t rocfft_field_t::get_field_for_lengths(const std::set<size_t>& len_dims) const
{
    std::vector<rocfft_brick_t> embedded_bricks;
    embedded_bricks.reserve(bricks.size());
    for(const auto& brick : bricks)
        embedded_bricks.emplace_back(brick.layout.get_layout_for_len_axes(len_dims),
                                     brick.location);
    rocfft_field_t embedded_field(std::move(embedded_bricks));
    // Complex-conjugate symmetry relationships no longer hold for sub-transforms that
    // exclude the innermost (halved) axis
    if(embedded_field.get_array_type() == rocfft_array_type_hermitian_interleaved
       && !len_dims.contains(0) && len_dims.size() < global_data_range.get_len_rank())
    {
        embedded_field.reinterpret_as(rocfft_array_type_complex_interleaved);
    }
    return embedded_field;
}

rocfft_field_t rocfft_field_t::get_embedding_field() const
{
    if(std::none_of(bricks.begin(), bricks.end(), [](const rocfft_brick_t& brick) {
           return brick.layout.is_embedded();
       }))
    {
        return *this;
    }
    std::vector<rocfft_brick_t> embedding_bricks;
    embedding_bricks.reserve(bricks.size());
    for(const auto& brick : bricks)
        embedding_bricks.emplace_back(brick.layout.get_embedding_layout(), brick.location);
    rocfft_field_t embedding_field(std::move(embedding_bricks));
    // Complex-conjugate symmetry relationships no longer hold upon re-interpretation into
    // a higher-dimensional data set
    if(embedding_field.get_array_type() == rocfft_array_type_hermitian_interleaved)
        embedding_field.reinterpret_as(rocfft_array_type_complex_interleaved);
    return embedding_field;
}

bool rocfft_field_t::is_same_up_to_hermitian_relabel(const rocfft_field_t& other) const
{
    const auto relabeled = [](rocfft_field_t field) {
        if(field.get_array_type() == rocfft_array_type_hermitian_interleaved)
            field.reinterpret_as(rocfft_array_type_complex_interleaved);
        else if(field.get_array_type() == rocfft_array_type_hermitian_planar)
            field.reinterpret_as(rocfft_array_type_complex_planar);
        return field;
    };
    return relabeled(*this) == relabeled(other);
}
