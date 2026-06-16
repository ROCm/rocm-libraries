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

#include "data_layout.h"
#include "../../shared/arithmetic.h"
#include "../../shared/ptrdiff.h"
#include "rocfft_current_function.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace
{
    template <typename T>
    std::vector<T> concatenate(const std::vector<T>& a, const std::vector<T>& b)
    {
        auto ret = a;
        std::copy(b.begin(), b.end(), std::back_inserter(ret));
        return ret;
    }
    template <typename T>
    std::vector<T> concatenate(const std::vector<T>& a, const T& b)
    {
        auto ret = a;
        ret.push_back(b);
        return ret;
    }
}

io_data_label other(io_data_label io)
{
    switch(io)
    {
    case io_data_label::INPUT:
        return io_data_label::OUTPUT;
    case io_data_label::OUTPUT:
        return io_data_label::INPUT;
    default:
        throw std::invalid_argument("Unknown io data label given to " + ROCFFT_CURRENT_FUNCTION);
    }
}

std::string to_str(io_data_label io)
{
    switch(io)
    {
    case io_data_label::INPUT:
        return "input";
    case io_data_label::OUTPUT:
        return "output";
    default:
        throw std::invalid_argument("Unknown io data label given to " + ROCFFT_CURRENT_FUNCTION);
    }
}

#define IO_DATA_LAYOUT_ACCESSOR_BODY(AXIS_IDX)                                        \
    do                                                                                \
    {                                                                                 \
        if(AXIS_IDX >= get_full_rank())                                               \
            throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION                       \
                                        + " accessing an out-of-range layout axis."); \
        return AXIS_IDX < len_axes.size() ? len_axes[AXIS_IDX]                        \
                                          : batch_axes[AXIS_IDX - len_axes.size()];   \
    } while(0)

const data_layout_t::axis_t& data_layout_t::operator[](size_t axis_idx) const
{
    IO_DATA_LAYOUT_ACCESSOR_BODY(axis_idx);
}

data_layout_t::axis_t& data_layout_t::operator[](size_t axis_idx)
{
    IO_DATA_LAYOUT_ACCESSOR_BODY(axis_idx);
}

data_layout_t::data_layout_t(const std::vector<size_t>& lower,
                             const std::vector<size_t>& upper,
                             const std::vector<size_t>& strides,
                             size_t                     batch_rank,
                             bool                       is_partial)
    : len_axes(batch_rank < lower.size() ? lower.size() - batch_rank : 0)
    , batch_axes(batch_rank)
{
    if(get_batch_rank() == 0 || get_len_rank() == 0)
        throw std::invalid_argument(
            ROCFFT_CURRENT_FUNCTION
            + " does not accept 0-dimensional length or batch coordinates.");
    if(lower.size() != upper.size() || lower.size() != strides.size())
        throw std::invalid_argument(
            ROCFFT_CURRENT_FUNCTION
            + " requires lower, upper, and strides to be of the same size.");

    for(size_t dim = 0; dim < lower.size(); dim++)
    {
        if(lower[dim] > upper[dim])
        {
            throw std::invalid_argument(
                ROCFFT_CURRENT_FUNCTION
                + " detected an invalid range of logical indices along layout axis "
                + std::to_string(dim) + ": lower (" + std::to_string(lower[dim])
                + ") is larger than upper (" + std::to_string(upper[dim]) + ").");
        }
        if(!is_partial && lower[dim] != 0)
            throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                        + " requires the logical range's lower bounds to be 0 for "
                                          "non-partial data layouts (lower["
                                        + std::to_string(dim) + "] is " + std::to_string(lower[dim])
                                        + ").");

        (*this)[dim].lower           = lower[dim];
        (*this)[dim].upper           = upper[dim];
        (*this)[dim].inbuffer_stride = strides[dim];
        (*this)[dim].is_partial      = is_partial;
    }
}

data_layout_t data_layout_t::full_layout(const std::vector<size_t>& lengths,
                                         const std::vector<size_t>& strides,
                                         size_t                     batch,
                                         size_t                     distance)
{
    return data_layout_t{std::vector<size_t>(lengths.size() + 1, 0),
                         concatenate(lengths, batch),
                         concatenate(strides, distance),
                         1,
                         false /* : is_partial */};
}

data_layout_t data_layout_t::default_full_layout(const std::vector<size_t>& lengths,
                                                 size_t                     batch,
                                                 bool                       real_case_with_padding)
{
    data_layout_t             ret;
    const std::vector<size_t> empty_for_default_strides_and_dist = {};
    ret.full_range_reset(lengths,
                         empty_for_default_strides_and_dist,
                         {batch},
                         empty_for_default_strides_and_dist,
                         real_case_with_padding);
    return ret;
}

size_t data_layout_t::get_len_rank() const
{
    return len_axes.size();
}

size_t data_layout_t::get_batch_rank() const
{
    return batch_axes.size();
}
size_t data_layout_t::get_full_rank() const
{
    return len_axes.size() + batch_axes.size();
}

std::vector<size_t> data_layout_t::lengths_and_batches() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].logical_span();

    return ret;
}

std::vector<size_t> data_layout_t::strides_and_distances() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].inbuffer_stride;

    return ret;
}

std::vector<size_t> data_layout_t::contiguous_strides_and_distances() const
{
    if(is_contiguous())
    {
        // a layout may be "contiguous" even if not ordered by increasing strides, e.g.,
        // lower := {0, 0}, upper := {a, b} and strides := {b, 1} is continuous...
        return strides_and_distances();
    }

    std::vector<size_t> ret(get_full_rank(), 1);
    for(size_t dim = 1; dim < get_full_rank(); dim++)
        ret[dim] = ret[dim - 1] * (*this)[dim - 1].logical_span();

    return ret;
}

std::vector<size_t> data_layout_t::lengths() const
{
    std::vector<size_t> ret(len_axes.size());
    for(size_t dim = 0; dim < len_axes.size(); dim++)
        ret[dim] = len_axes[dim].logical_span();

    return ret;
}

std::vector<size_t> data_layout_t::strides() const
{
    std::vector<size_t> ret(len_axes.size());
    for(size_t dim = 0; dim < len_axes.size(); dim++)
        ret[dim] = len_axes[dim].inbuffer_stride;

    return ret;
}

std::vector<size_t> data_layout_t::batches() const
{
    std::vector<size_t> ret(batch_axes.size());
    for(size_t dim = 0; dim < batch_axes.size(); dim++)
        ret[dim] = batch_axes[dim].logical_span();

    return ret;
}

std::vector<size_t> data_layout_t::distances() const
{
    std::vector<size_t> ret(batch_axes.size());
    for(size_t dim = 0; dim < batch_axes.size(); dim++)
        ret[dim] = batch_axes[dim].inbuffer_stride;

    return ret;
}

size_t data_layout_t::batch() const
{
    if(batch_axes.size() != 1)
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " is restricted to data layouts with only one batch axis.");
    return batch_axes[0].logical_span();
}

size_t data_layout_t::distance() const
{
    if(batch_axes.size() != 1)
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " is restricted to data layouts with only one batch axis.");
    return batch_axes[0].inbuffer_stride;
}

std::vector<size_t> data_layout_t::upper() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].upper;

    return ret;
}

std::vector<size_t> data_layout_t::lower() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].lower;

    return ret;
}

bool data_layout_t::is_empty() const
{
    bool ret = len_axes.empty() || batch_axes.empty();
    for(size_t dim = 0; !ret && dim < get_full_rank(); dim++)
        ret |= (*this)[dim].logical_span() == 0;
    return ret;
}

size_t data_layout_t::logical_count() const
{
    auto len_and_batches = lengths_and_batches();
    return product(len_and_batches.begin(), len_and_batches.end());
}

size_t data_layout_t::buffer_element_count() const
{
    return compute_ptrdiff(lengths_and_batches(), strides_and_distances());
}

bool data_layout_t::is_contiguous() const
{
    return buffer_element_count() == logical_count();
}

bool data_layout_t::operator==(const data_layout_t& other) const
{
    return len_axes == other.len_axes && batch_axes == other.batch_axes;
}

bool data_layout_t::has_same_logical_range_as(const data_layout_t& other) const
{
    if(!is_dimensionally_consistent_with(other))
        throw std::invalid_argument(
            "This object is not dimensionally consistent with the argument of "
            + ROCFFT_CURRENT_FUNCTION);
    bool ret = true;
    for(size_t dim = 0; ret && dim < get_full_rank(); dim++)
        ret &= (*this)[dim].has_same_logical_range_as(other[dim]);
    return ret;
}

data_layout_t data_layout_t::make_contiguous_intersection_of(const data_layout_t& first,
                                                             const data_layout_t& second)
{
    if(!first.is_dimensionally_consistent_with(second))
        throw std::invalid_argument("Dimensionally-inconsistent arguments for "
                                    + ROCFFT_CURRENT_FUNCTION);
    if(first.embedding != second.embedding)
        throw std::invalid_argument("Different embeddings for arguments of "
                                    + ROCFFT_CURRENT_FUNCTION);

    data_layout_t ret;
    ret.len_axes.resize(first.len_axes.size());
    ret.batch_axes.resize(first.batch_axes.size());
    for(size_t dim = 0; dim < first.get_full_rank(); dim++)
    {
        ret[dim].lower = std::max(first[dim].lower, second[dim].lower);
        ret[dim].upper = std::max(std::min(first[dim].upper, second[dim].upper), ret[dim].lower);
        ret[dim].is_partial = first[dim].is_partial || second[dim].is_partial
                              || !first[dim].has_same_logical_range_as(second[dim]);
        if(dim == 0)
            ret[dim].inbuffer_stride = 1;
        else
            ret[dim].inbuffer_stride = ret[dim - 1].inbuffer_stride * ret[dim - 1].logical_span();
    }
    ret.embedding = first.embedding;
    return ret;
}

std::vector<size_t> data_layout_t::length_axes_by_increasing_strides(bool pin_innermost_axis) const
{
    std::vector<size_t> ret(get_len_rank(), 0);
    if(ret.size() <= 1)
        return ret;
    std::iota(ret.begin(), ret.end(), 0);
    if(ret.size() == 2 && pin_innermost_axis)
        return ret;
    std::sort(ret.begin() + (pin_innermost_axis ? 1 : 0),
              ret.end(),
              [this](size_t dim_idx_i, size_t dim_idx_j) {
                  return len_axes[dim_idx_i].inbuffer_stride < len_axes[dim_idx_j].inbuffer_stride;
              });
    return ret;
}

void data_layout_t::reorder_length_axes(const std::vector<size_t>& len_axis_order)
{
    std::vector<size_t> current_order(get_len_rank(), 0);
    std::iota(current_order.begin(), current_order.end(), 0);
    if(!std::is_permutation(len_axis_order.begin(),
                            len_axis_order.end(),
                            current_order.begin(),
                            current_order.end()))
    {
        throw std::invalid_argument("Invalid re-ordering of length axes requested to "
                                    + ROCFFT_CURRENT_FUNCTION);
    }
    if(current_order != len_axis_order) // nothing to do otherwise
    {
        const auto len_axis_copy = len_axes;
        for(size_t dim = 0; dim < get_len_rank(); dim++)
            len_axes[dim] = len_axis_copy[len_axis_order[dim]];

        if(embedding)
        {
            // Update map according to new order
            for(auto& axis_idx : *embedding)
            {
                if(axis_idx >= get_len_rank())
                    continue; // batch axes are unmodified herein
                axis_idx = std::distance(
                    len_axis_order.begin(),
                    std::find(len_axis_order.begin(), len_axis_order.end(), axis_idx));
            }
        }
    }
}

size_t data_layout_t::slowest_varying_axis() const
{
    size_t ret = 0;
    for(size_t dim = 1; dim < get_full_rank(); dim++)
    {
        if((*this)[dim].logical_span() == 1)
            continue; // irrelevant unit length
        if((*this)[ret].logical_span() == 1 /*current max is irrelevant (unit length)*/
           || (*this)[ret].inbuffer_stride < (*this)[dim].inbuffer_stride)
        {
            ret = dim;
        }
    }
    return ret;
}

size_t data_layout_t::offset_in(const data_layout_t& other) const
{
    if(!is_dimensionally_consistent_with(other))
        throw std::invalid_argument(
            "This object is not dimensionally consistent with the argument of "
            + ROCFFT_CURRENT_FUNCTION);
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        if(!other[dim].logically_contains((*this)[dim].lower))
            throw std::invalid_argument(
                ROCFFT_CURRENT_FUNCTION
                + " requires this object's lower coordinate to be included in other.");

    size_t offset = 0;
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        offset += other[dim].inbuffer_stride * ((*this)[dim].lower - other[dim].lower);

    return offset;
}

bool data_layout_t::is_continuous_in(const data_layout_t& other) const
{
    if(!is_dimensionally_consistent_with(other))
        throw std::invalid_argument(
            "This object is not dimensionally consistent with the argument of "
            + ROCFFT_CURRENT_FUNCTION);

    bool ret = true;
    for(size_t dim = 0; ret && dim < get_full_rank(); dim++)
    {
        if(dim == other.slowest_varying_axis())
        {
            ret &= other[dim].logically_contains((*this)[dim]);
            if((*this)[dim].logical_span() > 1)
                ret &= (*this)[dim].inbuffer_stride == other[dim].inbuffer_stride;
        }
        else
            ret &= other[dim] == (*this)[dim];
    }
    return ret;
}

std::string data_layout_t::str() const
{
    std::string ret;
    ret += "lower";
    for(const auto& l : lower())
    {
        ret += " ";
        ret += std::to_string(l);
    }

    ret += " upper";
    for(const auto& u : upper())
    {
        ret += " ";
        ret += std::to_string(u);
    }

    ret += " stride";
    for(const auto& s_or_d : strides_and_distances())
    {
        ret += " ";
        ret += std::to_string(s_or_d);
    }

    if(is_embedded())
    {
        ret += " embedded within a " + std::to_string(embedding->size()) + "-dimensional data set";
    }
    return ret;
}

void data_layout_t::clear()
{
    len_axes.clear();
    batch_axes.clear();
    embedding.reset();
}

void data_layout_t::full_range_reset(const std::vector<size_t>& lengths,
                                     const std::vector<size_t>& strides,
                                     const std::vector<size_t>& batches,
                                     const std::vector<size_t>& distances,
                                     bool                       real_case_with_padding)
{
    if(lengths.empty() || batches.empty())
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires non-empty lengths and batches");
    if(!strides.empty() && strides.size() != lengths.size())
        throw std::invalid_argument(
            ROCFFT_CURRENT_FUNCTION
            + " requires strides to be of the same size as lengths if not empty.");
    if(!distances.empty() && distances.size() != batches.size())
        throw std::invalid_argument(
            ROCFFT_CURRENT_FUNCTION
            + " requires distances to be of the same size as batches if not empty.");
    clear();
    len_axes.resize(lengths.size());
    batch_axes.resize(batches.size());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
    {
        (*this)[dim].is_partial = false;
        (*this)[dim].lower      = 0;
        (*this)[dim].upper = dim < lengths.size() ? lengths[dim] : batches[dim - lengths.size()];
        const bool set_default_axis_stride = (dim < lengths.size() && strides.empty())
                                             || (dim >= lengths.size() && distances.empty());
        if(dim == 0)
        {
            if(set_default_axis_stride)
                (*this)[dim].inbuffer_stride = 1;
            else
            {
                // dim == 0 cannot be a batch axis as empty lengths are not accepted
                (*this)[dim].inbuffer_stride = strides[dim];
            }
        }
        else if(real_case_with_padding && dim == 1 && set_default_axis_stride)
            (*this)[dim].inbuffer_stride
                = 2 * (lengths[dim - 1] / 2 + 1) * (*this)[dim - 1].inbuffer_stride;
        else
        {
            if(set_default_axis_stride)
                (*this)[dim].inbuffer_stride
                    = (*this)[dim - 1].inbuffer_stride * (*this)[dim - 1].logical_span();
            else if(dim < lengths.size())
                (*this)[dim].inbuffer_stride = strides[dim];
            else
                (*this)[dim].inbuffer_stride = distances[dim - lengths.size()];
        }
    }
}

bool data_layout_t::is_dimensionally_consistent_with(const data_layout_t& other) const
{
    return get_batch_rank() == other.get_batch_rank() && get_len_rank() == other.get_len_rank();
}

bool data_layout_t::has_some_partial_length_axis() const
{
    return std::any_of(
        len_axes.begin(), len_axes.end(), [](const auto& len_axis) { return len_axis.is_partial; });
}

std::optional<data_layout_t>
    data_layout_t::get_other_inplace_layout_for(io_data_label         other_io,
                                                rocfft_transform_type fft_type,
                                                bool other_innermost_length_is_odd) const
{
    if(other_io != io_data_label::INPUT && other_io != io_data_label::OUTPUT)
        throw std::invalid_argument("Unknown I/O data label given to " + ROCFFT_CURRENT_FUNCTION);

    if(is_empty())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " queried on an empty layout");

    if(has_some_partial_length_axis())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " queried on a layout involving partial length axes");

    if(fft_type == rocfft_transform_type_complex_forward
       || fft_type == rocfft_transform_type_complex_inverse)
    {
        // simple copy
        return std::make_optional<data_layout_t>(*this);
    }
    else if(fft_type != rocfft_transform_type_real_forward
            && fft_type != rocfft_transform_type_real_inverse)
    {
        throw std::invalid_argument("Unknown type of transform given to "
                                    + ROCFFT_CURRENT_FUNCTION);
    }

    // real transform: unit stride is required along the innermost axis
    if((*this)[0].logical_span() > 1 && (*this)[0].inbuffer_stride != 1)
        return std::nullopt;
    if(is_hermitian_domain(fft_type, other_io))
    {
        bool strides_are_consistent = true;
        for(size_t dim = 1; strides_are_consistent && dim < get_full_rank(); dim++)
        {
            if((*this)[dim].logical_span() == 1)
                continue;
            strides_are_consistent
                &= (*this)[dim].inbuffer_stride % 2 == 0
                   && (*this)[dim].inbuffer_stride >= 2 * ((*this)[0].logical_span() / 2 + 1);
        }
        if(!strides_are_consistent)
            return std::nullopt;
    }
    // copy layout and modify what needs be
    auto ret = std::make_optional<data_layout_t>(*this);
    (*ret)[0].upper
        = is_real_domain(fft_type, other_io)
              ? 2 * ((*this)[0].logical_span() - 1) + (other_innermost_length_is_odd ? 1 : 0)
              : (*this)[0].logical_span() / 2 + 1;
    for(size_t dim = 1; dim < ret->get_full_rank(); dim++)
    {
        (*ret)[dim].inbuffer_stride = is_real_domain(fft_type, other_io)
                                          ? 2 * (*this)[dim].inbuffer_stride
                                          : (*this)[dim].inbuffer_stride / 2;
    }
    return ret;
}

data_layout_t data_layout_t::get_layout_for_len_axes(const std::set<size_t>& len_indices) const
{
    const auto len_rank = get_len_rank();
    if(len_indices.empty() || len_indices.size() > len_rank
       || std::any_of(len_indices.begin(), len_indices.end(), [&len_rank](const auto& len_idx) {
              return len_idx >= len_rank;
          }))
    {
        throw std::invalid_argument("Invalid subset of length axis indices given to "
                                    + ROCFFT_CURRENT_FUNCTION);
    }
    if(len_indices.size() == len_rank)
        return *this;

    data_layout_t ret;
    // map of current object's to returned object's axis indices
    std::vector<size_t> current_to_ret(get_full_rank());
    // Turn non-desired length axes into additional batch axes
    for(size_t axis_idx = 0; axis_idx < get_full_rank(); axis_idx++)
    {
        if(len_indices.contains(axis_idx))
        {
            current_to_ret[axis_idx] = ret.len_axes.size();
            ret.len_axes.push_back((*this)[axis_idx]);
        }
        else
        {
            current_to_ret[axis_idx] = len_indices.size() + ret.batch_axes.size();
            ret.batch_axes.push_back((*this)[axis_idx]);
        }
    }
    // Set returned object's embedding map
    ret.embedding = this->embedding;
    if(!ret.embedding)
    {
        // If no embedding in parent, initialize to default indices
        ret.embedding = std::vector<size_t>(len_rank);
        std::iota(ret.embedding->begin(), ret.embedding->end(), 0);
    }
    for(auto& embed_idx : *ret.embedding)
        embed_idx = current_to_ret[embed_idx];
    return ret;
}

bool data_layout_t::has_consistent_embedding() const
{
    if(!embedding)
        return true;
    // Embedding data sets cannot have empty length_axes nor empty batch_axes
    bool ret = !embedding->empty() && embedding->size() < get_full_rank();
    // All axis indices in the embedding must be within bounds
    ret &= std::all_of(embedding->begin(), embedding->end(), [&](const size_t& axis_idx) {
        return axis_idx < get_full_rank();
    });
    // All length axes of current object must be in the embedding
    for(size_t len_idx = 0; ret && len_idx < get_len_rank(); len_idx++)
        ret &= std::find(embedding->begin(), embedding->end(), len_idx) != embedding->end();
    // Embedding cannot have duplicated axis indices
    for(size_t i = 0; ret && i < embedding->size(); i++)
        for(size_t j = i + 1; ret && j < embedding->size(); j++)
            ret &= (*embedding)[i] != (*embedding)[j];
    return ret;
}

data_layout_t data_layout_t::get_embedding_layout() const
{
    if(!embedding)
        return *this;
    if(!has_consistent_embedding())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " detected an inconsistent embedding map");

    data_layout_t ret;
    for(const auto& axis_idx : *embedding)
        ret.len_axes.push_back((*this)[axis_idx]);
    // all other batch axes that aren't part of the embedding remain batch axes
    // (conserve relative ordering)
    for(auto axis_idx = get_len_rank(); axis_idx < get_full_rank(); axis_idx++)
    {
        if(std::find(embedding->begin(), embedding->end(), axis_idx) != embedding->end())
            continue;
        ret.batch_axes.push_back((*this)[axis_idx]);
    }
    return ret;
}

bool data_layout_t::is_embedded() const
{
    return embedding.has_value() && embedding->size() > get_len_rank();
}

std::vector<size_t> data_layout_t::corresponding_axes_in_embedding() const
{
    std::vector<size_t> ret;
    if(!embedding)
    {
        ret.resize(get_len_rank());
        std::iota(ret.begin(), ret.end(), 0);
        return ret;
    }
    if(!has_consistent_embedding())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " detected an inconsistent embedding map");
    for(size_t len_idx = 0; len_idx < get_len_rank(); len_idx++)
        ret.push_back(std::distance(embedding->begin(),
                                    std::find(embedding->begin(), embedding->end(), len_idx)));

    return ret;
}
