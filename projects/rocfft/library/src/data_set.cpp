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

#include "data_set.h"
#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "../../shared/ptrdiff.h"
#include "enum_printer.h"
#include "rocfft_current_function.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace
{
    // true iff a and b are identical or are the hermitian and complex variants
    // of the same format (interleaved or planar)
    bool same_format_up_to_hermitian(rocfft_array_type a, rocfft_array_type b)
    {
        return a == b
               || (array_type_is_complex(a) && array_type_is_complex(b)
                   && array_type_is_interleaved(a) == array_type_is_interleaved(b));
    }

    // complex counterpart of a Hermitian array type, other array types unchanged
    rocfft_array_type complex_counterpart(rocfft_array_type t)
    {
        switch(t)
        {
        case rocfft_array_type_hermitian_interleaved:
            return rocfft_array_type_complex_interleaved;
        case rocfft_array_type_hermitian_planar:
            return rocfft_array_type_complex_planar;
        default:
            return t;
        }
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

#define IO_DATA_SET_ACCESSOR_BODY(AXIS_IDX)                                         \
    do                                                                              \
    {                                                                               \
        if(AXIS_IDX >= get_full_rank())                                             \
            throw std::out_of_range(ROCFFT_CURRENT_FUNCTION                         \
                                    + " accessing an out-of-range data set axis."); \
        return AXIS_IDX < len_axes.size() ? len_axes[AXIS_IDX]                      \
                                          : batch_axes[AXIS_IDX - len_axes.size()]; \
    } while(0)

const data_set_t::axis_t& data_set_t::operator[](size_t axis_idx) const
{
    IO_DATA_SET_ACCESSOR_BODY(axis_idx);
}

data_set_t::axis_t& data_set_t::operator[](size_t axis_idx)
{
    IO_DATA_SET_ACCESSOR_BODY(axis_idx);
}

data_set_t::data_set_t(const std::vector<size_t>& lower,
                       const std::vector<size_t>& upper,
                       const std::vector<size_t>& strides,
                       size_t                     batch_rank)
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
                + " detected an invalid range of logical indices along data set axis "
                + std::to_string(dim) + ": lower (" + std::to_string(lower[dim])
                + ") is larger than upper (" + std::to_string(upper[dim]) + ").");
        }

        (*this)[dim].lower           = lower[dim];
        (*this)[dim].upper           = upper[dim];
        (*this)[dim].inbuffer_stride = strides[dim];
    }
}

data_set_t data_set_t::full_range(const std::vector<size_t>& lengths,
                                  const std::vector<size_t>& strides,
                                  size_t                     batch,
                                  size_t                     distance,
                                  size_t                     batch_rank)
{
    if(lengths.size() != strides.size() || batch_rank == 0 || batch_rank > lengths.size())
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires matching length/stride "
                                      "sizes and nonzero ranks.");
    data_set_t ret;
    ret.len_axes.resize(lengths.size() + 1 - batch_rank);
    ret.batch_axes.resize(batch_rank);
    for(size_t dim = 0; dim < lengths.size(); ++dim)
        ret[dim] = {0, lengths[dim], strides[dim]};
    ret.batch_axes.back() = {0, batch, distance};
    return ret;
}

data_set_t data_set_t::default_full_range(const std::vector<size_t>& lengths,
                                          const std::vector<size_t>& batches,
                                          bool                       real_case_with_padding)
{
    data_set_t                ret;
    const std::vector<size_t> empty_for_default_strides_and_dist = {};
    ret.full_range_reset(lengths,
                         empty_for_default_strides_and_dist,
                         batches,
                         empty_for_default_strides_and_dist,
                         real_case_with_padding);
    return ret;
}

size_t data_set_t::get_len_rank() const
{
    return len_axes.size();
}

size_t data_set_t::get_batch_rank() const
{
    return batch_axes.size();
}
size_t data_set_t::get_full_rank() const
{
    return len_axes.size() + batch_axes.size();
}

rocfft_array_type data_set_t::get_array_type() const
{
    if(!has_array_type())
    {
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " called on a data set without array type.");
    }

    return array_type;
}

bool data_set_t::has_array_type() const
{
    return array_type != rocfft_array_type_unset;
}

rocfft_precision data_set_t::get_precision() const
{
    if(!has_precision())
    {
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " called on a data set without precision.");
    }

    return static_cast<rocfft_precision>(precision);
}

bool data_set_t::has_precision() const
{
    return precision != precision_unset;
}

void data_set_t::set_precision(rocfft_precision p)
{
    precision = static_cast<std::underlying_type_t<rocfft_precision>>(p);
}

void data_set_t::set_array_type_and_offsets(rocfft_array_type            new_array_type,
                                            const std::array<size_t, 2>& new_offsets)
{
    if(new_offsets[1] != 0 && !array_type_is_planar(new_array_type))
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires a zero second offset for non-planar array types.");
    array_type = new_array_type;
    offsets    = new_offsets;
}

size_t data_set_t::num_planes() const
{
    return array_type_is_planar(array_type) ? 2 : 1;
}

size_t data_set_t::get_offset(data_part part) const
{
    if(array_type_is_planar(array_type) == (part == data_part::ALL))
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires data_part::ALL for non-planar data sets, and"
                                      " data_part::REAL or data_part::IMAGINARY for planar ones.");
    return offsets[part == data_part::IMAGINARY ? 1 : 0];
}

size_t data_set_t::axis_span(size_t axis) const
{
    return (*this)[axis].logical_span();
}

size_t data_set_t::axis_stride(size_t axis) const
{
    return (*this)[axis].inbuffer_stride;
}

bool data_set_t::same_batch_range(const data_set_t& other) const
{
    return std::equal(batch_axes.begin(),
                      batch_axes.end(),
                      other.batch_axes.begin(),
                      other.batch_axes.end(),
                      [](const auto& a, const auto& b) { return a.has_same_logical_range_as(b); });
}

bool data_set_t::same_logical_range(const data_set_t& other) const
{
    return same_batch_range(other)
           && std::equal(
               len_axes.begin(),
               len_axes.end(),
               other.len_axes.begin(),
               other.len_axes.end(),
               [](const auto& a, const auto& b) { return a.has_same_logical_range_as(b); });
}

void data_set_t::set_strides_and_distances(std::span<const size_t> values)
{
    if(values.size() != get_full_rank())
        throw std::invalid_argument("Wrong number of strides and distances");
    for(size_t i = 0; i < values.size(); ++i)
        (*this)[i].inbuffer_stride = values[i];
}

void data_set_t::set_axis_stride(size_t axis, size_t stride)
{
    (*this)[axis].inbuffer_stride = stride;
}

void data_set_t::reset_inbuffer(const data_set_t& other)
{
    if(!same_logical_range(other))
        throw std::invalid_argument("Cannot change the logical range of a data set");
    for(size_t i = 0; i < get_full_rank(); ++i)
        (*this)[i].inbuffer_stride = other.axis_stride(i);
    set_array_type_and_offsets(other.array_type, other.offsets);
}

data_set_t data_set_t::with_axis_span(size_t axis, size_t span) const
{
    auto result        = *this;
    result[axis].upper = result[axis].lower + span;
    return result;
}

data_set_t data_set_t::with_element_type(rocfft_array_type            type,
                                         rocfft_precision             p,
                                         const std::array<size_t, 2>& new_offsets) const
{
    auto result = *this;
    result.set_array_type_and_offsets(type, new_offsets);
    result.set_precision(p);
    return result;
}

std::vector<size_t> data_set_t::lengths_and_batches() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].logical_span();

    return ret;
}

std::vector<size_t> data_set_t::strides_and_distances() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].inbuffer_stride;

    return ret;
}

std::vector<size_t> data_set_t::contiguous_strides_and_distances() const
{
    if(is_contiguous())
    {
        // a data set may be "contiguous" even if not ordered by increasing strides, e.g.,
        // lower := {0, 0}, upper := {a, b} and strides := {b, 1} is continuous...
        return strides_and_distances();
    }

    std::vector<size_t> ret(get_full_rank(), 1);
    for(size_t dim = 1; dim < get_full_rank(); dim++)
        ret[dim] = ret[dim - 1] * (*this)[dim - 1].logical_span();

    return ret;
}

std::vector<size_t> data_set_t::lengths() const
{
    std::vector<size_t> ret(len_axes.size());
    for(size_t dim = 0; dim < len_axes.size(); dim++)
        ret[dim] = len_axes[dim].logical_span();

    return ret;
}

std::vector<size_t> data_set_t::strides() const
{
    std::vector<size_t> ret(len_axes.size());
    for(size_t dim = 0; dim < len_axes.size(); dim++)
        ret[dim] = len_axes[dim].inbuffer_stride;

    return ret;
}

std::vector<size_t> data_set_t::batches() const
{
    std::vector<size_t> ret(batch_axes.size());
    for(size_t dim = 0; dim < batch_axes.size(); dim++)
        ret[dim] = batch_axes[dim].logical_span();

    return ret;
}

std::vector<size_t> data_set_t::distances() const
{
    std::vector<size_t> ret(batch_axes.size());
    for(size_t dim = 0; dim < batch_axes.size(); dim++)
        ret[dim] = batch_axes[dim].inbuffer_stride;

    return ret;
}

size_t data_set_t::batch() const
{
    if(batch_axes.size() != 1)
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " is restricted to data sets with only one batch axis.");
    return batch_axes[0].logical_span();
}

size_t data_set_t::distance() const
{
    if(batch_axes.size() != 1)
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " is restricted to data sets with only one batch axis.");
    return batch_axes[0].inbuffer_stride;
}

std::vector<size_t> data_set_t::upper() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].upper;

    return ret;
}

std::vector<size_t> data_set_t::lower() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].lower;

    return ret;
}

bool data_set_t::is_empty() const
{
    bool ret = len_axes.empty() || batch_axes.empty();
    for(size_t dim = 0; !ret && dim < get_full_rank(); dim++)
        ret |= (*this)[dim].logical_span() == 0;
    return ret;
}

size_t data_set_t::logical_count() const
{
    auto len_and_batches = lengths_and_batches();
    return product(len_and_batches.begin(), len_and_batches.end());
}

size_t data_set_t::buffer_element_count(data_part part) const
{
    const auto offset = get_offset(part);
    const auto count  = compute_ptrdiff(lengths_and_batches(), strides_and_distances());
    // an empty data set needs no buffer, wherever it would start
    return count == 0 ? 0 : offset + count;
}

size_t data_set_t::element_byte_size() const
{
    size_t real_bytes = 0;
    switch(get_precision())
    {
    case rocfft_precision_half:
        real_bytes = 2;
        break;
    case rocfft_precision_single:
        real_bytes = 4;
        break;
    case rocfft_precision_double:
        real_bytes = 8;
        break;
    default:
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " found an unexpected precision.");
    }
    switch(get_array_type())
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        return 2 * real_bytes;
    case rocfft_array_type_real:
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        return real_bytes;
    default:
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " found an unexpected array type.");
    }
}

size_t data_set_t::buffer_size_bytes(data_part part) const
{
    return buffer_element_count(part) * element_byte_size();
}

size_t data_set_t::max_offset() const
{
    if(is_empty())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " is undefined for an empty logical index range.");
    size_t ret = *std::max_element(offsets.begin(), offsets.begin() + num_planes());
    for(size_t dim = 0; dim < get_full_rank(); ++dim)
    {
        const auto& axis = (*this)[dim];
        ret += (axis.logical_span() - 1) * axis.inbuffer_stride;
    }
    return ret;
}

size_t data_set_t::max_stride() const
{
    size_t ret = 0;
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret = std::max(ret, (*this)[dim].inbuffer_stride);
    return ret;
}

data_set_t data_set_t::reinterpret_as(rocfft_array_type new_array_type) const
{
    if(array_type == rocfft_array_type_unset || new_array_type == rocfft_array_type_unset)
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + " requires set array types.");
    if(array_type_is_real(array_type) == array_type_is_real(new_array_type))
    {
        if(!same_format_up_to_hermitian(array_type, new_array_type))
            throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                        + " cannot switch between interleaved and planar formats.");
        // allowed relabels keep the number of planes, hence the offsets
        auto ret = *this;
        ret.set_array_type_and_offsets(new_array_type, offsets);
        return ret;
    }

    const bool to_complex = array_type_is_real(array_type);
    // real <-> complex/hermitian: length axis 0 is split/merged, so it must be the
    // embedding's length axis 0 for get_embedding_data_set() to commute with this
    if(embedding && (*embedding)[0] != 0)
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires length axis 0 of an embedded data set to be"
                                      " the embedding's length axis 0.");

    if(array_type_is_planar(to_complex ? new_array_type : array_type))
    {
        // Real elements 2k and 2k + 1 are element k in planes 0 and 1, respectively.
        // Other strides and distances stay in real-element units.
        auto ret = *this;
        if(to_complex)
        {
            if(len_axes[0].lower % 2 != 0 || len_axes[0].upper % 2 != 0)
                throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                            + " requires even bounds along length axis 0 to"
                                              " reinterpret real data as planar data.");
            ret.len_axes[0].lower /= 2;
            ret.len_axes[0].upper /= 2;
            ret.len_axes[0].inbuffer_stride *= 2;
            ret.set_array_type_and_offsets(new_array_type,
                                           {offsets[0], offsets[0] + len_axes[0].inbuffer_stride});
        }
        else
        {
            if(len_axes[0].inbuffer_stride % 2 != 0
               || offsets[1] != offsets[0] + len_axes[0].inbuffer_stride / 2)
                throw std::invalid_argument(
                    ROCFFT_CURRENT_FUNCTION
                    + " requires an even stride along length axis 0 and planes offset by half of"
                      " it to reinterpret planar data as real data.");
            ret.len_axes[0].lower *= 2;
            ret.len_axes[0].upper *= 2;
            ret.len_axes[0].inbuffer_stride /= 2;
            ret.set_array_type_and_offsets(new_array_type, {offsets[0], 0});
        }
        return ret;
    }

    // real element 2k (resp. 2k + 1) is the real (resp. imaginary) part of complex
    // element k: positions in real units are twice the positions in complex units
    if(len_axes[0].inbuffer_stride != 1)
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires a unit stride along length axis 0 to"
                                      " reinterpret between real and interleaved data.");
    if(to_complex)
    {
        bool all_even
            = len_axes[0].lower % 2 == 0 && len_axes[0].upper % 2 == 0 && offsets[0] % 2 == 0;
        for(size_t dim = 1; dim < get_full_rank(); dim++)
            all_even &= (*this)[dim].inbuffer_stride % 2 == 0;
        if(!all_even)
            throw std::invalid_argument(
                ROCFFT_CURRENT_FUNCTION
                + " requires even bounds along length axis 0 and even other strides,"
                  " distances and offset to reinterpret real data as interleaved data.");
    }
    const auto convert    = [to_complex](size_t x) { return to_complex ? x / 2 : 2 * x; };
    auto       ret        = *this;
    ret.len_axes[0].lower = convert(len_axes[0].lower);
    ret.len_axes[0].upper = convert(len_axes[0].upper);
    for(size_t dim = 1; dim < get_full_rank(); dim++)
        ret[dim].inbuffer_stride = convert((*this)[dim].inbuffer_stride);
    ret.set_array_type_and_offsets(new_array_type, {convert(offsets[0]), 0});
    return ret;
}

data_set_t data_set_t::get_packed_data_set(bool real_inplace_padding) const
{
    // keeps precision and array type; full_range_reset drops the embedding
    auto ret = *this;
    ret.full_range_reset(lengths(), {}, batches(), {}, real_inplace_padding);
    ret.offsets = {0, 0};
    for(size_t dim = 0; dim < get_full_rank(); dim++)
    {
        ret[dim].lower = (*this)[dim].lower;
        ret[dim].upper = (*this)[dim].upper;
    }
    return ret;
}

bool data_set_t::is_contiguous() const
{
    return compute_ptrdiff(lengths_and_batches(), strides_and_distances()) == logical_count();
}

bool data_set_t::operator==(const data_set_t& other) const
{
    return len_axes == other.len_axes && batch_axes == other.batch_axes
           && array_type == other.array_type && precision == other.precision
           && offsets == other.offsets;
}

bool data_set_t::has_same_logical_range_as(const data_set_t& other) const
{
    return has_same_batch_range_as(other) && has_same_length_range_as(other);
}

bool data_set_t::has_same_length_range_as(const data_set_t& other) const
{
    if(get_len_rank() != other.get_len_rank())
        throw std::invalid_argument("Inconsistent length ranks given to "
                                    + ROCFFT_CURRENT_FUNCTION);
    return std::equal(
        len_axes.begin(),
        len_axes.end(),
        other.len_axes.begin(),
        [](const axis_t& a, const axis_t& b) { return a.has_same_logical_range_as(b); });
}

bool data_set_t::has_same_batch_range_as(const data_set_t& other) const
{
    if(get_batch_rank() != other.get_batch_rank())
        throw std::invalid_argument("Inconsistent batch ranks given to " + ROCFFT_CURRENT_FUNCTION);
    return std::equal(
        batch_axes.begin(),
        batch_axes.end(),
        other.batch_axes.begin(),
        [](const axis_t& a, const axis_t& b) { return a.has_same_logical_range_as(b); });
}

data_set_t data_set_t::make_contiguous_intersection_of(const data_set_t& first,
                                                       const data_set_t& second)
{
    if(!first.is_dimensionally_consistent_with(second))
        throw std::invalid_argument("Dimensionally-inconsistent arguments for "
                                    + ROCFFT_CURRENT_FUNCTION);
    if(first.embedding != second.embedding)
        throw std::invalid_argument("Different embeddings for arguments of "
                                    + ROCFFT_CURRENT_FUNCTION);
    if(first.precision != second.precision || first.array_type != second.array_type)
        throw std::invalid_argument("Different precisions or array types for arguments of "
                                    + ROCFFT_CURRENT_FUNCTION);

    data_set_t ret;
    ret.len_axes.resize(first.len_axes.size());
    ret.batch_axes.resize(first.batch_axes.size());
    for(size_t dim = 0; dim < first.get_full_rank(); dim++)
    {
        ret[dim].lower = std::max(first[dim].lower, second[dim].lower);
        ret[dim].upper = std::max(std::min(first[dim].upper, second[dim].upper), ret[dim].lower);
        if(dim == 0)
            ret[dim].inbuffer_stride = 1;
        else
            ret[dim].inbuffer_stride = ret[dim - 1].inbuffer_stride * ret[dim - 1].logical_span();
    }
    // zero offsets: the intersection describes a fresh contiguous buffer
    ret.embedding  = first.embedding;
    ret.array_type = first.array_type;
    ret.precision  = first.precision;
    return ret;
}

std::vector<data_set_t>
    data_set_t::compress_batch_axes_consistently_in(std::vector<data_set_t> data_sets)
{
    compress_batch_axes_consistently_in(data_sets, false);
    return data_sets;
}

void data_set_t::compress_batch_axes_consistently_in(std::span<data_set_t> data_sets,
                                                     bool                  legacy_packed_spans)
{
    compress_axes_consistently_in(data_sets, false, legacy_packed_spans);
}

std::vector<data_set_t>
    data_set_t::compress_length_axes_consistently_in(std::vector<data_set_t> data_sets)
{
    compress_axes_consistently_in(data_sets, true);
    return data_sets;
}

void data_set_t::compress_axes_consistently_in(std::span<data_set_t> data_sets,
                                               bool                  length_axes,
                                               bool                  legacy_packed_spans)
{
    const auto axes = length_axes ? &data_set_t::len_axes : &data_set_t::batch_axes;
    // flattened index of the first axis of the group being compressed
    const auto first = [&](const data_set_t& d) { return length_axes ? 0 : d.get_len_rank(); };

    for(const auto& d : data_sets)
    {
        if(!d.is_dimensionally_consistent_with(data_sets.front()))
            throw std::invalid_argument("Dimensionally-inconsistent arguments for "
                                        + ROCFFT_CURRENT_FUNCTION);
        // length axes may differ in span (e.g., real vs. complex data); such axes are
        // simply not merged (see `mergeable` below)
        for(size_t i = 0; !length_axes && !legacy_packed_spans && i < d.get_batch_rank(); i++)
            if(d.batch_axes[i].logical_span() != data_sets.front().batch_axes[i].logical_span())
                throw std::invalid_argument("Different batch spans for arguments of "
                                            + ROCFFT_CURRENT_FUNCTION);
    }

    // true iff axis `outer` may be merged over axis `inner` (of the group) in `d`, with
    // the same spans as in the first data set
    auto mergeable = [&](const data_set_t& d, size_t outer, size_t inner) {
        const auto& o         = (d.*axes)[outer];
        const auto& i         = (d.*axes)[inner];
        const auto& reference = data_sets.front().*axes;
        if(legacy_packed_spans)
        {
            const auto keep = std::max(outer, inner);
            const auto del  = std::min(outer, inner);
            if(keep == reference.size() - 1
               && (d.*axes)[del].logical_span() != reference[del].logical_span())
                return false;
        }
        else if(o.logical_span() != reference[outer].logical_span()
                || i.logical_span() != reference[inner].logical_span())
            return false;
        if(d.embedding)
        {
            const auto& e = *d.embedding;
            if(std::find(e.begin(), e.end(), first(d) + outer) != e.end()
               || std::find(e.begin(), e.end(), first(d) + inner) != e.end())
                return false;
        }
        return o.inbuffer_stride == i.inbuffer_stride * i.logical_span()
               || (o.logical_span() == 1 && o.lower == 0);
    };

    auto can_merge = [&](size_t outer, size_t inner) {
        return std::all_of(data_sets.begin(), data_sets.end(), [&](const auto& d) {
            return mergeable(d, outer, inner);
        });
    };
    bool merged = true;
    while(merged && !data_sets.empty())
    {
        merged         = false;
        const auto num = (data_sets.front().*axes).size();
        // Keep original axis order. Try lower-indexed pairs first, preferring
        // their lower axis as inner when both directions are valid (unit spans).
        for(size_t del = 0; !merged && del < num; ++del)
        {
            for(size_t keep = del + 1; !merged && keep < num; ++keep)
            {
                const bool lower_is_inner = can_merge(keep, del);
                if(!lower_is_inner && !can_merge(del, keep))
                    continue;
                for(auto& d : data_sets)
                {
                    auto&      group  = d.*axes;
                    const auto o      = group[lower_is_inner ? keep : del];
                    const auto i      = group[lower_is_inner ? del : keep];
                    auto&      k      = group[keep];
                    k.lower           = i.lower + i.logical_span() * o.lower;
                    k.upper           = k.lower + i.logical_span() * o.logical_span();
                    k.inbuffer_stride = i.inbuffer_stride;
                    const auto erased = first(d) + del;
                    group.erase(group.begin() + del);
                    if(d.embedding)
                        for(auto& axis_idx : *d.embedding)
                            if(axis_idx > erased)
                                axis_idx--;
                }
                merged = true;
            }
        }
    }
}

std::vector<size_t> data_set_t::length_axes_by_increasing_strides(bool pin_innermost_axis) const
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

void data_set_t::reorder_length_axes(const std::vector<size_t>& len_axis_order)
{
    reorder_axes(len_axis_order, axis_scope::length_only);
}

std::vector<size_t> data_set_t::axes_by_increasing_inbuffer_strides(axis_scope scope) const
{
    const auto&         axes = scope == axis_scope::length_only ? len_axes : batch_axes;
    std::vector<size_t> order(axes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::tie(axes[a].inbuffer_stride, a) < std::tie(axes[b].inbuffer_stride, b);
    });
    return order;
}

void data_set_t::reorder_axes(std::span<const size_t> order, axis_scope scope)
{
    const bool length = scope == axis_scope::length_only;
    auto&      axes   = length ? len_axes : batch_axes;
    if(order.size() != axes.size())
        throw std::invalid_argument("Wrong number of axes to reorder");
    bool identity = true;
    for(size_t i = 0; i < order.size(); ++i)
    {
        if(order[i] >= order.size()
           || std::find(order.begin(), order.begin() + i, order[i]) != order.begin() + i)
            throw std::invalid_argument("Invalid axis permutation");
        identity &= order[i] == i;
    }
    if(identity)
        return;

    const auto old = axes;
    for(size_t i = 0; i < order.size(); ++i)
        axes[i] = old[order[i]];

    if(embedding)
    {
        const size_t first = length ? 0 : get_len_rank();
        for(auto& axis : *embedding)
            if(axis >= first && axis - first < order.size())
                axis = first
                       + std::distance(order.begin(),
                                       std::find(order.begin(), order.end(), axis - first));
    }
}

size_t data_set_t::slowest_varying_axis() const
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

size_t data_set_t::offset_in(const data_set_t& other) const
{
    if(!is_dimensionally_consistent_with(other))
        throw std::invalid_argument(
            "This object is not dimensionally consistent with the argument of "
            + ROCFFT_CURRENT_FUNCTION);
    if(precision != other.precision || array_type != other.array_type)
        throw std::invalid_argument(
            "This object's precision or array type differs from the argument of "
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

bool data_set_t::is_continuous_in(const data_set_t& other) const
{
    if(!is_dimensionally_consistent_with(other))
        throw std::invalid_argument(
            "This object is not dimensionally consistent with the argument of "
            + ROCFFT_CURRENT_FUNCTION);
    if(precision != other.precision || array_type != other.array_type)
        throw std::invalid_argument(
            "This object's precision or array type differs from the argument of "
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

std::string data_set_t::str() const
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

    ret += " precision ";
    ret += has_precision() ? PrintPrecision(get_precision()) : "unset";
    ret += " array type ";
    ret += PrintArrayType(array_type);
    ret += " offsets";
    for(size_t plane = 0; plane < num_planes(); plane++)
        ret += " " + std::to_string(offsets[plane]);

    if(is_embedded())
    {
        ret += " embedded within a " + std::to_string(embedding->size()) + "-dimensional data set";
    }
    return ret;
}

void data_set_t::clear()
{
    len_axes.clear();
    batch_axes.clear();
    embedding.reset();
}

void data_set_t::full_range_reset(const std::vector<size_t>& lengths,
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
        (*this)[dim].lower = 0;
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

bool data_set_t::is_dimensionally_consistent_with(const data_set_t& other) const
{
    return get_batch_rank() == other.get_batch_rank() && get_len_rank() == other.get_len_rank();
}

std::optional<data_set_t>
    data_set_t::get_other_inplace_data_set_for(io_data_label         other_io,
                                               rocfft_transform_type fft_type,
                                               bool other_innermost_length_is_odd) const
{
    if(other_io != io_data_label::INPUT && other_io != io_data_label::OUTPUT)
        throw std::invalid_argument("Unknown I/O data label given to " + ROCFFT_CURRENT_FUNCTION);

    if(is_empty())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " queried on an empty data set");

    if(std::any_of(
           len_axes.begin(), len_axes.end(), [](const auto& axis) { return axis.lower != 0; }))
    {
        throw std::logic_error(
            ROCFFT_CURRENT_FUNCTION
            + " is assuming full length axes, non-zero lower bounds are rejected for length axes.");
    }

    const bool complex_fft = fft_type == rocfft_transform_type_complex_forward
                             || fft_type == rocfft_transform_type_complex_inverse;
    if(!complex_fft && fft_type != rocfft_transform_type_real_forward
       && fft_type != rocfft_transform_type_real_inverse)
    {
        throw std::invalid_argument("Unknown type of transform given to "
                                    + ROCFFT_CURRENT_FUNCTION);
    }

    // a set array type must match this data set's side of the transform
    const auto io = other(other_io);
    if(has_array_type()
       && !(complex_fft ? array_type_is_complex_but_not_hermitian(array_type)
            : is_real_domain(fft_type, io)
                ? (array_type == rocfft_array_type_real)
                : (array_type == rocfft_array_type_hermitian_interleaved)))
    {
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " found array type "
                               + PrintArrayType(array_type)
                               + " on the wrong side of the transform.");
    }

    if(complex_fft)
    {
        // simple copy (identical offsets in all planes)
        return std::make_optional<data_set_t>(*this);
    }

    // a Hermitian innermost length of 1 can only come from a real innermost length of 1
    if(is_real_domain(fft_type, other_io) && (*this)[0].logical_span() == 1
       && !other_innermost_length_is_odd)
    {
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires an odd real innermost length for a Hermitian"
                                      " innermost length of 1.");
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
    // copy data set and modify what needs be
    auto ret = std::make_optional<data_set_t>(*this);
    // the returned layout is never planar: a second plane's offset cannot be expressed
    // and real offsets are twice the hermitian ones
    if(offsets[1] != 0 || (!is_real_domain(fft_type, other_io) && offsets[0] % 2 != 0))
        return std::nullopt;
    const auto ret_array_type = !has_array_type() ? array_type
                                : is_real_domain(fft_type, other_io)
                                    ? rocfft_array_type_real
                                    : rocfft_array_type_hermitian_interleaved;
    ret->set_array_type_and_offsets(
        ret_array_type, {is_real_domain(fft_type, other_io) ? 2 * offsets[0] : offsets[0] / 2, 0});
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

data_set_t data_set_t::get_data_set_for_len_axes(const std::set<size_t>& len_indices) const
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

    data_set_t ret;
    ret.array_type = complex_counterpart(array_type);
    ret.precision  = precision;
    ret.offsets    = offsets;
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

bool data_set_t::has_consistent_embedding() const
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

data_set_t data_set_t::get_embedding_data_set() const
{
    if(!embedding)
        return *this;
    if(!has_consistent_embedding())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " detected an inconsistent embedding map");

    data_set_t ret;
    ret.array_type = is_embedded() ? complex_counterpart(array_type) : array_type;
    ret.precision  = precision;
    ret.offsets    = offsets;
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

bool data_set_t::is_embedded() const
{
    return embedding.has_value() && embedding->size() > get_len_rank();
}

std::vector<size_t> data_set_t::corresponding_axes_in_embedding() const
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

data_set_t data_set_t::split_length_axis(size_t dim, std::optional<size_t> factor) const
{
    if(dim >= get_len_rank())
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + " got length axis index "
                                    + std::to_string(dim) + " for a layout of "
                                    + std::to_string(get_len_rank()) + " length axes.");
    const auto& axis = len_axes[dim];
    const auto  span = axis.logical_span();
    if(!factor)
    {
        // the largest divisor d with d * d <= span is the one closest to sqrt(span)
        factor = 1;
        for(size_t d = 2; d * d <= span; d++)
            if(span % d == 0)
                factor = d;
        if(*factor == 1)
            throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + " found no non-trivial factor of "
                                        + std::to_string(span) + " (length axis "
                                        + std::to_string(dim) + ").");
    }
    if(*factor <= 1 || *factor >= span || span % *factor != 0)
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + ": " + std::to_string(*factor)
                                    + " is not a non-trivial factor of " + std::to_string(span)
                                    + " (length axis " + std::to_string(dim) + ").");
    if(axis.lower % *factor != 0)
        throw std::invalid_argument(
            ROCFFT_CURRENT_FUNCTION + ": partial range [" + std::to_string(axis.lower) + ", "
            + std::to_string(axis.upper) + ") along length axis " + std::to_string(dim)
            + " is not aligned on multiples of " + std::to_string(*factor) + ".");

    data_set_t ret    = *this;
    ret.len_axes[dim] = axis_t{0, *factor, axis.inbuffer_stride};
    ret.len_axes.insert(
        ret.len_axes.begin() + dim + 1,
        axis_t{axis.lower / *factor, axis.upper / *factor, *factor * axis.inbuffer_stride});
    if(embedding)
    {
        // flattened axis indices past dim shift by one; dim's embedding axis is split alike
        ret.embedding->clear();
        for(const auto axis_idx : *embedding)
        {
            ret.embedding->push_back(axis_idx > dim ? axis_idx + 1 : axis_idx);
            if(axis_idx == dim)
                ret.embedding->push_back(dim + 1);
        }
    }
    return ret;
}
