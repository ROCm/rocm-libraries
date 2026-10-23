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

#include "data_set_descriptor.h"
#include "../../shared/arithmetic.h"
#include "../../shared/precision_type.h"
#include "../../shared/ptrdiff.h"
#include "../../shared/rocfft_enums_vs_fft_enums.h"
#include "rocfft/rocfft.h"
#include "rocfft_current_function.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <type_traits>

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

    // true iff a and b are identical or are the hermitian and complex variants
    // of the same format (interleaved or planar)
    bool same_format_up_to_hermitian(rocfft_array_type a, rocfft_array_type b)
    {
        return a == b
               || (array_type_is_complex(a) && array_type_is_complex(b)
                   && array_type_is_interleaved(a) == array_type_is_interleaved(b));
    }

    // smallest (resp. largest) integer x such that x * b >= a (resp. x * b <= a), for b > 0
    ptrdiff_t div_up(ptrdiff_t a, ptrdiff_t b)
    {
        return a / b + (a % b > 0 ? 1 : 0);
    }
    ptrdiff_t div_down(ptrdiff_t a, ptrdiff_t b)
    {
        return a / b - (a % b < 0 ? 1 : 0);
    }

    // Extended Euclid algorithm: returns gcd(a, b) and sets p, q such that
    // p * a + q * b == gcd(a, b) (requires a > 0 and b > 0).
    size_t ext_euclid_gcd(size_t a, size_t b, ptrdiff_t& p, ptrdiff_t& q)
    {
        p                 = 1;
        q                 = 0;
        ptrdiff_t  p_next = 0, q_next = 1;
        const bool swapped = a < b;
        if(swapped)
            std::swap(a, b);
        while(b != 0)
        {
            const ptrdiff_t ratio  = static_cast<ptrdiff_t>(a / b);
            const size_t    next_b = a % b;
            const ptrdiff_t p_tmp  = p - ratio * p_next;
            const ptrdiff_t q_tmp  = q - ratio * q_next;
            a                      = b;
            b                      = next_b;
            p                      = p_next;
            q                      = q_next;
            p_next                 = p_tmp;
            q_next                 = q_tmp;
        }
        if(swapped)
            std::swap(p, q);
        return a;
    }

    // Self-aliasing detection for strided index ranges, ported from the O(1)-storage
    // implementation of array_valid in shared/array_validator.cpp (branch
    // users/regan-amd/array_validation_alt_impl, commit d644cffc0c), minus its
    // diagnostics (equation counters, collision examples).
    // `axes` holds non-trivial axes only (span > 1, stride > 0), sorted by increasing stride.
    struct strided_axes_t
    {
        struct axis
        {
            size_t span;
            size_t stride;
        };
        std::vector<axis> axes;

        size_t strides_gcd() const
        {
            return std::accumulate(
                axes.begin() + 1, axes.end(), axes[0].stride, [](size_t acc, const axis& a) {
                    return std::gcd(acc, a.stride);
                });
        }
        // position of multi-index (1, 1, ..., 1)
        size_t first_nondegenerate_entry() const
        {
            return std::accumulate(axes.begin(),
                                   axes.end(),
                                   size_t(0),
                                   [](size_t acc, const axis& a) { return acc + a.stride; });
        }
        // position of multi-index (span_0 - 1, span_1 - 1, ...)
        size_t last_entry() const
        {
            return std::accumulate(
                axes.begin(), axes.end(), size_t(0), [](size_t acc, const axis& a) {
                    return acc + a.stride * (a.span - 1);
                });
        }

        // true iff sum_i x_i * axes[i].stride == address for some 0 <= x_i < axes[i].span
        bool reaches(size_t address) const
        {
            if(address == 0)
                return true;
            if(address > last_entry() || address % strides_gcd() != 0)
                return false;
            if(axes.size() == 1)
                return true;
            if(axes.size() == 2)
            {
                // particular solution p * s0 + q * s1 == address from Bezout's coefficients,
                // then look for k such that 0 <= p + k * s1 / g < l0 and 0 <= q - k * s0 / g < l1
                ptrdiff_t       p, q;
                const size_t    g     = ext_euclid_gcd(axes[0].stride, axes[1].stride, p, q);
                const ptrdiff_t scale = static_cast<ptrdiff_t>(address / g);
                p *= scale;
                q *= scale;
                const ptrdiff_t l0    = static_cast<ptrdiff_t>(axes[0].span);
                const ptrdiff_t l1    = static_cast<ptrdiff_t>(axes[1].span);
                const ptrdiff_t s0_g  = static_cast<ptrdiff_t>(axes[0].stride / g);
                const ptrdiff_t s1_g  = static_cast<ptrdiff_t>(axes[1].stride / g);
                const ptrdiff_t min_k = std::max(div_up(-p, s1_g), div_up(q - l1 + 1, s0_g));
                const ptrdiff_t max_k = std::min(div_down(l0 - 1 - p, s1_g), div_down(q, s0_g));
                return min_k <= max_k;
            }
            // more than 2 unknowns: enumerate the one of largest stride (smallest range)
            // and recurse on the others
            const strided_axes_t sub{{axes.begin(), axes.end() - 1}};
            const axis&          last  = axes.back();
            const size_t         max_k = std::min(last.span - 1, address / last.stride);
            for(size_t k = 0; k <= max_k; k++)
            {
                if(sub.reaches(address - k * last.stride))
                    return true;
            }
            return false;
        }

        // Two distinct multi-indices x, y collide iff the axes split into disjoint, non-empty
        // sets A and B with sum_{A} u_i * s_i == sum_{B} u_i * s_i, u_i = |x_i - y_i| < span_i.
        // `seed_call` is false only for the sub-arrays checked herein: their own sub-arrays
        // are sub-arrays of this one, already covered by other splits of the seed call.
        bool is_self_aliasing(bool seed_call) const
        {
            if(axes.size() < 2)
                return false;
            bool nested = true;
            for(size_t i = 0; i + 1 < axes.size(); i++)
                nested &= axes[i + 1].stride >= axes[i].span * axes[i].stride;
            if(nested)
                return false;

            std::vector<size_t> mask(axes.size());
            for(size_t lhs_size = axes.size() / 2; lhs_size > 0; lhs_size--)
            {
                std::fill(mask.begin(), mask.begin() + lhs_size, 0);
                std::fill(mask.begin() + lhs_size, mask.end(), 1);
                do
                {
                    // skip swapped copies of other splits
                    if(2 * lhs_size == axes.size() && mask[0] == 1)
                        continue;
                    strided_axes_t lhs, rhs;
                    for(size_t i = 0; i < axes.size(); i++)
                        (mask[i] ? rhs : lhs).axes.push_back(axes[i]);
                    if(seed_call && (lhs.is_self_aliasing(false) || rhs.is_self_aliasing(false)))
                        return true;
                    // solutions with any u_i == 0 are collisions of smaller sub-arrays (checked
                    // above), so a match is a non-zero common multiple of both sides' strides gcd,
                    // reached by both sides with all u_i >= 1
                    const size_t lcm = std::lcm(lhs.strides_gcd(), rhs.strides_gcd());
                    const size_t match_min
                        = lcm
                          * DivRoundingUp(std::max(lhs.first_nondegenerate_entry(),
                                                   rhs.first_nondegenerate_entry()),
                                          lcm);
                    const size_t match_max = std::min(lhs.last_entry(), rhs.last_entry());
                    for(size_t match = match_min; match <= match_max; match += lcm)
                    {
                        if(lhs.reaches(match) && rhs.reaches(match))
                            return true;
                    }
                } while(std::next_permutation(mask.begin(), mask.end()));
            }
            return false;
        }
    };
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

#define IO_DATA_DESCRIPTOR_ACCESSOR_BODY(AXIS_IDX)                                  \
    do                                                                              \
    {                                                                               \
        if(AXIS_IDX >= get_full_rank())                                             \
            throw std::out_of_range(ROCFFT_CURRENT_FUNCTION                         \
                                    + " accessing an out-of-range layout axis.");   \
        return AXIS_IDX < len_axes.size() ? len_axes[AXIS_IDX]                      \
                                          : batch_axes[AXIS_IDX - len_axes.size()]; \
    } while(0)

const data_descriptor_t::axis_t& data_descriptor_t::operator[](size_t axis_idx) const
{
    IO_DATA_DESCRIPTOR_ACCESSOR_BODY(axis_idx);
}

data_descriptor_t::axis_t& data_descriptor_t::operator[](size_t axis_idx)
{
    IO_DATA_DESCRIPTOR_ACCESSOR_BODY(axis_idx);
}

data_descriptor_t::data_descriptor_t(const std::vector<size_t>& lower,
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
                + " detected an invalid range of logical indices along layout axis "
                + std::to_string(dim) + ": lower (" + std::to_string(lower[dim])
                + ") is larger than upper (" + std::to_string(upper[dim]) + ").");
        }

        (*this)[dim].lower           = lower[dim];
        (*this)[dim].upper           = upper[dim];
        (*this)[dim].inbuffer_stride = strides[dim];
    }
}

data_descriptor_t::data_descriptor_t(rocfft_precision             precision,
                                     rocfft_array_type            array_type,
                                     const std::vector<size_t>&   lower,
                                     const std::vector<size_t>&   upper,
                                     const std::vector<size_t>&   strides,
                                     size_t                       batch_rank,
                                     const std::array<size_t, 2>& offsets)
    : data_descriptor_t(lower, upper, strides, batch_rank)
{
    set_array_type_and_offsets(array_type, offsets);
    set_precision(precision);
}

data_descriptor_t data_descriptor_t::full_layout(const std::vector<size_t>& lengths,
                                                 const std::vector<size_t>& strides,
                                                 size_t                     batch,
                                                 size_t                     distance)
{
    return data_descriptor_t{std::vector<size_t>(lengths.size() + 1, 0),
                             concatenate(lengths, batch),
                             concatenate(strides, distance),
                             1};
}

data_descriptor_t data_descriptor_t::default_full_layout(const std::vector<size_t>& lengths,
                                                         const std::vector<size_t>& batches,
                                                         bool real_case_with_padding)
{
    data_descriptor_t         ret;
    const std::vector<size_t> empty_for_default_strides_and_dist = {};
    ret.full_range_reset(lengths,
                         empty_for_default_strides_and_dist,
                         batches,
                         empty_for_default_strides_and_dist,
                         real_case_with_padding);
    return ret;
}

size_t data_descriptor_t::get_len_rank() const
{
    return len_axes.size();
}

size_t data_descriptor_t::get_batch_rank() const
{
    return batch_axes.size();
}
size_t data_descriptor_t::get_full_rank() const
{
    return len_axes.size() + batch_axes.size();
}

rocfft_array_type data_descriptor_t::get_array_type() const
{
    return array_type;
}

rocfft_precision data_descriptor_t::get_precision() const
{
    if(!has_precision())
    {
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " called on a descriptor without precision.");
    }

    return static_cast<rocfft_precision>(precision);
}

bool data_descriptor_t::has_precision() const
{
    return precision != precision_unset;
}

void data_descriptor_t::set_precision(rocfft_precision p)
{
    precision = static_cast<std::underlying_type_t<rocfft_precision>>(p);
}

void data_descriptor_t::set_array_type_and_offsets(rocfft_array_type            new_array_type,
                                                   const std::array<size_t, 2>& new_offsets)
{
    if(new_offsets[1] != 0 && !array_type_is_planar(new_array_type))
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires a zero second offset for non-planar array types.");
    array_type = new_array_type;
    offsets    = new_offsets;
}

size_t data_descriptor_t::num_planes() const
{
    return array_type_is_planar(array_type) ? 2 : 1;
}

size_t data_descriptor_t::get_offset(size_t plane) const
{
    if(plane >= num_planes())
        throw std::out_of_range(ROCFFT_CURRENT_FUNCTION + " accessing an out-of-range plane.");
    return offsets[plane];
}

std::vector<size_t> data_descriptor_t::lengths_and_batches() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].logical_span();

    return ret;
}

std::vector<size_t> data_descriptor_t::strides_and_distances() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].inbuffer_stride;

    return ret;
}

std::vector<size_t> data_descriptor_t::contiguous_strides_and_distances() const
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

std::vector<size_t> data_descriptor_t::lengths() const
{
    std::vector<size_t> ret(len_axes.size());
    for(size_t dim = 0; dim < len_axes.size(); dim++)
        ret[dim] = len_axes[dim].logical_span();

    return ret;
}

std::vector<size_t> data_descriptor_t::strides() const
{
    std::vector<size_t> ret(len_axes.size());
    for(size_t dim = 0; dim < len_axes.size(); dim++)
        ret[dim] = len_axes[dim].inbuffer_stride;

    return ret;
}

std::vector<size_t> data_descriptor_t::batches() const
{
    std::vector<size_t> ret(batch_axes.size());
    for(size_t dim = 0; dim < batch_axes.size(); dim++)
        ret[dim] = batch_axes[dim].logical_span();

    return ret;
}

std::vector<size_t> data_descriptor_t::distances() const
{
    std::vector<size_t> ret(batch_axes.size());
    for(size_t dim = 0; dim < batch_axes.size(); dim++)
        ret[dim] = batch_axes[dim].inbuffer_stride;

    return ret;
}

size_t data_descriptor_t::batch() const
{
    if(batch_axes.size() != 1)
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " is restricted to data layouts with only one batch axis.");
    return batch_axes[0].logical_span();
}

size_t data_descriptor_t::distance() const
{
    if(batch_axes.size() != 1)
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " is restricted to data layouts with only one batch axis.");
    return batch_axes[0].inbuffer_stride;
}

std::vector<size_t> data_descriptor_t::upper() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].upper;

    return ret;
}

std::vector<size_t> data_descriptor_t::lower() const
{
    std::vector<size_t> ret(get_full_rank());
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret[dim] = (*this)[dim].lower;

    return ret;
}

bool data_descriptor_t::is_empty() const
{
    bool ret = len_axes.empty() || batch_axes.empty();
    for(size_t dim = 0; !ret && dim < get_full_rank(); dim++)
        ret |= (*this)[dim].logical_span() == 0;
    return ret;
}

size_t data_descriptor_t::logical_count() const
{
    auto len_and_batches = lengths_and_batches();
    return product(len_and_batches.begin(), len_and_batches.end());
}

size_t data_descriptor_t::buffer_element_count() const
{
    return compute_ptrdiff(lengths_and_batches(), strides_and_distances());
}

size_t data_descriptor_t::element_size() const
{
    if(array_type == rocfft_array_type_unset)
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " called on a descriptor without array type.");
    return ::element_size(get_precision(), array_type);
}

size_t data_descriptor_t::buffer_size_bytes() const
{
    return buffer_element_count() * element_size();
}

size_t data_descriptor_t::max_offset() const
{
    if(is_empty())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION
                               + " is undefined for an empty logical index range.");
    return *std::max_element(offsets.begin(), offsets.begin() + num_planes())
           + buffer_element_count() - 1;
}

size_t data_descriptor_t::max_stride() const
{
    size_t ret = 0;
    for(size_t dim = 0; dim < get_full_rank(); dim++)
        ret = std::max(ret, (*this)[dim].inbuffer_stride);
    return ret;
}

data_descriptor_t data_descriptor_t::reinterpret_as(rocfft_array_type new_array_type) const
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

    // real <-> complex/hermitian: length axis 0 is split/merged, so it must be the
    // embedding's length axis 0 for get_embedding_layout() to commute with this
    if(embedding && (*embedding)[0] != 0)
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION
                                    + " requires length axis 0 of an embedded data set to be"
                                      " the embedding's length axis 0.");
    const bool to_complex = array_type_is_real(array_type);
    auto       ret        = *this;

    if(array_type_is_planar(to_complex ? new_array_type : array_type))
    {
        // real element 2k (resp. 2k + 1) is element k of plane 0 (resp. 1): with
        // stride s and offset o along real length axis 0, the planes have stride 2s
        // and offsets {o, o + s} (both planes then lie in the real data's buffer).
        // AIFFT-566 writes these offsets as "{o, o  s}" (lost symbol): "o + s" is
        // the only reading consistent with its reverse rule (offset1 - offset0 ==
        // stride0 / 2) and with the in-buffer positions of odd real elements.
        if(to_complex)
        {
            if(len_axes[0].lower % 2 != 0 || len_axes[0].upper % 2 != 0)
                throw std::invalid_argument(
                    ROCFFT_CURRENT_FUNCTION
                    + " requires even bounds along length axis 0 to reinterpret real data"
                      " as planar data.");
            ret.len_axes[0].lower /= 2;
            ret.len_axes[0].upper /= 2;
            ret.len_axes[0].inbuffer_stride *= 2;
            ret.set_array_type_and_offsets(new_array_type,
                                           {offsets[0], offsets[0] + len_axes[0].inbuffer_stride});
            return ret;
        }
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
    ret.len_axes[0].lower = convert(len_axes[0].lower);
    ret.len_axes[0].upper = convert(len_axes[0].upper);
    for(size_t dim = 1; dim < get_full_rank(); dim++)
        ret[dim].inbuffer_stride = convert((*this)[dim].inbuffer_stride);
    ret.set_array_type_and_offsets(new_array_type, {convert(offsets[0]), 0});
    return ret;
}

bool data_descriptor_t::is_contiguous() const
{
    return buffer_element_count() == logical_count();
}

bool data_descriptor_t::is_self_aliasing() const
{
    // an empty range has no two distinct indices
    if(is_empty())
        return false;
    strided_axes_t nontrivial;
    for(size_t dim = 0; dim < get_full_rank(); dim++)
    {
        const auto& axis = (*this)[dim];
        if(axis.logical_span() < 2)
            continue;
        if(axis.inbuffer_stride == 0)
            return true;
        nontrivial.axes.push_back({axis.logical_span(), axis.inbuffer_stride});
    }
    std::sort(nontrivial.axes.begin(), nontrivial.axes.end(), [](const auto& a, const auto& b) {
        return a.stride < b.stride;
    });
    return nontrivial.is_self_aliasing(true);
}

bool data_descriptor_t::operator==(const data_descriptor_t& other) const
{
    return len_axes == other.len_axes && batch_axes == other.batch_axes
           && array_type == other.array_type && precision == other.precision
           && offsets == other.offsets;
}

bool data_descriptor_t::has_same_logical_range_as(const data_descriptor_t& other) const
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

data_descriptor_t
    data_descriptor_t::make_contiguous_intersection_of(const data_descriptor_t& first,
                                                       const data_descriptor_t& second)
{
    if(!first.is_dimensionally_consistent_with(second))
        throw std::invalid_argument("Dimensionally-inconsistent arguments for "
                                    + ROCFFT_CURRENT_FUNCTION);
    if(first.embedding != second.embedding)
        throw std::invalid_argument("Different embeddings for arguments of "
                                    + ROCFFT_CURRENT_FUNCTION);
    if(first.precision != second.precision)
        throw std::invalid_argument("Different precisions for arguments of "
                                    + ROCFFT_CURRENT_FUNCTION);
    if(!same_format_up_to_hermitian(first.array_type, second.array_type))
        throw std::invalid_argument("Incompatible array types for arguments of "
                                    + ROCFFT_CURRENT_FUNCTION);

    data_descriptor_t ret;
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

std::vector<size_t>
    data_descriptor_t::length_axes_by_increasing_strides(bool pin_innermost_axis) const
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

void data_descriptor_t::reorder_length_axes(const std::vector<size_t>& len_axis_order)
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

size_t data_descriptor_t::slowest_varying_axis() const
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

size_t data_descriptor_t::offset_in(const data_descriptor_t& other) const
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

bool data_descriptor_t::is_continuous_in(const data_descriptor_t& other) const
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

std::string data_descriptor_t::str() const
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
    ret += has_precision() ? precision_name(get_precision()) : "unset";
    ret += " array type ";
    ret += fft_enum_to_string(fft_array_type_from_rocfft_array_type(array_type));
    ret += " offsets";
    for(size_t plane = 0; plane < num_planes(); plane++)
        ret += " " + std::to_string(offsets[plane]);

    if(is_embedded())
    {
        ret += " embedded within a " + std::to_string(embedding->size()) + "-dimensional data set";
    }
    return ret;
}

void data_descriptor_t::clear()
{
    len_axes.clear();
    batch_axes.clear();
    embedding.reset();
}

void data_descriptor_t::full_range_reset(const std::vector<size_t>& lengths,
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

bool data_descriptor_t::is_dimensionally_consistent_with(const data_descriptor_t& other) const
{
    return get_batch_rank() == other.get_batch_rank() && get_len_rank() == other.get_len_rank();
}

std::optional<data_descriptor_t>
    data_descriptor_t::get_other_inplace_layout_for(io_data_label         other_io,
                                                    rocfft_transform_type fft_type,
                                                    bool other_innermost_length_is_odd) const
{
    if(other_io != io_data_label::INPUT && other_io != io_data_label::OUTPUT)
        throw std::invalid_argument("Unknown I/O data label given to " + ROCFFT_CURRENT_FUNCTION);

    if(is_empty())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " queried on an empty layout");

    if(std::any_of(
           len_axes.begin(), len_axes.end(), [](const auto& axis) { return axis.lower != 0; }))
    {
        throw std::logic_error(
            ROCFFT_CURRENT_FUNCTION
            + " is assuming full length axes, non-zero lower bounds are rejected for length axes.");
    }

    if(fft_type == rocfft_transform_type_complex_forward
       || fft_type == rocfft_transform_type_complex_inverse)
    {
        // simple copy (identical offsets in all planes)
        return std::make_optional<data_descriptor_t>(*this);
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
    auto ret = std::make_optional<data_descriptor_t>(*this);
    // the returned layout is never planar: a second plane's offset cannot be expressed
    // and real offsets are twice the hermitian ones
    if(offsets[1] != 0 || (!is_real_domain(fft_type, other_io) && offsets[0] % 2 != 0))
        return std::nullopt;
    const auto ret_array_type = array_type == rocfft_array_type_unset ? array_type
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

data_descriptor_t
    data_descriptor_t::get_layout_for_len_axes(const std::set<size_t>& len_indices) const
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

    data_descriptor_t ret;
    ret.array_type = array_type;
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

bool data_descriptor_t::has_consistent_embedding() const
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

data_descriptor_t data_descriptor_t::get_embedding_layout() const
{
    if(!embedding)
        return *this;
    if(!has_consistent_embedding())
        throw std::logic_error(ROCFFT_CURRENT_FUNCTION + " detected an inconsistent embedding map");

    data_descriptor_t ret;
    ret.array_type = array_type;
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

bool data_descriptor_t::is_embedded() const
{
    return embedding.has_value() && embedding->size() > get_len_rank();
}

std::vector<size_t> data_descriptor_t::corresponding_axes_in_embedding() const
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

data_descriptor_t data_descriptor_t::split_length_axis(size_t                dim,
                                                       std::optional<size_t> factor) const
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

    data_descriptor_t ret = *this;
    ret.len_axes[dim]     = axis_t{0, *factor, axis.inbuffer_stride};
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
