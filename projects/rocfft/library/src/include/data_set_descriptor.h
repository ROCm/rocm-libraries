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

#ifndef DATA_SET_DESCRIPTOR_H
#define DATA_SET_DESCRIPTOR_H

#include "rocfft/rocfft.h"

#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

// Label distinguishing between input and output data layouts where or
// when such distinction may be relevant.
enum class io_data_label
{
    INPUT,
    OUTPUT
};

/**
 * @return `io_data_label::OUTPUT` for the argument value `io_data_label::INPUT` and vice versa.
 * @throw An `std::invalid_argument` is thrown if `io` is not `io_data_label::INPUT`
 * nor `io_data_label::OUTPUT`.
 */
io_data_label other(io_data_label io);

/**
 * @return An `std::string` of value "input" (resp. "output") if `io` is
 * `io_data_label::INPUT` (resp. `io_data_label::OUTPUT`).
 * 
 * @throw An `std::invalid_argument` is thrown if `io` is not `io_data_label::INPUT`
 * nor `io_data_label::OUTPUT`.
 */
std::string to_str(io_data_label io);

constexpr bool is_real_domain(rocfft_transform_type fft_type, io_data_label io)
{
    return (fft_type == rocfft_transform_type_real_forward && io == io_data_label::INPUT)
           || (fft_type == rocfft_transform_type_real_inverse && io == io_data_label::OUTPUT);
}

constexpr bool is_hermitian_domain(rocfft_transform_type fft_type, io_data_label io)
{
    return (fft_type == rocfft_transform_type_real_forward && io == io_data_label::OUTPUT)
           || (fft_type == rocfft_transform_type_real_inverse && io == io_data_label::INPUT);
}

/**
 * @brief Helper structure encapsulating the details pertaining to the description
 * of data layouts mapping hyper-rectangular logical index ranges (full or partial)
 * to locations ("offsets") within data allocations.
 *
 * The set of all logical coordinates in the data layout is defined as the
 * Cartesian product between
 * - the set of logical "length" coordinates;
 * - the set of logical "batch" coordinates.
 * Elements having different batch coordinates are considered strictly independent
 * of one another.
 *
 * The descriptor also captures one in-buffer offset (in elements) per plane of the
 * data set: two for planar array types, one otherwise (an unset array type counts
 * as non-planar). Invariant: the second offset is `0` unless the array type is
 * planar; the array type and the offsets only change together to preserve it.
 */
struct data_descriptor_t
{
    /**
     * @brief Constructs a new `data_descriptor_t` object with explicit member values.
     * 
     * @param[in] lower lower bounds along all axes of the logical index range
     * (lower bounds are included).
     * @param[in] upper upper bounds along all axes of the logical index range
     * (upper bounds are excluded).
     * @param[in] strides in-buffer strides associated with all axes of the
     * logical index range.
     * @param[in] batch_rank number of batch axes in the logical index range.
     * Default value is `1`.
     * 
     * @note All vector arguments implicitly consider that length axes are listed
     * first followed by `batch_rank` batch axes.
     * 
     * @throw An `std::invalid_argument` is thrown if any of the following is detected:
     * 
     * - the number of batch axes, i.e., `batch_rank`, or the (deduced) number of
     * length axes is `0`;
     * 
     * - `lower`, `upper`, and/or `strides` do not have the same size;
     * 
     * - any element of `lower` is found strictly larger than the corresponding
     *   element of `upper`.
     * 
     */
    data_descriptor_t(const std::vector<size_t>& lower,
                      const std::vector<size_t>& upper,
                      const std::vector<size_t>& strides,
                      size_t                     batch_rank = 1);

    /**
     * @brief Constructs a new `data_descriptor_t` object with explicit member values,
     * including the precision, array type and offsets of the data set.
     *
     * @param[in] precision precision of the data set's elements.
     * @param[in] array_type array type of the data set.
     * @param[in] lower lower bounds along all axes of the logical index range
     * (lower bounds are included).
     * @param[in] upper upper bounds along all axes of the logical index range
     * (upper bounds are excluded).
     * @param[in] strides in-buffer strides associated with all axes of the
     * logical index range.
     * @param[in] batch_rank number of batch axes in the logical index range.
     * Default value is `1`.
     * @param[in] offsets in-buffer offsets of the data set's planes (only the
     * first one is used for non-planar array types). Default value is `{0, 0}`.
     *
     * @note `precision` and `array_type` are deliberately listed first: both are
     * unscoped enums that implicitly convert to `size_t`, so trailing positions
     * would let a call with a missing argument silently bind to the overload
     * without them (using an enum value as `batch_rank`).
     *
     * @note All vector arguments implicitly consider that length axes are listed
     * first followed by `batch_rank` batch axes.
     *
     * @throw An `std::invalid_argument` is thrown in the same cases as the
     * overload without `precision` and `array_type`, or if `offsets[1]` is not
     * `0` while `array_type` is not planar.
     */
    data_descriptor_t(rocfft_precision             precision,
                      rocfft_array_type            array_type,
                      const std::vector<size_t>&   lower,
                      const std::vector<size_t>&   upper,
                      const std::vector<size_t>&   strides,
                      size_t                       batch_rank = 1,
                      const std::array<size_t, 2>& offsets    = {0, 0});

    /**
     * @brief Constructs a new `data_descriptor_t` object capturing a full range of logical
     * indices with one batch axis.
     * 
     * @param[in] lengths spans of the logical index range along all length axes.
     * @param[in] strides in-buffer strides associated with all length axes.
     * @param[in] batch span of the logical index range along its batch axis.
     * @param[in] distance in-buffer stride associated with the batch axis.
     * 
     * @throw An `std::invalid_argument` is thrown if `lengths` or `strides`
     * are empty or have different sizes.
     */
    static data_descriptor_t full_layout(const std::vector<size_t>& lengths,
                                         const std::vector<size_t>& strides,
                                         size_t                     batch,
                                         size_t                     distance);

    /**
     * @brief Constructs a new `data_descriptor_t` object capturing a full range of logical
     * indices with possibly many batch axes, and default in-buffer strides (enforcing
     * in-buffer contiguity for the innermost length axis).
     * 
     * @param[in] lengths spans of the logical index range along all length axes.
     * @param[in] batches spans of the logical index range along all batch axes.
     * @param[in] real_case_with_padding flag setting the in-buffer stride of the
     * layout's first non-contiguous axis to the value that is required in real
     * domain for real, in-place Discrete Fourier Transforms.
     * 
     * @throw An `std::invalid_argument` is thrown if `lengths` is empty.
     */
    static data_descriptor_t default_full_layout(const std::vector<size_t>& lengths,
                                                 const std::vector<size_t>& batches,
                                                 bool real_case_with_padding = false);

    // convenience overload for the common unidimensional batch case
    inline static data_descriptor_t default_full_layout(const std::vector<size_t>& lengths,
                                                        size_t                     batch,
                                                        bool real_case_with_padding = false)
    {
        return default_full_layout(lengths, std::vector<size_t>(1, batch), real_case_with_padding);
    }

    /**
     * @return The number of length axes.
     */
    size_t get_len_rank() const;
    /**
     * @return The number of batch axes.
     */
    size_t get_batch_rank() const;
    /**
     * @return The total number of axes (length and batch axes).
     */
    size_t get_full_rank() const;
    /**
     * @return The array type of the data set (`rocfft_array_type_unset` if not provided).
     */
    rocfft_array_type get_array_type() const;
    /**
     * @return The precision of the data set.
     * @throw std::logic_error if no precision was provided (see `has_precision`).
     */
    rocfft_precision get_precision() const;
    /**
     * @return Whether a precision was provided for the data set.
     */
    bool has_precision() const;
    /**
     * @return The number of planes of the data set: `2` for planar array types,
     * `1` otherwise (including an unset array type).
     */
    size_t num_planes() const;
    /**
     * @param[in] plane index of the plane of interest. Default value is `0`.
     * @return The in-buffer offset (in elements) of the data set's plane `plane`.
     * @throw An `std::out_of_range` is thrown if `plane >= num_planes()`.
     */
    size_t get_offset(size_t plane = 0) const;
    /**
     * @return The spans of the logical index range along all its axes (length followed by batch axes).
     */
    std::vector<size_t> lengths_and_batches() const;
    /**
     * @return The in-buffer strides associated with all axes (length followed by batch axes).
     */
    std::vector<size_t> strides_and_distances() const;
    /**
     * @return The upper bounds of the logical index range along all its axes (length and batch).
     */
    std::vector<size_t> upper() const;
    /**
     * @return The lower bounds of the logical index range along all its axes (length and batch).
     */
    std::vector<size_t> lower() const;
    /**
     * @return The spans of the logical index range along its length axes.
     */
    std::vector<size_t> lengths() const;
    /**
     * @return The in-buffer strides associated with length axes.
     */
    std::vector<size_t> strides() const;
    /**
     * @return The spans of the logical index range along its batch axes.
     */
    std::vector<size_t> batches() const;
    /**
     * @return The in-buffer strides associated with batch axes.
     */
    std::vector<size_t> distances() const;

    /**
     * @return The span of the logical index range along its (lone) batch axis.
     * 
     * @throw An `std::logic_error` is thrown if the object's number of batch axes is not 1.
     */
    size_t batch() const;

    /**
     * @return The in-buffer stride associated with the (lone) batch axis.
     * 
     * @throw An `std::logic_error` is thrown if the object's number of batch axes is not 1.
     */
    size_t distance() const;

    /**
     * @return `true` iff the logical index range is empty.
     */
    bool is_empty() const;

    /**
     * @return A vector of in-buffer strides that would make the data layout contiguous
     * in memory. If the current layout is found to be contiguous, this object's in-buffer
     * strides are returned unchanged. Otherwise, the returned values are such that the
     * innermost layout axis is given a unit in-buffer stride.
     */
    std::vector<size_t> contiguous_strides_and_distances() const;

    /**
     * @param[in] other data descriptor which must be dimensionally consistent with this one
     * (same number of length and batch axes) and must logically contain this layout's
     * very first element.
     * 
     * @return The in-buffer offset for this layout's very first element in a buffer that
     * observes the `other` data layout. This is a plane-independent offset relative to
     * `other`'s first element, i.e., the offsets captured by either object are ignored.
     * 
     * @throw An `std::invalid_argument` is thrown if `other` is not dimensionally consistent
     * with this object or does not contain its very first element.
     */
    size_t offset_in(const data_descriptor_t& other) const;

    /**
     * @param[in] other data descriptor which must be dimensionally consistent with this one
     * (same number of length and batch axes).
     * 
     * @return `true` iff this layout represents a continuous chunk of `other`, i.e., iff
     * - both layouts have identical strides along all axes that have non-unit logical spans;
     * - both layouts have identical logical index ranges along all axes except
     *   for the `other`'s slowest-varying one;
     * - this layout's logical range is contained by the `other` along its slowest-varying axis.
     *
     * @note The offsets captured by either object are ignored.
     * 
     * @throw An `std::invalid_argument` is thrown if `other` is not dimensionally consistent
     * with this object.
     */
    bool is_continuous_in(const data_descriptor_t& other) const;

    /**
     * @return A string reporting this object's state, including its precision
     * and array type ("unset" for either if not provided), and the offsets of all
     * its planes (see `num_planes`).
     */
    std::string str() const;

    /**
     * @return The volume of this object's hyper-rectangular logical index range.
     */
    size_t logical_count() const;
    /**
     * @return The (minimum) number of elements required in a buffer to store the
     * data layout described by this object, *excluding* its offsets, i.e., counted
     * from the element at `get_offset(plane)` in every plane (plane `plane` of a
     * buffer must hold `get_offset(plane) + buffer_element_count()` elements).
     */
    size_t buffer_element_count() const;

    /**
     * @return The size in bytes of one element of the data set, as defined by its
     * precision and array type. For planar array types, this is the size of one
     * real component (elements are split across two planes).
     * @throw An `std::logic_error` is thrown if the precision or the array type
     * is unset.
     */
    size_t element_size() const;

    /**
     * @return The (minimum) number of bytes required in a buffer to store the data
     * layout described by this object, i.e., `buffer_element_count() * element_size()`.
     * For planar array types, this is the number of bytes required *per plane*.
     * @throw An `std::logic_error` is thrown if the precision or the array type
     * is unset.
     */
    size_t buffer_size_bytes() const;

    /**
     * @return The largest in-buffer position (in elements, relative to the start of the
     * buffer, offsets included) of an element of the data set, over all its planes, i.e.,
     * the maximum over all planes of `get_offset(plane) + sum_d (span_d - 1) * stride_d`
     * (equivalently, the maximum over planes of
     * `get_offset(plane) + buffer_element_count() - 1`). Lower bounds do not contribute:
     * as for `buffer_element_count`, the logical range's first element sits at
     * `get_offset(plane)`. `max_offset() - get_offset(plane)` is thus the largest index
     * reached relative to the start of plane `plane`, for any plane.
     * @note Computed in `size_t`, so values above 32-bit limits are represented exactly
     * (wrap-around beyond `size_t` is not detected).
     * @throw An `std::logic_error` is thrown if the logical index range is empty
     * (see `is_empty`): no element is reached.
     */
    size_t max_offset() const;

    /**
     * @return The largest in-buffer stride (in elements) over all axes, length and batch
     * axes alike, including axes of unit or zero logical span (their strides are still
     * well-defined values a kernel may be given). `0` if the object has no axis.
     * @note Not bounded by `max_offset`: the stride of a unit-span axis may be arbitrarily
     * large without contributing to any position.
     */
    size_t max_stride() const;

    /**
     * @param[in] new_array_type array type to be given to the returned object.
     * @return A copy of this object reinterpreted as `new_array_type`, describing
     * the very same buffer bytes. The precision is never changed.
     * - Between non-real array types of the same format (interleaved or planar), or
     *   from real to real, this is a relabel: axes and offsets are kept.
     * - From real to interleaved complex/hermitian: the bounds of length axis 0 are
     *   halved (its stride stays 1), all other strides and distances and the offset
     *   are halved, i.e., real element `2k` (resp. `2k + 1`) is the real (resp.
     *   imaginary) part of complex element `k`.
     * - From interleaved complex/hermitian to real: the reverse, i.e., the bounds
     *   of length axis 0, all other strides and distances and the offset are doubled.
     * - From real to planar complex/hermitian: with stride `s` along length axis 0
     *   and offset `o`, the bounds of length axis 0 are halved, its stride becomes
     *   `2s` and the offsets become `{o, o + s}`; nothing else changes, i.e., real
     *   element `2k` (resp. `2k + 1`) is element `k` of plane 0 (resp. plane 1).
     * - From planar complex/hermitian to real: the reverse, i.e., the bounds of
     *   length axis 0 are doubled, its stride is halved and the offsets become
     *   `{get_offset(0), 0}`.
     *
     * @note A planar object obtained from real data describes two planes within
     * the real data's buffer: callers must give both planes that same buffer.
     * Likewise, reinterpreting planar data as real data is only meaningful if both
     * planes lie in the same buffer, which this function cannot verify.
     *
     * Embedded objects are supported iff their length axis 0 is the embedding data
     * set's length axis 0: the embedding map is kept and `get_embedding_layout()`
     * of the result is the reinterpreted `get_embedding_layout()` of this object.
     * @throw An `std::invalid_argument` is thrown if this object's array type or
     * `new_array_type` is unset, if the request switches between interleaved and
     * planar formats, or, for requests between real and complex/hermitian array
     * types, if this object is embedded with a length axis 0 that is not the
     * embedding's, or if
     * - (real <-> interleaved) the stride of length axis 0 is not 1, or (from real
     *   only) any bound of length axis 0, any other stride or distance, or the
     *   offset is odd;
     * - (real to planar) any bound of length axis 0 is odd;
     * - (planar to real) the stride of length axis 0 is odd or
     *   `get_offset(1) != get_offset(0) + stride / 2` for that stride.
     */
    data_descriptor_t reinterpret_as(rocfft_array_type new_array_type) const;

    /**
     * @return `true` iff this object describes a contiguous data layout, i.e., iff all
     * in-buffer elements between offset 0 and logical_count() (excluded) correspond
     * to a distinct element in logical range. Offsets are ignored, i.e., in-buffer
     * locations are relative to `get_offset(plane)` in every plane.
     */
    bool is_contiguous() const;

    /**
     * @param[in] other data descriptor to be compared to this one.
     * @return `true` iff this object and `other` are strictly identical, including
     * their precisions, array types (an unset value only equals an unset value) and
     * offsets.
     */
    bool operator==(const data_descriptor_t& other) const;

    /**
     * @param[in] other data descriptor which must be dimensionally consistent with this one
     * (same number of length and batch axes).
     * @return `true` iff all axes of either layout cover the same logical index range.
     * 
     * @throw An `std::invalid_argument` is thrown if `other` is not dimensionally consistent
     * with this object.
     */
    bool has_same_logical_range_as(const data_descriptor_t& other) const;

    /**
     * @param[in] first a data_descriptor_t object
     * @param[in] second a data_descriptor_t object that's dimensionally consistent with `first`
     * @return A data_descriptor_t object capturing the intersection of the logical ranges
     * between `first` and `second`. The in-buffer strides of the constructed intersection
     * are set to their default contiguous values and its offsets are zero: it describes
     * a fresh contiguous buffer, regardless of where `first` and `second` start in theirs.
     * The returned object has `first`'s array type and the common precision.
     * 
     * @throw An `std::invalid_argument` is thrown if `first` and `second` are not
     * dimensionally consistent, have different embeddings (if any), have different
     * precisions (unset only matches unset), or have array types that are neither
     * identical nor the hermitian and complex variants of the same format
     * (interleaved or planar).
     */
    static data_descriptor_t make_contiguous_intersection_of(const data_descriptor_t& first,
                                                             const data_descriptor_t& second);

    /**
     * @param[in] other data descriptor against which ranks are to be compared
     * @return `true` if this data descriptor is dimensionally consistent with `other` 
     */
    bool is_dimensionally_consistent_with(const data_descriptor_t& other) const;

    /**
     * @brief Reports the order of length axis indices if sorting them by increasing
     * in-buffer strides (possibly pinning the innermost axis).
     * 
     * @param[in] pin_innermost_axis flag enforcing the front element of the returned
     * vector to be 0 regardless of the actual stride associated with the innermost
     * length axis.
     * @return A vector `v` that's a permutation of `{0, 1, ..., get_len_rank() - 1}`
     * and such that `len_axes[v[i]].stride` is no greater than `len_axes[v[j]].stride`
     * if `i` < `j` (excluding `i == 0` and `j == 0` if `pin_innermost_axis` is true)
     */
    std::vector<size_t> length_axes_by_increasing_strides(bool pin_innermost_axis) const;

    /**
     * @brief Verifies whether this object's layout is consistent as input
     * (resp. output) for specific types of in-place Discrete Fourier Transforms
     * and, if so, returns the corresponding output (resp. input) layout. This
     * object must have full data range representation for all length axes (all
     * lower bounds equal to zero).
     * 
     * @param[in] other_io I/O label for the data descriptor to be returned. Explicitly,
     * the calling object's layout is considered an input (resp. output) layout
     * if the argument value is `io_data_label::OUTPUT` (resp. `io_data_label::INPUT`)
     * @param[in] fft_type intended type of (in-place) Fourier Transform
     * @param[in] other_innermost_length_is_odd flag indicating whether the
     * logical span of the innermost length axis in the corresponding layout is
     * odd (if `true`) or not (if `false`). This is ignored (and can be safely
     * omitted in calls) *unless* the data layout to be returned corresponds to
     * the input of a real forward transform or the output of a real inverse
     * transform.
     * 
     * @return An `std::optional<data_descriptor_t>` object which has a value set
     * iff a corresponding layout for in-place operation does actually exist. Its
     * precision is this object's. For complex transforms, its array type and its
     * offsets are this object's; for real transforms, its array type is
     * `rocfft_array_type_real` (resp. `rocfft_array_type_hermitian_interleaved`) if
     * the returned layout is in real (resp. hermitian) domain, unless this object's
     * array type is unset, in which case it remains unset, and its (single) offset is
     * such that real offsets are twice the hermitian ones (no value is set if a real
     * offset to be halved is odd, or if this object's second offset is non-zero).
     * 
     * @note This function does not verify if either layout is self-aliasing.
     * 
     * @warning Should not be used for distributed layouts, unless length axes are
     * all undistributed.
     * 
     * @throw An `std::logic_error` is thrown if the current object is an empty
     * layout or has any non-zero lower bound for a length axis.
     * An `std::invalid_argument` is thrown if `fft_type` is not an expected value
     * or if `other_io` is not an expected value.
     * 
     */
    std::optional<data_descriptor_t> get_other_inplace_layout_for(io_data_label         other_io,
                                                                  rocfft_transform_type fft_type,
                                                                  bool other_innermost_length_is_odd
                                                                  = false) const;

    /**
     * @return The data descriptor corresponding to a (possibly) sub-dimensional data set
     * embedded in what the current object describes. 
     * 
     * @param[in] len_indices set of indices of the length axes of the current object
     * that define the length axes of the sub-dimensional data set of interest.
     * @note Relative ordering of length axes is unchanged in the returned object when
     * compared to this object's. Remaining length axes (of the current object) are
     * turned into batch axes (in the returned object).
     * @throw An `std::invalid_argument` exception is thrown if `len_indices` is empty,
     * or if any of the values in `len_indices` is out of bounds.
     * @note If `len_indices = {0, 1, ..., get_len_rank() - 1}`, this function is
     * equivalent to a simple copy of the current object.
     * @note Precision, array type and offsets are passed through unchanged.
     */
    data_descriptor_t get_layout_for_len_axes(const std::set<size_t>& len_indices) const;

    /**
     * @return The data descriptor of the data set in which the data set described by the
     * current object is embedded.
     * @note If this object is not representing the data layout of an embedded data set,
     * this function is equivalent to a simple copy of the current object.
     * @note Precision, array type and offsets are passed through unchanged.
     * @throw An `std::logic_error` exception is thrown if the object's embedding map is
     * found inconsistent.
     */
    data_descriptor_t get_embedding_layout() const;

    /**
     * @return An `std::vector<size_t>` of the length axis indices in the embedding data
     * set to which the current object's length axes correspond.
     * @note If this object is not representing the data layout of an embedded data set,
     * this function returns `{0, 1, ..., get_len_rank() - 1}`.
     * @throw An `std::logic_error` exception is thrown if the object's embedding map is
     * found inconsistent.
     */
    std::vector<size_t> corresponding_axes_in_embedding() const;

    /**
     * @return `true` iff the current object is registered as the data descriptor of a
     * data set embedded within another, higher-dimensional one.
     */
    bool is_embedded() const;

    /**
     * @brief Splits a length axis into two length axes describing the same data.
     *
     * Length axis `dim`, of logical range [lower, upper) and in-buffer stride `s`, is
     * replaced by an inner length axis of full range [0, factor) and stride `s` (at
     * index `dim`) followed by an outer length axis of range [lower / factor,
     * upper / factor) and stride `factor * s` (at index `dim + 1`); former length axes
     * of index `> dim` are shifted by one. Logical coordinate `i` along `dim` maps to
     * `(i % factor, i / factor)` so every element keeps its in-buffer offset. Batch
     * axes are unchanged. If this object is embedded, the corresponding axis of the
     * embedding data set is split alike, i.e., the returned object's embedding layout
     * is the current embedding layout split along its corresponding axis. The array
     * type, precision (or lack thereof) and per-plane offsets are kept: the split only
     * re-indexes elements within each plane.
     *
     * @param[in] dim index of the length axis to split.
     * @param[in] factor span of the inner axis. If omitted, the divisor of the logical
     * span `L = upper - lower` closest to `sqrt(L)` is used, ties going to the smaller
     * one (equivalently, the largest divisor `d` of `L` such that `d * d <= L`).
     * @return The split data descriptor.
     *
     * @note A partial range along `dim` is representable after the split iff `lower`
     * and `upper` are both multiples of `factor`.
     *
     * @throw An `std::invalid_argument` is thrown if any of the following is detected:
     *
     * - `dim` is not smaller than `get_len_rank()`;
     *
     * - `factor` is omitted and `L` has no divisor other than 1 and `L` (`L` is 0, 1 or prime);
     *
     * - `factor` is 1, not smaller than `L`, or does not divide `L`;
     *
     * - `lower` (hence `upper`) is not a multiple of `factor`.
     */
    data_descriptor_t split_length_axis(size_t                dim,
                                        std::optional<size_t> factor = std::nullopt) const;

    //-------------------------------------------------------------------------
    //                        DEFAULT COPIES AND MOVES
    //-------------------------------------------------------------------------

    data_descriptor_t(const data_descriptor_t&) = default;
    data_descriptor_t& operator=(const data_descriptor_t&) = default;
    data_descriptor_t(data_descriptor_t&&)                 = default;
    data_descriptor_t& operator=(data_descriptor_t&&) = default;

    // Prevent ill-defined objects as much as possible by privatizing
    // content-modifying members and default constructor.
private:
    struct axis_t
    {
        size_t lower;
        size_t upper;
        size_t inbuffer_stride;

        inline bool has_same_logical_range_as(const axis_t& other) const
        {
            return lower == other.lower && upper == other.upper;
        }
        inline size_t logical_span() const
        {
            return upper - lower;
        }
        inline bool operator==(const axis_t& other) const
        {
            // stride is irrelevant when comparing two layout's axes of unit logical range
            return has_same_logical_range_as(other)
                   && (logical_span() == 1 || inbuffer_stride == other.inbuffer_stride);
        }
        inline bool logically_contains(const size_t& coordinate) const
        {
            return lower <= coordinate && coordinate < upper;
        }

        inline bool logically_contains(const axis_t& other) const
        {
            return logically_contains(other.lower) && logically_contains(other.upper - 1);
        }
        inline bool empty() const
        {
            return lower == upper;
        }
    };

    data_descriptor_t() = default;

    // Storing length and batch axes in two vectors is an internal representation
    // detail (hidden behind semantic accessors). It's preferred over a single flat
    // vector plus a batch_rank counter because the split *is* the rank bookkeeping.
    std::vector<axis_t> len_axes;
    std::vector<axis_t> batch_axes;

    // Array type of the data set's elements; unset until provided.
    rocfft_array_type array_type = rocfft_array_type_unset;

    // In-buffer offsets of the data set's planes, see set_array_type_and_offsets.
    std::array<size_t, 2> offsets = {0, 0};

    // Marker for "no precision provided". Stored as the enum's underlying integer
    // type because rocfft_precision has no unset value and out-of-range enum
    // values are undefined behavior. Never exposed outside this struct.
    static constexpr std::underlying_type_t<rocfft_precision> precision_unset
        = std::numeric_limits<std::underlying_type_t<rocfft_precision>>::max();

    // Precision of the data set's elements; precision_unset until provided.
    std::underlying_type_t<rocfft_precision> precision = precision_unset;

    // If the current object captures the data layout for a sub-dimensional data set
    // embedded within another, an embedding map is required to keep track of which of
    // the current object's axes correspond to the embedding set's length axes. If
    // `embedding` has a value set, the current object captures the layout for data
    // embedded within a higher dimensional data set of rank `embedding->size()`, for
    // which the `j`-th length axis (`0 <= j < embedding->size()`) is the current
    // object's axis of (flattened) index `(*embedding)[j]`.
    std::optional<std::vector<size_t>> embedding;

    inline bool has_consistent_embedding() const;

    // Stores the underlying value of `p`; used by the constructor and by friends.
    void set_precision(rocfft_precision p);

    /**
     * @brief Sets the array type and the offsets of the data set together.
     * @throw An `std::invalid_argument` is thrown (and nothing is modified) if
     * `new_offsets[1]` is not `0` while `new_array_type` is not planar.
     */
    void set_array_type_and_offsets(rocfft_array_type            new_array_type,
                                    const std::array<size_t, 2>& new_offsets);

    /**
     * @brief Implementation-simplifying helper accessor for the length and batch
     * axes of the layout.
     * 
     * @param[in] axis_idx flattened axis index in [0, get_full_rank() [
     * @return `len_axes[axis_idx]` if `axis_idx < get_len_rank()`,
     * `batch_axes[axis_idx - get_len_rank()]` otherwise.
     */
    const axis_t& operator[](size_t axis_idx) const;
    axis_t&       operator[](size_t axis_idx);

    /**
     * @return Flattened index of the layout's axis of largest in-buffer
     * stride (and non-trivial logical range)
     */
    size_t slowest_varying_axis() const;

    /**
     * @brief Empties all layout axes and resets the embedding map. Precision,
     * array type and offsets are left untouched.
     */
    void clear();

    /**
     * @brief Shuffles the length axes of this layout such that, upon
     * return, `len_axes[dim]` is the former `len_axes[len_axis_order[dim]]`
     * 
     * @param[in] len_axis_order a vector that is a permutation of
     * `{0, 1, ..., get_len_rank() - 1}`
     *
     * @warning Unless trivial, this operation breaks consistency between this
     * object and (previously-created) embedded ones obtained from the current
     * object, if any. If this object is embedded itself, consistency with its
     * embedding data set is maintained under this re-ordering operation.
     * 
     * @throw An `std::invalid_argument` is thrown if `len_axis_order`
     * is not a permutation of `{0, 1, ..., get_len_rank() - 1}`.
     */
    void reorder_length_axes(const std::vector<size_t>& len_axis_order);

    /**
     * @brief Resets the object's state to capture a full range of logical
     * indices with either prescribed or default strides and/or distance.
     * Precision, array type and offsets are left untouched.
     * 
     * @param[in] lengths spans of the logical index range along all length axes.
     * @param[in] strides in-buffer strides associated with all length axes.
     * If empty, default in-buffer strides (enforcing in-buffer contiguity for
     * the innermost length axis) are set.
     * @param[in] batches spans of the logical index range along all batch axes.
     * @param[in] distances in-buffer strides associated with all batch axes.
     * If empty, default in-buffer strides deduced from the last length axis are
     * set.
     * @param[in] real_case_with_padding flag setting the in-buffer stride of the
     * layout's first non-contiguous axis to the value that is required in real
     * domain for real, in-place Discrete Fourier Transforms, if a default value
     * must be set. This flag is this irrelevant if `strides.size() > 1` or if
     * `lengths.size() == 1 && !distances.empty()`.
     * 
     * @throw An `std::invalid_argument` is thrown if any of the following is detected:
     * 
     * - `lengths` or `batches` is empty;
     * 
     * - `strides` (resp. `distances`) is not empty and does not have the same
     * size as `lengths` (resp. `batches`).
     */
    void full_range_reset(const std::vector<size_t>& lengths,
                          const std::vector<size_t>& strides,
                          const std::vector<size_t>& batches,
                          const std::vector<size_t>& distances,
                          bool                       real_case_with_padding);

    // Friends that need access to private members and/or default constructor below.
    // -----------------------------------------------------------------------------
    // Strides/distances are set by users via descriptions before even knowing the
    // lengths/ranges
    friend rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description,
                                                                 const rocfft_array_type,
                                                                 const rocfft_array_type,
                                                                 const size_t*,
                                                                 const size_t*,
                                                                 const size_t,
                                                                 const size_t*,
                                                                 const size_t,
                                                                 const size_t,
                                                                 const size_t*,
                                                                 const size_t);
    // Descriptions need access to private members to
    // - remove trivial axes (of unit logical range) from full and partial layouts upon finalization;
    // - possibly re-order relevant layouts' length axes by increasing in-buffer strides.
    friend struct rocfft_plan_description_t;
    // Fields need access to private members to erase or reorder length axes in their bricks'
    // layouts and to compare individual axes for undistributed-dimension detection.
    friend struct rocfft_field_t;
    // rocfft-internal-test drives private members directly (see tests/data_descriptor_test_access.h).
    friend struct data_descriptor_test_access;
};

#endif // DATA_SET_DESCRIPTOR_H
