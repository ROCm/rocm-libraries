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

#ifndef DATA_LAYOUT_H
#define DATA_LAYOUT_H

#include "rocfft/rocfft.h"

#include <cstring>
#include <optional>
#include <set>
#include <string>
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
 */
struct data_layout_t
{
    /**
     * @brief Constructs a new `data_layout_t` object with explicit member values.
     * 
     * @param[in] lower lower bounds along all axes of the logical index range
     * (lower bounds are included).
     * @param[in] upper upper bounds along all axes of the logical index range
     * (upper bounds are excluded).
     * @param[in] strides in-buffer strides associated with all axes of the
     * logical index range.
     * @param[in] batch_rank number of batch axes in the logical index range.
     * Default value is `1`.
     * @param[in] is_partial flag indicating whether the constructed object covers
     * a full data range (if `false`) or not (if `true`). Default value is `true`.
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
     *   element of `upper`;
     * 
     * - any element of `lower` is different than 0 yet `is_partial` is `false`.
     * 
     */
    data_layout_t(const std::vector<size_t>& lower,
                  const std::vector<size_t>& upper,
                  const std::vector<size_t>& strides,
                  size_t                     batch_rank = 1,
                  bool                       is_partial = true);

    /**
     * @brief Constructs a new `data_layout_t` object capturing a full range of logical
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
    static data_layout_t full_layout(const std::vector<size_t>& lengths,
                                     const std::vector<size_t>& strides,
                                     size_t                     batch,
                                     size_t                     distance);

    /**
     * @brief Constructs a new `data_layout_t` object capturing a full range of logical
     * indices with one batch axis, and default in-buffer strides (enforcing in-buffer
     * contiguity for the innermost length axis).
     * 
     * @param[in] lengths spans of the logical index range along all length axes.
     * @param[in] batch span of the logical index range along its batch axis.
     * @param[in] real_case_with_padding flag setting the in-buffer stride of the
     * layout's first non-contiguous axis to the value that is required in real
     * domain for real, in-place Discrete Fourier Transforms.
     * 
     * @throw An `std::invalid_argument` is thrown if `lengths` is empty.
     */
    static data_layout_t default_full_layout(const std::vector<size_t>& lengths,
                                             size_t                     batch,
                                             bool real_case_with_padding = false);

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
     * @param[in] other data layout which must be dimensionally consistent with this one
     * (same number of length and batch axes) and must logically contain this layout's
     * very first element.
     * 
     * @return The in-buffer offset for this layout's very first element in a buffer that
     * observes the `other` data layout.
     * 
     * @throw An `std::invalid_argument` is thrown if `other` is not dimensionally consistent
     * with this object or does not contain its very first element.
     */
    size_t offset_in(const data_layout_t& other) const;

    /**
     * @param[in] other data layout which must be dimensionally consistent with this one
     * (same number of length and batch axes).
     * 
     * @return `true` iff this layout represents a continuous chunk of `other`, i.e., iff
     * - both layouts have identical strides along all axes that have non-unit logical spans;
     * - both layouts have identical logical index ranges along all axes except
     *   for the `other`'s slowest-varying one;
     * - this layout's logical range is contained by the `other` along its slowest-varying axis.
     * 
     * @throw An `std::invalid_argument` is thrown if `other` is not dimensionally consistent
     * with this object.
     */
    bool is_continuous_in(const data_layout_t& other) const;

    /**
     * @return A string reporting this object's state.
     */
    std::string str() const;

    /**
     * @return The volume of this object's hyper-rectangular logical index range.
     */
    size_t logical_count() const;
    /**
     * @return The (minimum) number of elements required in a buffer to store the
     * data layout described by this object.
     */
    size_t buffer_element_count() const;

    /**
     * @return `true` iff this object describes a contiguous data layout, i.e., iff all
     * in-buffer elements between offset 0 and logical_count() (excluded) correspond
     * to a distinct element in logical range.
     */
    bool is_contiguous() const;

    /**
     * @param[in] other data layout to be compared to this one.
     * @return `true` iff this object and `other` are strictly identical.
     */
    bool operator==(const data_layout_t& other) const;

    /**
     * @param[in] other data layout which must be dimensionally consistent with this one
     * (same number of length and batch axes).
     * @return `true` iff all axes of either layout cover the same logical index range.
     * 
     * @throw An `std::invalid_argument` is thrown if `other` is not dimensionally consistent
     * with this object.
     */
    bool has_same_logical_range_as(const data_layout_t& other) const;

    /**
     * @param[in] first a data_layout_t object
     * @param[in] second a data_layout_t object that's dimensionally consistent with `first`
     * @return A data_layout_t object capturing the intersection of the logical ranges
     * between `first` and `second`. The in-buffer strides of the constructed intersection
     * are set to their default contiguous values.
     * 
     * @throw An `std::invalid_argument` is thrown if `first` and `second` are not
     * dimensionally consistent, or have different embeddings (if any).
     */
    static data_layout_t make_contiguous_intersection_of(const data_layout_t& first,
                                                         const data_layout_t& second);

    /**
     * @param[in] other data layout against which ranks are to be compared
     * @return `true` if this data layout is dimensionally consistent with `other` 
     */
    bool is_dimensionally_consistent_with(const data_layout_t& other) const;

    /**
     * @return `true` if any length axis covers a partial range.
     */
    bool has_some_partial_length_axis() const;

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
     * object must have no partial length axis.
     * 
     * @param[in] other_io I/O label for the data layout to be returned. Explicitly,
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
     * @return An `std::optional<data_layout_t>` object which has a value set
     * iff a corresponding layout for in-place operation does actually exist.
     * 
     * @note This function does not verify if either layout is self-aliasing and
     * ignores offsets as `data_layout_t` objects do not capture them.
     * 
     * @throw An `std::logic_error` is thrown if the current object is an empty
     * layout or involves some partial length axes. An `std::invalid_argument` is
     * thrown if `fft_type` is not an expected value or if `other_io` is not an
     * expected value.
     * 
     */
    std::optional<data_layout_t> get_other_inplace_layout_for(io_data_label         other_io,
                                                              rocfft_transform_type fft_type,
                                                              bool other_innermost_length_is_odd
                                                              = false) const;

    /**
     * @return The data layout corresponding to a (possibly) sub-dimensional data set
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
     */
    data_layout_t get_layout_for_len_axes(const std::set<size_t>& len_indices) const;

    /**
     * @return The data layout of the data set in which the data set described by the
     * current object is embedded.
     * @note If this object is not representing the data layout of an embedded data set,
     * this function is equivalent to a simple copy of the current object.
     * @throw An `std::logic_error` exception is thrown if the object's embedding map is
     * found inconsistent.
     */
    data_layout_t get_embedding_layout() const;

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
     * @return `true` iff the current object is registered as the data layout of a
     * data set embedded within another, higher-dimensional one.
     */
    bool is_embedded() const;

    //-------------------------------------------------------------------------
    //                        DEFAULT COPIES AND MOVES
    //-------------------------------------------------------------------------

    data_layout_t(const data_layout_t&) = default;
    data_layout_t& operator=(const data_layout_t&) = default;
    data_layout_t(data_layout_t&&)                 = default;
    data_layout_t& operator=(data_layout_t&&) = default;

    // Prevent ill-defined objects as much as possible by privatizing
    // content-modifying members and default constructor.
private:
    struct axis_t
    {
        size_t lower;
        size_t upper;
        size_t inbuffer_stride;
        bool   is_partial;

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
            return has_same_logical_range_as(other) && is_partial == other.is_partial
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

    data_layout_t() = default;

    std::vector<axis_t> len_axes;
    std::vector<axis_t> batch_axes;
    // If the current object captures the data layout for a sub-dimensional data set
    // embedded within another, an embedding map is required to keep track of which of
    // the current object's axes correspond to the embedding set's length axes. If
    // `embedding` has a value set, the current object captures the layout for data
    // embedded within a higher dimensional data set of rank `embedding->size()`, for
    // which the `j`-th length axis (`0 <= j < embedding->size()`) is the current
    // object's axis of (flattened) index `(*embedding)[j]`.
    std::optional<std::vector<size_t>> embedding;

    inline bool has_consistent_embedding() const;

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
     * @brief Empties all layout axes.
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
    // - complete the definitions data layouts once the lengths (full logical ranges) are known;
    // - remove trivial axes (of unit logical range) from full and partial layouts upon finalization;
    // - possibly re-order relevant layouts' length axes by increasing in-buffer strides.
    friend struct rocfft_plan_description_t;
    // Fields set the `is_partial` flags for axes of its bricks that are found to span the
    // entire range of logical indices in the corresponding full layout's length or batch
    // axes (upon field finalization)
    friend struct rocfft_field_t;
};

#endif // DATA_LAYOUT_H
