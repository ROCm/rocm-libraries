// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef CLIENT_DATA_LAYOUT_HELPERS_H
#define CLIENT_DATA_LAYOUT_HELPERS_H

#include "fft_enums.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// type traits to distinguish between std::array and other supposedly dynamically-sized containers
template <typename C>
struct is_std_array : std::false_type
{
};
template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type
{
};

/**
 * @brief calculates the default strides for a given Discrete Fourier transform (observing row-major convention by default).
 * @note This function assumes interleaved complex representation for hermitian-symmetric data.
 * @note Empty strides are returned for empty lengths.
 * 
 * @tparam C container type for the transform's lengths (container's value_type must be an integral type).
 * The same type is used for the returned strides.
 * @param[in] dft_type type of transform
 * @param[in] placement placement of the transform
 * @param[in] io input/output flag
 * @param[in] lengths container of lengths for the transform of interest
 * @param[in] dim_order optional vector of dimension ordering for ``lengths``. If used, the content
 * of this vector must be a permutation of {0, 1, ..., lengths.size() - 1} (:= default natural order)
 * and the corresponding dimension ordering is observed when generating the strides. For instance,
 * column-major ordering is observed when setting ``dim_order`` to {lengths.size()-1, ..., 1, 0}.
 * @return desired default strides (same type as ``lengths`` input argument)
 */
template <typename C, std::enable_if_t<std::is_integral_v<typename C::value_type>, bool> = true>
static C default_strides(fft_transform_type                        dft_type,
                         fft_result_placement                      placement,
                         fft_io                                    io,
                         const C&                                  lengths,
                         const std::optional<std::vector<size_t>>& dim_order = std::nullopt)
{
    validate_enums_or_throw("default_strides", dft_type, placement, io);
    if(dim_order)
    {
        if(dim_order->size() != lengths.size())
            throw std::invalid_argument(
                "default_strides: size mismatch between dim_order and lengths");
        std::vector<size_t> natural_order(lengths.size());
        std::iota(natural_order.begin(), natural_order.end(), 0);
        if(!std::is_permutation(dim_order->begin(), dim_order->end(), natural_order.begin()))
            throw std::invalid_argument("default_strides: invalid dim_order (not a permutation of "
                                        "{0, 1, ..., lengths.size() - 1}");
    }

    C                      ret(lengths);
    typename C::value_type def_stride = 1;
    for(auto l_idx = lengths.size(); l_idx-- > 0;)
    {
        const auto dim_idx = dim_order ? dim_order->at(l_idx) : l_idx;
        ret[dim_idx]       = def_stride;
        if(dim_idx == lengths.size() - 1 && is_real(dft_type))
        {
            if((io == fft_io_out) == is_fwd(dft_type))
                def_stride *= (lengths[dim_idx] / 2 + 1);
            else
            {
                if(placement == fft_placement_inplace)
                    def_stride *= 2 * (lengths[dim_idx] / 2 + 1);
                else
                    def_stride *= lengths[dim_idx];
            }
        }
        else
            def_stride *= lengths[dim_idx];
    }
    return ret;
}

/**
 * @brief calculates the default distances for a given Discrete Fourier transform,
 * batched possibly multiple times (observing row-major convention by default).
 * @note This function assumes interleaved complex representation for hermitian-symmetric data.
 * @note Empty distances are returned for empty batches.
 * 
 * @tparam T integral value type for lengths and batches, the same type is used for the returned distances
 * @param[in] dft_type type of transform
 * @param[in] placement placement of the transform
 * @param[in] io input/output flag
 * @param[in] lengths vector of lengths for the transform of interest
 * @param[in] batches vector of (possibly many) batch sizes for the transform of interest
 * @param[in] len_dim_order optional vector of dimension ordering for ``lengths``. If used, the content
 * of this vector must be a permutation of {0, 1, ..., lengths.size() - 1} (:= default natural order)
 * and the corresponding dimension ordering is observed when generating the strides. For instance,
 * column-major ordering is observed when setting ``len_dim_order`` to {lengths.size()-1, ..., 1, 0}.
 * @return std::vector<T> desired default distances (of size batches.size())
 */
template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
static std::vector<T> default_distances(fft_transform_type                        dft_type,
                                        fft_result_placement                      placement,
                                        fft_io                                    io,
                                        const std::vector<T>&                     lengths,
                                        const std::vector<T>&                     batches,
                                        const std::optional<std::vector<size_t>>& len_dim_order
                                        = std::nullopt)
{
    validate_enums_or_throw("default_distances", dft_type, placement, io);
    if(batches.empty())
        return std::vector<T>(); // empty as well
    auto temp_lengths = batches;
    std::copy(lengths.begin(), lengths.end(), std::back_inserter(temp_lengths));
    std::vector<T> ret;
    if(!len_dim_order)
    {
        ret = default_strides(dft_type, placement, io, temp_lengths);
    }
    else
    {
        if(len_dim_order->size() != lengths.size())
            throw std::invalid_argument(
                "default_distances: size mismatch between len_dim_order and lengths");
        std::vector<size_t> dim_order(batches.size() + lengths.size());
        std::iota(dim_order.begin(), dim_order.begin() + batches.size(), 0);
        for(size_t len_idx = 0; len_idx < lengths.size(); len_idx++)
            dim_order[batches.size() + len_idx] = batches.size() + len_dim_order->at(len_idx);
        ret = default_strides(dft_type, placement, io, temp_lengths, dim_order);
    }

    ret.resize(batches.size());
    return ret;
}
/**
 * @brief equivalent of default_distances for one-dimensional batches 
 */
template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
static T default_distance(fft_transform_type                        dft_type,
                          fft_result_placement                      placement,
                          fft_io                                    io,
                          const std::vector<T>&                     lengths,
                          const T&                                  batch_sz,
                          const std::optional<std::vector<size_t>>& len_dim_order = std::nullopt)
{
    validate_enums_or_throw("default_distance", dft_type, placement, io);
    if(lengths.empty())
        throw std::invalid_argument("empty lengths rejected by default_distance");
    const auto tmp = default_distances(
        dft_type, placement, io, lengths, std::vector<T>(1, batch_sz), len_dim_order);
    return tmp.front();
}

struct ionembed_exception : public std::runtime_error
{
    ionembed_exception()
        : std::runtime_error("")
    {
    }
};

/**
 * @brief exception thrown by ionembed_t objects when a type conversion error is detected.
 */
struct ionembed_type_conversion_error : public ionembed_exception
{
    using ionembed_exception::ionembed_exception;
};

/**
 * @brief exception thrown at construction of ionembed_t objects from generalized strides,
 * if/when those strides are incompatible with the "advanced layout" parameters.
 */
struct strides_are_not_nembed_compatible : ionembed_exception
{
    using ionembed_exception::ionembed_exception;
};

/**
 * @brief helper structure for converting hipFFT or hipFFTW advanced plans data
 * layout parameters [istride, ostride, inembed, onembed] to/from general strides.
 * @note This structure observes row-major ordering convention.
 * 
 * @tparam T integral type to be considered for the istride, ostride, inembed,
 * and onembed structure members.
 * @tparam ignore_elementary_stride_if_null_nembed boolean flag distinguishing whether the
 * elementary strides istride or ostride are ignored (and assumed equal to 1) whenever inembed
 * or onembed is not used (aka "implicit default").
 * If set to false, elementary strides are not ignored (behavior consistent with FFTW3 and hipFFTW).
 * If set to true,  elementary strides are ignored (behavior consistent with hipFFT and cuFFT).
 */
template <typename T,
          bool ignore_elementary_stride_if_null_nembed,
          std::enable_if_t<std::is_integral_v<T>, bool> = true>
struct ionembed_t
{

    // default constructor := default layout
    ionembed_t()
        : istride(T(1))
        , ostride(T(1))
        , inembed(std::nullopt)
        , onembed(std::nullopt)
    {
    }

    /**
     * @brief Construct a new ionembed t object
     * 
     * @tparam U value type for the constructor's input arguments (may be different than
     * the internally-used value type).
     * @param[in] istride_ elementary stride for input data (along the last dimension);
     * @param[in] inembed_ vector of input-embedding lengths, implicit default embedding is used if empty;
     * @param[in] ostride_ elementary stride for output data (along the last dimension);
     * @param[in] onembed_ vector of output-embedding lengths, implicit default embedding is used if empty;
     * @note: if non empty, the sizes of inembed_ and onembed_ must be equal (an std::invalid_argument
     * exception is thrown otherwise)
     * @note: if U is different than the internally-used value type, a ionembed_type_conversion_error may
     * be thrown by this constructor.
     */
    template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
    ionembed_t(U                     istride_,
               const std::vector<U>& inembed_,
               U                     ostride_,
               const std::vector<U>& onembed_)
        : istride(convert_to<T>(istride_))
        , ostride(convert_to<T>(ostride_))
        , inembed(make_optional_nembed_vector(inembed_))
        , onembed(make_optional_nembed_vector(onembed_))
    {
        if(inembed && onembed && inembed->size() != onembed->size())
            throw std::invalid_argument("inconsistent sizes of inembed and onembed used when "
                                        "constructing an ionembed_t object");
    }

    /**
     * @brief Construct a new ionembed t object
     *
     * @tparam U value type for the constructor's input arguments (may be different than
     * the internally-used value type).
     * @param[in] rank     dimensionality of the transform (must be strictly positive unless both inembed_ and onembed_ are ``nullptr``);
     * @param[in] istride_ elementary stride for input data (along the last dimension);
     * @param[in] inembed_ array of ``rank`` input-embedding lengths, implicit default embedding is used if nullptr;
     * @param[in] ostride_ elementary stride for output data (along the last dimension);
     * @param[in] onembed_ array of ``rank`` output-embedding lengths, implicit default embedding is used if nullptr;
     * @note: if U is different than the internally-used value type, a ionembed_type_conversion_error may
     * be thrown by this constructor.
     */
    template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
    ionembed_t(int rank, U istride_, const U* inembed_, U ostride_, const U* onembed_)
        : ionembed_t(istride_,
                     make_nembed_vector_from_array(rank, inembed_),
                     ostride_,
                     make_nembed_vector_from_array(rank, onembed_))
    {
    }

    /**
     * @brief Construct a new ionembed t object from generalized strides, if compatible.
     * 
     * @tparam U value type for the constructor's input arguments (may be different than
     * the internally-used value type).
     * @param[in] istrides generalized input strides (row-major ordering)
     * @param[in] ostrides generalized output strides (row-major ordering)
     * @param[in] lengths transform lengths (row-major ordering)
     * @param[in] dft_kind type of the discrete Fourier transform of interest.
     * @param[in] placement placement of the discrete Fourier transform of interest.
     * @param[in] use_nullptr_for_default_inembed self-exaplanatory boolean flag
     * @param[in] use_nullptr_for_default_onembed self-exaplanatory boolean flag
     * @note this constructor may throw
     * - a ionembed_type_conversion_error exception if a U value cannot be safely
     *   converted to the internally-used value type;
     * - a strides_are_not_nembed_compatible exception if the generalized strides are
     *   not compatible with an "advanced layout" parameterization.
     */
    template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
    ionembed_t(const std::vector<U>& istrides,
               const std::vector<U>& ostrides,
               const std::vector<U>& lengths,
               fft_transform_type    dft_kind,
               fft_result_placement  placement,
               bool                  use_nullptr_for_default_inembed,
               bool                  use_nullptr_for_default_onembed)
        : istride(convert_to<T>(istrides.empty() ? 1 : istrides.back()))
        , ostride(convert_to<T>(ostrides.empty() ? 1 : ostrides.back()))
        , inembed(calc_nembed_from_generalized_strides<U>(istrides,
                                                          lengths,
                                                          fft_io::fft_io_in,
                                                          dft_kind,
                                                          placement,
                                                          use_nullptr_for_default_inembed))
        , onembed(calc_nembed_from_generalized_strides<U>(ostrides,
                                                          lengths,
                                                          fft_io::fft_io_out,
                                                          dft_kind,
                                                          placement,
                                                          use_nullptr_for_default_onembed))
    {
        // istrides.size() == lengths.size() and ostrides.size() == lengths.size() verified
        // separately in initialization steps, so istrides.size() == ostrides.size()
    }

    // default copy and move constructors/assignment operators
    ionembed_t(const ionembed_t& other) = default;
    ionembed_t& operator=(const ionembed_t& other) = default;
    ionembed_t(ionembed_t&& other)                 = default;
    ionembed_t& operator=(ionembed_t&& other) = default;

    /**
     * @brief strict equality comparison operator.
     * @note While unset {i,o}nembed are conceptually equivalent to explicitly-set default values,
     * this operator cannot verify that possibility without additional information (like the kind
     * of transform and its placement): it's a simple member-wise equality check.
     * @return true iff the two structures have equal members
     */
    bool operator==(const ionembed_t& other) const
    {
        return istride == other.istride && ostride == other.ostride && inembed == other.inembed
               && onembed == other.onembed;
    }
    bool operator!=(const ionembed_t& other) const
    {
        return !(*this == other);
    }

    /**
     * @brief calculates generalized (row-major) strides from the advanced plan's data layout parameters
     * 
     * @tparam C container type for the generalized strides and the transform's lengths (container's
     * value_type must be an integral type).
     * @param[in] io flag for requesting input (fft_io::fft_io_in) or output (fft_io::fft_io_out) strides.
     * @param[in] dft_kind type of the discrete Fourier transform of interest.
     * @param[in] placement placement of the discrete Fourier transform of interest.
     * @param[in] lengths lengths of the discrete Fourier transform of interest (row-major ordering).
     * @return generalized strides calculated from the advanced plan's data layout parameters
     * or from the provided args if implicit default layout is used. ``dft_kind``, ``placement``,
     * and ``lengths`` are all ignored if implicit default layout is NOT used (i.e., if
     * {i,o}nembed is explicitly set internally). The returned value is of the same type as ``lengths``.
     * @note a ionembed_type_conversion_error may be thrown if an error is detected when converting
     * internally-calculated values to C::value_type (e.g. if a negative value is to be assigned but
     * C::value_type is unsigned, or if C::value_type := int32_t and T := int64_t and an overflow/underflow
     * is detected internally).
     */
    template <typename C, std::enable_if_t<std::is_integral_v<typename C::value_type>, bool> = true>
    C as_generalized_strides(fft_io               io,
                             fft_transform_type   dft_kind,
                             fft_result_placement placement,
                             const C&             lengths) const
    {
        using C_val_t = typename C::value_type;
        validate_enums_or_throw("ionembed_t::as_generalized_strides", io, dft_kind, placement);
        const auto&   nembed = io == fft_io::fft_io_in ? inembed : onembed;
        const C_val_t elem_stride
            = convert_to<C_val_t>(io == fft_io::fft_io_in ? istride : ostride);
        if(!nembed)
        {
            // --> default strides, use lengths, type of transform, etc.
            // note: value type used below is same of lengths', i.e., C_val_t
            auto ret = default_strides(dft_kind, placement, io, lengths);
            if constexpr(!ignore_elementary_stride_if_null_nembed)
            {
                if(elem_stride != 1)
                    std::for_each(
                        ret.begin(), ret.end(), [&](C_val_t& val) { val *= elem_stride; });
            }
            return ret;
        }
        C ret;
        if constexpr(is_std_array<C>::value)
        {
            if(ret.size() != nembed->size())
                throw std::invalid_argument(
                    "Incompatible size of std::array ``lengths`` w.r.t. internal {i,o}nembed "
                    "detected in ionembed_t::as_generalized_strides");
        }
        else
        {
            ret.resize(nembed->size());
        }
        ret.back() = elem_stride;
        for(auto dim = ret.size() - 1; dim-- > 0;)
        {
            ret[dim] = ret[dim + 1] * convert_to<C_val_t>(nembed->at(dim + 1));
        }
        return ret;
    }

    const T* get_nembed(fft_io io) const
    {
        validate_enums_or_throw("ionembed_t::get_nembed", io);
        const auto& tmp = io == fft_io::fft_io_in ? inembed : onembed;
        if(!tmp)
            return nullptr;
        return tmp->data();
    }

    size_t get_nembed_size(fft_io io) const
    {
        validate_enums_or_throw("ionembed_t::get_nembed", io);
        const auto& tmp = io == fft_io::fft_io_in ? inembed : onembed;
        if(!tmp)
            return 0;
        return tmp->size();
    }
    T get_elementary_stride(fft_io io) const
    {
        validate_enums_or_throw("ionembed_t::get_elementary_stride", io);
        return io == fft_io::fft_io_in ? istride : ostride;
    }

private:
    using nembed_t = std::optional<std::vector<T>>;
    T        istride;
    T        ostride;
    nembed_t inembed;
    nembed_t onembed;

    template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
    static nembed_t make_optional_nembed_vector(const std::vector<U>& nembed_vec)
    {
        if(nembed_vec.empty())
            return std::nullopt;
        typename nembed_t::value_type ret(nembed_vec.size());
        for(auto dim = ret.size(); dim-- > 0;)
            ret[dim] = convert_to<T>(nembed_vec[dim]);
        return nembed_t(ret);
    }

    template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
    static std::vector<U> make_nembed_vector_from_array(int rank, const U* nembed_array)
    {
        std::vector<U> ret;
        if(!nembed_array)
            return ret;
        if(rank < 0)
            throw std::invalid_argument(
                "invalid rank encountered by ionembed_t::make_nembed_vector_from_array");
        ret.assign(nembed_array, nembed_array + rank);
        return ret;
    }

    template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
    static nembed_t calc_nembed_from_generalized_strides(const std::vector<U>& strides,
                                                         const std::vector<U>& lengths,
                                                         fft_io                io,
                                                         fft_transform_type    dft_kind,
                                                         fft_result_placement  placement,
                                                         bool use_nullptr_if_default)
    {
        validate_enums_or_throw("ionembed_t::calc_nembed_from_strides", io, dft_kind, placement);

        if(strides.size() != lengths.size())
            throw std::invalid_argument("inconsistent strides.size() vs lengths.size() encountered "
                                        "by ionembed_t::calc_nembed_from_generalized_strides");
        if(strides.empty())
        {
            // degenerate case: lengths and strides are empty (possible use case supposedly considered
            // only for edge-case testing)
            // --> no actual value can (nor should) be legitimately set
            return std::nullopt;
        }

        const auto elem_stride = strides.back();
        const auto def_strides = default_strides(dft_kind, placement, io, lengths);
        if(use_nullptr_if_default
           && std::equal(strides.begin(), strides.end(), def_strides.begin(), [&](U s, U def_s) {
                  if constexpr(ignore_elementary_stride_if_null_nembed)
                      return s == def_s;
                  else
                      return s == elem_stride * def_s;
              }))
        {
            return std::nullopt;
        }
        else
        {
            typename nembed_t::value_type nembed(strides.size());
            for(auto nembed_dim = nembed.size(); nembed_dim-- > 0;)
            {
                U nembed_val;
                if(nembed_dim > 0)
                {
                    if(strides[nembed_dim] == 0
                       || strides[nembed_dim - 1] % strides[nembed_dim] != 0)
                        throw strides_are_not_nembed_compatible();
                    nembed_val = strides[nembed_dim - 1] / strides[nembed_dim];
                }
                else
                {
                    // nembed_dim == 0, i.e., actually irrelevant/unused value...
                    // set it to the minimum value documented to be valid by FFTW3
                    // to guard against (reference) plan creation failures
                    if(is_real(dft_kind) && lengths.size() == 1)
                    {
                        const auto cmplx_stride = lengths[nembed_dim] / 2 + 1;
                        if((io == fft_io_in) == is_bwd(dft_kind))
                            nembed_val = cmplx_stride;
                        else
                        {
                            if(placement == fft_placement_inplace)
                                nembed_val = 2 * cmplx_stride;
                            else
                                nembed_val = lengths[nembed_dim];
                        }
                    }
                    else
                        nembed_val = lengths[nembed_dim];
                }
                nembed[nembed_dim] = convert_to<T>(nembed_val);
            }
            return nembed_t(nembed);
        }
    }

    template <
        typename dest_t,
        typename src_t,
        std::enable_if_t<std::is_integral_v<src_t> && std::is_integral_v<dest_t>, bool> = true>
    static dest_t convert_to(const src_t& val)
    {
        if constexpr(std::is_signed_v<src_t>)
        {
            if constexpr(std::is_unsigned_v<dest_t>)
            {
                // do not use std::numeric_limits<dest_t>::lowest()
                // to avoid implicit promotion of val to dest_t:
                if(val < 0)
                    throw ionembed_type_conversion_error(); // underflow
            }
            else if constexpr(sizeof(dest_t) < sizeof(src_t))
            {
                if(val < std::numeric_limits<dest_t>::lowest())
                    throw ionembed_type_conversion_error(); // underflow
            }
        }
        if constexpr(sizeof(dest_t) < sizeof(src_t)
                     || (sizeof(dest_t) == sizeof(src_t)
                         && std::is_signed_v<dest_t> && std::is_unsigned_v<src_t>))
        {
            if(val > std::numeric_limits<dest_t>::max())
                throw ionembed_type_conversion_error(); // overflow
        }
        return static_cast<dest_t>(val);
    }
};

// in hipFFT/cuFFT:
// @ T := int or T := long long int
// @ T := size_t internally to the implementation details on AMD platforms
// @ elementary strides are ignored when implicit default inembed/onembed are used
//   --> "true" template specialization value for second template arg of ionembed_t
template <typename T>
using hipfft_ionembed_t = ionembed_t<T, true>;

// in FFTW3/hipFFTW
// @ T := int
// @ T := ptrdiff_t internally to the implementation details of hipFFTW
// @ elementary strides are not ignored when implicit default inembed/onembed are used
//   --> "false" template specialization value for second template arg of ionembed_t
using hipfftw_ionembed_t          = ionembed_t<int, false>;
using fftw_ionembed_t             = ionembed_t<int, false>;
using hipfftw_internal_ionembed_t = ionembed_t<ptrdiff_t, false>;

#endif // CLIENT_DATA_LAYOUT_HELPERS_H
