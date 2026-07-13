// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_UTILS_TENSOR_DESC_HPP
#define GUARD_MIOPEN_UTILS_TENSOR_DESC_HPP

// RAII wrapper around miopenTensorDescriptor_t that uses only the public
// MIOpen C API.  Replaces direct use of miopen::TensorDescriptor (an
// internal, MIOPEN_INTERNALS_EXPORT type) in driver, test, and
// miopen_utils code.

#include <miopen/miopen.h>

#include <cassert>
#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// When building tests, include the internal TensorDescriptor type so that
// TensorDesc can provide an implicit conversion to it (see operator below).
#ifdef MIOPEN_BUILD_TESTING
#include <miopen/tensor.hpp>
#endif

class TensorDesc
{
    miopenTensorDescriptor_t handle_ = nullptr;
    // Cached lengths/strides so the instance GetLengths()/GetStrides() can return
    // a const reference to a persistent container, matching the contract of the
    // internal miopen::TensorDescriptor. This is required for correctness, not
    // just performance: code such as miopen::tien<N>(desc.GetLengths()) builds a
    // tuple of references into the returned container (std::tie), so returning a
    // temporary by value would leave dangling references. The cache is refreshed
    // whenever the underlying descriptor is (re)configured.
    std::vector<size_t> lengths_;
    std::vector<size_t> strides_;

    void create()
    {
        auto s = miopenCreateTensorDescriptor(&handle_);
        if(s != miopenStatusSuccess)
            throw std::runtime_error("miopenCreateTensorDescriptor failed");
    }

    void destroy() noexcept
    {
        if(handle_ != nullptr)
        {
            miopenDestroyTensorDescriptor(handle_);
            handle_ = nullptr;
        }
    }

    // Repopulate lengths_/strides_ from the current handle_. Call after every
    // operation that sets or changes the descriptor.
    void refresh()
    {
        lengths_.clear();
        strides_.clear();
        if(handle_ == nullptr)
            return;
        int ndim = 0;
        miopenGetTensorDescriptorSize(handle_, &ndim);
        if(ndim <= 0)
            return;
        lengths_.resize(ndim);
        strides_.resize(ndim);
        miopenGetTensorDescriptorV2(handle_, nullptr, lengths_.data(), strides_.data());
    }

    void copy_from(miopenTensorDescriptor_t src)
    {
        if(src == nullptr)
            return;

        int ndim = 0;
        miopenGetTensorDescriptorSize(src, &ndim);
        if(ndim == 0)
            return;

        miopenDataType_t dt;
        std::vector<size_t> lens(ndim), strs(ndim);
        miopenGetTensorDescriptorV2(src, &dt, lens.data(), strs.data());
        miopenSetTensorDescriptorV2(handle_, dt, ndim, lens.data(), strs.data());
        refresh();
    }

public:
    // ---------------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------------

    /// Default: creates an empty descriptor handle.
    TensorDesc() { create(); }

    /// From data type + lengths (packed strides computed automatically).
    TensorDesc(miopenDataType_t type, const std::vector<size_t>& lens)
    {
        create();
        miopenSetTensorDescriptorV2(
            handle_, type, static_cast<int>(lens.size()), lens.data(), nullptr);
        refresh();
    }

    /// From data type + lengths + explicit strides.
    TensorDesc(miopenDataType_t type,
               const std::vector<size_t>& lens,
               const std::vector<size_t>& strides)
    {
        create();
        assert(lens.size() == strides.size());
        miopenSetTensorDescriptorV2(
            handle_, type, static_cast<int>(lens.size()), lens.data(), strides.data());
        refresh();
    }

    /// From data type + layout + lengths (strides derived from layout).
    TensorDesc(miopenDataType_t type, miopenTensorLayout_t layout, const std::vector<size_t>& lens)
    {
        create();
        std::vector<int> ilens(lens.begin(), lens.end());
        miopenSetNdTensorDescriptorWithLayout(
            handle_, type, layout, ilens.data(), static_cast<int>(ilens.size()));
        refresh();
    }

    /// Convenience: 4D from individual dimensions (packed).
    TensorDesc(miopenDataType_t type, size_t n, size_t c, size_t h, size_t w)
        : TensorDesc(type, std::vector<size_t>{n, c, h, w})
    {
    }

    /// Convenience: 5D from individual dimensions (packed).
    TensorDesc(miopenDataType_t type, size_t n, size_t c, size_t d, size_t h, size_t w)
        : TensorDesc(type, std::vector<size_t>{n, c, d, h, w})
    {
    }

    /// Deep copy from an existing opaque handle.
    explicit TensorDesc(miopenTensorDescriptor_t src)
    {
        create();
        copy_from(src);
    }

    // ---------------------------------------------------------------
    // Copy / move
    // ---------------------------------------------------------------

    TensorDesc(const TensorDesc& other)
    {
        create();
        copy_from(other.handle_);
    }

    TensorDesc& operator=(const TensorDesc& other)
    {
        if(this != &other)
            copy_from(other.handle_);
        return *this;
    }

    TensorDesc(TensorDesc&& other) noexcept
        : handle_(other.handle_),
          lengths_(std::move(other.lengths_)),
          strides_(std::move(other.strides_))
    {
        other.handle_ = nullptr;
    }

    TensorDesc& operator=(TensorDesc&& other) noexcept
    {
        if(this != &other)
        {
            destroy();
            handle_       = other.handle_;
            lengths_      = std::move(other.lengths_);
            strides_      = std::move(other.strides_);
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~TensorDesc() { destroy(); }

    // ---------------------------------------------------------------
    // Static query functions (work on raw miopenTensorDescriptor_t)
    // ---------------------------------------------------------------

    static int GetNumDims(miopenTensorDescriptor_t desc)
    {
        int n = 0;
        miopenGetTensorDescriptorSize(desc, &n);
        return n;
    }

    static miopenDataType_t GetType(miopenTensorDescriptor_t desc)
    {
        miopenDataType_t dt;
        miopenGetTensorDescriptorV2(desc, &dt, nullptr, nullptr);
        return dt;
    }

    static std::vector<size_t> GetLengths(miopenTensorDescriptor_t desc)
    {
        int ndim = GetNumDims(desc);
        std::vector<size_t> v(ndim);
        miopenGetTensorDescriptorV2(desc, nullptr, v.data(), nullptr);
        return v;
    }

    static std::vector<size_t> GetStrides(miopenTensorDescriptor_t desc)
    {
        int ndim = GetNumDims(desc);
        std::vector<size_t> v(ndim);
        miopenGetTensorDescriptorV2(desc, nullptr, nullptr, v.data());
        return v;
    }

    static size_t GetNumBytes(miopenTensorDescriptor_t desc)
    {
        size_t n = 0;
        miopenGetTensorNumBytes(desc, &n);
        return n;
    }

    static size_t GetElementSize(miopenTensorDescriptor_t desc)
    {
        auto lens = GetLengths(desc);
        size_t n  = 1;
        for(auto l : lens)
            n *= l;
        return n;
    }

    static size_t GetElementSpace(miopenTensorDescriptor_t desc)
    {
        size_t s = 0;
        miopenGetTensorElementSpace(desc, &s);
        return s;
    }

    static bool IsPacked(miopenTensorDescriptor_t desc)
    {
        bool p = false;
        miopenIsTensorPacked(desc, &p);
        return p;
    }

    static size_t GetVectorLength(miopenTensorDescriptor_t desc)
    {
        size_t v = 0;
        miopenGetTensorVectorLength(desc, &v);
        return v;
    }

    static miopenTensorLayout_t GetLayout(miopenTensorDescriptor_t desc)
    {
        miopenTensorLayout_t layout;
        miopenGetTensorLayout(desc, &layout);
        return layout;
    }

    /// Mirrors miopen::TensorDescriptor::GetLayoutEnum() for above-the-library
    /// consumers that used the internal type's optional-returning accessor.
    static std::optional<miopenTensorLayout_t> GetLayoutEnum(miopenTensorDescriptor_t desc)
    {
        return GetLayout(desc);
    }

    /// Returns the byte size of a MIOpen data type.
    static size_t GetTypeSize(miopenDataType_t dt)
    {
        switch(dt)
        {
        case miopenHalf: return 2;
        case miopenFloat: return 4;
        case miopenDouble: return 8;
        case miopenBFloat16: return 2;
        case miopenInt8: return 1;
        case miopenInt32: return 4;
        case miopenInt64: return 8;
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz: return 1;
        }
        return 0; // unreachable for known types; silences -Wreturn-type
    }

    /// Given a spatial dimension count and a vector of lengths (or strides),
    /// return a tuple of (N, C, D, H, W) with D=1 for 2D tensors.
    template <class TElement>
    static auto GetNCDHW(unsigned spatial_dims, const std::vector<TElement>& data)
    {
        if(spatial_dims == 3)
        {
            assert(data.size() >= 5);
            return std::make_tuple(data[0], data[1], data[2], data[3], data[4]);
        }
        else
        {
            assert(data.size() >= 4);
            return std::make_tuple(data[0], data[1], static_cast<TElement>(1), data[2], data[3]);
        }
    }

    static std::string GetLayout_str(miopenTensorDescriptor_t desc)
    {
        switch(GetLayout(desc))
        {
        case miopenTensorNCHW: return "NCHW";
        case miopenTensorNHWC: return "NHWC";
        case miopenTensorCHWN: return "CHWN";
        case miopenTensorNCHWc4: return "NCHWc4";
        case miopenTensorNCHWc8: return "NCHWc8";
        case miopenTensorCHWNc4: return "CHWNc4";
        case miopenTensorCHWNc8: return "CHWNc8";
        case miopenTensorNCDHW: return "NCDHW";
        case miopenTensorNDHWC: return "NDHWC";
        }
        return "Unknown"; // unreachable for known layouts; silences -Wreturn-type
    }

    // ---------------------------------------------------------------
    // Instance getters (delegate to static versions)
    // ---------------------------------------------------------------

    int GetNumDims() const { return static_cast<int>(lengths_.size()); }
    /// Alias for GetNumDims(), mirroring miopen::TensorDescriptor::GetSize().
    std::size_t GetSize() const { return lengths_.size(); }
    miopenDataType_t GetType() const { return GetType(handle_); }
    // Return a const reference to the cached vectors (NOT a temporary), so that
    // callers like miopen::tien<N>(desc.GetLengths()) — which build a tuple of
    // references into the argument — do not observe dangling references.
    const std::vector<size_t>& GetLengths() const { return lengths_; }
    const std::vector<size_t>& GetStrides() const { return strides_; }
    size_t GetNumBytes() const { return GetNumBytes(handle_); }
    size_t GetElementSize() const { return GetElementSize(handle_); }
    size_t GetElementSpace() const { return GetElementSpace(handle_); }
    bool IsPacked() const { return IsPacked(handle_); }
    size_t GetVectorLength() const { return GetVectorLength(handle_); }
    miopenTensorLayout_t GetLayout() const { return GetLayout(handle_); }
    std::optional<miopenTensorLayout_t> GetLayoutEnum() const { return GetLayoutEnum(handle_); }
    std::string GetLayout_str() const { return GetLayout_str(handle_); }

    /// Compute linear index from multi-dimensional coordinates using strides.
    template <typename... Ts>
    size_t GetIndex(Ts... xs) const
    {
        auto strides    = GetStrides();
        size_t coords[] = {static_cast<size_t>(xs)...};
        size_t idx      = 0;
        size_t n        = sizeof...(xs);
        assert(n <= strides.size());
        for(size_t i = 0; i < n; ++i)
            idx += coords[i] * strides[i];
        return idx;
    }

    /// Debug string: "layout dim[...] stride[...]"
    std::string ToString() const
    {
        auto lens = GetLengths();
        auto strs = GetStrides();
        std::ostringstream ss;
        ss << GetLayout_str() << " dim[";
        for(size_t i = 0; i < lens.size(); ++i)
        {
            if(i > 0)
                ss << ',';
            ss << lens[i];
        }
        ss << "] stride[";
        for(size_t i = 0; i < strs.size(); ++i)
        {
            if(i > 0)
                ss << ',';
            ss << strs[i];
        }
        ss << ']';
        return ss.str();
    }

    // ---------------------------------------------------------------
    // Reshape operations
    // ---------------------------------------------------------------

    /// Collapse a 5D tensor descriptor (NCDHW or NDHWC) to 4D (NCHW or NHWC)
    /// by merging the D and H dimensions: [N,C,D,H,W] -> [N,C,D*H,W].
    /// Used by batch normalization to handle 3D spatial inputs.
    TensorDesc Reshaped5Dto4D() const
    {
        auto dims   = GetLengths();
        auto layout = GetLayout();
        auto dt     = GetType();

        assert(dims.size() == 5);

        miopenTensorLayout_t layout4d;
        if(layout == miopenTensorNCDHW)
            layout4d = miopenTensorNCHW;
        else if(layout == miopenTensorNDHWC)
            layout4d = miopenTensorNHWC;
        else
            throw std::runtime_error("Reshaped5Dto4D: unsupported layout");

        // Merge D*H: [N, C, D, H, W] -> [N, C, D*H, W]
        dims[2] *= dims[3];
        dims[3] = dims[4];
        dims.pop_back();

        return {dt, layout4d, dims};
    }

    // ---------------------------------------------------------------
    // Comparison
    // ---------------------------------------------------------------

    bool operator==(const TensorDesc& other) const
    {
        return GetType() == other.GetType() && GetLengths() == other.GetLengths() &&
               GetStrides() == other.GetStrides();
    }

    bool operator!=(const TensorDesc& other) const { return !(*this == other); }

    // ---------------------------------------------------------------
    // C API interop
    // ---------------------------------------------------------------

    /// Get the raw handle for passing to MIOpen C API functions.
    miopenTensorDescriptor_t get() const { return handle_; }

    /// Implicit conversion so code like miopenSetTensor(h, desc, ...) works.
    operator miopenTensorDescriptor_t() const { return handle_; }

#ifdef MIOPEN_BUILD_TESTING
    // Test code (Layer 4) legitimately needs access to the internal
    // miopen::TensorDescriptor type for constructing conv::ProblemDescription,
    // calling DeriveBNTensorDescriptor, etc. This conversion dereferences the
    // opaque handle to return a reference to the underlying internal object.
    // Driver and miopen_utils builds do NOT define MIOPEN_BUILD_TESTING,
    // so this conversion is not available outside of test code.
    operator const miopen::TensorDescriptor&() const { return miopen::deref(handle_); }

    // Test-only: assign from the internal miopen::TensorDescriptor value type,
    // for test code that still builds descriptors via the internal API.
    TensorDesc& operator=(const miopen::TensorDescriptor& rhs)
    {
        *this = TensorDesc(rhs.GetType(),
                           std::vector<size_t>(rhs.GetLengths().begin(), rhs.GetLengths().end()),
                           std::vector<size_t>(rhs.GetStrides().begin(), rhs.GetStrides().end()));
        return *this;
    }
#endif
};

// tensor_view_t utilities -- build tensor views from descriptor handles
// without requiring miopen::TensorDescriptor internals.
#include <miopen_utils/tensor_view.hpp>

/// Build a tensor_view_t<N> from an opaque tensor descriptor handle.
/// Replacement for miopen::get_inner_expanded_tv<N>(miopen::deref(desc)).
template <int N>
inline tensor_view_t<N> GetInnerExpandedTv(miopenTensorDescriptor_t desc)
{
    auto dims    = TensorDesc::GetLengths(desc);
    auto strides = TensorDesc::GetStrides(desc);

    tensor_view_t<N> tv{};
    for(int i = 0; i < N; ++i)
    {
        if(dims.empty())
        {
            tv.stride[i] = 0;
            tv.size[i]   = 0;
        }
        else if(static_cast<size_t>(i) < dims.size())
        {
            tv.stride[i] = strides[i];
            tv.size[i]   = dims[i];
        }
        else
        {
            tv.stride[i] = strides.back();
            tv.size[i]   = 1;
        }
    }
    return tv;
}

#endif // GUARD_MIOPEN_UTILS_TENSOR_DESC_HPP
