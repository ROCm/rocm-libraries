// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "ck/utility/data_type.hpp"
#include "ck/utility/span.hpp"
#include "ck/utility/type_convert.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/ranges.hpp"

template <typename Range>
std::ostream& LogRange(std::ostream& os, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << v;
    }
    return os;
}

template <typename T, typename Range>
std::ostream& LogRangeAsType(std::ostream& os, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;

        using RangeType = ck::remove_cvref_t<decltype(v)>;
        if constexpr(std::is_same_v<RangeType, ck::f8_t> || std::is_same_v<RangeType, ck::bf8_t> ||
                     std::is_same_v<RangeType, ck::bhalf_t>)
        {
            os << ck::type_convert<float>(v);
        }
        else if constexpr(std::is_same_v<RangeType, ck::pk_i4_t> ||
                          std::is_same_v<RangeType, ck::f4x2_pk_t>)
        {
            const auto packed_floats = ck::type_convert<ck::float2_t>(v);
            const ck::vector_type<float, 2> vector_of_floats{packed_floats};
            os << vector_of_floats.template AsType<float>()[ck::Number<0>{}] << delim
               << vector_of_floats.template AsType<float>()[ck::Number<1>{}];
        }
        else
        {
            os << static_cast<T>(v);
        }
    }
    return os;
}

template <typename F, typename T, std::size_t... Is>
auto call_f_unpack_args_impl(F f, T args, std::index_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
auto call_f_unpack_args(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t... Is>
auto construct_f_unpack_args_impl(T args, std::index_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <typename F, typename T>
auto construct_f_unpack_args(F, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}

struct HostTensorDescriptor
{
    HostTensorDescriptor() = default;

    void CalculateStrides();

    template <typename X, typename = std::enable_if_t<std::is_convertible_v<X, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    HostTensorDescriptor(const std::initializer_list<ck::long_index_t>& lens)
        : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename Lengths,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck::ranges::range_value_t<Lengths>, std::size_t> ||
                  std::is_convertible_v<ck::ranges::range_value_t<Lengths>, ck::long_index_t>>>
    HostTensorDescriptor(const Lengths& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename X,
              typename Y,
              typename = std::enable_if_t<std::is_convertible_v<X, std::size_t> &&
                                          std::is_convertible_v<Y, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens,
                         const std::initializer_list<Y>& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    HostTensorDescriptor(const std::initializer_list<ck::long_index_t>& lens,
                         const std::initializer_list<ck::long_index_t>& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    template <typename Lengths,
              typename Strides,
              typename = std::enable_if_t<
                  (std::is_convertible_v<ck::ranges::range_value_t<Lengths>, std::size_t> &&
                   std::is_convertible_v<ck::ranges::range_value_t<Strides>, std::size_t>) ||
                  (std::is_convertible_v<ck::ranges::range_value_t<Lengths>, ck::long_index_t> &&
                   std::is_convertible_v<ck::ranges::range_value_t<Strides>, ck::long_index_t>)>>
    HostTensorDescriptor(const Lengths& lens, const Strides& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    std::size_t GetNumOfDimension() const;
    std::size_t GetElementSize() const;
    std::size_t GetElementSpaceSize() const;

    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetStrides() const;

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        assert(sizeof...(Is) == this->GetNumOfDimension());
        std::initializer_list<std::size_t> iss{static_cast<std::size_t>(is)...};
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    std::size_t GetOffsetFromMultiIndex(std::vector<std::size_t> iss) const
    {
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    friend std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc);

    private:
    std::vector<std::size_t> mLens;
    std::vector<std::size_t> mStrides;
};

template <typename New2Old>
HostTensorDescriptor transpose_host_tensor_descriptor_given_new2old(const HostTensorDescriptor& a,
                                                                    const New2Old& new2old)
{
    std::vector<std::size_t> new_lengths(a.GetNumOfDimension());
    std::vector<std::size_t> new_strides(a.GetNumOfDimension());

    for(std::size_t i = 0; i < a.GetNumOfDimension(); i++)
    {
        new_lengths[i] = a.GetLengths()[new2old[i]];
        new_strides[i] = a.GetStrides()[new2old[i]];
    }

    return HostTensorDescriptor(new_lengths, new_strides);
}

struct joinable_thread : std::thread
{
    template <typename... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread(joinable_thread&&) = default;
    joinable_thread& operator=(joinable_thread&&) = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

template <typename F, typename... Xs>
struct ParallelTensorFunctor
{
    F mF;
    static constexpr std::size_t NDIM = sizeof...(Xs);
    std::array<std::size_t, NDIM> mLens;
    std::array<std::size_t, NDIM> mStrides;
    std::size_t mN1d;

    ParallelTensorFunctor(F f, Xs... xs) : mF(f), mLens({static_cast<std::size_t>(xs)...})
    {
        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
        mN1d = mStrides[0] * mLens[0];
    }

    std::array<std::size_t, NDIM> GetNdIndices(std::size_t i) const
    {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = i / mStrides[idim];
            i -= indices[idim] * mStrides[idim];
        }

        return indices;
    }

    void operator()(std::size_t num_thread = 1) const
    {
        std::size_t work_per_thread = (mN1d + num_thread - 1) / num_thread;

        std::vector<joinable_thread> threads(num_thread);

        for(std::size_t it = 0; it < num_thread; ++it)
        {
            std::size_t iw_begin = it * work_per_thread;
            std::size_t iw_end   = std::min((it + 1) * work_per_thread, mN1d);

            auto f = [=, *this] {
                for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                {
                    call_f_unpack_args(mF, GetNdIndices(iw));
                }
            };
            threads[it] = joinable_thread(f);
        }
    }
};

template <typename F, typename... Xs>
auto make_ParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

template <typename T>
struct Tensor
{
    using Descriptor = HostTensorDescriptor;
    using Data       = std::vector<T>;

    template <typename X>
    Tensor(std::initializer_list<X> lens) : mDesc(lens), mData(GetElementSpaceSize())
    {
    }

    template <typename X, typename Y>
    Tensor(std::initializer_list<X> lens, std::initializer_list<Y> strides)
        : mDesc(lens, strides), mData(GetElementSpaceSize())
    {
    }

    template <typename Lengths>
    Tensor(const Lengths& lens) : mDesc(lens), mData(GetElementSpaceSize())
    {
    }

    template <typename Lengths, typename Strides>
    Tensor(const Lengths& lens, const Strides& strides)
        : mDesc(lens, strides), mData(GetElementSpaceSize())
    {
    }

    Tensor(const Descriptor& desc) : mDesc(desc), mData(GetElementSpaceSize()) {}

    template <typename OutT>
    Tensor<OutT> CopyAsType() const
    {
        Tensor<OutT> ret(mDesc);

        ck::ranges::transform(
            mData, ret.mData.begin(), [](auto value) { return ck::type_convert<OutT>(value); });

        return ret;
    }

    Tensor()              = delete;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&)      = default;

    ~Tensor() = default;

    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    template <typename FromT>
    explicit Tensor(const Tensor<FromT>& other) : Tensor(other.template CopyAsType<T>())
    {
    }
    void savetxt(std::string file_name, std::string dtype = "float")
    {
        std::ofstream file(file_name);

        if(file.is_open())
        {
            for(auto& itm : mData)
            {
                if(dtype == "float")
                    file << ck::type_convert<float>(itm) << std::endl;
                else if(dtype == "int")
                    file << ck::type_convert<int>(itm) << std::endl;
                else
                    // TODO: we didn't implement operator<< for all custom
                    // data types, here fall back to float in case compile error
                    file << ck::type_convert<float>(itm) << std::endl;
            }
            file.close();
        }
        else
        {
            // Print an error message to the standard error
            // stream if the file cannot be opened.
            throw std::runtime_error(std::string("unable to open file:") + file_name);
        }
    }
    decltype(auto) GetLengths() const { return mDesc.GetLengths(); }

    decltype(auto) GetStrides() const { return mDesc.GetStrides(); }

    std::size_t GetNumOfDimension() const { return mDesc.GetNumOfDimension(); }

    std::size_t GetElementSize() const { return mDesc.GetElementSize(); }

    std::size_t GetElementSpaceSize() const
    {
        if constexpr(ck::is_packed_type_v<ck::remove_cvref_t<T>>)
        {
            return (mDesc.GetElementSpaceSize() + 1) / ck::packed_size_v<ck::remove_cvref_t<T>>;
        }
        else
        {
            return mDesc.GetElementSpaceSize();
        }
    }

    std::size_t GetElementSpaceSizeInBytes() const { return sizeof(T) * GetElementSpaceSize(); }

    void SetZero() { ck::ranges::fill<T>(mData, T{0}); }

    template <typename F>
    void ForEach_impl(F&& f, std::vector<size_t>& idx, size_t rank)
    {
        if(rank == mDesc.GetNumOfDimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.GetLengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(F&& f)
    {
        std::vector<size_t> idx(mDesc.GetNumOfDimension(), 0);
        ForEach_impl(std::forward<F>(f), idx, size_t(0));
    }

    template <typename F>
    void ForEach_impl(const F&& f, std::vector<size_t>& idx, size_t rank) const
    {
        if(rank == mDesc.GetNumOfDimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.GetLengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<const F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(const F&& f) const
    {
        std::vector<size_t> idx(mDesc.GetNumOfDimension(), 0);
        ForEach_impl(std::forward<const F>(f), idx, size_t(0));
    }

    template <typename G>
    void GenerateTensorValue(G g, std::size_t num_thread = 1)
    {
        switch(mDesc.GetNumOfDimension())
        {
        case 1: {
            auto f = [&](auto i) { (*this)(i) = g(i); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0])(num_thread);
            break;
        }
        case 2: {
            auto f = [&](auto i0, auto i1) { (*this)(i0, i1) = g(i0, i1); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0], mDesc.GetLengths()[1])(num_thread);
            break;
        }
        case 3: {
            auto f = [&](auto i0, auto i1, auto i2) { (*this)(i0, i1, i2) = g(i0, i1, i2); };
            make_ParallelTensorFunctor(
                f, mDesc.GetLengths()[0], mDesc.GetLengths()[1], mDesc.GetLengths()[2])(num_thread);
            break;
        }
        case 4: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3) {
                (*this)(i0, i1, i2, i3) = g(i0, i1, i2, i3);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3])(num_thread);
            break;
        }
        case 5: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4) {
                (*this)(i0, i1, i2, i3, i4) = g(i0, i1, i2, i3, i4);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4])(num_thread);
            break;
        }
        case 6: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4, auto i5) {
                (*this)(i0, i1, i2, i3, i4, i5) = g(i0, i1, i2, i3, i4, i5);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4],
                                       mDesc.GetLengths()[5])(num_thread);
            break;
        }
        case 12: {
            auto f = [&](auto i0,
                         auto i1,
                         auto i2,
                         auto i3,
                         auto i4,
                         auto i5,
                         auto i6,
                         auto i7,
                         auto i8,
                         auto i9,
                         auto i10,
                         auto i11) {
                (*this)(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) =
                    g(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4],
                                       mDesc.GetLengths()[5],
                                       mDesc.GetLengths()[6],
                                       mDesc.GetLengths()[7],
                                       mDesc.GetLengths()[8],
                                       mDesc.GetLengths()[9],
                                       mDesc.GetLengths()[10],
                                       mDesc.GetLengths()[11])(num_thread);
            break;
        }
        default: throw std::runtime_error("unspported dimension");
        }
    }

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        return mDesc.GetOffsetFromMultiIndex(is...) / ck::packed_size_v<ck::remove_cvref_t<T>>;
    }

    template <typename... Is>
    T& operator()(Is... is)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...) /
                     ck::packed_size_v<ck::remove_cvref_t<T>>];
    }

    template <typename... Is>
    const T& operator()(Is... is) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...) /
                     ck::packed_size_v<ck::remove_cvref_t<T>>];
    }

    T& operator()(std::vector<std::size_t> idx)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(idx) / ck::packed_size_v<ck::remove_cvref_t<T>>];
    }

    const T& operator()(std::vector<std::size_t> idx) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(idx) / ck::packed_size_v<ck::remove_cvref_t<T>>];
    }

    typename Data::iterator begin() { return mData.begin(); }

    typename Data::iterator end() { return mData.end(); }

    typename Data::pointer data() { return mData.data(); }

    typename Data::const_iterator begin() const { return mData.begin(); }

    typename Data::const_iterator end() const { return mData.end(); }

    typename Data::const_pointer data() const { return mData.data(); }

    typename Data::size_type size() const { return mData.size(); }

    template <typename U = T>
    auto AsSpan() const
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::add_const_t<std::remove_reference_t<U>>;
        return ck::span<Element>{reinterpret_cast<Element*>(data()), size() * FromSize / ToSize};
    }

    template <typename U = T>
    auto AsSpan()
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::remove_reference_t<U>;
        return ck::span<Element>{reinterpret_cast<Element*>(data()), size() * FromSize / ToSize};
    }

    Descriptor mDesc;
    Data mData;
};
