/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_TENSOR_HOLDER_HPP
#define GUARD_TENSOR_HOLDER_HPP

#include <miopen_utils/network_data.hpp>
#include <miopen_utils/tensor_desc.hpp>
#include <common_utils/tuple_utils.hpp>
#include <common_utils/ford.hpp>
#include <common_utils/functional.hpp>
#include <common_utils/type_name.hpp>
#include <common_utils/each_args.hpp>
#include <common_utils/bfloat16.hpp>
#include <common_utils/random.hpp>

// When building tests, include internal MIOpen tensor type so test code can
// access miopen::TensorDescriptor, miopen::tien, miopen::deref, etc. via
// tensor_holder.hpp's transitive includes. Test code (Layer 4) legitimately
// needs these internals for constructing conv::ProblemDescription, calling
// DeriveBNTensorDescriptor, and similar internal test infrastructure.
// Driver and miopen_utils builds do NOT define MIOPEN_BUILD_TESTING, so they
// remain clean of internal includes.
#ifdef MIOPEN_BUILD_TESTING
#include <miopen/tensor.hpp>
#endif

#include <miopen_utils/serialize.hpp>

#include <half/half.hpp>
using half         = half_float::half;
using hip_bfloat16 = bfloat16;
#include <common_utils/float8.hpp>

#include <iomanip>
#include <fstream>

template <class F>
void visit_tensor_size(std::size_t n, F f)
{
    switch(n)
    {
    case 0: {
        f(std::integral_constant<std::size_t, 0>{});
        break;
    }
    case 1: {
        f(std::integral_constant<std::size_t, 1>{});
        break;
    }
    case 2: {
        f(std::integral_constant<std::size_t, 2>{});
        break;
    }
    case 3: {
        f(std::integral_constant<std::size_t, 3>{});
        break;
    }
    case 4: {
        f(std::integral_constant<std::size_t, 4>{});
        break;
    }
    case 5: {
        f(std::integral_constant<std::size_t, 5>{});
        break;
    }
    default: throw std::runtime_error("Unknown tensor size");
    }
}

template <class T>
struct miopen_type;

template <>
struct miopen_type<float> : std::integral_constant<miopenDataType_t, miopenFloat>
{
};

template <>
struct miopen_type<double> : std::integral_constant<miopenDataType_t, miopenDouble>
{
};

template <>
struct miopen_type<half_float::half> : std::integral_constant<miopenDataType_t, miopenHalf>
{
};
template <>
struct miopen_type<bfloat16> : std::integral_constant<miopenDataType_t, miopenBFloat16>
{
};

template <>
struct miopen_type<int8_t> : std::integral_constant<miopenDataType_t, miopenInt8>
{
};

template <>
struct miopen_type<int> : std::integral_constant<miopenDataType_t, miopenInt32>
{
};

template <>
struct miopen_type<int64_t> : std::integral_constant<miopenDataType_t, miopenInt64>
{
};

template <>
struct miopen_type<float8_fnuz> : std::integral_constant<miopenDataType_t, miopenFloat8_fnuz>
{
};

template <>
struct miopen_type<bfloat8_fnuz> : std::integral_constant<miopenDataType_t, miopenBFloat8_fnuz>
{
};

template <>
struct miopen_type<uint8_t> : std::integral_constant<miopenDataType_t, miopenInt8>
{
};

template <>
struct miopen_type<uint16_t> : std::integral_constant<miopenDataType_t, miopenHalf>
{
};

template <>
struct miopen_type<uint64_t> : std::integral_constant<miopenDataType_t, miopenInt64>
{
};

template <class T>
struct tensor
{
    using value_type = T;
    TensorDesc desc;
    std::vector<T> data;

    tensor() {}

    template <class X>
    tensor(const std::vector<X>& dims)
        : desc(miopen_type<T>{}, std::vector<size_t>(dims.begin(), dims.end())),
          data(desc.GetElementSpace())
    {
    }

    template <class X>
    tensor(const std::vector<X>& dims, const std::vector<X>& strides)
        : desc(miopen_type<T>{},
               std::vector<size_t>(dims.begin(), dims.end()),
               std::vector<size_t>(strides.begin(), strides.end())),
          data(desc.GetElementSpace())
    {
        assert(dims.size() == strides.size());
    }

    template <class X>
    tensor(miopenTensorLayout_t layout, const std::vector<X>& dims)
        : desc(miopen_type<T>{}, layout, std::vector<size_t>(dims.begin(), dims.end())),
          data(desc.GetElementSpace())
    {
    }

    template <class X>
    tensor(miopenTensorLayout_t /*layout*/,
           const std::vector<X>& dims,
           const std::vector<X>& strides)
        : desc(miopen_type<T>{},
               std::vector<size_t>(dims.begin(), dims.end()),
               std::vector<size_t>(strides.begin(), strides.end())),
          data(desc.GetElementSpace())
    {
        assert(dims.size() == strides.size());
    }

    tensor(std::size_t n, std::size_t c, std::size_t h, std::size_t w)
        : desc(miopen_type<T>{}, std::vector<size_t>{n, c, h, w}), data(n * c * h * w)
    {
    }

    tensor(miopenTensorLayout_t layout, std::size_t n, std::size_t c, std::size_t h, std::size_t w)
        : desc(miopen_type<T>{}, layout, std::vector<size_t>{n, c, h, w}),
          data(desc.GetElementSpace())
    {
    }

    tensor(std::size_t n, std::size_t c, std::size_t d, std::size_t h, std::size_t w)
        : desc(miopen_type<T>{}, std::vector<size_t>{n, c, d, h, w}), data(n * c * d * h * w)
    {
    }

    tensor(std::size_t n) : desc(miopen_type<T>{}, std::vector<size_t>{n}), data(n) {}

    /// Construct from a TensorDesc (deep copy). Exact match so that
    /// tensor<T>{other.desc} is unambiguous despite TensorDesc's conversion
    /// operators to both the opaque handle and (in tests) the internal type.
    explicit tensor(const TensorDesc& rhs) : desc(rhs), data(rhs.GetElementSpace()) {}

    /// Construct from an opaque tensor descriptor handle (deep copy).
    explicit tensor(miopenTensorDescriptor_t rhs) : desc(rhs)
    {
        assert(desc.GetType() == miopen_type<T>{}
               /// In the driver, T is input tensor type, but output tensor holders
               /// are instantiated with T as well. This leads to false assertion
               /// failures when T is INT8 because output type is different.
               /// \todo Get rid of this hack when the driver is improved:
               || (miopen_type<T>{} == miopenInt8 && desc.GetType() == miopenInt32));
        data.resize(desc.GetElementSpace());
    }

#ifdef MIOPEN_BUILD_TESTING
    /// Test-only: construct from the internal miopen::TensorDescriptor value type.
    /// Test code (Layer 4) still produces miopen::TensorDescriptor from internal
    /// helpers (e.g. ProblemDescription queries); this bridges it to TensorDesc
    /// without exposing the internal type to driver/miopen_utils builds.
    tensor(const miopen::TensorDescriptor& rhs)
        : desc(rhs.GetType(),
               std::vector<size_t>(rhs.GetLengths().begin(), rhs.GetLengths().end()),
               std::vector<size_t>(rhs.GetStrides().begin(), rhs.GetStrides().end())),
          data(desc.GetElementSpace())
    {
    }
#endif

    size_t GetDataByteSize() const { return GetSize() * sizeof(T); }

    size_t GetSize() const { return desc.GetElementSpace(); }

    template <class G>
    tensor& generate(G g) &
    {
        if(this->desc.GetVectorLength() > 1)
            this->generate_vect_impl(g);
        else
            this->generate_impl(g);
        return *this;
    }

    template <class G>
    tensor&& generate(G g) &&
    {
        if(this->desc.GetVectorLength() > 1)
            this->generate_vect_impl(g);
        else
            this->generate_impl(g);
        return std::move(*this);
    }

    template <class G>
    void generate_impl(G g)
    {
        auto seed = std::accumulate(desc.GetLengths().begin(),
                                    desc.GetLengths().end(),
                                    std::size_t{521288629},
                                    [](auto x, auto y) {
                                        x ^= x << 1U;
                                        return x ^ y;
                                    });
        seed ^= data.size();
        seed ^= desc.GetLengths().size();
        prng::reset_seed(seed);
        auto iterator = data.begin();
        auto assign   = [&](T x) {
            *iterator = x;
            ++iterator;
        };
        this->for_each(
            miopen::compose(miopen::compose(assign, miopen::cast_to<T>()), std::move(g)));
    }

    template <class G>
    void generate_vect_impl(G g)
    {
        auto seed = std::accumulate(desc.GetLengths().begin(),
                                    desc.GetLengths().end(),
                                    std::size_t{521288629},
                                    [](auto x, auto y) {
                                        x ^= x << 1U;
                                        return x ^ y;
                                    });
        seed ^= data.size();
        seed ^= desc.GetLengths().size();
        prng::reset_seed(seed);
        auto iterator     = data.begin();
        auto vectorLength = desc.GetVectorLength();
        auto assign       = [&](T x) {
            assert(iterator < data.end());
            // for debugging
            for(auto i = 0; i < vectorLength; i++)
            {
                *(iterator + i) = x;
            }
            iterator += vectorLength;
        };
        this->for_each(
            miopen::compose(miopen::compose(assign, miopen::cast_to<T>()), std::move(g)));
    }

    template <class Loop, class F>
    struct for_each_unpacked
    {
        Loop loop;
        F f;
        template <class... Ts>
        auto operator()(Ts... xs) const -> decltype(f(xs...), void())
        {
            loop(xs...)(std::move(f));
        }

        struct any
        {
            any() {}
            template <class X>
            any(X)
            {
            }
        };

        [[noreturn]] void operator()(any = {},
                                     any = {},
                                     any = {},
                                     any = {},
                                     any = {},
                                     any = {},
                                     any = {},
                                     any = {},
                                     any = {}) const
        {
            throw std::runtime_error(
                "Arguments to for_each do not match tensor size or the function " +
                miopen::get_type_name<F>() + " can not be called.");
        }
    };

    struct for_each_handler
    {
        template <class Self, class Loop, class F, class Size>
        void operator()(Self* self, Loop loop, F f, Size /*size*/) const
        {
            auto dims = Tien<Size::value>(self->desc.GetLengths());
            miopen::unpack(for_each_unpacked<Loop, F>{loop, std::move(f)}, dims);
        }
    };

    template <class F>
    void for_each(F f) const
    {
        visit_tensor_size(
            desc.GetLengths().size(),
            std::bind(for_each_handler{}, this, miopen::ford, std::move(f), std::placeholders::_1));
    }

    template <class F>
    void par_for_each(F f) const
    {
        visit_tensor_size(
            desc.GetLengths().size(),
            std::bind(
                for_each_handler{}, this, miopen::par_ford, std::move(f), std::placeholders::_1));
    }

    template <class... Ts>
    T& operator()(Ts... xs)
    {
        assert(this->desc.GetIndex(xs...) < data.size());
        return this->data[this->desc.GetIndex(xs...)];
    }

    template <class... Ts>
    const T& operator()(Ts... xs) const
    {
        assert(this->desc.GetIndex(xs...) < data.size());
        return this->data[this->desc.GetIndex(xs...)];
    }

    template <class Integer, Integer N>
    const T& operator()(const std::array<Integer, N>& multi_id) const
    {
        auto f = [&](auto... is) { return this->desc.GetIndex(is...); };
        assert(miopen::unpack(f, multi_id) < data.size());
        return this->data[miopen::unpack(f, multi_id)];
    }

    T& operator[](std::size_t i) { return data.at(i); }

    const T& operator[](std::size_t i) const { return data.at(i); }

    typename std::vector<T>::iterator begin() { return data.begin(); }

    typename std::vector<T>::iterator end() { return data.end(); }

    typename std::vector<T>::const_iterator begin() const { return data.begin(); }

    typename std::vector<T>::const_iterator end() const { return data.end(); }

    friend std::ostream& operator<<(std::ostream& stream, const tensor& t)
    {
        return stream << t.desc.ToString();
    }

    template <size_t N, typename Stream>
    void dump_inner(size_t dim, std::array<size_t, N>& coord, Stream& stream) const
    {
        const auto lengths = this->desc.GetLengths();
        if(lengths.size() == 0)
        {
            // 0D special case: Just print the one value that we have and return.
            stream << (*this)(coord);
        }
        else if(dim + 1 == lengths.size())
        {
            // 1D special case: dump everything on one line
            for(size_t i = 0; i < lengths[dim]; ++i)
            {
                if(i != 0)
                    stream << ' ';

                coord[dim] = i;
                stream << std::setw(4) << (*this)(coord);
            }

            stream << '\n';
        }
        else
        {
            if(dim + 2 == lengths.size())
            {
                // 2D special case: Also print which 2D slice we are currently printing
                // Note: this is not needed for higher dimensions, as they will also pass
                // through this branch.
                stream << "slice [";
                for(size_t i = 0; i < dim; ++i)
                {
                    stream << coord[i] << ", ";
                }
                stream << ":, :]\n";
            }

            for(size_t i = 0; i < lengths[dim]; ++i)
            {
                coord[dim] = i;
                this->dump_inner<N>(dim + 1, coord, stream);
            }
        }
    }

    template <typename Stream = decltype(std::cout)>
    void dump(const char* name, Stream& stream = std::cout) const
    {
        const auto n = this->desc.GetLengths().size();
        stream << "==== " << name << ": " << *this << n << '\n';
        stream.fill(' ');

        const auto flags = stream.flags();

        visit_tensor_size(n, [&](const auto size) {
            constexpr size_t N = decltype(size)::value;
            std::array<size_t, N> coord;
            this->dump_inner<N>(0, coord, stream);
        });

        stream.flags(flags);
    }
};

template <class T>
void serialize(std::istream& s, tensor<T>& x)
{
    std::vector<std::size_t> lens;
    serialize(s, lens);
    std::vector<std::size_t> strides;
    serialize(s, strides);
    x.desc = TensorDesc{miopen_type<T>{}, lens, strides};
    serialize(s, x.data);
}

template <class T>
void serialize(std::ostream& s, const tensor<T>& x)
{
    auto lens    = x.desc.GetLengths();
    auto strides = x.desc.GetStrides();
    serialize(s, lens);
    serialize(s, strides);
    serialize(s, x.data);
}

struct tensor_generate
{
    template <class Tensor, class G>
    Tensor&& operator()(Tensor&& t, G g) const
    {
        return std::forward<Tensor>(t.generate(g));
    }
};

struct tensor_elem_gen_integer
{
    uint64_t max_value = 17;

    template <class... Ts>
    double operator()(Ts... Xs) const
    {
        static_assert(sizeof...(Ts) < 6,
                      "Dimensions in tensor_elem_gen_integer must be less than 6.");
        assert(max_value > 0);
        std::array<uint64_t, sizeof...(Ts)> left = {{Xs...}};
        std::array<uint64_t, 5> right            = {{613, 547, 701, 877, 1049}};
        uint64_t dot =
            std::inner_product(left.begin(), left.end(), right.begin(), static_cast<uint64_t>(173));
        return static_cast<double>(dot % max_value);
    }
};

#endif
