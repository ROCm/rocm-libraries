#pragma once

#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#define BUNNIES_DEFINE_MODE(NAME, IDX_T)             \
    class NAME : public ::bunnies::mode<NAME, IDX_T> \
    {                                                \
    };

namespace bunnies
{

///////////////////////////
/////// Tensor view ///////
///////////////////////////

template <typename IdxT = int>
struct slice
{
    IdxT offset = 0, size = 0;
};

namespace detail
{
template <typename IdxT>
__device__ auto offset(IdxT i)
{
    return i;
}
template <typename IdxT>
__device__ auto offset(slice<IdxT> i)
{
    return i.offset;
}
template <typename IdxT>
__device__ auto size(IdxT i)
{
    return 0;
}
template <typename IdxT>
__device__ auto size(slice<IdxT> i)
{
    return i.size;
}
} // namespace detail

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
struct tensor_view
{
    static constexpr int dim = Dim;
    using tuple_t            = std::array<IdxT, Dim>;
    using idx_t              = IdxT;
    using offset_t           = OffsetT;

    OffsetT offset;
    std::array<IdxT, Dim> shape, stride;

    __device__ auto delta(std::array<IdxT, Dim> const& idx) const -> IdxT
    {
        IdxT p = 0;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            p += idx[i] * stride[i];
        }
        return p;
    }
    template <std::integral... I>
    __device__ auto delta(I... idx) const -> IdxT
    {
        static_assert(sizeof...(I) == Dim);

        std::array<IdxT, Dim> offsets = {static_cast<IdxT>(idx)...};
        return delta(offsets);
    }
    __device__ auto operator()(std::array<IdxT, Dim> const& idx) const -> OffsetT
    {
        return offset + delta(idx);
    }
    template <std::integral... I>
    __device__ auto operator()(I... idx) const -> OffsetT
    {
        return offset + delta(std::forward<I>(idx)...);
    }

    // Checks whether a multi-index is within bounds; does not check whether index is negative
    __device__ auto in_bounds(std::array<IdxT, Dim> const& idx) const -> bool
    {
        bool ok = true;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            ok = ok && idx[i] < shape[i];
        }
        return ok;
    }
    template <std::integral... I>
    __device__ auto in_bounds(I... idx) const -> bool
    {
        std::array<IdxT, Dim> offsets = {static_cast<IdxT>(idx)...};
        return in_bounds(offsets);
    }
    // Checks whether a multi-index is within bounds; checks that indices are non-negative
    __device__ auto in_bounds_maybe_negative(std::array<IdxT, Dim> const& idx) const -> bool
    {
        bool ok = true;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            ok = ok && idx[i] >= 0 && idx[i] < shape[i];
        }
        return ok;
    }
    template <std::integral... I>
    __device__ auto in_bounds_maybe_negative(I... idx) const -> bool
    {
        std::array<IdxT, Dim> offsets = {static_cast<IdxT>(idx)...};
        return in_bounds_maybe_negative(offsets);
    }

    template <typename... I>
    __device__ auto subview(I&&... idx_or_slice) const
    {
        static_assert(sizeof...(I) == Dim);
        static_assert(((std::is_same_v<std::decay_t<I>, IdxT> ||
                        std::is_same_v<std::decay_t<I>, slice<IdxT>>) &&
                       ...));

        constexpr int SubDim =
            (static_cast<int>(std::is_same_v<std::decay_t<I>, slice<IdxT>>) + ...);

        std::array<IdxT, Dim> offsets  = {detail::offset(idx_or_slice)...};
        std::array<bool, Dim> is_slice = {std::is_same_v<std::decay_t<I>, slice<IdxT>>...};

        OffsetT suboffset = offset;
        std::array<IdxT, SubDim> subshape, substride;
        int j = 0;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            suboffset += offsets[i] * stride[i];
            if(is_slice[i])
            {
                subshape[j]  = shape[i];
                substride[j] = stride[i];
                ++j;
            }
        }
        return tensor_view<SubDim, IdxT, OffsetT>(suboffset, subshape, substride);
    }
};

template <int Dim, typename T, typename IdxT = int>
using memref = tensor_view<Dim, IdxT, T*>;

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto
make_view(OffsetT offset, std::array<IdxT, Dim> const& shape, std::array<IdxT, Dim> const& stride)
{
    return tensor_view<Dim, IdxT, OffsetT>{offset, shape, stride};
}

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto make_view_col_major(std::array<IdxT, Dim> const& shape)
{
    std::array<IdxT, Dim> stride;
    stride[0] = 1;
    for(int mode = 0; mode < Dim - 1; ++mode)
    {
        stride[mode + 1] = stride[mode] * shape[mode];
    }
    return tensor_view<Dim, IdxT, OffsetT>{0, shape, stride};
}

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto make_view_row_major(std::array<IdxT, Dim> const& shape)
{
    std::array<IdxT, Dim> stride;
    stride[Dim - 1] = 1;
    for(int mode = Dim - 1; mode > 0; --mode)
    {
        stride[mode - 1] = stride[mode] * shape[mode];
    }
    return tensor_view<Dim, IdxT, OffsetT>{0, shape, stride};
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto
make_memref(T* ptr, std::array<IdxT, Dim> const& shape, std::array<IdxT, Dim> const& stride)
{
    return make_view<Dim, IdxT, T*>(ptr, shape, stride);
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto make_memref_col_major(std::array<IdxT, Dim> const& shape)
{
    return make_view_col_major<Dim, IdxT, T*>(shape);
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto make_memref_row_major(std::array<IdxT, Dim> const& shape)
{
    return make_view_row_major<Dim, IdxT, T*>(shape);
}

///////////////////////////
/////// Named view ////////
///////////////////////////

template <typename Derived, typename IdxT = int>
class mode
{
public:
    using idx_t = IdxT;

    __host__ __device__ constexpr mode() : value(0) {}
    __host__ __device__ constexpr mode(idx_t v) : value{v} {}
    template <typename... Ts>
    __host__ __device__ constexpr mode(std::tuple<Ts...> const& tpl)
        : value{std::get<Derived>(tpl).get()}
    {
    }

    __host__ __device__ constexpr auto get() const -> idx_t { return value; }

    // clang-format off
    __host__ __device__ constexpr auto operator<=> (mode const&) const = default;
    // clang-format on

    __host__ __device__ constexpr auto operator+(mode const& other) const
    {
        return Derived{value + other.value};
    }
    __host__ __device__ constexpr auto operator-(mode const& other) const
    {
        return Derived{value - other.value};
    }
    __host__ __device__ constexpr auto operator-() const { return Derived{-value}; }

private:
    idx_t value;
};


template <typename Mode>
concept mode_concept = requires(Mode m)
{
    typename Mode::idx_t;
    {
        m.get()
    } -> std::same_as<typename Mode::idx_t>;
}
&&std::is_base_of_v<mode<Mode, typename Mode::idx_t>, Mode>;

template <mode_concept Mode, std::integral Scalar>
__host__ __device__ inline constexpr auto operator*(Mode m, Scalar alpha)
{
    return Mode{m.get() * alpha};
}
template <std::integral Scalar, mode_concept Mode>
__host__ __device__ inline constexpr auto operator*(Scalar alpha, Mode m)
{
    return Mode{m.get() * alpha};
}
template <mode_concept Mode, std::integral Scalar>
__host__ __device__ inline constexpr auto operator/(Mode m, Scalar alpha)
{
    return Mode{m.get() / alpha};
}

template <typename Tuple>
concept mode_tuple_concept = requires
{
    typename std::tuple_size<Tuple>::type;
}
&&[]<std::size_t... Is>(std::index_sequence<Is...>) {
    return (mode_concept<std::remove_cvref_t<std::tuple_element_t<Is, Tuple>>> && ...);
}(std::make_index_sequence<std::tuple_size_v<Tuple>>{});

namespace detail
{
template <typename T, typename Tuple>
struct has_type;
template <typename T, typename... Us>
struct has_type<T, std::tuple<Us...>> : std::disjunction<std::is_same<T, Us>...>
{
};
template <typename T, typename Tuple>
constexpr bool has_type_v = has_type<T, Tuple>();
} // namespace detail

template <mode_concept M1, mode_concept M2>
requires(!std::is_same_v<M1, M2>) __host__ __device__ constexpr auto operator+(M1 a, M2 b)
{
    return std::tuple<M1, M2>{a, b};
}
template <mode_tuple_concept MT, mode_concept M>
__host__ __device__ constexpr auto operator+(MT a, M b)
{
    if constexpr(detail::has_type_v<M, MT>)
    {
        auto& av = std::get<M>(a);
        av       = av + b;
        return a;
    }
    else
    {
        return std::tuple_cat(a, std::make_tuple(b));
    }
}
template <mode_concept M, mode_tuple_concept MT>
__host__ __device__ constexpr auto operator+(M a, MT b)
{
    if constexpr(detail::has_type_v<M, MT>)
    {
        auto& bv = std::get<M>(b);
        bv       = a + bv;
        return b;
    }
    else
    {
        return std::tuple_cat(std::make_tuple(a), b);
    }
}
template <mode_tuple_concept MT, mode_concept... Modes>
__host__ __device__ constexpr auto operator+(MT a, std::tuple<Modes...> b)
{
    return (a + ... + std::get<Modes>(b));
}

template <mode_tuple_concept MT>
__host__ __device__ constexpr auto operator-(MT a)
{
    std::apply([&]<mode_concept... Modes>(Modes&... modes) { ((modes = -modes), ...); }, a);
    return a;
}

template <mode_concept M1, mode_concept M2>
requires(!std::is_same_v<M1, M2>) __host__ __device__ constexpr auto operator-(M1 a, M2 b)
{
    return std::tuple<M1, M2>{a, -b};
}
template <mode_tuple_concept MT, mode_concept M>
__host__ __device__ constexpr auto operator-(MT a, M b)
{
    if constexpr(detail::has_type_v<M, MT>)
    {
        auto& av = std::get<M>(a);
        av       = av - b;
        return a;
    }
    else
    {
        return std::tuple_cat(a, std::make_tuple(-b));
    }
}
template <mode_concept M, mode_tuple_concept MT>
__host__ __device__ constexpr auto operator-(M a, MT b)
{
    if constexpr(detail::has_type_v<M, MT>)
    {
        b        = -b;
        auto& bv = std::get<M>(b);
        bv       = a + bv;
        return b;
    }
    else
    {
        return std::tuple_cat(std::make_tuple(a), -b);
    }
}
template <mode_tuple_concept MT, mode_concept... Modes>
__host__ __device__ constexpr auto operator-(MT a, std::tuple<Modes...> b)
{
    return (a - ... - std::get<Modes>(b));
}

template <mode_tuple_concept MT, std::integral Scalar>
__host__ __device__ constexpr auto operator*(MT a, Scalar alpha)
{
    std::apply([&]<mode_concept... Modes>(Modes&... modes) { ((modes = modes * alpha), ...); }, a);
    return a;
}
template <std::integral Scalar, mode_tuple_concept MT>
__host__ __device__ constexpr auto operator*(Scalar alpha, MT a)
{
    return a * alpha;
}
template <mode_tuple_concept MT, std::integral Scalar>
__host__ __device__ constexpr auto operator/(MT a, Scalar alpha)
{
    std::apply([&]<mode_concept... Modes>(Modes&... modes) { ((modes = modes / alpha), ...); }, a);
    return a;
}

namespace detail
{
template <typename Tuple>
struct named_view_traits;

template <mode_concept... Modes>
struct named_view_traits<std::tuple<Modes...>>
{
    static constexpr int dim = sizeof...(Modes);
    using idx_acc_t          = std::common_type_t<typename Modes::idx_t...>;
};
} // namespace detail


// Modes should be generated from
//
// BUNNIES_DEFINE_MODE(mode_name, integer_type)
//
// Example:
//
// BUNNIES_DEFINE_MODE(I, int)
// BUNNIES_DEFINE_MODE(J, int)
//
// Modes can be added to form a tuple, e.g.
//
// I(4) + J(8)
//
// gives an (I,J) tuple.
//
// Example usage of named view:
//
// auto nv = named_view(0, I(4) + J(8), I(8) + J(1));
// nv(I(1) + J(1)) -> 1 * 8 + 1 * 1 = 9
//
template <mode_tuple_concept ModeTuple,
          typename OffsetT = detail::named_view_traits<ModeTuple>::idx_acc_t>
class named_view
{
public:
    using traits             = detail::named_view_traits<ModeTuple>;
    static constexpr int dim = traits::dim;
    using idx_acc_t          = traits::idx_acc_t;
    using offset_t           = OffsetT;
    using mode_tuple         = ModeTuple;

    __host__ __device__ constexpr named_view(offset_t offset, mode_tuple shape, mode_tuple stride)
        : offset_{std::move(offset)}
        , shape_{std::move(shape)}
        , stride_{std::move(stride)}
    {
    }

    template <mode_tuple_concept Tuple>
    __host__ __device__ constexpr auto delta(Tuple const& idx) const -> idx_acc_t
    {
        return std::apply([&]<mode_concept... Modes>(Modes const&... modes) -> idx_acc_t {
            return ((modes.get() * std::get<Modes>(stride_).get()) + ...);
        }, idx);
    }
    template <mode_concept Mode>
    __host__ __device__ constexpr auto delta(Mode const& idx) const -> idx_acc_t
    {
        return idx.get() * std::get<Mode>(stride_).get();
    }
    template <typename T>
    __host__ __device__ constexpr auto operator()(T const& idx) const -> offset_t
    {
        return offset_ + delta(idx);
    }
    template <mode_tuple_concept Tuple>
    __host__ __device__ constexpr auto in_bounds(Tuple const& idx) const -> bool
    {
        return std::apply([&]<mode_concept... Modes>(Modes const&... modes) -> bool {
            return ((modes < std::get<Modes>(shape_)) && ...);
        }, idx);
    }
    template <mode_concept Mode>
    __host__ __device__ constexpr auto in_bounds(Mode const& idx) const -> bool
    {
        return idx < std::get<Mode>(shape_);
    }
    template <mode_tuple_concept Tuple>
    __host__ __device__ constexpr auto in_bounds_maybe_negative(Tuple const& idx) const -> bool
    {
        return std::apply([&]<mode_concept... Modes>(Modes const&... modes) -> bool {
            return ((modes >= Modes{0} && modes < std::get<Modes>(shape_)) && ...);
        }, idx);
    }
    template <mode_concept Mode>
    __host__ __device__ constexpr auto in_bounds_maybe_negative(Mode const& idx) const -> bool
    {
        return idx >= Mode{0} && idx < std::get<Mode>(shape_);
    }

    template <mode_tuple_concept Tuple>
    __host__ __device__ constexpr auto
    moved_by(Tuple const& by) const -> named_view<mode_tuple, offset_t>
    {
        const auto new_offset = offset_ + delta(by);
        const auto new_shape =
            std::apply([&]<mode_concept... Modes>(Modes const&... off) -> mode_tuple {
            auto s = shape_;
            ((std::get<Modes>(s) = std::get<Modes>(s) - off), ...);
            return s;
        }, by);
        return named_view<mode_tuple, offset_t>(new_offset, new_shape, stride_);
    }
    template <mode_concept Mode>
    __host__ __device__ constexpr auto
    moved_by(Mode const& by) const -> named_view<mode_tuple, offset_t>
    {
        return moved_by(std::make_tuple(by));
    }

    __host__ __device__ constexpr auto offset() const -> offset_t { return offset_; }
    __host__ __device__ constexpr auto shape() const -> mode_tuple const& { return shape_; }
    __host__ __device__ constexpr auto stride() const -> mode_tuple const& { return stride_; }

private:
    offset_t offset_;
    mode_tuple shape_, stride_;
};

template <mode_concept... Modes>
struct mode_order
{
};

template <mode_tuple_concept ModeTuple, mode_concept... Modes>
__host__ __device__ constexpr auto make_canonical_stride(ModeTuple const& shape,
                                                         mode_order<Modes...>)
{
    static_assert(sizeof...(Modes) == std::tuple_size_v<ModeTuple>,
                  "Number of modes in mode order must match the shape tuple size");
    static_assert(((detail::has_type_v<Modes, ModeTuple>) && ...),
                  "Every mode in mode order must be contained in the mode shape tuple");

    using idx_acc_t = detail::named_view_traits<ModeTuple>::idx_acc_t;

    idx_acc_t last_stride = 1;
    auto stride           = ModeTuple{};
    ([&] {
        std::get<Modes>(stride) = Modes{last_stride};
        last_stride *= std::get<Modes>(shape).get();
    }(), ...);
    return stride;
}
template <mode_concept Mode>
__host__ __device__ constexpr auto make_canonical_stride(Mode const& mode,
                                                         mode_order<Mode> order = {})
{
    return make_canonical_stride(std::make_tuple(mode), order);
}

template <mode_concept Mode>
__host__ __device__ constexpr auto make_named_view(Mode const& shape, mode_order<Mode> order = {})
{
    return named_view(0, std::make_tuple(shape), make_canonical_stride(shape, order));
}
template <mode_tuple_concept ModeTuple, mode_concept... Modes>
__host__ __device__ constexpr auto make_named_view(ModeTuple const& shape,
                                                   mode_order<Modes...> order)
{
    return named_view(0, shape, make_canonical_stride(shape, order));
}
template <typename T, mode_concept Mode>
__host__ __device__ constexpr auto
make_named_memref(T* ptr, Mode const& shape, mode_order<Mode> order = {})
{
    return named_view(ptr, std::make_tuple(shape), make_canonical_stride(shape, order));
}
template <typename T, mode_tuple_concept ModeTuple, mode_concept... Modes>
__host__ __device__ constexpr auto
make_named_memref(T* ptr, ModeTuple const& shape, mode_order<Modes...> order)
{
    return named_view(ptr, shape, make_canonical_stride(shape, order));
}
} // namespace bunnies
