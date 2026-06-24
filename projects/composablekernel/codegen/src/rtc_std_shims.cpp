// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/rtc_std_shims.hpp"

namespace ck {
namespace host {

namespace {

// Bridge a standard header name onto the corresponding rocm-cxx header by
// pulling rocm's names into namespace std. Qualified lookup (std::X) honors the
// using-directive, so existing ck_tile code that writes std:: keeps working.
std::string bridge(const char* rocm_header)
{
    return std::string("#pragma once\n#include <rocm/") + rocm_header +
           ">\nnamespace std { using namespace rocm; }\n";
}

// ---- Minimal self-contained headers (no rocm-cxx equivalent) ----------------

// Compiler-coupled: layout must match what the compiler expects (pointer, size).
constexpr const char* kInitializerList = R"cpp(#pragma once
namespace std {
template <class E>
class initializer_list
{
public:
    using value_type      = E;
    using reference       = const E&;
    using const_reference = const E&;
    using size_type       = decltype(sizeof(0));
    using iterator        = const E*;
    using const_iterator  = const E*;

    constexpr initializer_list() noexcept : data_(nullptr), size_(0) {}

    constexpr size_type size() const noexcept { return size_; }
    constexpr const E* begin() const noexcept { return data_; }
    constexpr const E* end() const noexcept { return data_ + size_; }

private:
    const E* data_;
    size_type size_;

    // Used by the compiler to construct the list.
    constexpr initializer_list(const E* d, size_type s) noexcept : data_(d), size_(s) {}
};

template <class E>
constexpr const E* begin(initializer_list<E> il) noexcept { return il.begin(); }
template <class E>
constexpr const E* end(initializer_list<E> il) noexcept { return il.end(); }
}
)cpp";

// ck_tile specializes std::tuple_size / std::tuple_element for its own tuple,
// and (via print.hpp) builds a real std::tuple and feeds it to std::apply, so a
// minimal but working std::tuple + get + apply is required.
// __SIZE_TYPE__ is used instead of size_t: a global ::size_t (from
// hiprtc_runtime.h) plus rocm::size_t (via the bridge) make unqualified size_t
// ambiguous inside namespace std.
constexpr const char* kTuple = R"cpp(#pragma once
#include <type_traits>
#include <utility>
namespace std {

template <class... Ts>
class tuple;

template <class T>
struct tuple_size;
template <class... Ts>
struct tuple_size<tuple<Ts...>> : integral_constant<__SIZE_TYPE__, sizeof...(Ts)> {};
template <class T>
struct tuple_size<const T> : integral_constant<__SIZE_TYPE__, tuple_size<T>::value> {};
template <class T>
inline constexpr __SIZE_TYPE__ tuple_size_v = tuple_size<T>::value;

template <__SIZE_TYPE__ I, class T>
struct tuple_element;
template <__SIZE_TYPE__ I, class Head, class... Tail>
struct tuple_element<I, tuple<Head, Tail...>> : tuple_element<I - 1, tuple<Tail...>> {};
template <class Head, class... Tail>
struct tuple_element<0, tuple<Head, Tail...>> { using type = Head; };
template <__SIZE_TYPE__ I, class T>
struct tuple_element<I, const T> { using type = const typename tuple_element<I, T>::type; };
template <__SIZE_TYPE__ I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

template <>
class tuple<> {};

template <class Head, class... Tail>
class tuple<Head, Tail...>
{
public:
    Head head_;
    tuple<Tail...> tail_;
    constexpr tuple() = default;
    template <class H, class... T>
    constexpr explicit tuple(H&& h, T&&... t)
        : head_(static_cast<H&&>(h)), tail_(static_cast<T&&>(t)...) {}
};

namespace __rocm_tuple {
template <__SIZE_TYPE__ I>
struct getter
{
    template <class Head, class... Tail>
    static constexpr decltype(auto) get(tuple<Head, Tail...>& t) { return getter<I - 1>::get(t.tail_); }
    template <class Head, class... Tail>
    static constexpr decltype(auto) get(const tuple<Head, Tail...>& t) { return getter<I - 1>::get(t.tail_); }
};
template <>
struct getter<0>
{
    template <class Head, class... Tail>
    static constexpr Head& get(tuple<Head, Tail...>& t) { return t.head_; }
    template <class Head, class... Tail>
    static constexpr const Head& get(const tuple<Head, Tail...>& t) { return t.head_; }
};
} // namespace __rocm_tuple

template <__SIZE_TYPE__ I, class... Ts>
constexpr decltype(auto) get(tuple<Ts...>& t) { return __rocm_tuple::getter<I>::get(t); }
template <__SIZE_TYPE__ I, class... Ts>
constexpr decltype(auto) get(const tuple<Ts...>& t) { return __rocm_tuple::getter<I>::get(t); }
template <__SIZE_TYPE__ I, class... Ts>
constexpr decltype(auto) get(tuple<Ts...>&& t)
{
    return static_cast<tuple_element_t<I, tuple<Ts...>>&&>(__rocm_tuple::getter<I>::get(t));
}

namespace __rocm_tuple {
template <class F, class Tuple, __SIZE_TYPE__... I>
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, index_sequence<I...>)
{
    return static_cast<F&&>(f)(std::get<I>(static_cast<Tuple&&>(t))...);
}
} // namespace __rocm_tuple

template <class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t)
{
    return __rocm_tuple::apply_impl(
        static_cast<F&&>(f),
        static_cast<Tuple&&>(t),
        make_index_sequence<tuple_size_v<remove_reference_t<Tuple>>>{});
}

template <class... Ts>
constexpr tuple<decay_t<Ts>...> make_tuple(Ts&&... xs)
{
    return tuple<decay_t<Ts>...>(static_cast<Ts&&>(xs)...);
}
}
)cpp";

// Bridge onto rocm-cxx, then add the standard traits ck_tile uses that rocm-cxx
// does not provide. Placed in namespace std (some, like reference_wrapper, are
// formally in <functional>) so they resolve wherever ck_tile expects them.
constexpr const char* kTypeTraits = R"cpp(#pragma once
#include <rocm/type_traits.hpp>
// Some ck_tile headers use std::array without including <array> and rely on it
// being transitively available; <type_traits> is the near-universal include, so
// pull the array bridge in here. (The bridge only includes rocm/* headers, so
// there is no cycle back to this shim.)
#include <array>
namespace std {
using namespace rocm;

template <class T> auto __rocm_add_lvref(int) -> type_identity<T&>;
template <class T> auto __rocm_add_lvref(...) -> type_identity<T>;
template <class T> struct add_lvalue_reference : decltype(__rocm_add_lvref<T>(0)) {};
template <class T> using add_lvalue_reference_t = typename add_lvalue_reference<T>::type;

template <class T> auto __rocm_add_rvref(int) -> type_identity<T&&>;
template <class T> auto __rocm_add_rvref(...) -> type_identity<T>;
template <class T> struct add_rvalue_reference : decltype(__rocm_add_rvref<T>(0)) {};
template <class T> using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;

template <class T> struct remove_extent { using type = T; };
template <class T> struct remove_extent<T[]> { using type = T; };
template <class T, __SIZE_TYPE__ N> struct remove_extent<T[N]> { using type = T; };
template <class T> using remove_extent_t = typename remove_extent<T>::type;

template <class T>
struct decay {
private:
    using U = remove_reference_t<T>;
public:
    using type = conditional_t<is_array_v<U>,
                               remove_extent_t<U>*,
                               conditional_t<is_function_v<U>, add_pointer_t<U>, remove_cv_t<U>>>;
};
template <class T> using decay_t = typename decay<T>::type;

template <class...> struct conjunction : true_type {};
template <class B1> struct conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct conjunction<B1, Bn...> : conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};
template <class... B> inline constexpr bool conjunction_v = conjunction<B...>::value;

template <class...> struct disjunction : false_type {};
template <class B1> struct disjunction<B1> : B1 {};
template <class B1, class... Bn>
struct disjunction<B1, Bn...> : conditional_t<bool(B1::value), B1, disjunction<Bn...>> {};
template <class... B> inline constexpr bool disjunction_v = disjunction<B...>::value;

template <class B> struct negation : bool_constant<!bool(B::value)> {};
template <class B> inline constexpr bool negation_v = negation<B>::value;

template <class T> struct is_copy_constructible
    : bool_constant<__is_constructible(T, add_lvalue_reference_t<const T>)> {};
template <class T> inline constexpr bool is_copy_constructible_v = is_copy_constructible<T>::value;
template <class T> struct is_move_constructible
    : bool_constant<__is_constructible(T, add_rvalue_reference_t<T>)> {};
template <class T> inline constexpr bool is_move_constructible_v = is_move_constructible<T>::value;
template <class T> struct is_default_constructible : bool_constant<__is_constructible(T)> {};
template <class T> inline constexpr bool is_default_constructible_v = is_default_constructible<T>::value;
template <class T> struct is_copy_assignable
    : bool_constant<__is_assignable(add_lvalue_reference_t<T>, add_lvalue_reference_t<const T>)> {};
template <class T> inline constexpr bool is_copy_assignable_v = is_copy_assignable<T>::value;
template <class T> struct is_move_assignable
    : bool_constant<__is_assignable(add_lvalue_reference_t<T>, add_rvalue_reference_t<T>)> {};
template <class T> inline constexpr bool is_move_assignable_v = is_move_assignable<T>::value;

namespace __rocm_detail {
template <class T, bool = is_arithmetic_v<T>>
struct is_signed_impl : bool_constant<(T(-1) < T(0))> {};
template <class T> struct is_signed_impl<T, false> : false_type {};
}
template <class T> struct is_signed : __rocm_detail::is_signed_impl<T> {};
template <class T> inline constexpr bool is_signed_v = is_signed<T>::value;

template <class T> struct underlying_type { using type = __underlying_type(T); };
template <class T> using underlying_type_t = __underlying_type(T);

template <class T> T&& __rocm_declval() noexcept;
template <class F, class... Args>
struct is_invocable {
private:
    template <class F1, class... A1>
    static auto test(int) -> decltype(__rocm_declval<F1>()(__rocm_declval<A1>()...), true_type{});
    template <class, class...>
    static false_type test(...);
public:
    static constexpr bool value = decltype(test<F, Args...>(0))::value;
};
template <class F, class... Args>
inline constexpr bool is_invocable_v = is_invocable<F, Args...>::value;

// std::min/max/clamp are placed here (the near-universal include) because
// ck_tile uses them without always including <algorithm>.
template <class T> constexpr const T& min(const T& a, const T& b) { return b < a ? b : a; }
template <class T> constexpr const T& max(const T& a, const T& b) { return a < b ? b : a; }
template <class T, class Compare>
constexpr const T& min(const T& a, const T& b, Compare c) { return c(b, a) ? b : a; }
template <class T, class Compare>
constexpr const T& max(const T& a, const T& b, Compare c) { return c(a, b) ? b : a; }
template <class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return v < lo ? lo : (hi < v ? hi : v);
}

// std::multiplies/plus/minus etc. are placed here (the near-universal include)
// because ck_tile uses them (e.g. std::multiplies<index_t> in static_asserts)
// without always including <functional>.
template <class T = void> struct plus {
    constexpr T operator()(const T& a, const T& b) const { return a + b; }
};
template <> struct plus<void> {
    template <class A, class B>
    constexpr auto operator()(A&& a, B&& b) const -> decltype(a + b) { return a + b; }
};
template <class T = void> struct minus {
    constexpr T operator()(const T& a, const T& b) const { return a - b; }
};
template <> struct minus<void> {
    template <class A, class B>
    constexpr auto operator()(A&& a, B&& b) const -> decltype(a - b) { return a - b; }
};
template <class T = void> struct multiplies {
    constexpr T operator()(const T& a, const T& b) const { return a * b; }
};
template <> struct multiplies<void> {
    template <class A, class B>
    constexpr auto operator()(A&& a, B&& b) const -> decltype(a * b) { return a * b; }
};
template <class T = void> struct divides {
    constexpr T operator()(const T& a, const T& b) const { return a / b; }
};
template <> struct divides<void> {
    template <class A, class B>
    constexpr auto operator()(A&& a, B&& b) const -> decltype(a / b) { return a / b; }
};

template <class T>
class reference_wrapper {
    T* ptr_;
public:
    using type = T;
    constexpr reference_wrapper(T& r) noexcept : ptr_(__builtin_addressof(r)) {}
    reference_wrapper(T&&) = delete;
    constexpr operator T&() const noexcept { return *ptr_; }
    constexpr T& get() const noexcept { return *ptr_; }
};
}
)cpp";

constexpr const char* kConcepts = R"cpp(#pragma once
#include <type_traits>
#include <utility>
namespace std {
template <class T, class U>
concept same_as = is_same_v<T, U> && is_same_v<U, T>;

template <class From, class To>
concept convertible_to = is_convertible_v<From, To>;

template <class Derived, class Base>
concept derived_from = is_base_of_v<Base, Derived>;

template <class T>
concept integral = is_integral_v<T>;

template <class T>
concept floating_point = is_floating_point_v<T>;
}
)cpp";

// Reuse hipRTC's own global C math declarations rather than redefining them, so
// signatures match exactly.
constexpr const char* kCmath = R"cpp(#pragma once
namespace std {
using ::abs;
using ::fabs;
using ::fabsf;
using ::sqrt;
using ::sqrtf;
using ::sin;
using ::sinf;
using ::cos;
using ::cosf;
using ::tan;
using ::tanf;
using ::asin;
using ::asinf;
using ::acos;
using ::acosf;
using ::atan;
using ::atanf;
using ::atan2;
using ::atan2f;
using ::sinh;
using ::sinhf;
using ::cosh;
using ::coshf;
using ::tanh;
using ::tanhf;
using ::exp;
using ::expf;
using ::exp2;
using ::exp2f;
using ::expm1;
using ::expm1f;
using ::log;
using ::logf;
using ::log2;
using ::log2f;
using ::log10;
using ::log10f;
using ::pow;
using ::powf;
using ::ldexp;
using ::ldexpf;
using ::ceil;
using ::ceilf;
using ::floor;
using ::floorf;
using ::round;
using ::roundf;
using ::trunc;
using ::truncf;
using ::fmod;
using ::fmodf;
using ::fma;
using ::fmaf;
using ::fmin;
using ::fminf;
using ::fmax;
using ::fmaxf;
using ::copysign;
using ::copysignf;
using ::isnan;
using ::isinf;
using ::isfinite;
}
)cpp";

// These must be __host__ __device__: ck_tile device code (e.g.
// amd_buffer_addressing_builtins.hpp) calls std::memcpy from __device__ context.
// The __builtin_* forms are valid on both host and device.
constexpr const char* kCstring = R"cpp(#pragma once
namespace std {
__attribute__((host, device)) inline void* memcpy(void* d, const void* s, __SIZE_TYPE__ n) { return __builtin_memcpy(d, s, n); }
__attribute__((host, device)) inline void* memmove(void* d, const void* s, __SIZE_TYPE__ n) { return __builtin_memmove(d, s, n); }
__attribute__((host, device)) inline void* memset(void* d, int c, __SIZE_TYPE__ n) { return __builtin_memset(d, c, n); }
__attribute__((host, device)) inline int   memcmp(const void* a, const void* b, __SIZE_TYPE__ n) { return __builtin_memcmp(a, b, n); }
}
)cpp";

// min/max/clamp live in the type_traits shim; just pull them in here.
constexpr const char* kAlgorithm = R"cpp(#pragma once
#include <type_traits>
)cpp";

// rocm::size_t is unsigned long long (__hip_uint64_t) whereas the platform
// size_t is unsigned long; redefine the std index-sequence aliases on the
// platform size_t so they match ck_tile code templated on size_t.
constexpr const char* kUtility = R"cpp(#pragma once
#include <rocm/utility.hpp>
namespace std {
using namespace rocm;
template <__SIZE_TYPE__... I>
using index_sequence = integer_sequence<__SIZE_TYPE__, I...>;
template <__SIZE_TYPE__ N>
using make_index_sequence = __make_integer_seq<integer_sequence, __SIZE_TYPE__, N>;
template <class... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
}
)cpp";

// Use hipRTC's __hip_internal types so std::size_t matches the platform size_t
// (avoids the rocm::size_t = unsigned long long vs ::size_t = unsigned long
// mismatch).
constexpr const char* kCstddef = R"cpp(#pragma once
#include <rocm/stddef.hpp>
namespace std {
using namespace rocm;
using ::size_t;
using ::ptrdiff_t;
using nullptr_t = decltype(nullptr);
enum class byte : unsigned char {};
}
)cpp";

constexpr const char* kCstdint = R"cpp(#pragma once
#include <rocm/stdint.hpp>
using __hip_internal::int8_t;
using __hip_internal::int16_t;
using __hip_internal::int32_t;
using __hip_internal::int64_t;
using __hip_internal::uint8_t;
using __hip_internal::uint16_t;
using __hip_internal::uint32_t;
using __hip_internal::uint64_t;
using rocm::intptr_t;
using rocm::uintptr_t;
namespace std {
using namespace rocm;
using __hip_internal::int8_t;
using __hip_internal::int16_t;
using __hip_internal::int32_t;
using __hip_internal::int64_t;
using __hip_internal::uint8_t;
using __hip_internal::uint16_t;
using __hip_internal::uint32_t;
using __hip_internal::uint64_t;
using rocm::intptr_t;
using rocm::uintptr_t;
}
)cpp";

// <cinttypes> brings in <cstdint> and the printf/scanf format-macro set. LP64:
// 64-bit fixed-width types are 'long', so use the 'l' length modifier.
constexpr const char* kCinttypes = R"cpp(#pragma once
#include <cstdint>
#define PRId8  "d"
#define PRId16 "d"
#define PRId32 "d"
#define PRId64 "ld"
#define PRIi8  "i"
#define PRIi16 "i"
#define PRIi32 "i"
#define PRIi64 "li"
#define PRIu8  "u"
#define PRIu16 "u"
#define PRIu32 "u"
#define PRIu64 "lu"
#define PRIx8  "x"
#define PRIx16 "x"
#define PRIx32 "x"
#define PRIx64 "lx"
#define PRIX8  "X"
#define PRIX16 "X"
#define PRIX32 "X"
#define PRIX64 "lX"
#define PRIdPTR "ld"
#define PRIiPTR "li"
#define PRIuPTR "lu"
#define PRIxPTR "lx"
#define PRIXPTR "lX"
)cpp";

constexpr const char* kCassert = R"cpp(#pragma once
#ifdef assert
#undef assert
#endif
#define assert(x) ((void)0)
)cpp";

// Forward declarations so host-only types named in non-stripped (HOST_DEVICE)
// signatures resolve without instantiation.
constexpr const char* kVector = R"cpp(#pragma once
namespace std {
template <class T> class allocator;
template <class T, class Alloc = allocator<T>> class vector;
}
)cpp";

constexpr const char* kString = R"cpp(#pragma once
namespace std {
template <class CharT> struct char_traits;
template <class T> class allocator;
template <class CharT, class Traits = char_traits<CharT>, class Alloc = allocator<CharT>>
class basic_string;
using string = basic_string<char>;

// Minimal constexpr std::string_view. ck_tile uses it for compile-time pipeline
// name comparisons in `if constexpr` (e.g. kPipelineName != "qr_async_trload"),
// so the comparison operators must be usable in constant expressions.
class string_view {
    const char* data_;
    __SIZE_TYPE__ size_;
    static constexpr __SIZE_TYPE__ slen(const char* s) {
        __SIZE_TYPE__ n = 0;
        if(s) { while(s[n] != '\0') ++n; }
        return n;
    }
public:
    constexpr string_view() noexcept : data_(nullptr), size_(0) {}
    constexpr string_view(const char* s) noexcept : data_(s), size_(slen(s)) {}
    constexpr string_view(const char* s, __SIZE_TYPE__ n) noexcept : data_(s), size_(n) {}
    constexpr const char* data() const noexcept { return data_; }
    constexpr __SIZE_TYPE__ size() const noexcept { return size_; }
    constexpr __SIZE_TYPE__ length() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr char operator[](__SIZE_TYPE__ i) const { return data_[i]; }
    constexpr bool operator==(string_view o) const noexcept {
        if(size_ != o.size_) return false;
        for(__SIZE_TYPE__ i = 0; i < size_; ++i)
            if(data_[i] != o.data_[i]) return false;
        return true;
    }
    constexpr bool operator!=(string_view o) const noexcept { return !(*this == o); }
};
}
)cpp";

} // namespace

const std::unordered_map<std::string, std::string>& GetRtcStdShims()
{
    static const std::unordered_map<std::string, std::string> shims = [] {
        std::unordered_map<std::string, std::string> m;

        // Bridges onto rocm-cxx.
        m.emplace("type_traits", kTypeTraits);
        m.emplace("utility", kUtility);
        m.emplace("cstdint", kCstdint);
        m.emplace("cstddef", kCstddef);
        m.emplace("limits", bridge("limits.hpp"));
        m.emplace("array", bridge("array.hpp"));
        m.emplace("functional", bridge("functional.hpp"));
        m.emplace("iterator", bridge("iterator.hpp"));
        m.emplace("bit", bridge("bit.hpp"));

        // Minimal self-contained.
        m.emplace("initializer_list", kInitializerList);
        m.emplace("tuple", kTuple);
        m.emplace("concepts", kConcepts);
        m.emplace("cmath", kCmath);
        m.emplace("cstring", kCstring);
        m.emplace("algorithm", kAlgorithm);
        m.emplace("cassert", kCassert);
        m.emplace("cinttypes", kCinttypes);
        m.emplace("inttypes.h", kCinttypes);

        // C header <stdint.h>: expose the fixed-width integer types in the
        // global namespace via hipRTC's own __hip_internal types.
        m.emplace("stdint.h",
                  "#pragma once\n#include <rocm/stdint.hpp>\n"
                  "using __hip_internal::int8_t;\nusing __hip_internal::uint8_t;\n"
                  "using __hip_internal::int16_t;\nusing __hip_internal::uint16_t;\n"
                  "using __hip_internal::int32_t;\nusing __hip_internal::uint32_t;\n"
                  "using __hip_internal::int64_t;\nusing __hip_internal::uint64_t;\n"
                  "using rocm::intptr_t;\nusing rocm::uintptr_t;\n");

        // Forward-declaration stubs.
        m.emplace("vector", kVector);
        m.emplace("string", kString);
        // <string_view> reuses the constexpr string_view defined in <string>.
        m.emplace("string_view", "#pragma once\n#include <string>\n");

        // Empty stubs for host-only headers (bodies stripped before embedding).
        for(const char* name : {"variant",
                                "iostream",
                                "ostream",
                                "istream",
                                "sstream",
                                "stdexcept",
                                "memory",
                                "numeric",
                                "optional",
                                "random",
                                "cstdlib",
                                "cstdio",
                                "cctype",
                                "cwchar",
                                "cwctype",
                                "iomanip",
                                "fstream",
                                "thread",
                                "mutex",
                                "queue",
                                "deque",
                                "condition_variable",
                                "unordered_map",
                                "unordered_set",
                                "map",
                                "set",
                                "list",
                                "span",
                                "memory_resource",
                                // C header whose system version transitively
                                // pulls <stddef.h> (compiler resource dir, no
                                // longer on the search path without -isystem).
                                "stdio.h"})
        {
            m.emplace(name, "#pragma once\n");
        }

        return m;
    }();
    return shims;
}

} // namespace host
} // namespace ck
