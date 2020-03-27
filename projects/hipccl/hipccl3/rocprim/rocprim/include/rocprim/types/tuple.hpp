// Copyright (c) 2018-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TYPES_TUPLE_HPP_
#define ROCPRIM_TYPES_TUPLE_HPP_

#include <tuple>
#include <type_traits>

#include "../config.hpp"
#include "../detail/all_true.hpp"

#include "integer_sequence.hpp"

/// \addtogroup utilsmodule_tuple
/// @{

BEGIN_ROCPRIM_NAMESPACE

// ////////////////////////
// tuple (FORWARD DECLARATION)
// ////////////////////////
template<class... Types>
class tuple;

// ////////////////////////
// tuple_size
// ////////////////////////

/// \brief Provides access to the number of elements in a tuple as a compile-time constant expression.
///
/// <tt>tuple_size<T></tt> is undefined for types \p T that are not tuples.
template<class T>
class tuple_size;

/// \brief For \p T that is <tt>tuple</tt>, \p tuple_size<T>::value is the
/// the number of elements in a tuple (equal to <tt>sizeof...(Types)</tt>).
///
/// \see std::integral_constant
template<class... Types>
class tuple_size<::rocprim::tuple<Types...>> : public std::integral_constant<size_t, sizeof...(Types)>
{
    // All member functions of std::integral_constant are constexpr, so it should work
    // without problems on HIP
};
/// <tt>const T</tt> specialization of \ref tuple_size
template<class T>
class tuple_size<const T>
    : public std::integral_constant<size_t, tuple_size<T>::value>
{

};
/// <tt>volatile T</tt> specialization of \ref tuple_size
template<class T>
class tuple_size<volatile T>
    : public std::integral_constant<size_t, tuple_size<T>::value>
{

};
/// <tt>const volatile T</tt> specialization of \ref tuple_size
template<class T>
class tuple_size<const volatile T>
    : public std::integral_constant<size_t, tuple_size<T>::value>
{

};

// ////////////////////////
// tuple_element
// ////////////////////////

/// \brief Provides compile-time indexed access to the types of the elements of the tuple.
///
/// <tt>tuple_element<I, T></tt> is undefined for types \p T that are not tuples.
template<size_t I, class T>
struct tuple_element; // rocprim::tuple_size is defined only for rocprim::tuple

namespace detail
{

template<size_t I, class T>
struct tuple_element_impl;

template<size_t I, class T, class... Types>
struct tuple_element_impl<I, ::rocprim::tuple<T, Types...>>
    : tuple_element_impl<I-1, ::rocprim::tuple<Types...>>
{

};

template<class T, class... Types>
struct tuple_element_impl<0, ::rocprim::tuple<T, Types...>>
{
    using type = T;
};

template<size_t I>
struct tuple_element_impl<I, ::rocprim::tuple<>>
{
    static_assert(I != I, "tuple_element index out of range");
};

} // end detail namespace

/// \brief For \p T that is <tt>tuple</tt>, \p tuple_element<I, T>::type is the
/// type of <tt>I</tt>th element of that tuple.
template<size_t I, class... Types>
struct tuple_element<I, ::rocprim::tuple<Types...>>
{
    /// \brief The type of <tt>I</tt>th element of the tuple, where \p I is in <tt>[0, sizeof...(Types))</tt>
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using type = typename detail::tuple_element_impl<I, ::rocprim::tuple<Types...>>::type;
    #else
    typedef type;
    #endif
};
/// <tt>const T</tt> specialization of \ref tuple_element
template<size_t I, class T>
struct tuple_element<I, const T>
{
    /// \brief The type of <tt>I</tt>th element of the tuple, where \p I is in <tt>[0, sizeof...(Types))</tt>
    using type = typename std::add_const<typename tuple_element<I, T>::type>::type;
};
/// <tt>volatile T</tt> specialization of \ref tuple_element
template<size_t I, class T>
struct tuple_element<I, volatile T>
{
    /// \brief The type of <tt>I</tt>th element of the tuple, where \p I is in <tt>[0, sizeof...(Types))</tt>
    using type = typename std::add_volatile<typename tuple_element<I, T>::type>::type;
};
/// <tt>const volatile T</tt> specialization of \ref tuple_element
template<size_t I, class T>
struct tuple_element<I, const volatile T>
{
    /// \brief The type of <tt>I</tt>th element of the tuple, where \p I is in <tt>[0, sizeof...(Types))</tt>
    using type = typename std::add_cv<typename tuple_element<I, T>::type>::type;
};

template <size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

// get<I> forward declaration
#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<size_t I, class... UTypes>
ROCPRIM_HOST_DEVICE
const tuple_element_t<I, tuple<UTypes...>>& get(const tuple<UTypes...>&) noexcept;

template<size_t I, class... UTypes>
ROCPRIM_HOST_DEVICE
tuple_element_t<I, tuple<UTypes...>>& get(tuple<UTypes...>&) noexcept;

template<size_t I, class... UTypes>
ROCPRIM_HOST_DEVICE
tuple_element_t<I, tuple<UTypes...>>&& get(tuple<UTypes...>&&) noexcept;
#endif

// ////////////////////////
// tuple
// ////////////////////////

namespace detail
{

#ifdef __cpp_lib_is_final
    template<class T>
    using is_final = std::is_final<T>;
#elif defined(__HCC__) // use clang extention
    template<class T>
    using is_final = std::integral_constant<bool, __is_final(T)>;
#else
    template<class T>
    struct is_final : std::false_type
    {
    };
#endif

// tuple_value - represents single element in a tuple
template<
    size_t I,
    class T,
    bool /* Empty base optimization switch */ = std::is_empty<T>::value && !is_final<T>::value
>
struct tuple_value
{
    T value;

    ROCPRIM_HOST_DEVICE inline
    constexpr tuple_value() noexcept : value()
    {
        static_assert(!std::is_reference<T>::value, "can't default construct a reference element in a tuple" );
    }

    ROCPRIM_HOST_DEVICE inline
    tuple_value(const tuple_value&) = default;

    ROCPRIM_HOST_DEVICE inline
    tuple_value(tuple_value&&) = default;

    ROCPRIM_HOST_DEVICE inline
    explicit tuple_value(T value) noexcept
        : value(value)
    {
        // This is workaround for hcc which fails during linking without
        // this constructor with undefine reference errors when U from ctors
        // below is exactly T. Example:
        // rocprim::tuple<int, int, int> t(1, 2, 3);
        // Produced error:
        // undefined reference to `rocprim::detail::tuple_value<0ul, int>::tuple_value(int)
    }

    template<
        class U,
        typename = typename std::enable_if<
            !std::is_same<typename std::decay<U>::type, tuple_value>::value
        >::type,
        typename = typename std::enable_if<
            std::is_constructible<T, const U&>::value
        >::type
    >
    ROCPRIM_HOST_DEVICE inline
    explicit tuple_value(const U& v) noexcept : value(v)
    {
    }

    template<
        class U,
        typename = typename std::enable_if<
            // So U can't be tuple_value<T>
            !std::is_same<typename std::decay<U>::type, tuple_value>::value
        >::type,
        typename = typename std::enable_if<
            std::is_constructible<T, U&&>::value
        >::type
    >
    ROCPRIM_HOST_DEVICE inline
    explicit tuple_value(U&& v) noexcept : value(std::forward<U>(v))
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~tuple_value() = default;

    template<class U>
    ROCPRIM_HOST_DEVICE inline
    tuple_value& operator=(U&& v) noexcept
    {
        value = std::forward<U>(v);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    void swap(tuple_value& v) noexcept
    {
        auto tmp = std::move(v.value);
        v.value = std::move(this->value);
        this->value = std::move(tmp);
    }

    ROCPRIM_HOST_DEVICE inline
    T& get() noexcept
    {
        return value;
    }

    ROCPRIM_HOST_DEVICE inline
    const T& get() const noexcept
    {
        return value;
    }
};

// Specialization for empty base optimization
template<size_t I, class T>
struct tuple_value<I, T, true> : private T
{
    ROCPRIM_HOST_DEVICE inline
    constexpr tuple_value() noexcept : T()
    {
        static_assert(!std::is_reference<T>::value, "can't default construct a reference element in a tuple" );
    }

    ROCPRIM_HOST_DEVICE inline
    tuple_value(const tuple_value&) = default;

    ROCPRIM_HOST_DEVICE inline
    tuple_value(tuple_value&&) = default;

    ROCPRIM_HOST_DEVICE inline
    explicit tuple_value(T value) noexcept
        : T(value)
    {
        // This is workaround for hcc which fails during linking without
        // this constructor with undefine reference errors when U from ctors
        // below is exactly T. Example:
        // rocprim::tuple<int, int, int> t(1, 2, 3);
        // Produced error:
        // undefined reference to `rocprim::detail::tuple_value<0ul, int>::tuple_value(int)
    }

    template<
        class U,
        typename = typename std::enable_if<
            !std::is_same<typename std::decay<U>::type, tuple_value>::value
        >::type,
        typename = typename std::enable_if<
            std::is_constructible<T, const U&>::value
        >::type
    >
    ROCPRIM_HOST_DEVICE inline
    explicit tuple_value(const U& v) noexcept : T(v)
    {
    }

    template<
        class U,
        typename = typename std::enable_if<
            // So U can't be tuple_value<T>
            !std::is_same<typename std::decay<U>::type, tuple_value>::value
        >::type,
        typename = typename std::enable_if<
            std::is_constructible<T, U&&>::value
        >::type
    >
    ROCPRIM_HOST_DEVICE inline
    explicit tuple_value(U&& v) noexcept : T(std::forward<U>(v))
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~tuple_value() = default;

    template<class U>
    ROCPRIM_HOST_DEVICE inline
    tuple_value& operator=(U&& v) noexcept
    {
        T::operator=(std::forward<U>(v));
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    void swap(tuple_value& v) noexcept
    {
        auto tmp = std::move(v);
        v = std::move(*this);
        *this = std::move(tmp);
    }

    ROCPRIM_HOST_DEVICE inline
    T& get() noexcept
    {
        return static_cast<T&>(*this);
    }

    ROCPRIM_HOST_DEVICE inline
    const T& get() const noexcept
    {
        return static_cast<const T&>(*this);
    }
};

template <class... Types>
ROCPRIM_HOST_DEVICE inline
void swallow(Types&&...) noexcept {}

template<class Sequences, class... Types>
struct tuple_impl;

template<size_t... Indices, class... Types>
struct tuple_impl<::rocprim::index_sequence<Indices...>, Types...>
    : tuple_value<Indices, Types>...
{
    ROCPRIM_HOST_DEVICE inline
    constexpr tuple_impl() = default;

    ROCPRIM_HOST_DEVICE inline
    tuple_impl(const tuple_impl&) = default;

    ROCPRIM_HOST_DEVICE inline
    tuple_impl(tuple_impl&&) = default;

    ROCPRIM_HOST_DEVICE inline
    explicit tuple_impl(Types... values)
        : tuple_value<Indices, Types>(values)...
    {
        // This is workaround for hcc which fails during linking without
        // this constructor with undefine reference errors when UTypes
        // are exactly Types (see constructor below). Example:
        // rocprim::tuple<int, int, int> t(1, 2, 3);
        // Produced error:
        // undefined reference to `rocprim::detail::tuple_impl<
        //   rocprim::integer_sequence<unsigned long, 0ul, 1ul, 2ul>, int, int, int
        // >::tuple_impl(int, int, int)'
    }

    template<
        class... UTypes,
        typename = typename std::enable_if<
            sizeof...(UTypes) == sizeof...(Types)
        >::type,
        typename = typename std::enable_if<
            sizeof...(Types) >= 1
        >::type
    >
    ROCPRIM_HOST_DEVICE inline
    explicit tuple_impl(UTypes&&... values)
        : tuple_value<Indices, Types>(std::forward<UTypes>(values))...
    {
    }

    template<
        class... UTypes,
        typename = typename std::enable_if<
            sizeof...(UTypes) == sizeof...(Types)
        >::type,
        typename = typename std::enable_if<
            sizeof...(Types) >= 1
        >::type
    >
    ROCPRIM_HOST_DEVICE inline
    tuple_impl(::rocprim::tuple<UTypes...>&& other)
        : tuple_value<Indices, Types>(std::forward<UTypes>(::rocprim::get<Indices>(other)))...
    {
    }

    template<
        class... UTypes,
        typename = typename std::enable_if<
            sizeof...(UTypes) == sizeof...(Types)
        >::type,
        typename = typename std::enable_if<
            sizeof...(Types) >= 1
        >::type
    >
    ROCPRIM_HOST_DEVICE inline
    tuple_impl(const ::rocprim::tuple<UTypes...>& other)
        : tuple_value<Indices, Types>(::rocprim::get<Indices>(other))...
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~tuple_impl() = default;

    ROCPRIM_HOST_DEVICE inline
    tuple_impl& operator=(const tuple_impl& other) noexcept
    {
        swallow(
            tuple_value<Indices, Types>::operator=(
                static_cast<const tuple_value<Indices, Types>&>(other).get()
            )...
        );
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    tuple_impl& operator=(tuple_impl&& other) noexcept
    {
        swallow(
            tuple_value<Indices, Types>::operator=(
                static_cast<tuple_value<Indices, Types>&>(other).get()
            )...
        );
        return *this;
    }

    template<class... UTypes>
    ROCPRIM_HOST_DEVICE inline
    tuple_impl& operator=(const ::rocprim::tuple<UTypes...>& other) noexcept
    {
        swallow(tuple_value<Indices, Types>::operator=(::rocprim::get<Indices>(other))...);
        return *this;
    }

    template<class... UTypes>
    ROCPRIM_HOST_DEVICE inline
    tuple_impl& operator=(::rocprim::tuple<UTypes...>&& other) noexcept
    {
        swallow(
            tuple_value<Indices, Types>::operator=(
                ::rocprim::get<Indices>(std::move(other))
            )...
        );
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    tuple_impl& swap(tuple_impl& other) noexcept
    {
        swallow(
            (static_cast<tuple_value<Indices, Types>&>(*this).swap(
                static_cast<tuple_value<Indices, Types>&>(other)
            ), 0)...
        );
        return *this;
    }
};

template<class... Types>
using tuple_base =
    tuple_impl<
        typename ::rocprim::index_sequence_for<Types...>,
        Types...
    >;

} // end detail namespace

/// \brief Fixed-size collection of heterogeneous values.
///
/// \tparam Types... the types (zero or more) of the elements that the tuple stores.
///
/// \pre
/// * For all types in \p Types... following operations should not throw exceptions:
/// construction, copy and move assignment, and swapping.
///
/// \see std::tuple
template<class... Types>
class tuple
{
    using base_type = detail::tuple_base<Types...>;
    // tuple_impl
    base_type base;

    template<class Dummy>
    struct check_constructor
    {
        template<class... Args>
        static constexpr bool enable_default()
        {
            return detail::all_true<std::is_default_constructible<Args>::value...>::value;
        }

        template<class... Args>
        static constexpr bool enable_copy()
        {
            return detail::all_true<std::is_copy_constructible<Args>::value...>::value;
        }
    };

    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<size_t I, class... UTypes>
    ROCPRIM_HOST_DEVICE
    friend const tuple_element_t<I, tuple<UTypes...>>& get(const tuple<UTypes...>&) noexcept;

    template<size_t I, class... UTypes>
    ROCPRIM_HOST_DEVICE
    friend tuple_element_t<I, tuple<UTypes...>>& get(tuple<UTypes...>&) noexcept;

    template<size_t I, class... UTypes>
    ROCPRIM_HOST_DEVICE
    friend tuple_element_t<I, tuple<UTypes...>>&& get(tuple<UTypes...>&&) noexcept;
    #endif

public:
    /// \brief Default constructor. Performs value-initialization of all elements.
    ///
    /// This overload only participates in overload resolution if:
    /// * <tt>std::is_default_constructible<Ti>::value</tt> is \p true for all \p i.
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<
        class Dummy = void,
        typename = typename std::enable_if<
            check_constructor<Dummy>::template enable_default<Types...>()
        >::type
    >
    #endif
    ROCPRIM_HOST_DEVICE inline
    constexpr tuple() noexcept : base() {};

    /// \brief Implicitly-defined copy constructor.
    ROCPRIM_HOST_DEVICE inline
    tuple(const tuple&) = default;

    /// \brief Implicitly-defined move constructor.
    ROCPRIM_HOST_DEVICE inline
    tuple(tuple&&) = default;

    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    ROCPRIM_HOST_DEVICE inline
    explicit tuple(Types... values) noexcept
        : base(values...)
    {
        // Workaround for HCC compiler, without this we get undefined reference
        // errors during linking. Example:
        // rocprim::tuple<int, double> t1(1, 2)
        // Produces error:
        // 'undefined reference to `rocprim::tuple<int, double>::tuple(int, double)'
    }
    #endif

    /// \brief Direct constructor. Initializes each element of the tuple with
    /// the corresponding input value.
    ///
    /// This overload only participates in overload resolution if:
    /// * <tt>std::is_copy_constructible<Ti>::value</tt> is \p true for all \p i.
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<
        class Dummy = void,
        typename = typename std::enable_if<
            check_constructor<Dummy>::template enable_copy<Types...>()
        >::type
    >
    #endif
    ROCPRIM_HOST_DEVICE inline
    explicit tuple(const Types&... values)
        : base(values...)
    {
    }

    /// \brief Converting constructor. Initializes each element of the tuple
    /// with the corresponding value in \p std::forward<UTypes>(values).
    ///
    /// This overload only participates in overload resolution if:
    /// * <tt>sizeof...(Types) == sizeof...(UTypes)</tt>,
    /// * <tt>sizeof...(Types) >= 1</tt>, and
    /// * <tt>std::is_constructible<Ti, Ui&&>::value</tt> is \p true for all \p i.
    template<
        class... UTypes
        #ifndef DOXYGEN_SHOULD_SKIP_THIS
        ,typename = typename std::enable_if<
            sizeof...(UTypes) == sizeof...(Types)
        >::type,
        typename = typename std::enable_if<
            sizeof...(Types) >= 1
        >::type,
        typename = typename std::enable_if<
            detail::all_true<std::is_constructible<Types, UTypes&&>::value...>::value
        >::type
        #endif
    >
    ROCPRIM_HOST_DEVICE inline
    explicit tuple(UTypes&&... values) noexcept
        : base(std::forward<UTypes>(values)...)
    {
    }

    /// \brief Converting copy constructor. Initializes each element of the tuple
    /// with the corresponding value from \p other.
    ///
    /// This overload only participates in overload resolution if:
    /// * <tt>sizeof...(Types) == sizeof...(UTypes)</tt>,
    /// * <tt>sizeof...(Types) >= 1</tt>, and
    /// * <tt>std::is_constructible<Ti, Ui&&>::value</tt> is \p true for all \p i.
    template<
        class... UTypes,
        #ifndef DOXYGEN_SHOULD_SKIP_THIS
        typename = typename std::enable_if<
            sizeof...(UTypes) == sizeof...(Types)
        >::type,
        typename = typename std::enable_if<
            sizeof...(Types) >= 1
        >::type,
        typename = typename std::enable_if<
            detail::all_true<std::is_constructible<Types, const UTypes&>::value...>::value
        >::type
        #endif
    >
    ROCPRIM_HOST_DEVICE inline
    tuple(const tuple<UTypes...>& other) noexcept
        : base(other)
    {
    }

    /// \brief Converting move constructor. Initializes each element of the tuple
    /// with the corresponding value from \p other.
    ///
    /// This overload only participates in overload resolution if:
    /// * <tt>sizeof...(Types) == sizeof...(UTypes)</tt>,
    /// * <tt>sizeof...(Types) >= 1</tt>, and
    /// * <tt>std::is_constructible<Ti, Ui&&>::value</tt> is \p true for all \p i.
    template<
        class... UTypes,
        #ifndef DOXYGEN_SHOULD_SKIP_THIS
        typename = typename std::enable_if<
            sizeof...(UTypes) == sizeof...(Types)
        >::type,
        typename = typename std::enable_if<
            sizeof...(Types) >= 1
        >::type,
        typename = typename std::enable_if<
            detail::all_true<std::is_constructible<Types, UTypes&&>::value...>::value
        >::type
        #endif
    >
    ROCPRIM_HOST_DEVICE inline
    tuple(tuple<UTypes...>&& other) noexcept
        : base(std::forward<tuple<UTypes...>>(other))
    {
    }

    /// \brief Implicitly-defined destructor.
    ROCPRIM_HOST_DEVICE inline
    ~tuple() noexcept = default;

    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<
        class T,
        typename = typename std::enable_if<
            std::is_assignable<base_type&, T>::value
        >::type
    >
    ROCPRIM_HOST_DEVICE inline
    tuple& operator=(T&& v) noexcept
    {
        base = std::forward<T>(v);
        return *this;
    }

    tuple& operator=(const tuple& other) noexcept
    {
        base = other.base;
        return *this;
    }
    #else // For documentation
    /// \brief Copy assignment operator.
    /// \param other tuple to replace the contents of this tuple
    tuple& operator=(const tuple& other) noexcept;
    /// \brief Move assignment operator.
    /// \param other tuple to replace the contents of this tuple
    tuple& operator=(tuple&& other) noexcept;
    /// \brief For all \p i, assigns \p rocprim::get<i>(other) to \p rocprim::get<i>(*this).
    /// \param other tuple to replace the contents of this tuple
    template<class... UTypes>
    tuple& operator=(const tuple<UTypes...>& other) noexcept;
    /// \brief For all \p i, assigns \p std::forward<Ui>(get<i>(other)) to \p rocprim::get<i>(*this).
    /// \param other tuple to replace the contents of this tuple
    template<class... UTypes>
    tuple& operator=(tuple<UTypes...>&& other) noexcept;
    #endif

    /// \brief Swaps the content of the tuple (\p *this) with the content \p other
    /// \param other tuple of values to swap
    void swap(tuple& other) noexcept
    {
        base.swap(other.base);
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<>
class tuple<>
{
public:
    ROCPRIM_HOST_DEVICE inline
    constexpr tuple() noexcept
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~tuple() = default;

    ROCPRIM_HOST_DEVICE inline
    void swap(tuple&) noexcept
    {
    }
};
#endif

namespace detail
{

template<size_t I>
struct tuple_equal_to
{
    template<class T, class U>
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const T& lhs, const U& rhs) const
    {
        return tuple_equal_to<I-1>()(lhs, rhs) && get<I-1>(lhs) == get<I-1>(rhs);
    }
};

template<>
struct tuple_equal_to<0>
{
    template<class T, class U>
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const T&, const U&) const
    {
        return true;
    }
};

template<size_t I>
struct tuple_less_than
{
    template<class T, class U>
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const T& lhs, const U& rhs) const
    {
        constexpr size_t idx = tuple_size<T>::value - I;
        if(get<idx>(lhs) < get<idx>(rhs))
            return true;
        if(get<idx>(rhs) < get<idx>(lhs))
            return false;
        return tuple_less_than<I-1>()(lhs, rhs);
    }
};

template<>
struct tuple_less_than<0>
{
    template<class T, class U>
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const T&, const U&) const
    {
        return false;
    }
};

} // end namespace detail

/// \brief Equal to operator for tuples.
///
/// \tparam TTypes... - the element types of \p lhs tuple.
/// \tparam UTypes... - the element types of \p rhs tuple.
///
/// Compares every element of the tuple lhs with the corresponding element
/// of the tuple rhs, and returns \p true if all are equal.
///
/// \param lhs tuple to compare with \p rhs
/// \param rhs tuple to compare with \p lhs
/// \return \p true if <tt>rocprim::get<i>(lhs) == rocprim::get<i>(rhs)</tt> for all
/// \p i in <tt>[0, sizeof...(TTypes))</tt>; otherwise - \p false. Comparing two
/// empty tuples returns \p true.
template<
    class... TTypes,
    class... UTypes,
    typename = typename std::enable_if<
        sizeof...(TTypes) == sizeof...(UTypes)
    >::type
>
ROCPRIM_HOST_DEVICE inline
bool operator==(const tuple<TTypes...>& lhs, const tuple<UTypes...>& rhs)
{
    return detail::tuple_equal_to<sizeof...(TTypes)>()(lhs, rhs);
}

/// \brief Not equal to operator for tuples.
///
/// \tparam TTypes... - the element types of \p lhs tuple.
/// \tparam UTypes... - the element types of \p rhs tuple.
///
/// Compares every element of the tuple lhs with the corresponding element
/// of the tuple rhs, and returns \p true if at least one of such pairs is
/// not equal.
///
/// \param lhs tuple to compare with \p rhs
/// \param rhs tuple to compare with \p lhs
/// \return <tt>!(lhr == rhs)</tt>
template<class... TTypes, class... UTypes>
ROCPRIM_HOST_DEVICE inline
bool operator!=(const tuple<TTypes...>& lhs, const tuple<UTypes...>& rhs)
{
    return !(lhs == rhs);
}

/// \brief Less than operator for tuples.
///
/// \tparam TTypes... - the element types of \p lhs tuple.
/// \tparam UTypes... - the element types of \p rhs tuple.
///
/// Compares lhs and rhs lexicographically.
///
/// \param lhs tuple to compare with \p rhs
/// \param rhs tuple to compare with \p lhs
/// \return <tt>(bool)(rocprim::get<0>(lhs) < rocprim::get<0>(rhs)) ||
/// (!(bool)(rocprim::get<0>(rhs) < rocprim::get<0>(lhs)) && lhstail < rhstail)</tt>, where
/// \p lhstail is \p lhs without its first element, and \p rhstail is \p rhs without its first
/// element. For two empty tuples, it returns \p false.
template<
    class... TTypes,
    class... UTypes,
    typename = typename std::enable_if<
        sizeof...(TTypes) == sizeof...(UTypes)
    >::type
>
ROCPRIM_HOST_DEVICE inline
bool operator<(const tuple<TTypes...>& lhs, const tuple<UTypes...>& rhs)
{
    return detail::tuple_less_than<sizeof...(TTypes)>()(lhs, rhs);
}

/// \brief Greater than operator for tuples.
///
/// \tparam TTypes... - the element types of \p lhs tuple.
/// \tparam UTypes... - the element types of \p rhs tuple.
///
/// Compares lhs and rhs lexicographically.
///
/// \param lhs tuple to compare with \p rhs
/// \param rhs tuple to compare with \p lhs
/// \return <tt>rhs < lhs</tt>
template<class... TTypes, class... UTypes>
ROCPRIM_HOST_DEVICE inline
bool operator>(const tuple<TTypes...>& lhs, const tuple<UTypes...>& rhs)
{
    return rhs < lhs;
}

/// \brief Less than or equal to operator for tuples.
///
/// \tparam TTypes... - the element types of \p lhs tuple.
/// \tparam UTypes... - the element types of \p rhs tuple.
///
/// Compares lhs and rhs lexicographically.
///
/// \param lhs tuple to compare with \p rhs
/// \param rhs tuple to compare with \p lhs
/// \return <tt>!(rhs < lhs)</tt>
template<class... TTypes, class... UTypes>
ROCPRIM_HOST_DEVICE inline
bool operator<=(const tuple<TTypes...>& lhs, const tuple<UTypes...>& rhs)
{
    return !(rhs < lhs);
}

/// \brief Greater than or equal to operator for tuples.
///
/// \tparam TTypes... - the element types of \p lhs tuple.
/// \tparam UTypes... - the element types of \p rhs tuple.
///
/// Compares lhs and rhs lexicographically.
///
/// \param lhs tuple to compare with \p rhs
/// \param rhs tuple to compare with \p lhs
/// \return <tt>!(lhs < rhs)</tt>
template<class... TTypes, class... UTypes>
ROCPRIM_HOST_DEVICE inline
bool operator>=(const tuple<TTypes...>& lhs, const tuple<UTypes...>& rhs)
{
    return !(lhs < rhs);
}

// ////////////////////////
// swap
// ////////////////////////

/// \brief Swaps the content of \p lhs tuple with the content \p rhs
/// \param lhs,rhs tuples whose contents to swap
template<class... Types>
ROCPRIM_HOST_DEVICE inline
void swap(tuple<Types...>& lhs, tuple<Types...>& rhs) noexcept
{
    lhs.swap(rhs);
}

// ////////////////////////
// get<Index>
// ////////////////////////

/// \brief Extracts the <tt>I</tt>-th element from the tuple, where \p I is
/// an integer value from range <tt>[0, sizeof...(Types))</tt>.
/// \param t tuple whose contents to extract
/// \return constant refernce to the selected element of input tuple \p t.
template<size_t I, class... Types>
ROCPRIM_HOST_DEVICE inline
const tuple_element_t<I, tuple<Types...>>& get(const tuple<Types...>& t) noexcept
{
    using type = detail::tuple_value<I, tuple_element_t<I, tuple<Types...>>>;
    return static_cast<const type&>(t.base).get();
}

/// \brief Extracts the <tt>I</tt>-th element from the tuple, where \p I is
/// an integer value from range <tt>[0, sizeof...(Types))</tt>.
/// \param t tuple whose contents to extract
/// \return refernce to the selected element of input tuple \p t.
template<size_t I, class... Types>
ROCPRIM_HOST_DEVICE inline
tuple_element_t<I, tuple<Types...>>& get(tuple<Types...>& t) noexcept
{
    using type = detail::tuple_value<I, tuple_element_t<I, tuple<Types...>>>;
    return static_cast<type&>(t.base).get();
}

/// \brief Extracts the <tt>I</tt>-th element from the tuple, where \p I is
/// an integer value from range <tt>[0, sizeof...(Types))</tt>.
/// \param t tuple whose contents to extract
/// \return rvalue refernce to the selected element of input tuple \p t.
template<size_t I, class... Types>
ROCPRIM_HOST_DEVICE inline
tuple_element_t<I, tuple<Types...>>&& get(tuple<Types...>&& t) noexcept
{
    using value_type = tuple_element_t<I, tuple<Types...>>;
    using type = detail::tuple_value<I, tuple_element_t<I, tuple<Types...>>>;
    return static_cast<value_type&&>(static_cast<type&>(t.base).get());
}

// ////////////////////////
// make_tuple
// ////////////////////////

namespace detail
{

template<class T>
struct make_tuple_return
{
    using type = T;
};

template<class T>
struct make_tuple_return<std::reference_wrapper<T>>
{
    using type = T&;
};

template <class T>
using make_tuple_return_t = typename make_tuple_return<typename std::decay<T>::type>::type;

} // end detail namespace

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class... Types>
ROCPRIM_HOST_DEVICE inline
tuple<detail::make_tuple_return_t<Types>...> make_tuple(Types&&... args) noexcept
{
    return tuple<detail::make_tuple_return_t<Types>...>(std::forward<Types>(args)...);
}
#else
/// \brief Creates a tuple, returned tuple type is deduced from the types of arguments.
///
/// Returned tuple type \p tuple<VTypes...> is deduced like this: For each \p Ti in
/// \p Types..., the corresponding type \p Vi in \p VTypes... is \p std::decay<Ti>::type
/// unless \p std::decay<Ti>::type results in \p std::reference_wrapper<U> for some type U,
/// in which case the deduced type is U&.
///
/// \param args - zero or more arguments to create tuple from
///
/// \see std::tuple
template<class... Types>
tuple<VTypes...> make_tuple(Types&&... args);
#endif

// ////////////////////////
// ignore
// ////////////////////////

namespace detail
{

struct ignore_t
{
    ROCPRIM_HOST_DEVICE inline
    ignore_t() = default;

    ROCPRIM_HOST_DEVICE inline
    ~ignore_t() = default;

    template<class T>
    ROCPRIM_HOST_DEVICE inline
    const ignore_t& operator=(const T&) const
    {
        return *this;
    }
};

}
#ifndef DOXYGEN_SHOULD_SKIP_THIS
using ignore_type = detail::ignore_t;
#else
struct ignore_type;
#endif
/// \brief Assigning value to ignore object has no effect.
///
/// Intended for use with \ref rocprim::tie when unpacking a \ref tuple,
/// as a placeholder for the arguments that are not used.
///
/// \see std::ignore
const ignore_type ignore;

// ////////////////////////
// tie
// ////////////////////////

/// \brief Creates a tuple of lvalue references to its arguments \p args or instances
/// of \ref rocprim::ignore.
///
/// \param args - zero or more input lvalue references used to create tuple
///
/// \see std::tie
template<class... Types>
ROCPRIM_HOST_DEVICE inline
tuple<Types&...> tie(Types&... args) noexcept
{
    return ::rocprim::tuple<Types&...>(args...);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule_tuple

#endif // ROCPRIM_TYPES_TUPLE_HPP_
