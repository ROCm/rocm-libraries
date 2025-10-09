#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/type_traits.h>
#include <thrust/device_free.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/for_each.h>

#include <compare>
#include <functional>
#include <type_traits>
#include <utility>

THRUST_NAMESPACE_BEGIN

template <class T, class = void>
struct default_delete;

template <class T>
struct default_delete<T, std::enable_if_t<!std::is_array_v<T>>>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete() noexcept = default;

  template <class U>
  THRUST_HOST default_delete(
    const default_delete<U>&,
    std::enable_if_t<std::is_convertible_v<thrust::device_ptr<U>, pointer>>* = nullptr) noexcept
  {}

  THRUST_HOST void operator()(pointer ptr) const noexcept
  {
    if (ptr.get() == nullptr)
    {
      return;
    }

    // The ideal implementation would be a simple call to `thrust::device_delete`:
    //
    //   thrust::device_delete(ptr);
    //
    // However, for non-trivially destructible types, `thrust::device_delete`
    // calls `thrust::destroy_range`, which requires an allocator with a
    // `value_type` typedef. The internal `thrust::detail::device_delete_allocator`
    // is an empty struct that lacks this typedef, causing a compilation error.
    //
    // As a workaround, we manually invoke the destructor on the device using
    // `thrust::for_each_n` and then separately free the memory.
    if constexpr (!std::is_trivially_destructible_v<T>)
    {
      thrust::for_each_n(ptr, 1, [] __device__(T & x) {
        x.~T();
      });
    }
    thrust::device_free(ptr);
  }
};

template <class T>
struct default_delete<T[], std::enable_if_t<!std::is_trivially_destructible_v<T>>>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete(size_t n = 0) noexcept
      : m_size(n){};

  template <class U>
  THRUST_HOST
  default_delete(const default_delete<U[]>& other,
                 std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>>* = nullptr) noexcept
      : m_size(other.size())
  {}

  THRUST_HOST void operator()(pointer ptr) const noexcept
  {
    if (ptr.get() == nullptr)
    {
      return;
    }

    // The ideal implementation would be a call to `thrust::device_delete`
    // with the number of elements:
    //
    //   thrust::device_delete(ptr, m_size);
    //
    // However, for non-trivially destructible types, `thrust::device_delete`
    // calls `thrust::destroy_range`, which requires an allocator with a
    // `value_type` typedef. The internal `thrust::detail::device_delete_allocator`
    // is an empty struct that lacks this typedef, causing a compilation error.
    //
    // As a workaround, we manually invoke the destructor on each element
    // using `thrust::for_each_n` and then separately free the memory.
    if (m_size)
    {
      thrust::for_each_n(ptr, m_size, [] __device__(T & x) {
        x.~T();
      });
    }
    thrust::device_free(ptr);
  }

  THRUST_HOST size_t size() const
  {
    return m_size;
  }

private:
  size_t m_size;
};

// For arrays of trivially destructible element types we intentionally do NOT
// store the element count. Their destruction is a no-op, so only the raw
// deallocation is required. Omitting the size keeps this deleter
// specialization an empty (zero-size) type, allowing unique_ptr<T[]>
// instantiations that use it to remain a zero-cost abstraction (the deleter
// need not increase the overall object size).
template <class T>
struct default_delete<T[], std::enable_if_t<std::is_trivially_destructible_v<T>>>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete(size_t = 0) noexcept {};

  template <class U>
  THRUST_HOST default_delete(
    const default_delete<U>&,
    std::enable_if_t<std::is_convertible_v<thrust::device_ptr<U>, pointer>>* = nullptr) noexcept
  {}

  THRUST_HOST void operator()(pointer ptr) const noexcept
  {
    thrust::device_free(ptr);
  }
};

namespace detail
{

template <class T, class D, class = void>
struct pointer_detector
{
  using type = thrust::device_ptr<T>;
};

template <class T, class D>
struct pointer_detector<T, D, std::void_t<typename D::pointer>>
{
  using type = typename D::pointer;
};

template <class Deleter>
struct unique_ptr_deleter_sfinae
{
  static_assert(!std::is_reference_v<Deleter>, "incorrect specialization");
  using lval_ref_type        = const Deleter&;
  using good_rval_ref_type   = Deleter&&;
  using bad_rval_ref_type    = void;
  using enable_rval_overload = thrust::detail::true_type;
};

template <class Deleter>
struct unique_ptr_deleter_sfinae<const Deleter&>
{
  using lval_ref_type        = const Deleter&;
  using good_rval_ref_type   = void;
  using bad_rval_ref_type    = const Deleter&&;
  using enable_rval_overload = thrust::detail::false_type;
};

template <class Deleter>
struct unique_ptr_deleter_sfinae<Deleter&>
{
  using lval_ref_type        = Deleter&;
  using good_rval_ref_type   = void;
  using bad_rval_ref_type    = Deleter&&;
  using enable_rval_overload = thrust::detail::false_type;
};

} // namespace detail

template <class T, class D = default_delete<T>>
class __attribute__((trivial_abi)) unique_ptr
{
public:
  using pointer      = typename thrust::detail::pointer_detector<T, D>::type;
  using element_type = T;
  using deleter_type = D;

  // TODO: When a standard “trivially relocatable” facility lands, add an
  // annotation/macro here to advertise that unique_ptr can be bitwise relocated

private:
  pointer m_ptr;
  [[no_unique_address]] deleter_type m_deleter;

  using DeleterSFINAE = thrust::detail::unique_ptr_deleter_sfinae<D>;

  // Next section implements SFINAE constraints for the unique_ptr constructors
  // using a pattern that mirrors the implementation in libc++.
  //
  // The `dependent_type` helper makes the type aliases below (e.g., LValRefType)
  // dependent on a dummy template parameter from the constructor itself. This
  // forces the compiler to defer constraint checking until function overload
  // resolution, rather than at class instantiation time.
  //
  // NOTE: A simpler SFINAE pattern using a `static constexpr bool` evaluated at
  // class-instantiation time also works correctly. However, we intentionally
  // follow the more complex libc++ pattern for consistency with a proven
  // implementation, aiming to inherit its robustness against
  // potential compiler-specific edge cases.

  template <bool Dummy>
  using LValRefType = typename thrust::detail::dependent_type<DeleterSFINAE, Dummy>::type::lval_ref_type;

  template <bool Dummy>
  using GoodRValRefType = typename thrust::detail::dependent_type<DeleterSFINAE, Dummy>::type::good_rval_ref_type;

  template <bool Dummy>
  using BadRValRefType = typename thrust::detail::dependent_type<DeleterSFINAE, Dummy>::type::bad_rval_ref_type;

  template <
    bool Dummy,
    class Deleter =
      typename thrust::detail::dependent_type<typename thrust::detail::identity_<deleter_type>::type, Dummy>::type>
  using EnableIfDeleterDefaultConstructible =
    std::enable_if_t<std::is_default_constructible_v<Deleter> && !std::is_pointer_v<Deleter>>;

  template <class ArgType>
  using EnableIfDeleterConstructible =
    std::enable_if_t<std::is_constructible_v<deleter_type, ArgType>>;

  template <class U, class E>
  using EnableIfMoveConvertible =
    std::enable_if_t<std::is_convertible_v<typename U::pointer, pointer> && !std::is_array_v<E>>;

  template <class E>
  using EnableIfDeleterConvertible =
    std::enable_if_t<(std::is_reference_v<D> && std::is_same_v<D, E>)
                            || (!std::is_reference_v<D> && std::is_convertible_v<E, D>)>;

  template <class E>
  using EnableIfDeleterAssignable = std::enable_if_t<std::is_assignable_v<D&, E&&>>;

  template <
    bool Dummy,
    class Deleter =
      typename thrust::detail::dependent_type<typename thrust::detail::identity_<deleter_type>::type, Dummy>::type>
  using EnableIfDeleterDefaultDelete = std::enable_if_t<std::is_same_v<Deleter, default_delete<T>>>;

public:
  //==========================================================================
  // Constructors
  //==========================================================================

  template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
  THRUST_HOST constexpr unique_ptr() noexcept
      : m_ptr()
      , m_deleter()
  {}

  template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
  THRUST_HOST constexpr unique_ptr(std::nullptr_t) noexcept
      : unique_ptr()
  {}

  template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(pointer p) noexcept
      : m_ptr(p)
      , m_deleter()
  {}

  template <bool Dummy = true, class = EnableIfDeleterConstructible<LValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(pointer p, LValRefType<Dummy> d) noexcept
      : m_ptr(p)
      , m_deleter(d)
  {}

  template <bool Dummy = true, class = EnableIfDeleterConstructible<GoodRValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(pointer p, GoodRValRefType<Dummy> d) noexcept
      : m_ptr(p)
      , m_deleter(std::move(d))
  {
    static_assert(!std::is_reference_v<deleter_type>, "rvalue deleter bound to reference");
  }

  template <bool Dummy = true, class = EnableIfDeleterConstructible<BadRValRefType<Dummy>>>
  unique_ptr(pointer p, BadRValRefType<Dummy> d) = delete;

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr&& u) noexcept
      : m_ptr(u.release())
      , m_deleter(std::forward<deleter_type>(u.get_deleter()))
  {}

  template <class U, class E, class = EnableIfMoveConvertible<unique_ptr<U, E>, U>, class = EnableIfDeleterConvertible<E>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr<U, E>&& u) noexcept
      : m_ptr(u.release())
      , m_deleter(std::forward<E>(u.get_deleter()))
  {}

  //==========================================================================
  // Assignment
  //==========================================================================
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(unique_ptr&& u) noexcept
  {
    reset(u.release());
    m_deleter = std::forward<deleter_type>(u.get_deleter());
    return *this;
  }

  template <class U, class E, class = EnableIfMoveConvertible<unique_ptr<U, E>, U>, class = EnableIfDeleterAssignable<E>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(unique_ptr<U, E>&& u) noexcept
  {
    reset(u.release());
    m_deleter = std::forward<E>(u.get_deleter());
    return *this;
  }

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(std::nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  //==========================================================================
  // Destructor
  //==========================================================================
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 ~unique_ptr()
  {
    reset();
  }

  //==========================================================================
  // Observers
  //==========================================================================
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 pointer get() const noexcept
  {
    return m_ptr;
  }

  template <bool Dummy = true, class = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 T* get_raw() const noexcept
  {
    return thrust::raw_pointer_cast(m_ptr);
  }

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 deleter_type& get_deleter() noexcept
  {
    return m_deleter;
  }

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 const deleter_type& get_deleter() const noexcept
  {
    return m_deleter;
  }

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit operator bool() const noexcept
  {
    return m_ptr != nullptr;
  }

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 auto operator*() const noexcept
  {
    return *m_ptr;
  }

  // In host code, attempting to use this will produce a clear diagnostic
  // instead of silently allowing an invalid dereference of device memory.
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 pointer operator->() const noexcept
  {
    static_assert(false,
                  "thrust::unique_ptr<T>::operator->(): cannot dereference device memory from host. "
                  "Copy the object to host first (T host = *ptr) or access members inside device code.");

    return m_ptr;
  }

  //==========================================================================
  // Modifiers
  //==========================================================================
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 pointer release() noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = pointer();
    return temp;
  }

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 void reset(pointer p = pointer()) noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = p;
    if (temp)
    {
      m_deleter(temp);
    }
  }

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr& u) noexcept
  {
    using std::swap;
    swap(m_ptr, u.m_ptr);
    swap(m_deleter, u.m_deleter);
  }
};

template <class T, class D>
class unique_ptr<T[], D>
{};

template <class T, class D, std::enable_if_t<std::is_swappable_v<D>, void>>
inline THRUST_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr<T, D>& x, unique_ptr<T, D>& y) noexcept
{
  x.swap(y);
}

//==============================================================================
// Comparison Operators
//==============================================================================
template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator==(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return x.get() == y.get();
}

#if THRUST_STD_VER <= 17
template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator!=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(x == y);
}
#endif

template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator<(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  using P1 = typename unique_ptr<T1, D1>::element_type*;
  using P2 = typename unique_ptr<T2, D2>::element_type*;
  using CTP = typename std::common_type<P1, P2>::type;
  return std::less<CTP>()(thrust::raw_pointer_cast(x.get()), thrust::raw_pointer_cast(y.get()));
}

template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator>(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return y < x;
}

template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator<=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(y < x);
}

template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator>=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(x < y);
}

#if THRUST_STD_VER >= 20
template <class T1, class D1, class T2, class D2>
  THRUST_HOST inline auto operator<=> (const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  // TODO: once thrust::device_ptr supports three_way_comparison, we should be using that
  return std::compare_three_way()(x.get_raw(), y.get_raw());
}
#endif

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator==(const unique_ptr<T, D>& x, std::nullptr_t) noexcept
{
  return !x;
}

#if THRUST_STD_VER <= 17
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator==(std::nullptr_t, const unique_ptr<T, D>& y) noexcept
{
  return !y;
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator!=(const unique_ptr<T, D>& x, std::nullptr_t) noexcept
{
  return static_cast<bool>(x);
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator!=(std::nullptr_t, const unique_ptr<T, D>& y) noexcept
{
  return static_cast<bool>(y);
}
#endif

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator<(const unique_ptr<T, D>& x, std::nullptr_t)
{
  return std::less<typename unique_ptr<T, D>::pointer>()(x.get(), nullptr); // x.get() < nullptr;
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator<(std::nullptr_t, const unique_ptr<T, D>& y)
{
  return std::less<typename unique_ptr<T, D>::pointer>()(nullptr, y.get()); // nullptr < y.get();
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator>(const unique_ptr<T, D>& x, std::nullptr_t)
{
  return nullptr < x;
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator>(std::nullptr_t, const unique_ptr<T, D>& y)
{
  return y < nullptr;
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator<=(const unique_ptr<T, D>& x, std::nullptr_t)
{
  return !(nullptr < x);
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator<=(std::nullptr_t, const unique_ptr<T, D>& y)
{
  return !(y < nullptr);
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator>=(const unique_ptr<T, D>& x, std::nullptr_t)
{
  return !(x < nullptr);
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator>=(std::nullptr_t, const unique_ptr<T, D>& y)
{
  return !(y < nullptr);
}

#if THRUST_STD_VER >= 20
template <class T, class D>
  THRUST_HOST inline auto operator<=> (const unique_ptr<T, D>& x, std::nullptr_t)
{
  // TODO: once thrust::device_ptr supports three_way_comparison, we should be using that
  return std::compare_three_way()(x.get_raw(), static_cast<T*>(nullptr));
}
#endif

//==============================================================================
// Make unique
//==============================================================================
template <class T, class... Args, class = std::enable_if_t<!std::is_array_v<T>>>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr<T> make_unique(Args&&... args)
{
  thrust::device_ptr<T> p = thrust::device_malloc<T>(1);
  return unique_ptr<T>(thrust::device_new<T>(p, T(std::forward<Args>(args)...), 1));
}

THRUST_NAMESPACE_END
