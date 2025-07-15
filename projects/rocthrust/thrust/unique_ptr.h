#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/device_free.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/device_reference.h>

#include <functional>
#include <type_traits>
#include <utility>

THRUST_NAMESPACE_BEGIN

template <class T, class = void>
struct default_delete;

template <class T>
struct default_delete<T, typename thrust::detail::enable_if<thrust::detail::not_<std::is_array<T>>::value>::type>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete() noexcept = default;

  template <class U>
  THRUST_HOST default_delete(
    const default_delete<U>&,
    typename thrust::detail::enable_if<thrust::detail::is_convertible<thrust::device_ptr<U>, pointer>::value>::type* =
      nullptr) noexcept
  {}

  THRUST_HOST
  void operator()(pointer ptr) const noexcept
  {
    // We use for_each_n to launch a kernel that executes the destructor on the device,
    // avoiding known issues with thrust::device_delete for user-defined types.
    if constexpr (!std::is_trivially_destructible<T>::value)
      thrust::for_each_n(ptr, 1, [] __device__(T& x) { x.~T(); });
    thrust::device_free(ptr);
  }
};

template <class T>
struct default_delete<
  T[],
  typename thrust::detail::enable_if<thrust::detail::not_<std::is_trivially_destructible<T>>::value>::type>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST
  constexpr default_delete(size_t n = 0) noexcept : m_size(n) {};

  template <class U>
  THRUST_HOST
  default_delete(const default_delete<U[]>& other,
              typename thrust::detail::enable_if<thrust::detail::is_convertible<U (*)[], T (*)[]>::value>::type* =
                nullptr) noexcept
      : m_size(other.size())
  {}

  THRUST_HOST
  void operator()(pointer ptr) const noexcept
  {
    // We use for_each_n to launch a kernel that executes the destructor on the device,
    // avoiding known issues with thrust::device_delete for user-defined types.
    if (m_size)
      thrust::for_each_n(ptr, m_size, [] __device__(T& x) { x.~T(); });
    thrust::device_free(ptr);
  }

  THRUST_HOST size_t size() const
  {
    return m_size;
  }

private:
  size_t m_size;
};

template <class T>
struct default_delete<T[], typename thrust::detail::enable_if<std::is_trivially_destructible<T>::value>::type>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete(size_t = 0) noexcept {};

  template <class U>
  THRUST_HOST default_delete(
    const default_delete<U>&,
    typename thrust::detail::enable_if<thrust::detail::is_convertible<thrust::device_ptr<U>, pointer>::value>::type* =
      nullptr) noexcept
  {}

  THRUST_HOST void operator()(pointer ptr) const noexcept
  {
    thrust::device_free(ptr);
  }
};

template <class Deleter>
struct unique_ptr_deleter_sfinae
{
  static_assert(thrust::detail::not_<thrust::detail::is_reference<Deleter>>::value, "incorrect specialization");
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

template <class T, class D = default_delete<T>>
class unique_ptr
{
public:
  using pointer      = typename D::pointer;
  using element_type = T;
  using deleter_type = D;

private:
  pointer                            m_ptr;
  [[no_unique_address]] deleter_type m_deleter;

  using DeleterSFINAE = unique_ptr_deleter_sfinae<D>;

  template <bool Dummy>
  using LValRefType = typename thrust::detail::dependent_type<DeleterSFINAE, Dummy>::type::lval_ref_type;

  template <bool Dummy>
  using GoodRValRefType = typename thrust::detail::dependent_type<DeleterSFINAE, Dummy>::type::good_rval_ref_type;

  template <bool Dummy>
  using BadRValRefType = typename thrust::detail::dependent_type<DeleterSFINAE, Dummy>::type::bad_rval_ref_type;

  template <bool Dummy,
            class Deleter = typename thrust::detail::dependent_type<typename thrust::detail::identity_<deleter_type>::type, Dummy>::type>
  using EnableIfDeleterDefaultConstructible = typename thrust::detail::enable_if<
    thrust::detail::and_<std::is_default_constructible<Deleter>,
                         thrust::detail::not_<thrust::detail::is_pointer<Deleter>>>::value>::type;

  template <class ArgType>
  using EnableIfDeleterConstructible =
    typename thrust::detail::enable_if<std::is_constructible<deleter_type, ArgType>::value>::type;

  template <class U, class E>
  using EnableIfMoveConvertible =
    typename thrust::detail::enable_if<thrust::detail::and_<thrust::detail::is_convertible<typename U::pointer, pointer>,
                                                            thrust::detail::not_<std::is_array<E>>>::value>::type;

  template <class E>
  using EnableIfDeleterConvertible = typename thrust::detail::enable_if<
    thrust::detail::or_<thrust::detail::and_<thrust::detail::is_reference<D>, thrust::detail::is_same<D, E>>,
                        thrust::detail::and_<thrust::detail::not_<thrust::detail::is_reference<D>>,
                                             thrust::detail::is_convertible<E, D>>>::value>::type;

  template <class E>
  using EnableIfDeleterAssignable =
    typename thrust::detail::enable_if<thrust::detail::is_assignable<D&, E&&>::value>::type;

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
    THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(pointer                p,
                                                                GoodRValRefType<Dummy> d) noexcept
        : m_ptr(p)
        , m_deleter(std::move(d))
    {
      static_assert(thrust::detail::not_<thrust::detail::is_reference<deleter_type>>::value,
                  "rvalue deleter bound to reference");
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

  // Calling `->` on a `unique_ptr` from host code (e.g., `my_unique_ptr->member`)
  // will attempt to dereference a device pointer on the host, leading to a
  // segmentation fault.
  //
  // To access the object's members, you must first explicitly copy the data
  // from the device to a host-side object (e.g., `host_copy = *my_unique_ptr;`).
  // THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 pointer operator->() const noexcept
  // {
  //     return m_ptr;
  // }

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

template <class T, class D, typename thrust::detail::enable_if<std::is_swappable<D>::value, void>::type>
inline THRUST_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr<T, D>& x, unique_ptr<T, D>& y) noexcept
{
  x.swap(y);
}

//==============================================================================
// Comparison Operators
//==============================================================================
template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator==(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  // The initial `std::common_type` approach was considered to support comparisons
  // between related types (e.g., a base and derived class pointer). However,
  // a bug in `thrust::device_delete` prevented testing with class hierarchies.
  //
  // Testing was pivoted to use unrelated types (e.g., `int*` and `float*`),
  // which revealed that `std::common_type` is ill-formed for such cases.
  //
  // The standard-compliant and robust solution for comparing any two object
  // pointers is to cast them to `const void*`. This correctly handles all
  // cases: related, unrelated, and incomplete types.
  return std::equal_to<const void*>()(x.get_raw(), y.get_raw()); // const void*
}

#if THRUST_STD_VER <= 17
template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator!=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(x == y);
}
#endif

template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator<(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  // Similar to `operator==`, the `const void*` cast provides a robust,
  // standard-compliant way to establish a strict total ordering for any two
  // object pointers, which was necessary after testing with unrelated types
  // proved the `std::common_type` approach to be insufficient.
  return std::less<const void*>()(x.get_raw(), y.get_raw()); // const void*
}

template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator>(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return y < x;
}

template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator<=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(y < x);
}

template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator>=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(x < y);
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator==(const unique_ptr<T, D>& x, std::nullptr_t) noexcept
{
  return !x;
}

#if THRUST_STD_VER <= 17
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator==(std::nullptr_t, const unique_ptr<T, D>& y) noexcept
{
  return !y;
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator!=(const unique_ptr<T, D>& x, std::nullptr_t) noexcept
{
  return static_cast<bool>(x);
}

template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool
operator!=(std::nullptr_t, const unique_ptr<T, D>& y) noexcept
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

//==============================================================================
// Make unique
//==============================================================================
template <class T,
          class... Args,
          class = typename thrust::detail::enable_if<
              thrust::detail::not_<std::is_array<T>>::value>::type>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr<T> make_unique(Args&&... args)
{
  thrust::device_ptr<T> p = thrust::device_malloc<T>(1);
  return unique_ptr<T>(thrust::device_new<T>(p, T(std::forward<Args>(args)...), 1));
}

THRUST_NAMESPACE_END
