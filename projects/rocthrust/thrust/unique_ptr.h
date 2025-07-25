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
struct default_delete<T, typename std::enable_if<!std::is_array<T>::value>::type>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete() noexcept = default;

  template <class U>
  THRUST_HOST default_delete(
    const default_delete<U>&,
    typename std::enable_if<std::is_convertible<thrust::device_ptr<U>, pointer>::value>::type* = nullptr) noexcept
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
    if constexpr (!std::is_trivially_destructible<T>::value)
    {
      thrust::for_each_n(ptr, 1, [] __device__(T & x) {
        x.~T();
      });
    }
    thrust::device_free(ptr);
  }
};

template <class T>
struct default_delete<T[], typename std::enable_if<!std::is_trivially_destructible<T>::value>::type>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete(size_t n = 0) noexcept
      : m_size(n){};

  template <class U>
  THRUST_HOST
  default_delete(const default_delete<U[]>& other,
                 typename std::enable_if<std::is_convertible<U (*)[], T (*)[]>::value>::type* = nullptr) noexcept
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
struct default_delete<T[], typename std::enable_if<std::is_trivially_destructible<T>::value>::type>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete(size_t = 0) noexcept {};

  template <class U>
  THRUST_HOST default_delete(
    const default_delete<U>&,
    typename std::enable_if<std::is_convertible<thrust::device_ptr<U>, pointer>::value>::type* = nullptr) noexcept
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
  static_assert(!std::is_reference<Deleter>::value, "incorrect specialization");
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
    typename std::enable_if<std::is_default_constructible<Deleter>::value && !std::is_pointer<Deleter>::value>::type;

  template <class ArgType>
  using EnableIfDeleterConstructible =
    typename std::enable_if<std::is_constructible<deleter_type, ArgType>::value>::type;

  template <class U, class E>
  using EnableIfMoveConvertible =
    typename std::enable_if<std::is_convertible<typename U::pointer, pointer>::value && !std::is_array<E>::value>::type;

  template <class E>
  using EnableIfDeleterConvertible =
    typename std::enable_if<(std::is_reference<D>::value && std::is_same<D, E>::value)
                            || (!std::is_reference<D>::value && std::is_convertible<E, D>::value)>::type;

  template <class E>
  using EnableIfDeleterAssignable = typename std::enable_if<std::is_assignable<D&, E&&>::value>::type;

  template <
    bool Dummy,
    class Deleter =
      typename thrust::detail::dependent_type<typename thrust::detail::identity_<deleter_type>::type, Dummy>::type>
  using EnableIfDeleterDefaultDelete = typename std::enable_if<std::is_same<Deleter, default_delete<T>>::value>::type;

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
    static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference");
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
{
public:
  using pointer      = typename thrust::detail::pointer_detector<T, D>::type;
  using element_type = T;
  using deleter_type = D;



private:
  template <class Up, class OtherDeleter>
  friend class unique_ptr;

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

  template <class Pp>
  using EnableIfPointerConvertible = typename thrust::detail::enable_if<thrust::detail::is_same<Pp, pointer>::value>::type;

  template <bool Dummy,
            class Tp = typename thrust::detail::dependent_type<typename thrust::detail::identity_<element_type>::type, Dummy>::type>
  using EnableIfTriviallyDestructible =
    typename thrust::detail::enable_if<std::is_trivially_destructible<Tp>::value>::type;

  template <bool Dummy,
            class Tp = typename thrust::detail::dependent_type<typename thrust::detail::identity_<element_type>::type, Dummy>::type>
  using EnableIfNotTriviallyDestructible =
    typename thrust::detail::enable_if<thrust::detail::not_<std::is_trivially_destructible<Tp>>::value>::type;

  template <class UPtr, class Up, class ElemT = typename UPtr::element_type>
  using EnableIfMoveConvertible = typename thrust::detail::enable_if<
    thrust::detail::and_<std::is_array<Up>,
                         thrust::detail::is_same<pointer, element_type*>,
                         thrust::detail::is_same<typename UPtr::pointer, ElemT*>,
                         thrust::detail::is_convertible<ElemT (*)[], element_type (*)[]>>::value>;

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

  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterDefaultConstructible<Dummy>,
            class      = EnableIfPointerConvertible<Pp>,
            class      = EnableIfTriviallyDestructible<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(Pp p) noexcept
      : m_ptr(p)
      , m_deleter()
  {}

  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterDefaultConstructible<Dummy>,
            class      = EnableIfPointerConvertible<Pp>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(Pp p, size_t size) noexcept
      : m_ptr(p)
      , m_deleter(size)
  {}

  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterConstructible<LValRefType<Dummy>>,
            class      = EnableIfPointerConvertible<Pp>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(Pp p, LValRefType<Dummy> deleter) noexcept
      : m_ptr(p)
      , m_deleter(deleter)
  {}

  template <bool Dummy = true, class = EnableIfDeleterConstructible<LValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(std::nullptr_t, LValRefType<Dummy> deleter) noexcept
      : m_ptr(nullptr)
      , m_deleter(deleter)
  {}

  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterConstructible<GoodRValRefType<Dummy>>,
            class      = EnableIfPointerConvertible<Pp>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(Pp p, GoodRValRefType<Dummy> deleter) noexcept
      : m_ptr(p)
      , m_deleter(std::move(deleter))
  {
    static_assert(thrust::detail::not_<thrust::detail::is_reference<deleter_type>>::value,
                  "rvalue deleter bound to reference");
  }

  template <bool Dummy = true, class = EnableIfDeleterConstructible<GoodRValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(std::nullptr_t, GoodRValRefType<Dummy> deleter) noexcept
      : m_ptr(nullptr)
      , m_deleter(std::move(deleter))
  {
    static_assert(thrust::detail::not_<thrust::detail::is_reference<deleter_type>>::value,
                  "rvalue deleter bound to reference");
  }

  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterConstructible<BadRValRefType<Dummy>>,
            class      = EnableIfPointerConvertible<Pp>>
  THRUST_HOST unique_ptr(Pp ptr, BadRValRefType<Dummy> deleter) = delete;

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr&& u) noexcept
      : m_ptr(u.release())
      , m_deleter(std::forward<deleter_type>(u.get_deleter()))
  {}

  template <class Up,
            class Ep,
            class = EnableIfMoveConvertible<unique_ptr<Up, Ep>, Up>,
            class = EnableIfDeleterConvertible<Ep>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr<Up, Ep>&& u) noexcept
      : m_ptr(u.release())
      , m_deleter(std::forward<Ep>(u.get_deleter()))
  {}

  //==========================================================================
  // Assignment
  //==========================================================================
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(unique_ptr&& p) noexcept
  {
    reset(p.release());
    m_deleter = std::forward<deleter_type>(p.get_deleter());
    return *this;
  }

  template <class Up,
            class Ep,
            class = EnableIfMoveConvertible<unique_ptr<Up, Ep>, Up>,
            class = EnableIfDeleterAssignable<Ep>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(unique_ptr<Up, Ep>&& p) noexcept
  {
    reset(p.release());
    m_deleter = std::forward<Ep>(p.get_deleter());
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

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 auto operator[](size_t i) const noexcept
  {
    return m_ptr[i];
  }

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

  //==========================================================================
  // Modifiers
  //==========================================================================
  



};

template <class T, class D, typename std::enable_if<std::is_swappable<D>::value, void>::type>
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
template <class T, class... Args, class = typename std::enable_if<!std::is_array<T>::value>::type>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr<T> make_unique(Args&&... args)
{
  thrust::device_ptr<T> p = thrust::device_malloc<T>(1);
  return unique_ptr<T>(thrust::device_new<T>(p, T(std::forward<Args>(args)...), 1));
}

THRUST_NAMESPACE_END
