#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/device_free.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/device_reference.h>

#include <type_traits>
#include <utility>

THRUST_NAMESPACE_BEGIN

template <class T, class = void>
struct default_delete;

template <class T>
struct default_delete<T, typename thrust::detail::enable_if<thrust::detail::not_<std::is_array<T>>::value>::type>
{
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST
  constexpr default_delete() noexcept = default;

  template <class U>
  THRUST_HOST
  default_delete(const default_delete<U>&,
              typename thrust::detail::enable_if<
                  thrust::detail::is_convertible<thrust::device_ptr<U>,
                                                  pointer>::value>::type* = nullptr) noexcept
  {
  }

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
  THRUST_HOST default_delete(
      const default_delete<U[]>& other,
      typename thrust::detail::enable_if<
          thrust::detail::is_convertible<U (*)[], T (*)[]>::value>::type* = nullptr) noexcept
      : m_size(other.size())
  {
  }

  THRUST_HOST
  void operator()(pointer ptr) const noexcept
  {
    // We use for_each_n to launch a kernel that executes the destructor on the device,
    // avoiding known issues with thrust::device_delete for user-defined types.
    if (m_size)
      thrust::for_each_n(ptr, m_size, [] __device__(T& x) { x.~T(); });
    thrust::device_free(ptr);
  }

  THRUST_HOST
  size_t size() const
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
    using EnableIfDeleterConstructible = typename thrust::detail::enable_if<
        std::is_constructible<deleter_type, ArgType>::value>::type;

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
    {
    }

    template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
    THRUST_HOST constexpr unique_ptr(std::nullptr_t) noexcept
        : unique_ptr()
    {
    }

    template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
    THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(pointer p) noexcept
        : m_ptr(p)
        , m_deleter()
    {
    }

    template <bool Dummy = true, class = EnableIfDeleterConstructible<LValRefType<Dummy>>>
    THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(pointer            p,
                                                                LValRefType<Dummy> d) noexcept
        : m_ptr(p)
        , m_deleter(d)
    {
    }

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
    {
    }

    template <class U,
              class E,
              class = EnableIfMoveConvertible<unique_ptr<U, E>, U>,
              class = EnableIfDeleterConvertible<E>>
    THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr<U, E>&& u) noexcept
        : m_ptr(u.release())
        , m_deleter(std::forward<E>(u.get_deleter()))
    {
    }


  //==========================================================================
  // Assignment
  //==========================================================================
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(unique_ptr&& u) noexcept
  {
    reset(u.release());
    m_deleter = std::forward<deleter_type>(u.get_deleter());
    return *this;
  }

    template <class U,
              class E,
              class = EnableIfMoveConvertible<unique_ptr<U, E>, U>,
              class = EnableIfDeleterAssignable<E>>
    THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr&
    operator=(unique_ptr<U, E>&& u) noexcept
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

    //==========================================================================
    // Modifiers
    //==========================================================================

};

template <class T, class D>
class unique_ptr<T[], D>
{
};



THRUST_NAMESPACE_END
