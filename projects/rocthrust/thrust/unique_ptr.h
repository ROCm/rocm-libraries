#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/device_free.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>

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
                typename std::enable_if<std::is_convertible<thrust::device_ptr<U>, pointer>::value>::type * = nullptr) noexcept {}

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
                typename std::enable_if<std::is_convertible<U(*)[], T(*)[]>::value>::type * = nullptr) noexcept 
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
template <class T, class D = default_delete<T>>
class unique_ptr
{
public:
    using pointer = typename D::pointer;
    using element_type = T;
    using deleter_type = D;

private:
    pointer                            m_ptr;
    [[no_unique_address]] deleter_type m_deleter;

public:
    //==========================================================================
    // Constructors
    //==========================================================================

    THRUST_HOST
    THRUST_CONSTEXPR_SINCE_CXX17 unique_ptr() noexcept : m_ptr(nullptr), m_deleter() {}

    THRUST_HOST
    THRUST_CONSTEXPR_SINCE_CXX17 unique_ptr(std::nullptr_t) noexcept : unique_ptr() {}

    THRUST_HOST
    explicit unique_ptr(pointer p) noexcept : m_ptr(p), m_deleter() {}

    THRUST_HOST
    unique_ptr(pointer p, const deleter_type& d) noexcept : m_ptr(p), m_deleter(d) {}

    THRUST_HOST
    unique_ptr(pointer p, deleter_type&& d) noexcept : m_ptr(p), m_deleter(std::move(d)) {}

    THRUST_HOST
    unique_ptr(unique_ptr&& u) noexcept : m_ptr(u.release()), m_deleter(std::forward<deleter_type>(u.get_deleter())) {}
    
    template <
        class U, class E,
        class = typename std::enable_if<
                std::is_convertible<typename unique_ptr<U, E>::pointer, pointer>::value &&
                !std::is_array<U>::value
            >::type,
        class = typename std::enable_if<
            (std::is_reference<deleter_type>::value && std::is_same<deleter_type, E>::value) ||
            (!std::is_reference<deleter_type>::value && std::is_convertible<E, deleter_type>::value)
        >::type
    >
    THRUST_HOST
    unique_ptr(unique_ptr<U, E>&& u) noexcept
        : m_ptr(u.release()), m_deleter(std::forward<E>(u.get_deleter())) {}


};

template <class T, class D>
class unique_ptr<T[], D>
{

};




THRUST_NAMESPACE_END
