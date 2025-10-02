/*! \file
 *  \brief A smart pointer that owns and manages another object through a
 *         pointer and disposes of that object when the \p unique_ptr goes
 *         out of scope.
 */

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

/*! \addtogroup memory_management Memory Management
 *  \{
 */

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

/*! \p thrust::unique_ptr is a smart pointer that owns and manages another object,
 *  allocated in device memory, via a pointer and subsequently disposes of that
 *  object when the \p unique_ptr goes out of scope.
 *
 *  The object is disposed of using the associated `Deleter` when either of the
 *  following happens:
 *  - the managing `unique_ptr` object is destroyed.
 *  - the managing `unique_ptr` object is assigned another pointer via `operator=` or `reset()`.
 *
 *  The object is disposed of by calling `get_deleter()(get())`. The default deleter,
 *  `thrust::default_delete`, deallocates the memory using `thrust::device_free`.
 *  For non-trivially destructible types, Deleter invokes the destructor of the
 *  managed object on the device before deallocation.
 *
 *  A `unique_ptr` may alternatively own no object, in which case it is described as
 *  *empty*.
 *
 *  There are two versions of `thrust::unique_ptr`:
 *  1. Manages a single object
 *  2. Manages a dynamically-allocated array of objects
 *
 *
 *  \tparam T The type of the managed object.
 *  \tparam D The type of the deleter.
 *
 *  \see https://en.cppreference.com/w/cpp/memory/unique_ptr
 */

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

  /*! \brief Constructs a \p unique_ptr that does not own an object.
   */
  template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
  THRUST_HOST constexpr unique_ptr() noexcept
      : m_ptr()
      , m_deleter()
  {}

  /*! \brief Constructs a \p unique_ptr that does not own an object.
   */
  template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
  THRUST_HOST constexpr unique_ptr(std::nullptr_t) noexcept
      : unique_ptr()
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p p.
   *  \param p A pointer to the object in device memory to manage.
   */
  template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(pointer p) noexcept
      : m_ptr(p)
      , m_deleter()
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p raw_p.
   *  \param raw_p A raw pointer to the object in device memory to manage.
   */
  template <bool Dummy = true,
            class      = EnableIfDeleterDefaultConstructible<Dummy>,
            class      = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(T* raw_p) noexcept
      : m_ptr(device_pointer_cast(raw_p))
      , m_deleter()
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p p and uses \p d as the deleter.
   *  \param p A pointer to the object in device memory to manage.
   *  \param d The deleter to use.
   */
  template <bool Dummy = true, class = EnableIfDeleterConstructible<LValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(pointer p, LValRefType<Dummy> d) noexcept
      : m_ptr(p)
      , m_deleter(d)
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p p and uses \p d as the deleter.
   *  \param p A pointer to the object in device memory to manage.
   *  \param d The deleter to use.
   */
  template <bool Dummy = true, class = EnableIfDeleterConstructible<GoodRValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(pointer p, GoodRValRefType<Dummy> d) noexcept
      : m_ptr(p)
      , m_deleter(std::move(d))
  {
    static_assert(!std::is_reference_v<deleter_type>, "rvalue deleter bound to reference");
  }

  template <bool Dummy = true, class = EnableIfDeleterConstructible<BadRValRefType<Dummy>>>
  unique_ptr(pointer p, BadRValRefType<Dummy> d) = delete;

  /*! \brief Move constructor. Constructs a \p unique_ptr by taking ownership of the object managed by \p u.
   *  \param u The \p unique_ptr to move from.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr&& u) noexcept
      : m_ptr(u.release())
      , m_deleter(std::forward<deleter_type>(u.get_deleter()))
  {}

  /*! \brief Converting move constructor. Constructs a \p unique_ptr by taking ownership of the object managed by \p u.
   *
   *  Allows converting from \p unique_ptr<U, E> to \p unique_ptr<T, D> when
   *  the pointer and deleter types are compatible.
   * 
   *  \param u The \p unique_ptr to move from.
   */
  template <class U, class E, class = EnableIfMoveConvertible<unique_ptr<U, E>, U>, class = EnableIfDeleterConvertible<E>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr<U, E>&& u) noexcept
      : m_ptr(u.release())
      , m_deleter(std::forward<E>(u.get_deleter()))
  {}

  //==========================================================================
  // Assignment
  //==========================================================================
  /*! \brief Move assignment operator. Replaces the managed object with the one from \p u.
   *  \param u The \p unique_ptr to move from.
   *  \return `*this`
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(unique_ptr&& u) noexcept
  {
    reset(u.release());
    m_deleter = std::forward<deleter_type>(u.get_deleter());
    return *this;
  }

  /*! \brief Converting assignment operator.. Replaces the managed object with the one from \p u.
   *  \param u The \p unique_ptr to move from.
   *  \return `*this`
   */
  template <class U, class E, class = EnableIfMoveConvertible<unique_ptr<U, E>, U>, class = EnableIfDeleterAssignable<E>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(unique_ptr<U, E>&& u) noexcept
  {
    reset(u.release());
    m_deleter = std::forward<E>(u.get_deleter());
    return *this;
  }

  /*! \brief Assigns a null pointer, deallocating the managed object. Effectively the same as calling reset().
   *  \return `*this`
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(std::nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  //==========================================================================
  // Destructor
  //==========================================================================
  /*! \brief Destroys the \p unique_ptr, the managed object is destroyed via `get_deleter()(get())`. If get() == nullptr there are no effects.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 ~unique_ptr()
  {
    reset();
  }

  //==========================================================================
  // Observers
  //==========================================================================
  /*! \brief Returns a pointer to the managed object or `nullptr` if no object is owned.
   *  \return Pointer to the managed object.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 pointer get() const noexcept
  {
    return m_ptr;
  }

  /*! \brief Returns a raw pointer to the managed object or `nullptr` if no object is owned.
   *  \return Raw pointer to the managed object.
   */
  template <bool Dummy = true, class = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 T* get_raw() const noexcept
  {
    return raw_pointer_cast(m_ptr);
  }

  /*! \brief Returns a reference to the deleter object which would be used for destruction of the managed object.
   *  \return A reference to the stored deleter.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 deleter_type& get_deleter() noexcept
  {
    return m_deleter;
  }

  /*! \brief Returns a reference to the deleter object which would be used for destruction of the managed object.
   *  \return A reference to the stored deleter.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 const deleter_type& get_deleter() const noexcept
  {
    return m_deleter;
  }

  /*! \brief Checks if the \p unique_ptr owns an object.
   *  \return `true` if an object is owned, `false` otherwise.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit operator bool() const noexcept
  {
    return m_ptr != nullptr;
  }

  /*! \brief Dereferences the stored pointer.
   * 
   *  The default `unique_ptr` implementation uses `thrust::device_ptr`.
   *  Dereferencing this pointer in host code is a valid operation that
   *  results in a copy of the object from device to host memory.
   *
   *  \return A reference to the managed object.
   */
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
  /*! \brief Releases ownership of the managed object, if any.
   *  \return A pointer to the released object.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 pointer release() noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = pointer();
    return temp;
  }

  /*! \brief Replaces the managed object.
   *  \param p The new object to manage.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 void reset(pointer p = pointer()) noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = p;
    if (temp)
    {
      m_deleter(temp);
    }
  }

  /*! \brief Swaps the managed object and deleter with another \p unique_ptr.
   *  \param u The \p unique_ptr to swap with.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr& u) noexcept
  {
    using std::swap;
    swap(m_ptr, u.m_ptr);
    swap(m_deleter, u.m_deleter);
  }
};

template <class T, class D>
class __attribute__((trivial_abi)) unique_ptr<T[], D>
{
public:
  using pointer      = typename thrust::detail::pointer_detector<T, D>::type;
  using element_type = T;
  using deleter_type = D;

private:
  template <class Up, class OtherDeleter>
  friend class unique_ptr;

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
    std::enable_if_t<std::is_default_constructible_v<Deleter> && !std::is_pointer_v<Deleter>>;

  template <class ArgType>
  using EnableIfDeleterConstructible =
    std::enable_if_t<std::is_constructible_v<deleter_type, ArgType>>;

  template <class Pp>
  using EnableIfPointerConvertible = std::enable_if_t<std::is_same_v<Pp, pointer>>;

  template <bool Dummy,
            class Tp = typename thrust::detail::dependent_type<typename thrust::detail::identity_<element_type>::type,
                                                               Dummy>::type>
  using EnableIfTriviallyDestructible = std::enable_if_t<std::is_trivially_destructible_v<Tp>>;

  template <bool Dummy,
            class Tp = typename thrust::detail::dependent_type<typename thrust::detail::identity_<element_type>::type,
                                                               Dummy>::type>
  using EnableIfNotTriviallyDestructible = std::enable_if_t<!std::is_trivially_destructible_v<Tp>>;

  template <class UPtr, class Up, class ElemT = typename UPtr::element_type>
  using EnableIfMoveConvertible =
    std::enable_if_t<std::is_array_v<Up> && std::is_same_v<pointer, element_type*>
                            && std::is_same_v<typename UPtr::pointer, ElemT*>
                            && std::is_convertible_v<ElemT (*)[], element_type (*)[]>>;

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
  using EnableIfDeleterDefaultDelete = std::enable_if_t<std::is_same_v<Deleter, default_delete<T[]>>>;

public:
  //==========================================================================
  // Constructors
  //==========================================================================

  /*! \brief Constructs an empty \p unique_ptr that does not own an array.
   */
  template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
  THRUST_HOST constexpr unique_ptr() noexcept
      : m_ptr()
      , m_deleter()
  {}

  /*! \brief Constructs an empty \p unique_ptr that does not own an array.
   */
  template <bool Dummy = true, class = EnableIfDeleterDefaultConstructible<Dummy>>
  THRUST_HOST constexpr unique_ptr(std::nullptr_t) noexcept
      : unique_ptr()
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p p.
   *
   *  This overload is only available for arrays of trivially-destructible types.
   *
   *  \param p A pointer to an array in device memory to manage.
   */
  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterDefaultConstructible<Dummy>,
            class      = EnableIfPointerConvertible<Pp>,
            class      = EnableIfTriviallyDestructible<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(Pp p) noexcept
      : m_ptr(p)
      , m_deleter()
  {}

  /*! \brief Constructs a \p unique_ptr from a raw device array pointer.
   *
   *  This overload is only available for arrays of trivially-destructible types.
   * 
   * \param raw_p A raw pointer to an array in device memory to manage.
   */
  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterDefaultConstructible<Dummy>,
            class      = EnableIfPointerConvertible<device_ptr<T>>,
            class      = EnableIfTriviallyDestructible<Dummy>,
            class      = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(Pp* raw_p) noexcept
      : m_ptr(device_pointer_cast(raw_p))
      , m_deleter()
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p p with known size. 
   *
   *  For arrays of non-trivially-destructible types, the size is required to ensure
   *  all element destructors are properly called during deletion.
   *
   *  \param p A pointer to an array in device memory to manage.
   *  \param size The number of elements in the array.
   */
  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterDefaultConstructible<Dummy>,
            class      = EnableIfPointerConvertible<Pp>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(Pp p, size_t size) noexcept
      : m_ptr(p)
      , m_deleter(size)
  {}

  /*! \brief Constructs a \p unique_ptr from a raw device array pointer with known size.
   *
   *  For arrays of non-trivially-destructible types, the size is required to ensure
   *  all element destructors are properly called during deletion.
   *
   *  \param raw_p A raw pointer to an array in device memory to manage.
   *  \param size The number of elements in the array.
   */
  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterDefaultConstructible<Dummy>,
            class      = EnableIfPointerConvertible<device_ptr<T>>,
            class      = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(Pp* raw_p, size_t size) noexcept
      : m_ptr(device_pointer_cast(raw_p))
      , m_deleter(size)
  {}

  /*! \brief Constructs a \p unique_ptr with a custom deleter (lvalue reference).
   *
   *  \param p A pointer to the array in device memory to manage.
   *  \param deleter The deleter to use for destroying the array.
   */
  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterConstructible<LValRefType<Dummy>>,
            class      = EnableIfPointerConvertible<Pp>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(Pp p, LValRefType<Dummy> deleter) noexcept
      : m_ptr(p)
      , m_deleter(deleter)
  {}

  /*! \brief Constructs an empty \p unique_ptr with a custom deleter (lvalue reference).
   *
   *  \param deleter The deleter to store.
   */
  template <bool Dummy = true, class = EnableIfDeleterConstructible<LValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(std::nullptr_t, LValRefType<Dummy> deleter) noexcept
      : m_ptr(nullptr)
      , m_deleter(deleter)
  {}

  /*! \brief Constructs a \p unique_ptr with a custom deleter (rvalue reference).
   *
   *  \param p A pointer to the array in device memory to manage.
   *  \param deleter The deleter to use for destroying the array (moved).
   */
  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterConstructible<GoodRValRefType<Dummy>>,
            class      = EnableIfPointerConvertible<Pp>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(Pp p, GoodRValRefType<Dummy> deleter) noexcept
      : m_ptr(p)
      , m_deleter(std::move(deleter))
  {
    static_assert(!std::is_reference_v<deleter_type>, "rvalue deleter bound to reference");
  }

  /*! \brief Constructs an empty \p unique_ptr with a custom deleter (rvalue reference).
   *
   *  \param deleter The deleter to store (moved).
   */
  template <bool Dummy = true, class = EnableIfDeleterConstructible<GoodRValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(std::nullptr_t, GoodRValRefType<Dummy> deleter) noexcept
      : m_ptr(nullptr)
      , m_deleter(std::move(deleter))
  {
    static_assert(!std::is_reference_v<deleter_type>, "rvalue deleter bound to reference");
  }

  template <class Pp,
            bool Dummy = true,
            class      = EnableIfDeleterConstructible<BadRValRefType<Dummy>>,
            class      = EnableIfPointerConvertible<Pp>>
  THRUST_HOST unique_ptr(Pp ptr, BadRValRefType<Dummy> deleter) = delete;

  /*! \brief Move constructor that transfers ownership from another array \p unique_ptr.
   *
   *  \param u The \p unique_ptr to move from.
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr&& u) noexcept
      : m_ptr(u.release())
      , m_deleter(std::forward<deleter_type>(u.get_deleter()))
  {}

  /*! \brief Converting move constructor from a compatible array \p unique_ptr.
   *
   *  Allows converting from \p unique_ptr<U[], E> to \p unique_ptr<T[], D> when
   *  the array element types and deleter types are compatible (e.g., derived to base).
   *
   *  \tparam Up An array element type convertible to \p T.
   *  \tparam Ep A deleter type convertible to \p D.
   *  \param u The \p unique_ptr to move from.
   */
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
  /*! \brief Move assignment operator. Replaces the managed object with the one from \p u.
   *
   *  Releases the currently managed array (if any) and takes ownership of
   *  the array managed by \p p.
   *
   *  \param p The \p unique_ptr to move from.
   *  \return `*this`
   */
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr& operator=(unique_ptr&& p) noexcept
  {
    reset(p.release());
    m_deleter = std::forward<deleter_type>(p.get_deleter());
    return *this;
  }

  /*! \brief Converting assignment operator. Replaces the managed object with the one from \p u.
   *  \param u The \p unique_ptr to move from.
   *  \return `*this`
   */
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

  /*! \brief Assigns a null pointer, deallocating the managed object. Effectively the same as calling reset().
   *  \return `*this`
   */
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

  template <bool Dummy = true, class = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 T* get_raw() const noexcept
  {
    return raw_pointer_cast(m_ptr);
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
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 pointer release() noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = pointer();
    return temp;
  }

  template <class Pp, class = EnableIfPointerConvertible<Pp>>
  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 void reset(Pp p) noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = p;
    if (temp)
    {
      m_deleter(temp);
    }
  }

  THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 void reset(std::nullptr_t = nullptr) noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = nullptr;
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

template <class T, class D, std::enable_if_t<std::is_swappable_v<D>, void>>
inline THRUST_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr<T, D>& x, unique_ptr<T, D>& y) noexcept
{
  x.swap(y);
}

//==============================================================================
// Comparison Operators
//==============================================================================
/*! \brief Compares two \p unique_ptr objects for equality.
 *
 *  Two \p unique_ptr objects are considered equal if they point to the same
 *  memory address or are both null.
 *
 *  \param x The first \p unique_ptr to compare.
 *  \param y The second \p unique_ptr to compare.
 *  \return `true` if the pointers are equal, `false` otherwise.
 */
template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator==(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return x.get() == y.get();
}

#if THRUST_STD_VER <= 17
template <class T1, class D1, class T2, class D2>
/*! \brief Compares two \p unique_ptr objects for inequality (C++17 and earlier).
 *
 *  \param x The first \p unique_ptr to compare.
 *  \param y The second \p unique_ptr to compare.
 *  \return `true` if the pointers are not equal, `false` otherwise.
 */
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 bool operator!=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(x == y);
}
#endif

/*! \brief Compares two \p unique_ptr objects using less-than ordering.
 *
 *  \param x The first \p unique_ptr to compare.
 *  \param y The second \p unique_ptr to compare.
 *  \return `true` if the pointer stored in \p x is less than the pointer stored in \p y, `false` otherwise.
 *  
 *  \note Operators `>`, `<=`, and `>=` are also provided and defined in terms of this operator.
 */
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
/*! \brief Constructs an object of type \p T in device memory and wraps it in a \p unique_ptr.
 *
 *  Allocates device memory for a single object of type \p T, constructs the object
 *  by forwarding the provided arguments, and returns a \p unique_ptr managing the
 *  allocated object.
 *
 *  This overload participates in overload resolution only if \p T is not an array type.
 *
 *  \tparam T The type of object to construct (must not be an array).
 *  \tparam Args The types of arguments to forward to the constructor of \p T.
 *  \param args Arguments to forward to the constructor of \p T.
 *  \return A \p unique_ptr<T> managing the newly created object.
 */
template <class T, class... Args, class = std::enable_if_t<!std::is_array_v<T>>>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr<T> make_unique(Args&&... args)
{
  thrust::device_ptr<T> p = thrust::device_malloc<T>(1);
  return unique_ptr<T>(thrust::device_new<T>(p, T(std::forward<Args>(args)...), 1));
}

/*! \brief Constructs an array of objects of type \p T in device memory and wraps it in a \p unique_ptr.
 *
 *  Allocates device memory for an array of \p n objects of type \p U (where \p T is \p U[]),
 *  and returns a \p unique_ptr managing the allocated array.
 *
 *  This overload participates in overload resolution only if \p T is an array of unknown
 *  bound (e.g., \p T[]).
 *
 *  \tparam T The array type (e.g., \p int[], \p MyClass[]).
 *  \param n The number of elements in the array.
 *  \return A \p unique_ptr<T> managing the newly created array.
 */
template <class T, class = std::enable_if_t<thrust::detail::is_unbounded_array<T>::value>>
THRUST_HOST inline THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr<T> make_unique(size_t n)
{
  using U = typename std::remove_extent<T>::type;
  return unique_ptr<T>(thrust::device_new<U>(n), n);
}

template <class T, class... Args, class = std::enable_if_t<thrust::detail::is_bounded_array<T>::value>>
THRUST_HOST void make_unique(Args&&...) = delete;

#if THRUST_STD_VER >= 20

/*! \brief Constructs an object of type \p T in device memory without initialization (C++20).
 *
 *  Allocates device memory for a single object of type \p T without initializing it,
 *  and returns a \p unique_ptr managing the allocated memory. The object has
 *  indeterminate value.
 *
 *  This overload participates in overload resolution only if \p T is not an array type.
 *
 *  \tparam T The type of object to allocate (must not be an array).
 *  \return A \p unique_ptr<T> managing the uninitialized memory.
 */
template <class T, class = std::enable_if_t<!std::is_array_v<T>>>
THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr<T> make_unique_for_overwrite()
{
  return unique_ptr<T>(thrust::device_malloc<T>(1));
}

/*! \brief Constructs an array without initialization (C++20).
 *
 *  Allocates device memory for an array of \p n objects of type \p U (where \p T is \p U[])
 *  without initializing the elements, and returns a \p unique_ptr managing the allocated
 *  array. The elements have indeterminate values.
 *
 *  This overload participates in overload resolution only if \p T is an array of unknown
 *  bound (e.g., \p T[]).
 * 
 *  \tparam T The array type (e.g., \p int[], \p MyClass[]).
 *  \param n The number of elements in the array.
 *  \return A \p unique_ptr<T> managing the uninitialized array.
 */
template <class T, class = std::enable_if_t<thrust::detail::is_unbounded_array<T>::value>>
THRUST_HOST THRUST_CONSTEXPR_SINCE_CXX23 unique_ptr<T> make_unique_for_overwrite(size_t n)
{
  using U = typename std::remove_extent<T>::type;

  return unique_ptr<T>(thrust::device_malloc<U>(n), n);
}

template <class T, class... Args, class = std::enable_if_t<thrust::detail::is_bounded_array<T>::value>>
THRUST_HOST void make_unique_for_overwrite(Args&&...) = delete;

#endif

/*! \} // end smart_pointers
 */

THRUST_NAMESPACE_END
