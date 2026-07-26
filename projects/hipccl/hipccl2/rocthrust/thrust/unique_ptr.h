// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  Apache-2.0

/*! \file unique_ptr.h
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

#ifndef THRUST_DOXYGEN_INVOKED
namespace internal
{
struct destroy_device_functor
{
  template <typename T>
  THRUST_DEVICE void operator()(T& x) const noexcept
  {
    x.~T();
  }
};
} // namespace internal
#endif

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \{
 */

/*! \brief Default deleter for \p unique_ptr.
 *
 *  The default deleter used by \p unique_ptr when no custom deleter is specified.
 *  Calls the destructor (if needed) and deallocates memory using \p thrust::device_free.
 *
 *  \tparam T The type of object to delete (single object form).
 *
 *  \see thrust::unique_ptr
 *  \see thrust::device_free
 */
template <class T, class = void>
struct default_delete;

/*! \brief Default deleter specialization for single objects.
 *
 *  This specialization handles deletion of single objects allocated in device memory.
 *
 *  \tparam T The type of object to delete.
 */
template <class T>
struct default_delete<T, std::enable_if_t<!std::is_array_v<T>>>
{
  /*! \typedef pointer
   *  \brief The pointer type (`thrust::device_ptr<T>`).
   */
  using pointer = thrust::device_ptr<T>;

  THRUST_HOST constexpr default_delete() noexcept;

  /*! \brief Converting constructor from compatible deleter.
   *
   *  Allows construction from a deleter for a convertible type.
   *
   *  \tparam U A type convertible to \p T.
   */
  template <class U>
  THRUST_HOST default_delete(
    const default_delete<U>&,
    std::enable_if_t<std::is_convertible_v<thrust::device_ptr<U>, pointer>>* = nullptr) noexcept
  {}

  /*! \brief Deletes the object pointed to by \p ptr.
   *
   *  Calls the destructor (if needed) and deallocates memory using thrust::device_free.
   *
   *  \param ptr Pointer to the object to delete.
   */
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
      thrust::for_each_n(ptr, 1, ::internal::destroy_device_functor{});
    }
    thrust::device_free(ptr);
  }
};

template <class T>
THRUST_HOST constexpr default_delete<T, std::enable_if_t<!std::is_array_v<T>>>::default_delete() noexcept = default;

/*! \brief Default deleter specialization for arrays of non-trivially destructible types.
 *
 *  This specialization handles deletion of arrays where element destructors must be called.
 *
 *  \tparam T The array element type (non-trivially destructible).
 */
template <class T>
struct default_delete<T[], std::enable_if_t<!std::is_trivially_destructible_v<T>>>
{
  /*! \typedef pointer
   *  \brief The pointer type (`thrust::device_ptr<T>`).
   */
  using pointer = thrust::device_ptr<T>;

  // Allow other instantiations to access private members for converting constructor
  template <class U, class>
  friend struct default_delete;

  /*! \brief Constructs a deleter with the specified array size.
   *
   *  \param n The number of elements in the array (default: 0).
   */
  THRUST_HOST constexpr default_delete(size_t n = 0) noexcept
      : m_size(n){};

  /*! \brief Copy and converting constructor from compatible array deleter.
   *
   *  Copies the size from another deleter for a convertible array type.
   *
   *  \tparam U An array element type convertible to \p T.
   *  \param other The deleter to copy from.
   */
  template <class U>
  THRUST_HOST default_delete(const default_delete<U[]>& other,
                             std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>>* = nullptr) noexcept
      : m_size(other.m_size)
  {}

  /*! \brief Deletes the array pointed to by \p ptr.
   *
   *  Calls destructors (if needed) and deallocates memory using thrust::device_free.
   *
   *  \param ptr Pointer to the array to delete.
   */
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
      thrust::for_each_n(ptr, m_size, ::internal::destroy_device_functor{});
    }
    thrust::device_free(ptr);
  }

private:
  size_t m_size;
};

/*! \brief Default deleter specialization for arrays of trivially destructible types.
 *
 *  For arrays of trivially destructible element types, this specialization intentionally
 *  does NOT store the element count. Their destruction is a no-op, so only the raw
 *  deallocation is required. Omitting the size keeps this deleter specialization an
 *  empty (zero-size) type, allowing \p unique_ptr<T[]> instantiations that use it to
 *  remain a low/zero-cost abstraction (the default deleter need not increase the overall
 *  size of unique_ptr<T[]> above the size of a simple pointer `T *`).
 *
 *  \tparam T The array element type (trivially destructible).
 */
template <class T>
struct default_delete<T[], std::enable_if_t<std::is_trivially_destructible_v<T>>>
{
  /*! \typedef pointer
   *  \brief The pointer type (`thrust::device_ptr<T>`).
   */
  using pointer = thrust::device_ptr<T>;

  /*! \brief Constructs a deleter (size parameter ignored for trivially destructible types).
   *
   *  The size parameter is accepted but ignored since trivially destructible types
   *  don't require element-wise destruction.
   */
  THRUST_HOST constexpr default_delete(size_t = 0) noexcept {};

  /*! \brief Copy and converting constructor from compatible deleter.
   *
   *  Allows construction from a deleter for a convertible type.
   *
   *  \tparam U A type convertible to \p T.
   */
  template <class U>
  THRUST_HOST default_delete(
    const default_delete<U>&,
    std::enable_if_t<std::is_convertible_v<thrust::device_ptr<U>, pointer>>* = nullptr) noexcept
  {}

  /*! \brief Deletes the array pointed to by \p ptr.
   *
   *  Deallocates memory using thrust::device_free.
   *
   *  \param ptr Pointer to the array to delete.
   */
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
 *  `thrust::default_delete`, calls the destructor (if needed) and deallocates memory
 *  using `thrust::device_free`.
 *
 *  A `unique_ptr` may alternatively own no object, in which case it is described as
 *  *empty*.
 *
 *  There are two versions of `thrust::unique_ptr`:
 *  1. Manages a single object
 *  2. Manages a dynamically-allocated array of objects
 *
 *  \par Nested Types
 *  - `pointer` - The type of the stored pointer (defaults to `thrust::device_ptr<T>`)
 *  - `element_type` - The type of the managed object (`T`)
 *  - `deleter_type` - The type of the deleter (`D`)
 *
 *  \tparam T The type of the managed object.
 *  \tparam D The type of the deleter.
 *
 *  \see thrust::default_delete
 *  \see thrust::make_unique
 *  \see thrust::device_ptr
 *  \see thrust::device_malloc
 *  \see thrust::device_new
 *  \see thrust::device_free
 *  \see https://en.cppreference.com/w/cpp/memory/unique_ptr
 */
template <class T, class D = default_delete<T>>
#if THRUST_HAS_ATTRIBUTE(trivial_abi)
class __attribute__((trivial_abi)) unique_ptr
#else
class unique_ptr
#endif
{
public:
  /*! \typedef pointer
   *  \brief The type of the stored pointer (defaults to `thrust::device_ptr<T>`).
   */
  using pointer = typename thrust::detail::pointer_detector<T, D>::type;

  /*! \typedef element_type
   *  \brief The type of the managed object (`T`).
   */
  using element_type = T;
  
  /*! \typedef deleter_type
   *  \brief The type of the deleter (`D`).
   */
  using deleter_type = D;

  // TODO: When a standard "trivially relocatable" facility lands, add an
  // annotation/macro here to advertise that unique_ptr can be bitwise relocated

private:
  pointer m_ptr;
#if THRUST_COMPILER(MSVC) && THRUST_HAS_CPP_ATTRIBUTE(msvc::no_unique_address)
  [[msvc::no_unique_address]] deleter_type m_deleter;
#elif THRUST_HAS_CPP_ATTRIBUTE(no_unique_address)
  [[no_unique_address]] deleter_type m_deleter;
#else
  deleter_type m_deleter;
#endif

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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 explicit unique_ptr(pointer p) noexcept
      : m_ptr(p)
      , m_deleter()
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p raw_p.
   *  \param raw_p A raw pointer to the object in device memory to manage.
   */
  template <bool Dummy = true,
            class      = EnableIfDeleterDefaultConstructible<Dummy>,
            class      = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 explicit unique_ptr(T* raw_p) noexcept
      : m_ptr(device_pointer_cast(raw_p))
      , m_deleter()
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p p and uses \p d as the deleter.
   *  \param p A pointer to the object in device memory to manage.
   *  \param d The deleter to use.
   */
  template <bool Dummy = true, class = EnableIfDeleterConstructible<LValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(pointer p, LValRefType<Dummy> d) noexcept
      : m_ptr(p)
      , m_deleter(d)
  {}

  /*! \brief Constructs a \p unique_ptr that owns the object pointed to by \p p and uses \p d as the deleter.
   *  \param p A pointer to the object in device memory to manage.
   *  \param d The deleter to use.
   */
  template <bool Dummy = true, class = EnableIfDeleterConstructible<GoodRValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(pointer p, GoodRValRefType<Dummy> d) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(unique_ptr&& u) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(unique_ptr<U, E>&& u) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr& operator=(unique_ptr&& u) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr& operator=(unique_ptr<U, E>&& u) noexcept
  {
    reset(u.release());
    m_deleter = std::forward<E>(u.get_deleter());
    return *this;
  }

  /*! \brief Assigns a null pointer, deallocating the managed object. Effectively the same as calling reset().
   *  \return `*this`
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr& operator=(std::nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  //==========================================================================
  // Destructor
  //==========================================================================
  /*! \brief Destroys the \p unique_ptr, the managed object is destroyed via `get_deleter()(get())`. If get() == nullptr there are no effects.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 ~unique_ptr()
  {
    reset();
  }

  //==========================================================================
  // Observers
  //==========================================================================
  /*! \brief Returns a pointer to the managed object or `nullptr` if no object is owned.
   *  \return Pointer to the managed object.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 pointer get() const noexcept
  {
    return m_ptr;
  }

  /*! \brief Returns a raw pointer to the managed object or `nullptr` if no object is owned.
   *  \return Raw pointer to the managed object.
   */
  template <bool Dummy = true, class = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 T* get_raw() const noexcept
  {
    return raw_pointer_cast(m_ptr);
  }

  /*! \brief Returns a reference to the deleter object which would be used for destruction of the managed object.
   *  \return A reference to the stored deleter.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 deleter_type& get_deleter() noexcept
  {
    return m_deleter;
  }

  /*! \brief Returns a reference to the deleter object which would be used for destruction of the managed object.
   *  \return A reference to the stored deleter.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 const deleter_type& get_deleter() const noexcept
  {
    return m_deleter;
  }

  /*! \brief Checks if the \p unique_ptr owns an object.
   *
   *  Equivalent to `get() != nullptr`.
   *
   *  \return `true` if an object is owned, `false` otherwise.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 explicit operator bool() const noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 auto operator*() const noexcept
  {
    return *m_ptr;
  }

  /*! \brief Disabled for device pointers - cannot dereference device memory from host.
   *
   *  \warning This operator is intentionally disabled because dereferencing
   *  device memory from host code is invalid. Use `operator*()` to copy the
   *  object to host first (`T host_obj = *ptr`), or access members inside
   *  device code.
   *
   *  Attempting to use this operator will produce a compile-time error with
   *  a clear diagnostic message.
   *
   *  \return This function does not return; it triggers a compile-time error.
   */
  template <bool NotUsed = false>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 pointer operator->() const noexcept
  {
    static_assert(NotUsed,
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 pointer release() noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = pointer();
    return temp;
  }

  /*! \brief Replaces the managed object.
   *  \param p The new object to manage.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 void reset(pointer p = pointer()) noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = p;
    if (temp)
    {
      m_deleter(temp);
    }
  }

  /*! \brief Swaps the managed object and deleter with another \p unique_ptr.
   *
   *  Exchanges the contents of `*this` and \p u. This operation is `noexcept`
   *  and swaps both the stored pointer and the deleter.
   *
   *  \param u The \p unique_ptr to swap with.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 void swap(unique_ptr& u) noexcept
  {
    using std::swap;
    swap(m_ptr, u.m_ptr);
    swap(m_deleter, u.m_deleter);
  }
};

/*! \brief Array specialization of \p unique_ptr for managing dynamically-allocated arrays.
 *
 *  This specialization manages arrays allocated in device memory. The array form
 *  provides `operator[]` for element access.
 *
 *  \par Array Destruction Behavior
 *  The array form has different constructor requirements and performance characteristics
 *  based on the element type's destructibility:
 *
 *  **Trivially-destructible types** (e.g., `int`, `float`, POD types):
 *  - Constructors do NOT require array size parameter
 *  - Deleter is zero-size (no memory overhead)
 *  - Destruction only deallocates memory (no destructors to call)
 *  - Example: `unique_ptr<int[]> ptr(device_malloc<int>(100));`
 *
 *  **Non-trivially-destructible types** (e.g., types with custom destructors):
 *  - Constructors REQUIRE array size parameter
 *  - Deleter stores the size (small memory overhead: one `size_t`)
 *  - Destruction calls destructor for each element on device before deallocation
 *  - Example: `unique_ptr<MyClass[]> ptr(device_new<MyClass>(100), 100);`
 *
 *  This design ensures zero overhead for simple types while properly handling
 *  complex types that require cleanup.
 *
 *  \tparam T The array element type.
 *  \tparam D The type of the deleter.
 *
 *  \see thrust::default_delete
 *  \see thrust::make_unique
 *  \see https://en.cppreference.com/w/cpp/memory/unique_ptr
 */
template <class T, class D>
#if THRUST_HAS_ATTRIBUTE(trivial_abi)
class __attribute__((trivial_abi)) unique_ptr<T[], D>
#else
class unique_ptr<T[], D>
#endif
{
public:
  /*! \typedef pointer
   *  \brief The type of the stored pointer (defaults to `thrust::device_ptr<T>`).
   */
  using pointer = typename thrust::detail::pointer_detector<T, D>::type;

  /*! \typedef element_type
   *  \brief The type of the array elements (`T`).
   */
  using element_type = T;
  
  /*! \typedef deleter_type
   *  \brief The type of the deleter (`D`).
   */
  using deleter_type = D;

private:
  template <class Up, class OtherDeleter>
  friend class unique_ptr;

  pointer m_ptr;
#if THRUST_COMPILER(MSVC) && THRUST_HAS_CPP_ATTRIBUTE(msvc::no_unique_address)
  [[msvc::no_unique_address]] deleter_type m_deleter;
#elif THRUST_HAS_CPP_ATTRIBUTE(no_unique_address)
  [[no_unique_address]] deleter_type m_deleter;
#else
  deleter_type m_deleter;
#endif

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
  using EnableIfMoveConvertible = std::enable_if_t<
    std::is_array_v<Up> && std::is_same_v<pointer, element_type*> && std::is_same_v<typename UPtr::pointer, ElemT*>
    && std::is_convertible_v<ElemT (*)[], element_type (*)[]>>;

  template <class E>
  using EnableIfDeleterConvertible = std::enable_if_t<
    (std::is_reference_v<D> && std::is_same_v<D, E>) || (!std::is_reference_v<D> && std::is_convertible_v<E, D>)>;

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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 explicit unique_ptr(Pp p) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 explicit unique_ptr(Pp* raw_p) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 explicit unique_ptr(Pp p, size_t size) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 explicit unique_ptr(Pp* raw_p, size_t size) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(Pp p, LValRefType<Dummy> deleter) noexcept
      : m_ptr(p)
      , m_deleter(deleter)
  {}

  /*! \brief Constructs an empty \p unique_ptr with a custom deleter (lvalue reference).
   *
   *  \param deleter The deleter to store.
   */
  template <bool Dummy = true, class = EnableIfDeleterConstructible<LValRefType<Dummy>>>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(std::nullptr_t, LValRefType<Dummy> deleter) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(Pp p, GoodRValRefType<Dummy> deleter) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(std::nullptr_t, GoodRValRefType<Dummy> deleter) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(unique_ptr&& u) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr(unique_ptr<Up, Ep>&& u) noexcept
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
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr& operator=(unique_ptr&& p) noexcept
  {
    reset(p.release());
    m_deleter = std::forward<deleter_type>(p.get_deleter());
    return *this;
  }

  /*! \brief Converting assignment operator. Replaces the managed object with the one from \p p.
   *  \param p The \p unique_ptr to move from.
   *  \return `*this`
   */
  template <class Up,
            class Ep,
            class = EnableIfMoveConvertible<unique_ptr<Up, Ep>, Up>,
            class = EnableIfDeleterAssignable<Ep>>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr& operator=(unique_ptr<Up, Ep>&& p) noexcept
  {
    reset(p.release());
    m_deleter = std::forward<Ep>(p.get_deleter());
    return *this;
  }

  /*! \brief Assigns a null pointer, deallocating the managed object. Effectively the same as calling reset().
   *  \return `*this`
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr& operator=(std::nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  //==========================================================================
  // Destructor
  //==========================================================================
  /*! \brief Destroys the \p unique_ptr, the managed array is destroyed via `get_deleter()(get())`. If get() == nullptr there are no effects.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 ~unique_ptr()
  {
    reset();
  }

  //==========================================================================
  // Observers
  //==========================================================================
  /*! \brief Accesses an element of the managed array.
   *
   *  For device arrays, accessing an element from host code triggers a copy
   *  from device to host memory.
   *
   *  \warning Using this operator when iterating over the array is STRONGLY discouraged
   *  as the excessive number of single-element copies usually results in poor performance.
   *
   *  \param i The index of the element to access.
   *  \return A reference to (or copy of) the element at index \p i.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 auto operator[](size_t i) const noexcept
  {
    return m_ptr[i];
  }

  /*! \brief Returns a pointer to the managed array or `nullptr` if no object is owned.
   *  
   *  \return Pointer to the managed array.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 pointer get() const noexcept
  {
    return m_ptr;
  }

  /*! \brief Returns a raw pointer to the managed array or `nullptr` if no object is owned.
   *  
   *  \return Pointer to the managed array.
   */
  template <bool Dummy = true, class = EnableIfDeleterDefaultDelete<Dummy>>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 T* get_raw() const noexcept
  {
    return raw_pointer_cast(m_ptr);
  }

  /*! \brief Returns a reference to the deleter object which would be used for destruction of the managed array.
   *  \return A reference to the stored deleter.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 deleter_type& get_deleter() noexcept
  {
    return m_deleter;
  }

  /*! \brief Returns a const reference to the deleter object which would be used for destruction of the managed array.
   *  \return A const reference to the stored deleter.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 const deleter_type& get_deleter() const noexcept
  {
    return m_deleter;
  }

  /*! \brief Checks if the \p unique_ptr owns an array.
   *  \return `true` if an array is owned, `false` otherwise.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 explicit operator bool() const noexcept
  {
    return m_ptr != nullptr;
  }

  //==========================================================================
  // Modifiers
  //==========================================================================
  /*! \brief Releases ownership of the managed array, if any.
   *  \return A pointer to the released array.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 pointer release() noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = pointer();
    return temp;
  }

  /*! \brief Replaces the managed array.
   *  \param p The new array to manage.
   */
  template <class Pp, class = EnableIfPointerConvertible<Pp>>
  THRUST_HOST THRUST_CONSTEXPR_CXX23 void reset(Pp p) noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = p;
    if (temp)
    {
      m_deleter(temp);
    }
  }

  /*! \brief Replaces the managed array with \p nullptr.
   *
   *  Destroys the currently managed array (if any) and resets to empty state.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 void reset(std::nullptr_t = nullptr) noexcept
  {
    pointer temp = m_ptr;
    m_ptr        = nullptr;
    if (temp)
    {
      m_deleter(temp);
    }
  }

  /*! \brief Swaps the managed array and deleter with another array \p unique_ptr.
   *  \param u The \p unique_ptr to swap with.
   */
  THRUST_HOST THRUST_CONSTEXPR_CXX23 void swap(unique_ptr& u) noexcept
  {
    using std::swap;
    swap(m_ptr, u.m_ptr);
    swap(m_deleter, u.m_deleter);
  }
};

/*! \brief Swaps two \p unique_ptr objects.
 *
 *  Exchanges the contents of \p x and \p y by calling `x.swap(y)`.
 *
 *  \param x The first \p unique_ptr.
 *  \param y The second \p unique_ptr.
 */
template <class T, class D, std::enable_if_t<std::is_swappable_v<D>, void>>
inline THRUST_CONSTEXPR_CXX23 void swap(unique_ptr<T, D>& x, unique_ptr<T, D>& y) noexcept
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
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator==(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return x.get() == y.get();
}

#if THRUST_STD_VER <= 2017
template <class T1, class D1, class T2, class D2>
/*! \brief Compares two \p unique_ptr objects for inequality (C++17 and earlier).
 *
 *  \param x The first \p unique_ptr to compare.
 *  \param y The second \p unique_ptr to compare.
 *  \return `true` if the pointers are not equal, `false` otherwise.
 */
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator!=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
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
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator<(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  using P1 = typename unique_ptr<T1, D1>::element_type*;
  using P2 = typename unique_ptr<T2, D2>::element_type*;
  using CTP = typename std::common_type<P1, P2>::type;
  return std::less<CTP>()(thrust::raw_pointer_cast(x.get()), thrust::raw_pointer_cast(y.get()));
}

/*! \brief Compares two \p unique_ptr objects using greater-than ordering.
 *
 *  \param x The first \p unique_ptr to compare.
 *  \param y The second \p unique_ptr to compare.
 *  \return `true` if \p x is greater than \p y, `false` otherwise.
 */
template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator>(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return y < x;
}

/*! \brief Compares two \p unique_ptr objects using less-than-or-equal ordering.
 *
 *  \param x The first \p unique_ptr to compare.
 *  \param y The second \p unique_ptr to compare.
 *  \return `true` if \p x is less than or equal to \p y, `false` otherwise.
 */
template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator<=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(y < x);
}

/*! \brief Compares two \p unique_ptr objects using greater-than-or-equal ordering.
 *
 *  \param x The first \p unique_ptr to compare.
 *  \param y The second \p unique_ptr to compare.
 *  \return `true` if \p x is greater than or equal to \p y, `false` otherwise.
 */
template <class T1, class D1, class T2, class D2>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator>=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  return !(x < y);
}

#if THRUST_STD_VER >= 2020
template <class T1, class D1, class T2, class D2>
  THRUST_HOST inline auto operator<=> (const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y)
{
  // TODO: once thrust::device_ptr supports three_way_comparison, we should be using that
  return std::compare_three_way()(x.get_raw(), y.get_raw());
}
#endif

/*! \brief Compares a \p unique_ptr with nullptr for equality.
 *
 *  \param x The \p unique_ptr to compare.
 *  \return `true` if \p x is empty, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator==(const unique_ptr<T, D>& x, std::nullptr_t) noexcept
{
  return !x;
}

#if THRUST_STD_VER <= 2017
/*! \brief Compares nullptr with a \p unique_ptr for equality (C++17 and earlier).
 *
 *  \param y The \p unique_ptr to compare.
 *  \return `true` if \p y is empty, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator==(std::nullptr_t, const unique_ptr<T, D>& y) noexcept
{
  return !y;
}

/*! \brief Compares a \p unique_ptr with nullptr for inequality (C++17 and earlier).
 *
 *  \param x The \p unique_ptr to compare.
 *  \return `true` if \p x is not empty, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator!=(const unique_ptr<T, D>& x, std::nullptr_t) noexcept
{
  return static_cast<bool>(x);
}

/*! \brief Compares nullptr with a \p unique_ptr for inequality (C++17 and earlier).
 *
 *  \param y The \p unique_ptr to compare.
 *  \return `true` if \p y is not empty, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator!=(std::nullptr_t, const unique_ptr<T, D>& y) noexcept
{
  return static_cast<bool>(y);
}
#endif

/*! \brief Compares a \p unique_ptr with nullptr using less-than ordering.
 *
 *  \param x The \p unique_ptr to compare.
 *  \return `true` if the pointer in \p x is less than nullptr, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator<(const unique_ptr<T, D>& x, std::nullptr_t)
{
  return std::less<typename unique_ptr<T, D>::pointer>()(x.get(), nullptr); // x.get() < nullptr;
}

/*! \brief Compares nullptr with a \p unique_ptr using less-than ordering.
 *
 *  \param y The \p unique_ptr to compare.
 *  \return `true` if nullptr is less than the pointer in \p y, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator<(std::nullptr_t, const unique_ptr<T, D>& y)
{
  return std::less<typename unique_ptr<T, D>::pointer>()(nullptr, y.get()); // nullptr < y.get();
}

/*! \brief Compares a \p unique_ptr with nullptr using greater-than ordering.
 *
 *  \param x The \p unique_ptr to compare.
 *  \return `true` if the pointer in \p x is greater than nullptr, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator>(const unique_ptr<T, D>& x, std::nullptr_t)
{
  return nullptr < x;
}

/*! \brief Compares nullptr with a \p unique_ptr using greater-than ordering.
 *
 *  \param y The \p unique_ptr to compare.
 *  \return `true` if nullptr is greater than the pointer in \p y, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator>(std::nullptr_t, const unique_ptr<T, D>& y)
{
  return y < nullptr;
}

/*! \brief Compares a \p unique_ptr with nullptr using less-than-or-equal ordering.
 *
 *  \param x The \p unique_ptr to compare.
 *  \return `true` if the pointer in \p x is less than or equal to nullptr, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator<=(const unique_ptr<T, D>& x, std::nullptr_t)
{
  return !(nullptr < x);
}

/*! \brief Compares nullptr with a \p unique_ptr using less-than-or-equal ordering.
 *
 *  \param y The \p unique_ptr to compare.
 *  \return `true` if nullptr is less than or equal to the pointer in \p y, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator<=(std::nullptr_t, const unique_ptr<T, D>& y)
{
  return !(y < nullptr);
}

/*! \brief Compares a \p unique_ptr with nullptr using greater-than-or-equal ordering.
 *
 *  \param x The \p unique_ptr to compare.
 *  \return `true` if the pointer in \p x is greater than or equal to nullptr, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator>=(const unique_ptr<T, D>& x, std::nullptr_t)
{
  return !(x < nullptr);
}

/*! \brief Compares nullptr with a \p unique_ptr using greater-than-or-equal ordering.
 *
 *  \param y The \p unique_ptr to compare.
 *  \return `true` if nullptr is greater than or equal to the pointer in \p y, `false` otherwise.
 */
template <class T, class D>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 bool operator>=(std::nullptr_t, const unique_ptr<T, D>& y)
{
  return !(y < nullptr);
}

#if THRUST_STD_VER >= 2020
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
 *  Allocates device memory for a single object of type \p T, direct-initializes the object
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
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 unique_ptr<T> make_unique(Args&&... args)
{
  thrust::device_ptr<T> p = thrust::device_malloc<T>(1);
  return unique_ptr<T>(thrust::device_new<T>(p, T(std::forward<Args>(args)...), 1));
}

/*! \brief Constructs an array of objects of type \p T in device memory and wraps it in a \p unique_ptr.
 *
 *  Allocates device memory for an array of \p n objects of type \p U (where \p T is \p U[]),
 *  default-initializes each element, and returns a \p unique_ptr managing the allocated array.
 *
 *  This overload participates in overload resolution only if \p T is an array of unknown
 *  bound (e.g., \p T[]).
 *
 *  \tparam T The array type (e.g., \p int[], \p MyClass[]).
 *  \param n The number of elements in the array.
 *  \return A \p unique_ptr<T> managing the newly created array.
 */
template <class T, class = std::enable_if_t<thrust::detail::is_unbounded_array<T>::value>>
THRUST_HOST inline THRUST_CONSTEXPR_CXX23 unique_ptr<T> make_unique(size_t n)
{
  using U = typename std::remove_extent<T>::type;
  return unique_ptr<T>(thrust::device_new<U>(n), n);
}

template <class T, class... Args, class = std::enable_if_t<thrust::detail::is_bounded_array<T>::value>>
THRUST_HOST void make_unique(Args&&...) = delete;

#if THRUST_STD_VER >= 2020

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
THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr<T> make_unique_for_overwrite()
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
THRUST_HOST THRUST_CONSTEXPR_CXX23 unique_ptr<T> make_unique_for_overwrite(size_t n)
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
