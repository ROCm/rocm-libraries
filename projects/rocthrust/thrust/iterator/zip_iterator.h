/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

//! \file thrust/iterator/zip_iterator.h
//! \brief An iterator which returns a tuple of the result of dereferencing a tuple of iterators when dereferenced

/*
 * Copyright David Abrahams and Thomas Becker 2000-2006.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

// TODO(libhipcxx): remove all the code in the path of !_THRUST_HAS_DEVICE_SYSTEM_STD
// once libhipcxx gets ready

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/advance.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>
#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <thrust/detail/tuple_meta_transform.h>
#  include <thrust/detail/tuple_transform.h>
#  include <thrust/iterator/iterator_categories.h>
#  include <thrust/tuple.h>
#  include <thrust/type_traits/integer_sequence.h>
#endif

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(tuple)
#endif

THRUST_NAMESPACE_BEGIN

template <typename IteratorTuple>
class zip_iterator;

namespace detail
{
template <typename... Ts>
using minimum_category = minimum_type<Ts...>;

#if _THRUST_HAS_DEVICE_SYSTEM_STD
template <typename IteratorTuple>
struct make_zip_iterator_base
{
  static_assert(!sizeof(IteratorTuple), "thrust::zip_iterator only supports cuda::std::tuple");
};

template <typename... Its>
struct make_zip_iterator_base<_THRUST_STD::tuple<Its...>>
{
  // reference type is the type of the tuple obtained from the iterators' reference types.
  using reference = tuple_of_iterator_references<iterator_reference_t<Its>...>;

  // Boost's Value type is the same as reference type. using value_type = reference;
  using value_type = _THRUST_STD::tuple<iterator_value_t<Its>...>;

  // Difference type is the first iterator's difference type
  using difference_type = iterator_difference_t<_THRUST_STD::tuple_element_t<0, _THRUST_STD::tuple<Its...>>>;

  // Iterator system is the minimum system tag in the iterator tuple
  using system = _THRUST_STD::__type_fold_left<_THRUST_STD::__type_list<iterator_system_t<Its>...>,
                                               any_system_tag,
                                               _THRUST_STD::__type_quote_trait<minimum_system>>;

  // Traversal category is the minimum traversal category in the iterator tuple
  using traversal_category =
    _THRUST_STD::__type_fold_left<_THRUST_STD::__type_list<iterator_traversal_t<Its>...>,
                                  random_access_traversal_tag,
                                  _THRUST_STD::__type_quote_trait<minimum_category>>;

  // The iterator facade type from which the zip iterator will be derived.
  using type =
    iterator_facade<zip_iterator<_THRUST_STD::tuple<Its...>>,
                    value_type,
                    system,
                    traversal_category,
                    reference,
                    difference_type>;
};
#else
// Functors to be used with tuple algorithms
//
template <typename DiffType>
class advance_iterator
{
public:
  inline THRUST_HOST_DEVICE advance_iterator(DiffType step)
      : m_step(step)
  {}

  THRUST_EXEC_CHECK_DISABLE
  template <typename Iterator>
  inline THRUST_HOST_DEVICE void operator()(Iterator& it) const
  {
    thrust::advance(it, m_step);
  }

private:
  DiffType m_step;
}; // end advance_iterator

struct increment_iterator
{
  THRUST_EXEC_CHECK_DISABLE
  template <typename Iterator>
  inline THRUST_HOST_DEVICE void operator()(Iterator& it)
  {
    ++it;
  }
}; // end increment_iterator

struct decrement_iterator
{
  THRUST_EXEC_CHECK_DISABLE
  template <typename Iterator>
  inline THRUST_HOST_DEVICE void operator()(Iterator& it)
  {
    --it;
  }
}; // end decrement_iterator

struct dereference_iterator
{
  template <typename Iterator>
  struct apply
  {
    using type = typename iterator_traits<Iterator>::reference;
  }; // end apply

  // XXX silence warnings of the form "calling a __host__ function from a __host__ __device__ function is not allowed
  THRUST_EXEC_CHECK_DISABLE
  template <typename Iterator>
  THRUST_HOST_DEVICE typename apply<Iterator>::type operator()(Iterator const& it)
  {
    return *it;
  }
}; // end dereference_iterator

// The namespace tuple_impl_specific provides two meta-
// algorithms and two algorithms for tuples.
namespace tuple_impl_specific
{

// define apply1 for tuple_meta_transform_impl
template <typename UnaryMetaFunctionClass, class Arg>
struct apply1 : UnaryMetaFunctionClass::template apply<Arg>
{}; // end apply1

// define apply2 for tuple_meta_accumulate_impl
template <typename UnaryMetaFunctionClass, class Arg1, class Arg2>
struct apply2 : UnaryMetaFunctionClass::template apply<Arg1, Arg2>
{}; // end apply2

// Meta-accumulate algorithm for tuples. Note: The template
// parameter StartType corresponds to the initial value in
// ordinary accumulation.
//
template <class Tuple, class BinaryMetaFun, class StartType>
struct tuple_meta_accumulate;

template <class BinaryMetaFun, typename StartType>
struct tuple_meta_accumulate<thrust::tuple<>, BinaryMetaFun, StartType>
{
  using type = typename thrust::detail::identity_<StartType>::type;
};

template <class BinaryMetaFun, typename StartType, typename T, typename... Ts>
struct tuple_meta_accumulate<thrust::tuple<T, Ts...>, BinaryMetaFun, StartType>
{
  using type =
    typename apply2<BinaryMetaFun,
                    T,
                    typename tuple_meta_accumulate<thrust::tuple<Ts...>, BinaryMetaFun, StartType>::type>::type;
};

template <typename Fun>
inline THRUST_HOST_DEVICE Fun tuple_for_each_helper(Fun f)
{
  return f;
}

template <typename Fun, typename T, typename... Ts>
inline THRUST_HOST_DEVICE Fun tuple_for_each_helper(Fun f, T& t, Ts&... ts)
{
  f(t);
  return tuple_for_each_helper(f, ts...);
}

// for_each algorithm for tuples.

template <typename Fun, typename... Ts, size_t... Is>
inline THRUST_HOST_DEVICE Fun tuple_for_each(thrust::tuple<Ts...>& t, Fun f, thrust::index_sequence<Is...>)
{
  return tuple_for_each_helper(f, thrust::get<Is>(t)...);
} // end tuple_for_each()

// for_each algorithm for tuples.
template <typename Fun, typename... Ts>
inline THRUST_HOST_DEVICE Fun tuple_for_each(thrust::tuple<Ts...>& t, Fun f)
{
  return tuple_for_each(t, f, thrust::make_index_sequence<thrust::tuple_size<thrust::tuple<Ts...>>::value>{});
}

} // namespace tuple_impl_specific

// Metafunction to obtain the type of the tuple whose element types
// are the value_types of an iterator tuple.
//
template <typename IteratorTuple>
struct tuple_of_value_types : tuple_meta_transform<IteratorTuple, iterator_value>
{}; // end tuple_of_value_types

struct minimum_category_lambda
{
  template <typename T1, typename T2>
  struct apply : minimum_category<T1, T2>
  {};
};

// Metafunction to obtain the minimal traversal tag in a tuple
// of iterators.
//
template <typename IteratorTuple>
struct minimum_traversal_category_in_iterator_tuple
{
  using tuple_of_traversal_tags = typename tuple_meta_transform<IteratorTuple, thrust::iterator_traversal>::type;

  using type = typename tuple_impl_specific::
    tuple_meta_accumulate<tuple_of_traversal_tags, minimum_category_lambda, thrust::random_access_traversal_tag>::type;
};

struct minimum_system_lambda
{
  template <typename T1, typename T2>
  struct apply : minimum_system<T1, T2>
  {};
};

// Metafunction to obtain the minimal system tag in a tuple
// of iterators.
template <typename IteratorTuple>
struct minimum_system_in_iterator_tuple
{
  using tuple_of_system_tags =
    typename thrust::detail::tuple_meta_transform<IteratorTuple, thrust::iterator_system>::type;

  using type = typename tuple_impl_specific::
    tuple_meta_accumulate<tuple_of_system_tags, minimum_system_lambda, thrust::any_system_tag>::type;
};

namespace zip_iterator_base_ns
{

template <typename Tuple, typename IndexSequence>
struct tuple_of_iterator_references_helper;

template <typename Tuple, size_t... Is>
struct tuple_of_iterator_references_helper<Tuple, thrust::index_sequence<Is...>>
{
  using type = thrust::detail::tuple_of_iterator_references<typename thrust::tuple_element<Is, Tuple>::type...>;
};

template <typename IteratorTuple>
struct tuple_of_iterator_references
{
  // get a thrust::tuple of the iterators' references
  using tuple_of_references = typename tuple_meta_transform<IteratorTuple, thrust::iterator_reference>::type;

  // map thrust::tuple<T...> to tuple_of_iterator_references<T...>
  using type = typename tuple_of_iterator_references_helper<
    tuple_of_references,
    thrust::make_index_sequence<thrust::tuple_size<tuple_of_references>::value>>::type;
};

} // namespace zip_iterator_base_ns

///////////////////////////////////////////////////////////////////
//
// Class make_zip_iterator_base
//
// Builds and exposes the iterator facade type from which the zip
// iterator will be derived.
//
template <typename IteratorTuple>
struct make_zip_iterator_base
{
  // private:
  //  reference type is the type of the tuple obtained from the
  //  iterators' reference types.
  using reference = typename zip_iterator_base_ns::tuple_of_iterator_references<IteratorTuple>::type;

  // Boost's Value type is the same as reference type.
  // using value_type = reference;
  using value_type = typename tuple_of_value_types<IteratorTuple>::type;

  // Difference type is the first iterator's difference type
  using difference_type =
    typename thrust::iterator_traits<typename thrust::tuple_element<0, IteratorTuple>::type>::difference_type;

  // Iterator system is the minimum system tag in the
  // iterator tuple
  using system = typename minimum_system_in_iterator_tuple<IteratorTuple>::type;

  // Traversal category is the minimum traversal category in the
  // iterator tuple
  using traversal_category = typename minimum_traversal_category_in_iterator_tuple<IteratorTuple>::type;

public:
  // The iterator facade type from which the zip iterator will
  // be derived.
  using type = thrust::
    iterator_facade<zip_iterator<IteratorTuple>, value_type, system, traversal_category, reference, difference_type>;
}; // end make_zip_iterator_base
#endif
} // namespace detail

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p zip_iterator is an iterator which represents a pointer into a range of \p tuples whose elements are themselves
//! taken from a \p tuple of input iterators. This iterator is useful for creating a virtual array of structures while
//! achieving the same performance and bandwidth as the structure of arrays idiom. \p zip_iterator also facilitates
//! kernel fusion by providing a convenient means of amortizing the execution of the same operation over multiple
//! ranges.
//!
//! The following code snippet demonstrates how to create a \p zip_iterator which represents the result of "zipping"
//! multiple ranges together.
//!
//! \code
//! #include <thrust/iterator/zip_iterator.h>
//! #include <thrust/tuple.h>
//! #include <thrust/device_vector.h>
//! ...
//! thrust::device_vector<int> int_v{0, 1, 2};
//! thrust::device_vector<float> float_v{0.0f, 1.0f, 2.0f};
//! thrust::device_vector<char> char_v{'a', 'b', 'c'};
//!
//! // aliases for iterators
//! using IntIterator = thrust::device_vector<int>::iterator;
//! using FloatIterator = thrust::device_vector<float>::iterator;
//! using CharIterator = thrust::device_vector<char>::iterator;
//!
//! // alias for a tuple of these iterators
//! using IteratorTuple = thrust::tuple<IntIterator, FloatIterator, CharIterator>;
//!
//! // alias the zip_iterator of this tuple
//! using ZipIterator = thrust::zip_iterator<IteratorTuple>;
//!
//! // finally, create the zip_iterator
//! ZipIterator iter(thrust::make_tuple(int_v.begin(), float_v.begin(), char_v.begin()));
//!
//! *iter;   // returns (0, 0.0f, 'a')
//! iter[0]; // returns (0, 0.0f, 'a')
//! iter[1]; // returns (1, 1.0f, 'b')
//! iter[2]; // returns (2, 2.0f, 'c')
//!
//! thrust::get<0>(iter[2]); // returns 2
//! thrust::get<1>(iter[0]); // returns 0.0f
//! thrust::get<2>(iter[1]); // returns 'b'
//!
//! // iter[3] is an out-of-bounds error
//! \endcode
//!
//! Defining the type of a \p zip_iterator can be complex. The next code example demonstrates how to use the \p
//! make_zip_iterator function with the \p make_tuple function to avoid explicitly specifying the type of the \p
//! zip_iterator. This example shows how to use \p zip_iterator to copy multiple ranges with a single call to \p
//! thrust::copy.
//!
//! \code
//! #include <thrust/zip_iterator.h>
//! #include <thrust/tuple.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   thrust::device_vector<int> int_in{0, 1, 2}, int_out(3);
//!   thrust::device_vector<float> float_in{0.0f, 10.0f, 20.0f}, float_out(3);
//!
//!   thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(int_in.begin(), float_in.begin())),
//!                thrust::make_zip_iterator(thrust::make_tuple(int_in.end(),   float_in.end())),
//!                thrust::make_zip_iterator(thrust::make_tuple(int_out.begin(),float_out.begin())));
//!
//!   // int_out is now [0, 1, 2]
//!   // float_out is now [0.0f, 10.0f, 20.0f]
//!
//!   return 0;
//! }
//! \endcode
//!
//! \see make_zip_iterator
//! \see make_tuple
//! \see tuple
//! \see get
template <typename IteratorTuple>
class THRUST_DECLSPEC_EMPTY_BASES zip_iterator : public detail::make_zip_iterator_base<IteratorTuple>::type
{
public:
  //! The underlying iterator tuple type. Alias to zip_iterator's first template argument.
  using iterator_tuple = IteratorTuple;

  //! Default constructor does nothing.
  zip_iterator() = default;

  //! This constructor creates a new \p zip_iterator from a \p tuple of iterators.
  //!
  //! \param iterator_tuple The \p tuple of iterators to copy from.
  inline THRUST_HOST_DEVICE zip_iterator(IteratorTuple iterator_tuple)
      : m_iterator_tuple(iterator_tuple)
  {}

  //! This copy constructor creates a new \p zip_iterator from another \p zip_iterator.
  //!
  //! \param other The \p zip_iterator to copy.
  template <typename OtherIteratorTuple, detail::enable_if_convertible_t<OtherIteratorTuple, IteratorTuple, int> = 0>
  inline THRUST_HOST_DEVICE zip_iterator(const zip_iterator<OtherIteratorTuple>& other)
      : m_iterator_tuple(other.get_iterator_tuple())
  {}

  //! This method returns a \c const reference to this \p zip_iterator's
  //! \p tuple of iterators.
  //!
  //! \return A \c const reference to this \p zip_iterator's \p tuple  of iterators.
  inline THRUST_HOST_DEVICE const IteratorTuple& get_iterator_tuple() const
  {
    return m_iterator_tuple;
  }

  //! \cond

private:
  using super_t = typename detail::make_zip_iterator_base<IteratorTuple>::type;

  friend class iterator_core_access;

#if _THRUST_HAS_DEVICE_SYSTEM_STD
  using index_seq = make_index_sequence<_THRUST_STD::tuple_size_v<IteratorTuple>>;

  THRUST_EXEC_CHECK_DISABLE
  template <size_t... Is>
  THRUST_HOST_DEVICE typename super_t::reference dereference_impl(index_sequence<Is...>) const
  {
    return {*_THRUST_STD::get<Is>(m_iterator_tuple)...};
  }
#endif

  // Dereferencing returns a tuple built from the dereferenced iterators in the iterator tuple.
  THRUST_HOST_DEVICE typename super_t::reference dereference() const
  {
#if _THRUST_HAS_DEVICE_SYSTEM_STD
    return dereference_impl(index_seq{});
#else
    using namespace detail::tuple_impl_specific;

    return thrust::detail::tuple_host_device_transform<detail::dereference_iterator::template apply>(
      get_iterator_tuple(), detail::dereference_iterator());
#endif
  }

  // Two zip_iterators are equal if the two first iterators of the tuple are equal. Note this differs from Boost's
  // implementation, which considers the entire tuple.
  THRUST_EXEC_CHECK_DISABLE
  template <typename OtherIteratorTuple>
  inline THRUST_HOST_DEVICE bool equal(const zip_iterator<OtherIteratorTuple>& other) const
  {
    return get<0>(get_iterator_tuple()) == get<0>(other.get_iterator_tuple());
  }

#if _THRUST_HAS_DEVICE_SYSTEM_STD
  THRUST_EXEC_CHECK_DISABLE
  template <size_t... Is>
  inline THRUST_HOST_DEVICE void advance_impl(typename super_t::difference_type n, index_sequence<Is...>)
  {
    (..., thrust::advance(_THRUST_STD::get<Is>(m_iterator_tuple), n));
  }
#endif

  // Advancing a zip_iterator means to advance all iterators in the tuple
  inline THRUST_HOST_DEVICE void advance(typename super_t::difference_type n)
  {
#if _THRUST_HAS_DEVICE_SYSTEM_STD
    advance_impl(n, index_seq{});
#else
    using namespace detail::tuple_impl_specific;
    tuple_for_each(m_iterator_tuple, detail::advance_iterator<typename super_t::difference_type>(n));
#endif
  }

#if _THRUST_HAS_DEVICE_SYSTEM_STD
  THRUST_EXEC_CHECK_DISABLE
  template <size_t... Is>
  inline THRUST_HOST_DEVICE void increment_impl(index_sequence<Is...>)
  {
    (..., ++_THRUST_STD::get<Is>(m_iterator_tuple));
  }
#endif

  // Incrementing a zip iterator means to increment all iterators in the tuple
  inline THRUST_HOST_DEVICE void increment()
  {
#if _THRUST_HAS_DEVICE_SYSTEM_STD
    increment_impl(index_seq{});
#else
    using namespace detail::tuple_impl_specific;
    tuple_for_each(m_iterator_tuple, detail::increment_iterator());
#endif
  }

#if _THRUST_HAS_DEVICE_SYSTEM_STD
  THRUST_EXEC_CHECK_DISABLE
  template <size_t... Is>
  inline THRUST_HOST_DEVICE void decrement_impl(index_sequence<Is...>)
  {
    (..., --_THRUST_STD::get<Is>(m_iterator_tuple));
  }
#endif

  // Decrementing a zip iterator means to decrement all iterators in the tuple
  inline THRUST_HOST_DEVICE void decrement()
  {
#if _THRUST_HAS_DEVICE_SYSTEM_STD
    decrement_impl(index_seq{});
#else
    using namespace detail::tuple_impl_specific;
    tuple_for_each(m_iterator_tuple, detail::decrement_iterator());
#endif
  }

  // Distance is calculated using the first iterator in the tuple.
  template <typename OtherIteratorTuple>
  inline THRUST_HOST_DEVICE typename super_t::difference_type
  distance_to(const zip_iterator<OtherIteratorTuple>& other) const
  {
    return get<0>(other.get_iterator_tuple()) - get<0>(get_iterator_tuple());
  }

  // The iterator tuple.
  IteratorTuple m_iterator_tuple;

  //! \endcond
};

//! \p make_zip_iterator creates a \p zip_iterator from a \p tuple of iterators.
//!
//! \param t The \p tuple of iterators to copy.
//! \return A newly created \p zip_iterator which zips the iterators encapsulated in \p t.
//! \see zip_iterator
template <typename... Iterators>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
inline THRUST_HOST_DEVICE zip_iterator<_THRUST_STD::tuple<Iterators...>>
make_zip_iterator(_THRUST_STD::tuple<Iterators...> t)
#else
inline THRUST_HOST_DEVICE zip_iterator<thrust::tuple<Iterators...>> make_zip_iterator(thrust::tuple<Iterators...> t)
#endif
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  return zip_iterator<_THRUST_STD::tuple<Iterators...>>(t);
#else
  return zip_iterator<thrust::tuple<Iterators...>>(t);
#endif
}

//! \p make_zip_iterator creates a \p zip_iterator from
//! iterators.
//!
//! \param its The iterators to copy.
//! \return A newly created \p zip_iterator which zips the iterators.
//!
//! \see zip_iterator
template <typename... Iterators>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
inline THRUST_HOST_DEVICE zip_iterator<_THRUST_STD::tuple<Iterators...>> make_zip_iterator(Iterators... its)
#else
inline THRUST_HOST_DEVICE zip_iterator<thrust::tuple<Iterators...>> make_zip_iterator(Iterators... its)
#endif
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  return make_zip_iterator(_THRUST_STD::make_tuple(its...));
#else
  return make_zip_iterator(thrust::make_tuple(its...));
#endif
}

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
