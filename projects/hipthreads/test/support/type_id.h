//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TYPE_ID_H
#define SUPPORT_TYPE_ID_H

#include <string>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER < 11
#error This header requires C++11 or greater
#endif

// TypeID - Represent a unique identifier for a type. TypeID allows equality
// comparisons between different types.
struct TypeID {
  __host__ __device__ friend bool operator==(TypeID const& LHS, TypeID const& RHS)
  {return LHS.m_id == RHS.m_id; }
  __host__ __device__ friend bool operator!=(TypeID const& LHS, TypeID const& RHS)
  {return LHS.m_id != RHS.m_id; }

  __host__ __device__ ::std::string name() const {
    return m_id;
  }

private:
  __host__ __device__ explicit constexpr TypeID(const char* xid) : m_id(xid) {}

  __host__ __device__ TypeID(const TypeID&) = delete;
  __host__ __device__ TypeID& operator=(TypeID const&) = delete;

  const char* const m_id;
  template <class T> __host__ friend TypeID const& makeTypeIDImp();
  template <class T> __device__ friend TypeID const& makeTypeIDImp();
};

// makeTypeID - Return the TypeID for the specified type 'T'.
template <class T>
__host__ inline TypeID const& makeTypeIDImp() {
#ifdef _MSC_VER
  static const TypeID id(__FUNCSIG__);
#else
  static const TypeID id(__PRETTY_FUNCTION__);
#endif // _MSC_VER
  return id;
}

// makeTypeID - Return the TypeID for the specified type 'T'.
template <class T>
__device__ inline TypeID const& makeTypeIDImp() {
#ifdef _MSC_VER
  static __device__ const TypeID id(__FUNCSIG__);
#else
  static __device__ const TypeID id(__PRETTY_FUNCTION__);
#endif // _MSC_VER
  return id;
}

template <class T>
struct TypeWrapper {};

template <class T>
__host__ __device__ inline TypeID const& makeTypeID() {
  return makeTypeIDImp<TypeWrapper<T>>();
}

template <class ...Args>
struct ArgumentListID {};

// makeArgumentID - Create and return a unique identifier for a given set
// of arguments.
template <class ...Args>
__host__ __device__ inline  TypeID const& makeArgumentID() {
  return makeTypeIDImp<ArgumentListID<Args...>>();
}

#endif // SUPPORT_TYPE_ID_H
