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

// Portions of this code are derived from
//
// Manjunath Kudlur's Carbon library
//
// and
//
// Based on Boost.Phoenix v1.2
// Copyright (c) 2001-2002 Joel de Guzman

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/functional/actor.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace functional
{

template <typename Eval>
struct actor;

template <typename T>
class value
{
public:
  template <typename Env>
  struct result
  {
    using type = T;
  };

  THRUST_HOST_DEVICE value(const T& arg)
      : m_val(arg)
  {}

  template <typename Env>
  THRUST_HOST_DEVICE T eval(const Env&) const
  {
    return m_val;
  }

private:
  T m_val;
}; // end value

template <typename T>
THRUST_HOST_DEVICE actor<value<T>> val(const T& x)
{
  return value<T>(x);
} // end val()

} // namespace functional
} // namespace detail
THRUST_NAMESPACE_END
