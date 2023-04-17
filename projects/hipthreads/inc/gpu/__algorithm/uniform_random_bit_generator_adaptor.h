//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_RANGES_UNIFORM_RANDOM_BIT_GENERATOR_ADAPTOR_H
#define __GPU___ALGORITHM_RANGES_UNIFORM_RANDOM_BIT_GENERATOR_ADAPTOR_H

#include "gpu/__config"

namespace gpu {

// Range versions of random algorithms (e.g. `gpu::shuffle`) are less constrained than their classic counterparts.
// Range algorithms only require the given generator to satisfy the `std::uniform_random_bit_generator` concept.
// Classic algorithms require the given generator to meet the uniform random bit generator requirements; these
// requirements include satisfying `std::uniform_random_bit_generator` and add a requirement for the generator to
// provide a nested `result_type` typedef (see `[rand.req.urng]`).
//
// To be able to reuse classic implementations, make the given generator meet the classic requirements by wrapping
// it into an adaptor type that forwards all of its interface and adds the required typedef.
template <class _Gen>
class _ClassicGenAdaptor {
private:
  // The generator is not required to be copyable or movable, so it has to be stored as a reference.
  _Gen& __gen_;

public:
  using result_type = std::invoke_result_t<_Gen&>;

  _LIBGPU_HIDE_FROM_ABI
  static constexpr auto min() { return std::remove_cv_t<std::remove_reference_t<_Gen>>::min(); }
  _LIBGPU_HIDE_FROM_ABI
  static constexpr auto max() { return std::remove_cv_t<std::remove_reference_t<_Gen>>::max(); }

  _LIBGPU_HIDE_FROM_ABI
  constexpr explicit _ClassicGenAdaptor(_Gen& __g) : __gen_(__g) {}

  _LIBGPU_HIDE_FROM_ABI
  constexpr auto operator()() const { return __gen_(); }
};

} // namespace gpu

#endif // __GPU___ALGORITHM_RANGES_UNIFORM_RANDOM_BIT_GENERATOR_ADAPTOR_H
