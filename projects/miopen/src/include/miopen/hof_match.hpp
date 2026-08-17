// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MLOPEN_HOF_MATCH_HPP
#define GUARD_MLOPEN_HOF_MATCH_HPP

#include <type_traits>
#include <utility>

namespace miopen {

template <typename... Fs>
struct hof_match : Fs...
{
    using Fs::operator()...;

    constexpr explicit hof_match(Fs&&... fs) : Fs(std::move(fs))... {}
};

template <typename... Fs>
hof_match(Fs...) -> hof_match<std::decay_t<Fs>...>;

} // namespace miopen

#endif
