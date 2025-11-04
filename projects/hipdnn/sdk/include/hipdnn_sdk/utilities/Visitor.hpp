// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace hipdnn_sdk::utilities
{

// Struct is initialized with a list of functors
// Typically passed to std:visit to provide the overloads for different
// variant values
template <class... Ts>
struct Visitor : Ts...
{
    using Ts::operator()...;
};

template <class... Ts>
Visitor(Ts...) -> Visitor<Ts...>;
}
