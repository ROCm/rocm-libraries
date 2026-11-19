// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Binding-local ownership of stack builders through Python exceptions. */
#ifndef ROCKE_BINDINGS_BUILDER_LIFETIME_HPP
#define ROCKE_BINDINGS_BUILDER_LIFETIME_HPP

#include <memory>

namespace rocke_bindings
{
// Own teardown of a stack-allocated builder after a successful kernel build.
// The deleter frees the builder's resources, never the stack object itself.
template <typename Builder>
auto own_builder(Builder* builder, void (*free_builder)(Builder*))
{
    return std::unique_ptr<Builder, decltype(free_builder)>(builder, free_builder);
}
} // namespace rocke_bindings

#endif // ROCKE_BINDINGS_BUILDER_LIFETIME_HPP
