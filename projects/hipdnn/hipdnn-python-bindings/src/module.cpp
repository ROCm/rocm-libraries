// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declarations for binding functions
void graph_bindings(nb::module_& m);
void tensor_bindings(nb::module_& m);
void attributes_bindings(nb::module_& m);
void types_bindings(nb::module_& m);

NB_MODULE(hipdnn_frontend, m)
{
    m.doc() = "Python bindings for the hipDNN frontend library";

    // Initialize bindings from other files
    types_bindings(m); // Types and enums first
    tensor_bindings(m); // Then tensor attributes
    attributes_bindings(m); // Then node attributes
    graph_bindings(m); // Finally graph which uses all of the above
}
