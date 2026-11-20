// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <nanobind/nanobind.h>

namespace hipdnn_frontend::graph
{
class Graph;
}

void graphBindings(nanobind::module_& m);
void tensorBindings(nanobind::module_& m);
void attributesBindings(nanobind::module_& m);
void typesBindings(nanobind::module_& m);
void autotuneBindings(nanobind::module_& m);
void handleBindings(nanobind::module_& m);
void memoryBindings(nanobind::module_& m);
void hipBindings(nanobind::module_& m);
void pygraphBindings(nanobind::module_& m);

// Registered on hipdnn.Graph after every existing overload, so existing call
// shapes keep resolving to the existing overloads first.
void graphKwargsBindings(nanobind::class_<hipdnn_frontend::graph::Graph>& cls);
void pointwiseBindings(nanobind::class_<hipdnn_frontend::graph::Graph>& cls);
