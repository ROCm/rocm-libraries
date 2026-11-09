// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/portable_ir/malformed_param_attrs.h -- the malformed attrs of a pointer
 * param 'P' that both portable-IR importers refuse, each with the one text
 * rocke_param_opts_set_attr gives. dom_decoders checks the concrete-IR importer
 * and recipe_vm_replay the recipe VM against this table, so the two doors
 * cannot drift apart.
 */
#ifndef ROCKE_TESTS_PORTABLE_IR_MALFORMED_PARAM_ATTRS_H
#define ROCKE_TESTS_PORTABLE_IR_MALFORMED_PARAM_ATTRS_H

#include "rocke/ir.h"

struct malformed_param_attr_t
{
    const char* what;
    const char* attrs; /* the param's "attrs" value, as JSON */
    const char* message;
};

inline const malformed_param_attr_t kMalformedParamAttrs[] = {
    {"unknown key",
     R"json({"el": 1048576})json",
     "param 'P' has unknown attr 'el' -- a param attr is one of " ROCKE_PARAM_ATTR_KEYS},
    {"non-bool flag", R"json({"nonnull": 1})json", "param 'P' attr nonnull is not a bool"},
    {"non-integral align", R"json({"align": "16"})json", "param 'P' align is not an integer"},
    {"align outside the int range",
     R"json({"align": 4294967296})json",
     "param 'P' align is outside the int range"},
    {"non-integral dereferenceable",
     R"json({"dereferenceable": 1.5})json",
     "param 'P' dereferenceable is not an integer"},
    {"non-string addr_space",
     R"json({"addr_space": 3})json",
     "param 'P' attr addr_space is not a string"},
    {"non-integral multiple_of",
     R"json({"multiple_of": 2.5})json",
     "param 'P' multiple_of is not an integer"},
    {"bound that is neither an integer nor a relation",
     R"json({"ge": true})json",
     "param 'P' ge is not an integer or a relation"},
    {"non-object attrs", R"json([])json", "param 'P' attrs is not an object"},
};

#endif /* ROCKE_TESTS_PORTABLE_IR_MALFORMED_PARAM_ATTRS_H */
