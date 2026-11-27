// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "rocke/instance_gfx950_attention_dense_builder.h"

#include "attention_dense_internal.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"

#include <cstring>

rocke_kernel_def_t* rocke_gfx950_build_attention_dense_new(
    rocke_ir_builder_t* b, const rocke_gfx950_attention_dense_spec_t* spec, const char* arch)
{
    if(b == nullptr)
        return nullptr;
    // Even a refusal or an exception before arena initialization must leave b
    // safe to inspect and free. Never reinterpret an invalid spec as a valid one.
    b->arena = {};
    b->kernel = nullptr;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if(arch != nullptr && std::strcmp(arch, "gfx950") != 0)
            rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "attention_dense is gfx950-only (got %s)", arch);
        char reason[ROCKE_ATTENTION_DENSE_REASON_CAP];
        if(!rocke_gfx950_supports_attention_dense(spec, arch, reason, sizeof(reason)))
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "unsupported gfx950 attention_dense spec: %s", reason);
        char name[ROCKE_ATTENTION_DENSE_NAME_CAP];
        const auto status = rocke_gfx950_attention_dense_kernel_name(spec, name, sizeof(name));
        if(status != ROCKE_OK)
            ckc::raise_status(status, "could not encode gfx950 dense kernel name");
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return nullptr;
        rocke_attn2d_c32_dist_reset();
        if(spec->base.persistent)
            rocke_dense_gfx950::build_persistent(b, *spec);
        else
            rocke_dense_gfx950::build_grid(b, *spec);
        return rocke_ir_builder_ok(b) ? rocke_ir_builder_kernel(b) : nullptr;
    });
}
