// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of the per-arch dense backward builders
 * library/kernels/gfx942/attention_dense_bwd.py and
 * library/kernels/gfx950/attention_dense_bwd.py (build_attention_dense_bwd,
 * as attention_bwd_run._build_dense calls it): the arch's module, the spec
 * class check, then emit_dense_bwd_main
 * (helper_rocke.instances.common._attention_dense_bwd_shell.h).
 */
#include <string.h>

#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.instances.common._attention_dense_bwd_shell.h"
#include "rocke/instance_attention_dense_bwd_spec.h"
#include "rocke/ir_internal.h"

using namespace rocke_attn_bwd;

extern "C" {

rocke_kernel_def_t* rocke_build_attn_dense_bwd_main_new(rocke_ir_builder_t* b,
                                                        const rocke_attn_bwd_dense_spec_t* spec,
                                                        const char* arch)
{
    if(b == NULL)
        return NULL;
    /* The checks and the validator raise before the Python builder exists;
     * the C builder must exist to carry the error, so a refused spec gets a
     * placeholder name. */
    std::string a = arch ? arch : "";
    std::string name;
    rocke_attn_bwd_spec_t body;
    rocke_status_t code = ROCKE_OK;
    std::string msg;
    try
    {
        if(spec == NULL)
            throw ckc::ValueError("spec must not be NULL");
        /* importlib.import_module(f"kernels.{arch}.attention_dense_bwd") */
        if(a != "gfx942" && a != "gfx950")
            throw ckc::ValueError("No module named 'kernels." + a + ".attention_dense_bwd'");
        /* check_dense_bwd_spec_class(spec, Gfx9xxAttentionDenseBwdSpec) */
        if(spec->arch_class == NULL || a != spec->arch_class)
        {
            std::string got = spec->arch_class == NULL ? "AttentionDenseBwdSpec"
                              : strcmp(spec->arch_class, "gfx942") == 0
                                  ? "Gfx942AttentionDenseBwdSpec"
                                  : "Gfx950AttentionDenseBwdSpec";
            std::string want
                = a == "gfx942" ? "Gfx942AttentionDenseBwdSpec" : "Gfx950AttentionDenseBwdSpec";
            throw ckc::TypeError("build_attention_dense_bwd on " + a + " takes a " + want + ", got "
                                 + got);
        }
        /* emit_dense_bwd_main: validate, then the not-built knobs */
        body = dense_validate_spec(*spec, a);
        std::vector<std::string> missing = dense_not_built_all(*spec, body);
        if(!missing.empty())
        {
            std::string j;
            for(size_t i = 0; i < missing.size(); ++i)
                j += (i ? ", " : "") + missing[i];
            throw ckc::NotImplError("the dense attention backward main kernel is not built for "
                                    + j);
        }
        name = dense_spec_kernel_name(*spec);
    }
    catch(const ckc::Error& e)
    {
        code = e.code();
        msg = e.what();
    }
    catch(const std::bad_alloc& e)
    {
        code = ROCKE_ERR_OOM;
        msg = e.what();
    }
    if(rocke_ir_builder_init(b, code == ROCKE_OK ? name.c_str() : "rocke_attn_dense_bwd_invalid")
       != ROCKE_OK)
        return NULL;
    if(code != ROCKE_OK)
    {
        rocke_i_set_err_msg(b, code, msg.c_str());
        return NULL;
    }
    return ckc::guard_builder(
        b, [&]() -> rocke_kernel_def_t* { return emit_dense_bwd_main(b, *spec, a, body); });
}

} /* extern "C" */
