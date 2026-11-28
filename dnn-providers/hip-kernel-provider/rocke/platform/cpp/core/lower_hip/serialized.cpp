// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * serialized.cpp -- rocke_lower_serialized_ir_to_hip and its char* twin
 * rocke_lower_serialized_ir_to_hip_text: serialized ck.dsl.ir/v1 text in, HIP
 * source or the engine's refusal out. They compose the shared parse step
 * (rocke/serialized_ir_internal.h) with the HIP lowerer
 * (rocke_lower_kernel_to_hip), so every host that holds a kernel as text -- the
 * rocke_engine binding, the ctypes transport rocke.core.native_engine and the
 * rocke_ir_lower_cli test program -- parses, lowers and reports a refusal the
 * same way.
 */
#include "rocke/lower_hip.h"

#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/serialized_ir_internal.h"
#include "rocke/strbuf.h"

rocke_status_t rocke_lower_serialized_ir_to_hip(const char* ir_text,
                                                const char* arch,
                                                rocke_strbuf_t* out,
                                                rocke_strbuf_t* message)
{
    if(!out)
    {
        ckc::serialized_ir_report(message, "no output buffer");
        return ROCKE_ERR_VALUE;
    }
    rocke_ir_builder_t b = {};
    rocke_kernel_def_t* kernel = nullptr;
    rocke_status_t status = ckc::serialized_ir_parse(ir_text, &b, &kernel, message);
    if(status != ROCKE_OK)
    {
        return status;
    }
    if(!ckc::serialized_ir_names_target(arch))
    {
        /* Unlike rocke_lower_kernel_to_hip, this entry has no default target,
         * as rocke_lower_serialized_ir_to_llvm has none; the checks run in the
         * same order as there. */
        rocke_ir_builder_free(&b);
        ckc::serialized_ir_report(message, ROCKE_NO_TARGET_REASON);
        return ROCKE_ERR_VALUE;
    }
    rocke_lower_hip_opts_t opts = {};
    opts.arch = arch;
    status = rocke_lower_kernel_to_hip(&b, kernel, &opts, out);
    if(status != ROCKE_OK)
    {
        ckc::serialized_ir_report(message,
                                  ckc::serialized_ir_step_message(&b, "unknown lowering error"));
    }
    rocke_ir_builder_free(&b);
    return status;
}

rocke_status_t rocke_lower_serialized_ir_to_hip_text(const char* ir_text,
                                                     const char* arch,
                                                     char** out_text,
                                                     char** out_message)
{
    if(out_message)
    {
        *out_message = nullptr;
    }
    rocke_strbuf_t message;
    rocke_strbuf_init(&message, 0);
    if(!out_text)
    {
        ckc::serialized_ir_report(&message, "no output buffer");
        ckc::serialized_ir_take_message(&message, out_message);
        return ROCKE_ERR_VALUE;
    }
    *out_text = nullptr;
    rocke_strbuf_t text;
    rocke_strbuf_init(&text, 0);
    rocke_status_t status = rocke_lower_serialized_ir_to_hip(ir_text, arch, &text, &message);
    if(status == ROCKE_OK && text.oom)
    {
        ckc::serialized_ir_report(&message, "output allocation failed (OOM)");
        status = ROCKE_ERR_OOM;
    }
    if(status == ROCKE_OK)
    {
        *out_text = rocke_strbuf_detach(&text);
        rocke_strbuf_free(&message);
    }
    else
    {
        ckc::serialized_ir_take_message(&message, out_message);
    }
    rocke_strbuf_free(&text);
    return status;
}
