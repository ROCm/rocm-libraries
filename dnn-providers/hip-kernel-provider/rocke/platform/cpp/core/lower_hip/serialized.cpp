// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * serialized.cpp -- rocke_lower_serialized_ir_to_hip: serialized ck.dsl.ir/v1
 * text in, HIP source or the engine's refusal out. It composes the IR parser
 * (rocke_ir_parse) with the HIP lowerer (rocke_lower_kernel_to_hip), so every
 * host that holds a kernel as text -- the rocke_engine binding and the
 * rocke_ir_lower_cli test program -- parses, lowers and reports a refusal the
 * same way.
 */
#include "rocke/lower_hip.h"

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/strbuf.h"

/* The builder's message for a failed step, or `fallback` when it set none. */
static const char* rocke_h_step_message(const rocke_ir_builder_t* b, const char* fallback)
{
    const char* message = rocke_ir_builder_error(b);
    return (message && *message) ? message : fallback;
}

/* Append `reason` to the caller's `message` buffer, if it passed one. */
static void rocke_h_report(rocke_strbuf_t* message, const char* reason)
{
    if(message)
    {
        rocke_strbuf_append(message, reason);
    }
}

rocke_status_t rocke_lower_serialized_ir_to_hip(const char* ir_text,
                                                const char* arch,
                                                rocke_strbuf_t* out,
                                                rocke_strbuf_t* message)
{
    if(!out)
    {
        rocke_h_report(message, "no output buffer");
        return ROCKE_ERR_VALUE;
    }
    if(!ir_text || !*ir_text)
    {
        rocke_h_report(message, "empty IR input");
        return ROCKE_ERR_VALUE;
    }
    rocke_ir_builder_t b = {};
    if(rocke_ir_builder_init(&b, "serialized_ir") != ROCKE_OK)
    {
        rocke_h_report(message, "builder init failed (OOM)");
        return ROCKE_ERR_OOM;
    }
    rocke_kernel_def_t* kernel = nullptr;
    rocke_status_t status = rocke_ir_parse(ir_text, &b, &kernel);
    if(status != ROCKE_OK || !kernel)
    {
        rocke_h_report(message, rocke_h_step_message(&b, "unknown parse error"));
        rocke_ir_builder_free(&b);
        /* The parser reports every failure with a status; a kernel-less
         * ROCKE_OK would still be a failure here, where ROCKE_OK means source. */
        return status != ROCKE_OK ? status : ROCKE_ERR_VALUE;
    }
    rocke_lower_hip_opts_t opts = {};
    opts.arch = arch;
    status = rocke_lower_kernel_to_hip(&b, kernel, &opts, out);
    if(status != ROCKE_OK)
    {
        rocke_h_report(message, rocke_h_step_message(&b, "unknown lowering error"));
    }
    rocke_ir_builder_free(&b);
    return status;
}
