// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * serialized_ir.cpp -- the parse step of the serialized-IR lowering entries
 * (rocke/serialized_ir_internal.h): serialized ck.dsl.ir/v1 text in, a kernel
 * or the reason it is not one out.
 */
#include "rocke/serialized_ir_internal.h"

#include "rocke/ir_serialize.h"

namespace ckc
{

const char* serialized_ir_step_message(const rocke_ir_builder_t* b, const char* fallback)
{
    const char* message = rocke_ir_builder_error(b);
    return (message && *message) ? message : fallback;
}

void serialized_ir_report(rocke_strbuf_t* message, const char* reason)
{
    if(message)
    {
        rocke_strbuf_append(message, reason);
    }
}

rocke_status_t serialized_ir_parse(const char* ir_text,
                                   rocke_ir_builder_t* b,
                                   rocke_kernel_def_t** out_kernel,
                                   rocke_strbuf_t* message)
{
    *out_kernel = nullptr;
    if(!ir_text || !*ir_text)
    {
        serialized_ir_report(message, "empty IR input");
        return ROCKE_ERR_VALUE;
    }
    if(rocke_ir_builder_init(b, "serialized_ir") != ROCKE_OK)
    {
        serialized_ir_report(message, "builder init failed (OOM)");
        return ROCKE_ERR_OOM;
    }
    rocke_kernel_def_t* kernel = nullptr;
    const rocke_status_t status = rocke_ir_parse(ir_text, b, &kernel);
    if(status != ROCKE_OK || !kernel)
    {
        serialized_ir_report(message, serialized_ir_step_message(b, "unknown parse error"));
        rocke_ir_builder_free(b);
        /* The parser reports every failure with a status; a kernel-less
         * ROCKE_OK would still be a failure here, where ROCKE_OK means a kernel. */
        return status != ROCKE_OK ? status : ROCKE_ERR_VALUE;
    }
    *out_kernel = kernel;
    return ROCKE_OK;
}

bool serialized_ir_names_target(const char* arch)
{
    return arch != nullptr && *arch != '\0';
}

void serialized_ir_take_message(rocke_strbuf_t* message, char** out_message)
{
    if(out_message)
    {
        *out_message = rocke_strbuf_detach(message);
    }
    rocke_strbuf_free(message);
}

} // namespace ckc
