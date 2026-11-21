// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* rocke_lower_serialized_ir_to_hip (rocke/lower_hip.h), called with and
 * without its optional `message` buffer.
 *
 * For each input -- a kernel that lowers, a kernel the HIP lowering refuses,
 * text that is not a kernel, empty text, no text, and no output buffer -- the
 * call returns the same status whether or not the caller passes a `message`
 * buffer. With one, a refusal appends a non-empty reason (for the documented
 * refusals, exactly the documented one) and a lowered kernel appends none;
 * without one, a lowered kernel's source is the same.
 */
#include <cstdio>
#include <cstdlib>
#include <string>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/strbuf.h"

namespace
{

int g_failures = 0;

void check(bool ok, const char* input, const char* what)
{
    if(!ok)
    {
        std::fprintf(stderr, "FAIL [%s]: %s\n", input, what);
        ++g_failures;
    }
}

/* A kernel with one s_waitcnt, serialized: it lowers on gfx942 and is refused
 * on a target without the monolithic s_waitcnt. */
std::string serialized_waitcnt_kernel()
{
    rocke_ir_builder_t builder = {};
    if(rocke_ir_builder_init(&builder, "waitcnt") != ROCKE_OK)
        return "";
    rocke_b_s_waitcnt(&builder, 0, 0, -1);
    rocke_b_ret(&builder);
    char* text = nullptr;
    std::string serialized;
    if(rocke_ir_builder_ok(&builder)
       && rocke_ir_serialize(rocke_ir_builder_kernel(&builder), &text) == ROCKE_OK && text)
        serialized = text;
    std::free(text);
    rocke_ir_builder_free(&builder);
    return serialized;
}

struct lowering
{
    rocke_status_t status;
    std::string source;
    std::string message;
};

/* Lower `ir_text` for `arch`, passing a `message` buffer only when
 * `with_message` is set, and an output buffer only when `with_out` is set. */
lowering lower(const char* ir_text, const char* arch, bool with_message, bool with_out = true)
{
    rocke_strbuf_t out;
    rocke_strbuf_t message;
    rocke_strbuf_init(&out, 0);
    rocke_strbuf_init(&message, 0);
    lowering result;
    result.status = rocke_lower_serialized_ir_to_hip(
        ir_text, arch, with_out ? &out : nullptr, with_message ? &message : nullptr);
    result.source = rocke_strbuf_cstr(&out);
    result.message = rocke_strbuf_cstr(&message);
    rocke_strbuf_free(&out);
    rocke_strbuf_free(&message);
    return result;
}

/* A documented refusal: the same `status` with and without a message buffer,
 * and exactly `reason` in the buffer. */
void check_refusal(const char* input,
                   const char* ir_text,
                   bool with_out,
                   rocke_status_t status,
                   const char* reason)
{
    const lowering reported = lower(ir_text, "gfx942", true, with_out);
    const lowering unreported = lower(ir_text, "gfx942", false, with_out);
    check(reported.status == status, input, "the reported status is not the documented one");
    check(unreported.status == status, input, "the unreported status is not the documented one");
    check(reported.message == reason, input, "the reason is not the documented one");
}

void check_input(const char* input, const char* ir_text, const char* arch, bool lowers)
{
    const lowering reported = lower(ir_text, arch, true);
    const lowering unreported = lower(ir_text, arch, false);
    check(unreported.status == reported.status,
          input,
          "the status depends on whether a message buffer is passed");
    if(lowers)
    {
        check(reported.status == ROCKE_OK, input, "the kernel did not lower");
        check(reported.message.empty(), input, "a lowered kernel reported a reason");
        check(!reported.source.empty() && unreported.source == reported.source,
              input,
              "the source differs without a message buffer");
    }
    else
    {
        check(reported.status != ROCKE_OK, input, "the input was not refused");
        check(!reported.message.empty(), input, "the refusal reported no reason");
    }
}

} // namespace

int main()
{
    const std::string waitcnt = serialized_waitcnt_kernel();
    check(!waitcnt.empty(), "waitcnt kernel", "the kernel did not build and serialize");
    check_input("lowers", waitcnt.c_str(), "gfx942", true);
    check_input("lowering refused", waitcnt.c_str(), "gfx1250", false);
    check_input("not a kernel", "not serialized IR", "gfx942", false);
    check_refusal("empty text", "", true, ROCKE_ERR_VALUE, "empty IR input");
    check_refusal("no text", nullptr, true, ROCKE_ERR_VALUE, "empty IR input");
    check_refusal("no output buffer", waitcnt.c_str(), false, ROCKE_ERR_VALUE, "no output buffer");
    if(g_failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("rocke_lower_serialized_ir_to_hip: all checks passed\n");
    return 0;
}
