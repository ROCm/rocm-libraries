// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* The char* serialized-IR entries: rocke_lower_serialized_ir_to_llvm
 * (rocke/lower_llvm.h) and rocke_lower_serialized_ir_to_hip_text
 * (rocke/lower_hip.h), the two an FFI caller such as rocke.core.native_engine
 * calls.
 *
 * The LLVM entry must emit what rocke_lower_kernel_to_llvm_ex emits for the
 * kernel the text describes, and refuse with the documented status and reason:
 * text that is not a kernel (the parser's reason, the one the HIP entry
 * reports), an unknown target, no target, an arch-domain refusal (status
 * ROCKE_ERR_ARCH_DOMAIN and the lane's own message), empty or no text, and no
 * output pointer. The HIP twin must report exactly what
 * rocke_lower_serialized_ir_to_hip reports for the same input. Both must return
 * the same status whether or not the caller asks for the reason, and no reason
 * on success.
 */
#include <cstdio>
#include <cstdlib>
#include <string>

#include "rocke/arch_domain.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"

namespace
{

int g_failures = 0;

void check(bool ok, const std::string& input, const std::string& what)
{
    if(!ok)
    {
        std::fprintf(stderr, "FAIL [%s]: %s\n", input.c_str(), what.c_str());
        ++g_failures;
    }
}

void set_arch_domain_mode(const char* value)
{
#ifdef _WIN32
    _putenv_s(ROCKE_ARCH_DOMAIN_ENV, value != nullptr ? value : "");
#else
    if(value == nullptr)
        unsetenv(ROCKE_ARCH_DOMAIN_ENV);
    else
        setenv(ROCKE_ARCH_DOMAIN_ENV, value, 1);
#endif
}

/* A kernel built by `emit` into a fresh builder, serialized; "" when it did
 * not build. The built kernel is lowered directly too, into `direct_llvm`, for
 * `arch` at `flavor`. */
template <typename Emit>
std::string serialized_kernel(const char* name,
                              Emit emit,
                              const char* arch,
                              rocke_llvm_flavor_t flavor,
                              std::string* direct_llvm)
{
    rocke_ir_builder_t builder = {};
    if(rocke_ir_builder_init(&builder, name) != ROCKE_OK)
        return "";
    emit(&builder);
    rocke_b_ret(&builder);
    std::string serialized;
    char* text = nullptr;
    if(rocke_ir_builder_ok(&builder)
       && rocke_ir_serialize(rocke_ir_builder_kernel(&builder), &text) == ROCKE_OK && text)
        serialized = text;
    std::free(text);
    if(direct_llvm)
    {
        char* llvm = nullptr;
        if(rocke_lower_kernel_to_llvm_ex(
               rocke_ir_builder_kernel(&builder), flavor, arch, &llvm, nullptr, 0)
               == ROCKE_OK
           && llvm)
            *direct_llvm = llvm;
        std::free(llvm);
    }
    rocke_ir_builder_free(&builder);
    return serialized;
}

/* One call's outcome; `text` and `message` are "" where the entry gave NULL,
 * and `*_null` records which it gave. */
struct outcome
{
    rocke_status_t status;
    std::string text;
    std::string message;
    bool text_null;
    bool message_null;
};

outcome take(rocke_status_t status, char* text, char* message)
{
    outcome result{status, text ? text : "", message ? message : "", !text, !message};
    rocke_text_free(text);
    rocke_text_free(message);
    return result;
}

outcome lower_llvm(const char* ir_text,
                   const char* arch,
                   rocke_llvm_flavor_t flavor,
                   bool with_message,
                   bool with_out = true)
{
    char* text = nullptr;
    char* message = nullptr;
    const rocke_status_t status = rocke_lower_serialized_ir_to_llvm(
        ir_text, arch, flavor, with_out ? &text : nullptr, with_message ? &message : nullptr);
    return take(status, text, message);
}

outcome
    lower_hip_text(const char* ir_text, const char* arch, bool with_message, bool with_out = true)
{
    char* text = nullptr;
    char* message = nullptr;
    const rocke_status_t status = rocke_lower_serialized_ir_to_hip_text(
        ir_text, arch, with_out ? &text : nullptr, with_message ? &message : nullptr);
    return take(status, text, message);
}

/* What rocke_lower_serialized_ir_to_hip reports for the same input. */
outcome lower_hip_strbuf(const char* ir_text, const char* arch)
{
    rocke_strbuf_t out;
    rocke_strbuf_t message;
    rocke_strbuf_init(&out, 0);
    rocke_strbuf_init(&message, 0);
    const rocke_status_t status = rocke_lower_serialized_ir_to_hip(ir_text, arch, &out, &message);
    outcome result{status,
                   status == ROCKE_OK ? rocke_strbuf_cstr(&out) : "",
                   rocke_strbuf_cstr(&message),
                   false,
                   false};
    rocke_strbuf_free(&out);
    rocke_strbuf_free(&message);
    return result;
}

/* A refusal: `status` with and without a reason requested, NULL text, and the
 * reason `reason` (any non-empty one when `reason` is NULL). */
void check_llvm_refusal(const std::string& input,
                        const char* ir_text,
                        const char* arch,
                        rocke_llvm_flavor_t flavor,
                        rocke_status_t status,
                        const char* reason,
                        bool with_out = true)
{
    const outcome reported = lower_llvm(ir_text, arch, flavor, true, with_out);
    const outcome unreported = lower_llvm(ir_text, arch, flavor, false, with_out);
    check(reported.status == status,
          input,
          "status " + std::to_string((int)reported.status) + ", expected "
              + std::to_string((int)status) + ": " + reported.message);
    check(unreported.status == status, input, "the status depends on asking for the reason");
    check(reported.text_null, input, "a refusal returned text");
    check(!reported.message.empty(), input, "the refusal gave no reason");
    if(reason)
        check(reported.message == reason, input, "reason '" + reported.message + "'");
}

/* The HIP twin reports what the buffer entry reports for `ir_text`. */
void check_hip_twin(const std::string& input, const char* ir_text, const char* arch)
{
    const outcome expected = lower_hip_strbuf(ir_text, arch);
    const outcome reported = lower_hip_text(ir_text, arch, true);
    const outcome unreported = lower_hip_text(ir_text, arch, false);
    check(reported.status == expected.status, input, "the twin's status differs");
    check(
        unreported.status == expected.status, input, "the status depends on asking for the reason");
    check(reported.text == expected.text, input, "the twin's source differs");
    check(reported.message == expected.message, input, "the twin's reason differs");
    if(expected.status == ROCKE_OK)
        check(!reported.text_null && reported.message_null && unreported.text == expected.text,
              input,
              "a lowered kernel returned no text, or a reason");
    else
        check(reported.text_null && !reported.message.empty(),
              input,
              "a refusal returned text, or no reason");
}

void emit_waitcnt(rocke_ir_builder_t* b)
{
    rocke_b_s_waitcnt(b, 0, 0, -1);
}

void emit_dpp8(rocke_ir_builder_t* b)
{
    rocke_b_mov_dpp8(b, rocke_b_const_i32(b, 1), 0x765432);
}

/* The first committed flavor whose arch-domain column refuses mov.dpp8.i32 on
 * gfx950 (a gfx10+ intrinsic measured arch_absent on gfx9), or NULL. */
const char* dpp8_refusing_flavor(const rocke_arch_domain_cell_t** cell)
{
    for(int i = 0; i < rocke_llvm_flavor_count(); ++i)
    {
        const char* flavor = rocke_llvm_flavor_at(i);
        const rocke_arch_domain_cell_t* found
            = rocke_arch_domain_negative(flavor, "mov.dpp8.i32", "gfx950");
        if(found != nullptr && !found->waived)
        {
            *cell = found;
            return flavor;
        }
    }
    return nullptr;
}

void test_llvm_entry()
{
    const rocke_llvm_flavor_t llvm20 = rocke_llvm_flavor_from_name("llvm20");
    std::string direct;
    const std::string waitcnt
        = serialized_kernel("waitcnt", emit_waitcnt, "gfx942", llvm20, &direct);
    check(!waitcnt.empty() && !direct.empty(), "waitcnt", "the kernel did not build and lower");

    const outcome reported = lower_llvm(waitcnt.c_str(), "gfx942", llvm20, true);
    const outcome unreported = lower_llvm(waitcnt.c_str(), "gfx942", llvm20, false);
    check(reported.status == ROCKE_OK, "lowers", "status " + std::to_string((int)reported.status));
    check(reported.text == direct, "lowers", "the text differs from rocke_lower_kernel_to_llvm_ex");
    check(reported.message_null, "lowers", "a lowered kernel returned a reason");
    check(unreported.text == direct, "lowers", "the text depends on asking for the reason");

    /* One parse step: the HIP entry's refusal of the same text. */
    const outcome parse_refusal = lower_hip_strbuf("not serialized IR", "gfx942");
    check_llvm_refusal("not a kernel",
                       "not serialized IR",
                       "gfx942",
                       llvm20,
                       parse_refusal.status,
                       parse_refusal.message.c_str());
    check_llvm_refusal("unknown arch",
                       waitcnt.c_str(),
                       "gfx000",
                       llvm20,
                       ROCKE_ERR_KEY,
                       "unknown arch backend gfx000");
    for(const char* arch : {"", (const char*)nullptr})
        check_llvm_refusal(arch ? "empty arch" : "no arch",
                           waitcnt.c_str(),
                           arch,
                           llvm20,
                           ROCKE_ERR_VALUE,
                           "lowering requires an explicit gfx target (there is no default)");
    check_llvm_refusal("empty text", "", "gfx942", llvm20, ROCKE_ERR_VALUE, "empty IR input");
    check_llvm_refusal("no text", nullptr, "gfx942", llvm20, ROCKE_ERR_VALUE, "empty IR input");
    check_llvm_refusal("no output pointer",
                       waitcnt.c_str(),
                       "gfx942",
                       llvm20,
                       ROCKE_ERR_VALUE,
                       "no output buffer",
                       false);
}

void test_llvm_arch_domain_refusal()
{
    const rocke_arch_domain_cell_t* cell = nullptr;
    const char* flavor = dpp8_refusing_flavor(&cell);
    check(flavor != nullptr, "arch domain", "no committed column refuses mov.dpp8.i32 on gfx950");
    if(flavor == nullptr)
        return;
    const std::string dpp8 = serialized_kernel(
        "dpp8", emit_dpp8, "gfx950", rocke_llvm_flavor_from_name(flavor), nullptr);
    char lane_message[ROCKE_ERR_MSG_CAP];
    rocke_arch_domain_message(cell, flavor, lane_message, sizeof lane_message);
    set_arch_domain_mode("error");
    check_llvm_refusal("arch domain",
                       dpp8.c_str(),
                       "gfx950",
                       rocke_llvm_flavor_from_name(flavor),
                       ROCKE_ERR_ARCH_DOMAIN,
                       lane_message);
    set_arch_domain_mode(nullptr);
}

void test_hip_twin()
{
    const std::string waitcnt
        = serialized_kernel("waitcnt", emit_waitcnt, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, nullptr);
    check_hip_twin("lowers", waitcnt.c_str(), "gfx942");
    check_hip_twin("lowering refused", waitcnt.c_str(), "gfx1250");
    check_hip_twin("not a kernel", "not serialized IR", "gfx942");
    check_hip_twin("empty text", "", "gfx942");
    check_hip_twin("no text", nullptr, "gfx942");

    const outcome no_out = lower_hip_text(waitcnt.c_str(), "gfx942", true, false);
    check(no_out.status == ROCKE_ERR_VALUE && no_out.message == "no output buffer",
          "no output pointer",
          "status " + std::to_string((int)no_out.status) + ": " + no_out.message);
}

} // namespace

int main()
{
    test_llvm_entry();
    test_llvm_arch_domain_refusal();
    test_hip_twin();
    if(g_failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("serialized-IR text entries: all checks passed\n");
    return 0;
}
