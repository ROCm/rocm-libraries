// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_arch_domain_refusal.cpp -- the C++ engine's arch-domain lane,
 * without the Python extension.
 *
 * The C++ engine consults its generated copy of the arch-domain table at its
 * demand chokepoint (rocke_ll_need) and, in `error` mode, refuses a kernel that
 * demands an intrinsic the target provably cannot lower. tests/core/
 * test_arch_domain_lane.py holds this engine to the Python lowerer through the
 * rocke_engine extension; this test covers the same contract where that
 * extension is not built, which includes the installed CTest lane:
 *
 *   - ROCKE_ARCH_DOMAIN is parsed by Python's rules, with Python's message;
 *   - `error` refuses with ROCKE_ERR_ARCH_DOMAIN and Python's message;
 *   - `warn` and `off` lower;
 *   - the message is cut where Python's MESSAGE_CAP cuts it;
 *   - the message text equals a sentence that the Python test also pins.
 *
 * The demand used is `mov.dpp8.i32` on gfx950: an RDNA intrinsic the committed
 * columns measure arch_absent on CDNA, which is a fact about the hardware. If no
 * column says so any more, the test fails and names it, rather than passing
 * without having refused anything.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rocke/arch_domain.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

namespace
{

constexpr const char* kKey = "mov.dpp8.i32";
constexpr const char* kArch = "gfx950";

int g_failures = 0;

void fail(const std::string& what, int line)
{
    fprintf(stderr, "FAIL: %s (%s:%d)\n", what.c_str(), __FILE__, line);
    ++g_failures;
}

#define EXPECT(cond, what)          \
    do                              \
    {                               \
        if(!(cond))                 \
            fail((what), __LINE__); \
    } while(0)

void set_mode(const char* value)
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

struct Lowered
{
    rocke_status_t status;
    std::string text; /* the IR on success, the error message otherwise */
};

/* One mov.dpp8.i32 demand, lowered for kArch at `flavor`. */
Lowered lower_dpp8(const char* flavor)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "rocke_arch_domain_dpp8") != ROCKE_OK)
    {
        return {ROCKE_ERR_OOM, "rocke_ir_builder_init"};
    }
    rocke_b_mov_dpp8(&b, rocke_b_const_i32(&b, 1), 0x765432);
    rocke_b_ret(&b);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    const rocke_status_t st = rocke_lower_kernel_to_llvm_ex(rocke_ir_builder_kernel(&b),
                                                            rocke_llvm_flavor_from_name(flavor),
                                                            kArch,
                                                            &ll,
                                                            err,
                                                            sizeof err);
    Lowered out{st, st == ROCKE_OK && ll != nullptr ? std::string(ll) : std::string(err)};
    std::free(ll);
    rocke_ir_builder_free(&b);
    return out;
}

/* A kernel that demands no intrinsic at all, lowered for kArch at `flavor`. */
Lowered lower_nothing(const char* flavor)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "rocke_arch_domain_nothing") != ROCKE_OK)
    {
        return {ROCKE_ERR_OOM, "rocke_ir_builder_init"};
    }
    rocke_b_const_i32(&b, 1);
    rocke_b_ret(&b);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    const rocke_status_t st = rocke_lower_kernel_to_llvm_ex(rocke_ir_builder_kernel(&b),
                                                            rocke_llvm_flavor_from_name(flavor),
                                                            kArch,
                                                            &ll,
                                                            err,
                                                            sizeof err);
    Lowered out{st, st == ROCKE_OK && ll != nullptr ? std::string(ll) : std::string(err)};
    std::free(ll);
    rocke_ir_builder_free(&b);
    return out;
}

/* The first committed flavor whose column refuses kKey on kArch. */
const char* refusing_flavor(const rocke_arch_domain_cell_t** cell)
{
    for(int i = 0; i < rocke_llvm_flavor_count(); ++i)
    {
        const char* flavor = rocke_llvm_flavor_at(i);
        const rocke_arch_domain_cell_t* c = rocke_arch_domain_negative(flavor, kKey, kArch);
        if(c != nullptr && !c->waived)
        {
            *cell = c;
            return flavor;
        }
    }
    return nullptr;
}

void test_mode_parsing()
{
    const std::string dflt = rocke_arch_domain_default_mode();
    rocke_arch_domain_mode_t mode;
    char err[ROCKE_ERR_MSG_CAP];

    const struct
    {
        const char* raw;
        rocke_arch_domain_mode_t want;
    } ok[] = {{"error", ROCKE_ARCH_DOMAIN_ERROR},
              {" ERROR\t", ROCKE_ARCH_DOMAIN_ERROR},
              {"warn", ROCKE_ARCH_DOMAIN_WARN},
              {"Warn", ROCKE_ARCH_DOMAIN_WARN},
              {"off", ROCKE_ARCH_DOMAIN_OFF},
              {"\nOFF ", ROCKE_ARCH_DOMAIN_OFF}};
    for(const auto& c : ok)
    {
        const rocke_status_t st = rocke_arch_domain_mode(c.raw, &mode, err, sizeof err);
        EXPECT(st == ROCKE_OK && mode == c.want, std::string("mode '") + c.raw + "'");
    }

    /* Unset, empty and blank all mean the default. */
    rocke_arch_domain_mode_t from_default;
    EXPECT(rocke_arch_domain_mode(dflt.c_str(), &from_default, err, sizeof err) == ROCKE_OK,
           "the generated default parses");
    for(const char* raw : {(const char*)nullptr, "", "   "})
    {
        EXPECT(rocke_arch_domain_mode(raw, &mode, err, sizeof err) == ROCKE_OK
                   && mode == from_default,
               "unset/empty/blank means the default");
    }

    /* A misspelling is refused with Python's message, not read as the default. */
    EXPECT(rocke_arch_domain_mode(" Warning ", &mode, err, sizeof err) == ROCKE_ERR_VALUE,
           "a misspelled mode is refused");
    const std::string want = "ROCKE_ARCH_DOMAIN='Warning' is not a mode; expected one of "
                             "error, warn, off (unset means "
                             + dflt + ").";
    EXPECT(want == err, std::string("mode message: ") + err);
}

void test_message_cap()
{
    const std::string evidence(400, 'x');
    const rocke_arch_domain_cell_t cell = {kKey, kArch, "arch_absent", evidence.c_str(), 0};
    char out[ROCKE_ERR_MSG_CAP];
    rocke_arch_domain_message(&cell, "llvm22", out, sizeof out);
    EXPECT(strlen(out) == ROCKE_ERR_MSG_CAP - 1, "the message is cut at the error buffer");
    const std::string head = "arch-domain: intrinsic 'mov.dpp8.i32' is not available on gfx950 "
                             "at llvm22 (arch_absent): xxx";
    EXPECT(std::string(out).rfind(head, 0) == 0, std::string("message text: ") + out);
}

/* The same sentence is pinned in test_arch_domain_lane.py
 * (test_the_message_text_is_the_one_the_cpp_engine_pins), where Python's
 * message() must produce it. The two tests run without the rocke_engine
 * extension, so a drift in either engine fails a CI test. */
void test_message_text()
{
    const rocke_arch_domain_cell_t cell
        = {kKey, kArch, "arch_absent", "no such intrinsic on gfx950", 0};
    char out[ROCKE_ERR_MSG_CAP];
    rocke_arch_domain_message(&cell, "llvm22", out, sizeof out);
    EXPECT(std::string(out)
               == "arch-domain: intrinsic 'mov.dpp8.i32' is not available on gfx950 at llvm22 "
                  "(arch_absent): no such intrinsic on gfx950",
           std::string("message text: ") + out);
    const rocke_arch_domain_cell_t bare = {kKey, kArch, "arch_absent", "", 0};
    rocke_arch_domain_message(&bare, "llvm22", out, sizeof out);
    EXPECT(std::string(out)
               == "arch-domain: intrinsic 'mov.dpp8.i32' is not available on gfx950 at llvm22 "
                  "(arch_absent)",
           std::string("message text without evidence: ") + out);
}

void test_lowering(const char* flavor, const rocke_arch_domain_cell_t* cell)
{
    char want[ROCKE_ERR_MSG_CAP];
    rocke_arch_domain_message(cell, flavor, want, sizeof want);

    set_mode("error");
    Lowered refused = lower_dpp8(flavor);
    EXPECT(refused.status == ROCKE_ERR_ARCH_DOMAIN,
           "error mode refuses: status " + std::to_string((int)refused.status) + ", "
               + refused.text);
    EXPECT(refused.text == want, "refusal message: " + refused.text);

    for(const char* mode : {"warn", "off"})
    {
        set_mode(mode);
        Lowered lowered = lower_dpp8(flavor);
        EXPECT(lowered.status == ROCKE_OK
                   && lowered.text.find("@llvm.amdgcn.mov.dpp8.i32") != std::string::npos,
               std::string(mode) + " mode lowers: " + lowered.text);
    }

    set_mode("warning");
    Lowered bad = lower_dpp8(flavor);
    EXPECT(bad.status == ROCKE_ERR_VALUE
               && bad.text.find("ROCKE_ARCH_DOMAIN='warning' is not a mode") == 0,
           "a misspelled mode fails the lowering: " + bad.text);

    /* Read once at entry, so the same bad value also fails a kernel that
     * demands no intrinsic, and a good one lets it lower. */
    Lowered nothing = lower_nothing(flavor);
    EXPECT(nothing.status == ROCKE_ERR_VALUE
               && nothing.text.find("ROCKE_ARCH_DOMAIN='warning' is not a mode") == 0,
           "a misspelled mode fails a kernel that demands nothing: " + nothing.text);
    set_mode("error");
    nothing = lower_nothing(flavor);
    EXPECT(nothing.status == ROCKE_OK,
           "error mode lowers a kernel that demands nothing: " + nothing.text);
    set_mode(nullptr);
}

} // namespace

int main()
{
    test_mode_parsing();
    test_message_cap();
    test_message_text();

    const rocke_arch_domain_cell_t* cell = nullptr;
    const char* flavor = refusing_flavor(&cell);
    if(flavor == nullptr)
    {
        fail(std::string("no committed column refuses ") + kKey + " on " + kArch
                 + "; this test needs a demand some column refuses",
             __LINE__);
    }
    else
    {
        test_lowering(flavor, cell);
    }

    if(g_failures != 0)
    {
        fprintf(stderr, "arch_domain_refusal: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("arch_domain_refusal: OK (refused at %s)\n", flavor);
    return 0;
}
