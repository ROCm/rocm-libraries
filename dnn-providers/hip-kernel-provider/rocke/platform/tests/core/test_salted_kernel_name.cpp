// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// rocke_sha1_hex / rocke_salted_kernel_name (helpers/spec.py
// salted_kernel_name): the digests hashlib gives, the Python test's pinned
// name, the stable sort by name, and the refusals.
#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.spec.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#define CHECK(expr)                                                 \
    do                                                              \
    {                                                               \
        if(!(expr))                                                 \
        {                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                               \
        }                                                           \
    } while(0)

using Pairs = std::vector<std::pair<std::string, std::string>>;

static int test_sha1()
{
    /* hashlib.sha1(m).hexdigest(); the lengths cover the one- and two-block
     * tails (55 / 56) and whole blocks (64). */
    CHECK(rocke_sha1_hex("") == "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    CHECK(rocke_sha1_hex("abc") == "a9993e364706816aba3e25717850c26c9cd0d89d");
    CHECK(rocke_sha1_hex(std::string(55, 'a')) == "c1c8bbdc22796e28c0e15163d20899b65621d65a");
    CHECK(rocke_sha1_hex(std::string(56, 'a')) == "c2db330f6083854c99d4b5bfb6e8f29f201be699");
    CHECK(rocke_sha1_hex(std::string(64, 'a')) == "0098ba824b5c16427bd7a1122a5a442a25ec644d");
    CHECK(rocke_sha1_hex(std::string(119, 'a')) == "ee971065aaa017e0632a8ca6c77bb3bf8b1dfc56");
    CHECK(rocke_sha1_hex(std::string(200, 'a')) == "e61cfffe0d9195a525fc6cf06ca2d77119c24a40");
    return 0;
}

static int test_salted_name()
{
    CHECK(rocke_salted_kernel_name("k", Pairs{}) == "k");
    /* tests/test_helpers_spec.py::test_salted_kernel_name, either order */
    CHECK(rocke_salted_kernel_name("k", Pairs{{"b", "(2, 'x')"}, {"a", "1"}}) == "k_420a851fb6");
    CHECK(rocke_salted_kernel_name("k", Pairs{{"a", "1"}, {"b", "(2, 'x')"}}) == "k_420a851fb6");
    /* equal names keep their input order (sorted() is stable) */
    CHECK(rocke_salted_kernel_name(
              "k", Pairs{{"dkv_mode", "'atomic'"}, {"b", "(2, 'x')"}, {"a", "1"}, {"b", "2"}})
          == "k_a68d0461ba");
    return 0;
}

static int test_refusals()
{
    const char* names[] = {"not a name"};
    const char* reprs[] = {"1"};
    char out[64];
    CHECK(rocke_salted_kernel_name("k", names, reprs, 1, out, sizeof(out), NULL)
          == ROCKE_ERR_VALUE);
    const char* ok[] = {"a"};
    /* "k_" + 10 digits needs 13 bytes with the NUL */
    CHECK(rocke_salted_kernel_name("k", ok, reprs, 1, out, 12, NULL) == ROCKE_ERR_VALUE);
    size_t len = 0;
    CHECK(rocke_salted_kernel_name("k", ok, reprs, 1, out, 13, &len) == ROCKE_OK);
    CHECK(len == 12);
    bool threw = false;
    try
    {
        rocke_salted_kernel_name("k", Pairs{{"a-b", "1"}});
    }
    catch(const ckc::ValueError&)
    {
        threw = true;
    }
    CHECK(threw);
    return 0;
}

int main()
{
    if(test_sha1() || test_salted_name() || test_refusals())
        return 1;
    std::printf("salted_kernel_name: ok\n");
    return 0;
}
