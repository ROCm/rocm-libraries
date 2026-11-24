// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "rocke/sha256.h"

#include <cstdio>
#include <cstring>

namespace
{
int failures = 0;
#define CHECK(cond)                                                              \
    do                                                                           \
    {                                                                            \
        if(!(cond))                                                              \
        {                                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while(0)

void check_sha(const void* data, size_t size, const char* expected)
{
    uint8_t digest[34];
    std::memset(digest, 0xa5, sizeof(digest));
    rocke_sha256(data, size, digest + 1);
    CHECK(digest[0] == 0xa5 && digest[33] == 0xa5);
    char hex[65];
    for(size_t i = 0; i < 32; ++i)
    {
        std::snprintf(hex + 2 * i, 3, "%02x", (unsigned)digest[i + 1]);
    }
    if(std::strcmp(hex, expected) != 0)
    {
        std::fprintf(stderr, "SHA-256 size %zu: got %s, expected %s\n", size, hex, expected);
        ++failures;
    }
}
} // namespace

int main()
{
    // Published NIST SHA-256 examples: empty, abc, 448-bit and 896-bit messages.
    check_sha(nullptr, 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check_sha("abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char m448[] = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    check_sha(
        m448, sizeof(m448) - 1, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const char m896[] = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
                        "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    check_sha(
        m896, sizeof(m896) - 1, "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");

    // Binary inputs (including NUL), independently generated with hashlib;
    // lengths straddle the padding-length field and complete-block boundaries.
    uint8_t bytes[129];
    for(size_t i = 0; i < sizeof(bytes); ++i)
    {
        bytes[i] = (uint8_t)i;
    }
    const struct
    {
        size_t size;
        const char* expected;
    } boundaries[] = {{55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
                      {56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
                      {63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
                      {64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
                      {65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
                      {119, "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"},
                      {120, "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"},
                      {127, "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976"},
                      {128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"}};
    for(const auto& b : boundaries)
    {
        // Offset the data to exercise unaligned input as well.
        std::memmove(bytes + 1, bytes, b.size);
        check_sha(bytes + 1, b.size, b.expected);
        for(size_t i = 0; i < sizeof(bytes); ++i)
        {
            bytes[i] = (uint8_t)i;
        }
    }
    if(!failures)
    {
        std::puts("SHA-256: PASS");
    }
    return failures ? 1 : 0;
}
