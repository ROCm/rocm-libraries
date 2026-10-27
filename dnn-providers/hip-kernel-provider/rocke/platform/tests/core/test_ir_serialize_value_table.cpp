// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_ir_serialize_value_table.cpp -- the name tables of the C++
 * ck.dsl.ir/v1 reader (cpp/core/ir/serialize.cpp), which mirror the Python
 * parser's `values` dict:
 *
 *   - a definition is visible to every later op of its scope, and the table
 *     grows past any initial size;
 *   - an scf.for's counter and carried values are visible in its body only,
 *     so sibling loops can reuse the names, and a body name that shadowed an
 *     outer one retires it too (Python pops the key);
 *   - redefining an id is an error;
 *   - rocke_ir_canonicalize numbers ids by first definition, so it is
 *     idempotent and blind to a consistent renaming.
 *
 * Exit 0 = all pass.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"

namespace
{

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    failures += ok ? 0 : 1;
}

const std::string kHead = "ckdsl.ir v1\nkernel @vt {\n  params {\n    %P : ptr<i32,global>\n"
                          "    %n : i32\n  }\n  region @entry {\n";
const std::string kTail = "    cf.return (  )\n  }\n}\n";

std::string op(const std::string& line)
{
    return "    " + line + "\n";
}

/* An scf.for over [0, %n) with counter `iv`, one carried i32 `arg` (initial
 * value `init`), whose body yields `res_next = arg + iv`; its result is `res`.
 * Body results stay defined after the loop, so each loop names its own. */
std::string loop(const std::string& res,
                 const std::string& iv,
                 const std::string& arg,
                 const std::string& init)
{
    return "    " + res + " : i32 = scf.for ( %zero, %n, %one, " + init
           + " ) { iter_args = l:[ { name = s:\"" + arg + "\", type = s:\"i32\" } ], iv = s:\"" + iv
           + "\", iv_type = s:\"i32\", num_iter_args = i:1 }\n" + "      region @body {\n        "
           + res + "_next : i32 = arith.add ( " + arg + ", " + iv + " )\n        scf.yield ( " + res
           + "_next ) { num = i:1 }\n      }\n";
}

const std::string kConsts
    = op("%zero : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }")
      + op("%one : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }");

/* Parse `text`; on success also returns serialize(parse(text)) in `back`. */
bool parse(const std::string& text, std::string* back, std::string* error)
{
    rocke_ir_builder_t b = {};
    rocke_ir_builder_init(&b, "vt");
    rocke_kernel_def_t* k = nullptr;
    const bool ok = rocke_ir_parse(text.c_str(), &b, &k) == ROCKE_OK && k != nullptr;
    if(!ok && error)
        *error = rocke_ir_builder_error(&b) ? rocke_ir_builder_error(&b) : "";
    if(ok && back)
    {
        char* s = nullptr;
        rocke_ir_serialize(k, &s);
        *back = s ? s : "";
        std::free(s);
    }
    rocke_ir_builder_free(&b);
    return ok;
}

std::string canonicalize(const std::string& text)
{
    char* out = nullptr;
    std::string s;
    if(rocke_ir_canonicalize(text.c_str(), &out) == ROCKE_OK && out)
        s = out;
    std::free(out);
    return s;
}

/* Every SSA id `%x` (params, results, counters, carried names in attrs)
 * becomes `%r_x`: a consistent renaming. */
std::string rename_ids(const std::string& text)
{
    std::string out;
    for(char c : text)
    {
        out += c;
        if(c == '%')
            out += "r_";
    }
    return out;
}

} // namespace

int main()
{
    /* 1. A long def-use chain: each op reads the previous result and the
     * first one, so lookups hit both ends of a table that has grown many
     * times. */
    std::string chain = kHead + op("%v0 : i32 = arith.add ( %n, %n )");
    for(int i = 1; i < 3000; ++i)
        chain += op("%v" + std::to_string(i) + " : i32 = arith.add ( %v" + std::to_string(i - 1)
                    + ", %v0 )");
    chain
        += op("memref.global_store_typed ( %P, %n, %v2999 ) { align = i:4, elem_type = s:\"i32\" }")
           + kTail;
    std::string back, error;
    check(parse(chain, &back, &error) && back == chain, "3000-value chain round-trips");

    /* 2. Sibling loops reuse their counter and carried names. */
    const std::string siblings = kHead + kConsts + loop("%a", "%k", "%acc", "%zero")
                                 + loop("%b", "%k", "%acc", "%a")
                                 + op("memref.global_store_typed ( %P, %n, %b ) { align = i:4, "
                                      "elem_type = s:\"i32\" }")
                                 + kTail;
    check(parse(siblings, &back, &error) && back == siblings, "sibling loops reuse block names");

    /* 3. A counter is gone once its loop is parsed. */
    const std::string escaped = kHead + kConsts + loop("%a", "%k", "%acc", "%zero")
                                + op("memref.global_store_typed ( %P, %n, %k ) { align = i:4, "
                                     "elem_type = s:\"i32\" }")
                                + kTail;
    check(!parse(escaped, nullptr, &error)
              && error.find("used before definition") != std::string::npos,
          "a counter used after its loop is rejected");

    /* 4. A carried name that shadows an outer value retires the outer one. */
    const std::string shadow = kHead + kConsts + op("%acc : i32 = arith.add ( %n, %n )")
                               + loop("%a", "%k", "%acc", "%acc")
                               + op("memref.global_store_typed ( %P, %n, %acc ) { align = i:4, "
                                    "elem_type = s:\"i32\" }")
                               + kTail;
    check(!parse(shadow, nullptr, &error)
              && error.find("used before definition") != std::string::npos,
          "a shadowed outer name is retired with the body's");

    /* 5. Redefinition. */
    const std::string twice = kHead + op("%x : i32 = arith.add ( %n, %n )")
                              + op("%x : i32 = arith.add ( %n, %n )") + kTail;
    check(!parse(twice, nullptr, &error) && error.find("redefined") != std::string::npos,
          "a redefined id is rejected");

    /* 6. Canonical ids. */
    const std::string canon = canonicalize(siblings);
    check(!canon.empty() && canonicalize(canon) == canon, "canonicalize is idempotent");
    check(canonicalize(rename_ids(siblings)) == canon,
          "canonicalize ignores a consistent renaming");
    const std::string canon_chain = canonicalize(chain);
    check(canon_chain.find("%3001 : i32 = arith.add ( %3000, %2 )") != std::string::npos,
          "ids number params first, then results by definition");

    std::printf(failures ? ">> VALUE TABLE FAILURES\n" : ">> ALL VALUE TABLE CHECKS PASSED\n");
    return failures ? 1 : 0;
}
