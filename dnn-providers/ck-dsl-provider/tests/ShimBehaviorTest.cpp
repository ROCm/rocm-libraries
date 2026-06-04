// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstring>
#include <mutex>
#include <string>

#include "python/EmbeddedInterpreter.hpp"

extern "C" {
#include "py/compile.h"
#include "py/lexer.h"
#include "py/parse.h"
#include "py/runtime.h"
}

// Behavioral tests for the hand-written stdlib shims (micropython/shims/*.py).
//
// The shims replace CPython stdlib modules that ck_dsl codegen depends on
// (dataclasses, functools, itertools, typing, os, time, ...). The compat lint
// (check_compat.py) only proves the ck_dsl SOURCE parses under MicroPython; it
// does NOT exercise shim BEHAVIOR. A subtle shim bug (wrong field order, a
// shared mutable default, eviction where there should be none) would produce a
// wrong-but-non-empty kernel that the host-only smoke tests cannot catch.
//
// These tests run Python source through the SAME embedded interpreter the
// provider ships (frozen shims, the embed mpconfigport.h), so they validate the
// real runtime contract. The runtime compiler is available -- mpconfigport.h
// sets MICROPY_ENABLE_COMPILER unconditionally -- so we can compile + run source
// strings directly rather than freezing a test module into the plugin.

using ck_dsl_provider::EmbeddedInterpreter;

namespace {

struct ExecResult {
    bool ok;
    std::string error;
};

// Format a MicroPython exception object into a fixed buffer, under its own nlr
// guard (POD-only, no C++ objects in the protected region). Mirrors
// CompileServiceBridge::raiseFromMpException's formatting.
void formatMpException(mp_obj_t exc, char* buf, size_t bufSize) {
    buf[0] = '\0';
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        vstr_t vstr;
        mp_print_t print;
        vstr_init_print(&vstr, 200, &print);
        mp_obj_print_helper(&print, exc, PRINT_EXC);
        size_t len = vstr_len(&vstr);
        if (len >= bufSize) {
            len = bufSize - 1;
        }
        std::memcpy(buf, vstr_str(&vstr), len);
        buf[len] = '\0';
        vstr_clear(&vstr);
        nlr_pop();
    }
}

// Compile and execute a Python source string in the embedded interpreter,
// holding the interpreter lock and resetting the GC C-stack root-scan top to
// this frame (the discipline every entry point that runs Python must follow --
// MicroPython has one global runtime, no GIL). Returns ok=false plus the
// formatted exception text on any raise (an `assert` failure surfaces as
// AssertionError here). The compile/parse/exec sequence matches
// ports/embed mp_embed_exec_str, but propagates failures to the caller instead
// of printing-and-swallowing them.
ExecResult execPython(const char* src) {
    EmbeddedInterpreter::ensureInitialized();
    std::lock_guard<std::mutex> lock(EmbeddedInterpreter::interpreterMutex());
    EmbeddedInterpreter::setCallStackTop(__builtin_frame_address(0));

    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t* lex =
            mp_lexer_new_from_str_len(qstr_from_str("<shim-test>"), src, std::strlen(src), 0);
        qstr sourceName = lex->source_name;
        mp_parse_tree_t parseTree = mp_parse(lex, MP_PARSE_FILE_INPUT);
        mp_obj_t moduleFun = mp_compile(&parseTree, sourceName, /*is_repl=*/false);
        mp_call_function_0(moduleFun);
        nlr_pop();
        return {true, std::string()};
    }
    char buf[512];
    formatMpException(MP_OBJ_FROM_PTR(nlr.ret_val), buf, sizeof(buf));
    return {false, std::string(buf[0] != '\0' ? buf : "<python exception>")};
}

}  // namespace

// Each test runs a self-contained Python snippet whose `assert`s encode the
// required behaviour; a failing assert surfaces as ok=false with the message.
#define RUN_PY(src)                      \
    do {                                 \
        ExecResult _r = execPython(src); \
        EXPECT_TRUE(_r.ok) << _r.error;  \
    } while (0)

// ---------------------------------------------------------------------------
// Object model: object.__setattr__ must work. ck_dsl's frozen-dataclass
// __init__s (transforms.py / tensor_view.py / spec.py) set fields via the
// canonical object.__setattr__ idiom; mpconfigport.h enables
// MICROPY_PY_DELATTR_SETATTR for exactly this. A regression flipping it off
// would break conv lowering -- pin it.
// ---------------------------------------------------------------------------
TEST(TestShimObjectModel, ObjectSetattrEnabled) {
    RUN_PY(R"py(
class C:
    pass
o = C()
object.__setattr__(o, 'x', 5)
assert o.x == 5, 'object.__setattr__ did not set the attribute'
)py");
}

// ---------------------------------------------------------------------------
// dataclasses shim -- the highest-use, highest-divergence shim. Convention
// (enforced by check_compat for ck_dsl): every field is declared `= field(...)`
// because MicroPython erases bare annotations and its dicts are unordered.
// ---------------------------------------------------------------------------

// Field order is recovered from field()'s creation counter, not dict order:
// positional construction must map args in source-declaration order.
TEST(TestShimDataclasses, DeclarationOrderRecovered) {
    RUN_PY(R"py(
from dataclasses import dataclass, field, fields
@dataclass(frozen=True)
class P:
    a: int = field()
    b: int = field()
    c: int = field()
p = P(1, 2, 3)
assert (p.a, p.b, p.c) == (1, 2, 3), 'positional args mapped out of order'
names = tuple(f.name for f in fields(p))
assert names == ('a', 'b', 'c'), 'field order wrong: %s' % (names,)
)py");
}

// Single-level inheritance: base fields first, then own, ordered by the global
// counter (the Transform base + subclass pattern that pervades ck_dsl).
TEST(TestShimDataclasses, InheritanceBaseThenOwnOrder) {
    RUN_PY(R"py(
from dataclasses import dataclass, field, fields
@dataclass
class Base:
    x: int = field()
    y: int = field()
@dataclass
class Sub(Base):
    z: int = field()
s = Sub(1, 2, 3)
assert (s.x, s.y, s.z) == (1, 2, 3), 'inherited fields mis-ordered'
names = tuple(f.name for f in fields(s))
assert names == ('x', 'y', 'z'), 'order: %s' % (names,)
)py");
}

// replace() returns a new instance with overrides applied and other fields
// copied; the original is untouched.
TEST(TestShimDataclasses, ReplaceCopiesAndOverrides) {
    RUN_PY(R"py(
from dataclasses import dataclass, field, replace
@dataclass(frozen=True)
class P:
    a: int = field()
    b: int = field()
p = P(1, 2)
q = replace(p, b=9)
assert (q.a, q.b) == (1, 9), 'replace did not apply override / copy'
assert (p.a, p.b) == (1, 2), 'replace mutated the original'
)py");
}

// default_factory produces a fresh object per instance (no shared mutable
// default) -- core/ir.py relies on this for operands/results/attrs lists.
TEST(TestShimDataclasses, DefaultFactoryFreshPerInstance) {
    RUN_PY(R"py(
from dataclasses import dataclass, field
@dataclass
class P:
    items: list = field(default_factory=list)
a = P()
b = P()
a.items.append(1)
assert a.items == [1]
assert b.items == [], 'default_factory shared a mutable default across instances'
)py");
}

// __post_init__ runs after fields are set, and a raising __post_init__
// propagates (ck_dsl validators, e.g. TensorDescriptor, rely on both).
TEST(TestShimDataclasses, PostInitInvokedAndPropagates) {
    RUN_PY(R"py(
from dataclasses import dataclass, field
seen = []
@dataclass
class P:
    a: int = field()
    def __post_init__(self):
        seen.append(self.a)
P(7)
assert seen == [7], '__post_init__ not called after fields set'

@dataclass
class Q:
    a: int = field()
    def __post_init__(self):
        raise ValueError('bad')
raised = False
try:
    Q(1)
except ValueError:
    raised = True
assert raised, 'raising __post_init__ did not propagate'
)py");
}

// frozen=True yields a working value-based __hash__ (equal instances hash equal,
// usable as dict keys) and a type-aware __eq__.
TEST(TestShimDataclasses, FrozenHashAndEq) {
    RUN_PY(R"py(
from dataclasses import dataclass, field
@dataclass(frozen=True)
class P:
    a: int = field()
    b: int = field()
p1 = P(1, 2)
p2 = P(1, 2)
p3 = P(1, 3)
assert p1 == p2 and p1 != p3, 'frozen __eq__ wrong'
assert hash(p1) == hash(p2), 'equal frozen instances must hash equal'
d = {p1: 'x'}
assert d[p2] == 'x', 'frozen instance not usable as an equal dict key'
class Other:
    pass
assert not (p1 == Other()), '__eq__ not type-aware'
)py");
}

// Deliberate divergence (locked): the shim does NOT enforce immutability -- a
// plain field assignment on a frozen instance must NOT raise. ck_dsl relies on
// hashability/eq, not on frozen raising. Locking this catches a future
// "CPython-parity fix" that would silently change embed behaviour.
TEST(TestShimDataclasses, FrozenDoesNotEnforceImmutability) {
    RUN_PY(R"py(
from dataclasses import dataclass, field
@dataclass(frozen=True)
class P:
    a: int = field()
p = P(1)
p.a = 99
assert p.a == 99, 'shim unexpectedly enforced frozen immutability'
)py");
}

// A user-defined __init__ is not overwritten by the decorator (CPython parity;
// ck_dsl's Transform subclasses define their own __init__).
TEST(TestShimDataclasses, UserInitRespected) {
    RUN_PY(R"py(
from dataclasses import dataclass, field
@dataclass
class P:
    a: int = field()
    def __init__(self, a):
        self.a = a * 2
p = P(5)
assert p.a == 10, 'decorator overwrote a user-defined __init__'
)py");
}

// Missing a required field raises TypeError.
TEST(TestShimDataclasses, MissingRequiredArgRaises) {
    RUN_PY(R"py(
from dataclasses import dataclass, field
@dataclass
class P:
    a: int = field()
    b: int = field()
raised = False
try:
    P(1)
except TypeError:
    raised = True
assert raised, 'missing required field did not raise TypeError'
)py");
}

// ---------------------------------------------------------------------------
// functools shim
// ---------------------------------------------------------------------------

// lru_cache memoizes; the bare (no-parens) decorator form works.
TEST(TestShimFunctools, LruCacheMemoizesBareForm) {
    RUN_PY(R"py(
import functools
calls = []
@functools.lru_cache
def f(x):
    calls.append(x)
    return x * x
assert f(3) == 9
assert f(3) == 9
assert calls == [3], 'lru_cache (bare form) did not memoize: %s' % (calls,)
)py");
}

// Deliberate divergence (locked): maxsize is ignored -- the cache never evicts,
// so re-calling an "evicted" key does NOT re-invoke the function. Safe here
// because ck_dsl's cached values (target.py arch lookups) are singletons.
TEST(TestShimFunctools, LruCacheMaxsizeIgnoredNoEviction) {
    RUN_PY(R"py(
import functools
calls = []
@functools.lru_cache(maxsize=1)
def h(x):
    calls.append(x)
    return x
h(1)
h(2)
h(1)
h(2)
assert calls == [1, 2], 'maxsize=1 evicted; shim should ignore maxsize: %s' % (calls,)
)py");
}

// Keyword arguments are part of the cache key.
TEST(TestShimFunctools, LruCacheKwargsKeying) {
    RUN_PY(R"py(
import functools
calls = []
@functools.lru_cache
def k(a, b=0):
    calls.append((a, b))
    return a + b
assert k(1, b=2) == 3
assert k(1, b=2) == 3
assert k(1, b=3) == 4
assert calls == [(1, 2), (1, 3)], 'kwargs not part of cache key: %s' % (calls,)
)py");
}

// ---------------------------------------------------------------------------
// itertools shim
// ---------------------------------------------------------------------------

// product: order (rightmost-fastest, CPython parity), repeat=, and empty input.
TEST(TestShimItertools, ProductOrderRepeatEmpty) {
    RUN_PY(R"py(
import itertools
r = list(itertools.product([1, 2], [3, 4]))
assert r == [(1, 3), (1, 4), (2, 3), (2, 4)], 'product order wrong: %s' % (r,)
r2 = list(itertools.product([0, 1], repeat=2))
assert r2 == [(0, 0), (0, 1), (1, 0), (1, 1)], 'product repeat= wrong: %s' % (r2,)
r3 = list(itertools.product())
assert r3 == [()], 'empty product should be a single empty tuple: %s' % (r3,)
)py");
}

// ---------------------------------------------------------------------------
// Trivial shims -- mostly locking deliberate "no real implementation"
// behaviour so a future change toward CPython parity fails loudly.
// ---------------------------------------------------------------------------

// typing names import, subscript as no-ops, and TYPE_CHECKING is False.
TEST(TestShimMisc, TypingNoOpsAndTypeChecking) {
    RUN_PY(R"py(
import typing
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
assert TYPE_CHECKING is False, 'TYPE_CHECKING must be False under the shim'
_ = List[int]
_ = Dict[str, int]
_ = Optional[int]
_ = Tuple[int, ...]
)py");
}

// os.getenv is default-only (no environment) -- locks the branch ck_dsl's
// lower_llvm takes for CK_DSL_LLVM_FLAVOR.
TEST(TestShimMisc, OsGetenvDefaultOnly) {
    RUN_PY(R"py(
import os
assert os.getenv('CK_DSL_NONEXISTENT_VAR_XYZ') is None
assert os.getenv('CK_DSL_NONEXISTENT_VAR_XYZ', 'dflt') == 'dflt'
)py");
}

// time clock is a constant 0.0 -- locks the "timings read as zero in embed"
// behaviour the comgr compile path produces.
TEST(TestShimMisc, TimeClockIsConstantZero) {
    RUN_PY(R"py(
import time
assert time.perf_counter() == 0.0
assert time.monotonic() == 0.0
)py");
}

// The host-only stub shims must IMPORT cleanly (they sit in ck_dsl's import
// graph via the unused hipcc backend) and raise OSError when actually used.
TEST(TestShimMisc, StubShimsImportAndRaiseWhenUsed) {
    RUN_PY(R"py(
import subprocess
import tempfile
import __future__
raised = False
try:
    subprocess.run(['noop'])
except OSError:
    raised = True
assert raised, 'subprocess.run should raise OSError in the embed'
raised = False
try:
    tempfile.TemporaryDirectory()
except OSError:
    raised = True
assert raised, 'tempfile.TemporaryDirectory should raise OSError in the embed'
)py");
}
