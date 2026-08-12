# Hardening: what each proof exists to stop

Status: proposed design, prototype-backed. Each hardening step below is real code.
Executable capabilities name the `ctest` that proves them; where a proof is partial or a
control is still future work, the step says so and cites
[07-status-and-roadmap.md](07-status-and-roadmap.md). Run one with
`ctest --test-dir <build> -R <name> --output-on-failure`.

The [architecture](01-architecture.md) describes a versioned boundary. A boundary is only
real if the symbols actually behave - if the version script exports what it says, the nodes
attach where they should, and the loader survives being hammered by threads. This chapter
walks each hardening step in the order it was built. For each: the threat first, then the
fix, then the test that would fail if the fix regressed.

`tests/CMakeLists.txt` defines the suite (grep `add_test`); how many register depends on
which optional linkers and sanitizers your toolchain offers. Cite test names, not counts -
names do not drift. A canonical amdclang++/ld.lld build registers the full applicable set
green; the exact number is a moving target and deliberately not pinned here.

Platform scope. Every proof in this chapter is Linux/ELF-specific. The export, version-node,
co-residency, ldconfig, and linker-guard tests are registered only under UNIX AND NOT APPLE in
tests/CMakeLists.txt, and the mechanism they exercise (ELF version scripts, SONAMEs, dlvsym,
ldconfig) has no Windows/PE or macOS/Mach-O analogue here. Windows/PE and Darwin are out of
scope and unproven; the DLL/PE ABI-versioning mechanism is not addressed by this contract.

## 1. Stop the provider leaking libstdc++ symbols (RES-02)

**Threat.** A recording provider is built with `CXX_VISIBILITY_PRESET hidden`, so it looks
locked down. It is not. Its runtime headers pull `std::filesystem` out-of-line symbols from
the libstdc++ archive, and those are default-visibility inside the archive - the whole-TU
`-fvisibility=hidden` flag never touches them. Result: roughly 170 leaked `std::` symbols in
the provider's `.dynsym`. Two such providers in one process share those symbols through the
dynamic loader, and one silently runs the other's code.

**Fix.** An explicit export allowlist. `providers/recording/recording_provider.map` names
the single symbol a provider may expose and sends everything else to `local: *`. The
`--version-script` is applied by the `add_recording_provider()` function in
`providers/recording/CMakeLists.txt`, so it covers all recording providers and the rocBLAS
bridge target uniformly.

**Proof.** `rocm_interfaces.exports` builds the provider DSOs on a fixed, manually
maintained list (the `-D<NAME>_PROVIDER=` arguments in `tests/CMakeLists.txt` and the
matching `foreach` in `tests/check_exports.cmake`) and asserts each exports exactly one
defined, non-absolute dynamic symbol (`rocm_interfaces_provider_query_v1`) - the checker
runs `nm -D --defined-only --with-symbol-versions` and ignores the absolute version-node
entry, so it is one callable export, not 176. A provider not on that list is not inspected
until it is added in both places (auto-derivation is COMMITTED-NEXT; see
[07-status-and-roadmap.md](07-status-and-roadmap.md#committed-next-the-immediate-plan)).
Landed in commit `a929517` (`fix(interfaces): stop recording providers leaking libstdc++
symbols`).

## 2. Give the exports names, so majors can coexist (ABI-03, named nodes)

**Threat.** Hiding the leak is not enough for co-residency. If the loader exports a bare
`rocblas_sgemm`, then when two majors load in one process the dynamic loader binds every
caller to whichever it saw first. The allowlist controls *what* is exported; it does not
control *which definition wins*.

**Fix.** Named ELF version nodes. The generated `rocblas_bridge.map` tags the bridge loader
entry points with `ROCBLAS_ABI_5`, so they emit as `rocblas_sgemm@@ROCBLAS_ABI_5` and friends;
the static `loader/rocblas_loader.map` applies the same `ROCBLAS_ABI_5` node to the 11-symbol
narrow loader (`rocblas_narrow_loader_shadow`). The `exports` check was upgraded from a
symbol-count assertion to a version-node assertion at the same time.

**Proof.** `rocm_interfaces.exports` (now node-aware). It inspects `rocblas_loader_shadow` and
`rocblas_narrow_v2_loader_shadow` (both built from the generated `rocblas_bridge.map`) plus the
provider DSOs; it does not inspect `rocblas_narrow_loader_shadow`, the target that carries the
static 11-symbol `loader/rocblas_loader.map`. That narrow loader is exercised today only by the
behavioral `rocm_interfaces.rocblas_narrow_shadow` test, so its node-versioning is not asserted
by the exports check (NEXT, tracked in
[07-status-and-roadmap.md](07-status-and-roadmap.md)). Landed in commit `ba093ad`
(`feat(interfaces): assign named version nodes to loader and provider exports`).

## 3. Show the nodes hold across co-resident majors (ABI-03, co-residency)

**Threat.** A named node is only worth having if co-resident majors actually resolve their
own definition and the nodeless case is genuinely worse. Assert nothing and you are
trusting the linker on faith.

**Fix and proof.** Two runtime tests, built from `abi03_fixture_rocblas.cpp` with three
maps - `abi03_provA.map` (`ROCBLAS_ABI_6`), `abi03_provB.map` (`ROCBLAS_ABI_7`), and
`abi03_anon.map` (no node, the negative control):

- `rocm_interfaces.abi03_coresidency` loads two majors in one process and asserts each
  `rocblas_sgemm` resolves to its own node - no cross-binding.
- `rocm_interfaces.abi03_interpose_hazard` is the discriminating half: with the anonymous
  (nodeless) build, the interposition **reproduces**. If the node mechanism were inert this
  test would not be able to show the hazard, so its passing proves the nodes are what
  prevent it.

Landed in commit `8832c9a` (`test(interfaces): prove ABI version-node co-residency defeats
interposition`).

## 4. Refuse a toolchain that cannot stamp version nodes (RES-03)

**Threat.** The version nodes are stamped by the linker. GCC link-time optimization with the
LLVM linker (lld) cannot resolve the version-script symbol-to-node assignments out of GCC LTO
IR - lld carries no GCC LTO plugin. The recorded RES-03 spike shows this pairing fails hard at
link time: ld.lld emits 'version script assignment of ROCBLAS_ABI_5 to symbol rocblas_sgemm
failed: symbol not defined' for every versioned symbol and produces no DSO (res03/build-lld-fat/err5
in the exec spike). Without a guard, this surfaces as a confusing raw linker error deep in an
otherwise ordinary build rather than a named, actionable failure - and any future lld change
that resolved the symbols instead of erroring would silently emit unversioned exports.

**Fix.** Fail the build instead. `rocm_interfaces_assert_lto_linker_supported()` in
`cmake/rocm_interfaces_lto_linker_guard.cmake` detects LTO (IPO or `-flto` in flags) plus
lld (`CMAKE_LINKER_TYPE=LLD`, `-fuse-ld=lld`, `--ld-path=*lld`, or a matching
`CMAKE_LINKER`) plus a non-Clang compiler, and raises `FATAL_ERROR` naming RES-03. It is
included and called in `interfaces/CMakeLists.txt`, so it is real enforcement, not a lint.

**Proof.** A `cmake -P` driver (`tests/check_lto_linker_guard.cmake`) exercises four cases.
Each case sets synthetic toolchain variables (`CMAKE_CXX_COMPILER_ID`, IPO, and linker flags)
and runs the `rocm_interfaces_assert_lto_linker_supported()` predicate, so these tests validate
the guard's detection logic; they do not themselves invoke a compiler or linker:

- `rocm_interfaces.lto_linker_guard_rejects_gnu_lld` - the dangerous combination, marked
  `WILL_FAIL TRUE`.
- `rocm_interfaces.lto_linker_guard_accepts_clang_lld` - Clang carries the plugin, allowed.
- `rocm_interfaces.lto_linker_guard_accepts_gnu_bfd` - GCC with GNU ld, allowed.
- `rocm_interfaces.lto_linker_guard_accepts_gnu_lld_without_lto` - no LTO, nothing to drop,
  allowed.

Landed in commit `3e56e73` (`feat(interfaces): guard against lld + LTO with a non-Clang
compiler`).

## 5. Lock the loader against concurrency regressions (OPS-04)

**Threat.** The provider registry and loader are shared across threads. A data race there -
two contexts selecting at once, a module refcount torn - corrupts every dispatch that
follows, and races do not fail deterministically, so a unit test can pass a thousand times
and still be broken.

**Fix and proof.** `tests/ops04_concurrency_test.cpp` drives five concurrent scenarios:
shared-registry `select`+`add_module`, multi-stream/multi-device dispatch, the bridge's
`call_once`, a hot-path in-flight probe, and the jit-cache forward-obligation double. It is
registered as `rocm_interfaces.ops04_concurrency` only under UNIX with a ThreadSanitizer
build (`ROCM_INTERFACES_SANITIZE=thread`), with `TSAN_OPTIONS=halt_on_error=0:exitcode=66`
so a race fails the test. Non-vacuity (one-time observation, not asserted by CTest): a local
TSan build's binary carried 171 `__tsan_` symbols, confirming it was genuinely instrumented
rather than a silent non-sanitized fallback; the registered test does not itself count
`__tsan_` symbols and relies on `TSAN_OPTIONS=halt_on_error=0:exitcode=66` to fail on a race.
Landed in commit
`df8512b` (`test(interfaces): regression-lock loader/registry concurrency under TSan`).

## 6. Prove the versioning holds across every symbol shape (ABI-01/02)

The nodes work for ordinary function symbols (steps 2-3). The remaining risk is that they
quietly fail on some *other* shape of symbol. This is the ABI-01/02 proof suite. Each test
follows the non-vacuity recipe in
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md): positive, negative
control, genuineness, except the ASan case (6d), whose node assertion is a substring match
with no negative control (see 6d).

### 6a. The core invariants (commit `897293e`)

Built from the ABI-03 fixture with `abi04_rb5.map` (`ROCBLAS_ABI_5`) plus the provA/provB
maps, producing `librocblas.so.{5,6,7}` and `-Bsymbolic` variants:

- `rocm_interfaces.abi04_three_line_order` - the three `dlvsym` lines resolve to `ABI_5/6/7`
  in both load orders, and each handle's cross-node lookup (the wrong node on the same symbol)
  returns null. That cross-node-nil assertion is what makes it discriminating: it is invoked
  only with the distinct `rb5`/`rb6`/`rb7` DSOs. `rocm_interfaces.abi04_same_node_negative`
  (commit `215ede4`) is the dedicated discrete negative control: three DSOs are built on the
  shared `ROCBLAS_ABI_6` node (with `ABI_VER` 5/6/7), each resolves its own value on that node,
  and the `ABI_5`/`ABI_7` lookups are nil everywhere. Fed the distinct-node DSOs instead, the
  same-node mode fails - so the control is not vacuous.
- `rocm_interfaces.abi04_bsymbolic_inert` - loads two `-Bsymbolic` DSOs plus a plain comparison
  DSO and asserts each `rocblas_sgemm` resolves to its own node with cross-node nil (the
  co-residency check), then reads `DT_FLAGS`/`DT_SYMBOLIC` via `dlinfo(RTLD_DI_LINKMAP)` and
  requires `DF_SYMBOLIC` present on both `-Bsymbolic` DSOs and absent on the plain one (commit
  `215ede4`). This is now a discriminating proof: the co-residency outcome is identical with or
  without `-Bsymbolic`, but the `DF_SYMBOLIC` delta fails if the flag is dropped from a bsym
  target or wrongly applied to the plain one - confirmed by feeding the plain DSO where a bsym
  is expected (the run fails). It inspects the ELF flag directly rather than adding an internal
  interposable call to the fixture; the conclusion is that versioning, not `-Bsymbolic`, is the
  co-residency mechanism.
- `rocm_interfaces.abi04_multiple_default_def_rejected` - links a version script that names
  `rocblas_sgemm` as a default (`global`) symbol in two nodes (`ROCBLAS_ABI_6` and
  `ROCBLAS_ABI_7`) and observes the toolchain's response. It passes on any link failure (read
  as the linker rejecting the duplicate default definition) and, when the link succeeds, fails
  only if `nm` reports more than one `rocblas_sgemm@@` definition; a zero- or one-`@@` result
  also passes. It does not synthesize a genuine two-`@@` DSO, so today it confirms the linker's
  own behavior rather than independently proving a two-default-definition DSO is rejected.
  Forcing a real two-`@@` input is COMMITTED-NEXT (see
  [07-status-and-roadmap.md](07-status-and-roadmap.md)).
- `rocm_interfaces.abi04_ldconfig_stub_preserved` - `ldconfig -n` leaves the
  `librocblas.so -> librocblas.so.6` stub intact and the `ROCBLAS_ABI_6` node survives.

### 6b. The same invariants under the second linker (commit `01c14eb`)

`add_abi04_provider` gained an optional linker argument and a configure-time
`ROCM_INTERFACES_HAVE_LLD` probe. When lld is present, five lld-built mirrors run:
`abi04_three_line_order_lld`, `abi04_same_node_negative_lld`, `abi04_bsymbolic_inert_lld`,
`abi04_multiple_default_def_rejected_lld`, `abi04_ldconfig_stub_preserved_lld`.
Non-vacuity (one-time observation, not asserted by CTest): the lld DSO's `.comment` stamps
`Linker: AMD LLD`, which a bfd DSO lacks - evidence it is genuinely lld and not a silent bfd
fallback. The registered `abi04_*_lld` tests are gated on the configure-time
`ROCM_INTERFACES_HAVE_LLD` probe and do not themselves inspect `.comment`.

### 6c. A real data object (commit `04ade2b`)

**Threat.** A version node might attach to functions but not to a data symbol.
**Proof.** `rocm_interfaces.abi06_data_version_node` versions a DSO built from the real
`sobol32` precomputed table (`abi06_data.map`, `ROCBLAS_ABI_6`) and asserts at runtime that
`dlsym`/`dlvsym` reach the exact node and that the first element is `0x80000000`; the
wrong-node lookup returns null.

### 6d. A build under AddressSanitizer (commit `0081ba5`)

**Threat.** ASan rewrites the binary; the node could be lost in the process.
**Proof (partial - node definition, not per-symbol binding).**
`rocm_interfaces.abi04_asan_version_node_survives` builds an `-fsanitize=address` DSO (via
`ROCM_INTERFACES_SANITIZE=address`), asserts the DSO carries `__asan_` symbols (genuineness, so
a no-ASan fallback cannot pass it), and asserts the string `ROCBLAS_ABI_6` appears in
`nm -D --with-symbol-versions` output. That second assertion is a loose substring match: the
linker emits an absolute node symbol `ROCBLAS_ABI_6@@ROCBLAS_ABI_6` whenever the version node
is defined, so the match passes on the node metadata alone and does not prove the `@@` default
binding on `rocblas_sgemm` the way 6a/6c/6e do, nor distinguish `@@` (default) from `@`
(non-default). Unlike the other node tests it has no negative control. Tightening it to assert
`rocblas_sgemm@@ROCBLAS_ABI_6` explicitly is tracked in
[07-status-and-roadmap.md](07-status-and-roadmap.md).

### 6e. C++ mangled names and RTTI (commit `769f4aa`, tightened in `f681132`)

**Threat.** This is the hardest shape. C++ methods, and especially the RTTI vtable/typeinfo,
are emitted as default-visibility weak symbols with mangled names; a naive map misses them.
**Fix and proof.** `rocm_interfaces.abi05_cpp_mangled_version_node` builds a DSO forcing
out-of-line emission of `rocrand_cpp::error` (fixture `abi05_fixture_rocrand_cpp.cpp`, which
must `#include <cstdio>` before the rocRAND header - see the header defect in
[audit-findings.md](audit-findings.md)) and asserts that the methods and the RTTI all carry `@@ROCBLAS_ABI_6`. The version script
`abi05_rocrand_cpp.map` assigns the node with namespace-wide globs (`_ZN11rocrand_cpp*`,
`_ZNK11rocrand_cpp*`, `_ZTVN..`, `_ZTIN..`, `_ZTSN..`); the checker
`check_cpp_mangled_version.cmake` narrows the positive count to the `rocrand_cpp::error` class
component (`_ZN11rocrand_cpp5error`, `_ZNK11rocrand_cpp5error`, `_ZTVN..5error`,
`_ZTIN..5error`), so only that class's members and RTTI are counted. Registration is gated on a
`ROCM_INTERFACES_HAVE_ROCRAND_CPP` host-compile probe so a toolchain without the header does
not break configure; an lld mirror runs too.

Non-vacuity here was proven with four mutations: node dropped -> fail; RTTI globs removed ->
fail (RTTI is a hard assertion, not best-effort); no ODR-use -> fail; anonymous map -> the
same eight symbols export with zero nodes. A final adversarial review caught one live defect
- the checker's method match was over-broad, matching the bare namespace prefix
`_ZN11rocrand_cpp` (which also catches namespace-scope free functions) rather than the class
component - fixed in commit `f681132` (which edits `check_cpp_mangled_version.cmake`, not the
map) to require `_ZN11rocrand_cpp5error`. The anonymous-control loop was left
broad on purpose: its job is to count everything and assert no node leaked, so a wider net
is stricter.

## The one-line map

| Step | Threat | ctest(s) |
| --- | --- | --- |
| 1 RES-02 | 170 leaked libstdc++ symbols | `exports` |
| 2 named nodes | bare symbols interpose across majors | `exports` |
| 3 co-residency | is the node mechanism real | `abi03_coresidency`, `abi03_interpose_hazard` |
| 4 RES-03 guard | g++ LTO + lld cannot stamp version nodes | `lto_linker_guard_rejects_gnu_lld` + 3 accepts |
| 5 OPS-04 | loader/registry data race | `ops04_concurrency` |
| 6a core | node fails on ordering / same-node control / `-Bsymbolic` genuineness / dup-def / ldconfig | `abi04_three_line_order`, `abi04_same_node_negative`, `abi04_bsymbolic_inert`, `abi04_multiple_default_def_rejected`, `abi04_ldconfig_stub_preserved` |
| 6b lld | node fails under the other linker | the five `abi04_*_lld` |
| 6c data | node fails on a data object | `abi06_data_version_node` |
| 6d ASan | node lost under ASan | `abi04_asan_version_node_survives` (node-definition + `__asan_` only; see 6d) |
| 6e C++/RTTI | node misses mangled + RTTI symbols | `abi05_cpp_mangled_version_node` (+ `_lld`) |

To add a new proof of your own, follow the recipe in [05-extending.md](05-extending.md).
