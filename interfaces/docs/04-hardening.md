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
`-fvisibility=hidden` flag never touches them. Result: roughly 174 leaked `std::` symbols in
the provider's `.dynsym` (176 defined dynamic symbols in all, of which one is the real export). Two such providers in one process share those symbols through the
dynamic loader, and one silently runs the other's code.

**Fix.** An explicit export allowlist. `providers/provider.map` names
the single symbol a provider may expose and sends everything else to `local: *`. The
`--version-script` is applied to every recording and system-backed provider target.

**Proof.** `rocm_interfaces.exports` derives its provider list from the global
`ROCM_INTERFACES_PROVIDER_TARGETS` build-system property populated by every provider target,
then asserts that each DSO exports exactly
one defined, non-absolute dynamic symbol (`rocm_interfaces_provider_query_v1`). The checker
runs `nm -D --defined-only --with-symbol-versions` and ignores the absolute version-node
entry, so it is one callable export, not 176. The independent
`rocm_interfaces.exports_provider_list_complete` control recursively enumerates every
provider `MODULE_LIBRARY` from the build system and requires that ground truth to equal the
derived registry list, so a newly registered provider cannot silently escape inspection.
The export allowlist landed in `a929517`; auto-derivation and its completeness control landed in
`61f8dc9` (`test(interfaces): make provider export coverage fail closed`).

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

The two tests above use `dlvsym`/`dlsym` on explicit handles, which resolve each handle's
own symbol whether or not a node is present - so they prove co-residency and the nodeless
hazard, but not that a node defeats interposition for a *linked* caller. A third pair closes
that gap with a real linked relocation:

- `rocm_interfaces.abi03_linked_consumer_versioned_binds` links a consumer against both
  providers with `libprovA.so.6` (`ABI_6`) `NEEDED` first, and pins its `rocblas_sgemm`
  reference to `ROCBLAS_ABI_7` with a `.symver` directive. The relocation binds to `ABI_7`
  (`-> 7`) despite the `ABI_6` provider being earlier in scope - `readelf -V` confirms the
  binary carries a `Verneed` requirement for `ROCBLAS_ABI_7` on `libprovB.so.7`.
- `rocm_interfaces.abi03_linked_consumer_plain_interposed` is the discriminating control:
  the identical link line with the `.symver` removed leaves an unversioned reference, which
  is interposed by the first-`NEEDED` `ABI_6` provider (`-> 6`, `Verneed` on
  `libprovA.so.6`). One directive flips the bound major, so the pair is not vacuous.

The defense has a boundary, and it too is locked by a test rather than left to prose:

- `rocm_interfaces.abi03_versioned_bare_lookup_uncovered` loads the same noded providers
  (`ABI_6`, `ABI_7`) both `GLOBAL` with `ABI_6` first, then shows a bare unversioned
  `dlsym(RTLD_DEFAULT, rocblas_sgemm)` still takes the first-loaded `ABI_6` (`-> 6`) while a
  version-aware `dlvsym(..., ROCBLAS_ABI_7)` reaches `ABI_7` (`-> 7`). The nodes are present
  and functional; a bare global lookup simply does not consult them. This discriminates from
  `abi03_interpose_hazard` (which uses nodeless DSOs) on lookup form alone, and is the reason
  the boundary routes callers through versioned relocations or `dlvsym`.

Landed in commits `8832c9a` (`test(interfaces): prove ABI version-node co-residency defeats
interposition`), `3396f66` (linked-consumer relocation proof), and `8235b3c` (bare
`RTLD_DEFAULT` boundary).

## 4. Refuse a toolchain that cannot stamp version nodes (RES-03)

**Threat.** The version nodes are stamped by the linker. GCC link-time optimization with the
LLVM linker (lld) cannot resolve the version-script symbol-to-node assignments out of GCC LTO
IR - lld carries no GCC LTO plugin. This pairing breaks the versioning, and the exact failure
mode is linker-version-dependent. The recorded RES-03 spike (an older lld, `res03/build-lld-fat/err5`
in the exec spike) failed hard at link time: ld.lld emitted 'version script assignment of
ROCBLAS_ABI_5 to symbol rocblas_sgemm failed: symbol not defined' for every versioned symbol and
produced no DSO. On the current shipping toolchain (GCC 13.3.1, AMD LLD 23.0.0) the same pairing
no longer errors: it exits 0 and produces a DSO, but ld.lld silently drops the versioned symbol
it cannot read out of the LTO IR - the `ROCBLAS_ABI_5` node is stamped yet carries no symbol, so
the export simply vanishes (`nm -D` shows no `rocblas_sgemm`). Either way the versioning is
broken; the silent-drop mode is strictly worse, and a lint keyed on the old error string would
miss it entirely - which is why the pairing is refused at configure time rather than diagnosed
after the fact.

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
control, genuineness.

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
- `rocm_interfaces.abi04_multiple_default_def_rejected` (strengthened in commit `b7f3f89`) -
  synthesizes a genuine two-default-definition DSO and proves it is rejected. Two objects each
  carry a default `.symver` alias of the same base name (`rocblas_sgemm@@ROCBLAS_ABI_6` and
  `rocblas_sgemm@@ROCBLAS_ABI_7`); the test asserts the link fails **and** that the diagnostic
  is specifically a duplicate/multiple-definition of `rocblas_sgemm`, so an unrelated link
  failure no longer passes. It is made discriminating by a single-`@` (non-default) control
  built from the same objects: changing one `@@` to `@` must flip the result to a clean link
  with exactly one `@@` and one `@`. Earlier the check compiled a single `rocblas_sgemm` and
  named it in two nodes, which can only ever yield 0 or 1 `@@` (the linker warns and assigns
  to the first node), so it observed the toolchain rather than forcing a two-`@@` rejection.
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

### 6d. A build under AddressSanitizer (commit `0081ba5`, tightened in `c5031f0`)

**Threat.** ASan rewrites the binary; the node could be lost in the process.
**Proof.** `rocm_interfaces.abi04_asan_version_node_survives` builds an `-fsanitize=address` DSO
(via `ROCM_INTERFACES_SANITIZE=address`), asserts the DSO carries `__asan_` symbols
(genuineness, so a no-ASan fallback cannot pass it), and asserts `rocblas_sgemm@@ROCBLAS_ABI_6`
appears in `nm -D --with-symbol-versions` output - the per-symbol `@@` default binding, not the
looser substring on the absolute node symbol. Its negative control is a second genuine ASan
build (also carrying `__asan_` and exporting `rocblas_sgemm`) linked with a nodeless version
script; the test fails if that control shows any `ROCBLAS_ABI_6` node, so the positive
assertion is discriminating in the same way as 6a/6c/6e.

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
| 1 RES-02 | ~174 leaked libstdc++ symbols | `exports` |
| 2 named nodes | bare symbols interpose across majors | `exports` |
| 3 co-residency | is the node mechanism real; does it defeat interposition for a linked caller; where is its boundary | `abi03_coresidency`, `abi03_interpose_hazard`, `abi03_linked_consumer_versioned_binds`, `abi03_linked_consumer_plain_interposed`, `abi03_versioned_bare_lookup_uncovered` |
| 4 RES-03 guard | g++ LTO + lld cannot stamp version nodes | `lto_linker_guard_rejects_gnu_lld` + 3 accepts |
| 5 OPS-04 | loader/registry data race | `ops04_concurrency` |
| 6a core | node fails on ordering / same-node control / `-Bsymbolic` genuineness / dup-def / ldconfig | `abi04_three_line_order`, `abi04_same_node_negative`, `abi04_bsymbolic_inert`, `abi04_multiple_default_def_rejected`, `abi04_ldconfig_stub_preserved` |
| 6b lld | node fails under the other linker | the five `abi04_*_lld` |
| 6c data | node fails on a data object | `abi06_data_version_node` |
| 6d ASan | node lost under ASan | `abi04_asan_version_node_survives` (`rocblas_sgemm@@ROCBLAS_ABI_6` + `__asan_` + nodeless negative control) |
| 6e C++/RTTI | node misses mangled + RTTI symbols | `abi05_cpp_mangled_version_node` (+ `_lld`) |

To add a new proof of your own, follow the recipe in [05-extending.md](05-extending.md).
