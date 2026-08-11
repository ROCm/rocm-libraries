# Hardening: what each proof exists to stop

Status: proposed design, prototype-backed. Every capability below is real code, and every
claim names the `ctest` that proves it. Run one with
`ctest --test-dir <build> -R <name> --output-on-failure`.

The [architecture](01-architecture.md) describes a versioned boundary. A boundary is only
real if the symbols actually behave - if the version script exports what it says, the nodes
attach where they should, and the loader survives being hammered by threads. This chapter
walks each hardening step in the order it was built. For each: the threat first, then the
fix, then the test that would fail if the fix regressed.

The whole suite runs green. In a canonical amdclang++/ld.lld build it registers 26 tests;
the definitions total 28 (some are gated on optional linkers and sanitizers). Names, not
counts, are what you should cite - they do not drift.

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

**Proof.** `rocm_interfaces.exports` builds every provider DSO and asserts each exports
exactly one dynamic symbol (`rocm_interfaces_provider_query_v1`) - one line from `nm -D`,
not 176. Landed in commit `a929517` (`fix(interfaces): stop recording providers leaking
libstdc++ symbols`).

## 2. Give the exports names, so majors can coexist (ABI-03, named nodes)

**Threat.** Hiding the leak is not enough for co-residency. If the loader exports a bare
`rocblas_sgemm`, then when two majors load in one process the dynamic loader binds every
caller to whichever it saw first. The allowlist controls *what* is exported; it does not
control *which definition wins*.

**Fix.** Named ELF version nodes. `loader/rocblas_loader.map` tags the 11 real loader entry
points with `ROCBLAS_ABI_5`, so they emit as `rocblas_sgemm@@ROCBLAS_ABI_5` and friends. The
`exports` check was upgraded from a symbol-count assertion to a version-node assertion at the
same time.

**Proof.** `rocm_interfaces.exports` (now node-aware). Landed in commit `ba093ad`
(`feat(interfaces): assign named version nodes to loader and provider exports`).

## 3. Prove the nodes actually defeat interposition (ABI-03, co-residency)

**Threat.** A named node is only worth having if it does what it claims. Assert nothing and
you are trusting the linker on faith.

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

## 4. Refuse a toolchain that silently drops versioning (RES-03)

**Threat.** The version nodes are stamped by the linker. GCC link-time optimization with the
LLVM linker (lld) drops the version-script assignments entirely - lld carries no GCC LTO
plugin and cannot resolve symbol-to-node assignments out of GCC LTO IR. The build succeeds.
The symbols come out unversioned. You discover this the day two majors interpose in the
field.

**Fix.** Fail the build instead. `rocm_interfaces_assert_lto_linker_supported()` in
`cmake/rocm_interfaces_lto_linker_guard.cmake` detects LTO (IPO or `-flto` in flags) plus
lld (`CMAKE_LINKER_TYPE=LLD`, `-fuse-ld=lld`, `--ld-path=*lld`, or a matching
`CMAKE_LINKER`) plus a non-Clang compiler, and raises `FATAL_ERROR` naming RES-03. It is
included and called in `interfaces/CMakeLists.txt`, so it is real enforcement, not a lint.

**Proof.** A `cmake -P` driver exercises four cases:

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
so a race fails the test. Non-vacuity: the binary carries 171 `__tsan_` symbols, confirming
it is genuinely instrumented rather than a silent non-sanitized fallback. Landed in commit
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
  in both orders. Negative control: give all three the same `.6` node and the check fails
  (the `ABI_5`/`ABI_7` nodes are absent). Discriminating.
- `rocm_interfaces.abi04_bsymbolic_inert` - a `-Bsymbolic` DSO carries `DT_FLAGS SYMBOLIC`
  (a plain DSO does not), yet co-residency resolution is identical. This proves `-Bsymbolic`
  is inert for co-residency and the *version node* is the mechanism.
- `rocm_interfaces.abi04_multiple_default_def_rejected` - a duplicate version-script
  collapses to a single `@@`; a synthetic two-`@@` input trips a `FATAL`. Proves you cannot
  ship two default definitions.
- `rocm_interfaces.abi04_ldconfig_stub_preserved` - `ldconfig -n` leaves the
  `librocblas.so -> librocblas.so.6` stub intact and the `ROCBLAS_ABI_6` node survives.

### 6b. The same invariants under the second linker (commit `01c14eb`)

`add_abi04_provider` gained an optional linker argument and a configure-time
`ROCM_INTERFACES_HAVE_LLD` probe. When lld is present, four lld-built mirrors run:
`abi04_three_line_order_lld`, `abi04_bsymbolic_inert_lld`,
`abi04_multiple_default_def_rejected_lld`, `abi04_ldconfig_stub_preserved_lld`.
Non-vacuity: the lld DSO's `.comment` stamps `Linker: AMD LLD`, which a bfd DSO lacks - proof
it is genuinely lld and not a silent bfd fallback.

### 6c. A real data object (commit `04ade2b`)

**Threat.** A version node might attach to functions but not to a data symbol.
**Proof.** `rocm_interfaces.abi06_data_version_node` versions a DSO built from the real
`sobol32` precomputed table (`abi06_data.map`, `ROCBLAS_ABI_6`) and asserts at runtime that
`dlsym`/`dlvsym` reach the exact node and that the first element is `0x80000000`; the
wrong-node lookup returns null.

### 6d. A build under AddressSanitizer (commit `0081ba5`)

**Threat.** ASan rewrites the binary; the node could be lost in the process.
**Proof.** `rocm_interfaces.abi04_asan_version_node_survives` builds an `-fsanitize=address`
DSO (via `ROCM_INTERFACES_SANITIZE=address`) and asserts `ROCBLAS_ABI_6` survives *and* the
DSO carries `__asan_` symbols - so a no-ASan fallback cannot pass it.

### 6e. C++ mangled names and RTTI (commit `769f4aa`, tightened in `f681132`)

**Threat.** This is the hardest shape. C++ methods, and especially the RTTI vtable/typeinfo,
are emitted as default-visibility weak symbols with mangled names; a naive map misses them.
**Fix and proof.** `rocm_interfaces.abi05_cpp_mangled_version_node` builds a DSO forcing
out-of-line emission of `rocrand_cpp::error` (fixture `abi05_fixture_rocrand_cpp.cpp`, which
must `#include <cstdio>` before the rocRAND header - see the header defect in
[audit-findings.md](audit-findings.md)) and asserts, via mangled-name globs in
`abi05_rocrand_cpp.map`, that the methods (`_ZN11rocrand_cpp5error*`, `_ZNK..`) and the RTTI
(`_ZTVN..`, `_ZTIN..`) all carry `@@ROCBLAS_ABI_6`. Registration is gated on a
`ROCM_INTERFACES_HAVE_ROCRAND_CPP` host-compile probe so a toolchain without the header does
not break configure; an lld mirror runs too.

Non-vacuity here was proven with four mutations: node dropped -> fail; RTTI globs removed ->
fail (RTTI is a hard assertion, not best-effort); no ODR-use -> fail; anonymous map -> the
same eight symbols export with zero nodes. A final adversarial review caught one live defect
- the method match was over-broad, matching the bare namespace prefix `_ZN11rocrand_cpp`
(which also catches namespace-scope free functions) rather than the class component - fixed
in commit `f681132` to require `_ZN11rocrand_cpp5error`. The anonymous-control loop was left
broad on purpose: its job is to count everything and assert no node leaked, so a wider net
is stricter.

## The one-line map

| Step | Threat | ctest(s) |
| --- | --- | --- |
| 1 RES-02 | 170 leaked libstdc++ symbols | `exports` |
| 2 named nodes | bare symbols interpose across majors | `exports` |
| 3 co-residency | is the node mechanism real | `abi03_coresidency`, `abi03_interpose_hazard` |
| 4 RES-03 guard | toolchain drops versioning silently | `lto_linker_guard_rejects_gnu_lld` + 3 accepts |
| 5 OPS-04 | loader/registry data race | `ops04_concurrency` |
| 6a core | node fails on ordering / dup-def / ldconfig | `abi04_three_line_order`, `abi04_bsymbolic_inert`, `abi04_multiple_default_def_rejected`, `abi04_ldconfig_stub_preserved` |
| 6b lld | node fails under the other linker | the four `abi04_*_lld` |
| 6c data | node fails on a data object | `abi06_data_version_node` |
| 6d ASan | node lost under ASan | `abi04_asan_version_node_survives` |
| 6e C++/RTTI | node misses mangled + RTTI symbols | `abi05_cpp_mangled_version_node` (+ `_lld`) |

To add a new proof of your own, follow the recipe in [05-extending.md](05-extending.md).
