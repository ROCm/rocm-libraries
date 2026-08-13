# Why a stable, versioned boundary

Status: proposed design, prototype-backed. The failures below are real and reproducible;
each fix is in this tree and names the CTest that locks it, except where the text marks a
claim as an intended contract or still-future proof.

Most of the time, a ROCm math library and its callers get along fine. The trouble starts
on the day you want to change something. This chapter is about that day - the ways the
boundary fails without deliberate versioning, and what each one costs. The threat-model
table at the end enumerates the six distinct threats; the sections below group them into
failure stories.

## Failure 1: a bare exported symbol locks you to one provider

Ship rocBLAS. A caller links `librocblas.so` and calls `rocblas_sgemm`. The exported C
symbol freezes the call ABI, not the internals, so changing how `sgemm` runs is already
allowed - rocBLAS routes some paths through hipBLASLt and falls back to its legacy solutions
(see [audit-findings.md](audit-findings.md)). A stable C function is, by itself, an
implementation seam for behavior.

What a bare symbol does not give you is provider replacement. The caller is bound to rocBLAS
as the sole provider of `rocblas_sgemm`, so you cannot drop in a different implementation of
the domain without relinking the caller. A provider that build-depends on another - rocBLAS
on hipBLASLt - cannot be swapped for it independently, and no package boundary says who owns
the symbol. You get one implementation, welded to one package, that only its own author can
replace.

Concretely, a minimal reproduction shows what that binding records. A `caller` that calls
`rocblas_sgemm`, linked against a `librocblas.so.0` that itself build-depends on
`libhipblaslt.so.0`, records only a bare name plus rocBLAS's SONAME:

```
$ nm -D caller
                 U rocblas_sgemm
$ readelf -d caller | grep NEEDED
 (NEEDED)  Shared library: [librocblas.so.0]
```

The import is undefined and unversioned - just a name - and the `NEEDED` entry binds it to
rocBLAS specifically. So provider replacement is not "put a different `.so` on the path": the
loader resolves by SONAME, and a caller built against rocBLAS ignores an independent
`libmyblas.so.0` that exports the very same `rocblas_sgemm`:

```
$ LD_LIBRARY_PATH=./onlyB ./caller
./caller: error while loading shared libraries: librocblas.so.0: cannot open shared object file: No such file or directory
```

The only swap that avoids relinking is impersonation - renaming the other provider to
`librocblas.so.0` - which lies about identity and silently drops rocBLAS's own dependency
edge on hipBLASLt:

```
$ readelf -d librocblas.so.0 | grep NEEDED
 (NEEDED)  Shared library: [libhipblaslt.so.0]
```

A clean swap instead relinks the caller against the new provider, after which its `NEEDED`
reads `libmyblas.so.0`. And because the exported symbol carries no version node at all:

```
$ readelf --dyn-syms librocblas.so.0 | grep rocblas_sgemm
     6: 0000000000001119    18 FUNC    GLOBAL DEFAULT   12 rocblas_sgemm
```

that bare import, once a caller records it, can never be changed or retired, and no new
contract can ship beside it. Failure 3 and [chapter 03](03-abi-and-versioning-contract.md)
are about closing that last gap.

The fix is a seam: the caller talks to a stable loader, the implementation lives behind a
provider protocol, and the two never share symbols by accident. Selection happens once, at
context creation, and is frozen for the life of the context. You can swap the provider
behind the loader and the caller never notices.

## Failure 2: the provider `.so` leaks ~174 symbols you never wrote

This one is quiet, and it is the one that bites first. Build a provider `.so` from C++ and
the linker exports far more than you asked for. Pull in `std::filesystem` and roughly 174
libstdc++ out-of-line symbols land in your dynamic table with default visibility.

Setting `-fvisibility=hidden` does not save you: those symbols are default-visibility
inside libstdc++, pulled from the archive by the runtime headers, so a whole-TU visibility
flag never touches them.

Now put two provider libraries in the same process. Both export the same `std::` symbols.
The dynamic loader picks one definition for the whole process, and the other library
silently calls code it was never compiled against. Nothing crashes at link time. It goes
wrong at runtime, intermittently, depending on load order.

The fix is an explicit export allowlist - a version script that names the one symbol a
provider may expose and hides everything else. Proven by `rocm_interfaces.exports`, which
derives every registered provider DSO and
asserts each exports exactly one symbol, not 176. The independent
`rocm_interfaces.exports_provider_list_complete` control requires that derived list to match
the complete build-system recording-provider enumeration.

## Failure 3: two library majors in one process interpose each other

Sooner or later two ABI majors of the same library coexist in a process - a plugin built
against the old rocBLAS loaded next to an app built against the new one. If both export a
bare `rocblas_sgemm`, the dynamic loader binds every caller to whichever it saw first. The
new library can end up calling the old library's `sgemm`. This is symbol interposition, and
it is silent.

The fix is named ELF version nodes. `rocblas_sgemm@@ROCBLAS_ABI_5` and
`rocblas_sgemm@@ROCBLAS_ABI_6` are distinct symbols to the loader even though the C name is
identical, so each caller binds to the major it was built against. Two tests lock this:
`rocm_interfaces.abi03_coresidency` (each handle resolves its own node, cross-version
lookup nil) and `rocm_interfaces.abi03_interpose_hazard` (remove the node and a bare global
lookup reproduces the hazard). The causal linked-consumer proof is still future work (see
[07-status-and-roadmap.md](07-status-and-roadmap.md#aspirational-direction-not-commitment)).

## Failure 4: the versioning silently stops working

The whole scheme rests on the linker actually stamping those version nodes. Several ordinary
build choices can break that stamping without any error:

- **A toolchain mismatch.** GCC link-time optimization with the LLVM linker (lld) cannot
  resolve the version-script assignments out of GCC LTO IR - lld carries no GCC LTO plugin.
  The failure mode is linker-version-dependent: in the recorded RES-03 spike (an older lld)
  the pairing failed hard - ld.lld errored "version script assignment of ROCBLAS_ABI_5 to
  symbol rocblas_sgemm failed: symbol not defined" and produced no DSO - whereas on the
  current shipping toolchain (GCC 13.3.1, AMD LLD 23.0.0) it exits 0 and produces a DSO but
  silently drops the versioned symbol (the node is stamped yet empty, the export vanishes).
  A loud error is confusing deep in a build; the silent-drop mode is worse still - so the
  combination is refused at configure time by
  `rocm_interfaces_assert_lto_linker_supported()`; proven by
  `rocm_interfaces.lto_linker_guard_rejects_gnu_lld` and the three `..._accepts_...` cases
  (see
  [04-hardening.md](04-hardening.md#4-refuse-a-toolchain-that-cannot-stamp-version-nodes-res-03)).
- **A different symbol shape.** A data object, a C++ mangled method, an RTTI vtable, or a
  build under AddressSanitizer are each a chance for the node to fail to attach. Proven to
  survive by `rocm_interfaces.abi06_data_version_node`,
  `rocm_interfaces.abi05_cpp_mangled_version_node`, and
  `rocm_interfaces.abi04_asan_version_node_survives`.
## Failure 5: a data race in the loader corrupts every dispatch

The failures above are about what the symbols say. This one is about the loader itself. The
registry that hands out providers is shared across threads; a data race there - two contexts
selecting at once, a torn module refcount - corrupts every downstream dispatch, and races do
not fail deterministically, so a unit test can pass a thousand times and still be broken.
Regression-locked under ThreadSanitizer by `rocm_interfaces.ops04_concurrency`.

## The threat model, on one page

| Threat | What breaks | The mechanism that stops it | Proven by |
| --- | --- | --- | --- |
| Locked to one provider | Cannot replace the whole implementation or swap providers without relinking | Loader/provider seam; selection frozen at context creation | `abi03_coresidency`, architecture design |
| C++ symbol leakage | Two libraries share `std::` symbols; wrong code runs | Version-script allowlist: export one symbol, hide the rest | `exports` |
| Interposition across majors | New library calls old library's implementation | Named ELF version nodes per major | `abi03_linked_consumer_versioned_binds`, `abi03_interpose_hazard`, `abi03_coresidency` |
| Toolchain cannot stamp versioning | GCC-LTO + lld cannot read the version-script symbols out of GCC LTO IR: the shipping lld silently drops the versioned export (an older lld errored hard and produced no DSO) | Configure-time GCC-LTO-plus-lld guard that fails the build with a named RES-03 error | `lto_linker_guard_rejects_gnu_lld` |
| Node fails on odd symbol shapes | Data / mangled / RTTI / ASan symbols unversioned | Proof suite exercises each shape | `abi06_data_version_node`, `abi05_cpp_mangled_version_node`, `abi04_asan_version_node_survives` |
| Loader data race | Dispatch corruption under concurrent use | TSan regression lock on registry + loader | `ops04_concurrency` |

Most `Proven by` entries name a `ctest` you can run
(`ctest --test-dir <build> -R <name>`); where a row cites "architecture design" instead,
the guarantee is structural, not a single executable test. The mechanisms are described in [01-architecture.md](01-architecture.md), the
contract that governs them in [03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md),
and the story behind each proof in [04-hardening.md](04-hardening.md).
