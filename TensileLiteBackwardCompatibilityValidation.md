# TensileLite generated-library compatibility with installed hipBLASLt

Investigation date: 2026-09-24 UTC

## Conclusion

No: without compatibility changes, the current from-source TensileLite in this checkout cannot generate a library that the hipBLASLt installed under `/opt/rocm` can load and use.

The installed consumer is hipBLASLt 1.2.2 from ROCm 7.2.4, not hipBLASLt 1.2.1. The available GPUs are gfx942, not gfx950. This experiment therefore directly answers the question for the software and hardware on this machine. It also empirically confirms the same packaging and serialized-schema incompatibilities described in `TensileLiteBackwardCompatibility.md`, even with the newer 1.2.2 consumer.

The differential test established all of the following:

1. The test harness, installed hipBLASLt, installed device library, and GPU are healthy. The baseline loaded the installed gfx942 solution code object and produced zero incorrect values.
2. Current TensileLite successfully consumed one committed production logic file and generated a complete gfx942 library.
3. The generated library could not be used as-is. The installed loader requested `TensileLibrary_lazy_gfx942.dat`, while current TensileLite emitted `TensileLibrary_lazy_gfx942.dat.zlib`.
4. Passing TensileLite's native `--no-compress` option did not produce uncompressed metadata. The `.dat.zlib` files were byte-for-byte identical to the default output; only the GPU solution code object changed from its compressed bundle to an uncompressed bundle.
5. A separate packaging-only diagnostic copy was decompressed and given the old mapping name. The installed metadata reader then failed with `invalid array<T, 5> index 5`, directly confirming that current `TypesEqual` has six entries while this consumer accepts five.
6. No generated solution kernel was launched. The consumer rejected the catalog before heuristic selection, so numerical validation of the generated kernel was unreachable without changing the metadata contract.

The generated metadata also records `KernArgsVersion=3`; the installed consumer source defaults to version 2. Deliberately rewriting the catalog to mislabel a version-3 kernel as version 2 and launching it was not attempted because that can pass pointers at the wrong kernarg offsets and cause a GPU memory fault. The earlier load failures are already sufficient to disprove unchanged compatibility.

## Reproducer

Run from the repository root:

```bash
./scripts/run_tensilelite_backward_compat_e2e.sh
```

The process must be allowed to open `/dev/kfd`. A restricted container or sandbox can enumerate gfx942 through sysfs while still failing the numerical baseline with `no ROCm-capable device is detected`.

Each invocation creates a new evidence directory under:

```text
build/tensilelite-backward-compat/run-<UTC timestamp>-<PID>/
```

Important outputs are:

| File or directory | Purpose |
| --- | --- |
| `run.log` | Full command and output transcript |
| `baseline.log` | Installed library load, kernel launch, and numerical result |
| `generated/` | Unmodified current-TensileLite output |
| `generated-raw.log` | Installed consumer's attempt to use unmodified output |
| `generated-no-compress/` | A second native generation using `--no-compress` |
| `generated-no-compress.log` | Installed consumer's attempt to use the native `--no-compress` output |
| `packaging-adapted-library/` | Diagnostic copy with only decompression and old mapping-name adaptation |
| `generated-packaging-adapted.log` | Installed consumer's metadata-reader result after packaging adaptation |
| `verdict.txt` | Machine-readable final verdict |

The script exits nonzero when setup, compilation, the installed-library baseline, or library generation fails. A completed compatibility experiment exits zero and writes one of these verdicts:

- `COMPATIBLE_AS_IS`
- `COMPATIBLE_WITH_NO_COMPRESS`
- `PACKAGING_INCOMPATIBLE_METADATA_AND_KERNEL_COMPATIBLE`
- `INCOMPATIBLE`

The observed verdict was `INCOMPATIBLE`.

## Pinned experiment

| Boundary | Value |
| --- | --- |
| Current producer checkout | `47ee6f53dd302ff533892d6f48c88b3813b87fef` |
| Logic file commit | `2b8462cabb63249c61458baad734a22767586ea0` |
| Installed hipBLASLt version | `1.2.2` |
| Installed package version | `1.2.2.70204-93~24.04` |
| Installed source tweak | `dabb6df2b9` (`dabb6df2b988f8eabed1e2fecefaaf4e818bc7ef`) |
| Loaded host library | `/opt/rocm-7.2.4/lib/libhipblaslt.so.1.2.70204` |
| ROCm version | `7.2.4` |
| GPU architecture | `gfx942:sramecc+:xnack-` |
| Visible devices | 8 |
| Canonical evidence directory | `build/tensilelite-backward-compat/run-20260924T171805Z-2799772/` |

The script verifies that the logic file is tracked and has no staged or unstaged modifications before generating anything.

## Selected committed logic

The experiment uses:

```text
projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/
  asm_full/aquavanjaram/gfx942/Equality/
  aquavanjaram_Cijk_Alik_Bljk_SB_UserArgs.yaml
```

This is a production gfx942 equality logic file for float32 GEMM with `transA=T`, `transB=N`. It contains the exact problem used by the harness:

```text
m=512, n=512, batch=1, k=512
data type: float32
compute type: float32
alpha=1, beta=1
```

TensileCreateLibrary parsed 17 solutions from the logic and generated 16 unique kernels.

## What the end-to-end script does

### 1. Establish provenance

The script records:

- the producer checkout revision;
- the commit that last changed the selected logic file;
- the installed hipBLASLt semantic version and source tweak;
- the real path of `libhipblaslt.so.1`;
- the GPU architectures returned by `rocm_agent_enumerator`;
- the exact Python `Tensile` and from-source `rocisa` modules used.

It imports `VCndMaskB16` as a small current-rocisa capability check. This catches the stale rocisa extension encountered during the earlier source-contract analysis. On this checkout the working module was:

```text
projects/hipblaslt/tensilelite/build_tmp/tensilelite/rocisa/rocisa/
```

If that current from-source build is unavailable, rebuild it with:

```bash
cd projects/hipblaslt/tensilelite
invoke rocisa
```

The script also accepts `TENSILE_PYTHON` and `ROCISA_PYTHONPATH` overrides.

### 2. Build a consumer-independent numerical harness

The script compiles `scripts/tensilelite_backward_compat_e2e.cpp` against only the installed headers and libraries:

```bash
/opt/rocm/bin/hipcc \
  -std=c++17 -O2 \
  scripts/tensilelite_backward_compat_e2e.cpp \
  -I/opt/rocm/include \
  -L/opt/rocm/lib \
  -Wl,-rpath,/opt/rocm/lib \
  -lhipblaslt -lamdhip64 -ldl
```

It checks the dynamic resolution with `ldd` and at runtime with `dlsym`/`dladdr`. This avoids using a client that might itself be linked to a from-source TensileLite host library and contaminating the consumer boundary.

The harness:

- creates deterministic float32 A, B, and C matrices;
- computes a CPU reference for `D = A^T * B + C`;
- asks hipBLASLt for one heuristic solution;
- executes it;
- copies D back to the host;
- validates every one of the 262,144 output values;
- prints the loaded hipBLASLt path, GPU architecture, hipBLASLt version integer, workspace, incorrect-value count, and maximum errors.

### 3. Prove the baseline

The baseline points the installed consumer at its paired installed device library:

```bash
HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/lib/hipblaslt/library \
HIPBLASLT_LOG_LEVEL=4 \
TENSILE_DB=0x2B060 \
<harness>
```

Observed evidence:

```text
hipblaslt_library=/opt/rocm-7.2.4/lib/libhipblaslt.so.1.2.70204
gpu_arch=gfx942:sramecc+:xnack- device_count=8
hipblaslt_version_integer=100202
Loading library mapping from file: .../TensileLiteLibrary_lazy_Mapping.dat
17 solutions loaded
loaded code object .../TensileLibrary_SS_SS_UA_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
incorrect_values=0 total_values=262144 max_absolute_error=0 max_relative_error=0
PASS numerical_validation
```

This is the control that makes the later failures meaningful. The same installed host library, GPU, API calls, dimensions, inputs, and correctness test work with the installed device library.

### 4. Generate a library from the one logic file

The central generation command is equivalent to:

```bash
PYTHONPATH=<current rocisa>:projects/hipblaslt/tensilelite \
python -m Tensile.TensileCreateLibrary \
  <directory containing the logic file> \
  <run directory>/generated \
  HIP \
  --architecture gfx942 \
  --logic-filter aquavanjaram_Cijk_Alik_Bljk_SB_UserArgs \
  --library-format msgpack \
  --no-enumerate \
  --jobs 8 \
  --verbose 2 \
  --cxx-compiler /opt/rocm/bin/amdclang++ \
  --c-compiler /opt/rocm/bin/amdclang \
  --assembler /opt/rocm/bin/amdclang++ \
  --offload-bundler /opt/rocm/llvm/bin/clang-offload-bundler
```

Generation completed successfully. The resulting files were:

| Artifact | Size |
| --- | ---: |
| `Kernels.so-000-gfx942.hsaco` | 176,168 bytes |
| `TensileLibrary_SS_SS_UA_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co` | 97,174 bytes |
| `TensileLibrary_SS_SS_UA_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.dat.zlib` | 5,087 bytes |
| `TensileLibrary_lazy_gfx942.dat.zlib` | 520 bytes |
| `TensileLiteLibrary_lazy_gfx942_Mapping.dat.zlib` | 75 bytes |

### 5. Audit what the current producer emitted

Before invoking hipBLASLt, the script decompresses the metadata in memory and records its contract:

```text
metadata_compute_fields=computeInputTypeA,computeInputTypeB
metadata_types_equal_length=6
metadata_task_predicates=And,LaunchLimits,WorkspaceCheck
metadata_kernargs_version=3
```

The emitted ordinary-GEMM problem predicates also include the newer types identified by the earlier analysis:

```text
FusedGemmA2A
GateResidualDataTypeWhiteList
MXBlockA
MXBlockB
UseGateResidual
```

The installed consumer source at tweak `dabb6df2b9` has these older contracts:

- required problem-type field `computeInputType`;
- `TypesEqual` stored as `std::array<rocisa::DataType, 5>`;
- default `KernArgsVersion=2`;
- direct reads of uncompressed `.dat` files;
- lazy mapping name `TensileLiteLibrary_lazy_Mapping.dat`.

### 6. Attempt the generated library without modifications

The script points the installed consumer directly at the current producer's per-architecture output:

```bash
HIPBLASLT_TENSILE_LIBPATH=<run directory>/generated/library/gfx942 \
<harness>
```

The first causal failure was:

```text
Cannot read ".../TensileLibrary_lazy_gfx942.dat": No such file or directory
Error loading .../TensileLibrary_lazy_gfx942.dat (msgpack):
Failed to open file
hipblasLtMatmulAlgoGetHeuristic(...): status 3
```

The current producer wrote `.dat.zlib`; the installed loader requested `.dat`. This alone disproves as-is compatibility.

The log also shows `Kernels.so-000-gfx942.hsaco` loading. That is a helper code object loaded during initialization and is not evidence that a selected solution kernel ran. The solution catalog failed to load, heuristic selection returned status 3, and the generated solution `.co` was never launched.

### 7. Test TensileLite's native `--no-compress` option

Current TensileLite exposes this option:

```bash
python -m Tensile.TensileCreateLibrary ... --no-compress
```

Its help text is precise: `Don't compress assembly code objects.` The option is passed to the kernel/code-object writer, while `LibraryIO.writeMsgPack()` independently and unconditionally applies zlib level 9 and writes `.dat.zlib`.

The native `--no-compress` generation therefore still produced:

```text
TensileLibrary_SS_SS_UA_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.dat.zlib
TensileLibrary_lazy_gfx942.dat.zlib
TensileLiteLibrary_lazy_gfx942_Mapping.dat.zlib
```

All three metadata files had exactly the same SHA-256 hashes as the default generation. The effect of the flag was visible only in the solution code object:

| Artifact | Default | `--no-compress` |
| --- | ---: | ---: |
| Solution `.co` | 97,174 bytes | 1,223,808 bytes |
| Solution metadata | 5,087-byte `.dat.zlib` | Same 5,087-byte `.dat.zlib` |
| Master metadata | 520-byte `.dat.zlib` | Same 520-byte `.dat.zlib` |
| Lazy mapping | 75-byte `.dat.zlib` | Same 75-byte `.dat.zlib` |

The installed consumer consequently failed at the same boundary:

```text
Cannot read ".../generated-no-compress/library/gfx942/TensileLibrary_lazy_gfx942.dat": No such file or directory
hipblasLtMatmulAlgoGetHeuristic(...): status 3
```

So the answer to whether current TensileLite can be told to emit uncompressed MessagePack through `--no-compress` is **no**. That flag controls assembly code-object compression, not catalog compression. There is currently no `TensileCreateLibrary` CLI switch for uncompressed MessagePack.

`--library-format yaml` emits uncompressed YAML, but that is a different serialization format, the installed release library has no `yaml-cpp` dependency and is built for MessagePack, and the newer serialized schema would still remain. It is not an uncompressed-MessagePack compatibility path.

### 8. Packaging-only diagnostic probe

To distinguish packaging failure from schema compatibility, the script copies the generated directory and changes only packaging:

1. Decompress every `*.dat.zlib` to the corresponding `*.dat`.
2. Copy `TensileLiteLibrary_lazy_gfx942_Mapping.dat` to the old expected name `TensileLiteLibrary_lazy_Mapping.dat`.
3. Leave all serialized fields, predicate values, solutions, and GPU kernels unchanged.

The installed consumer then reached its MessagePack-to-C++ conversion and failed with:

```text
Error loading msgpack data:
invalid array<T, 5> index 5
hipblasLtMatmulAlgoGetHeuristic(...): status 3
```

This is the runtime manifestation of the `TypesEqual` schema break: the current producer emitted six data types, while the installed consumer's `std::array<..., 5>` rejects the sixth entry.

## Attempts and corrections made while building the reproducer

### Direct `amdclang++` harness compile

The first harness compile used `/opt/rocm/bin/amdclang++` directly. It failed with:

```text
Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__
```

This was a harness compiler-driver issue, not a hipBLASLt compatibility result. The script now uses `/opt/rocm/bin/hipcc`, which supplies the HIP platform configuration.

### Restricted GPU access

The first baseline executed in a restricted environment that could enumerate gfx942 but could not open `/dev/kfd`. It reported:

```text
gpu_unavailable=no ROCm-capable device is detected device_count=0
```

The complete experiment was rerun with GPU device access. It saw eight gfx942 devices and the installed-library baseline passed.

### Initial metadata retry lacked detailed errors

The installed reader initially printed only `Could not load`. Its detailed MessagePack errors are gated by Tensile's `0x1000` debug bit. The script now uses `TENSILE_DB=0x2B060`, which records mapping loads, code-object paths, selected kernels, kernargs, library details, and metadata-reader errors. This exposed `invalid array<T, 5> index 5`.

## Relationship to `TensileLiteBackwardCompatibility.md`

The earlier report studied current TensileLite versus hipBLASLt 1.2.1 on gfx950 from source contracts and did not claim a GPU end-to-end run. This experiment differs in two pinned dimensions:

| Report | Consumer | GPU | Evidence |
| --- | --- | --- | --- |
| `TensileLiteBackwardCompatibility.md` | hipBLASLt 1.2.1 | gfx950 | Source and serialization-contract analysis |
| This report | installed hipBLASLt 1.2.2 | gfx942 | Actual generation, dynamic loading, GPU baseline, and consumer rejection |

The empirical result supports, rather than weakens, the earlier conclusion:

- compression and mapping names are real load-time boundaries;
- the native `--no-compress` option affects code-object bundling but not MessagePack catalog compression;
- adapting names and compression does not repair the schema;
- the five-versus-six `TypesEqual` mismatch is observed directly in the installed reader;
- current ordinary GEMM output contains newer predicates;
- the generated catalog declares kernarg version 3 while the installed consumer generation defaults to version 2.

This run does not claim that a gfx950 kernel was executed or that the exact 1.2.1 binary was tested. It proves the requested current-machine case and demonstrates that even installed 1.2.2 cannot consume current output unchanged.

## What would be required to make it work

There are two safe routes:

1. Generate the device library with the TensileLite revision paired with the installed hipBLASLt source tweak/release.
2. Upgrade the consuming hipBLASLt together with its paired TensileLite-generated device library.

Keeping current TensileLite with this unchanged installed consumer would require a real compatibility target, not just a filename conversion:

- emit `computeInputType` and the old five-entry `TypesEqual` representation;
- suppress or translate predicates unknown to the old reader;
- generate the actual version-2 kernel argument layout and audit all later host/device argument additions;
- emit uncompressed `.dat` catalogs and the old lazy mapping name/layout;
- run numerical validation on every supported problem family and architecture.

Changing only metadata labels is unsafe. In particular, changing `internalArgsSupport.version` from 3 to 2 without regenerating the kernel signature would tell the old host to launch a version-3 kernel using version-2 pointer and stride offsets.

## Files added by this investigation

- `scripts/run_tensilelite_backward_compat_e2e.sh`: full generation, baseline, load, metadata audit, packaging probe, and verdict workflow.
- `scripts/tensilelite_backward_compat_e2e.cpp`: installed-library-only numerical harness.
- `TensileLiteBackwardCompatibilityValidation.md`: this record.

Generated binaries, libraries, and logs are kept under the ignored `build/tensilelite-backward-compat/` directory.

## Local git history of `--no-compress`

The observed behavior is **expected, not a regression in `--no-compress`**. Local history shows that assembly code-object compression and MessagePack catalog compression were introduced as separate features, in separate PRs, about 18 months apart. The later MessagePack change deliberately made `.dat.zlib` the build artifact and did not extend `--no-compress` to metadata.

### The flag was introduced for assembly code objects

The original hipBLASLt commit is `c1f9582f7ca4aa5e60ff7ac91710eed1e0fdb4fc`. Its monorepo-imported equivalent is `36defa5e8c4af75ec23dc10dcc8e5f8673cb565a`, dated 2024-12-06, with the same subject, `Code object compression via bundling (#1374)`. The imported commit body says:

```text
* feat: compress code objects
* feat: add --no-compress flag

[ROCm/hipBLASLt commit: c1f9582f7ca4aa5e60ff7ac91710eed1e0fdb4fc]
```

Thus the originating PR was [ROCm/hipBLASLt PR #1374](https://github.com/ROCm/hipBLASLt/pull/1374). Both the original object `c1f9582f...` and the imported object `36defa5e8c...` are available in the local git object database, and the latter's trailer records the former.

The exact user-facing wording added by that commit was:

- `TensileCreateLibrary --help`: `Don't compress assembly code objects.`
- `projects/hipblaslt/install.sh --help`: `don't compress assembly code objects generated by tensilelite`

The implementation added a `compress` argument only to the assembly code-object path. After linking assembly `.o` files into a temporary `.co.raw`, it either called the Clang offload bundler's `--compress` operation to produce the destination `.co`, or, for `--no-compress`, moved the raw linked code object to that `.co` path unchanged. The source/helper code-object builder was called separately and received no compression flag. Primary source: `git show 36defa5e8c4af75ec23dc10dcc8e5f8673cb565a`, especially:

- `projects/hipblaslt/tensilelite/Tensile/TensileCreateLibrary.py` in that commit, where `useCompression = not args.NoCompress` is passed only as `compress=useCompression` to the kernel writer;
- `projects/hipblaslt/tensilelite/Tensile/BuildCommands/AssemblyCommands.py` in that commit, lines 64-106, where `compressCodeObject(...)` and `shutil.move(...)` are the two branches;
- `projects/hipblaslt/tensilelite/Tensile/BuildCommands/SharedCommands.py` in that commit, lines 8-30, where `compressCodeObject()` invokes the bundler with `--compress`;
- `projects/hipblaslt/install.sh` in that commit, whose help text names “assembly code objects generated by tensilelite.”

At the original `c1f9582f...` commit, `tensilelite/Tensile/LibraryIO.py` was unchanged from its parent and `writeMsgPack()` still wrote raw MessagePack directly to the requested `.dat` file. In other words, PR #1374 introduced the flag when metadata compression did not yet exist, and the flag's diff had no metadata-writing branch to control.

The PR discussion makes the separation explicit. When a reviewer suggested compressing the MessagePack files too, the author replied: “for the scope of this PR we'll keep it to code object files and add the .dat file compression in another PR.”

The current tree preserves the same scope in several exact descriptions:

- `Tensile/TensileCreateLibrary/ParseArguments.py`: `Don't compress assembly code objects.`
- `projects/hipblaslt/tasks.py`: `Don't compress TensileLite assembly objects.`
- `projects/hipblaslt/install.sh`: `Don't compress asm objects`
- `projects/hipblaslt/README.md` and `projects/hipblaslt/device-library/CMakeLists.txt`: `Do not compress device code object files` / `Do not compress device code object files.`

Current data flow also remains assembly-only: `ParseArguments.py` sets `UseCompression = not NoCompress`; `Run.py` passes that value to `buildAssemblyCodeObjectFiles()`; and `Toolchain/Assembly.py` either calls `bundler.compress(...)` or moves the uncompressed `.co.raw`. The adjacent `buildSourceCodeObjectFiles()` call has no such parameter, and `Common/GlobalParameters.py` labels `UseCompression` a `code-object compression toggle`. These are visible at current-tree lines `ParseArguments.py:91-96,235`, `Run.py:707-728,1279`, `Toolchain/Assembly.py:86-148`, and `Common/GlobalParameters.py:830-840`.

### MessagePack compression was a later, independent artifact contract

Compressed MessagePack was introduced by monorepo commit `79c2e3f11fad12477cec52aa0dc5a50619226b4f`, dated 2026-06-24, with subject `Compress .dat files with zlib to reduce installed package size (#8294)`. This is [ROCm/rocm-libraries PR #8294](https://github.com/ROCm/rocm-libraries/pull/8294). Its summary explicitly calls the affected files “TensileLite `.dat` files (msgpack solution-selection catalogs)” and says compressed files use the `.dat.zlib` extension. Its build-time description says `writeMsgPack()` “compresses with `zlib.compress(level=9)`, and writes `<name>.dat.zlib`.”

That commit changed `LibraryIO.writeMsgPack()` from writing raw MessagePack to unconditionally applying zlib level 9 and writing `filename + ".zlib"`. Its new tests explicitly asserted that `library.dat` does **not** exist and `library.dat.zlib` does exist. The PR also changed the C++ reader to prefer `.zlib` and fall back to uncompressed `.dat`; that fallback is a reader compatibility feature, not a generator option. Primary source: `git show --format=fuller 79c2e3f11fad12477cec52aa0dc5a50619226b4f`, particularly `Tensile/LibraryIO.py`, `Tensile/Tests/unit/test_library_io.py`, and `src/msgpack/MessagePack.cpp` in that commit.

The ancestry is unambiguous: `36defa5e...` is an ancestor of `79c2e3f1...`, and both are ancestors of the tested producer revision. PR #8294 did not modify `TensileCreateLibrary/ParseArguments.py`, `TensileCreateLibrary/Run.py`, or the assembly toolchain; it modified `LibraryIO.py` and the MessagePack readers/tests. Current `Run.py:1324-1347` routes the lazy mapping, master catalog, and lazy solution catalogs through `LibraryIO.write(...)`. Current `LibraryIO.writeMsgPack()` still documents `Writes data to file in compressed Message Pack format (.dat.zlib).` and unconditionally executes `zlib.compress(raw, 9)` before writing `<filename>.zlib`; `writeMsgPackIndexed()` declares the same `.dat.zlib` contract.

Therefore the artifact split is intentional:

| Artifact class | Default | With `--no-compress` | Controlling feature |
| --- | --- | --- | --- |
| Linked assembly solution code object (`*.co`) | Offload-bundler-compressed | Raw linked code object at the same `.co` path | PR #1374 / `UseCompression` |
| HIP source/helper code object (`*.hsaco`) | Unchanged by this flag | Unchanged by this flag | Separate source builder |
| MessagePack solution catalogs and lazy mappings (`*.dat.zlib`) | zlib level 9 | Still zlib level 9, byte-identical for identical input | PR #8294 / `LibraryIO.writeMsgPack*()` |

The generic-looking option name can invite a broader reading, but every original and current help description limits it to code objects, and the implementation has never connected it to MessagePack serialization. An uncompressed-MessagePack generation mode would require a new `LibraryIO` policy or CLI option; it is not what `--no-compress` means in this history.

## Introducing PR provenance

Here, **direct commit** means the commit whose diff first changed the relevant producer/consumer contract in the history that reached `develop`. For the two large gfx1250 PRs, that source-changing commit is listed separately from the PR's integration commit.

| Compatibility boundary | Direct introducing commit and PR | Contract introduced | Confidence / ambiguity |
| --- | --- | --- | --- |
| Compressed MessagePack output | [`79c2e3f11fad12477cec52aa0dc5a50619226b4f`](https://github.com/ROCm/rocm-libraries/commit/79c2e3f11fad12477cec52aa0dc5a50619226b4f), [PR #8294, “Compress .dat files with zlib to reduce installed package size”](https://github.com/ROCm/rocm-libraries/pull/8294) | `writeMsgPack()` and indexed MessagePack output became zlib-compressed `<name>.dat.zlib`; the new reader prefers `.zlib` and only then falls back to `.dat`. | **High.** The commit directly changes writer, reader, and tests. The older [ROCm/hipBLASLt PR #1374](https://github.com/ROCm/hipBLASLt/pull/1374) is only the assembly code-object compression feature and is not the `.dat.zlib` introducer. |
| Architecture-qualified lazy mapping filename | [`5dc844a6443a5bc6743aaefa180364b6befcbc28`](https://github.com/ROCm/rocm-libraries/commit/5dc844a6443a5bc6743aaefa180364b6befcbc28), [PR #6840, “Fix/hipblaslt per arch mapping kpack collision”](https://github.com/ROCm/rocm-libraries/pull/6840) | Replaced the shared `TensileLiteLibrary_lazy_Mapping.dat` contract with `TensileLiteLibrary_lazy_<arch>_Mapping.dat` on both generation and loading paths, deliberately without a legacy-name fallback. PR #8294 subsequently made the complete current name end in `.dat.zlib`. | **High.** [`74a8aa55f9e271dc8c13598c3f68dfc44acdcef2`](https://github.com/ROCm/rocm-libraries/commit/74a8aa55f9e271dc8c13598c3f68dfc44acdcef2) / [PR #579, “Don't load all dat for index related API”](https://github.com/ROCm/rocm-libraries/pull/579) introduced the original global mapping mechanism. [`e3d39443132d40001ba56e09f5c9ebeea49022e4`](https://github.com/ROCm/rocm-libraries/commit/e3d39443132d40001ba56e09f5c9ebeea49022e4) / [PR #7717, “[hipblaslt] unconditional per arch build output”](https://github.com/ROCm/rocm-libraries/pull/7717) later moved the already architecture-qualified artifacts into `library/<arch>/`; neither is the filename-change introducer. |
| `computeInputType` to `computeInputTypeA`/`computeInputTypeB`, and `TypesEqual` arity 5 to 6 | Direct source change [`46488928eeb1beae3c52281db3a417730e9ff624`](https://github.com/ROCm/rocm-libraries/commit/46488928eeb1beae3c52281db3a417730e9ff624), integrated by [`e70c7fc4f340559ecc39b12c2b74847069783849`](https://github.com/ROCm/rocm-libraries/commit/e70c7fc4f340559ecc39b12c2b74847069783849) through [PR #6375, “[hipblaslt] Initial 1250 Support Part 2”](https://github.com/ROCm/rocm-libraries/pull/6375) | The serialized solution fields became two required keys and the `TypesEqual` value changed from `array<DataType, 5>` to `array<DataType, 6>`. | **High.** The two changes occur together in the direct commit; the separate integration hash is given because #6375 merged a long multi-commit branch. |
| `GateResidualDataTypeWhiteList` and serialized `UseGateResidual` | [`c42e983504188ed4af24d3c26bc9e7f07247c975`](https://github.com/ROCm/rocm-libraries/commit/c42e983504188ed4af24d3c26bc9e7f07247c975), [PR #9121, “feat(hipsparselt-tensilelite): Gate Residual Epilogue - Phase 1 : FP16 on gfx942”](https://github.com/ROCm/rocm-libraries/pull/9121) | Added both emitted problem predicates and their C++ subclass/serialization registrations (`UseGateResidual` is implemented by `UseGateResidualEqual`). | **High.** The merge commit directly contains the producer and reader additions. |
| `FusedGemmA2A` | [`59f8fd37e861c5956a10b7683dc285fe1d818c47`](https://github.com/ROCm/rocm-libraries/commit/59f8fd37e861c5956a10b7683dc285fe1d818c47), [PR #10925, “feat(tensilelite): fuse an all-to-all epilogue into the GEMM store path”](https://github.com/ROCm/rocm-libraries/pull/10925) | Made `FusedGemmA2A` a serialized `ProblemType` predicate and registered the corresponding C++ predicate class. | **High.** [`3cb75b39b890c24f6d9807bec7144430c09fa737`](https://github.com/ROCm/rocm-libraries/commit/3cb75b39b890c24f6d9807bec7144430c09fa737) / [PR #11342, “feat(hipblaslt): add host API for the fused GEMM + all-to-all epilogue”](https://github.com/ROCm/rocm-libraries/pull/11342) later added host API/build gating; it did not introduce this catalog predicate. |
| `MXBlockA` and `MXBlockB` | Reader/type registration in [`3335bb1f0854a3968c00b6e7cc6b972006ffa273`](https://github.com/ROCm/rocm-libraries/commit/3335bb1f0854a3968c00b6e7cc6b972006ffa273), unconditional producer emission in [`4aa16ecc9211d2efec9b612718ec4dd11be4bf4b`](https://github.com/ROCm/rocm-libraries/commit/4aa16ecc9211d2efec9b612718ec4dd11be4bf4b), integrated by [`2d55d744840a09db752ee348518672fb74471f9d`](https://github.com/ROCm/rocm-libraries/commit/2d55d744840a09db752ee348518672fb74471f9d) through [PR #6374, “[hipblaslt] Initial 1250 Support Part 1”](https://github.com/ROCm/rocm-libraries/pull/6374) | Added the two predicate types/registrations, then made both values appear in generated problem predicates even when their value is zero. | **High.** [`9e0422cfc34d2f7eb6cfc5a9a69085fc157da15f`](https://github.com/ROCm/rocm-libraries/commit/9e0422cfc34d2f7eb6cfc5a9a69085fc157da15f) / [PR #4702, “[hipBLASLt] Add block size into predicate for correct solution selection”](https://github.com/ROCm/rocm-libraries/pull/4702) is related but merged only to `gfx950_mx_rebase`; it is not the `develop` integration provenance. |
| `LaunchLimits` | [`5c74fe3d3c0ae7afd8f1fed1812cd4bc3389c1a8`](https://github.com/ROCm/rocm-libraries/commit/5c74fe3d3c0ae7afd8f1fed1812cd4bc3389c1a8), [PR #2747, “Detect if kernel launch would overflow hip grid limits”](https://github.com/ROCm/rocm-libraries/pull/2747) | Added the task predicate, its C++ implementation, and its serialization registration. | **High.** [`c19508578838d98937bba1cd9384b59bce7641f0`](https://github.com/ROCm/rocm-libraries/commit/c19508578838d98937bba1cd9384b59bce7641f0) / [PR #4033, “[hipblaslt] Tabulate debug predicates”](https://github.com/ROCm/rocm-libraries/pull/4033) later changed debug-predicate presentation, not the predicate contract. |
| Default `KernArgsVersion` 2 to 3 | [`33c75f0b6e32c854844b909bf5cbd9896e35952d`](https://github.com/ROCm/rocm-libraries/commit/33c75f0b6e32c854844b909bf5cbd9896e35952d), [PR #11923, “pkareorder”](https://github.com/ROCm/rocm-libraries/pull/11923) | Changed the default to version 3 and reordered the actual host/device kernarg layout while retaining explicit version-1/2 generation paths. | **High.** This is an ABI change, not merely a catalog-version label. |
