# ROCm architecture catalog

`rocm-arch` provides a common catalog of AMD GPU target identities for CMake,
Python, and C++. Each architecture registers through a JSON definition in
`python/rocm_arch/architectures/`. The catalog has no GPU or HIP dependency.

## Definitions

The filename matches the canonical ID:

```json
{
  "schema_version": 1,
  "id": "gfx942",
  "kind": "processor",
  "release_year": 2023
}
```

`kind` is `processor` or `generic`. Processor identities include distinct
steppings such as `gfx1250-strict`. Names are exact and case-sensitive. Feature
suffixes, marketing aliases, compiler acceptance, and hardware resources are
outside the catalog. Library support and tuning are owned by each consumer.

Processor definitions may include `release_year`, describing the first publicly
released hardware using that processor target, at year granularity. It does not
describe compiler introduction or library support. Omit the field when the
release year is unknown. Generic targets have no hardware year.
The catalog contains no resource or tuning metadata.

Python reads the packaged definitions directly. CMake generates the native
catalog and C++ header in the build directory using the same validated loader.
Adding, changing, or removing a definition triggers reconfiguration on the next
build. Generated headers are not checked in.

## Build and install

Native generation requires CMake 3.25.2+ and host Python 3.9+. With tests disabled,
no compiler is required. Installed CMake and C++ consumers do not run the generator
or require Python, HIP, or a JSON parser. The header supports C++14 and later.

```sh
cmake -S shared/arch -B build/arch \
  -DROCM_ARCH_BUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --install build/arch
python3 -m pip install ./shared/arch
```

Native installation includes the Python package under `share/rocm-arch/python`.
The Python distribution can also be installed into a virtual environment.

## Interfaces

```cmake
include(rocm-arch-dependency)
rocm_arch_list(processors KIND PROCESSOR)
rocm_arch_list(recent KIND PROCESSOR MIN_YEAR 2020 MAX_YEAR 2025 ORDER RELEASE)
rocm_arch_list(generics KIND GENERIC)
rocm_arch_is_known(known ID gfx942)
target_link_libraries(my_consumer PRIVATE rocm::arch)
```

```cpp
#include <rocm/arch/catalog.hpp>
using architecture = rocm::arch::architecture_t;
static_assert(rocm::arch::from_name("gfx942") == architecture::gfx942, "identity");
static_assert(rocm::arch::is_known("gfx1250-strict"), "distinct stepping");
constexpr auto recent = rocm::arch::architectures_between<2020, 2025>();
static_assert(rocm::arch::released_between(architecture::gfx942, 2020, 2025), "release year");
```

C++ exposes name arrays `architectures` and `generic_targets`, typed processor
identities in `architecture_ids`, and canonical constants such as
`names::gfx942`. `from_name()` returns `architecture_t::Count` for an unknown
processor; `name()` returns a null pointer for an invalid identity. Generic
targets are separate from the processor enum. Enum ordinals are local to the
catalog version: persist canonical names, and rebuild consumers when updating
the native package.

```python
import rocm_arch
processors = rocm_arch.architectures()
recent = rocm_arch.architectures(min_year=2020, max_year=2025, order="release")
generics = rocm_arch.generic_targets()
assert rocm_arch.is_known("gfx942")
```

Year bounds are inclusive and independently optional. With no bounds, all
registered processors are returned, including undated identities. With either
bound, undated identities are excluded. Name order is the CMake/Python default;
release order sorts by year, breaks ties by canonical ID, and places undated
identities last. C++ `architectures_between<MinYear, MaxYear>()` returns a constexpr
name array in release order, with `0` for an omitted bound. `release_year()` returns
`0` in C++ or `None` in Python for an unknown year. Invalid or inverted bounds are
rejected by CMake/Python and the C++ template; the C++ predicate returns false.

Python lists are immutable tuples. All language views expose the same
SHA-256 catalog digest. Native arrays use constant initialization; C++17 uses
inline constants and C++14 uses constants local to each translation unit.

## Build dependencies

The root superbuild always adds `shared/arch` before selected components.
Consumers use `include(rocm-arch-dependency)` followed by a catalog query; no
`find_package()` or initialization call is needed. CMake finds the module through
`CMAKE_MODULE_PATH`, which points to `shared/arch/cmake` in the checkout or
`<prefix>/share/rocm-arch/cmake` for an installation. The library CMakeLists add
the checkout module path; standalone installed builds can set it at configure time:

```sh
cmake -S /path/to/consumer -B build/consumer \
  -DCMAKE_MODULE_PATH=/path/to/prefix/share/rocm-arch/cmake
```

Including the source module resolves an installed package or adds the shared
source. Source installation includes the dependency alongside its consumer.
Including the installed module loads the adjacent package configuration without
source access or generation. Repeated inclusion reuses the existing target.

rocFFT and hipFFT derive build defaults from the processor catalog with local
exclusions. rocFFT also derives its Python configuration enum from the catalog.
Origami uses the shared processor type and enumerates model-supported identities
by applying explicit exclusions; registration without a model or exclusion fails
its compile-time model check. Its Python enum uses that same supported set.

rocPRIM's legacy `GPU_TARGETS=all` defaults query the catalog with local exclusions
and XNACK feature policy. `ROCPRIM_MIN_ARCH_YEAR` and `ROCPRIM_MAX_ARCH_YEAR` are
optional CMake cache strings for those defaults; empty values are unbounded.
The separate ASAN defaults and explicit target selection use their own policy.
rocRAND's `rmake.py` derives defaults from the catalog with local exclusions and
accepts `--min-arch-year` and `--max-arch-year`; `--architecture` selects explicit
targets. Its tool requires the checkout or installed `rocm_arch` Python package.
rocSPARSE uses shared canonical names in its private C++ dispatch headers and
links the native catalog as a build dependency.

See [TESTING.md](TESTING.md) for validation commands.
