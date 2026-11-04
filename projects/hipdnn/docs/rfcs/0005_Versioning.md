# Versioning

- Contributors: Jeremy Hart
- **Status**: In progress
- **Status notes**: Per-component `version.json`, generated `version.h` headers, git-hash tweak, and minimum dependency versions shipped (#4176, #4311, #4654, #5147). The frontend checks the backend major version and has a runtime dlopen mode (#8792); `scripts/bump-hipdnn-versions.sh` reports and applies bumps (#9350). Schema versioning in `file_identifier`, compatibility CI tests, and fully automated bumps are not built.
- **Last verified**: 2026-10-15 against `develop` (2501d0b28cc)

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Risks](#6-risks)
7. [Execution Plan](#7-execution-plan)
8. [Testing Plan](#8-testing-plan)
9. [Future Considerations](#9-future-considerations)
10. [Glossary](#10-glossary)

## 1. Executive Summary

This RFC outlines the details of a versioning system and guidelines that should be adhered to following the release of hipDNN.

To help provide clarity to developers and consumers of hipDNN, this document seeks to establish a clear set of rules for what falls under the public API, how it can change as the version number increases and what guarantees for backwards and forwards compatibility will exist.

This document also outlines the work and code infrastructure that is intended to facilitate the versioning system in a way that is as minimally error-prone and unobtrusive as possible.
## 2. Problem Statement

hipDNN consists of many different components which all work together for the purpose of running DNN workflows. While many of these components are likely to be compiled or used in a context that guarantees compatible versions, there are contexts and components where this can't be relied upon. The consumer of each of these components, be they internal or external to hipDNN, needs a versioning system to ensure that component is in a state that they can support.

We want the following guarantees from this system:
- The frontend is forward and eventually backward compatible with a dynamically linked hipdnn_backend within a major version
- Within a major version, serialized schemas are forwards and backwards compatible with all other components of the library
	- If they represent an operation that hasn't yet been supported by a plugin, it will gracefully fail to match
- Increasing the version of a component of hipDNN is easy and unobtrusive
- The components of hipDNN are able to communicate their version in such a way that their dependent components are able to properly determine what operations and features can be supported
## 3. Current System Overview

Each component of hipDNN has its own version (section 4.5). The umbrella `hipdnn` project in the top-level `CMakeLists.txt` has no version.

Components check compatibility in two places:
- At build time, each package config file requires its internal dependencies at a minimum version (section 4.3).
- At runtime, the frontend checks the backend major version (section 4.6.3), and the backend checks the plugin API major version of each plugin (section 4.6.4).
## 4. Proposed Design
### 4.1 Version specification

Each component has a `version.json` and a `version.h.in` file in its component directory (for example `backend/version.json`). CMake reads the version from `version.json` and generates a version header. The rocm-libraries short git commit hash is the tweak version. If CMake cannot get a commit hash (for example, git is not installed or the source is not a git checkout), the tweak is `unknown`.

`cmake/VersionUtils.cmake` provides these functions:
- `hipdnn_setup_version(<component>)` reads the `<component>_version` key from `version.json` and gets the tweak. It sets `<COMPONENT>_VERSION`, `<COMPONENT>_VERSION_MAJOR`, `<COMPONENT>_VERSION_MINOR`, `<COMPONENT>_VERSION_PATCH`, `<COMPONENT>_VERSION_TWEAK`, and `<COMPONENT>_VERSION_STRING` (`MAJOR.MINOR.PATCH.TWEAK`).
- `hipdnn_generate_version_header(<component>)` configures `version.h.in` into `include/<component>/version.h` in the build directory.
- `hipdnn_get_component_version(<component> <output_var>)` reads the version of another component (section 4.3).

The component directory is the component name without the `hipdnn_` prefix. For example, `hipdnn_data_sdk` reads `data_sdk/version.json`.

__data_sdk/version.json__
```json
{
    "hipdnn_data_sdk_version": "0.2.0"
}
```

__data_sdk/CMakeLists.txt__
```cmake
# Version handling
hipdnn_setup_version(hipdnn_data_sdk)

# Set project with version
project(hipdnn_data_sdk VERSION ${HIPDNN_DATA_SDK_VERSION} LANGUAGES CXX)

add_library(hipdnn_data_sdk INTERFACE)

# Generate version header
hipdnn_generate_version_header(hipdnn_data_sdk)
```

Each component `CMakeLists.txt` calls `hipdnn_setup_version()` and then calls `project()` with the version that it sets.

__data_sdk/version.h.in__
```c
#define HIPDNN_DATA_SDK_VERSION_MAJOR @HIPDNN_DATA_SDK_VERSION_MAJOR@
#define HIPDNN_DATA_SDK_VERSION_MINOR @HIPDNN_DATA_SDK_VERSION_MINOR@
#define HIPDNN_DATA_SDK_VERSION_PATCH @HIPDNN_DATA_SDK_VERSION_PATCH@
#define HIPDNN_DATA_SDK_VERSION_TWEAK "@HIPDNN_DATA_SDK_VERSION_TWEAK@"
#define HIPDNN_DATA_SDK_VERSION_STRING "@HIPDNN_DATA_SDK_VERSION_STRING@"
```

Note: The `configure_file` cmake function replaces the names between the at signs with matching cmake variables. The SDK and frontend components install the generated header as `<component>/version.h`. The backend uses its header internally and reports its version through `hipdnnVersionString_ext()`.

### 4.2 Version bumps

Our versioning follows the guidelines of [semantic versioning](https://semver.org/), unless otherwise specified. The version follows the format `MAJOR.MINOR.PATCH.TWEAK`.

If a PR requires a version bump, the PR must update the version in `version.json`. `scripts/bump-hipdnn-versions.sh` helps with this. It compares each component against a release ref (by default, the highest `origin/release/therock-*` branch) and reports the components that changed but did not bump `version.json`. With `--apply`, it writes the bumps. The script cannot choose the bump level; the author must choose it.

Not yet built. Manual bumps cause merge conflicts between PRs that change the same component. To prevent this, automation should apply the bump when the PR merges, for example from PR labels or PR comments.

**Note**: Comments and documentation do not require a version update (though the tweak version will still change)

#### Major version
In major version updates, API breaking changes may be made, and deprecated features may be removed. Updates must coincide with a rocm major release. Note that a rocm major release will not require a hipDNN major version bump. However, if any component undergoes a major version bump every component should, ensuring that all components will share a major version.

The process for major version bumps is TBD, however we will be ensuring that breaking changes are deprecated ahead of time to give consumers advanced notice.

#### Minor version
API and features additions can be added within a minor version bump, but nothing can be removed. If a function or class is no longer needed, it should be marked as deprecated using the `deprecated` attribute. If appropriate, logging should be added to warn about its usage.

Any API feature additions should have an associated comment identifying the version they were introduced. This will require special handling when version bumping is automated and the version isn't known until the PR has been merged.

**Note**: There is a distinction between C and C++ APIs. In our C++ APIs, new overloads can be added. In C APIs, overloads are considered a breaking change, and only new functions may be added.

#### Patch version
The patch is updated on backward compatible bug fixes.

#### Tweak version
The tweak version is a fourth identifier added to the version determined by the short commit hash of rocm-libraries, and updates automatically as a result.

#### Public dependencies

For any components with dependencies that expose headers from another component, their version *must* also increase when that dependency's version increases.

**Ex**:
- The hipdnn_plugin_sdk version is 1.6.3
- hipdnn_data_sdk version increases from 1.3.5 to 1.4.0
- hipdnn_plugin_sdk version must be increased to 1.7.0

### 4.3 Version requirements

Each component requires its internal dependencies at the version in the `version.json` of that dependency. `hipdnn_get_component_version()` reads this version at build time, and the package config file passes it to `find_dependency()`. The package version files use `SameMajorVersion`, so CMake accepts an installed dependency with the same major version and a version equal to or greater than the required version.

At runtime, the frontend accepts any backend with the same major version (section 4.6.3).

### 4.4 Public API
Excluding the exception below, everything included in headers available to the user are part of the public API. Some of the library components have been split into multiple APIs with a user-facing header.

If something must be included in a user-facing header, but shouldn't be part of the public API, it should be put inside a `detail` namespace. Anything in a detail namespace should not be used outside of the library internals, as it can be changed or removed at any time.

### 4.5 Versioned components

The following table lists the versioned components of hipDNN (omitting the `hipdnn_` at the start of each). The table also shows the components whose version each component requires (section 4.3), and how each component reports its version at runtime.

| Target | Requirements | Dynamic version query? |
| -- | -- | -- |
| flatbuffers_sdk [^1] | - data_sdk | No |
| data_sdk | | No |
| plugin_sdk | - data_sdk | No [^2] |
| backend | | Yes, `hipdnnVersionString_ext()` |
| frontend | - data_sdk<br>- backend | No |
| test_sdk | - data_sdk<br>- plugin_sdk | No |
[^1]: `flatbuffers_sdk` contains the schemas: both the flatbuffer schema files and the json serialization. There is no separate schema version.
[^2]: Plugins built with `plugin_sdk` report their versions with `hipdnnPluginGetVersion()` and `hipdnnPluginGetApiVersion()` (section 4.6.4).

### 4.6 Individual component details
#### 4.6.1 Schema

The schema covers both the flatbuffer serialization and the json serialization. Both are in `hipdnn_flatbuffers_sdk`. These two serializations should change in lockstep as new fields, enums and structs are added. Schema changes follow the `hipdnn_flatbuffers_sdk` version.

The `file_identifier` of a schema is a fixed type tag: `"HDGP"` in `serialized_graph_and_plan.fbs` and `"HDDP"` in `device_properties.fbs`. A graph can state the minimum engine plugin API version that it needs in the `min_required_engine_api_version` field of the `Graph` table in `graph.fbs` (see [RFC 0016](0016_RuntimePassByValueTensors.md)).

Not yet built. A schema version should be stored in each serialization. The design puts it in the `file_identifier` field, which encodes it at the beginning of every serialization. Unlike other versions, this version only increases on a major release in which the schema has breaking changes.

Schemas should be backwards and forward compatible within a major release. To this aim, the following changes can be made within the same major version:
- Adding new structs, tables, enums, unions
- Adding a new field to the end of a table
- Adding a new value to the end of an enum (TODO: Need to verify this)
- Adding a new variant to the end of a union
- Adding the `deprecated` attribute to a field

The following is a non-exhaustive list of changes that cannot be made within the same major version:
- Changing the type of a field
- Changing the order of fields
-  Removing a field
- Modifying a struct in any way

On a new major release, all `deprecated` fields should be removed, and other breaking changes may occur.

Json serialization changes should be made in concert with flatbuffer schema files.

#### 4.6.2 Backend
The backend has two components: hipdnn_backend (the API) and hipdnn_backend_private (the implementation). These components share a single version.

The backend must be entirely agnostic to the schema version within the same major version.

#### 4.6.3 Frontend

The frontend must be forward and backward compatible with the hipdnn_backend within a major version. The frontend should be robust to all compatible backend libraries, even those that it was not compiled against.

The frontend has two CMake targets:
- `hipdnn_frontend` links the backend directly.
- `hipdnn_frontend_dynamic` defines `HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND`. It does not link the backend. It loads the backend shared library at runtime. The `HIPDNN_BACKEND_LIBRARY_PATH` environment variable can set the directory that the frontend loads the backend from.

Both targets call `checkBackendVersion()` in `frontend/include/hipdnn_frontend/detail/BackendWrapper.hpp` on the string from `hipdnnVersionString_ext()`. If the string is missing, cannot be parsed, or has a major version other than `HIPDNN_FRONTEND_VERSION_MAJOR`, the frontend logs an error and uses `IncompatibleBackendWrapper`. Its status calls return `HIPDNN_STATUS_NOT_INITIALIZED`. With `hipdnn_frontend_dynamic`, a call to a backend entry point that the loaded backend does not export returns `HIPDNN_STATUS_VERSION_MISMATCH`.

When new functionality is added dependent on a backend feature that was added since the last major release, there must be dynamic guards to ensure that the backend version is new enough to support that feature. The `IHipdnnBackend` interface from `frontend/include/hipdnn_frontend/detail/HipdnnBackendInterface.hpp` is a good location to place these guards when appropriate. `IHipdnnBackend::version()` returns the parsed backend version.

When a function is called that is not supported by the linked backend version, the frontend should fail gracefully, return the appropriate error status and log the event as needed.

#### 4.6.4 Plugins

Each plugin exports `hipdnnPluginGetVersion()` and `hipdnnPluginGetApiVersion()`, declared in `plugin_sdk/include/hipdnn_plugin_sdk/PluginApi.h`. `hipdnnPluginGetVersion()` reports the version of the plugin. `hipdnnPluginGetApiVersion()` reports the version of the plugin API that the plugin implements. The plugin must implement every function in its plugin API for this version. The reported API version does not need to match the header that the plugin was compiled against.

The engine plugin API and the heuristic plugin API have their own versions, independent of the backend version: `HIPDNN_ENGINE_API_VERSION` in `engine_api_version.h` and `HIPDNN_HEURISTIC_API_VERSION` in `heuristic_api_version.h`. When the backend loads a plugin, it:
- Resolves the required plugin API symbols, and rejects the plugin if one is missing.
- Rejects an engine plugin with an API version that cannot be parsed, or with a major version other than `HIPDNN_ENGINE_API_VERSION_MAJOR` (`backend/src/plugin/EnginePluginManager.hpp`).
- Rejects a heuristic plugin with a major version other than `HIPDNN_HEURISTIC_API_VERSION_MAJOR` (`backend/src/plugin/HeuristicPluginManager.hpp`).

If an engine plugin does not export `hipdnnPluginGetApiVersion()`, the backend uses API version `1.0.0` for it.

> [!NOTE]
> The backend also loads engine plugins that report API major version 0, and logs an info message. This transitional case goes away at the next engine API major version.

Plugins are responsible for failing gracefully on unsupported schemas. Typically this will involve logging a warning, and returning false in the `isApplicable` API.

### 4.7 Compatibility summary

#### Schema &#8594; All consuming components
	- **Forwards compatible** within major versions
	- **Backwards compatibile** within major versions
#### Backend &#8594; Frontend
	- **Forwards compatible** within major versions
	- **Backwards compatible** within major versions
#### All components &#8594; All consuming components
	- **Forwards compatible** within major versions

## 5. Key Design Decisions

### 5.1 Version syncing between CMake and C++
**Decision**: Set version in json file, read into cmake, output into header version file
**Rationale**:
- Provides a simple and consistent pattern for updating the version using automation
- Allows for the tweak version to be set dynamically based on the rocm-libraries commit hash
	- This alleviates the need for CI to run in serial due to merge conflict resolution if multiple PRs make changes to the same library component
**Drawbacks**:
- Requires additional files
- Requires git to get the tweak version. Without git, the tweak is `unknown`
- The tweak version changes on every commit, even if the component did not change

#### 5.1.1 Alternatives

##### Set in header file

Set the version in the header file

```c++
#define <COMPONENT>_VERSION_MAJOR 0
#define <COMPONENT>_VERSION_MINOR 0
#define <COMPONENT>_VERSION_PATCH 1
```
And query it from cmake using the following pattern
```cmake
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/<component>_version.h" VERSION_HEADER)
string(REGEX MATCH "<COMPONENT>_VERSION_MAJOR ([0-9]+)" _ ${VERSION_HEADER})
set(VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "<COMPONENT>_VERSION_MINOR ([0-9]+)" _ ${VERSION_HEADER})
set(VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "<COMPONENT>_VERSION_PATCH ([0-9]+)" _ ${VERSION_HEADER})
set(VERSION_PATCH ${CMAKE_MATCH_1})

project(<component> VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})
```
**Drawbacks**:
- Can't dynamically set tweak version

### 5.2 Dependency versions requirements match the current version in version.json
**Decision**: All internal dependencies have a required version that is set to the current version in the `version.json` of that dependency
**Rationale**:
- This guarantees that the required version will always be compatible with the library
- No automation or manual effort is required to maintain this. Implementing this automation would be difficult (maybe even infeasible) and likely require a large amount of CI time for compilation. Handling it manually would be likely be error prone and require dedicated effort
**Drawbacks**:
- While this guarantees that required versions will always be compatible, the required versions will have stronger requirements than necessary. Ideally, the required version would only be updated when a new function or feature is needed.


## 6. Risks
### 6.1 Flatbuffer and Json schema definitions getting out of sync
**Risk**: The flatbuffer schema might be updated without also updating the json serialization
**Mitigation**
- Create tests creating serializations for all flatbuffer Unions and ensure that json can properly serialize them as well
- This test exists already for `NodeAttributes`
### 6.2 Backwards/Forwards compatibility breaking
**Risk** Backwards or Forwards compatibility could become broken accidentally for some time before it's noticed. While this can often be remedied, it could lead to versions of library components that are not forward and backward compatible with each other.
**Mitigation**
- Have CI run backward/forward compatibility tests on PRs. Not yet built.

## 7. Execution Plan

### 7.1 Phase 1 (Prerequisites)
- Make hipdnn_backend agnostic to the schema version
	- Ensure that it doesn't fail with an older or newer schema
- Separate hipDNN into multiple cmake projects
- Ensure flatbuffer and json deserializations are robust to new parameters, unions, enums, etc...
	- This impacts plugins, and the data_sdk
- Add note to our documentation that the `detail` namespace is not part of the public API (`docs/CodingStyleAndNamingGuidelines.md`, section 10.1)
	- Move anything into the detail namespace that shouldn't be part of the public API
### 7.2 Phase 2 (Add necessary versioning)
- Add versions to json and flatbuffer serializations. Not yet built.
- Add version.json files
	- Have the version propagated properly from this file, to the cmake, to the version header
	- Also fetch the rocm-libraries git commit for the tweak version
- Have components use current dependencies version.json as their minimum required versions
- Add api functions for reporting library version
### 7.3 Phase 3 (Expand on versioning)
- Create automation for bumping version
	- `scripts/bump-hipdnn-versions.sh` reports and applies bumps
	- Applying bumps automatically when a PR merges is not yet built
- Add frontend infrastructure for checking backend version
- Expand testing to ensure forward and backwards compatibility between components. Not yet built.
## 8. Testing Plan
### 8.1 Backend backwards compatible with frontend
Not yet built.

**Backend API test**
- Build frontend with a hipdnn_backend from the start of the major release
- Run a test suite to ensure all new features fail gracefully and as intended
**Backend test (updated API)**
- Build the test suite with `hipdnn_frontend_dynamic` and the current hipdnn_backend
- Set `HIPDNN_BACKEND_LIBRARY_PATH` to the directory of an older hipdnn_backend library
- Run the same test suite to ensure all new features fail gracefully and as intended
### 8.2 Schema files forwards and backwards compatibility
Not yet built.

- Generate old serializations and ensure they can be loaded by data_sdk and plugins
	- By either functioning properly or failing gracefully
- Add to schema to generate "future" serializations, and ensure that they can be handled properly
	- Need to expand Unions, add new values to enums, and add new fields to tables
### 8.3 Testing for version bumping automation
Not yet built.

- Details pending
## 9. Future Considerations
- Determine procedure for major version bumps
- Create a requirements file where forward and backwards compatibility is not guaranteed and ensure that each component compiles and runs properly with the versions allowed for in their requirements
## 10. Glossary
**Backward compatibility**: The ability of a newer version of a component to work correctly with older versions of its dependencies or consumers.

**Conventional commits**: A specification guideline on commits that conveys whether a change requires a patch, minor, or major version update. Often useful for automating version updates.

**Deprecated attribute**: A marker (e.g., `[[deprecated]]` in C++) applied to functions, classes, or fields to indicate they are obsolete and should no longer be used. Deprecated items may be removed in the next major version.

**Flatbuffer**: A cross-platform serialization library developed by Google that enables efficient serialization of data with zero-copy deserialization. Used by hipDNN for graph and configuration serialization.

**Forward compatibility**: The ability of an older version of a component to work correctly with newer versions of its dependencies or consumers.

**Major version**: The first number in a semantic version (X.y.z). Incremented when backward-incompatible changes are made. In hipDNN, major version updates coincide with ROCm major releases.

**Minor version**: The second number in a semantic version (x.Y.z). Incremented when new features or API additions are made in a backward-compatible manner.

**Patch version**: The third number in a semantic version (x.y.Z). Incremented for backward-compatible bug fixes.

**Public API**: The set of functions, classes, types, and interfaces that are officially supported for external use. Changes to the public API follow strict versioning rules.

**Schema**: The flatbuffer schema files (`.fbs`) that are used by hipDNN to define graphs, engine configurations, and engine details. In this document, it also refers to the corresponding JSON serialization of the same information.

**Semantic versioning (SemVer)**: A versioning specification that uses a three-part version number (MAJOR.MINOR.PATCH) with defined rules for when each part should be incremented based on the type of changes made.

**Tweak version**: An optional fourth component of a version number, used in hipDNN to represent the git commit hash of the rocm-libraries repository for precise build identification.
