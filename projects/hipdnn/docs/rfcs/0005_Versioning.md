# Versioning

- Contributors: Jeremy Hart
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

hipDNN currently has a single version defined in the top level CMakeLists.txt file. There is no mechanism in place for ensuring that different components that interact together are compatible, aside from a possible compilation failure at build time.
## 4. Proposed Design
### 4.1 Version specification

Version information should be set in a version.json file, read into cmake, and then output into a header file with the rocm-libraries git hash used in place of the tweak.

Below is sample of how this project could be implemented

__version.json__
```json
{
	"<component>_version": 1.3.0
}
```

__CMakeLists.txt__
```cmake
# Read version from json
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.json VERSION_JSON)

string(JSON <COMPONENT>_VERSION GET ${VERSION_JSON} "<component>_version")

# Get commit hash
execute_command(COMMAND "git rev-parse --short HEAD"
                OUTPUT_VARIABLE <COMPONENT>_TWEAK
                OUTPUT_STRIP_TRAILING_WHITESPACE)

string(APPEND <COMPONENT>_VERSION ".${<COMPONENT>_TWEAK}")

project(<COMPONENT> VERSION ${<COMPONENT>_VERSION} LANGUAGES CXX)

# Generate version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/<component>_version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/<component>_version.h
    @ONLY
)
```

The `CMakeLists.txt` file is then responsible for parsing the json file for the version number, adding the git commit hash as the tweak version, and configuring the version header.

__include/\<component\>\_version.h.in__
```c
#pragma once
#define <COMPONENT>_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define <COMPONENT>_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define <COMPONENT>_VERSION_PATCH @PROJECT_VERSION_PATCH@
#define <COMPONENT>_VERSION_STRING "@PROJECT_VERSION@"
```

Note: The `configure_file` cmake function will replace the names between the at signs with matching cmake variables, and the modified file will be installed with the component.

### 4.2 Version bumps
Our versioning follows the guidelines of [semantic versioning](https://semver.org/), unless otherwise specified. The version follows the format `MAJOR.MINOR.PATCH.TWEAK`.

If a PR requires a version bump, the version must be updated as part of the PR or its merging process. Doing this manually within the PR will be necessary initially, but this will create merge conflicts for any PRs changing the same component. To alleviate this, this process should be automated using labels, PR comments, or some similar mechanism as soon as feasible.

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

Excluding components that are compatible within major versions (See section 4.6), the required version for all internal dependencies should be read from the version.json file in that component.

### 4.4 Public API
Excluding the exception below, everything included in headers available to the user are part of the public API. Some of the library components have been split into multiple APIs with a user-facing header.

If something must be included in a user-facing header, but shouldn't be part of the public API, it should be put inside a `detail` namespace. Anything in a detail namespace should not be used outside of the library internals, as it can be changed or removed at any time.

### 4.5 Versioned components
In the proposed design, the following table lists the components of hipDNN, and the other internal components that they depend on (omitting the hipdnn_ at the start of each)

| Target | Requirements | Dynamic version query? |
| -- | -- | -- |
| schema [^1] | | Yes |
| data_sdk | - schema | No |
| plugin_sdk | - data_sdk | No |
| backend | - data_sdk<br>- plugin_sdk<br> | Yes |
| frontend | - backend<br>- data_sdk<br>- schema | No |
[^1]: Schema refers to both serialized formats of flatbuffers and json.

### 4.6 Individual component details
#### 4.6.1 Schema
The schema covers both the flatbuffer serialization and the json serialization. These two serializations should change in lockstep as new fields, enums and structs are added. The version should be stored in the `file_identifier` field of the schema, which will encode it at the beginning of every serialization. Unlike other versions, this should only be increased on a major release in which the schema undergoes breaking changes.

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

The backend must be entirely agnostic to the schema version within the same major version (Note, this is not the current behaviour).

#### 4.6.3 Frontend
For the time being, the frontend must be forward compatible with the hipdnn_backend.

In the future, when the library is updated to use dlopen to link to the backend, backwards compatibility will be required as well. The frontend should be robust to all compatible backend libraries, even those that it wasn't compiled against.

When new functionality is added dependent on a backend feature that was added since the last major release (or the moment when we commit to backwards compatibility), there must dynamic guards placed to ensure that the backend version is new enough to support that feature. The `IHipdnnBackend` interface from `frontend/include/HipdnnBackendInterface.hpp` is a good location to place these guards when appropriate.

When a function is called that is not supported by the linked backend version, the frontend should fail gracefully, return the appropriate error status and log the event as needed.

#### 4.6.4 Plugins
Plugins must have a function that reports the plugin version they support. The plugin is required to have implemented every function in its corresponding plugin API for this version, and tests will be put in place to verify that every function has a definition. Note, the reported API version doesn't need to match the header they were compiled against. It's expected that plugin's will report the API version they have implemented.

Plugins are responsible for failing gracefully on unsupported schemas. Typically this will involve logging a warning, and returning false in the `isApplicable` API.

### 4.7 Compatibility summary

#### Schema &#8594; All consuming components
	- **Forwards compatible** within major versions
	- **Backwards compatibile** within major versions
#### Backend &#8594; Frontend
	- **Forwards compatible** within major versions
	- In the future, **Backwards compatible** within major versions
#### All components &#8594; All consuming components
	- **Forwards compatible** within major versions

## 5. Key Design Decisions

### 5.1 Version syncing between CMake and C++
**Decision**: Set version in json file, read into cmake, output into header version file
**Rationale**:
- Provides a simple and consistent pattern for updating the version using automation
- Allows for patch version to be set dynamically based on the rocm-libraries commit hash
	- This alleviates the need for CI to run in serial due to merge conflict resolution if multiple PRs make changes requiring a bump to the same library component patch
**Drawbacks**:
- Requires additional files
- Requires git to be installed to get patch number
- Patch number increases even if no change has been made

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
**Decision**: Excluding the dependencies that must be compatible within major versions, all dependencies should have a required version that's set to the current version in that component's version.json
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
- Have CI run backward/forward compatibility tests on PRs

## 7. Execution Plan

### 7.1 Phase 1 (Prerequisites)
- Make hipdnn_backend agnostic to the schema version
	- Ensure that it doesn't fail with an older or newer schema
- Separate hipDNN into multiple cmake projects
	- If a version is required at the point of release, set it to 1.0.0
- Ensure flatbuffer and json deserializations are robust to new parameters, unions, enums, etc...
	- This impacts plugins, and the data_sdk
- Add note to our documentation that the `detail` namespace is not part of the public API
	- Move anything into the detail namespace that shouldn't be part of the public API
### 7.2 Phase 2 (Add necessary versioning)
- Add versions to json and flatbuffer serializations
- Add version.json files
	- Have the version propagated properly from this file, to the cmake, to the version header
	- Also fetch the rocm-libraries git commit for the tweak version
- Have components use current dependencies version.json as their minimum required versions
- Add api functions for reporting library version
### 7.3 Phase 3 (Expand on versioning)
- Create automation for bumping version
- Add frontend infrastructure for checking backend version
	- Should be timed close to the first minor change added
- Expand testing to ensure forward and backwards compatibility between components
## 8. Testing Plan
### 8.1 Backend backwards compatible with frontend
**Backend API test**
- Build frontend with a hipdnn_backend from the start of the major release
- Run a test suite to ensure all new features fail gracefully and as intended
**Backend test (updated API)
- Build frontend with current hipdnn_backend
- Use dllopen (or another utility) to switch to an old hipdnn_backend_private library
- Run the same test suite to ensure all new features fail gracefully and as intended
- Dependent on ability to dynamically change backend library
### 8.2 Schema files forwards and backwards compatibility
- Generate old serializations and ensure they can be loaded by data_sdk and plugins
	- By either functioning properly or failing gracefully
- Add to schema to generate "future" serializations, and ensure that they can be handled properly
	- Need to expand Unions, add new values to enums, and add new fields to tables
### 8.3 Testing for version bumping automation
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
