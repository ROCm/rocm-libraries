# Lane: public API (`API`)

Every change to a public surface must be intentional and described in the PR.

## Surfaces

| Surface | Where |
|---|---|
| Backend C API and ABI | `projects/hipdnn/backend/include/` |
| Frontend C++ | `projects/hipdnn/frontend/include/`; the cuDNN shim `projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/` (default OFF, `HIPDNN_ENABLE_CUDNN_COMPATIBILITY`) |
| Plugin and heuristic API | `projects/hipdnn/plugin_sdk/include/`, including ABI version headers |
| Data and FlatBuffers SDKs | `projects/hipdnn/data_sdk/include/`, `projects/hipdnn/flatbuffers_sdk/include/` |
| Test SDK | `projects/hipdnn/test_sdk/` (installed in the Development component; providers consume it) |
| Serialized format | `projects/hipdnn/flatbuffers_sdk/schemas/*.fbs` |
| Python | `hipdnn_frontend` exports and nanobind bindings: `projects/hipdnn/python/frontend_bindings`, `projects/hipdnn/python/frontend_wheel_package` |
| CMake package | Exported targets, config files, option names consumers set |
| Runtime contract | Environment variables, file formats and paths, status codes, log formats others parse |
| Provider contracts | `dnn-providers/**` non-test source: environment variables a provider reads, engine names and IDs, `config/<ENGINE>.toml` and support-claim sidecar schemas, installed provider CMake packages and exported targets |

A header outside `detail/` is public if it is installed. The `detail` namespace and `detail/` directories are not public even
though the frontend install rule ships them; flag `detail` types in public signatures. Behavior is a surface too: status codes,
defaults, environment variables, file formats or log formats changed in source files behind unchanged signatures are API changes.

## Method

1. Diff the surfaces between `$BASE` and `$HEAD`. Classify each changed declaration: added, removed, signature changed,
   semantics changed, default changed.
2. Find each in the PR description. Described and intended → note it. Not described → **Needs human decision**: unintended API
   change, with the exact declaration ([../policy.md](../policy.md) *Review rules*).
3. Breaking changes (removal, signature or semantics change, enum value reuse, struct layout change, `.fbs` field reorder or ID
   change) are `BLOCKING` unless described and allowed by `projects/hipdnn/docs/rfcs/0005_Versioning.md`: breaking changes only at
   a major release after deprecation; JSON and FlatBuffers change together; enum and union values only appended; frontend
   features needing a newer backend carry a backend-version guard. The RFC marks some sections as not yet current; cite the
   section you apply. Bump the plugin ABI version where the plugin contract changed.
4. `.fbs` changes keep backward and forward compatibility (field order and IDs, defaults) or declare and justify the break; check
   the generated object wrappers' behavior, not only the schema text.
5. A public API meant to match cuDNN: compare signature, defaults, status behavior and ownership against `NVIDIA/cudnn-frontend`
   at the tag `v<major>.<minor>.<patch>` that the shim's `CUDNN_FRONTEND_*_VERSION` macros declare (`gh api
   repos/NVIDIA/cudnn-frontend/contents/<path>?ref=<tag>` or a clone under `$SCRATCH/API/`). If that source is unreachable, put the
   point under **Needs human decision**; never compare from memory. Divergences in `projects/hipdnn/docs/KNOWN_DIVERGENCES.md` are
   intentional; any other is a finding. Files adapted from cudnn-frontend keep the NVIDIA notice, and a version-macro bump updates
   `projects/hipdnn/frontend/THIRD_PARTY_LICENSES.md`.
6. New public declarations carry documentation per `projects/hipdnn/docs/doxygen-guidelines.md`.
