# Lane: correctness and code quality (`CORR<k>`)

Read every changed line of your owned files, and the outside code each change depends on.

## Correctness

- Trace each change from producer to consumer and from guard to caller to test. State the smallest input that breaks it.
- Error paths: every failure returns a status or throws as the surrounding API does; nothing is swallowed; partial state unwinds.
- Resource ownership (`projects/hipdnn/docs/ai-rules.md`, RAII & Resource Management): wrap owning raw pointers in RAII at once.
  FlatBuffers `Table::UnPack()` returns an owning raw pointer: wrap it in `std::unique_ptr<T>`. Only the `root_type` tables
  (`git -C "$CLONE" grep -n root_type "$HEAD" -- 'projects/hipdnn/flatbuffers_sdk/schemas/*.fbs'`) have a generated `UnPack<Name>()` returning
  `std::unique_ptr`. `getAttribute()` with `HIPDNN_TYPE_BACKEND_DESCRIPTOR` allocates a descriptor the caller owns. Provider
  handles, workspace buffers, streams and external-library resources are released on every failure path.
- PR CI does not catch leaks: its ASAN job builds sanitizer binaries without running tests
  (`projects/hipdnn/docs/testing/KNOWN_TESTING_GAPS.md`). Prove a leak by reasoning over the ownership path, or request a local
  `-DBUILD_ADDRESS_SANITIZER=ON` run of the affected tests from `RUN` under **Handoffs**.
- Providers: applicability predicates match what the implementation supports; registration names the right op, layouts, types and
  behavior notes; workspace size, stream use, async behavior and external-library calls are correct; unsupported inputs fail
  predictably.
- New operation: walk the PR Checklist in `projects/hipdnn/docs/AddingNewOperations.md`; report each layer the diff misses
  (schema and `SCHEMAS` list, `NodeAttributes` union, backend descriptor and factories, frontend
  attributes/packer/unpacker/node/Graph API, enum string utilities, CMake lists, tests per layer).
- Numerics: accumulation type, overflow, tolerance, NaN/Inf, denormals.
- Concurrency: shared state, caches, lazy initialization, handle and stream lifetimes.
- Portability: Windows paths and DLL search, `lib` vs `bin`, path separators, 32-bit sizes.
- Arch-specific code: gfx names, wave size, LDS limits and feature checks match the targets.

## Code quality

- Names and structure follow `projects/hipdnn/docs/CodingStyleAndNamingGuidelines.md`; `detail/` symbols never appear in public
  signatures.
- Dead code, unused parameters, commented-out code, unexplained magic numbers.
- Complexity that a simpler construct in the same codebase already solves.

Missing or weak tests are not findings here: hand them off to `TESTQ` (weak tests) or `TESTB` (missing coverage). Test registration in
`test_categories.yaml` is owned by `BUILD`: hand it off, do not file it. Record anything else for another lane under **Handoffs**; do not file it.
