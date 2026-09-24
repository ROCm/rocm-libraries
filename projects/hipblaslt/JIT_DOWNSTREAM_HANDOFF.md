# hipBLASLt JIT downstream handoff

Development continues on `downstream/hipblaslt-jit` from the complete upstream
review-stack tip `d742375dbbef5ef44e9890e199741de70d573f14`. The full JIT
implementation and existing names are preserved: the explicit TensileLite API,
its sample, the optional generic API and separate sample, provider prediction,
benchmark integration, validation driver and workflow.

This consolidation adds contributor handoff/design documents only. Every
freshly fetched PR head and every scoped local branch head is an ancestor of
the new branch. Original branches are retained as historical safety refs.
Closing the upstream reviews is an administrative transition, not a merge or
an approval of their design. Review discussions remain on GitHub.

## Preserved upstream heads

| PR | Source head |
| --- | --- |
| [#12459](https://github.com/ROCm/rocm-libraries/pull/12459) | `58b4799d28c7a16649344aa6bb368985bf2625f4` |
| [#12460](https://github.com/ROCm/rocm-libraries/pull/12460) | `4984c944b4412f4641179cb7cdb6ab927f8cafa3` |
| [#12552](https://github.com/ROCm/rocm-libraries/pull/12552) | `4f89f196932adf6cd7683f4d5494b115092ad152` |
| [#12563](https://github.com/ROCm/rocm-libraries/pull/12563) | `0010f19acafb91bd175d6afc9f3ec77fa0c9b8bb` |
| [#12564](https://github.com/ROCm/rocm-libraries/pull/12564) | `6a1ebb5765051ad9e44d7bbe95e67247f34c00b2` |
| [#12565](https://github.com/ROCm/rocm-libraries/pull/12565) | `b25107056a334745c699fdac0fa4e94f1e1b30ff` |
| [#12461](https://github.com/ROCm/rocm-libraries/pull/12461) | `5d9826f41640dd3a79f26d810775bb802afc60da` |
| [#12462](https://github.com/ROCm/rocm-libraries/pull/12462) | `e89931fd167758ddaf6554e786dcabdaa7499de3` |
| [#12463](https://github.com/ROCm/rocm-libraries/pull/12463) | `7a307c190abf258d403711fbbcead647d11a76cc` |
| [#12430](https://github.com/ROCm/rocm-libraries/pull/12430) | `d742375dbbef5ef44e9890e199741de70d573f14` |

Native stack #12567 records the historical order. Its PR heads were read and
fetched again before this branch was created on September 24, 2026.

## Local environment and publication

Use the existing `.venv` at the repository root and the configured
`projects/hipblaslt/build/release` directory. The final validated build has
`HIPBLASLT_ENABLE_JIT=ON`; no installation or rebuild is needed for this
branch/document-only consolidation. The original untracked `.venv/` and
`SESSION_HANDOFF.md` remain untouched and uncommitted.

Only `origin`, pointing to `git@github.com:ROCm/rocm-libraries.git`, was configured
at consolidation. No personal/downstream remote was configured. This branch
has not been pushed upstream and no replacement PR was created. A future
publication requires an explicitly selected downstream repository.

An incremental Git bundle is retained locally under
`build/downstream-backup/hipblaslt-jit-20260924.bundle` relative to this project.
It contains the downstream branch and its commits beyond the common upstream
base; `git bundle verify` identifies the prerequisite commit. This supplements
the preserved local branch refs. It is a generated backup, not tracked source.

## Reusable design material

The [design-note index](jit-design/README.md) preserves the Confluence discussion
draft, host/Python timing and progress plans, and producer-first KFA discovery
and convergence plan. The current [roadmap](JIT_ROADMAP.md) keeps implemented
behavior separate from those planned features. No external page was published.

The `HIPBLASLT_JIT_DEBUG` plan retains independent `timing` and `progress`
categories, including `timing,progress`; unset/empty adds no new collection,
observer or files. Existing logs, errors and benchmark timing boundaries remain
unchanged until that separately planned implementation is undertaken.

KFA convergence starts with complete metadata emitted by TensileLite. Shared
selection/execution follows evidence of identical packed arguments, launch
geometry, predicates, workspace, helper order and synchronization. Broader
planning/cache protocols and modeled-input coverage remain future work.

## Reusable validation evidence

Before consolidation, native gfx950 validation passed the ten basic driver
routes and twelve integrated generic routes, including C/C++ numerical output,
algorithm ownership, malformed artifacts, helper preflight and disabled APIs.
Both separate samples produced zero maximum error. The final prediction layer
passed sixteen targeted benchmark checks and six retained API/sample/disabled
routes, and the enabled build was restored.

The prediction count includes three numerical C/mixed/C++ cases, two expected
prediction/recipe failures and eleven negative checks. gfx1250 SIA4 generation
and compilation passed separately; this is not native gfx1250 execution.
Configured CI architectures are not evidence of completed runs. These results
remain applicable because consolidation preserves executable, build, fixture
and test source exactly; no new tests were run for the added documents.

The upstream reviews include unresolved scope/naming objections to the builder
and process integration. A factual comparison with `Tensile --build-only` is
recorded in the discussion draft. Continuing downstream does not change those
reviews into acceptance.
