# hkp_pack -- descriptor packaging

Build-time UKD/KMD/KDP -> kpack packaging. Provider-internal (`tools/hkp_pack.py`);
see `python/hkp_pack/` for the pipeline itself and `examples/descriptors/` for a
real, minimal authored source root.

## Running the tests

```bash
cd dnn-providers/hip-kernel-provider
PYTHONPATH=descriptor-packaging/python:rocke/library:rocke/platform/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests -q
```

`rocm_kpack` (the third `PYTHONPATH` entry, or `--kpack-python-dir` /
`HIPKERNELPROVIDER_ROCM_KPACK_DIR`) is required by most of the suite -- it is the
kpack archive reader/writer every packing test round-trips through. Its absence is
diagnosed ONCE, clearly, by the `rocm_kpack_dir` fixture in `tests/conftest.py`: every
test needing it skips with one message naming the missing dependency, rather than each
failing separately deep inside `run_pipeline` with the same raw import error repeated
per test. Set `HIPKERNELPROVIDER_KPACK_REQUIRE_ROCM_KPACK=1` (mirrors the existing
`_REQUIRE_HIPCC`/`_REQUIRE_COMGR` pattern) to turn that skip into a hard failure, e.g.
in CI where a missing dependency must not go silent.

`-m quick` selects the load-time/pure-unit subset that needs neither `hipcc` nor
`rocm_kpack`/comgr -- useful on a box provisioned for neither.

### Desk-check a variant set (`hkp_pack.desk_check`, `tools/hkp_desk_check.py`)

```bash
python3 descriptor-packaging/tools/hkp_desk_check.py <path/to/some.kdp.json>
```

RUNBOOK.md step 5d's four invariants over a shipped variant set used to live only as a
shell-embedded Python snippet inside markdown -- untestable prose, and wrong on the
exact data it told an agent to point it at. Invariant 1 (metadata/spec drift) read
`kernel_source.spec`, which packing rewrites away (the authored spec moves to
`provenance.spec`; `kernel_source` becomes `{kind: kpack, library, toc_key, symbol,
sha256}`), so on a real packed tree the check silently printed "none" regardless of
real drift. Nothing ever ran the snippet, so nothing ever noticed.

`python/hkp_pack/desk_check.py` is the fix, shipped as real, importable, testable code
instead: it checks `kernel_source.spec` then falls back to `provenance.spec`, and
reports COULD-NOT-CHECK (a FAILING result, not a silent pass) when a kernel has neither.
It also handles pre-pack authored trees, where `toc_key`/`symbol` do not exist yet --
those invariants report NOT-APPLICABLE rather than a false "None == None" collision.
`tools/hkp_desk_check.py` is the thin CLI: exit 0 means every invariant this check can
enforce is clean (including "found nothing to check" counting as a failure, never a
pass); exit 1 means a real violation or a spec that could not be found anywhere.
Invariant 4 (symbol non-uniqueness) is informational only and never affects the exit
code on its own.

```bash
PYTHONPATH=descriptor-packaging/python:rocke/library:rocke/platform/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests/test_desk_check_invariants.py -q
```

Exercises the SHIPPED module directly (not a private copy of its logic -- a private
copy is exactly how invariant 1 went dead the first time) over
`tests/fixtures/desk_check/` (a small real rocKE `attention_dense` bundle: two
genuinely distinct variants, head_size 64 and 128). Each invariant carries both a
positive case (a clean real pack) and a negative case (a fixture engineered to violate
it, packed for real, proving the check catches the defect it exists for), plus a
`TestCliEndToEnd` class that runs the actual `tools/hkp_desk_check.py` subprocess
end-to-end -- against a clean real pack (exit 0), a real pack with an injected genuine
drift (exit 1), and the pre-pack authored tree (exit 0, NOT-APPLICABLE for
toc_key/symbol) -- so the CLI's argument parsing and exit-code mapping are covered, not
just the library functions underneath it.

### Real-corpus builder-signature guards (`tests/test_hkp_pack_rocke.py`)

`test_real_gfx942_attention_dense_is_accepted` and `test_real_gfx942_tiled_2d_is_accepted`
assert the real gfx942 `build_*` functions in `rocke/library/kernels/gfx942/` satisfy
`_require_spec_arch_signature`'s `(spec, *, arch)` contract. gfx942's
`build_attention_dense` used to be the corpus's one real REFUSAL case (a keyword-only
`tuning` parameter no descriptor could supply); PR #11237 folded that parameter into the
spec dataclass, so the corpus no longer contains a real unsuppliable-parameter builder.
The guard itself remains covered by a SYNTHETIC one in
`tests/test_hkp_pack_producer_guards.py::test_rejects_keyword_only_parameter` -- run
both files together to see full coverage of this guard:

```bash
PYTHONPATH=descriptor-packaging/python:rocke/library:rocke/platform/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests/test_hkp_pack_rocke.py \
        descriptor-packaging/tests/test_hkp_pack_producer_guards.py -q
```
