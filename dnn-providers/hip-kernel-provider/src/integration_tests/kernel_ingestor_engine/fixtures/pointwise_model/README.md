# pointwise_model — a model-backed UHD

Every other pack in this tree declares `"adapter": "native"`: its heuristic is a compiled
function resolved by symbol. This one declares `"adapter": "tree_data"`, so its heuristic
is a trained artifact loaded at plan build (RFC 0019 §7). It exists so a model-backed UHD is
generated, packed and validated with the rest of the test descriptors.

## What is deliberate here

**It reuses the native pack's symbols.** The graph matcher, the ADD and dtype matchers and
the dispatch handler resolve to the same native symbols `pointwise/` uses. A model-backed
heuristic resolves no score symbol, so this pack adds no C++ to the provider — the only
difference from `pointwise/` is which heuristic its UED names.

**Its model disagrees with the native scorer, on purpose.** `hipkernel.pointwise.score`
returns `block_size`, so the native engine prefers the 256 kernel. This model prefers the
64 one. Two engines over one catalog, choosing opposite kernels, is what makes a test able
to tell them apart.

**The kernel ids are ordered against the model.** Both kernels sit at `priority: 0`, so the
declared-order fallback breaks the tie on descriptor id, and the 256 kernel's id is the
lower of the two. Declared order therefore picks 256 and the model picks 64 — so a model
that silently failed to load is visible, instead of landing on the same answer by luck.
`get_knobs_for_engine` reports the top-ranked kernel's `block_size` as the knob default,
which is how a test reads the outcome.

## Where it lives

Authored under `src/integration_tests/`, assembled into a build-tree stage directory (with
the generated artifacts below), and packed into the unit test descriptor root only, excluded
from `install`. Its kernel is `embedded_source`, so only `hip_kernel_provider_tests`, which
embeds `kernels/PointwiseModelAdd.cpp`, can serve it; the dlopened plugin the integration
binary loads carries no such source. A two-leaf model over one feature has no business in a
customer's plugin directory; it is scaffolding, not product.

## What this is not

**Not a worked example of a good feature set.** The signature is one field,
`$kernel.block_size`. The pointwise graph matcher binds tensor uids rather than shapes, so
there is no useful `$q.*` to split on here. A real pack wants problem features — that is
where a heuristic's value comes from, and a kernel-only model cannot express it.

**Not a trained model.** `tools/uhd_model_gen` writes a two-leaf tree by hand. A real one
comes from `projects/hipdnn/tools/uhd_gen` and a benchmark corpus.

## The artifacts

`pointwise_model.uhd.json` and `pointwise_model.bin` are generated into the staged
descriptor directory at build time, not committed. The model has to be: it is binary, so a
committed copy could not be read in a diff. The descriptor is text and could be committed,
but is generated with it because the two are tied by `features_hash` — a committed copy
would be free to drift the moment the feature signature changed. The generator computes
that hash with the same function the runtime validates it against, so the pair cannot
disagree.
