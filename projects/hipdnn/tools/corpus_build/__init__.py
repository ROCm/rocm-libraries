# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One SDPA graph corpus, assembled from the three sources that know different things.

`uhd_gen generate --graphs` needs a problem set, and until now there was no way to
produce one of a useful size: `make_sdpa_bundles` emits the geometries one kernel pack
happens to carry, `corpus_gen` samples an operation's declared space, and the shapes
real models run are written down in prose. Each answers a question the others cannot.

    kernel geometry   which problems a descriptor engine will admit at all, and which
                      of those have enough competing kernels for L2's ranking to have
                      something to choose between
    model shapes      the prefill and decode shapes a heuristic must get right,
                      because being wrong on those is being wrong at the job
    declaration sweep the rest of the space, so the model is not excellent on the
                      corners someone happened to compile and untested elsewhere

The output is engine-agnostic: RFC 0019 §4.1's UHD carries no engine, role or arch,
and an engine binds a heuristic by naming it from its own provider code -- so one
corpus trains the descriptor-backed engines and the ones with no UED alike, which is
what makes their L1 estimates comparable at A level.

Every graph is tagged with a regime (RFC 0019.13 §5.2) in its name and in the
manifest, because `uhd_gen evaluate` reports §11.2's per-regime table -- the PRIMARY
form of the regret report, and §11.4 MUST-3's check that an aggregate improvement is
not hiding a regression in the one regime the heuristic was built for -- as
UNAVAILABLE for want of a regime column on every corpus that exists today.

    python -m corpus_build --out ./corpus --count 1000
"""
