"""Gate a set of variant sets on the five properties a comparison depends on.

Run against generated bundles or installed trees. Exits non-zero on any failure, so
it can sit in front of a build or a sweep.

WHY FIVE AND NOT ONE. Each of these failed separately, and each was invisible to the
check that caught the others:

  1. BINARY NESTING. "Do more variants help?" is only answerable if the larger set can
     still choose everything the smaller one could. That is about compiled binaries.
     Checking metadata instead is what hid the defect for several rounds: normalising
     labels made the sets look nested while the binaries diverged. 43 shapes shipped
     where the larger set could not reproduce the smaller one's kernel.

  2. LOADER-TUPLE UNIQUENESS. The loader substitutes a KMD field's `default_value` for
     an absent key, then requires the resulting tuple to be unique per device. A
     duplicate is not a dropped entry -- it rejects THE WHOLE ENGINE, and the arm then
     serves every graph from another engine while exiting 0 and passing a
     descriptor-count check. That shipped once and cost a full sweep.

  3. NO SENTINEL IN A DESCRIPTOR. `-1` means "unresolved". Every compiled artifact has
     a definite setting, so a descriptor claiming otherwise describes nothing that
     exists, and downstream it aliases onto the KMD default and triggers (2).

  4. METADATA MATCHES ITS BINARY. The matcher selects on metadata; the spec decides
     what was built. When they disagree the runtime picks a kernel on false pretences.
     364 descriptors advertised "policy decides" while pinning an override.

  5. VOCABULARY. Metadata carries the hipDNN spelling the matcher compares ("BF16");
     the spec carries the builder's ("bf16"). A descriptor written in the builder's
     vocabulary loads cleanly, reconciles on every count, and matches NOTHING.

A count check answers none of these. Counts are about disk.

HOW THE SCHEMA IS REACHED. By reference, never by filename. A bundle's KMD is found
by walking the id chain the documents themselves declare -- `KDP.engine` names a UED,
`UED.metadata` names a KMD -- across every descriptor under the selected root. A
dangling hop fails naming the hop and the id; two documents claiming one id fail
naming both files. Binding a KDP to a same-stem sibling instead is how a bundle gets
gated against a schema nothing wires it to: the check passes, and the engine the
loader actually assembles was never examined.

TWO MODES, and `--mode` is required because they make different claims.

  --mode full asks whether the shipped binary agrees with the metadata that selects
  it. The answer comes from the producing compiler's own evidence, carried by the
  descriptor: `provenance.specialization_contract` declares which metadata fields a
  compiled kernel specialises on and how each is read off the builder's spec --
  carried by the kernel, or once by the KDP it is inline in -- and
  `provenance.effective_spec` records what the compiler observed when it built the
  bytes this descriptor names. The gate rebuilds the consumer records from the
  descriptors in front of it and hands them, with the named payload bytes, to
  `hkp_pack.agreement.verify`. Nothing imports a producer: a packed artifact is
  checkable on a machine that has never had rocKE installed. A missing, unsupported
  or mismatched record is a FAILURE -- an artifact that cannot say what it was built
  from has not established agreement.

  A kernel declaring `metadata_fields: []` -- the mandatory declaration for a
  non-compiled source, which is every AOT hip bundle -- has no producing-build
  record to bind, so it is reported as NOT VERIFIED HERE. That is a third outcome,
  neither a failure nor part of the pass: `hkp_pack.desk_check` states the same
  thing about the same artifact, and two readers of one tree must not disagree.

  Because full mode claims every property, a check it could not RUN is a gap in
  that claim, and the run FAILS with exit 1 naming the unrun checks. Passing on a
  narrowed full run would put the strong claim behind an exit code that did not
  earn it, and leave the caveat to whatever reads the output.

  --mode structural runs only what needs no compiled evidence: binary nesting on the
  identities it can compute, loader-tuple uniqueness, sentinel absence, plain
  spec-vs-metadata agreement, and vocabulary. It reports compiled specialization
  agreement as NOT CHECKED, by name, in the output and in the exit summary, and its
  passing line says so. A structural pass can never be read as compiled agreement:
  a gate that quietly stops checking is the failure mode this whole file exists to
  prevent.

A PROFILE beside the config carries the two facts the artifacts do not: which bundle
to gate when a tree hosts more than one engine, and the matcher's vocabulary where a
declaration does not already state it. Everything else is read off the descriptors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: KMD value meaning "unresolved -- the kernel's own policy decides". Never legal in a
#: shipped descriptor; see property 3.
SENTINEL = -1

#: The two claims this gate can make. Selected explicitly on every run: a default
#: would let the weaker run print under the stronger one's name, or the stronger
#: one fail a tree that was only ever meant to be checked structurally.
MODES = ("full", "structural")


def _agreement_python_root() -> Path:
    """The descriptor-packaging python directory, anchored on this file's location.

    Located by ascending to the ancestor that actually contains the subtree rather
    than by counting path components, so the tool answers the same run from the repo
    root, from an installed checkout, or from three directories down. The gate cannot
    reimplement declaration or evidence semantics -- there is one implementation of
    those and it lives there -- so failing to find it is fatal rather than a
    degraded run.
    """
    relative = Path("dnn-providers/hip-kernel-provider/descriptor-packaging/python")
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / relative).is_dir():
            return candidate / relative
    raise SystemExit(
        f"FAIL: cannot locate {relative} above {here}. The specialization "
        f"declaration and evidence semantics live there and are not reimplemented "
        f"here."
    )


sys.path.insert(0, str(_agreement_python_root()))

from hkp_pack import agreement, descriptor_context  # noqa: E402
from hkp_pack.errors import HkpPackError  # noqa: E402


class GateError(RuntimeError):
    """An input the gate cannot proceed from: an unresolvable reference, an
    ambiguous tree, a payload it was told to read and could not. Never a finding
    about a descriptor's content -- those are reported as failures and counted."""


class Profile:
    """Per-kernel facts the descriptors do not carry.

    A profile is a small JSON/YAML document beside the generator config::

        bundle: gfx942_attention_dense
        vocabulary:
          dtype: [BF16, FP16]

    ``bundle`` pins which engine to gate when a tree hosts more than one; without it
    a multi-engine tree is refused rather than guessed at, since gating the wrong
    engine would pass while the one under test is broken.

    ``vocabulary`` declares the matcher's legal spellings for a field, as either a
    builder-to-matcher mapping or the legal set. A UKD's
    ``specialization_contract`` -- its own, or the one it inherits from its KDP --
    states the same thing for the fields it specialises on and is merged in
    automatically, so a profile only has to speak for the fields no declaration
    covers.

    Nothing in a profile names a policy function, a module or a provider root. What
    a binary was built with is answered by the producing compiler's evidence under
    ``--mode full``, not by importing the kernel on the machine running the gate.
    """

    def __init__(self, raw: dict, path: str | None = None):
        self.path = path
        self.bundle = raw.get("bundle")
        self.vocabulary = dict(raw.get("vocabulary") or {})
        # ABSENT and EXPLICITLY EMPTY are different claims, and collapsing them is
        # the same mistake this gate exists to catch one layer down. (4a) below must
        # tell "no vocabulary block exists, so an undeclared string field is
        # genuinely ambiguous" from "a vocabulary block exists and simply never
        # mentions this field", which is not ambiguous -- the author had the exact
        # place to declare a translation and did not.
        self.vocabulary_declared = "vocabulary" in raw

    @classmethod
    def load(cls, path: str) -> "Profile":
        with open(path) as fh:
            text = fh.read()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError:  # pragma: no cover - environment-dependent
                raise SystemExit(
                    f"FAIL: {path} is not JSON and PyYAML is not installed to read it "
                    f"as YAML."
                )
            raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise SystemExit(f"FAIL: profile {path} must be a mapping.")
        return cls(raw, path)

    @classmethod
    def empty(cls) -> "Profile":
        return cls({})


def select(
    bundles: list[descriptor_context.Bundle], profile: Profile
) -> list[descriptor_context.Bundle]:
    """The bundles of the ONE engine this run gates.

    Several KDPs may declare one engine -- that is the shape the engine-wide desk
    check exists for -- but two engines under one root is a question only the author
    can answer, and answering it by guessing would gate the wrong engine while the
    one under test is broken.
    """
    if not bundles:
        raise GateError("no *.kdp.json under this root -- is this a descriptor tree?")
    if profile.bundle:
        wanted = f"{profile.bundle}.kdp.json"
        bundles = [b for b in bundles if os.path.basename(b.kdp_path) == wanted]
        if not bundles:
            raise GateError(f"no {wanted} under this root")
    engines = sorted({b.engine["id"] for b in bundles})
    if len(engines) > 1:
        names = ", ".join(
            sorted(os.path.basename(b.kdp_path)[: -len(".kdp.json")] for b in bundles)
        )
        raise GateError(
            f"{len(engines)} engines under this root ({names}). Set 'bundle' in the "
            f"profile to say which one this gate is about -- checking the wrong "
            f"engine would pass while the one under test is broken."
        )
    return bundles


def _binary_key(descriptor: dict) -> str:
    """Identity of the compiled artifact this descriptor names.

    Two dialects, strongest first, because a bundle is checkable at two points in
    its life:

      * PACKED. The spec is compiled away and the descriptor names a symbol in the
        archive. The sha256 IS the binary, a stronger identity than any
        reconstruction, so prefer it when present.
      * AUTHORED. Pre-build there is only the builder and the spec AS AUTHORED. An
        omitted key stays omitted: absence is the author's intent that the kernel
        decide, and rewriting it to a default or to false would make two genuinely
        different requests look like one.
    """
    source = descriptor["kernel_source"]
    if source.get("sha256") or source.get("symbol"):
        return json.dumps(
            {"sha256": source.get("sha256"), "symbol": source.get("symbol")},
            sort_keys=True,
        )
    spec = dict(source.get("spec") or {})
    return json.dumps(
        {
            "builder": source.get("builder"),
            "spec": sorted((k, repr(v)) for k, v in spec.items()),
        },
        sort_keys=True,
    )


def _shape_key(descriptor: dict, knob: str) -> str:
    """Identity of a descriptor's shape with `knob` erased.

    Two descriptors sharing this key describe the SAME shape at different
    settings of `knob` -- candidates to be a "specialization twin" pair. Built from
    metadata, not spec, because metadata is what a bigger set's author reads
    when deciding "do I already carry this shape".
    """
    metadata = descriptor["metadata"]
    return json.dumps(
        sorted((k, repr(v)) for k, v in metadata.items() if k != knob),
        sort_keys=True,
    )


def _specialization_twins(order: list, by_label: dict, knobs: set) -> list:
    """Shapes where a bigger set overrides a kernel-decided knob instead of adding to it.

    `knobs` are the fields the UKDs' own declarations mark as compiled-specialization
    fields: the ones a descriptor may legitimately leave out of its spec because the
    kernel settles them at build time. Nothing else can be left out that way, so
    nothing else can have a twin.

    Binary nesting (property 1) already reports the symptom here -- "N binaries
    absent from BIG" -- and that is correct, but it names no knob and implies no
    fix. The fix that actually applies is never "resolve a conflict": the
    smaller set's descriptor left `knob` for the kernel to settle at build time,
    so the bigger set must carry THAT variant unchanged, with the override
    sitting BESIDE it as a second binary -- not in its place. A reader told only
    "a binary is missing" may reach for the wrong fix and repin the override
    instead of restoring the twin, which is the same class of mistake that
    shipped 364 mislabelled descriptors under property 4.

    A pinned value that happens to equal what the kernel would have settled on is
    not a violation -- it is the SAME binary under another spelling, which is
    exactly what `_binary_key()` already normalises, so that equality is
    delegated to it rather than re-decided here from raw spec values.
    """
    violations = []
    for small_label, big_label in zip(order, order[1:]):
        small_descs = by_label[small_label]
        big_descs = by_label[big_label]
        for knob in sorted(knobs):
            big_by_shape: dict = {}
            for c in big_descs:
                big_by_shape.setdefault(_shape_key(c, knob), []).append(c)
            for d in small_descs:
                spec = d["kernel_source"].get("spec") or {}
                if not spec or spec.get(knob) is not None:
                    continue  # nothing to settle, or this descriptor pins it itself
                candidates = big_by_shape.get(_shape_key(d, knob))
                if not candidates:
                    continue  # no shape match at all: a plain nesting gap, not a twin
                candidate_keys = {_binary_key(c) for c in candidates}
                if _binary_key(d) in candidate_keys:
                    continue  # the twin is present, or the override resolves the same
                if any(
                    (c["kernel_source"].get("spec") or {}).get(knob) is None
                    for c in candidates
                ):
                    continue  # a differently-settled twin already covers it
                shape = {k: v for k, v in d["metadata"].items() if k != knob}
                violations.append(
                    f"specialization twin missing: {big_label} carries only a pinned "
                    f"'{knob}' at shape {shape}, none left to the kernel like "
                    f"{small_label}'s {d['name']} -- carry BOTH variants, the "
                    f"override alone drops {small_label}'s kernel from the "
                    f"candidate list"
                )
    return violations


def effective_arch(
    bundles: list[descriptor_context.Bundle], requested: str | None
) -> str:
    """The single architecture a full-mode run is about.

    The producing compiler wrote its evidence for one arch, so the reader has to be
    checking the same one. A shard that pins exactly one arch answers this itself;
    anything wider is a question, and `--arch` is how the caller answers it rather
    than the gate picking the first entry of a list.
    """
    if requested:
        return requested
    covered = {tuple(entry.arch) for b in bundles for entry in b.entries}
    single = {c[0] for c in covered if len(c) == 1}
    if len(covered) == 1 and len(single) == 1:
        return single.pop()
    raise GateError(
        "this tree does not pin exactly one architecture, so the evidence written "
        "for one arch cannot be matched against it. Pass --arch to say which shard "
        "is being checked."
    )


class Payloads:
    """The named bytes a packed descriptor points at, read once per archive.

    A descriptor that names payload bytes and cannot produce them has not shown that
    its evidence is about the artifact it ships, so an unreadable payload is a
    failure rather than a property left unchecked.
    """

    def __init__(self, kpack_python_dir: str | None = None):
        self._dir = kpack_python_dir
        self._archives: dict = {}
        self._module = None

    def _archive(self, path: Path):
        key = str(path)
        if key not in self._archives:
            if self._module is None:
                from hkp_pack.kpack_resolver import load_kpack  # noqa: PLC0415

                self._module, _compression = load_kpack(self._dir)
            self._archives[key] = self._module.PackedKernelArchive.read(path)
        return self._archives[key]

    def read(self, entry: descriptor_context.Entry, arch: str) -> bytes:
        source = entry.ukd["kernel_source"]
        kind = source.get("kind")
        if kind != "kpack":
            raise GateError(
                f"{entry.ukd.get('name')}: --mode full needs the packed dialect, and "
                f"kernel_source.kind is {kind!r}. The producing compiler's evidence "
                f"exists only once the bytes do; check the packed tree."
            )
        library, toc_key = source.get("library"), source.get("toc_key")
        if not library or not toc_key:
            raise GateError(
                f"{entry.ukd.get('name')}: packed kernel_source names no "
                f"library/toc_key, so there are no payload bytes to bind the "
                f"evidence to."
            )
        archive_path = (Path(entry.origin_dir) / library).resolve()
        if not archive_path.is_file():
            raise GateError(
                f"{entry.ukd.get('name')}: kernel_source.library resolves to "
                f"{archive_path}, which does not exist."
            )
        try:
            blob = self._archive(archive_path).get_kernel(toc_key, arch)
        except HkpPackError:
            raise
        except Exception as exc:
            raise GateError(
                f"{entry.ukd.get('name')}: cannot read {archive_path}: {exc}"
            ) from exc
        if blob is None:
            raise GateError(
                f"{entry.ukd.get('name')}: {archive_path} carries no member "
                f"{toc_key!r} for {arch}."
            )
        return blob


def _comparable(value):
    """One representation for values the two layers spell differently.

    A spec carries Python `True`; metadata carries `1`. Normalising here rather
    than special-casing bool keeps the check from reporting a mislabelling that is
    only a spelling difference.
    """
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def check(
    label: str,
    root: str,
    profile: Profile,
    mode: str,
    arch: str | None = None,
    payloads: Payloads | None = None,
):
    """Run every property this mode can honestly claim, and name the rest.

    Returns `(binaries, descriptors, failures, unchecked, unverified, knobs)`, where
    `knobs` are the compiled-specialization fields the declarations name -- the set
    the twin check keys on.

    `unchecked` and `unverified` are DIFFERENT outcomes and must not be merged.
    `unchecked` is a check this run could not run, which under `--mode full` is a
    gap in the claim and fails the gate. `unverified` is a check that ran and
    reached a stated result: the kernel declares no specialized metadata field, so
    there is no producing-build record to bind. That is a legitimate outcome, so it
    neither fails the gate nor joins the pass line's list of things established.
    """
    index = descriptor_context.Index(root)
    schemas = index.schemas()
    all_bundles = descriptor_context.resolve_bundles(index)
    bundles = select(all_bundles, profile)
    kmd = bundles[0].kmd
    engine_id = bundles[0].engine["id"]
    descriptors = [entry.ukd for b in bundles for entry in b.entries]
    failures: list[str] = []
    unchecked: list[str] = []
    # Kernels reported as NOT VERIFIED HERE: the check ran and there was nothing
    # for it to bind, which is neither a failure nor a property established.
    unverified: list[str] = []

    declared = descriptor_context.declarations(bundles, schemas)
    knobs = {f for d in declared.values() for f in d["metadata_fields"]}
    vocabulary = dict(profile.vocabulary)
    for declaration in declared.values():
        vocabulary.update(declaration["vocabulary"])
    # A declaration's vocabulary speaks only for the fields that declaration
    # specialises on, so it is merged into the translations above but does NOT
    # answer "was there a place to declare a translation for THIS field". Only the
    # profile's block does that, and treating a contract as a blanket answer would
    # start comparing every matcher-only string field raw against a spec that spells
    # it in the builder's vocabulary on purpose.
    vocabulary_declared = profile.vocabulary_declared

    # (2) Loader-tuple uniqueness, engine-wide and arch-aware, with each value
    # canonicalised per its declared KMD type. Engine-wide because the loader
    # assembles ONE catalog per engine per device: two KDPs of one engine that each
    # look unique alone still collide there. Arch-aware because two candidates whose
    # coverage is disjoint never meet in that catalog -- a gfx942 pack and a gfx950
    # pack may legitimately carry the same tuple -- while a wildcard overlaps
    # everything. Canonicalised because `1` and `1.0` on a FLOAT field are one
    # catalog entry and a BOOL `true` is not an INT `1`; comparing raw values would
    # miss the first collision and invent the second.
    completed: list = []
    for bundle in bundles:
        for entry in bundle.entries:
            try:
                values = agreement.complete_metadata(entry.ukd["metadata"], kmd)
            except HkpPackError as exc:
                failures.append(
                    f"{entry.ukd.get('name')}: metadata does not complete against "
                    f"the engine's KMD: {exc}"
                )
                continue
            completed.append((entry, agreement.digest(values)))
    collisions = []
    for i, (left, left_key) in enumerate(completed):
        for right, right_key in completed[i + 1 :]:
            if left_key != right_key or not agreement.overlap(left.arch, right.arch):
                continue
            # A wildcard covers whatever the other side names, so the overlap it
            # reports is that side's list rather than an empty intersection.
            both = set(left.arch) & set(right.arch)
            either = set(left.arch) | set(right.arch)
            where = (
                ", ".join(sorted(both if left.arch and right.arch else either))
                or "every arch"
            )
            collisions.append(
                f"{left.ukd.get('name')} and {right.ukd.get('name')} complete to one "
                f"tuple on {where}"
            )
    if collisions:
        failures.append(
            f"{len(collisions)} loader-tuple collisions (engine would be dropped), "
            f"e.g. {collisions[0]}"
        )

    # (3) No sentinel anywhere in shipped metadata.
    sentinels = [
        k["name"]
        for k in descriptors
        if any(v == SENTINEL for v in k["metadata"].values())
    ]
    if sentinels:
        failures.append(
            f"{len(sentinels)} descriptors ship the unset sentinel, e.g. {sentinels[0]}"
        )

    # (5) Vocabulary, per declared field. Without a declaration there is nothing to
    # compare against -- the right spelling is a matcher fact, not a derivable one.
    if vocabulary:
        for field, declared_spellings in vocabulary.items():
            # A mapping declares builder-spelling -> matcher-spelling, which the
            # parity generator uses to translate; the legal set is its VALUES. A
            # bare list declares the legal set directly. Both spellings are
            # accepted so one profile serves both tools without restating itself.
            allowed = (
                declared_spellings.values()
                if isinstance(declared_spellings, dict)
                else declared_spellings
            )
            allowed_set = {str(a) for a in allowed}
            wrong = sorted(
                {
                    str(k["metadata"].get(field))
                    for k in descriptors
                    if field in k["metadata"]
                    and str(k["metadata"][field]) not in allowed_set
                }
            )
            if wrong:
                failures.append(
                    f"{field} written in the wrong vocabulary: {wrong} "
                    f"(the matcher compares {sorted(allowed_set)})"
                )
    else:
        unchecked.append("vocabulary (no 'vocabulary' in profile or declaration)")

    # (4) Metadata matches the binary it names.
    #
    # TWO KINDS OF FIELD, and only one of them needs the compiler's evidence.
    #
    # (4a) PLAIN FIELDS: a metadata key that is ALSO a spec key. The spec is what the
    # builder compiles and the metadata is what the matcher compares, so if a
    # descriptor carries both they must agree -- no evidence, no profile, no kernel
    # knowledge required. This is a property of the descriptor against ITSELF, and it
    # runs in both modes.
    #
    # A review demonstrated what its absence costs, by mutation: flipping a
    # descriptor's metadata.ragged to 0 while its spec still said True passed the
    # gate 84/84 OK. That is the "dangerous direction" the shipping commit names --
    # an aligned-labelled descriptor whose binary is boundary-padded -- caught
    # downstream by the C++ matcher unit tests but invisible at the STATIC rung,
    # which coverage_gate.py's own docstring insists is a separate rung precisely
    # because each catches what the other cannot. The same hole swallowed a head_size
    # mismatch and, once they existed, varlen/paged both ways.
    #
    # ANY field with a declared vocabulary is exempt from the plain comparison, not
    # just the dict-form ones. A vocabulary declaration means the two layers spell the
    # value DIFFERENTLY ON PURPOSE -- builder "bf16", matcher "BF16" -- so comparing
    # them raw reports a mislabelling that is only a translation. Property (5) checks
    # the metadata side against the legal set, which is the check that applies to a
    # translated field.
    translated_fields = set(vocabulary)
    vocabulary_maps = {
        field: mapping
        for field, mapping in vocabulary.items()
        if isinstance(mapping, dict)
    }

    plain_mismatches = []
    undeclared_string_fields = set()
    for descriptor in descriptors:
        spec = descriptor["kernel_source"].get("spec") or {}
        if not spec:
            continue
        for field, meta_value in descriptor["metadata"].items():
            if field not in spec or field in knobs:
                continue
            spec_value = spec[field]
            if field in translated_fields:
                # Translated on purpose. Where the mapping is given, apply it and
                # still compare -- that catches a descriptor whose metadata names a
                # different dtype than its spec builds. Where only the legal set is
                # declared, there is nothing to translate WITH, so property (5) owns
                # the field entirely and this comparison must not guess.
                if field not in vocabulary_maps:
                    continue
                if isinstance(spec_value, str):
                    spec_value = vocabulary_maps[field].get(spec_value, spec_value)
            elif isinstance(spec_value, str) or isinstance(meta_value, str):
                # An UNDECLARED string field: no vocabulary entry names it. Whether
                # that means "same spelling both layers" or "translated, just not
                # written down" depends on whether a vocabulary was declared AT ALL.
                #
                # Nothing declared anywhere: there is nowhere an author COULD have
                # said "translated", so an undeclared string is genuinely ambiguous
                # -- the common, correct `spec: "bf16"` / `metadata: "BF16"` pairing
                # must not become a false failure. Recorded rather than silently
                # dropped: a field nobody can judge is a fact the author should see,
                # not a check that quietly asked nothing.
                #
                # A vocabulary exists and just never mentions this field: the author
                # had the exact place to say "this field is translated" and did not,
                # so there is nothing left to guess -- compare it raw like any plain
                # field. This is the branch that closes the escape a review found on
                # the real gfx950 tree: `persist_decode` written only in
                # kernel_source.spec, with a vocabulary that declares `dtype` and
                # says nothing about `persist_decode`, let 84 descriptors carry a
                # contradicting `metadata.persist_decode` and still pass.
                if not vocabulary_declared:
                    undeclared_string_fields.add(field)
                    continue
            if _comparable(spec_value) != _comparable(meta_value):
                plain_mismatches.append(
                    f"{descriptor['name']}: {field} spec={spec_value!r} "
                    f"metadata={meta_value!r}"
                )
    if plain_mismatches:
        failures.append(
            f"{len(plain_mismatches)} descriptor field(s) whose metadata contradicts "
            f"the spec their binary is built from, e.g. {plain_mismatches[0]}"
        )
    if undeclared_string_fields:
        # A field nobody can judge is a liability the author should SEE, not a check
        # that quietly asked nothing -- named rather than folded into an unqualified
        # pass. Only reachable with no vocabulary declared anywhere: once one exists,
        # an unmentioned field is compared raw above instead of landing here.
        unchecked.append(
            "metadata-matches-binary for UNDECLARED STRING field(s) "
            f"{', '.join(sorted(undeclared_string_fields))} (no 'vocabulary' in "
            f"profile or declaration to judge them against)"
        )

    # (4b) COMPILED SPECIALIZATION: the fields a declaration marks as settled by the
    # compiler. A descriptor may leave one out of its spec entirely, so nothing in
    # the descriptor says what the binary was built with -- only the producing
    # compile does, through the evidence it wrote onto the descriptor. Full mode
    # checks that evidence against the descriptors and the payload bytes in hand;
    # structural mode cannot, and says so by name.
    if mode == "full":
        records = descriptor_context.consumer_records(all_bundles, schemas, arch)
        for bundle in bundles:
            for entry in bundle.entries:
                name = entry.ukd.get("name")
                enclosing = bundle.kdp_doc if entry.inline else None
                if agreement.resolved_contract(entry.ukd, enclosing) is None:
                    failures.append(
                        f"{name}: no specialization declaration for engine "
                        f"{engine_id}, so this descriptor never states what its "
                        f"binary specialises on and agreement cannot be established"
                    )
                    continue
                agreement.select_declaration(
                    entry.ukd, bundle.engine, bundle.kmd, schemas, enclosing
                )
                kind = entry.ukd.get("kernel_source", {}).get("kind")
                if kind != "kpack":
                    failures.append(
                        f"{name}: --mode full needs the packed dialect, and "
                        f"kernel_source.kind is {kind!r}. Check the packed tree."
                    )
                    continue
                if entry.ukd["id"] not in records:
                    failures.append(
                        f"{name}: no consumer records for requested architecture "
                        f"{arch}, so compiled specialization agreement cannot be "
                        f"established"
                    )
                    continue
                bound = records[entry.ukd["id"]]
                # A declaration with no `metadata_fields` states that the compiler
                # specialized on nothing, which is the legitimate and MANDATORY
                # declaration for a non-compiled source. There is no
                # producing-build record for it to bind, so it is reported rather
                # than failed -- and rather than absorbed into the pass, which
                # would put "the record binds these bytes" behind a kernel for
                # which no record was ever read. `hkp_pack.desk_check` says the
                # same thing about the same artifact; the two readers of one tree
                # have to agree.
                #
                # A descriptor that CARRIES `provenance.effective_spec` is checked
                # whatever its declaration claims. The evidence is in the file, so
                # the check can run -- and skipping it would make relabelling the
                # specialized field as matcher-only a waiver that turns a stale or
                # forged record into a pass. Nothing in this tool reports "no
                # record to bind" about a descriptor holding one.
                claimed = any(r["declaration"]["metadata_fields"] for r in bound)
                if not claimed and "effective_spec" not in (
                    entry.ukd.get("provenance") or {}
                ):
                    unverified.append(
                        f"{name}: declares no specialized metadata_fields, so there "
                        f"is no producing-build record to bind and nothing here was "
                        f"verified against a binary"
                    )
                    continue
                try:
                    payload = payloads.read(entry, arch)
                    agreement.verify(entry.ukd, bound, payload)
                except GateError as exc:
                    # Already names the descriptor it is about; the payload reader
                    # is reached from other callers too and says so itself.
                    failures.append(str(exc))
                    continue
                except HkpPackError as exc:
                    failures.append(f"{name}: {exc}")
                    continue
    else:
        unchecked.append(
            "COMPILED SPECIALIZATION AGREEMENT (--mode structural reads no "
            "producing-build evidence)"
        )

    binaries = {_binary_key(k) for k in descriptors}
    verdict = "OK" if not failures else "FAIL"
    print(
        f"  {label}: descriptors={len(descriptors):5d} "
        f"distinct-binaries={len(binaries):5d} {verdict}"
    )
    for f in failures:
        print(f"      ! {f}")
    for u in unchecked:
        print(f"      ? NOT CHECKED: {u}")
    for u in unverified:
        print(f"      ? NOT VERIFIED HERE: {u}")
    return binaries, descriptors, failures, unchecked, unverified, knobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate variant sets on nesting, tuple uniqueness, sentinels, "
        "metadata/binary agreement and vocabulary.",
    )
    parser.add_argument(
        "pairs",
        nargs="+",
        metavar="LABEL ROOT",
        help="Label and descriptor root, repeated. Nesting is checked in the order "
        "given: each set must be a binary subset of the next.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="full: check the producing compiler's evidence "
        "(provenance.specialization_contract and provenance.effective_spec) against "
        "these descriptors, this schema, this architecture and the named payload "
        "bytes; a missing or mismatched record FAILS. structural: run only the "
        "checks that need no compiled evidence and report compiled specialization "
        "agreement as NOT CHECKED. There is no default: the two make different "
        "claims.",
    )
    parser.add_argument(
        "--arch",
        help="The architecture whose shard is being checked, for --mode full. "
        "Required when the tree does not pin exactly one.",
    )
    parser.add_argument(
        "--profile",
        help="Per-kernel profile (JSON or YAML) declaring the bundle name to gate "
        "and the matcher's vocabulary for fields no descriptor declaration covers.",
    )
    parser.add_argument(
        "--kpack-python-dir",
        help="Directory holding the rocm_kpack package, for reading the payload "
        "bytes a packed descriptor names under --mode full. Omit it to use the "
        "installed one.",
    )
    args = parser.parse_args(argv)

    if len(args.pairs) % 2:
        parser.error("arguments must be LABEL ROOT pairs")
    roots = list(zip(args.pairs[::2], args.pairs[1::2]))

    profile = Profile.load(args.profile) if args.profile else Profile.empty()

    print("variant-set gate")
    print(f"  mode: {args.mode}")

    sets, by_label, bad, skipped, knobs = {}, {}, [], [], set()
    unverified: list[str] = []
    payloads = Payloads(args.kpack_python_dir) if args.mode == "full" else None
    try:
        arch = None
        if args.mode == "full":
            probe = descriptor_context.resolve_bundles(
                descriptor_context.Index(roots[0][1])
            )
            arch = effective_arch(probe, args.arch)
        for label, root in roots:
            binaries, descriptors, failures, unchecked, unbound, declared = check(
                label, root, profile, args.mode, arch, payloads
            )
            sets[label] = binaries
            by_label[label] = descriptors
            bad += [(label, f) for f in failures]
            skipped += unchecked
            unverified += [f"{label}: {u}" for u in unbound]
            knobs |= declared
    except (GateError, HkpPackError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    # (1) Binary nesting, pairwise along the given order.
    order = [lbl for lbl, _ in roots]
    for small, big in zip(order, order[1:]):
        missing = sets[small] - sets[big]
        ok = not missing
        print(
            f"  {small} binaries subset of {big}: {ok}"
            + ("" if ok else f"  MISSING {len(missing)}")
        )
        if not ok:
            bad.append((small, f"{len(missing)} binaries absent from {big}"))

    # (1b) Specialization twins: a special case of (1) worth naming on its own,
    # because the fix it needs ("carry both variants") is not the fix "binaries do
    # not nest" suggests ("resolve the conflict").
    if len(order) > 1 and knobs:
        for violation in _specialization_twins(order, by_label, knobs):
            print(f"  {violation}")
            bad.append(("specialization-twins", violation))

    if unverified:
        # Stated by name and kept out of both verdict lists. It is not a failure:
        # a kernel declaring no specialized metadata field is a bundle whose
        # compiler settled nothing the matcher can be checked against, which is
        # the mandatory declaration for a non-compiled source. It is not part of
        # the pass either: no producing-build record was read for these, so the
        # pass line must not say their binaries were bound.
        print(f"  {len(unverified)} kernel(s) NOT VERIFIED HERE:")
        for u in unverified:
            print(f"      ? {u}")
        print(
            "  Only rocKE-origin kernels currently carry compiled-specialization "
            "evidence; a hip kernel AOT-compiled with specializing preprocessor "
            "defines is a real compiled specialization that this check does not "
            "yet verify, so absence of a claim is a limit of this tool, not a "
            "property of the kernel."
        )

    print()
    if bad:
        print(f"GATE FAILED ({len(bad)} problem(s))")
        return 1
    if skipped:
        names = ", ".join(sorted(set(skipped)))
        if args.mode == "full":
            # A full run claims every property, so a check that could not run is
            # a gap in the claim rather than a narrower pass. Reported here, on
            # this tool's own exit code: a caller reading the claim off the exit
            # status must not need a second tool to scrape the caveat back out.
            print(f"GATE FAILED ({len(set(skipped))} check(s) NOT RUN: {names})")
            print(
                "  --mode full claims compiled specialization agreement and "
                "vocabulary as well as the structural properties. Supply what "
                "the unrun check needs, or ask for --mode structural and take "
                "the narrower claim in writing."
            )
            return 1
        print(
            "GATE PASSED on what it checked: binaries nest, tuples unique, no "
            "sentinel."
        )
        print(f"  {len(set(skipped))} check(s) NOT RUN: {names}")
        print(
            "  This run did NOT check that any shipped binary agrees with the "
            "metadata that selects it. Only --mode full reads the producing "
            "compiler's evidence, and only it can make that claim."
        )
        return 0
    if unverified:
        print(
            "GATE PASSED: binaries nest, tuples unique, no sentinel, vocabulary "
            "correct, and compiled specialization agrees with metadata for every "
            "kernel that declares one -- see NOT VERIFIED HERE above for the rest"
        )
        return 0
    print(
        "GATE PASSED: binaries nest, tuples unique, no sentinel, compiled "
        "specialization agrees with metadata, vocabulary correct"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
