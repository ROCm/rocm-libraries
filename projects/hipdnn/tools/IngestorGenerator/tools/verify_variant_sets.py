"""Gate a set of variant sets on five properties, over generated bundles or installed
trees. Exits non-zero on any failure, so it can sit in front of a build or a sweep.
The numbered sections in `check()` are keyed to this list.

  1. BINARY NESTING. The larger set can still choose everything the smaller one
     could, compared on compiled binaries: normalising metadata labels instead makes
     diverging sets look nested.

  2. LOADER-TUPLE UNIQUENESS. The loader substitutes a KMD field's `default_value`
     for an absent key and requires the resulting tuple to be unique per device. A
     duplicate is not a dropped entry -- it rejects THE WHOLE ENGINE, while exiting 0
     and passing a descriptor-count check.

  3. NO SENTINEL IN A DESCRIPTOR. `-1` means "unresolved". Every compiled artifact has
     a definite setting, and downstream a sentinel aliases onto the KMD default and
     triggers (2).

  4. METADATA MATCHES ITS BINARY. The matcher selects on metadata; the spec decides
     what was built. When they disagree the runtime picks a kernel on false pretences.

  5. VOCABULARY. Metadata carries the hipDNN spelling the matcher compares ("BF16");
     the spec carries the builder's ("bf16"). A descriptor written in the builder's
     vocabulary loads cleanly, reconciles on every count, and matches NOTHING.

`--mode` is required rather than defaulted because the two modes make different
claims; see its `--help` for which. Schemas are reached by reference through the id
chain the documents declare, never by filename -- binding a KDP to a same-stem
sibling gates a bundle against a schema nothing wires it to.
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
#: would let the weaker run print under the stronger one's name.
MODES = ("full", "structural")


def _agreement_python_root() -> Path:
    """The descriptor-packaging python directory, anchored on this file's location.

    Located by ascending to the ancestor that actually contains the subtree rather
    than by counting path components, so the tool answers the same run from the repo
    root, from an installed checkout, or from three directories down.
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

        bundle: <bundle-folder-name>
        vocabulary:
          dtype: [BF16, FP16]

    ``bundle`` pins which engine to gate when a tree hosts more than one; see
    `select`. ``vocabulary`` declares the matcher's legal spellings for a field, as
    either a builder-to-matcher mapping or the legal set. A UKD's
    ``specialization_contract`` -- its own, or the one it inherits from its KDP --
    states the same thing for the fields it specialises on and is merged in
    automatically, so a profile only has to speak for the fields no declaration
    covers.
    """

    def __init__(self, raw: dict, path: str | None = None):
        self.path = path
        self.bundle = raw.get("bundle")
        self.vocabulary = dict(raw.get("vocabulary") or {})
        # ABSENT and EXPLICITLY EMPTY are different claims: (4a) below treats an
        # undeclared string field as ambiguous only when no vocabulary block exists at
        # all, and `bool(self.vocabulary)` cannot tell those two apart.
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
    can answer.
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

    Two dialects, strongest first: a PACKED descriptor names a symbol whose sha256 IS
    the binary, so prefer it. AUTHORED pre-build there is only the builder and the
    spec as written, and an omitted key stays omitted -- absence is the author's
    intent that the kernel decide, and defaulting it would make two genuinely
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

    Two descriptors sharing this key are candidates to be a "specialization twin"
    pair. Built from metadata, not spec, because metadata is what a bigger set's
    author reads when deciding "do I already carry this shape".
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
    checking the same one. A shard that pins exactly one answers this itself; anything
    wider needs `--arch` rather than the gate picking the first entry of a list.
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

    A spec carries Python `True`; metadata carries `1`. Without normalising, that
    spelling difference is reported as a mislabelling.
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
    `unchecked` is a check this run could not run, which under `--mode full` is a gap
    in the claim and fails the gate. `unverified` is a check that ran and found
    nothing to bind; it neither fails the gate nor joins the pass line.
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
    unverified: list[str] = []

    declared = descriptor_context.declarations(bundles, schemas)
    knobs = {f for d in declared.values() for f in d["metadata_fields"]}
    vocabulary = dict(profile.vocabulary)
    for declaration in declared.values():
        vocabulary.update(declaration["vocabulary"])
    # A declaration's vocabulary speaks only for the fields that declaration
    # specialises on, so it is merged into the translations above but does NOT answer
    # "was there a place to declare a translation for THIS field". Only the profile's
    # block does that.
    vocabulary_declared = profile.vocabulary_declared

    # (2) Loader-tuple uniqueness, engine-wide, arch-aware, and canonicalised per the
    # declared KMD type. Engine-wide because the loader assembles ONE catalog per
    # engine per device; arch-aware because candidates whose coverage is disjoint
    # never meet in that catalog while a wildcard overlaps everything; canonicalised
    # because `1` and `1.0` on a FLOAT field are one entry and a BOOL `true` is not an
    # INT `1`.
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
            # A mapping declares builder-spelling -> matcher-spelling and its VALUES
            # are the legal set; a bare list declares the legal set directly. Both are
            # accepted so one profile serves this tool and the parity generator.
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
    # descriptor carries both they must agree -- a property of the descriptor against
    # ITSELF, needing no evidence and running in both modes.
    #
    # ANY field with a declared vocabulary is exempt from the plain comparison, not
    # just the dict-form ones. A vocabulary declaration means the two layers spell the
    # value DIFFERENTLY ON PURPOSE -- builder "bf16", matcher "BF16" -- so comparing
    # them raw reports a mislabelling that is only a translation. Property (5) owns
    # the metadata side of such a field.
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
                # still compare. Where only the legal set is declared there is nothing
                # to translate WITH, so property (5) owns the field entirely.
                if field not in vocabulary_maps:
                    continue
                if isinstance(spec_value, str):
                    spec_value = vocabulary_maps[field].get(spec_value, spec_value)
            elif isinstance(spec_value, str) or isinstance(meta_value, str):
                # An UNDECLARED string field. Whether that means "same spelling in
                # both layers" or "translated, just not written down" depends on
                # whether a vocabulary was declared AT ALL. With none, there is
                # nowhere an author COULD have said "translated", so the common and
                # correct `spec: "bf16"` / `metadata: "BF16"` pairing must not become
                # a false failure -- it is recorded instead. With one, the author had
                # the exact place to say "translated" and did not, so there is nothing
                # left to guess: compare it raw like any plain field.
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
        # A field nobody can judge is named rather than folded into an unqualified
        # pass. Only reachable with no vocabulary declared anywhere: once one exists,
        # an unmentioned field is compared raw above instead of landing here.
        unchecked.append(
            "metadata-matches-binary for UNDECLARED STRING field(s) "
            f"{', '.join(sorted(undeclared_string_fields))} (no 'vocabulary' in "
            f"profile or declaration to judge them against)"
        )

    # (4b) COMPILED SPECIALIZATION: the fields a declaration marks as settled by the
    # compiler. A descriptor may leave one out of its spec entirely, so only the
    # producing compile's own evidence says what the binary was built with. Full mode
    # checks it against the payload bytes in hand; structural mode cannot, and says so
    # by name.
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
                # A declaration with no `metadata_fields` is the MANDATORY declaration
                # for a non-compiled source: no producing-build record exists to bind,
                # so it is reported rather than failed or absorbed into the pass.
                # `hkp_pack.desk_check` says the same about the same artifact.
                #
                # A descriptor that CARRIES `provenance.effective_spec` is checked
                # whatever its declaration claims -- otherwise relabelling the
                # specialized field as matcher-only would waive a stale or forged
                # record into a pass.
                #
                # The waiver is keyed on origin too. `provenance.origin_kind ==
                # "rocke"` means the packer published the evidence when it shipped
                # this kernel, so dropping `effective_spec` and moving the
                # specialized fields to matcher-only would otherwise retire the
                # check and leave the archive bytes unread. An ABSENT `origin_kind`
                # is NOT rocKE: descriptors packed before the field existed and
                # hand-authored inputs have none.
                #
                # This reaches evidence lost by accident, not evidence removed on
                # purpose. `origin_kind` is bound only by the digest inside the
                # record being dropped, so one edit can remove the record and set
                # the origin to `"hip"` together, presenting as a source that never
                # owed evidence.
                provenance = entry.ukd.get("provenance") or {}
                claimed = any(r["declaration"]["metadata_fields"] for r in bound)
                if not claimed and "effective_spec" not in provenance:
                    if provenance.get("origin_kind") == "rocke":
                        failures.append(
                            f"{name}: provenance.origin_kind is 'rocke', so the "
                            f"packer published this kernel's compiler-owned "
                            f"provenance.effective_spec when it shipped it. The "
                            f"descriptor in hand declares no specialized "
                            f"metadata_fields AND carries no effective_spec, so "
                            f"there is no record left to bind and the archive bytes "
                            f"were never read. A rocKE-produced kernel is required "
                            f"to carry its compiler evidence; relabelling its "
                            f"specialized fields as matcher-only does not make it an "
                            f"unspecialized source."
                        )
                        continue
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
        # Stated by name and kept out of both verdict lists: not a failure, and not
        # part of the pass either -- no producing-build record was read for these, so
        # the pass line must not say their binaries were bound.
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
            # A full run claims every property, so a check that could not run is a gap
            # in the claim rather than a narrower pass -- and it lands on this tool's
            # own exit code, so a caller need not scrape the caveat out of the output.
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
