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

WHAT IS DECLARED AND WHAT IS DISCOVERED. Properties 1-3 are structural: they read the
KMD and the descriptors and need to know nothing about the kernel. Properties 4 and 5
cannot be: resolving "policy decides" means asking the kernel's own policy function,
and knowing that "BF16" is right and "bf16" is wrong means knowing the matcher's
vocabulary. Both are per-kernel facts, so both live in a PROFILE beside the config
rather than in this file.

The first version of this tool hardcoded one knob (`use_exp2_fast`), one bundle
(`gfx942_attention_dense`) and one vocabulary (`BF16`/`FP16`), which made it a gate for
exactly one integration. A sweep over 22 knobs cannot be gated by a tool that knows
about one, and the knob-by-knob sweep is precisely when properties 3 and 4 matter most.

WITHOUT A PROFILE the structural checks still run and the policy-dependent ones are
reported as NOT CHECKED, by name, in the output and the exit summary. They are never
silently skipped: a gate that quietly stops checking is the failure mode this whole
file exists to prevent.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
from collections import Counter

#: KMD value meaning "unresolved -- the kernel's own policy decides". Never legal in a
#: shipped descriptor; see property 3.
SENTINEL = -1


class Profile:
    """Per-kernel facts the structural checks cannot derive from the artifacts.

    A profile is a small JSON/YAML document beside the generator config::

        bundle: gfx942_attention_dense
        provider_root: dnn-providers/hip-kernel-provider
        vocabulary:
          dtype: [BF16, FP16]
        policies:
          use_exp2_fast:
            module: kernels.gfx942.attention_dense
            function: _use_exp2_fast
            args: [head_size, dtype, seqlen_q]

    ``policies`` is a MAPPING, not a single field, so a second tri-state knob is a
    profile edit rather than a change to this tool. ``args`` names spec keys, passed
    positionally to the policy function in the order given -- the same order the
    kernel's own resolver takes them.
    """

    def __init__(self, raw: dict, path: str | None = None):
        self.path = path
        self.bundle = raw.get("bundle")
        self.provider_root = raw.get("provider_root")
        self.vocabulary = dict(raw.get("vocabulary") or {})
        self.policies = dict(raw.get("policies") or {})
        # ABSENT and EXPLICITLY EMPTY are different claims, and collapsing them is
        # the same mistake this gate exists to catch one layer down. A profile with
        # no `policies:` key has not been asked the question -- the check is narrowed
        # and says so. A profile that writes `policies: {}` ASSERTS the kernel has no
        # policy-owned knob, which is a fact about the kernel (gfx950 attention_dense
        # takes the shared spec; every tri-state on its gfx942 sibling belongs to a
        # private subclass that does not exist here). That assertion is checkable and
        # is checked: an empty declaration with a policy-shaped field present would
        # still fail below.
        self.policies_declared = "policies" in raw
        # Same ABSENT-vs-EXPLICIT distinction as `policies_declared`, and needed for
        # the same reason: (4a) below must tell "no vocabulary block exists, so an
        # undeclared string field is genuinely ambiguous" from "a vocabulary block
        # exists and simply never mentions this field", which is not ambiguous -- the
        # author had the exact place to declare a translation and did not.
        self.vocabulary_declared = "vocabulary" in raw
        self._resolvers: dict = {}

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

    def bind(self) -> None:
        """Import each policy function once, failing loudly if one cannot be found.

        Resolution is deliberately eager. A policy that fails to import at descriptor
        number 900 would leave the earlier 899 reported as checked when they were not.
        """
        root = self.provider_root or os.environ.get("ROCKE_PROVIDER_ROOT")
        if root:
            root = os.path.abspath(os.path.expanduser(root))
            for sub in ("rocke/library", "rocke/platform/python"):
                candidate = os.path.join(root, sub)
                if os.path.isdir(candidate) and candidate not in sys.path:
                    sys.path.insert(0, candidate)

        for knob, decl in self.policies.items():
            module_name = decl.get("module")
            func_name = decl.get("function")
            args = list(decl.get("args") or [])
            if not module_name or not func_name:
                raise SystemExit(
                    f"FAIL: profile policy '{knob}' needs both 'module' and "
                    f"'function'."
                )
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                raise SystemExit(
                    f"FAIL: profile policy '{knob}' names module '{module_name}', "
                    f"which will not import ({exc}). Set provider_root in the profile "
                    f"or ROCKE_PROVIDER_ROOT in the environment so the kernel library "
                    f"is importable -- the gate cannot resolve 'policy decides' "
                    f"without asking the kernel."
                )
            func = getattr(module, func_name, None)
            if func is None:
                raise SystemExit(
                    f"FAIL: profile policy '{knob}' names '{func_name}' in "
                    f"'{module_name}', which does not define it."
                )
            self._resolvers[knob] = (func, args)

    def resolve(self, knob: str, spec: dict):
        """What the BINARY was built with for `knob`, given its spec.

        An explicit value in the spec is the answer. Absent means the kernel's policy
        decides, so ask the policy the same question the builder asked it.
        """
        value = spec.get(knob)
        if value is not None:
            return int(bool(value)) if isinstance(value, bool) else value
        func, argnames = self._resolvers[knob]
        try:
            argv = [spec[a] for a in argnames]
        except KeyError as exc:
            raise SystemExit(
                f"FAIL: policy for '{knob}' needs spec key {exc}, which this "
                f"descriptor's kernel_source.spec does not carry."
            )
        result = func(*argv)
        return int(bool(result)) if isinstance(result, bool) else result


def _binary_key(descriptor: dict, profile: Profile) -> str:
    """Identity of the compiled artifact this descriptor names.

    Two dialects, because the same bundle is checkable before and after packing:

      * `rocke` (generated, pre-build): builder plus the spec it will be built from,
        with every policy-owned knob resolved so that "absent" and "explicitly the
        value the policy would have chosen" name the SAME binary -- which they do.
      * `kpack` (installed, post-build): the spec is compiled away and the descriptor
        names a symbol in the archive. The sha256 IS the binary, a stronger identity
        than any reconstruction, so prefer it when present.
    """
    source = descriptor["kernel_source"]
    if source.get("sha256") or source.get("symbol"):
        return json.dumps(
            {"sha256": source.get("sha256"), "symbol": source.get("symbol")},
            sort_keys=True,
        )
    spec = dict(source.get("spec") or {})
    for knob in profile.policies:
        spec[knob] = profile.resolve(knob, spec)
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
    settings of `knob` -- candidates to be a "policy twin" pair. Built from
    metadata, not spec, because metadata is what a bigger set's author reads
    when deciding "do I already carry this shape".
    """
    metadata = descriptor["metadata"]
    return json.dumps(
        sorted((k, repr(v)) for k, v in metadata.items() if k != knob),
        sort_keys=True,
    )


def _policy_twins(order: list, by_label: dict, profile: "Profile") -> list:
    """Shapes where a bigger set overrides a policy-decided knob instead of adding to it.

    Binary nesting (property 1) already reports the symptom here -- "N binaries
    absent from BIG" -- and that is correct, but it names no knob and implies no
    fix. The fix that actually applies is never "resolve a conflict": the
    smaller set's descriptor left `knob` for the kernel's own policy to decide
    at build time, so the bigger set must carry THAT variant unchanged, with
    the override sitting BESIDE it as a second binary -- not in its place. A
    reader told only "a binary is missing" may reach for the wrong fix and
    repin the override instead of restoring the policy-decided twin, which is
    the same class of mistake that shipped 364 mislabelled descriptors under
    property 4.

    A pinned value that happens to equal what the policy would have resolved
    is not a violation -- it is the SAME binary under another spelling, which
    is exactly what `_binary_key()` already normalises, so that equality is
    delegated to it rather than re-decided here from raw spec values.
    """
    violations = []
    for small_label, big_label in zip(order, order[1:]):
        small_descs = by_label[small_label]
        big_descs = by_label[big_label]
        for knob in profile.policies:
            big_by_shape: dict = {}
            for c in big_descs:
                big_by_shape.setdefault(_shape_key(c, knob), []).append(c)
            for d in small_descs:
                spec = d["kernel_source"].get("spec") or {}
                if not spec or spec.get(knob) is not None:
                    continue  # nothing to resolve, or this descriptor pins it itself
                candidates = big_by_shape.get(_shape_key(d, knob))
                if not candidates:
                    continue  # no shape match at all: a plain nesting gap, not a twin
                candidate_keys = {_binary_key(c, profile) for c in candidates}
                if _binary_key(d, profile) in candidate_keys:
                    continue  # the twin is present, or the override resolves the same
                if any(
                    (c["kernel_source"].get("spec") or {}).get(knob) is None
                    for c in candidates
                ):
                    continue  # a differently-resolved policy twin already covers it
                shape = {k: v for k, v in d["metadata"].items() if k != knob}
                violations.append(
                    f"policy twin missing: {big_label} carries only a pinned "
                    f"'{knob}' at shape {shape}, none left to policy like "
                    f"{small_label}'s {d['name']} -- carry BOTH variants, the "
                    f"override alone drops {small_label}'s kernel from the "
                    f"candidate list"
                )
    return violations


def load(root: str, profile: Profile):
    """(kmd, descriptors) for the bundle under `root`.

    The bundle is discovered rather than named, so this works for any engine. A
    profile may pin `bundle` when a tree carries more than one, which is the normal
    case for an installed tree hosting several engines.
    """
    pattern = f"{profile.bundle}.kdp.json" if profile.bundle else "*.kdp.json"
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(
            f"FAIL: no {pattern} under {root}"
            + ("" if profile.bundle else " -- is this a descriptor tree?")
        )
    if len(hits) > 1:
        names = ", ".join(
            sorted({os.path.basename(h)[: -len(".kdp.json")] for h in hits})
        )
        raise SystemExit(
            f"FAIL: {len(hits)} bundles under {root} ({names}). Set 'bundle' in the "
            f"profile to say which one this gate is about -- checking the wrong "
            f"engine would pass while the one under test is broken."
        )
    kdp = hits[0]
    kmd = kdp.replace(".kdp.json", ".kmd.json")
    if not os.path.exists(kmd):
        raise SystemExit(
            f"FAIL: {kdp} has no sibling .kmd.json; cannot read the schema."
        )
    with open(kmd) as fh:
        kmd_doc = json.load(fh)
    with open(kdp) as fh:
        descriptors = json.load(fh)["kernelDescriptors"]
    return kmd_doc, descriptors


def check(label: str, root: str, profile: Profile):
    kmd, descriptors = load(root, profile)
    names = [f["name"] for f in kmd["fields"]]
    defaults = {f["name"]: f.get("default_value") for f in kmd["fields"]}
    failures: list[str] = []
    unchecked: list[str] = []

    # (2) Loader-tuple uniqueness, with KMD defaults substituted exactly as the loader
    # substitutes them. Structural: needs no kernel knowledge.
    tuples = Counter(
        tuple(k["metadata"].get(n, defaults.get(n)) for n in names) for k in descriptors
    )
    collisions = sum(n - 1 for n in tuples.values() if n > 1)
    if collisions:
        failures.append(
            f"{collisions} loader-tuple collisions (engine would be dropped)"
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
    if profile.vocabulary:
        for field, declared in profile.vocabulary.items():
            # A mapping declares builder-spelling -> matcher-spelling, which the
            # parity generator uses to translate; the legal set is its VALUES. A
            # bare list declares the legal set directly. Both spellings are
            # accepted so one profile serves both tools without restating itself.
            allowed = declared.values() if isinstance(declared, dict) else declared
            allowed_set = {str(a) for a in allowed}
            wrong = sorted(
                {
                    str(k["metadata"].get(field))
                    for k in descriptors
                    if str(k["metadata"].get(field)) not in allowed_set
                }
            )
            if wrong:
                failures.append(
                    f"{field} written in the wrong vocabulary: {wrong} "
                    f"(the matcher compares {sorted(allowed_set)})"
                )
    else:
        unchecked.append("vocabulary (no 'vocabulary' in profile)")

    # (4) Metadata matches the binary it names.
    #
    # TWO KINDS OF FIELD, and only one of them needs kernel knowledge.
    #
    # (4a) PLAIN FIELDS: a metadata key that is ALSO a spec key. The spec is what the
    # builder compiles and the metadata is what the matcher compares, so if a
    # descriptor carries both they must agree -- no policy, no profile, no kernel
    # knowledge required. This is a property of the descriptor against ITSELF.
    #
    # This check used to not exist. Property (4) iterated `profile.policies` and
    # nothing else, so a kernel with no policy-owned knobs -- which `policies: {}`
    # correctly declares for gfx950 -- had property (4) checking NOTHING while the
    # gate still printed a clean pass. A review demonstrated the consequence by
    # mutation: flipping a descriptor's metadata.ragged to 0 while its spec still
    # said True passed the gate 84/84 OK. That is the exact "dangerous direction" the
    # shipping commit names -- an aligned-labelled descriptor whose binary is
    # boundary-padded -- caught downstream by the C++ matcher unit tests but invisible
    # at the STATIC rung, which coverage_gate.py's own docstring insists is a separate
    # rung precisely because each catches what the other cannot. The same hole
    # swallowed a head_size mismatch and, once they existed, varlen/paged both ways.
    # ANY field with a declared vocabulary is exempt from the plain comparison, not
    # just the dict-form ones. A vocabulary declaration means the two layers spell the
    # value DIFFERENTLY ON PURPOSE -- builder "bf16", matcher "BF16" -- so comparing
    # them raw reports a mislabelling that is only a translation. The dict form maps
    # the translation and the list form declares the matcher's legal set; both say
    # "this field is translated". Property (5) checks the metadata side against the
    # legal set, which is the check that actually applies to a translated field.
    translated_fields = set(profile.vocabulary)
    vocabulary_maps = {
        field: mapping
        for field, mapping in profile.vocabulary.items()
        if isinstance(mapping, dict)
    }

    def _comparable(value):
        """One representation for values the two layers spell differently.

        A spec carries Python `True`; metadata carries `1`. Normalising here rather
        than special-casing bool keeps the check from reporting a mislabelling that is
        only a spelling difference.
        """
        if isinstance(value, bool):
            return str(int(value))
        return str(value)

    plain_mismatches = []
    undeclared_string_fields = set()
    for descriptor in descriptors:
        spec = descriptor["kernel_source"].get("spec") or {}
        if not spec:
            continue
        for field, meta_value in descriptor["metadata"].items():
            if field not in spec or field in profile.policies:
                continue
            spec_value = spec[field]
            if field in translated_fields:
                # Translated on purpose. Where the profile gives the MAPPING, apply it
                # and still compare -- that catches a descriptor whose metadata names
                # a different dtype than its spec builds. Where it declares only the
                # legal set, there is nothing to translate WITH, so property (5) owns
                # the field entirely and this comparison must not guess.
                if field not in vocabulary_maps:
                    continue
                if isinstance(spec_value, str):
                    spec_value = vocabulary_maps[field].get(spec_value, spec_value)
            elif isinstance(spec_value, str) or isinstance(meta_value, str):
                # An UNDECLARED string field: no `vocabulary:` entry names it. Whether
                # that means "same spelling both layers" or "translated, just not
                # written down" depends on whether the profile has a vocabulary
                # section AT ALL.
                #
                # No section at all: there is nowhere an author COULD have declared a
                # translation, so an undeclared string is genuinely ambiguous -- the
                # common, correct `spec: "bf16"` / `metadata: "BF16"` pairing must not
                # become a false failure. Recorded rather than silently dropped: a
                # field nobody can judge is a fact the author should see, not a check
                # that quietly asked nothing.
                #
                # A section exists and just never mentions this field: the author had
                # the exact place to say "this field is translated" and did not, so
                # there is nothing left to guess -- compare it raw like any plain
                # field. This is the branch that closes the escape a review found on
                # the real gfx950 tree: `persist_decode` written only in
                # kernel_source.spec, with a vocabulary section present that declares
                # `dtype` and says nothing about `persist_decode`, let 84 descriptors
                # carry a contradicting `metadata.persist_decode` and still pass.
                if not profile.vocabulary_declared:
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
        # pass. Only reachable with no `vocabulary:` block in the profile at all: once
        # one exists, an unmentioned field is compared raw above instead of landing
        # here.
        unchecked.append(
            "metadata-matches-binary for UNDECLARED STRING field(s) "
            f"{', '.join(sorted(undeclared_string_fields))} (no 'vocabulary' in "
            f"profile to judge them against)"
        )

    # (4b) POLICY-OWNED KNOBS: absent from the spec means the kernel's own policy
    # decided at build time, so the binary is definite but only the policy can say
    # what it chose. This is the half that needs kernel knowledge.
    if profile.policies:
        for knob in profile.policies:
            mislabelled = []
            for k in descriptors:
                spec = k["kernel_source"].get("spec") or {}
                if not spec:
                    continue
                want = profile.resolve(knob, spec)
                got = k["metadata"].get(knob)
                if got is None or int(got) != int(want):
                    mislabelled.append(k["name"])
            if mislabelled:
                failures.append(
                    f"{len(mislabelled)} descriptors mislabel their binary on "
                    f"'{knob}', e.g. {mislabelled[0]}"
                )
    elif profile.policies_declared:
        # `policies: {}` -- the author asserts this kernel has no policy-owned knob.
        # Nothing for (4b) to compare, and nothing MISSING either. This no longer
        # means property (4) as a whole checks nothing: (4a) above runs
        # unconditionally and needs no profile at all.
        pass
    else:
        unchecked.append(
            "metadata-matches-binary for POLICY knobs (no 'policies' in profile)"
        )

    binaries = {_binary_key(k, profile) for k in descriptors}
    verdict = "OK" if not failures else "FAIL"
    print(
        f"  {label}: descriptors={len(descriptors):5d} "
        f"distinct-binaries={len(binaries):5d} {verdict}"
    )
    for f in failures:
        print(f"      ! {f}")
    for u in unchecked:
        print(f"      ? NOT CHECKED: {u}")
    return binaries, descriptors, failures, unchecked


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
        "--profile",
        help="Per-kernel profile (JSON or YAML) declaring the bundle name, the "
        "matcher's vocabulary and each policy-owned knob's resolver. Without it the "
        "structural checks still run and the rest are reported as NOT CHECKED.",
    )
    args = parser.parse_args(argv)

    if len(args.pairs) % 2:
        parser.error("arguments must be LABEL ROOT pairs")
    roots = list(zip(args.pairs[::2], args.pairs[1::2]))

    profile = Profile.load(args.profile) if args.profile else Profile.empty()
    profile.bind()

    print("variant-set gate")
    if not args.profile:
        print("  (no profile: policy and vocabulary checks are NOT CHECKED)")

    sets, by_label, bad, skipped = {}, {}, [], []
    for label, root in roots:
        binaries, descriptors, failures, unchecked = check(label, root, profile)
        sets[label] = binaries
        by_label[label] = descriptors
        bad += [(label, f) for f in failures]
        skipped += unchecked

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

    # (1b) Policy twins: a special case of (1) worth naming on its own, because the
    # fix it needs ("carry both variants") is not the fix "binaries do not nest"
    # suggests ("resolve the conflict").
    if len(order) > 1 and profile.policies:
        for violation in _policy_twins(order, by_label, profile):
            print(f"  {violation}")
            bad.append(("policy-twins", violation))

    print()
    if bad:
        print(f"GATE FAILED ({len(bad)} problem(s))")
        return 1
    if skipped:
        print(
            "GATE PASSED on what it checked: binaries nest, tuples unique, no "
            "sentinel."
        )
        print(
            f"  {len(set(skipped))} check(s) NOT RUN for want of a profile: "
            f"{', '.join(sorted(set(skipped)))}"
        )
        return 0
    print(
        "GATE PASSED: binaries nest, tuples unique, no sentinel, no mislabelling, "
        "vocabulary correct"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
