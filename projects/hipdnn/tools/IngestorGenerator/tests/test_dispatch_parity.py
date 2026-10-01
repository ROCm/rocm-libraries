# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What `dispatch_parity.py` reports, and what it refuses to report.

A shape that is not served has one per-shape explanation from the layer that said no:
the eligibility predicate (declined), the spec factory's ValueError, which the
dispatcher's own candidate treats as a decline (refused), or a graph feature outside
the profile's graph contract (out_of_contract). Request construction failing, or the
factory raising anything else, aborts the command instead, because a corpus the
request class cannot hydrate makes every remaining count untrustworthy, and a
`rejected` bucket that can only print 0 claims a failure was checked for. The
dispatcher, request class and predicate are stubs, except in the classes that bind
the shipped profile.

Also what the tool BINDS before it can report anything: where a profile's
``provider_root`` resolves from, and which dispatch arm the shipped profile pins.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dispatch_parity  # noqa: E402
import launch_surface  # noqa: E402

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_SHIPPED_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "gfx950_attention_dense.profile.yaml"
)

#: The decline carries a real reason rather than a blanket refusal so the served
#: control survives alongside it.
_STUB_PROVIDER = '''
import dataclasses


@dataclasses.dataclass
class Request:
    seqlen_q: int
    head_size: int = 128


@dataclasses.dataclass
class Spec:
    seqlen_q: int
    head_size: int
    block_n: int


def resolve(request):
    """Derive a field rather than defaulting it, as a real dispatcher would."""
    return Spec(
        seqlen_q=request.seqlen_q,
        head_size=request.head_size,
        block_n=64 if request.seqlen_q >= 1024 else 32,
    )


def resolve_but_raise(request):
    """A dispatcher that fails operationally once the per-shape loop calls it.

    The message names the request it was handed, so a test can tell "the factory
    ran and threw" apart from "the factory was never reached". Not a ValueError:
    that is the dispatcher's shape-refusal path, which is a decline.
    """
    raise RuntimeError(f"dispatcher exploded on seqlen_q {request.seqlen_q}")


def resolve_refusing(request):
    """A spec factory that refuses one shape the way rocKE specs do: ValueError
    from construction, which the dispatcher's own candidate reports as a
    decline."""
    if request.head_size == 192:
        raise ValueError(f"head_size must be 64 or 128, got {request.head_size}")
    return resolve(request)


def supports(spec, arch=None):
    if spec.seqlen_q == 777:
        return False, "seqlen_q 777 is not a supported prefill length"
    return True, ""
'''


@pytest.fixture
def parity(tmp_path, monkeypatch):
    """A profile, a corpus and a provider root the tool can bind. Returns a callable
    over the shape list, so each test states its own corpus."""
    library = tmp_path / "provider" / "rocke" / "library"
    library.mkdir(parents=True)
    (tmp_path / "provider" / "rocke" / "platform" / "python").mkdir(parents=True)
    (library / "stub_provider.py").write_text(_STUB_PROVIDER)
    # The tool inserts the provider dirs itself; popping the module keeps one test's
    # import from satisfying the next one's from a stale sys.modules entry.
    monkeypatch.delitem(sys.modules, "stub_provider", raising=False)

    profile = {
        "slug": "stub_attention",
        "source": "kernels/stub.py",
        "builder": "build_stub",
        "engine": {"name": "stub:Engine"},
        "kmd_fields": [{"name": "seqlen_q", "type": "int", "default_value": 256}],
        "provider_root": str(tmp_path / "provider"),
        "dispatch": {"module": "stub_provider", "function": "resolve"},
        "request": {"module": "stub_provider", "class": "Request"},
        "predicate": {"module": "stub_provider", "function": "supports"},
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))

    def argv(shapes: list, *extra: str, **profile_overrides) -> list:
        profile_path.write_text(json.dumps({**profile, **profile_overrides}))
        shapes_path = tmp_path / "shapes.json"
        shapes_path.write_text(json.dumps(shapes))
        return [
            "--profile",
            str(profile_path),
            "--shapes",
            str(shapes_path),
            *extra,
        ]

    return argv


_SERVED_AND_DECLINED = [{"seqlen_q": 256}, {"seqlen_q": 2048}, {"seqlen_q": 777}]


class TestTheReportCarriesNoUnpopulatableBucket:
    def test_the_summary_names_no_rejected_bucket(self, parity, capsys):
        """A request-construction failure returns 2 long before the summary prints,
        so no `rejected` bucket exists to print."""
        assert dispatch_parity.main(parity(_SERVED_AND_DECLINED)) == 0
        out = capsys.readouterr().out
        assert "rejected" not in out, (
            "the summary still prints a bucket nothing can populate; a count that "
            "is structurally always 0 reads as a check that passed"
        )
        assert "spec construction raised" not in out, (
            "the summary still offers spec construction as a per-shape outcome, but "
            "that path aborts the command instead of bucketing the shape"
        )

    def test_the_counts_that_remain_are_still_right(self, parity, capsys):
        """A control: the live counts are right, so an absent `rejected` line is a
        report with a bucket missing rather than a harness that printed nothing."""
        assert dispatch_parity.main(parity(_SERVED_AND_DECLINED)) == 0
        out = capsys.readouterr().out
        assert "shapes in         3" in out
        assert "servable          2" in out
        assert "declined          1" in out

    def test_report_gaps_lists_the_decline_with_its_reason(self, parity, capsys):
        """`--report-gaps` is the tool's whole answer to an uncovered shape, and it
        prints from the same loop the dead bucket would join."""
        assert dispatch_parity.main(parity(_SERVED_AND_DECLINED, "--report-gaps")) == 0
        out = capsys.readouterr().out
        assert "[declined]" in out
        assert "seqlen_q 777 is not a supported prefill length" in out

    def test_report_gaps_prints_nothing_when_every_shape_is_served(
        self, parity, capsys
    ):
        """No gaps means no gap lines, not a bucket header with 0 under it."""
        assert dispatch_parity.main(parity([{"seqlen_q": 256}], "--report-gaps")) == 0
        out = capsys.readouterr().out
        assert "[declined]" not in out
        assert "rejected" not in out


class TestConstructionFailureAbortsRatherThanBuckets:
    def test_an_unhydratable_shape_exits_2_naming_the_failure(self, parity, capsys):
        """A corpus key the request class does not accept is not a shape-level verdict:
        the tool cannot say whether the kernel would serve it."""
        shapes = [{"seqlen_q": 256}, {"seqlen_q": 512, "nonexistent_field": 1}]
        assert dispatch_parity.main(parity(shapes)) == 2

        captured = capsys.readouterr()
        assert "request/spec construction failed" in captured.err
        assert captured.err.startswith("FAIL:")
        assert "dispatcher parity" not in captured.out, (
            "a summary was printed for a corpus that failed to hydrate; the counts "
            "would describe only the shapes processed before the failure"
        )

    def test_a_dispatcher_that_raises_also_exits_2(self, parity, capsys, monkeypatch):
        """A non-ValueError from the factory is operational, never a decline. The
        dispatcher's own message is asserted so an exit 2 raised while resolving the
        symbol does not pass."""
        shapes = [{"seqlen_q": 256}]
        argv = parity(shapes)
        real_resolve_shapes = dispatch_parity.resolve_shapes

        def with_raising_dispatcher(shapes_arg, profile):
            profile = dict(profile)
            profile["dispatch"] = {
                "module": "stub_provider",
                "function": "resolve_but_raise",
            }
            return real_resolve_shapes(shapes_arg, profile)

        monkeypatch.setattr(dispatch_parity, "resolve_shapes", with_raising_dispatcher)
        assert dispatch_parity.main(argv) == 2

        captured = capsys.readouterr()
        assert "FAIL:" in captured.err
        assert "dispatcher exploded on seqlen_q 256" in captured.err
        assert "request/spec construction failed" in captured.err
        assert "dispatcher parity" not in captured.out, (
            "a summary was printed for a corpus whose dispatcher raised; the counts "
            "would describe only the shapes resolved before the failure"
        )

    def test_a_predicate_decline_is_not_promoted_to_an_abort(self, parity, capsys):
        """The abort policy must not swallow the one outcome that IS a per-shape
        verdict."""
        assert dispatch_parity.main(parity([{"seqlen_q": 777}])) == 1
        assert "no shape resolved" in capsys.readouterr().err


class TestASpecFactoryRefusalIsADecline:
    """rocKE specs refuse unsupported shapes by raising ValueError, and the
    dispatcher's own candidate `support()` reports that as a decline. One such
    shape in a real corpus must not abort the whole run."""

    _REFUSING = {
        "dispatch": {"module": "stub_provider", "function": "resolve_refusing"}
    }

    def test_a_refused_shape_is_reported_and_the_rest_resolves(self, parity, capsys):
        shapes = [{"seqlen_q": 256}, {"seqlen_q": 256, "head_size": 192}]
        argv = parity(shapes, "--report-gaps", **self._REFUSING)
        assert dispatch_parity.main(argv) == 0, capsys.readouterr().err
        out = capsys.readouterr().out
        assert "servable          1" in out
        assert "refused           1" in out
        assert "[refused]" in out
        assert "head_size must be 64 or 128, got 192" in out

    def test_a_corpus_of_only_refusals_exits_1_not_2(self, parity, capsys):
        """Every shape refused is a corpus nothing serves, not an operational
        failure."""
        argv = parity([{"seqlen_q": 256, "head_size": 192}], **self._REFUSING)
        assert dispatch_parity.main(argv) == 1
        assert "no shape resolved" in capsys.readouterr().err


class TestGraphFeaturesOutsideTheContractAreNotServed:
    """A mined graph's varlen/layout features have no request field, so the
    dispatcher's yes for the bare request does not make the graph servable."""

    def test_a_bound_feature_the_contract_does_not_admit_is_out_of_contract(
        self, parity, capsys
    ):
        shapes = [
            {"seqlen_q": 256, "_graph_features": {"layouts": ["BSHD"], "features": []}},
            {
                "seqlen_q": 256,
                "_graph_features": {
                    "layouts": ["BSHD"],
                    "features": ["seq_len_kv", "seq_len_q"],
                },
            },
        ]
        assert dispatch_parity.main(parity(shapes, "--report-gaps")) == 0
        out = capsys.readouterr().out
        assert "servable          1" in out
        assert "out of contract   1" in out
        assert "[out_of_contract]" in out and "seq_len_kv, seq_len_q" in out

    def test_a_layout_outside_the_contract_is_out_of_contract(self, parity, capsys):
        shapes = [
            {"seqlen_q": 256, "_graph_features": {"layouts": ["BHSD"], "features": []}},
            {
                "seqlen_q": 512,
                "_graph_features": {"layouts": ["BHSD", "BSHD"], "features": []},
            },
        ]
        argv = parity(shapes, "--report-gaps", graph_contract={"layouts": ["BSHD"]})
        assert dispatch_parity.main(argv) == 0
        out = capsys.readouterr().out
        assert "servable          1" in out, "a single-head tensor is also BSHD"
        assert "operand layout BHSD is not in graph_contract.layouts" in out

    def test_a_contract_feature_admits_the_graph(self, parity, capsys):
        shapes = [
            {"seqlen_q": 256, "_graph_features": {"layouts": [], "features": ["x"]}}
        ]
        argv = parity(shapes, graph_contract={"features": ["x"]})
        assert dispatch_parity.main(argv) == 0
        assert "servable          1" in capsys.readouterr().out


#: The relative root the shipped profile names. Spelled out so the decoy below can
#: reproduce it exactly: under a cwd-relative resolution the decoy is what binds.
_REPO_RELATIVE_ROOT = "dnn-providers/hip-kernel-provider"


def _make_provider(root: Path) -> Path:
    """The two directories ``_bind_provider`` requires of a provider root."""
    (root / "rocke" / "library").mkdir(parents=True)
    (root / "rocke" / "platform" / "python").mkdir(parents=True)
    return root


class TestProviderBindingIsIndependentOfTheInvocationDirectory:
    """``provider_root`` is repository-relative, anchored on the tool's own location.

    A root resolved against the current directory makes one profile correct from one
    directory and silently wrong from every other: the import fails where the tree is
    absent, and -- worse -- binds a same-shaped tree that happens to sit under the
    caller's cwd.
    """

    @pytest.fixture
    def bind(self, monkeypatch):
        """Bind a root and return the entries it ADDED to ``sys.path``.

        The difference, not a substring scan of the whole path: earlier tests in this
        module bind stub providers of their own, and a scan would report those too. A
        copy of ``sys.path`` is swapped in for the duration, so one test's provider
        cannot satisfy the next one's import and no real rocKE library outlives it.
        """
        monkeypatch.setattr(sys, "path", list(sys.path))
        baseline = list(sys.path)

        def _bind(root):
            dispatch_parity._bind_provider(root)
            return [entry for entry in sys.path if entry not in baseline]

        return _bind

    def test_a_relative_root_binds_the_checkout_not_a_look_alike_under_the_cwd(
        self, bind, tmp_path, monkeypatch
    ):
        """The decoy has the SAME relative layout and sits at the cwd, so a pass here
        cannot be explained by the tool simply failing to find anything."""
        decoy = _make_provider(tmp_path / "decoy" / _REPO_RELATIVE_ROOT)
        monkeypatch.chdir(tmp_path / "decoy")

        added = bind(_REPO_RELATIVE_ROOT)

        assert added, "nothing was bound at all"
        assert not [entry for entry in added if str(decoy) in entry], (
            f"the look-alike tree under the current directory was bound: {added}. "
            "The root was resolved against the cwd rather than the checkout"
        )
        expected = launch_surface.find_repo_root(_TOOLS) / _REPO_RELATIVE_ROOT
        for entry in added:
            assert str(expected) in entry, (
                f"{entry!r} is not under the checkout's {expected} -- the anchor is "
                "neither the cwd nor the repository root"
            )

    def test_the_shipped_profile_binds_from_an_unrelated_directory(
        self, bind, tmp_path, monkeypatch
    ):
        """End to end on the value that actually ships, so a correct resolver paired
        with a stale profile string still fails."""
        profile = dispatch_parity._load_profile(str(_SHIPPED_PROFILE))
        root = profile["provider_root"]
        assert not os.path.isabs(root), (
            f"the shipped profile names an absolute provider root ({root!r}); it "
            "would only resolve on the machine that wrote it"
        )
        monkeypatch.chdir(tmp_path)

        added = bind(root)

        assert added, (
            f"the shipped provider_root {root!r} did not bind from an unrelated "
            "directory"
        )

    def test_a_relative_root_that_is_absent_names_the_checkout_it_looked_under(
        self, bind, tmp_path, monkeypatch
    ):
        """The failure has to say where it looked, or a mis-anchored root reads as a
        missing checkout."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(dispatch_parity.ParityError) as excinfo:
            bind("no/such/provider")
        message = str(excinfo.value)
        assert str(launch_surface.find_repo_root(_TOOLS)) in message, message
        assert (
            str(tmp_path) not in message
        ), f"the message names the current directory: {message}"

    def test_an_absolute_root_is_still_taken_verbatim(
        self, bind, tmp_path, monkeypatch
    ):
        """Repository-relative resolution is for relative values only; an absolute
        root may legitimately point outside the checkout."""
        provider = _make_provider(tmp_path / "elsewhere")
        monkeypatch.chdir(tmp_path)
        added = bind(str(provider))
        assert [entry for entry in added if str(provider) in entry], added

    def test_a_user_relative_root_is_still_expanded(self, bind, tmp_path, monkeypatch):
        """``~`` expands to an absolute path, so it must not fall into the
        repository-relative branch and be looked for inside the checkout."""
        provider = _make_provider(tmp_path / "home" / "provider")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        added = bind(os.path.join("~", "provider"))
        assert [entry for entry in added if str(provider) in entry], added

    def test_the_environment_is_the_fallback_when_the_profile_names_none(
        self, bind, tmp_path, monkeypatch
    ):
        provider = _make_provider(tmp_path / "from_env")
        monkeypatch.setenv("ROCKE_PROVIDER_ROOT", str(provider))
        added = bind(None)
        assert [entry for entry in added if str(provider) in entry], added

    def test_a_nonempty_profile_value_still_outranks_the_environment(
        self, bind, tmp_path, monkeypatch
    ):
        chosen = _make_provider(tmp_path / "from_profile")
        ignored = _make_provider(tmp_path / "from_env")
        monkeypatch.setenv("ROCKE_PROVIDER_ROOT", str(ignored))
        added = bind(str(chosen))
        assert [entry for entry in added if str(chosen) in entry], added
        assert not [entry for entry in added if str(ignored) in entry], added

    def test_the_sibling_tools_share_this_one_binding(self):
        """``knob_sweep`` and ``reconcile_applicability`` read the same ``provider_root``
        out of the same profiles. They must reach it through THIS function rather than
        resolving a root of their own, or the fix above holds for one tool only."""
        import knob_sweep
        import reconcile_applicability

        assert knob_sweep._bind_provider is dispatch_parity._bind_provider
        assert reconcile_applicability._bind_provider is dispatch_parity._bind_provider


class TestTheShippedProfilePinsTheDispatchArmItsCatalogWasBuiltFrom:
    """The request defaults decide which kernel the dispatcher resolves, and this
    catalog contains one arm of that choice only."""

    def test_dense_persistent_is_the_string_off_not_a_yaml_boolean(self):
        defaults = dispatch_parity._load_profile(str(_SHIPPED_PROFILE))["request"][
            "defaults"
        ]
        assert "dense_persistent" in defaults, (
            "the profile leaves dense_persistent unset, so AttentionRequest defaults "
            "it to 'auto' and the dispatcher resolves the persistent arm once "
            "work >= dense_num_persistent -- a kernel this catalog does not ship"
        )
        value = defaults["dense_persistent"]
        assert isinstance(value, str), (
            f"dense_persistent parsed as {type(value).__name__} ({value!r}): the key "
            "was written unquoted and PyYAML read `off` as a boolean. The dispatcher "
            "calls .strip().lower() on it, so the tool aborts rather than pinning "
            "the non-persistent arm"
        )
        assert value == "off", value

    @staticmethod
    def _resolve_with_the_real_dispatcher(monkeypatch, **overrides):
        """B1, Sq=Skv=8192, Hq=Hkv=8, D=128, bf16, causal through the dispatcher and
        request class the shipped profile binds, on its own ``request.defaults``.

        At that shape ``work = 32 * 8 * 1 = 256 = dense_num_persistent``, so the
        unpinned ``auto`` arm resolves persistent and, at D=128 causal bf16, wide DMA
        with it: the one shape where the pin is the whole difference.
        """
        import importlib

        profile = _real_dispatcher_or_skip(monkeypatch)
        dispatch, request = profile["dispatch"], profile["request"]
        try:
            factory_module = importlib.import_module(dispatch["module"])
            request_module = importlib.import_module(request["module"])
        except ImportError as exc:
            pytest.skip(
                f"the rocKE library cannot be imported here ({exc}) -- run with an "
                "interpreter that has its dependencies, e.g. <build-dir>/dnn-providers/"
                "hip-kernel-provider/descriptor-packaging/hkp-rocke-venv/bin/python"
            )
        factory = getattr(factory_module, dispatch["function"])
        request_cls = getattr(request_module, request["class"])
        fields = {
            **request["defaults"],
            "batch": 1,
            "seqlen_q": 8192,
            "seqlen_k": 8192,
            "nhead_q": 8,
            "nhead_k": 8,
            "hdim_q": 128,
            "hdim_v": 128,
            "dtype": "bf16",
            "mask_type": 1,
            **overrides,
        }
        result = factory(request_cls(**fields))
        attribute = dispatch.get("spec_attribute")
        return getattr(result, attribute) if attribute else result

    def test_the_real_dispatcher_resolves_the_arm_the_catalog_ships(self, monkeypatch):
        """Checking the YAML value is only half of it: this is what the dispatcher does
        with that value, so a profile that parses correctly but no longer reaches the
        non-persistent eight-argument kernel still fails."""
        spec = self._resolve_with_the_real_dispatcher(monkeypatch)
        assert spec.persistent is False, (
            "the shipped profile resolves the persistent arm at B1/Sq8192/H8/D128 -- "
            "a kernel with a different argument contract that this catalog does not "
            "ship"
        )
        assert spec.wide_lds_dma is False, spec

    def test_the_unpinned_control_does_resolve_the_persistent_arm(self, monkeypatch):
        """A control: without the pin this shape IS persistent, so the case above
        exercises the pin rather than a shape that is never persistent."""
        spec = self._resolve_with_the_real_dispatcher(
            monkeypatch, dense_persistent="auto"
        )
        assert spec.persistent is True, spec


def _real_dispatcher_or_skip(monkeypatch) -> dict:
    """The shipped gfx950 profile, bound, or a skip when the rocKE tree is absent
    or cannot import. The root is found from this file's location (a checkout or
    a `git archive` extract alike), so a skip here means the tree really lacks it."""
    import importlib

    profile = dispatch_parity._load_profile(str(_SHIPPED_PROFILE))
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        dispatch_parity._bind_provider(profile["provider_root"])
    except dispatch_parity.ParityError as exc:
        pytest.skip(f"the rocKE tree is not present in this tree ({exc})")
    try:
        importlib.import_module(profile["dispatch"]["module"])
    except ImportError as exc:
        pytest.skip(f"the rocKE library cannot be imported here ({exc})")
    return profile


_D128 = dict(nhead_q=8, nhead_k=8, hdim_q=128, hdim_v=128, dtype="bf16", mask_type=1)


class TestTheRealDispatcherBoundary:
    """On the shipped gfx950 profile: the spec's own ValueError refusal is a decline
    (what `_make_gfx950_attention_dense_candidate().support` returns), and the
    runtime-shape fields collapse onto the canonical B1/Sq512/Skv512 entry."""

    def _run(self, monkeypatch, tmp_path, shapes, *extra):
        _real_dispatcher_or_skip(monkeypatch)
        path = tmp_path / "shapes.json"
        path.write_text(json.dumps(shapes))
        argv = ["--profile", str(_SHIPPED_PROFILE), "--shapes", str(path), *extra]
        return dispatch_parity.main(argv)

    def test_a_shape_only_the_bm128_variant_serves_is_servable(
        self, monkeypatch, tmp_path, capsys
    ):
        """Sq=128 does not divide the auto variant's block_m=256, but the bm128
        sibling variant serves it and the catalog ships bm128 tiles. Asking only
        the auto variant's factory reported it as a gap the engine does not have."""
        shapes = [
            {**_D128, "batch": 1, "seqlen_q": 128, "seqlen_k": 256, "mask_type": 0}
        ]
        out = tmp_path / "config.yaml"
        rc = self._run(monkeypatch, tmp_path, shapes, "--out", str(out))
        printed = capsys.readouterr()
        assert rc == 0, printed.err
        assert "servable          1" in printed.out

        profile = dispatch_parity._load_profile(str(_SHIPPED_PROFILE))
        [resolution] = dispatch_parity.resolve_shapes(shapes, profile)
        assert resolution.spec.block_m == 128, resolution.spec

    def test_head_size_192_is_a_refusal_not_an_abort(
        self, monkeypatch, tmp_path, capsys
    ):
        shapes = [
            {**_D128, "batch": 1, "seqlen_q": 512, "seqlen_k": 512},
            {
                **_D128,
                "batch": 1,
                "seqlen_q": 512,
                "seqlen_k": 512,
                "hdim_q": 192,
                "hdim_v": 192,
            },
        ]
        assert self._run(monkeypatch, tmp_path, shapes, "--report-gaps") == 0
        out = capsys.readouterr().out
        assert "[refused]" in out and "head_size must be 64 or 128, got 192" in out

    _RUNTIME_ONLY = [
        {**_D128, "batch": 1, "seqlen_q": 512, "seqlen_k": 512},
        {**_D128, "batch": 2, "seqlen_q": 1024, "seqlen_k": 1024},
        {**_D128, "batch": 4, "seqlen_q": 4096, "seqlen_k": 4096},
        # Ragged self-attention bakes its length: kept at its own values.
        {**_D128, "batch": 1, "seqlen_q": 300, "seqlen_k": 300},
    ]

    def test_shapes_differing_only_in_runtime_fields_are_one_entry(
        self, monkeypatch, tmp_path, capsys
    ):
        out = tmp_path / "config.yaml"
        rc = self._run(monkeypatch, tmp_path, self._RUNTIME_ONLY, "--out", str(out))
        assert rc == 0
        printed = capsys.readouterr().out
        assert "2 kernels for 4 servable shapes" in printed, printed
        assert "spec bakes its shape" in printed

        profile = dispatch_parity._load_profile(str(_SHIPPED_PROFILE))
        catalog, _ = dispatch_parity.canonicalise(
            dispatch_parity.resolve_shapes(self._RUNTIME_ONLY, profile), profile
        )
        kernels = dispatch_parity.build_config(catalog, profile)["packs"][0]["kernels"]
        shapes = sorted(
            (k["metadata"]["batch"], k["metadata"]["seqlen_q"], k["metadata"]["ragged"])
            for k in kernels
        )
        assert shapes == [(1, 300, 1), (1, 512, 0)], shapes

    def test_per_shape_keeps_one_kernel_per_shape(self, monkeypatch, tmp_path, capsys):
        out = tmp_path / "config.yaml"
        rc = self._run(
            monkeypatch, tmp_path, self._RUNTIME_ONLY, "--out", str(out), "--per-shape"
        )
        assert rc == 0
        assert "4 kernels for 4 servable shapes" in capsys.readouterr().out


class TestEmittedNamesDoNotDependOnShapeOrder:
    """Names land in the descriptors and the kpack. A name carrying the shape's
    position in the corpus churns both whenever the corpus is re-mined in another
    order (942:S3-1)."""

    _SHAPES = [
        {**_D128, "batch": 1, "seqlen_q": 512, "seqlen_k": 512},
        {**_D128, "batch": 2, "seqlen_q": 1024, "seqlen_k": 1024, "mask_type": 0},
        {**_D128, "batch": 1, "seqlen_q": 4096, "seqlen_k": 4096, "nhead_k": 2},
        {**_D128, "batch": 1, "seqlen_q": 300, "seqlen_k": 300},
        {**_D128, "batch": 1, "seqlen_q": 128, "seqlen_k": 256, "mask_type": 0},
    ]

    def _emit(self, monkeypatch, tmp_path, shapes, *extra) -> str:
        _real_dispatcher_or_skip(monkeypatch)
        path = tmp_path / "shapes.json"
        path.write_text(json.dumps(shapes))
        out = tmp_path / "config.yaml"
        argv = ["--profile", str(_SHIPPED_PROFILE), "--shapes", str(path)]
        assert dispatch_parity.main([*argv, "--out", str(out), *extra]) == 0
        return out.read_text()

    @pytest.mark.parametrize("extra", [(), ("--per-shape",)])
    def test_a_reordered_corpus_emits_the_identical_config(
        self, monkeypatch, tmp_path, extra
    ):
        forward = self._emit(monkeypatch, tmp_path, self._SHAPES, *extra)
        backward = self._emit(monkeypatch, tmp_path, self._SHAPES[::-1], *extra)
        assert forward == backward

    def test_names_are_the_catalog_key_not_a_position(self, monkeypatch, tmp_path):
        _real_dispatcher_or_skip(monkeypatch)
        profile = dispatch_parity._load_profile(str(_SHIPPED_PROFILE))
        config = dispatch_parity.build_config(
            dispatch_parity.resolve_shapes(self._SHAPES[:1], profile), profile
        )
        [kernel] = config["packs"][0]["kernels"]
        assert kernel["name"] == (
            "gfx950_attention_dense_dtBF16_hs128_nqh8_nkh8_ca1_ra0_sw0_ba1_sq512_sk512"
            "_bm256_bn64"
        )


class TestTheGraphContractAppliesToEverySource:
    """The gfx950 engine's graph_match declines sinks, so no catalog entry may carry
    use_sinks=true, even though the gfx950 kernel and dispatcher serve sinks. A
    rocKE trace states sinks as a request field, with no `_graph_features`, and
    used to reach the catalog as a second kernel under the same catalog key."""

    _TRACE = (
        Path(__file__).resolve().parents[5]
        / "dnn-providers/hip-kernel-provider/rocke/library/benchmarks/gfx950"
        / "attention/prefill/gpt_oss_sink_prefill_shapes.json"
    )

    def test_a_real_sink_prefill_trace_is_out_of_contract(self, monkeypatch, tmp_path):
        import mine_shapes

        profile = _real_dispatcher_or_skip(monkeypatch)
        record = json.loads(self._TRACE.read_text().splitlines()[0])
        assert record["variant"] == "full_sink_prefill_s512" and record["has_sinks"]
        tree = tmp_path / "bench"
        tree.mkdir()
        (tree / "sink_shapes.json").write_text(json.dumps(record) + "\n")
        [shape] = mine_shapes.from_rocke_bench(tree, "bf16")
        assert shape["use_sinks"] is True

        sink, control = dispatch_parity.resolve_shapes(
            [shape, {**shape, "use_sinks": False}], profile
        )
        assert sink.kind == "out_of_contract", sink
        assert "sink_token" in sink.reason
        assert control.spec is not None, "the sinkless twin is still served"


#: A builder whose policy-owned field follows the tile, as gfx942's v_row_pad
#: follows block_n, and whose spec refuses a tile that does not divide the length.
_KNOB_STUB = """
import dataclasses


@dataclasses.dataclass(frozen=True)
class Spec:
    seqlen_q: int
    head_size: int
    block_n: int = 64
    v_row_pad: object = None

    def __post_init__(self):
        if self.seqlen_q % self.block_n:
            raise ValueError(f"seqlen_q must be a multiple of block_n={self.block_n}")

    def resolved_v_row_pad(self):
        return v_row_pad(self.block_n)


def v_row_pad(block_n):
    return 32 if block_n == 32 else 0


def supports(spec, arch=None):
    return True, ""
"""


class TestKnobPinsComeBeforeWhatDependsOnThem:
    """942:S6-16: `--knobs` pinned the tile after the policy-owned metadata was
    resolved at the dispatcher's tile, so the catalog said v_row_pad 0 where the
    block_n=32 binary is built with 32, and packing refused it. 942:S6-9: an
    approved tile SET could only be written as a cross-product."""

    @staticmethod
    def _build(tmp_path, monkeypatch, knobs):
        (tmp_path / "knob_stub.py").write_text(_KNOB_STUB)
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "knob_stub", raising=False)
        import knob_stub

        fields = ["seqlen_q", "head_size", "block_n", "v_row_pad"]
        profile = {
            "slug": "stub",
            "arch": "gfxstub",
            "source": "stub.py",
            "builder": "build_stub",
            "engine": {"name": "stub:Engine"},
            "kmd_fields": [],
            "metadata_fields": fields,
            "predicate": {"module": "knob_stub", "function": "supports"},
            "arch_spec": {"module": "knob_stub", "class": "Spec"},
            "policies": {
                "v_row_pad": {
                    "module": "knob_stub",
                    "function": "v_row_pad",
                    "args": ["block_n"],
                }
            },
            "specialization": {
                "metadata_fields": fields,
                "matcher_only_fields": [],
                "bindings": {
                    **{f: {"field": f} for f in fields[:3]},
                    "v_row_pad": {"method": "resolved_v_row_pad"},
                },
            },
        }
        served = [
            dispatch_parity.Resolution(
                {"seqlen_q": 512}, spec=knob_stub.Spec(seqlen_q=512, head_size=128)
            )
        ]
        config = dispatch_parity.build_config(served, profile, knobs)
        return [k["metadata"] for k in config["packs"][0]["kernels"]]

    def test_policy_metadata_is_resolved_at_the_pinned_tile(
        self, tmp_path, monkeypatch
    ):
        [metadata] = self._build(tmp_path, monkeypatch, {"block_n": [32]})
        assert metadata["block_n"] == 32
        assert metadata["v_row_pad"] == 32, "resolved at the dispatcher's block_n=64"

    def test_an_explicit_combination_list_emits_exactly_those(
        self, tmp_path, monkeypatch
    ):
        rows = self._build(tmp_path, monkeypatch, [{"block_n": 64}, {"block_n": 32}])
        assert sorted((m["block_n"], m["v_row_pad"]) for m in rows) == [
            (32, 32),
            (64, 0),
        ]

    def test_a_combination_the_builder_refuses_is_dropped_and_reported(
        self, tmp_path, monkeypatch, capsys
    ):
        rows = self._build(tmp_path, monkeypatch, {"block_n": [64, 48]})
        assert [m["block_n"] for m in rows] == [64]
        assert "multiple of block_n=48" in capsys.readouterr().err
