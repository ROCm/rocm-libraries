"""The Python tier algebra over the shared generic target table."""

import json
from pathlib import Path

import pytest

from hkp_pack import generic_targets
from hkp_pack.errors import HkpPackError
from hkp_pack.generic_targets import GenericTargets

pytestmark = pytest.mark.quick

_SDK_DIR = Path(__file__).resolve().parents[4] / "projects" / "hipdnn" / "plugin_sdk"
_TABLE = _SDK_DIR / "data" / "gpu_generic_targets.json"

_TIER_NAMES = {
    generic_targets.TIER_EXPLICIT: "explicit",
    generic_targets.TIER_GENERIC: "generic",
    generic_targets.TIER_UNRESTRICTED: "unrestricted",
    None: "none",
}


@pytest.fixture(scope="module")
def table():
    return GenericTargets.load(_TABLE)


@pytest.mark.parametrize(
    "doc",
    [
        {"schemaVersion": 1},
        {"schemaVersion": 2, "generics": {"gfx11-generic": ["gfx1100"]}},
        {"generics": []},
        {"generics": {"gfx11-generic": 5}},
        [],
    ],
)
def test_rejects_a_malformed_table(tmp_path, doc):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(HkpPackError):
        GenericTargets.load(path)


@pytest.mark.parametrize(
    "arch, device, expect",
    [
        ([], "gfx942:sramecc+:xnack-", "unrestricted"),
        (["gfx1151", "gfx11-generic"], "gfx1151:sramecc+:xnack-", "explicit"),
        (["gfx1151", "gfx11-generic"], "gfx1100", "generic"),
        (["gfx11-generic"], "gfx1151:sramecc+", "generic"),
        (["gfx11-generic"], "gfx1154", "none"),
        (["gfx12-generic"], "gfx1100", "none"),
        (["gfx1250"], "gfx1250-strict", "none"),
        (["gfx1250-strict"], "gfx1250-strict", "explicit"),
        (["gfx94"], "gfx942", "none"),
        (["gfx9-4-generic", "gfx942"], "gfx942", "explicit"),
    ],
)
def test_list_tier(table, arch, device, expect):
    assert _TIER_NAMES[generic_targets.list_tier(arch, device, table)] == expect


@pytest.mark.parametrize(
    "outer, inner, expect",
    [
        ([], ["gfx942"], True),
        (["gfx942"], [], True),
        (["gfx942", "gfx950"], ["gfx942"], True),
        (["gfx942"], ["gfx942", "gfx950"], False),
        (["gfx11-generic"], ["gfx1100", "gfx1153"], True),
        (["gfx11-generic"], ["gfx1154"], False),
        (["gfx1151"], ["gfx11-generic"], False),
        (["gfx11-generic"], ["gfx11-generic"], True),
        (["gfx11-generic"], ["gfx12-generic"], False),
        (["gfx9-4-generic"], ["gfx1151"], False),
        (["gfx942", "gfx11-generic"], ["gfx942", "gfx1100"], True),
        (["gfx942", "gfx11-generic"], ["gfx950", "gfx1100"], False),
    ],
)
def test_covers(table, outer, inner, expect):
    assert generic_targets.covers(outer, inner, table) is expect


@pytest.mark.parametrize(
    "a, b, expect",
    [
        ([], [], True),
        ([], ["gfx942"], False),
        (["gfx942"], ["gfx942"], True),
        (["gfx942"], ["gfx950"], False),
        (["gfx11-generic"], ["gfx1151"], False),
        (["gfx11-generic"], [], False),
        (["gfx11-generic"], ["gfx12-generic"], False),
        (["gfx11-generic"], ["gfx11-generic"], True),
        (["gfx1151", "gfx11-generic"], ["gfx11-generic"], True),
        (["gfx11-generic"], ["gfx11-generic", "gfx942"], True),
        (["gfx1151"], ["gfx11-generic", "gfx1151"], True),
        (["gfx1250"], ["gfx1250-strict"], False),
    ],
)
def test_compete(table, a, b, expect):
    assert generic_targets.compete(a, b, table) is expect
    assert generic_targets.compete(b, a, table) is expect
