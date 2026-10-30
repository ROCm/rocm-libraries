# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The GEMM routes of hipTensor's kernel pack per target (builders/hiptensor/make_rocke_pack.py):
the body tables follow rocke.dispatch.gemm.cf32 (gfx950: the DMA ring bodies; gfx942: the
register-staged body sets per route, their 64-bit-address bodies and operand splits), the
descriptors of every route x data type x body (non-batched, batched) and the 64-bit-address
body, the route-table entries of each target's body rule, and the committed manifests' GEMM
kernels (body, split mode and launch-limit attributes). No device, nothing lowered."""

import re
from pathlib import Path

import pytest

_LIBROOT = Path(__file__).resolve().parents[1]
_WRITER = _LIBROOT / "builders" / "hiptensor" / "make_rocke_pack.py"
_MANIFESTS = _LIBROOT / "builders" / "hiptensor" / "manifests"


def _load_writer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_hiptensor_make_rocke_pack", _WRITER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def writer():
    return _load_writer()


def test_gemm_tables_are_rockes(writer):
    from rocke.dispatch.gemm import cf32

    assert writer.GEMM_ARCHES == ("gfx950", "gfx942")
    assert tuple(writer.GEMM_BODIES["gfx942"]) == writer.GEMM_ROUTES
    for route in writer.GEMM_ROUTES:
        assert writer.GEMM_BODIES["gfx942"][route] == cf32.route_bodies("gfx942", route)
        for arch in writer.GEMM_ARCHES:
            kind = "h16" if route == "h16" else "f32"
            assert writer.GEMM_WIDE_BODY[arch][kind] == cf32.wide_body(arch, route)
            assert writer.gemm_split_mode(arch, route) == (
                "rne" if route == "h16" else cf32.split_mode(arch, route)
            )
        assert set(writer.GEMM_BODIES["gfx950"][route]) <= set(
            cf32.CF32_BODIES["gfx950"]
        )


def test_gfx942_gemm_descriptors(writer):
    descs = [p for f, p in writer.kernel_descriptors("gfx942", 304) if f == "gemm"]
    want = []
    for route in writer.GEMM_ROUTES:
        for dt in ("fp16", "bf16") if route == "h16" else ("bf16",):
            for body in writer.GEMM_BODIES["gfx942"][route]:
                for batched in (False, True):
                    want.append(dict(route=route, dtype=dt, body=body, batched=batched))
            want.append(
                dict(
                    route=route,
                    dtype=dt,
                    body=writer._wide_body("gfx942", route),
                    batched=False,
                    wide=True,
                )
            )
    assert descs == want
    assert len(descs) == 150
    # every descriptor makes a valid spec (prologue and split mode attached)
    for p in descs:
        spec, why = writer._gemm_spec(p, "gfx942")
        assert spec is not None, (p, why)
        assert spec.trait.vgpr_prefetch == 4 and spec.trait.wide_addr == bool(
            p.get("wide")
        )
        pro = spec._prologue
        assert getattr(pro, "split_mode", "rne") == writer.gemm_split_mode(
            "gfx942", p["route"]
        )


def test_gemm_route_entries(writer):
    from rocke.dispatch.gemm import cf32

    r = writer.routes_for("gfx942", 304)
    gemm = {k: v for k, v in r.items() if k.startswith("gemm.")}
    assert gemm["gemm.rule"] == "cost"
    assert gemm["gemm.x6_admits_gett32"] == "1"
    assert (
        float(gemm["gemm.x6_gett32_first.x6"]) == cf32.CF32_NATIVE_FIRST["gfx942"]["x6"]
    )
    assert (
        float(gemm["gemm.x6_gett32_first.cf32"])
        == cf32.CF32_NATIVE_FIRST["gfx942"]["cf32"]
    )
    n = 0
    for route in writer.GEMM_ROUTES:
        for b in writer.GEMM_BODIES["gfx942"][route]:
            a, bb, occ = cf32.gemm_body_coef(b, route)
            key = f"gemm.cost.{route}.{b.lower()}"
            assert float(gemm[key + ".a"]) == a and float(gemm[key + ".b"]) == bb
            assert float(gemm[key + ".occ"]) == occ
            n += 3
    assert len(gemm) == n + 4
    # gfx950: the waves rule's thresholds only (as before)
    r950 = {k for k in writer.routes_for("gfx950", 256) if k.startswith("gemm.")}
    assert r950 == {f"gemm.{k.lower()}" for k in writer.GEMM_ROUTE_KEYS}


def _gemm_lines(arch):
    out = []
    for line in (_MANIFESTS / f"{arch}.manifest").read_text().splitlines():
        if line.startswith("kernel ") and 'family="gemm"' in line:
            attrs = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
            out.append({k: v.strip('"') for k, v in attrs.items()})
    return out


def test_manifest_gemm_kernels(writer):
    from rocke.dispatch.gemm import cf32

    g = _gemm_lines("gfx942")
    assert len(g) == 150
    vp = dict(
        limit_operand_bytes=str(1 << 32),
        limit_c_elements=str(1 << 31),
        limit_tile_operand_bytes=str(0x7FFF0000),
        limit_c_tile_elements=str(1 << 31),
        limit_grid_yz="65535",
    )
    seen = set()
    for k in g:
        route = k["route"]
        assert k["body"] in writer.GEMM_BODIES["gfx942"][route]
        assert k["split_mode"] == (
            "none" if route == "h16" else cf32.split_mode("gfx942", route)
        )
        assert all(k[a] == v for a, v in vp.items()), k
        assert k["streamk"] == "0" and k["persistent_ctas"] == "0"
        seen.add((route, k["dtype"], k["body"], k["batched"], k["wide"]))
    assert len(seen) == 150
    for k in _gemm_lines("gfx950"):
        assert "split_mode" not in k
        assert k["limit_operand_bytes"] == str(0x7FFF0000)
        assert k["limit_c_elements"] == str((1 << 31) - 1)


def test_parity_family_builds_the_pack_bodies(writer):
    """The gemm_cf32 parity family (platform/tests/instances/parity/gemm_cf32_emit.py) byte-compares
    the bodies the gfx942 pack builds: its body sets and 64-bit-address bodies are the writer's
    (the C++ pack builder's own copy is checked by the C++ pack == Python pack cross-check).
    """
    import importlib.util

    path = (
        _LIBROOT.parent
        / "platform"
        / "tests"
        / "instances"
        / "parity"
        / "gemm_cf32_emit.py"
    )
    spec = importlib.util.spec_from_file_location("_gemm_cf32_emit", path)
    emit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emit)
    for route in ("h16", "x6", "cf32"):
        assert emit._G942[route] == writer.GEMM_BODIES["gfx942"][route], route
        assert emit._G942_WIDE[route] == writer._wide_body("gfx942", route), route
    pk = {
        (c["name"], c["body"], c["dtype"], bool(c.get("batched")), bool(c.get("wide")))
        for c in emit.CONFIGS
        if c.get("arch") == "gfx942" and c["name"].startswith("pk942_")
    }
    want = set()
    for route in ("h16", "x6", "cf32"):
        for dt in ("fp16", "bf16") if route == "h16" else ("bf16",):
            for b in writer.GEMM_BODIES["gfx942"][route]:
                for batched in (False, True):
                    want.add((f"pk942_{route}_{b.lower()}", b, dt, batched, False))
            w = writer._wide_body("gfx942", route)
            want.add((f"pk942_{route}_{w.lower()}_w64", w, dt, False, True))
    assert pk == want
