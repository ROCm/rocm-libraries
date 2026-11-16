# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""gfx942 and gfx90a: one arch, two builder families (16-bit rows RDNA-family over the MFMA16 layer, fp32 rows CDNA-family).

The pieces that make it one manifest: the instance table (``_instances``) spells each row by its family, the S1 tables tool renders
each row with its family's columns, the merge module applies the R-MIR verdict to both families' rows, and ``gen_descriptors``
accepts a manifest that holds both ways of carrying the dBias reduce kernel (a 16-bit companion, an fp32 combined object).
Host-only: nothing compiles.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "tools"))
import gen_descriptors as gd  # noqa: E402
import sdpa_bwd_tables_from_sweep as tables_tool  # noqa: E402
from generators import _instances  # noqa: E402
from generators import _sdpa_bwd_cdna_build as cdna  # noqa: E402
from generators import _sdpa_bwd_rdna_build as rdna  # noqa: E402
from generators._sdpa_bwd_spec import GENERIC_KERNELS, Spec, to_spec  # noqa: E402

# One of each kind of row the arch holds: prep, dK/dV, dQ (plain, dB-storing bias) of both families, a wide 16-bit rung with a
# spelled knob and the other grid order, the 512 half-head fp32 rung, and the standalone reduce companions of the 16-bit dtypes.
KEYS = {
    "gfx90a": [
        "prep,f16,128,plain,bk,0,1,tail,0,1;arch={a}",
        "prep,fp32,128,plain,bk,0,1,tail,0,1;arch={a}",
        "dkdv,f16,128,plain,bk,0,1,g1,0,1;arch={a}",
        "dkdv,f16,64,plain,bk,0,1,g1,0,1;arch={a};grid_order={other}",
        "dkdv,bf16,512,causal,bk,1,1,g1,0,1;arch={a};transposed_source=derived",
        "dkdv,fp32,512,plain,bk,0,1,g1,0,1;arch={a};block_q=16",
        "dq,f16,128,plain,bk,0,1,g1,0,1;arch={a}",
        "dq,f16,128,bias,bk,0,1,g1,0,1;arch={a}",
        "dq,fp32,128,bias,bk,0,1,g1,0,1;arch={a};block_m=32;block_n=16",
        "dbias_reduce,f16,32,plain,bk,0,0,tail,0,1;arch={a}",
        "dbias_reduce,bf16,32,plain,bk,0,0,tail,0,1;arch={a}",
    ],
}
KEYS["gfx942"] = KEYS["gfx90a"]
ARCHES = sorted(KEYS)
# the grid order a Spec may spell: the one that is not the arch's default
OTHER_GRID_ORDER = {"gfx942": "kv_fastest", "gfx90a": "head_fastest"}


def _specs(arch: str) -> list[Spec]:
    return [
        Spec.parse(k.format(a=arch, other=OTHER_GRID_ORDER[arch])) for k in KEYS[arch]
    ]


def _table_text(arch: str) -> str:
    pins = {s.hash(): {"spec": s.key(), "hash": s.hash()} for s in _specs(arch)}
    return tables_tool.render_tables(arch, pins)


@pytest.fixture
def populated(monkeypatch):
    """The instance table of an arch holding ``KEYS``, as the S1 tables tool writes it."""

    def install(arch: str) -> None:
        scope: dict = {}
        exec(
            _table_text(arch).split("# fmt: off\n")[1].split("# fmt: on")[0], scope
        )  # noqa: S102
        monkeypatch.setattr(
            _instances,
            f"SDPA_BWD_{arch.upper()}_TABLES",
            scope[f"SDPA_BWD_{arch.upper()}_TABLES"],
        )

    return install


def test_the_instance_columns_agree_with_the_spec_grammar() -> None:
    from generators._sdpa_bwd_spec import KNOBS_CDNA9  # noqa: PLC0415

    assert _instances.SDPA_BWD_CDNA9_KNOBS == {
        kernel: KNOBS_CDNA9[kernel] for kernel in GENERIC_KERNELS
    }


@pytest.mark.parametrize("arch", ARCHES)
def test_a_table_rendered_by_the_tool_gives_back_the_specs_through_both_builders(
    arch: str, populated
) -> None:
    """Spec -> table row -> instance row -> Spec: the 16-bit rows through the RDNA builder, the fp32 rows through the CDNA builder."""
    populated(arch)
    wanted = {s.key() for s in _specs(arch) if s.kernel != "dbias_reduce"}
    got = set()
    for kind in ("prep", "dkdv", "dq"):
        got |= {s.key() for s in rdna.specs_for(kind, arch)}
        _, specs = cdna.rows_of(f"sdpa_bwd_{kind}", arch)
        got |= {s.key() for s in specs}
    assert got == wanted
    assert {s.key() for s in rdna.specs_for("dbias_reduce", arch)} == {
        s.key() for s in _specs(arch) if s.kernel == "dbias_reduce"
    }
    # every table row is an instance of its op, and the two builders partition them
    for kind in ("prep", "dkdv", "dq"):
        rows = _instances.instances_for(f"sdpa_bwd_{kind}", arch)
        assert len(rows) == len(rdna.rows_of(f"sdpa_bwd_{kind}", arch)[0]) + len(
            cdna.rows_of(f"sdpa_bwd_{kind}", arch)[0]
        )


@pytest.mark.parametrize("arch", ARCHES)
def test_the_rows_of_both_families_have_the_field_set_of_the_gfx950_rows_of_their_op(
    arch: str, populated
) -> None:
    """The KMD of an op is one for every arch: same fields, same order, whichever builder spells the row."""
    populated(arch)
    for kind in ("prep", "dkdv", "dq"):
        op = f"sdpa_bwd_{kind}"
        gfx950 = list(_instances.instances_for(op, "gfx950")[0].knobs)
        rows = _instances.instances_for(op, arch)
        assert {r.dtype == "fp32" for r in rows} == {False, True}
        for row in rows:
            assert list(row.knobs) == gfx950, row.name


@pytest.mark.parametrize("arch", ARCHES)
def test_the_reduce_kernel_of_a_bias_dq_row_is_its_families(
    arch: str, populated
) -> None:
    """A 16-bit dB-storing bias dQ row names the standalone reduce kernel of its companion; an fp32 one the symbol inside its object."""
    populated(arch)
    by_key = {
        to_spec_key(r, arch): r for r in _instances.instances_for("sdpa_bwd_dq", arch)
    }
    f16 = by_key[f"dq,f16,128,bias,bk,0,1,g1,0,1;arch={arch}"]
    fp32 = by_key[f"dq,fp32,128,bias,bk,0,1,g1,0,1;arch={arch};block_m=32;block_n=16"]
    plain = by_key[f"dq,f16,128,plain,bk,0,1,g1,0,1;arch={arch}"]
    assert f16.knobs["reduce_symbol"] == _instances.SDPA_BWD_GFX11_REDUCE_SYMBOL["f16"]
    assert fp32.knobs["reduce_symbol"] == _instances.SDPA_BWD_REDUCE_SYMBOL["fp32"]
    assert plain.knobs["reduce_symbol"] == ""
    # and the CDNA builder agrees with the row: the object it builds carries that symbol
    assert cdna.reduce_symbol_of(to_spec(fp32, arch)) == fp32.knobs["reduce_symbol"]


def to_spec_key(row, arch: str) -> str:
    """The key of an instance row of either family (the RDNA rows carry no perf columns, so the key comes from the build's rows)."""
    spec = next(
        (
            s
            for s in (
                *rdna.specs_for(row.op.removeprefix("sdpa_bwd_"), arch),
                *cdna.rows_of(row.op, arch)[1],
            )
            if s.tile == row.knobs["head_dim"]
            and s.dtype == row.dtype
            and (s.cls == "bias") == bool(row.knobs["has_bias"])
            and (s.cls == "causal") == bool(row.knobs["causal"])
            and s.dropout == row.knobs["has_dropout"]
        ),
        None,
    )
    assert spec is not None
    return spec.key()


@pytest.mark.parametrize("arch", ARCHES)
def test_a_smaller_rung_outranks_a_larger_one_in_both_families(
    arch: str, populated
) -> None:
    """One priority formula for the rows of the arch: a padded graph takes the smallest rung that covers it, never the 512 half-head
    object (equal priorities would tie on kernel id)."""
    populated(arch)
    for kind in ("dkdv", "dq"):
        by_dtype: dict[str, dict[int, int]] = {}
        for row in _instances.instances_for(f"sdpa_bwd_{kind}", arch):
            by_dtype.setdefault(row.dtype, {})[row.knobs["head_dim"]] = row.priority
        for dtype in ("f16", "fp32"):
            ranks = by_dtype[dtype]
            tiles = sorted(ranks)
            assert [ranks[t] for t in tiles] == sorted(
                (ranks[t] for t in tiles), reverse=True
            ), (kind, dtype, ranks)
            assert len(set(ranks.values())) == len(ranks)
    fp32 = {
        r.knobs["head_dim"]: r.priority
        for r in _instances.instances_for("sdpa_bwd_dkdv", arch)
        if r.dtype == "fp32"
    }
    assert fp32[512] == _instances.PRIORITY_SPECIALIZED + 1 + (256 - 512) // 16 == 85


def test_the_tool_refuses_a_knob_its_arch_has_no_column_for() -> None:
    """A spelled knob without a column would be dropped from the table and the pin lost: the tool must stop instead."""
    from generators._sdpa_bwd_spec import GENERIC_TABLE_KNOBS  # noqa: PLC0415

    spec = Spec.parse(
        "dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx90a;grid_order=head_fastest"
    )
    with pytest.raises(SystemExit, match="grid_order"):
        tables_tool.table_row(spec, GENERIC_TABLE_KNOBS["dkdv"])
    row = tables_tool.table_row(
        spec, tables_tool.ARCH_OUTPUTS["gfx90a"].columns_of(spec)
    )
    assert '"head_fastest"' in row


@pytest.mark.parametrize("arch", ARCHES)
def test_the_tool_renders_each_row_with_the_columns_of_its_family(arch: str) -> None:
    out = tables_tool.ARCH_OUTPUTS[arch]
    for spec in _specs(arch):
        columns = out.columns_of(spec)
        expected = (
            _instances.SDPA_BWD_KNOBS
            if spec.dtype == "fp32" and spec.kernel != "dbias_reduce"
            else _instances.SDPA_BWD_CDNA9_KNOBS
        )[spec.kernel]
        assert columns == expected, spec.key()


def test_the_gfx11_and_gfx950_tables_keep_their_columns() -> None:
    """The mixed-arch columns must not leak into the arches whose generated tables are already committed."""
    assert tables_tool.ARCH_OUTPUTS["gfx11-generic"].cdna_columns is None
    assert tables_tool.ARCH_OUTPUTS["gfx950"].cdna_columns is None
    assert "grid_order" not in tables_tool.ARCH_OUTPUTS["gfx11-generic"].columns["dkdv"]


def _record(verdict: str, status: str = "ok") -> dict:
    return {
        "name": "row",
        "build_status": status,
        "gates": {"rmir": {"verdict": verdict}},
    }


def test_the_standalone_pin_of_a_combined_row_is_its_dq_object_alone() -> None:
    spec = Spec.parse(
        "dq,fp32,128,bias,bk,0,1,g1,0,1;arch=gfx90a;block_m=32;block_n=16"
    )
    assert cdna.reduce_symbol_of(spec)
    assert cdna.reduce_symbol_of(spec, standalone=True) == ""
    plain = Spec.parse("dq,fp32,128,plain,bk,0,1,g1,0,1;arch=gfx90a")
    assert (
        cdna.reduce_symbol_of(plain)
        == cdna.reduce_symbol_of(plain, standalone=True)
        == ""
    )
    # a merged object has no standalone form: it always carries its reduce kernel
    merged = Spec.parse("merged,bf16,64,bias,bk,0,0,tail,0,1")
    assert (
        cdna.reduce_symbol_of(merged, standalone=True)
        == cdna.reduce_symbol_of(merged)
        != ""
    )


# --- gen_descriptors: one manifest, two ways to carry the dBias reduce kernel ---


def _manifest(content: Path, rows: list[dict], companions: list[dict]) -> None:
    folder = content / "sdpa_bwd_dq" / "gfx90a"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for entry in (*rows, *companions):
        (folder / entry["file"]).write_bytes(entry["file"].encode())
        entry["sha256"] = hashlib.sha256(entry["file"].encode()).hexdigest()
    manifest = {
        "op": "sdpa_bwd_dq",
        "arch": "gfx90a",
        "toolchain": {"inputs_sha256": "synthetic"},
        "instances": rows,
        "companions": companions,
    }
    (folder / gd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def _row(name: str, dtype: str, reduce_symbol: str, ship: str | None) -> dict:
    return {
        "name": name,
        "file": f"{name}.hsaco",
        "symbol": f"{name}_kernel",
        "dtype": dtype,
        "priority": 100,
        "knobs": {"head_dim": 128, "reduce_symbol": reduce_symbol},
        **({"ship": ship} if ship else {}),
    }


def _companion(dtype: str, symbol: str) -> dict:
    return {
        "hash": f"reduce_{dtype}",
        "spec": f"dbias_reduce,{dtype},32,plain,bk,0,0,tail,0,1;arch=gfx90a",
        "file": f"reduce_{dtype}.hsaco",
        "symbol": symbol,
    }


def _tailzero(dtype: str) -> dict:
    return {
        "hash": f"tz_{dtype}",
        "spec": f"tailzero,{dtype};arch=gfx90a",
        "file": f"tz_{dtype}.hsaco",
        "symbol": "tz_kernel",
    }


@pytest.fixture
def collect(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "inputs_digest", lambda op, arch: "synthetic")
    monkeypatch.setattr(gd, "describe", lambda data, where, symbol=None: None)
    monkeypatch.setattr(gd, "verify_arch", lambda *args: None)
    monkeypatch.setattr(gd, "verify_generic", lambda *args: None)

    def run(rows, companions):
        # every dQ row launches the tail-zero companion of its dtype
        tailzero = [_tailzero(d) for d in sorted({row["dtype"] for row in rows})]
        _manifest(tmp_path, rows, [*companions, *tailzero])
        return gd.collect(tmp_path, "gfx90a")

    return run


def test_a_manifest_may_hold_a_companion_row_and_a_combined_row(collect) -> None:
    entries = collect(
        [
            _row("f16_bias", "f16", "reduce_f16_kernel", None),
            _row("fp32_bias", "fp32", "reduce_f32_kernel_1", "combined"),
            _row("f16_plain", "f16", "", None),
        ],
        [_companion("f16", "reduce_f16_kernel")],
    )
    by_name = {e.record["name"]: e.companions for e in entries}
    assert by_name["f16_bias"][gd.REDUCE_ROLE]["symbol"] == "reduce_f16_kernel"
    # A combined row launches the reduce kernel of its own object, never the standalone companion.
    own = by_name["fp32_bias"][gd.REDUCE_ROLE]
    assert (own["file"], own["symbol"]) == ("fp32_bias.hsaco", "reduce_f32_kernel_1")
    assert gd.REDUCE_ROLE not in by_name["f16_plain"]
    assert all(gd.TAILZERO_ROLE in roles for roles in by_name.values())


def test_a_row_naming_a_reduce_kernel_nothing_holds_is_refused(collect) -> None:
    with pytest.raises(
        ValueError, match="the manifest has no f16 dbias_reduce companion"
    ):
        collect(
            [_row("f16_bias", "f16", "reduce_f16_kernel", None)],
            [_companion("bf16", "reduce_bf16_kernel")],
        )


def test_a_companion_of_another_symbol_is_refused(collect) -> None:
    with pytest.raises(ValueError, match="the f16 companion is"):
        collect(
            [_row("f16_bias", "f16", "reduce_f16_kernel", None)],
            [_companion("f16", "other_kernel")],
        )


# --- the pin driver ---


def test_the_pin_driver_records_one_attempt_per_row_and_stops_on_a_dual_object(
    tmp_path, monkeypatch
) -> None:
    """Every row has ``attempts == 1``; a row whose dump held two objects is recorded as ``dual_object`` with no object, so the S1
    selection's ``expect`` count stops the run; the exit status is non-zero unless every row is ``ok``.
    """
    import csv  # noqa: PLC0415

    import sdpa_bwd_pin_build as pin  # noqa: PLC0415
    from generators import _flydsl_env as env  # noqa: PLC0415
    from generators import _sdpa_bwd_split_build as split  # noqa: PLC0415

    specs = [
        "dkdv,f16,128,plain,bk,0,1,g1,0,1;arch=gfx90a",
        "dq,fp32,128,bias,bk,0,1,g1,0,1;arch=gfx90a;block_m=32;block_n=16",
        "dkdv,fp32,64,plain,bk,0,1,g1,0,1;arch=gfx90a",
    ]
    listing = tmp_path / "specs.txt"
    listing.write_text("# gfx90a pins\n\n" + "\n".join(specs) + "\n")
    resources = {c: 7 for c in pin.RESOURCES}
    outcomes = [
        cdna.Outcome(
            "ok", b"\x7fELF-a", gate={"resources": resources, "rmir": {"verdict": "ok"}}
        ),
        cdna.Outcome(
            "ok",
            b"\x7fELF-b",
            gate={"resources": resources, "rmir": {"verdict": "known"}},
        ),
        cdna.Outcome("dual_object", detail="embedded objects are not identical"),
    ]
    seen = []

    def compile_all(compile_one, tasks, arch, label):
        seen.append((compile_one, tasks, arch))
        return outcomes

    monkeypatch.setenv(
        "FLYDSL_GEN_JOBS", "1"
    )  # the driver sets it; restored with the test
    monkeypatch.setattr(env, "assert_flydsl_version", lambda: None)
    monkeypatch.setattr(env, "prepare", lambda arch: None)
    monkeypatch.setattr(cdna, "_compile_all", compile_all)
    out = tmp_path / "pins"
    assert pin.main([str(listing), str(out), "--workers", "1"]) == 1
    ((fn, tasks, arch),) = seen
    assert (
        fn is split.compile_one and arch == "gfx90a" and tasks == [(k,) for k in specs]
    )
    rows = list(csv.DictReader((out / "manifest.csv").open()))
    assert [r["status"] for r in rows] == ["ok", "ok", "dual_object"]
    assert {r["attempts"] for r in rows} == {"1"}
    assert [r["rmir"] for r in rows] == ["ok", "known", ""]
    assert [bool(r["sha256"]) for r in rows] == [True, True, False]
    assert (out / "objs" / rows[0]["hash"] / "obj.hsaco").read_bytes() == b"\x7fELF-a"
    assert not (out / "objs" / rows[2]["hash"]).exists()
    assert rows[2]["detail"].startswith("embedded objects")


def test_the_pin_driver_needs_canonical_specs_of_one_arch(
    tmp_path, monkeypatch
) -> None:
    import sdpa_bwd_pin_build as pin  # noqa: PLC0415

    monkeypatch.setenv("FLYDSL_GEN_JOBS", "1")

    listing = tmp_path / "specs.txt"
    listing.write_text("dkdv,f16,128,plain,bk,0,1,g1,0,1;arch=gfx90a;arch=gfx90a\n")
    with pytest.raises(ValueError, match="arch is spelled"):
        pin.main([str(listing), str(tmp_path / "o"), "--workers", "1"])
    listing.write_text(
        "dkdv,f16,128,plain,bk,0,1,g1,0,1;arch=gfx90a\ndkdv,f16,128,plain,bk,0,1,g1,0,1;arch=gfx942\n"
    )
    with pytest.raises(SystemExit, match="one arch"):
        pin.main([str(listing), str(tmp_path / "o"), "--workers", "1"])


# --- dK/dV's launch grid follows the Spec's grid order, and the kernel's knob record must agree ---


@pytest.mark.parametrize(
    ("key", "grid"),
    [
        ("dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx942", "kheads_by_kvblocks_by_seqs"),
        (
            "dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx942;grid_order=kv_fastest",
            "kvblocks_by_kheads_by_seqs",
        ),
        ("dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx90a", "kvblocks_by_kheads_by_seqs"),
        (
            "dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx90a;grid_order=head_fastest",
            "kheads_by_kvblocks_by_seqs",
        ),
        (
            "dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx11-generic",
            "kvblocks_by_kheads_by_seqs",
        ),
        ("dq,f16,64,plain,bk,0,1,g1,0,1;arch=gfx942", "qheads_by_qblocks_by_seqs"),
    ],
)
def test_the_grid_rule_of_a_row_follows_its_grid_order(key: str, grid: str) -> None:
    assert rdna._grid_rule(Spec.parse(key)) == grid


def test_a_knob_record_that_launches_the_other_order_is_refused() -> None:
    """The manifest's grid string is the Spec's; a kernel that decodes block ids the other way would read the wrong tiles."""
    from types import SimpleNamespace  # noqa: PLC0415

    from generators._codeobject import GeneratorError  # noqa: PLC0415

    def built(key: str, launched: str | None):
        knobs = dict(num_waves=4, block_q=32, waves_per_eu=0, block_kv=64)
        if launched:
            knobs["grid_order"] = launched
        return SimpleNamespace(spec=Spec.parse(key), knobs=SimpleNamespace(**knobs))

    gfx942 = "dkdv,f16,64,plain,bk,0,1,g1,0,1;arch=gfx942"
    assert (
        rdna.geometry_of(built(gfx942, "head_fastest"))["block_threads"] == 256
    )  # 4 waves of 64 lanes
    with pytest.raises(
        GeneratorError, match="launches dK/dV kv_fastest, the Spec says head_fastest"
    ):
        rdna.geometry_of(built(gfx942, "kv_fastest"))
    # the generic fp32 family's record has no grid order and launches kv_fastest
    gfx11 = "dkdv,fp32,64,plain,bk,0,1,g1,0,1;arch=gfx11-generic"
    assert (
        rdna.geometry_of(built(gfx11, None))["block_threads"] == 128
    )  # 4 waves of 32 lanes


def test_a_manifest_without_companions_keeps_its_reduce_kernels_inside_its_objects(
    collect,
) -> None:
    """gfx950 lists tail-zero companions only: a row naming a reduce symbol is not asked for a reduce companion, it launches the reduce of its own object."""
    (entry,) = collect([_row("bias", "f16", "reduce_kernel_1", None)], [])
    own = entry.companions[gd.REDUCE_ROLE]
    assert (own["file"], own["symbol"]) == ("bias.hsaco", "reduce_kernel_1")


def test_grid_order_defaults_are_the_rdna_family_arches() -> None:
    from generators._sdpa_bwd_spec import GRID_ORDER_DEFAULT, family  # noqa: PLC0415

    assert set(GRID_ORDER_DEFAULT) == {
        "gfx11-generic",
        "gfx12-generic",
        "gfx942",
        "gfx90a",
    }
    assert all(family(a, "f16") == "rdna" for a in GRID_ORDER_DEFAULT)


# --- the merge module: a reason of either builder, or a red R-MIR verdict, stops the run; --only names rows of either family; tail-zero companions ---


class _Rig:
    """``split.generate`` over one 16-bit and one fp32 row of gfx90a with both builders stubbed."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        from generators import _flydsl_env as env  # noqa: PLC0415
        from generators import _sdpa_bwd_split_build as split  # noqa: PLC0415

        self.split = split
        self.tmp = tmp_path
        self.ref = tmp_path / "ref"
        self.ref.mkdir()
        self.s16 = Spec.parse("dkdv,f16,128,plain,bk,0,1,g1,0,1;arch=gfx90a")
        self.s32 = Spec.parse(
            "dkdv,fp32,128,plain,bk,0,1,g1,0,1;arch=gfx90a;block_q=16"
        )
        self.pins = {}
        for spec in (self.s16, self.s32):
            blob = spec.key().encode()
            (self.ref / f"{spec.hash()}.hsaco").write_bytes(blob)
            self.pins[spec.hash()] = {
                "hash": spec.hash(),
                "spec": spec.key(),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        self.i16, self.i32 = SimpleNamespace(name="r16"), SimpleNamespace(name="c32")
        self.verdict16 = "ok"
        self.r_reasons: list[str] = []
        self.c_reasons: list[str] = []
        self.entries: list[dict] = []
        self.listed_companions: list[dict] = []
        self.rows_of_only: dict[str, object] = {}
        self.cdna_kwargs: dict = {}
        self.manifest: list = []

        def rows_of(family):
            inst, spec = (
                (self.i16, self.s16) if family == "rdna" else (self.i32, self.s32)
            )

            def stub(op, arch, only=None):
                self.rows_of_only[family] = only
                keep = only is None or spec.hash() in only
                rows = ([inst], [spec]) if keep else ([], [])
                return (*rows, []) if family == "rdna" else rows

            return stub

        def rdna_build_rows(
            op,
            arch,
            instances,
            specs,
            companions,
            reference,
            op_dir,
            reference_dir,
            nondeterministic_csv,
        ):
            records = [
                {
                    "name": i.name,
                    "build_status": "ok",
                    "spec_hash": s.hash(),
                    "sha256": self.pins[s.hash()]["sha256"],
                    "gates": {"rmir": {"verdict": self.verdict16, "spill_findings": 1}},
                }
                for i, s in zip(instances, specs)
            ]
            return records, self.entries, list(self.r_reasons)

        def cdna_build_rows(
            op, arch, instances, specs, op_dir, reference_dir, reference_csv, **kwargs
        ):
            self.cdna_kwargs = kwargs
            records = [
                {
                    "name": i.name,
                    "build_status": "ok",
                    "spec_hash": s.hash(),
                    "gates": {"rmir": {"verdict": "ok"}},
                }
                for i, s in zip(instances, specs)
            ]
            return (
                records,
                list(self.c_reasons),
                [SimpleNamespace(attempts=1)] * len(records),
            )

        monkeypatch.setattr(env, "rocm_version", lambda: "10.2.0")
        monkeypatch.setattr(env, "assert_flydsl_version", lambda: None)
        monkeypatch.setattr(env, "prepare", lambda arch: None)
        monkeypatch.setattr(env, "provenance", lambda arch: {})
        monkeypatch.setattr(
            split,
            "rdna",
            SimpleNamespace(
                **{
                    **vars(rdna),
                    "rows_of": rows_of("rdna"),
                    "build_rows": rdna_build_rows,
                }
            ),
        )
        monkeypatch.setattr(
            split,
            "cdna",
            SimpleNamespace(
                **{
                    **vars(cdna),
                    "rows_of": rows_of("cdna"),
                    "build_rows": cdna_build_rows,
                    "read_reference": lambda _p: self.pins,
                    "write_companions": lambda d, c: self.listed_companions.extend(c),
                }
            ),
        )
        monkeypatch.setattr(
            split, "instances_for", lambda op, arch: [self.i16, self.i32]
        )
        monkeypatch.setattr(
            split,
            "write_op_manifest",
            lambda d, a, op, prov, records: self.manifest.extend(records),
        )
        monkeypatch.setattr(
            split, "refresh_source_md", lambda d: tmp_path / "SOURCE.md"
        )
        split.cdna._complete_source_md = lambda *a: tmp_path / "SOURCE.md"

    def run(self, only=None, build_tailzero=None) -> int:
        return self.split.generate(
            "sdpa_bwd_dkdv",
            "gfx90a",
            self.tmp / "out",
            only,
            self.ref,
            build_tailzero=build_tailzero,
        )


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return _Rig(tmp_path, monkeypatch)


def test_the_merge_module_ships_both_families_rows_in_table_order(rig) -> None:
    assert rig.run() == 0
    assert [r["name"] for r in rig.manifest] == ["r16", "c32"]
    # an unstable CDNA-family build is rebuilt, as on a shipped gfx950 op
    assert rig.cdna_kwargs["rebuild_unstable"] is True


@pytest.mark.parametrize("family", ["r", "c"])
def test_a_reason_of_either_builder_stops_the_run_and_ships_nothing(
    rig, family: str
) -> None:
    from generators._codeobject import GeneratorError  # noqa: PLC0415

    getattr(rig, f"{family}_reasons").append(f"{family}: the build is not its pin")
    with pytest.raises(GeneratorError, match=f"{family}: the build is not its pin"):
        rig.run()
    assert not rig.manifest
    assert not list((rig.tmp / "out" / "sdpa_bwd_dkdv" / "gfx90a").iterdir())


def test_a_pin_object_that_is_not_its_reference_row_is_refused_before_any_build(
    rig,
) -> None:
    from generators._codeobject import GeneratorError  # noqa: PLC0415

    (rig.ref / f"{rig.s32.hash()}.hsaco").write_bytes(b"other")
    with pytest.raises(GeneratorError, match="is not the pin its reference row names"):
        rig.run()


def test_a_red_r_mir_verdict_of_a_16_bit_row_stops_the_run(rig) -> None:
    from generators._codeobject import GeneratorError  # noqa: PLC0415

    rig.verdict16 = "known"
    assert rig.run() == 0
    rig.manifest.clear()
    rig.verdict16 = "red"
    with pytest.raises(GeneratorError, match="R-MIR spill finding"):
        rig.run()
    assert not rig.manifest


def test_only_may_name_rows_of_either_family(rig) -> None:
    from generators._codeobject import GeneratorError  # noqa: PLC0415

    assert rig.run({rig.s32.hash()}) == 0
    assert rig.rows_of_only["rdna"] == set() and rig.rows_of_only["cdna"] == {
        rig.s32.hash()
    }
    assert [r["name"] for r in rig.manifest] == ["c32"]
    rig.manifest.clear()
    assert rig.run({rig.s16.hash(), rig.s32.hash()}) == 0
    assert [r["name"] for r in rig.manifest] == ["r16", "c32"]
    with pytest.raises(GeneratorError, match="no row with hash"):
        rig.run({"0123456789ab"})


def test_a_red_r_mir_verdict_of_a_companion_stops_a_gated_arch(tmp_path) -> None:
    from generators import _codeobject  # noqa: PLC0415

    spec = Spec.parse("dbias_reduce,f16,32,plain,bk,0,0,tail,0,1;arch=gfx90a")
    pin = {"hash": spec.hash(), "sha256": "0" * 64}
    red = cdna.Outcome("ok", b"x", gate={"rmir": {"verdict": "red"}})
    with pytest.raises(_codeobject.GeneratorError, match="R-MIR spill finding"):
        rdna._companion(spec, red, pin, tmp_path)
    # the verdict is the gate's on gfx942 and gfx90a only: a generic arch has no gate on its companions
    generic = Spec.parse("dbias_reduce,f16,32,plain,bk,0,0,tail,0,1;arch=gfx11-generic")
    with pytest.raises(_codeobject.GeneratorError, match="not the S1 object"):
        rdna._companion(generic, cdna.Outcome("ok", b"x"), pin, tmp_path)


def test_the_tail_zero_companions_are_listed_after_the_reduce_companions(rig) -> None:
    rig.entries = [{"spec": "dbias_reduce,f16"}]
    calls = []

    def build_tailzero(arch, op_dir):
        calls.append((arch, op_dir))
        return [{"spec": "tailzero,f16;arch=gfx90a"}]

    assert rig.run(build_tailzero=build_tailzero) == 0
    assert calls == [("gfx90a", rig.tmp / "out" / "sdpa_bwd_dkdv" / "gfx90a")]
    assert [c["spec"] for c in rig.listed_companions] == [
        "dbias_reduce,f16",
        "tailzero,f16;arch=gfx90a",
    ]


def test_a_tail_zero_failure_stops_the_run_and_a_stopped_run_builds_none(rig) -> None:
    from generators._codeobject import GeneratorError  # noqa: PLC0415

    def failing(arch, op_dir):
        raise GeneratorError("tailzero,f16: R-MIR is red")

    with pytest.raises(GeneratorError, match="R-MIR is red"):
        rig.run(build_tailzero=failing)
    assert not rig.manifest
    assert not list((rig.tmp / "out" / "sdpa_bwd_dkdv" / "gfx90a").iterdir())

    rig.c_reasons.append("c: not ok")

    def never(arch, op_dir):
        raise AssertionError("the tail-zero objects are built after the rows pass")

    with pytest.raises(GeneratorError, match="c: not ok"):
        rig.run(build_tailzero=never)
