"""The kernel identity rule of each arch (lib/kernel_identity.py, lib/dat.py):
gfx1250's `cd_th_v1` keys a kernel by its cluster shape and raw temporal
hints, every other arch keeps `tdm_th_v1`, and a record of one rule joins a
library read under the other."""

from bench_fakes import LIBRARY_STEM, solution, write_logic
from lib import correctness, dat
from lib import kernel_identity as ki

GFX950_STEM = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_"
    "Cijk_Dijk_ID75a0_gfx950"
)


def tdm_solution(index, ha=0, hb=0, cluster=(1, 1, 1), tdm=3, mt=(128, 128, 64)):
    """A solution whose A and B the tensor data mover loads (TDMInst 3), with
    temporal hints `ha` and `hb` and cluster shape `cluster`, its name
    spelling them; solutions that differ only in their index share a name."""
    sol = solution(
        index,
        index,
        mt,
        tdmInst=tdm,
        hasTemporalHint=True,
        temporalHintA=ha,
        temporalHintB=hb,
        clusterDim=list(cluster),
    )
    sol["name"] = (
        f"Cijk_Alik_Bljk_F8B_MT{mt[0]}x{mt[1]}x{mt[2]}_MI16x16x128_CL{cluster[0]}x{cluster[1]}"
        f"_THA{ha}_THB{hb}_TDMI{tdm}"
    )
    return sol


def cd(sol):
    return ki.with_rule(dat.kernel_dat_info(sol), ki.CLUSTER_RULE)


def test_each_arch_has_its_identity_rule():
    assert dat.identity_rule_of_stem(LIBRARY_STEM) == ki.CLUSTER_RULE == "cd_th_v1"
    assert dat.identity_rule_of_stem(f"{LIBRARY_STEM}.dat.zlib") == "cd_th_v1"
    assert dat.identity_rule_of_stem("TensileLibrary_X_CU96_gfx1250.dat") == "cd_th_v1"
    assert dat.identity_rule_of_stem(GFX950_STEM) == ki.IDENTITY_RULE == "tdm_th_v1"
    assert ki.rule_of_arch("gfx1250") == ki.rule_of_arch("gfx1250-strict") == "cd_th_v1"
    assert ki.rule_of_arch("gfx950") == ki.rule_of_arch(None) == "tdm_th_v1"


def test_the_loader_stamps_the_rule_of_gfx1250_files_only(tmp_path):
    write_logic(
        tmp_path / f"{LIBRARY_STEM}.dat", [tdm_solution(1), tdm_solution(2, ha=1)]
    )
    write_logic(tmp_path / f"{GFX950_STEM}.dat", [solution(5, 5)])
    gfx1250 = dat.load_library_kernels(tmp_path, LIBRARY_STEM)
    gfx950 = dat.load_library_kernels(tmp_path, GFX950_STEM)
    assert [k["identity_rule"] for k in gfx1250] == ["cd_th_v1", "cd_th_v1"]
    assert all("identity_rule" not in k for k in gfx950)
    assert ki.identity_of_files([(LIBRARY_STEM, gfx1250, None)]).rule == "cd_th_v1"
    assert ki.identity_of_files([(GFX950_STEM, gfx950, None)]).rule == "tdm_th_v1"


def test_a_gfx1250_signature_names_the_cluster_shape_and_raw_hints():
    k = cd(tdm_solution(1, ha=1, hb=3, cluster=(2, 1, 1)))
    text = "MT128x128x64_MI16x16x128_CD2x1_THA1_THB3"
    assert ki.signature_text(k) == text
    assert ki.canonical_name(k) == k["kernel_name"]
    assert ki.kernel_id(k) == f"{text}:{ki.name_hash(k['kernel_name'])}"
    assert ki.identity_attributes(k)["temporal_hint_a"] == 1


def test_th_twins_are_two_kernels_under_cd_and_one_under_tdm():
    a, b = (dat.kernel_dat_info(tdm_solution(i, ha=h)) for i, h in ((1, 1), (2, 3)))
    assert ki.kernel_id(a) == ki.kernel_id(b)
    assert ki.alias_map([a, b]) == {2: 1}
    ca, cb = ki.with_rule(a, ki.CLUSTER_RULE), ki.with_rule(b, ki.CLUSTER_RULE)
    assert ki.kernel_id(ca) != ki.kernel_id(cb)
    assert ki.alias_map([ca, cb]) == {}


def test_only_exact_duplicates_alias_under_cd():
    kernels = [cd(tdm_solution(1)), cd(tdm_solution(2)), cd(tdm_solution(3, hb=1))]
    assert ki.alias_map(kernels) == {2: 1}
    plain = [tdm_solution(4, tdm=0), tdm_solution(5, tdm=0)]
    assert ki.alias_map([cd(s) for s in plain]) == {5: 4}
    assert ki.alias_map([dat.kernel_dat_info(s) for s in plain]) == {}


def test_cd_serves_the_hint_attributes_and_refuses_the_cache_hint_rules():
    rules = {"non_temporal_rule": True}
    hints = ["temporal_hint_a"]
    assert set(ki.tdm_conflicts([cd(tdm_solution(1, ha=1))], rules, hints)) == {
        "non_temporal_rule"
    }
    old = [dat.kernel_dat_info(tdm_solution(1, ha=1))]
    assert set(ki.tdm_conflicts(old, rules, hints)) == {
        "non_temporal_rule",
        "attribute_spec",
    }


def test_a_record_of_the_old_rule_joins_the_library_under_the_new(tmp_path):
    sols = [tdm_solution(1, ha=1), tdm_solution(2, ha=3), tdm_solution(3, hb=1)]
    write_logic(tmp_path / f"{LIBRARY_STEM}.dat", sols)
    now = ki.identity_of_files(
        [(LIBRARY_STEM, dat.load_library_kernels(tmp_path, LIBRARY_STEM), None)]
    )
    then = ki.record(now.under(ki.IDENTITY_RULE))
    assert (now.rule, then["rule"]) == ("cd_th_v1", "tdm_th_v1")
    assert then["kernel_ids"] != ki.record(now)["kernel_ids"]
    same = ki.Remap.of(then, now, "an earlier round")
    assert same.recorded and same.same and not same.files_changed and not same.changed
    other = tmp_path / "rebuilt"
    other.mkdir()
    write_logic(
        other / f"{LIBRARY_STEM}.dat",
        sols[:2] + [tdm_solution(3, hb=1, mt=(256, 128, 64))],
    )
    rebuilt = ki.identity_of_files(
        [(LIBRARY_STEM, dat.load_library_kernels(other, LIBRARY_STEM), None)]
    )
    moved = ki.Remap.of(then, rebuilt, "an earlier round")
    assert not moved.same and 3 in {sid for sid, _ in moved.changed}


def test_gfx950_ids_keep_their_form(tmp_path):
    write_logic(tmp_path / f"{GFX950_STEM}.dat", [solution(7, 7, nta=4)])
    (k,) = dat.load_library_kernels(tmp_path, GFX950_STEM)
    assert ki.rule_of(k) == "tdm_th_v1" and "identity_rule" not in k
    assert (
        ki.kernel_id(k)
        == f"MT128x128x64_MI16x16x128_NTA4_NTB0:{ki.name_hash(k['kernel_name'])}"
    )
    assert (
        ki.record(ki.identity_of_files([(GFX950_STEM, [k], None)]))["rule"]
        == "tdm_th_v1"
    )


def test_a_blocklist_names_a_gfx1250_kernel_under_either_rule():
    k = cd(tdm_solution(1, ha=1))
    raw = k["kernel_name"]
    masked = dat.canonical_name(raw, 3)
    assert masked != raw
    for name in (raw, masked):
        listed = correctness.Blocklist([correctness.Entry(name=name, reason="listed")])
        assert listed.blocks(k)
        assert listed.blocks_id(ki.kernel_id(ki.with_rule(k, ki.IDENTITY_RULE)))
    # A name names the kernel's id under either rule; a masked name, which a
    # list of the old rule may hold, cannot spell the raw hints the new id hashes.
    assert correctness.Blocklist(
        [correctness.Entry(name=raw, reason="listed")]
    ).blocks_id(ki.kernel_id(k))
    assert not correctness.Blocklist(
        [correctness.Entry(name=masked, reason="listed")]
    ).blocks_id(ki.kernel_id(k))
    # Under the gfx1250 rule a raw name blocks its kernel alone, not its TH
    # twins; an old list's canonical name still blocks its old alias group.
    twin = cd(tdm_solution(2, hb=2))
    assert dat.canonical_name(twin["kernel_name"], 3) == masked
    assert not correctness.Blocklist(
        [correctness.Entry(name=raw, reason="listed")]
    ).blocks(twin)
    assert correctness.Blocklist(
        [correctness.Entry(name=masked, reason="listed")]
    ).blocks(twin)
    guarded = correctness.Blocklist(
        [
            correctness.Entry(
                name=raw, reason="listed", guards=(correctness.Guard("M", 64),)
            )
        ]
    )
    assert guarded.guards(k) and not guarded.guards(twin)
