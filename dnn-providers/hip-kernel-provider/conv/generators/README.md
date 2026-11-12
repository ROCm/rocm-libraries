# Conv descriptor generators

Each script writes one conv engine's complete descriptor set (`*.ued.json`, `*.kmd.json`, `*.umd.json`, `*.udd.json`, `*.kdp.json`) into the directory given as its only argument. The descriptors under `src/engines/kernel_ingestor_engine/descriptors/conv/<engine>/` are exactly what these scripts produce: the small ones in git, the `*.kdp.json` catalogs in DVC (one `<catalog>.dvc` pointer each). Running a script into that directory reproduces the files byte for byte.

`regen_all.py` runs every script below; [../REGEN.md](../REGEN.md) is the procedure (regenerate, `dvc add`, upload, commit the pointers).

```bash
# From this directory; Python 3, standard library only.
python3 regen_all.py --check          # every family reproduces, every pointer matches
python3 regen_all.py --dvc-add        # regenerate all, record the catalogs in DVC
python3 gen_igemm.py ../../src/engines/kernel_ingestor_engine/descriptors/conv/igemm   # one family
```

| script | descriptor dir | engines |
|---|---|---|
| `gen_igemm.py` | `igemm` | `CONV_IGEMM_FWD`, `CONV_IGEMM_BWD` |
| `gen_igemm_nchw.py` | `igemm_nchw` | `CONV_IGEMM_NCHW_BWD` |
| `gen_direct.py` | `direct` | `CONV_DIRECT_FWD`, `CONV_DIRECT_BWD` |
| `gen_winograd.py` | `winograd` | `CONV_WINOGRAD_FWD`, `CONV_WINOGRAD_BWD`, `CONV_WINOGRAD_WRW` |
| `gen_conv_wrw.py` | `wrw` | `CONV_WRW` |
| `gen_conv_stem_bwd.py` | `stem_bwd` | `CONV_STEM_BWD` |
| `gen_smallc.py` | `smallc` | `CONV_SMALLC_FWD`, `CONV_SMALLC_BWD` |
| `gen_depthwise_valu.py` | `depthwise_valu` | `CONV_DEPTHWISE_VALU_FWD`, `CONV_DEPTHWISE_VALU_BWD` |
| `gen_depthwise_valu_wrw.py` | `depthwise_valu_wrw` | `CONV_DEPTHWISE_VALU_WRW` |
| `gen_depthwise_rows.py` | `depthwise_rows` | `CONV_DEPTHWISE_ROWS_FWD`, `CONV_DEPTHWISE_ROWS_BWD` |
| `gen_depthwise_mfma.py` | `depthwise_mfma` | `CONV_DEPTHWISE_MFMA_FWD`, `CONV_DEPTHWISE_MFMA_BWD` |
| `gen_group.py` | `group` | `CONV_GROUP_FWD`, `CONV_GROUP_BWD`, `CONV_GROUP_WRW` |

The naive engines (`descriptors/conv/naive`, 9 configs) are maintained by hand.

`gen_conv_stem_bwd.py` enumerates a config space and ships only the configs listed in `conv_stem_bwd_kept.txt`, chosen by a per-config bench (the file's header says how). `python3 gen_conv_stem_bwd.py <dir> --universe` writes the whole space, the input for re-selecting the list.

Rules the generators follow, which the engine packs and tests rely on:
- **Ids are stable.** Every id is a `uuid5` of the config's content, so regenerating doesn't churn the files.
- **Configs are told apart only by integer knobs.** Candidate enumeration and the winner cache distinguish configs only by their integer knob fields. A config field that changes the compiled kernel must therefore be an int knob. The engines' catalog tests fail if two configs share a knob tuple, and some generators also assert it.
- **Each config is admitted on its own gates.** A config is a candidate for a graph only if its pack's gates accept both the graph and the config. The packs (`src/engines/kernel_ingestor_engine/packs/Conv*Problem.*`) decline every field they don't support, and each gate has a test.
- **Packs stay under 500 KB per file,** the repo's pre-commit limit. Large catalogs (`wrw`, `stem_bwd`) are split per dtype and kernel family, and their generators assert the limit.

After regenerating, run the engine's tests; see the conv section of `../README.md` for the filters.
