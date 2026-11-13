<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# Regenerating the conv engine descriptors

Every conv engine family except naive is **generated**: `generators/gen_<family>.py` writes the
family's complete descriptor set into
`src/engines/kernel_ingestor_engine/descriptors/conv/<family>/`. Below, `$CONV` names that
`conv/` folder.

| Files | Stored in | Why |
|---|---|---|
| `*.ued.json`, `*.udd.json`, `*.kmd.json`, `*.umd.json` | git | small, read in review |
| `kernels/` | git | the HIP sources, embedded into the plugin at build time |
| `*.kdp.json` (instance catalogs) | DVC, remote `ingestor`; git holds `<catalog>.kdp.json.dvc` | generated, 684K lines across the families |
| `naive/*.kdp.json` | git | three small catalogs written by hand, no generator |

Each catalog is its own DVC output, beside the descriptors that name it, rather than one
folder per family: an `embedded_source` UKD names its source relative to its own folder
(`kernels/<File>.cpp`), and neither the packer nor `hkp_verify_embedded_sources` accepts a
`..`, so a KDP must sit in a folder above `kernels/`, which a DVC folder output cannot
contain (`src/engines/kernel_ingestor_engine/descriptors/README.md`).

Generation is **deterministic**: ids are `uuid5` of the config content and the files are
written with fixed formatting, so regenerating an unchanged generator reproduces every file
byte for byte, and every pointer's md5 with it.

The descriptors, the catalogs and the pointers are **one unit**: a change to a generator lands
as the regenerated descriptors, the new pointers and the uploaded content, in one commit. A
pointer whose content was never pushed fails every pull, CI's included.

Everything below runs from the repository root. Python 3, standard library only, plus `dvc`
(3.x) and `git`.

---

## 1. Fetch

Reads are anonymous:

```bash
dvc pull -r ingestor dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/conv/*/*.kdp.json.dvc
```

TheRock's `fetch_sources.py` pulls every `*.dvc` pointer, honouring `remote: ingestor`.
Without the catalogs a family is not served (`conv/ConvContent.cmake`): configure warns, or
fails with `-DHIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON`.

## 2. Verify the tree reproduces

Writes nothing. Needs the catalogs pulled.

```bash
python3 dnn-providers/hip-kernel-provider/conv/generators/regen_all.py --check
```

Every family prints `ok` when each generated file equals the tree's byte for byte and each
catalog's pointer records that catalog's md5, size and `remote: ingestor`. A stale catalog
or pointer (one no generator writes any more) is reported too.

## 3. Regenerate and record

After a change to a generator (a new config, a new family, a field):

```bash
# (1) Regenerate every family into the tree, then record the catalogs in DVC:
#     `dvc add` per catalog (rewrites its pointer), `remote: ingestor` on each pointer,
#     and `git add` of the pointers, the .gitignore entries and the small descriptors.
python3 dnn-providers/hip-kernel-provider/conv/generators/regen_all.py --dvc-add
#     A catalog a generator no longer writes fails the run; --prune deletes it and its
#     pointer. --families igemm,wrw restricts the run.

# (2) The family's tests, as for any descriptor change (see ../README.md for the filters).

# (3) Upload the changed catalogs (needs write credentials; reads stay anonymous). The
#     committed .dvc/config sets allow_anonymous_login = true, which makes DVC send unsigned
#     requests, so writing needs the --local override first. Credentials go in the
#     environment, never in a config file.
dvc remote modify --local ingestor allow_anonymous_login false
D=dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/conv
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... dvc push -r ingestor $D/<family>/*.kdp.json.dvc

# (4) Commit the pointers with the change that produced them. The pointer diff shows only
#     md5 and size, so say in the message what the catalogs gained or lost.
git commit
```

`dvc push` with pointer targets uploads only those outputs; an unchanged catalog is already
on the remote and is skipped. Push before the PR goes up.

A **new family** is one more row in `FAMILIES` (`generators/regen_all.py`) and a
`HIPDNN_ENGINE_CONV_<FAMILY>` row in `IngestorPacks.cpp` (`../README.md`, *Adding an engine
family*); `--dvc-add` creates its pointers and `.gitignore`.

## 4. Merging a branch that still writes catalogs into git

A branch cut before the catalogs moved to DVC still commits `*.kdp.json` as ordinary git
files. Merging it conflicts on every catalog it touched (modify/delete: git sees the file
deleted here, modified there). The catalogs are generated, so never resolve them by hand.
Take the code, rerun the generators, record:

```bash
git merge <lead-branch>                 # conflicts in descriptors/conv/<family>/*.kdp.json
# Resolve every non-catalog conflict (generators, kernels, packs, tests) as usual. Then
# regenerate, which overwrites each catalog with what the merged generator writes,
# untracks it from git, dvc-adds it and stages the pointers:
python3 dnn-providers/hip-kernel-provider/conv/generators/regen_all.py --dvc-add
python3 dnn-providers/hip-kernel-provider/conv/generators/regen_all.py --check
git status --short -- dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/conv \
  | grep 'kdp.json$'                    # must print nothing: no catalog left in git
git commit                              # the merge commit
# then upload (section 3, step 3) before pushing the branch.
```

The same recipe covers a branch that added a family or split a catalog: the new catalogs get
new pointers, and `--prune` removes the ones the merged generators no longer write. It is
safe to rerun: an unchanged catalog keeps its pointer byte for byte.

## 5. When the bytes differ

`--check` reporting a difference on a tree nobody regenerated means a generator is no longer
deterministic (an unordered set or dict iteration, a timestamp, a float printed through a
locale) or a catalog was edited by hand. Fix the generator and regenerate; never edit a
catalog directly, since the next regeneration discards the edit.
