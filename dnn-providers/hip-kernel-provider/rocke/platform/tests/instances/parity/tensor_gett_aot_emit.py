#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/tensor_gett_aot_emit.py -- Python reference emitter for the launches of the
# tensor GETT AOT route table (dispatch/families/tensor_gett_aot.json): per row, one entry's GETT
# kernel (step -1) or pack kernel (step j >= 0) of aot_route_plan, each distinct pack kernel once.
# C twin: tensor_gett_aot_emit.c (the same rows through rocke::gett::table_plan on the table the
# engine embeds). Generated rows: edit both files together; the tests check that the rows cover
# the table.
import sys

from _emit_common import run_emit

ARCH = "gfx950"

ROWS = (
    ("012b6c12610da328685c", 0),
    ("012b6c12610da328685c", 1),
    ("012b6c12610da328685c", -1),
    ("042080ce0ea15d2c8338", 0),
    ("042080ce0ea15d2c8338", 1),
    ("042080ce0ea15d2c8338", -1),
    ("04411be57b9d6cdfaa58", 0),
    ("04411be57b9d6cdfaa58", 1),
    ("04411be57b9d6cdfaa58", -1),
    ("0671da5aeecbc4dc5d59", 0),
    ("0671da5aeecbc4dc5d59", 1),
    ("0671da5aeecbc4dc5d59", -1),
    ("068306b0a9c5fc7d4536", 0),
    ("068306b0a9c5fc7d4536", 1),
    ("068306b0a9c5fc7d4536", -1),
    ("0770044cc7ca1fa88832", -1),
    ("0b39144c8b8b751a8725", 0),
    ("0b39144c8b8b751a8725", 1),
    ("0b39144c8b8b751a8725", -1),
    ("0b9e31d91d6e7878e0fe", 0),
    ("0b9e31d91d6e7878e0fe", -1),
    ("0d90bcf9688e8b5d4b38", 0),
    ("0d90bcf9688e8b5d4b38", 1),
    ("0d90bcf9688e8b5d4b38", -1),
    ("10f57e12fdacd73dbb5c", 0),
    ("10f57e12fdacd73dbb5c", 1),
    ("10f57e12fdacd73dbb5c", -1),
    ("11d514989528ae806df4", 0),
    ("11d514989528ae806df4", 1),
    ("11d514989528ae806df4", -1),
    ("11d936235c29d7dd517d", -1),
    ("1341f9be172eef32d358", 0),
    ("1341f9be172eef32d358", 1),
    ("1341f9be172eef32d358", -1),
    ("13a9b6bdc703e2f4f7be", -1),
    ("14c774c944b1df2a8344", 0),
    ("14c774c944b1df2a8344", 1),
    ("14c774c944b1df2a8344", -1),
    ("15a2315f4cc36aec8fce", 0),
    ("15a2315f4cc36aec8fce", 1),
    ("15a2315f4cc36aec8fce", -1),
    ("19055accd7eb601abec3", 0),
    ("19055accd7eb601abec3", -1),
    ("1ad4ac253748ee823a2f", -1),
    ("1b1eee7d55d6168fcc03", 0),
    ("1b1eee7d55d6168fcc03", 1),
    ("1b1eee7d55d6168fcc03", -1),
    ("1d9e4c79a6c647ad89ce", 0),
    ("1d9e4c79a6c647ad89ce", 1),
    ("1d9e4c79a6c647ad89ce", -1),
    ("1dbd0056a4f6f81f22cc", -1),
    ("20ae5db3f6c0dbca135e", -1),
    ("20c13310ac6dca6286dc", 0),
    ("20c13310ac6dca6286dc", 1),
    ("20c13310ac6dca6286dc", -1),
    ("212547c5205bcac98846", -1),
    ("218ed92663f33d8d6059", -1),
    ("22292947aca822332ed4", 0),
    ("22292947aca822332ed4", -1),
    ("2301408c52b4cc400f26", 0),
    ("2301408c52b4cc400f26", 1),
    ("2301408c52b4cc400f26", -1),
    ("243772ad7becb2aae1ac", -1),
    ("265ac0a86977580413ee", -1),
    ("29a9b26e3b9da61a0de7", -1),
    ("2b072cf7c44451b9a22b", 0),
    ("2b072cf7c44451b9a22b", 1),
    ("2b072cf7c44451b9a22b", -1),
    ("319ea218224e3a5911d7", 0),
    ("319ea218224e3a5911d7", 1),
    ("319ea218224e3a5911d7", -1),
    ("34fe4eedd18b712568f1", -1),
    ("357c8e0b8501b97cb402", 0),
    ("357c8e0b8501b97cb402", -1),
    ("35e0ea04bdc50b609ebe", -1),
    ("362c16f8298fb8fea3d6", 0),
    ("362c16f8298fb8fea3d6", 1),
    ("362c16f8298fb8fea3d6", -1),
    ("36eabef19bddb9ff4c7e", 0),
    ("36eabef19bddb9ff4c7e", -1),
    ("37c1bfd79a0ed90ff4df", 0),
    ("37c1bfd79a0ed90ff4df", -1),
    ("38df70ace0e4e3595a6a", -1),
    ("39d0da1a799f4be83bde", -1),
    ("3a879d228f3f143279d3", -1),
    ("3ae58da4f91192c78cad", -1),
    ("3aedde6067977fe9e520", -1),
    ("3b34a9ecfe0ad39b08b9", -1),
    ("3b8477f8489fd395b29f", -1),
    ("3c190de32ba27cbb96b4", 0),
    ("3c190de32ba27cbb96b4", 1),
    ("3c190de32ba27cbb96b4", -1),
    ("3c7b89bcfadb70631601", 0),
    ("3c7b89bcfadb70631601", -1),
    ("3dc7c7cb5682e2eb33d1", 0),
    ("3dc7c7cb5682e2eb33d1", 1),
    ("3dc7c7cb5682e2eb33d1", -1),
    ("3e909183bd210a5b2ab9", 0),
    ("3e909183bd210a5b2ab9", 1),
    ("3e909183bd210a5b2ab9", -1),
    ("3f58c7b05d1facb7809e", -1),
    ("412dc358627097aeb24f", 0),
    ("412dc358627097aeb24f", -1),
    ("425f3e142e7d3da220e4", 0),
    ("425f3e142e7d3da220e4", 1),
    ("425f3e142e7d3da220e4", -1),
    ("450058cb49cf441cf955", -1),
    ("450fc43b7ff4e0b8b57e", -1),
    ("46a00571165e57adc37a", -1),
    ("47ccd41b8e967df95b49", -1),
    ("48c6eac33cbc003b3f6d", -1),
    ("4a494e95ec17d9ac1361", -1),
    ("4bb3630cdf061141702f", 1),
    ("4bb3630cdf061141702f", -1),
    ("4dacfebeb3360a8ba529", -1),
    ("4e30d6eaab1e50a4e980", -1),
    ("4e83cd53b9fd820a3612", -1),
    ("4fe4c3d818a8bb897d96", -1),
    ("509faec6559433adc026", 0),
    ("509faec6559433adc026", 1),
    ("509faec6559433adc026", -1),
    ("51638de5e285825473ec", -1),
    ("516620d506de046aaf98", -1),
    ("51a30f7971debe0a39b7", 1),
    ("51a30f7971debe0a39b7", -1),
    ("52430bfbc034416ed36b", -1),
    ("52b6c760217d04057123", -1),
    ("5360aae7cb77eaddf6ef", 1),
    ("5360aae7cb77eaddf6ef", -1),
    ("550c7023d13af5c3f1ed", -1),
    ("5577f946a3c1bc5d33ef", -1),
    ("57cd1ffb8a7acfa0bdd2", -1),
    ("588ea39ecf78773703e2", 0),
    ("588ea39ecf78773703e2", 1),
    ("588ea39ecf78773703e2", -1),
    ("5aecb40c02fc43d212d0", -1),
    ("5b22d8b82450b251d9d8", -1),
    ("5ca71abe87fbc552bd78", 0),
    ("5ca71abe87fbc552bd78", 1),
    ("5ca71abe87fbc552bd78", -1),
    ("5fa7fe54d67e22563d86", -1),
    ("617f2a8067c222509ff2", 0),
    ("617f2a8067c222509ff2", 1),
    ("617f2a8067c222509ff2", -1),
    ("647589f5b4097e4e5581", 0),
    ("647589f5b4097e4e5581", -1),
    ("659bde83710bd6cf7167", -1),
    ("66c9d643415a6296c4ea", 0),
    ("66c9d643415a6296c4ea", 1),
    ("66c9d643415a6296c4ea", -1),
    ("69356c1951954352c50a", -1),
    ("696716eec81192dfe39a", -1),
    ("696cdf50424a6eec9449", -1),
    ("69752f84d93522746436", -1),
    ("6ad5607b5570684da16e", -1),
    ("6c77bee73ac3473fbd0f", -1),
    ("6daab3c7a066c1198aeb", -1),
    ("708226678b8c4a7f05ea", -1),
    ("720668d8806319e7909f", -1),
    ("721aebd5228f484549a6", -1),
    ("74f6affc3c52e64e3223", 0),
    ("74f6affc3c52e64e3223", 1),
    ("74f6affc3c52e64e3223", -1),
    ("75f1b7537ddc881ea0c2", 0),
    ("75f1b7537ddc881ea0c2", -1),
    ("760a7f36242c2f9a8cad", -1),
    ("78710b16f819b469b2ae", -1),
    ("794bdd1fbbab76157087", 0),
    ("794bdd1fbbab76157087", 1),
    ("794bdd1fbbab76157087", -1),
    ("79653c2c066b0179a277", -1),
    ("7ba55270a3cd3b9bfc61", -1),
    ("7d26abfd2cab28a39a2f", -1),
    ("7d4e82b32aa6ebd0b380", -1),
    ("7e65e423df3899b85a19", -1),
    ("8066b48ca2459137764d", -1),
    ("83028061080f1bdd23ce", -1),
    ("84ee4233611a51ec29db", -1),
    ("85e8856282e87493e161", -1),
    ("866474f939a0c68f3a03", 0),
    ("866474f939a0c68f3a03", -1),
    ("87c62abd09ff6936fd0d", -1),
    ("89d0f7bffe50630a4b5a", -1),
    ("8a15d1300ae965c8472d", -1),
    ("8ada6d7515e80092f100", -1),
    ("8afd1b4ce908400adf52", 0),
    ("8afd1b4ce908400adf52", 1),
    ("8afd1b4ce908400adf52", -1),
    ("8f3987c044a1a83ade67", -1),
    ("93d26c401ae95fd02373", -1),
    ("94dadbe85a8ed30f0681", -1),
    ("95550e7977c6376a5da1", 0),
    ("95550e7977c6376a5da1", 1),
    ("95550e7977c6376a5da1", -1),
    ("95dbbc40b38cfa3eed30", 0),
    ("95dbbc40b38cfa3eed30", -1),
    ("95e6f66a79e003b6ba6b", -1),
    ("96abf9a2478664ed30a7", 0),
    ("96abf9a2478664ed30a7", 1),
    ("96abf9a2478664ed30a7", -1),
    ("96dd8f530f115e716118", -1),
    ("98389223c6041c88e018", -1),
    ("98889fc0099547216273", -1),
    ("9abead683faaff7e9390", 0),
    ("9abead683faaff7e9390", -1),
    ("9be52e56f5b8737bd95e", 0),
    ("9be52e56f5b8737bd95e", -1),
    ("9c71ef87c98fc88a3277", -1),
    ("9f355551e864a0c8fed9", -1),
    ("9faf16c1bf0ebf142a89", -1),
    ("9fc361de5c9ff5a2d003", -1),
    ("a1c293bbb866b70b5bcb", -1),
    ("a497b91a74fed5a6fbae", -1),
    ("a5854263f42dbc6383df", 0),
    ("a5854263f42dbc6383df", 1),
    ("a5854263f42dbc6383df", -1),
    ("a643c7edf6cd165376c9", -1),
    ("a721aecb01516974a32d", -1),
    ("a76388ffb741318d8fbe", -1),
    ("a798e2bfdd53ac194bda", -1),
    ("aaa5320369eb9725876b", -1),
    ("b4615eb852d2a2ce2516", -1),
    ("b94d992805d5c3e61a13", -1),
    ("bad7e64c9a9650e7b418", -1),
    ("bb3c92eb8d93e69b3d87", -1),
    ("bf6c934fc4ae8bf93bc9", -1),
    ("bf9d8eea17f66abce404", -1),
    ("c6de67344c326b800912", -1),
    ("c7220eb851dea3d3dae8", -1),
    ("c98ab7f60ad8f46cd9f8", -1),
    ("ca49ba6273d351af9514", -1),
    ("cb59ea6dca8cec1030fa", -1),
    ("cd59464559b2acf86af8", 0),
    ("cd59464559b2acf86af8", 1),
    ("cd59464559b2acf86af8", -1),
    ("cf4fafcba9256525353a", 0),
    ("cf4fafcba9256525353a", 1),
    ("cf4fafcba9256525353a", -1),
    ("d15536c867f798af8a19", -1),
    ("d1e0544b8ed56d8b2a9e", 0),
    ("d1e0544b8ed56d8b2a9e", 1),
    ("d1e0544b8ed56d8b2a9e", -1),
    ("d4363eb0c5951656f4d1", -1),
    ("d58c8d31aa16bbdedbf9", -1),
    ("d58d20c4e4598c7a6bb8", -1),
    ("d8ff6eb475bf74452d36", -1),
    ("d936c73465f80705a77a", -1),
    ("d949e80c4ef20e09d38f", -1),
    ("db0f6efa5b33e8c2fb68", -1),
    ("db69fee9690e14b022c4", -1),
    ("dcd5c8fccec0b7cc1071", -1),
    ("dcfee7da850b57a22f96", -1),
    ("dd8318e003604b53e9c7", -1),
    ("ddf7c3a4d8e205342a3e", -1),
    ("e2693d430ffd132d28f5", -1),
    ("e6706f3378c2bf4e139e", -1),
    ("e67e9de79de356bd172a", -1),
    ("e80cf8c6d7f3185b478e", -1),
    ("e8386b1f2df992fe551a", -1),
    ("e89b240120dbcaa28df0", -1),
    ("eafc7c398ee16ffa2f83", -1),
    ("ed74fc2fb8a4fc3826e5", -1),
    ("ee61f9a827bf2f0ebfda", 0),
    ("ee61f9a827bf2f0ebfda", -1),
    ("ef9e568ae7501f16b047", -1),
    ("f19c593bf01e47d5b4d7", -1),
    ("f2102d7149f0b9342df1", -1),
    ("f228b33c15f417271d1d", -1),
    ("f272b6075ed9b5d6f0ec", -1),
    ("f2f148e4579f3d0943c2", -1),
    ("f3938f9ecb551f4e69fe", -1),
    ("f5692629036bdce871b6", -1),
    ("f662fb2f19293ae6c684", -1),
    ("f6c61f8370237e158689", -1),
    ("f7d3938e61e3ed3bebc7", -1),
    ("fd532eaab62421ea656f", -1),
    ("fddc489510469bc69b67", -1),
    ("fe61bf6e633a208def9b", -1),
    ("ffb4b07a24eb3f1a8ad0", -1),
)


def _spec(idx: int):
    from rocke.dispatch.families.tensor_gett import (
        TensorGettRequest,
        aot_entry_problem,
        aot_route_plan,
        tensor_gett_route_key,
    )
    from rocke.instances.common.tensor.gett import (
        GettKnobs,
        TensorGettSpec,
        is_valid_spec,
    )

    if not 0 <= idx < len(ROWS):
        raise SystemExit(f"unknown config index {idx}")
    key, step = ROWS[idx]
    p = aot_entry_problem(ARCH, key)
    if p is None:
        sys.stderr.write(f"no AOT entry {key}\n")
        sys.exit(1)
    prob = TensorGettSpec.make(
        p["modes_a"],
        p["modes_b"],
        p["modes_d"],
        p["extents"],
        dtype=p["dtype"],
        knobs=GettKnobs(bilinear=p["bilinear"]),
    )
    req = TensorGettRequest(problem=prob, arch=ARCH)
    if tensor_gett_route_key(req) != key:
        sys.stderr.write("route key mismatch\n")
        sys.exit(1)
    pl = aot_route_plan(req)
    if pl is None:
        sys.stderr.write("no AOT route plan\n")
        sys.exit(1)
    if step < 0:
        ok, why = is_valid_spec(pl.gett.spec, arch=ARCH)
        if not ok:
            sys.stderr.write(f"invalid spec: {why}\n")
            sys.exit(1)
        return ("gett", pl.gett.spec), ARCH
    if step >= len(pl.packs):
        sys.stderr.write("no pack step\n")
        sys.exit(1)
    return ("pack", pl.packs[step].spec), ARCH


def _build(spec, arch):
    from rocke.instances.common.tensor.gett import build_tensor_gett
    from rocke.instances.common.tensor.pack import build_tensor_pack

    kind, s = spec
    if kind == "gett":
        return build_tensor_gett(s, arch=arch)
    return build_tensor_pack(s, arch=arch)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: tensor_gett_aot_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    sys.exit(main())
