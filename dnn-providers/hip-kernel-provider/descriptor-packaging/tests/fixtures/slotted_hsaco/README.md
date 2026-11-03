# Slotted hsaco fixture

Slots `prep`, `dkdv` and `dq`, each with its own KMD and dispatch descriptor, every UKD an
authored `hsaco`, shaped like the SDPA backward split engine.

- `prep` and `dkdv` have one pack, for gfx950.
- `dq` has two: one for gfx950 and one for `gfx11-generic`, whose object is the gfx942
  one (packing does not read the object's arch, as in `test_hkp_pack_generic.py`).
- `dq`'s UKD is a two-launch kernel: an hsaco `sources` map with `add`
  (`HsacoFixtureAdd`) and `scale` (`HsacoFixtureScale`), both symbols of one code object.
  Both packs (gfx950 and `gfx11-generic`) carry it; a handler for `dq` declares the launch
  set `add`, `scale`. `prep` and `dkdv` are single-launch.

`objects/<arch>/HsacoFixture.co` are copies of `../hsaco/<arch>/HsacoFixture.co`; see
`../hsaco/README.md` for how they are built.
