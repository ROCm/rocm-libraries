# Slotted hsaco fixture

A slotted engine shaped like the SDPA backward split engine: slots `prep`, `dkdv` and
`dq`, each with its own KMD and dispatch descriptor, and every UKD an authored `hsaco`.

- `prep` and `dkdv` have one pack, for gfx950.
- `dq` has two: one for gfx950 and one for `gfx11-generic`, whose object is the gfx942
  one (packing does not read the object's arch, as in `test_hkp_pack_generic.py`).
- `dq`'s KMD declares `reduce_symbol`, which names a second kernel of the same code
  object (`HsacoFixtureScale`). The handler launches it by symbol; it is no `sources`
  entry, since the packer takes no authored hsaco `sources`.

`objects/<arch>/HsacoFixture.co` are copies of `../hsaco/<arch>/HsacoFixture.co`; see
`../hsaco/README.md` for how they are built.
