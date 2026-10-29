# Slotted hsaco fixture

Slots `prep`, `dkdv` and `dq`, each with its own KMD and dispatch descriptor, every UKD an
authored `hsaco`, shaped like the SDPA backward split engine.

- `prep` and `dkdv` have one pack, for gfx950.
- `dq` has two: one for gfx950 and one for `gfx11-generic`, whose object is the gfx942
  one (packing does not read the object's arch, as in `test_hkp_pack_generic.py`).
- `dq`'s KMD declares `reduce_symbol`, naming a second kernel of the same code object
  (`HsacoFixtureScale`) that the handler launches by symbol. It is no `sources` entry: the
  packer takes no authored hsaco `sources`.

`objects/<arch>/HsacoFixture.co` are copies of `../hsaco/<arch>/HsacoFixture.co`; see
`../hsaco/README.md` for how they are built.
