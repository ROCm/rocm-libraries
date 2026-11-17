# Bare-ELF code-object fixture

Test data for the authored-`hsaco` tests: one single-target amdgcn code object per arch,
a bare ELF rather than a clang offload bundle. Packer-only test data; never packed by
CMake.

Both objects are compiled from `HsacoFixture.cpp`, which defines two `extern "C"` kernels:
`HsacoFixtureAdd` (4 arguments) and `HsacoFixtureScale` (3 arguments).

## Regenerating

From this directory, for `<arch>` in `gfx942` and `gfx950`:

```
mkdir -p <arch>
amdclang++ -x hip --offload-arch=<arch> --offload-device-only --no-gpu-bundle-output -O3 -fuse-cuid=none HsacoFixture.cpp -o <arch>/HsacoFixture.co
chmod 0644 <arch>/HsacoFixture.co
```

Compiler the committed objects came from (`amdclang++ --version`):

- `AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git 528402e03c23ae6cdddd4a7d3269df3f9636142b)`

## Objects

| File | Bytes | sha256 |
|---|---|---|
| `gfx942/HsacoFixture.co` | 7088 | `c36305a26817525bc9941a4067360c9c4d12950e8e49682678bf67a4a47b6217` |
| `gfx950/HsacoFixture.co` | 7088 | `75386103fdda3e022c92f120cd322a3c2eb347899d7d0c50c872a1793fafb623` |

`llvm-readelf -h`:

```
gfx942/HsacoFixture.co
  ABI Version:                       4
  Flags:                             0x54c, gfx942, xnack, sramecc
gfx950/HsacoFixture.co
  ABI Version:                       4
  Flags:                             0x54f, gfx950, xnack, sramecc
```
