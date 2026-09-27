# Source Provenance — gfx942 / fmha_v3_fwd

The CSV and `.co` binaries in this directory are a snapshot of the FMHA forward
kernels published by the AITER project.

| Field | Value |
|---|---|
| Upstream repository | https://github.com/ROCm/aiter |
| Source commit | `17d4a33b6f9535e820353ebc6217769efc3766d6` |
| Source path | `hsa/gfx942/fmha_v3_fwd/` |

All binaries in this directory are unmodified upstream AITER snapshots. AITER
splits gfx942 forward kernels into `MI300/` (MI300X) and `MI308/` (MI300A APU)
because the two parts have different fp8 support; both are vendored here.

Only the batch-mode (`mode=0`) kernels are vendored, matching what the
`asm_sdpa_engine` dispatcher can select. AITER's group-mode (`mode=1`) and
alternate bf16-conversion (`bf16_cvt` other than the values in `fmha_fwd.csv`)
variants are intentionally omitted.

FP8 forward kernels are only available for gfx942 (hd128); there are no gfx950
fp8 forward kernels and no hd192x128 fp8 kernels in this snapshot.

## SHA256 manifest — gfx942/fmha_v3_fwd/MI300

```
379ccbd65ce645d7d61a018b56c096a541df4d91a9fddbe23b143adae4b889cb  fwd_hd128_bf16_rtne.co
ae0c7e5705cbde27a61dfcc71fb9be39dc48c7614098c4c2a3cf3f073add9bf5  fwd_hd128_bf16_causal_rtne.co
7bffc09e0fea2165b609e3661c896839bff3d0418e4511e99ef62bc2101c8b04  fwd_hd128_fp8.co
90c0d4bb363435de42be4edd582f945f75ee6ba743f4dfd87a05ec0b3529ce87  fwd_hd128_fp8_causal.co
0889ec96435cd84caccb5e769504bcf6c15a01374e41f95664f59fae25cdcfcb  fwd_hd192x128_bf16_rtne.co
9cd4466504b8ced764fb9211356c39bfd0efdb1f2e3143eec709d0b002d7470e  fwd_hd192x128_bf16_causal_rtne.co
```

## SHA256 manifest — gfx942/fmha_v3_fwd/MI308

```
79ff4711e6284621f030b0afd71cc6757729de7faad60b325d2c217926b11d75  fwd_hd128_bf16_rtne.co
d50ecd551041089fc63b26b1f2b5511df282fd6eef7ba5d5f4225f093b961d00  fwd_hd128_bf16_causal_rtne.co
5bf756114109794ea4ab90a026afeb88b46b1888cad219bdf34b3da03ba998d4  fwd_hd128_fp8.co
6af44f3c297dc821f5c0450eb48629732d2bcbe4d83d8b0cb6d418d7285034e8  fwd_hd128_fp8_causal.co
0ef786448fa01bdb681a6fcbb80923a7696b64154401f5ccb96872ed5ac9ab16  fwd_hd192x128_bf16_rtne.co
bfd97b7caf9de2fbe4e0f9a4ff2a4eccce3cb74062dcf2503d2b406831ab5933  fwd_hd192x128_bf16_causal_rtne.co
```
