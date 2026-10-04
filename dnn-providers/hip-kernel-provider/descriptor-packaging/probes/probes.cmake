# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Packaging probes, one declaration per integration and architecture. Loaded by
# hkp_load_packaging_probes() under HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES; see
# HkpPackagingProbes.cmake for the arguments.

# gfx950 dense attention, rocKE producer. The probe packs the production descriptors
# trimmed to one instance. INSTANCE names the UKD whose compile path every dense
# attention variant shares, so one compile stands for the family. The key is the UKD
# `name`, not its `id`: ids are regenerated on every ingestor run, names encode the
# specialization. When that instance is renamed or removed, configure fails naming it;
# pick another instance of the same family from the KDP and update INSTANCE.
hkp_add_packaging_probe(
    NAME gfx950_attention_dense
    ARCH gfx950
    KIND rocke
    DERIVE_FROM
        "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}/rocKE/gfx950_attention_dense"
    KDP gfx950_attention_dense.kdp.json
    INSTANCE "attention_dense.bf16_d64_hq8_kv1_c1_bm256_bn64.gfx950")
