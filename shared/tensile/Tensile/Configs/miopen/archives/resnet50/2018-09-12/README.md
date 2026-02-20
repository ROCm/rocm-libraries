Start with the 6 asm_full logic files

  - vega20_Cijk_Ailk_Bjlk_HB.yaml
  - vega20_Cijk_Ailk_Bljk_HB.yaml
  - vega20_Cijk_Alik_Bljk_HB.yaml
  - vega20_Cijk_Ailk_Bjlk_SB.yaml
  - vega20_Cijk_Ailk_Bljk_SB.yaml
  - vega20_Cijk_Alik_Bljk_SB.yaml

from

  - rocBLAS commit a85df88648587a0d2880a74c6c57964366ab02a1 for HGEMM
  - rocBLAS commit 0ceb1ad64c8bda5473a1e1c3a74ab9ff204acbf8 for SGEMM

we merge the 6 Resnet50-specific logic files archived in the "logic" directory
into the corresponding asm_full logic files of the same name, resulting in the
6 combined asm_full logic files in

  - rocBLAS commit ea27b3aba339b4fd48795153995d24dd96cd6457 for HGEMM+SGEMM

The 6 YAML configuration files used to generate the Resnet50-specific logic
files are archived in the "config" directory correspondingly named

  - hgemm_resnet50_nt.yaml
  - hgemm_resnet50_nn.yaml
  - hgemm_resnet50_tn.yaml
  - sgemm_resnet50_nt.yaml
  - sgemm_resnet50_nn.yaml
  - sgemm_resnet50_tn.yaml

Note that we explicitly purged the 6 sizes with either n=49 or k=49 from
the Resnet50-specific logic files for HGEMM because they won't be using
the assembly kernels.
