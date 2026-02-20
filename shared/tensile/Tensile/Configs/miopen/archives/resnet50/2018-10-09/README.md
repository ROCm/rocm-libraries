Start with the 9 asm_full logic files archived under "logic/main":

  - vega20_Cijk_Ailk_Bjlk_HB.yaml
  - vega20_Cijk_Ailk_Bljk_HB.yaml
  - vega20_Cijk_Alik_Bljk_HB.yaml
  - vega20_Cijk_Ailk_Bjlk_HBH.yaml
  - vega20_Cijk_Ailk_Bljk_HBH.yaml
  - vega20_Cijk_Alik_Bljk_HBH.yaml
  - vega20_Cijk_Ailk_Bjlk_SB.yaml
  - vega20_Cijk_Ailk_Bljk_SB.yaml
  - vega20_Cijk_Alik_Bljk_SB.yaml


The 9 Resnet50-specific logic files archived in "logic/resnet50" were merged
into the corresponding asm_full logic files of the same name, resulting in
9 merged asm_full logic files archived in "logic/merged".  These merged
logic files were checked in to the following rocBLAS commit:

  - rocBLAS commit 40a121382beaa7345cdd64190fc246ce93585e54

The 9 YAML configuration files used to generate the Resnet50-specific logic
files are archived in the "config" directory correspondingly named

  - hgemm_resnet50_nt.yaml
  - hgemm_resnet50_nn.yaml
  - hgemm_resnet50_tn.yaml
  - hpa_resnet50_nt.yaml
  - hpa_resnet50_nn.yaml
  - hpa_resnet50_tn.yaml
  - sgemm_resnet50_nt.yaml
  - sgemm_resnet50_nn.yaml
  - sgemm_resnet50_tn.yaml
