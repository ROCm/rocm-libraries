export LD_LIBRARY_PATH=$PWD/../build_tmp/tensilelite:$LD_LIBRARY_PATH

CAT0=bf16_mi450_mc_cat0_qkv_output_gate_v2
rm -rf $CAT0
HIP_VISIBLE_DEVICES=0 ../tensilelite/bin/Tensile $CAT0.yaml $CAT0 > $CAT0.log 2>&1

CAT1=bf16_mi450_mc_cat1_o_projection_v2
# CAT1=bf16_mi450_mc_cat1_o_projection_v2_m1
rm -rf $CAT1
HIP_VISIBLE_DEVICES=1 ../tensilelite/bin/Tensile $CAT1.yaml $CAT1 > $CAT1.log 2>&1