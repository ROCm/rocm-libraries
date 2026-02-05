#!/bin/bash

ProfilerPath="../../../build/bin/ckProfiler"

# Layout: NHWGC-GKYXC-NHWGK (channels last)
fwd_layout=1
bwd_weight_layout=2
bwd_data_layout=1

# FWD configs
mkdir -p forward/profiler

# 2D
dim=2
$ProfilerPath grouped_conv_fwd 0 $fwd_layout $dim --instances > forward/profiler/nhwgc_fp32.conf 
$ProfilerPath grouped_conv_fwd 1 $fwd_layout $dim --instances > forward/profiler/nhwgc_fp16.conf 
$ProfilerPath grouped_conv_fwd 2 $fwd_layout $dim --instances > forward/profiler/nhwgc_bf16.conf

# 3D
dim=3
$ProfilerPath grouped_conv_fwd 2 $fwd_layout $dim --instances > forward/profiler/ndhwgc_bf16.conf 
$ProfilerPath grouped_conv_fwd 1 $fwd_layout $dim --instances > forward/profiler/ndhwgc_fp16.conf 
$ProfilerPath grouped_conv_fwd 0 $fwd_layout $dim --instances > forward/profiler/ndhwgc_fp32.conf

# BWD weight configs
mkdir -p backward_weight/profiler

# 2D
dim=2
$ProfilerPath grouped_conv_bwd_weight 0 $bwd_weight_layout $dim --instances > backward_weight/profiler/nhwgc_fp32.conf 
$ProfilerPath grouped_conv_bwd_weight 1 $bwd_weight_layout $dim --instances > backward_weight/profiler/nhwgc_fp16.conf 
$ProfilerPath grouped_conv_bwd_weight 5 $bwd_weight_layout $dim --instances > backward_weight/profiler/nhwgc_bf16.conf

#3D
dim=3
$ProfilerPath grouped_conv_bwd_weight 5 $bwd_weight_layout $dim --instances > backward_weight/profiler/ndhwgc_bf16.conf 
$ProfilerPath grouped_conv_bwd_weight 1 $bwd_weight_layout $dim --instances > backward_weight/profiler/ndhwgc_fp16.conf 
$ProfilerPath grouped_conv_bwd_weight 0 $bwd_weight_layout $dim --instances > backward_weight/profiler/ndhwgc_fp32.conf

# BWD data configs
mkdir -p backward_data/profiler

# 2D
dim=2
$ProfilerPath grouped_conv_bwd_data 0 $bwd_data_layout $dim --instances > backward_data/profiler/nhwgc_fp32.conf 
$ProfilerPath grouped_conv_bwd_data 1 $bwd_data_layout $dim --instances > backward_data/profiler/nhwgc_fp16.conf 
$ProfilerPath grouped_conv_bwd_data 2 $bwd_data_layout $dim --instances > backward_data/profiler/nhwgc_bf16.conf

#3D
dim=3
$ProfilerPath grouped_conv_bwd_data 2 $bwd_data_layout $dim --instances > backward_data/profiler/ndhwgc_bf16.conf 
$ProfilerPath grouped_conv_bwd_data 1 $bwd_data_layout $dim --instances > backward_data/profiler/ndhwgc_fp16.conf 
$ProfilerPath grouped_conv_bwd_data 0 $bwd_data_layout $dim --instances > backward_data/profiler/ndhwgc_fp32.conf
