#!/bin/bash

test_mode_1='full-3d'

# requires changing code generator to skip steps 1-2
test_mode_2='direction_1' 

test_mode_3='direction_1_step_1_2'

test_mode_4='direction_1_step_1_2_3_4' 

# -------------------------------------------------------------------
# input parameters
# -------------------------------------------------------------------

length=( 64 64 64 )
batch=( 5 )
pp_dim=( 2 )
pp_radices=( 16 4 )
test_mode=$test_mode_2
# -------------------------------------------------------------------

in_len_file="in_len.txt"
in_batch_file="in_batch.txt"
in_pp_dim_file="in_pp_dim.txt"
in_pp_radices_file="in_pp_radices.txt"
in_test_mode_file="in_test_mode.txt"
rocfft_input_data_file="rocfft_input_data.m"
rocfft_output_data_file="rocfft_output_data.m"
# -------------------------------------------------------------------

echo ${length[@]} > $in_len_file
echo ${batch[@]} > $in_batch_file
echo ${pp_dim[@]} > $in_pp_dim_file
echo ${pp_radices[@]} > $in_pp_radices_file
echo ${test_mode} > $in_test_mode_file

# ===================================================================
rocfft_script_dir=$(pwd)
rofft_dir=$(pwd)/../..
rocfft_exec_dir=${rofft_dir}/build/clients/staging/

cd $rocfft_exec_dir

ROCFFT_LAYER=16 ./rocfft-bench --precision double --length ${length[0]} ${length[1]} ${length[2]} -b ${batch[0]} &> out.txt

cd $rocfft_script_dir

if [ $test_mode = $test_mode_1 ]; then
    buffer_arg_1=0
    buffer_arg_2=2
elif [ $test_mode = $test_mode_2 ]; then
    buffer_arg_1=0
    buffer_arg_2=1
elif [ $test_mode = $test_mode_3 ]; then
    buffer_arg_1=0
    buffer_arg_2=1
elif [ $test_mode = $test_mode_4 ]; then
    buffer_arg_1=0
    buffer_arg_2=2
fi

./rocfft_to_octave.sh 1 $buffer_arg_1 ${rocfft_exec_dir}out.txt
./rocfft_to_octave.sh 0 $buffer_arg_2 ${rocfft_exec_dir}out.txt

rm $rocfft_exec_dir/out.txt

octave -W run_test.m

# ===================================================================

rm $in_len_file
rm $in_batch_file
rm $in_pp_dim_file
rm $in_pp_radices_file
rm $in_test_mode_file



