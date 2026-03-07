#!/bin/bash

# ===================================================================
# test modes
# ===================================================================

# need to comment out steps 1 and 2 + full pass in 
# first kernel code generator (checks first kernel 
# output buffer)
test_mode_0='input'

# (checks second kernel output buffer)
test_mode_1='full-3d'

# need to comment out first kernel code generator
# call to perform_partial_pass_step_1_2() (checks first 
# kernel output buffer)
test_mode_2='direction_1' 

# (checks first kernel output buffer)
test_mode_3='direction_1_step_1_2'

# need to comment out call to forward_pp_length in 
# second kernel generator's generate_global_function()
# (checks second kernel output buffer)
test_mode_4='direction_1_step_1_2_3_4' 

# requires commenting out call to perform_partial_pass_step_3_4()
# in second kernel generator's generate_global_function() and
# call to local_transpose_pp_length in store_to_global()
# (checks second kernel output buffer)
test_mode_5='direction_1_step_1_2_direction_2'

# ===================================================================
# transform types
# ===================================================================
transform_type_0='c2c'
transform_type_1='r2c'
transform_type_2='c2r'

# -------------------------------------------------------------------
# input parameters
# -------------------------------------------------------------------

length=( 224 108 52 )
precision='single'
batch=( 1 )
pp_dim=( 2 )
pp_radices=( 54 2 )
test_mode=$test_mode_1
transform_type=$transform_type_0
# -------------------------------------------------------------------

in_len_file="in_len.txt"
in_batch_file="in_batch.txt"
in_pp_dim_file="in_pp_dim.txt"
in_pp_radices_file="in_pp_radices.txt"
in_test_mode_file="in_test_mode.txt"
in_transform_type_file="in_transform_type.txt"
rocfft_input_data_file="rocfft_input_data.m"
rocfft_output_data_file="rocfft_output_data.m"
# -------------------------------------------------------------------

echo ${length[@]} > $in_len_file
echo ${batch[@]} > $in_batch_file
echo ${pp_dim[@]} > $in_pp_dim_file
echo ${pp_radices[@]} > $in_pp_radices_file
echo ${test_mode} > $in_test_mode_file
echo ${transform_type} > $in_transform_type_file

# ===================================================================
rocfft_script_dir=$(pwd)
rofft_dir=$(pwd)/../..
rocfft_exec_dir=${rofft_dir}/build/clients/staging/

cd $rocfft_exec_dir


if [ $transform_type = $transform_type_0 ]; then
    ttype='-t0'
elif [ $transform_type = $transform_type_1 ]; then
    ttype='-t2'
elif [ $transform_type = $transform_type_2 ]; then
    ttype='-t3'
fi

ROCFFT_LAYER=16 ./rocfft-bench --precision $precision --length ${length[0]} ${length[1]} ${length[2]} -b ${batch[0]} ${ttype} &> out.txt

cd $rocfft_script_dir

if [ $test_mode = $test_mode_0 ]; then
    buffer_arg_1=0
    buffer_arg_2=1
elif [ $test_mode = $test_mode_1 ]; then
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
elif [ $test_mode = $test_mode_5 ]; then
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
rm $in_transform_type_file



