#!/bin/bash -ex
INSTR_WIDTH=$1

# Due how rocprofv3 works, sometimes data is not recorded and thus needs to be re-ran
# If the output is too short (i.e. only csv headers), repeat the run
CHAR_LIMIT=76

DIR=stats_$INSTR_WIDTH
rm $DIR -fr; mkdir $DIR

ROCPROF_DIR=rocprof_$INSTR_WIDTH

for (( i=0; i<=8; i++ )); do
    EXE="./prog_${i}_${INSTR_WIDTH}".out

    hipcc ../hip.cpp -DBYTE_STRIDE="$i" -DINSTR_WIDTH=$INSTR_WIDTH -O3 -o ./$EXE

    while true; do
        echo "Trying BYTE_STRIDE=$i"

        rm $ROCPROF_DIR/ -rf

        ROCPROF=/opt/rocm/bin/rocprofv3

        export ROCROLLER_BUILD_DIR=./
        # EXE="../scripts/rrperf run --suite fp4_target_d2lds_mi16x16x128_pf4x1_wgm"
        # EXE="./client/rocroller-gemm --m=1024 --n=1024 --k=512 generate validate"
        $ROCPROF --att \
        -d ${ROCPROF_DIR}/ \
        --att-perfcounter-ctrl=8 \
        --att-perfcounters="SQ_INST_LEVEL_VMEM,SQ_INST_LEVEL_LDS,SQ_LDS_BANK_CONFLICT,SQ_VALU_MFMA_BUSY_CYCLES" \
        --att-target-cu=1 \
        --att-shader-engine-mask=0x1 -- $EXE

        { output="$(cat $ROCPROF_DIR/stats_ui_output_agent_*_dispatch_1.csv)"; }

        len=${#output}

        if (( len > CHAR_LIMIT )); then
            mv $ROCPROF_DIR/ $DIR/$i/
            break
        fi
    done
    
    exit

    rm ./$EXE

done