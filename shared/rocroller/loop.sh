#!/bin/bash -ex
INSTR_WIDTH=$1

# Due how rocprofv3 works, sometimes data is not recorded and thus needs to be re-ran
# If the output is too short (i.e. only csv headers), repeat the run
CHAR_LIMIT=76

DIR=stats_$INSTR_WIDTH
rm $DIR -r; mkdir $DIR

ROCPROF_DIR=rocprof_$INSTR_WIDTH

for (( i=0; i<=8; i++ )); do
    EXE="prog_${i}_${INSTR_WIDTH}".out

    hipcc ../hip.cpp -DBYTE_STRIDE="$i" -DINSTR_WIDTH=$INSTR_WIDTH -O3 -o ./$EXE

    while true; do
        echo "Trying BYTE_STRIDE=$i"

        rm $ROCPROF_DIR/ -rf

        # ~/repos/rocprofiler-sdk-build/bin/rocprofv3 -i input.json -d $ROCPROF_DIR/ -- ./$EXE 2> /dev/null
        ~/repos/rocprofiler-sdk-build/bin/rocprofv3 --att -d $ROCPROF_DIR/ -- ./$EXE 2> /dev/null
        { output="$(cat $ROCPROF_DIR/stats_ui_output_agent_*_dispatch_1.csv)"; } > /dev/null 2>&1

        len=${#output}

        if (( len > CHAR_LIMIT )); then
            mv $ROCPROF_DIR/ $DIR/$i/
            break
        fi
    done
    
    rm ./$EXE

done