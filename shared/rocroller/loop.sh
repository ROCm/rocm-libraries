#!/bin/bash -e
INSTR_WIDTH=128

# Due how rocprofv3 works, sometimes data is not recorded and thus needs to be re-ran
# If the output is too short (i.e. only csv headers), repeat the run
CHAR_LIMIT=76

DIR=stats_$INSTR_WIDTH
mkdir $DIR -p

ROCPROF_DIR=rocprof_$INSTR_WIDTH

for (( i=0; i<=512; i++ )); do
    hipcc ../hip.cpp -DBYTE_STRIDE="$i" -DINSTR_WIDTH=$INSTR_WIDTH -O3 -o "prog_${i}_${INSTR_WIDTH}".out

    while true; do
        echo "Trying BYTE_STRIDE=$i"

        rm $ROCPROF_DIR/ -rf

        HSA_CU_MASK=0 rocprofv3 --att -i input.json -d $ROCPROF_DIR/ -- ./a.out 2> /dev/null
        { output="$(cat $ROCPROF_DIR/stats_ui_output_agent_*_dispatch_1.csv)"; } > /dev/null 2>&1

        len=${#output}

        if (( len > CHAR_LIMIT )); then
            echo "$output" > "$DIR/$i.csv"
            break
        fi
    done

done