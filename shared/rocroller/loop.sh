#!/bin/bash -e

for (( i=0; i<=8; i++ )); do

    CHAR_LIMIT=76

    hipcc ../hip.cpp -DCOUNTER="$i" -O3

    while true; do
        echo "Trying COUNTER=$i"

        rm ./rocprofv3_out/ -rf

        HSA_CU_MASK=0 rocprofv3 --att -d rocprofv3_out/ -- ./a.out 2> /dev/null
        { output="$(cat rocprofv3_out/stats_ui_output_agent_*_dispatch_1.csv)"; } > /dev/null 2>&1

        len=${#output}

        if (( len > CHAR_LIMIT )); then
            echo "$output" > "stats_$i.txt"
            break
        fi
    done

done