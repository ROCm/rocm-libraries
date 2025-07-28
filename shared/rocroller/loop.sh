#!/bin/bash -e

for (( i=0; i<5; i++ )); do

    CHAR_LIMIT=76

    hipcc ../hip.cpp -DSTRIDE="$i"

    while true; do
        rm ./rocprofv3_out/ -rf

        HSA_CU_MASK=0 rocprofv3 --att -i input.json -d rocprofv3_out/ -- ./a.out > /dev/null 2>&1
        { output="$(cat rocprofv3_out/stats_ui_output_agent_*_dispatch_1.csv)"; } > /dev/null 2>&1

        len=${#output}
        echo "Num characters: ${#output}"

        if (( len > CHAR_LIMIT )); then
            echo "$output"
            break
        fi
    done

done