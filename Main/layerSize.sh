#!/bin/bash

iterations=10
layer_1_sizes="6 7 8 9 10 11 12 13 14 15 16"
layer_2_sizes="0 1 2 3 4 5 6 7 8 9 10"
implementations="fixedInclusive fixedExclusive"


for impl in $implementations; do
    echo > measurement_layerSize_times_setup_$impl.csv
    echo > measurement_layerSize_times_apply_$impl.csv
    echo > measurement_layerSize_bandwidth_setup_$impl.csv
    echo > measurement_layerSize_bandwidth_apply_$impl.csv
    for l2Size in $layer_2_sizes; do
        for l1Size in $layer_1_sizes; do
            echo "$block_size"
            output=$(./bin/project --benchmark --$impl --i $iterations --s 4194304 --blockSize 128 --sparsity 0.5 --layer1Size $l1Size --layer2Size $l2Size --pack)
            echo "$output"
            time_setup_ms=`echo "$output" | awk '/Setup Time:/ {print $3}'`
            time_apply_ms=`echo "$output" | awk '/Apply Time:/ {print $3}'`
            time_total_ms=`echo "$output" | awk '/Total Time:/ {print $3}'`

            echo -n "$time_setup_ms, " >> measurement_layerSize_times_setup_$impl.csv
            echo -n "$time_apply_ms, " >> measurement_layerSize_times_apply_$impl.csv

            bw_setup=`echo "$output" | awk '/Setup Bandwidth:/ {print $3}'`
            bw_apply=`echo "$output" | awk '/Apply Bandwidth:/ {print $3}'`
            bw_total=`echo "$output" | awk '/Total Bandwidth:/ {print $3}'`

            echo -n "$bw_setup, " >> measurement_layerSize_bandwidth_setup_$impl.csv
            echo -n "$bw_apply, " >> measurement_layerSize_bandwidth_apply_$impl.csv
        done
        echo "" >> measurement_layerSize_times_setup_$impl.csv
        echo "" >> measurement_layerSize_times_apply_$impl.csv
        echo "" >> measurement_layerSize_bandwidth_setup_$impl.csv
        echo "" >> measurement_layerSize_bandwidth_apply_$impl.csv
    done
done