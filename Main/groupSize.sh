#!/bin/bash

iterations=10
implementations="dynamicExclusive fixedInclusive fixedExclusive"
groupSizes="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536"

for impl in $implementations; do

    echo "bitmask_groupSize, time_setup, time_apply" > measurement_groupSize_times_$impl.csv
    echo "bitmask_groupSize, bw_setup, bw_apply" > measurement_groupSize_bandwidth_$impl.csv
    for size in $groupSizes; do

        echo -n "$size" >> measurement_groupSize_times_$impl.csv
        echo -n "$size" >> measurement_groupSize_bandwidth_$impl.csv

        output=$(./bin/project --benchmark --$impl --i $iterations --s 4194304 --sparsity 0.5 --pack --groupSize $size)
        echo "$output"
        time_setup_ms=`echo "$output" | awk '/Setup Time:/ {print $3}'`
        time_apply_ms=`echo "$output" | awk '/Apply Time:/ {print $3}'`

        echo -n ", $time_setup_ms, $time_apply_ms" >> measurement_groupSize_times_$impl.csv

        bw_setup=`echo "$output" | awk '/Setup Bandwidth:/ {print $3}'`
        bw_apply=`echo "$output" | awk '/Apply Bandwidth:/ {print $3}'`

        echo -n ", $bw_setup, $bw_apply" >> measurement_groupSize_bandwidth_$impl.csv

        echo "" >> measurement_groupSize_times_$impl.csv
        echo "" >> measurement_groupSize_bandwidth_$impl.csv
    done
done