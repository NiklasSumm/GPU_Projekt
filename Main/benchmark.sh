#!/bin/bash

iterations=10
block_sizes="32 64 128 256 512 1024"
size_options="1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216"
#implementations="dynamicExclusive fixedInclusive fixedExclusive baseline baselineSetupLess"
implementation=fixedExclusive

# Create headline for output csv
echo "bitmask_size_bytes, time_setup_115_ms, time_apply_115_ms, time_total_115_ms, time_setup_78_ms, time_apply_78_ms, time_total_78_ms, time_setup_88_ms, time_apply_88_ms, time_total_88_ms, time_setup_baseline_ms, time_apply_baseline_ms, time_total_baseline_ms, time_setup_baseline_sl_ms, time_apply_baseline_sl_ms, time_total_baseline_sl_ms" > measurement_times.csv
echo "bitmask_size_bytes, bw_setup_115, bw_apply_115, bw_total_115, bw_setup_78, bw_apply_78, bw_total_78, bw_setup_88, bw_apply_88, bw_total_88, bw_setup_baseline, bw_apply_baseline, bw_total_baseline, bw_setup_baseline_sl, bw_apply_baseline_sl, bw_total_baseline_sl" > measurement_bandwidth.csv

for size in $size_options; do

    echo -n "$size" >> measurement_times.csv
    echo -n "$size" >> measurement_bandwidth.csv
    for block_size in $block_sizes; do
        echo "$block_size"
        output=$(./bin/project --benchmark --$implementation --i $iterations --s $size --blockSize $block_size)
        echo "$output"
        time_setup_ms=`echo "$output" | awk '/Setup Time:/ {print $3}'`
        time_apply_ms=`echo "$output" | awk '/Apply Time:/ {print $3}'`
        time_total_ms=`echo "$output" | awk '/Total Time:/ {print $3}'`
        
        echo -n ", $time_setup_ms, $time_apply_ms, $time_total_ms" >> measurement_times.csv

        bw_setup=`echo "$output" | awk '/Setup Bandwidth:/ {print $3}'`
        bw_apply=`echo "$output" | awk '/Apply Bandwidth:/ {print $3}'`
        bw_total=`echo "$output" | awk '/Total Bandwidth:/ {print $3}'`

        echo -n ", $bw_setup, $bw_apply, $bw_total" >> measurement_bandwidth.csv
    done
    echo "" >> measurement_times.csv
    echo "" >> measurement_bandwidth.csv
done
