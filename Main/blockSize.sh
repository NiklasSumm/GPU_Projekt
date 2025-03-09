#!/bin/bash

iterations=10
size_options="1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216"
implementations="dynamicExclusive fixedInclusive fixedExclusive"
blockSizes="32 64 128 256 512 1024"

for impl in $implementations; do

    echo "bitmask_size_bytes, time_setup_32, time_apply_32, time_setup_64, time_apply_64, time_setup_128, time_apply_128, time_setup_256, time_apply_256, time_setup_512, time_apply_512, time_setup_1024, time_apply_1024" > measurement_blockSize_times_$impl.csv
    echo "bitmask_size_bytes, bw_setup_32, bw_apply_32, bw_setup_64, bw_apply_64, bw_setup_128, bw_apply_128, bw_setup_256, bw_apply_256, bw_setup_512, bw_apply_512, bw_setup_1024, bw_apply_1024" > measurement_blockSize_bandwidth_$impl.csv
    for size in $size_options; do

        echo -n "$size" >> measurement_blockSize_times_$impl.csv
        echo -n "$size" >> measurement_blockSize_bandwidth_$impl.csv
        for blockSize in $blockSizes; do
            echo "$impl"
            output=$(./bin/project --benchmark --$impl --i $iterations --s $size --blockSize $blockSize --sparsity 0.5 --pack)
            echo "$output"
            time_setup_ms=`echo "$output" | awk '/Setup Time:/ {print $3}'`
            time_apply_ms=`echo "$output" | awk '/Apply Time:/ {print $3}'`

            echo -n ", $time_setup_ms, $time_apply_ms" >> measurement_blockSize_times_$impl.csv

            bw_setup=`echo "$output" | awk '/Setup Bandwidth:/ {print $3}'`
            bw_apply=`echo "$output" | awk '/Apply Bandwidth:/ {print $3}'`

            echo -n ", $bw_setup, $bw_apply" >> measurement_blockSize_bandwidth_$impl.csv
        done
        echo "" >> measurement_blockSize_times_$impl.csv
        echo "" >> measurement_blockSize_bandwidth_$impl.csv
    done
done