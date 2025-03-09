#!/bin/bash

iterations=10
size=8388608
sparsity_options="0.20 0.60 0.80 0.90 0.95 0.975 0.9875 0.99375 0.996875 0.9984375 0.99921875"
implementations="dynamicExclusive fixedInclusive fixedExclusive baseline baselineSetupLess"

# Create headline for output csv
echo "sparsity, bw_apply_115, bw_apply_78, bw_apply_88, bw_apply_baseline, bw_apply_baseline_sl" > sparsity_bandwidth_high.csv

for sparsity in $sparsity_options; do
    echo -n "$sparsity" >> sparsity_bandwidth_high.csv
    for impl in $implementations; do
        echo "$impl"
        output=$(./bin/project --benchmark --$impl --i $iterations --s $size --sparsity $sparsity --pack)
        echo "$output"

        bw_apply=`echo "$output" | awk '/Apply Bandwidth:/ {print $3}'`

        echo -n ", $bw_apply" >> sparsity_bandwidth_high.csv
    done
    echo "" >> sparsity_bandwidth_high.csv
done
