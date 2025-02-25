#!/bin/bash

iterations=10
size=16777216
sparsity_options="0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95"
implementations="dynamicExclusive fixedInclusive fixedExclusive baseline baselineSetupLess"

# Create headline for output csv
echo "sparsity, bw_apply_115, bw_apply_78, bw_apply_88, bw_apply_baseline, bw_apply_baseline_sl" > sparsity_bandwidth.csv

for sparsity in $sparsity_options; do
    echo -n "$sparsity" >> sparsity_bandwidth.csv
    for impl in $implementations; do
        echo "$impl"
        output=$(./bin/project --benchmark --$impl --i $iterations --s $size --sparsity $sparsity)
        echo "$output"

        bw_apply=`echo "$output" | awk '/Apply Bandwidth:/ {print $3}'`

        echo -n ", $bw_apply" >> sparsity_bandwidth.csv
    done
    echo "" >> sparsity_bandwidth.csv
done
