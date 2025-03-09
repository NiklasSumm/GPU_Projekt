#!/bin/bash

implementations="dynamicExclusive fixedInclusive fixedExclusive baseline baselineSetupLess"

for impl in $implementations; do
    ncu -f --set full -o $impl ./bin/project --s 1048576 --$impl --sparsity 0.5 --pack
done
