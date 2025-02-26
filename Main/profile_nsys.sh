#!/bin/bash

implementations="dynamicExclusive fixedInclusive fixedExclusive baseline baselineSetupLess"

for impl in $implementations; do
    nsys profile -o $impl ./bin/project --s 1048576 --$impl --sparsity 0.5
done
