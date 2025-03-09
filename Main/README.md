# Binary Scan Encoding with Backward Application

## Overview

The **Binary Scan Encoding with Backward Application** project is a CUDA-based implementation utilizing **CUDA 12.8** for efficient bitmask encoding and permutation generation. It combines CUDA libraries such as **CUB, Thrust, and cuRAND** with custom CUDA kernels to implement and benchmark different encoding strategies, most notably **dynamic and fixed-layer tree approaches**.

## Features

- Built with **CUDA 12.8**
- Implements multiple encoding strategies:
  - **Dynamic Exclusive**
  - **Fixed Inclusive**
  - **Fixed Exclusive**
  - **Baseline using Thrust**
- Includes **benchmarking** and **validation** options
- Supports **bitmask sparsity control**
- Implements **packing and unpacking of arrays**

## Dependencies

Ensure the following libraries and tools are installed before building:

- **CUDA 12.8**
- **NVIDIA CUB**
- **Thrust**
- **cuRAND**

## Build Instructions

To build the project, simply run the provided Makefile:

```sh
make
```

This will handle all necessary compilation steps, including linking required libraries.

## Usage

Run the compiled program with various options:

```sh
./bin/project [options]
```

### General Options

| Option                                 | Description                                                                |
| -------------------------------------- | -------------------------------------------------------------------------- |
| `-s <size>` or `--size <size>`         | Set the bitmask size (in `uint64_t` steps)                                 |
| `-i <num>` or `--num-iterations <num>` | Number of iterations for benchmarking                                      |
| `--benchmark`                          | Measure execution times and throughput                                     |
| `--validate`                           | Validate results against a CPU reference                                   |
| `--sparsity <fraction>`                | Control bitmask sparsity (default: `0.0`)                                  |
| `--pack` / `--unpack`                  | Pack or unpack an array instead of producing a permutation for validation. |

### Encoding Strategy Options (Mutually Exclusive)

| Option                   | Description                                         |
| ------------------------ | --------------------------------------------------- |
| `--dynamicExclusive`     | Use a tree-based encoding with dynamic layer count  |
| `--dynamicExclusiveSolo` | Dynamic tree encoding without collaborative descent |
| `--fixedInclusive`       | Use a two-layer encoding with inclusive scan        |
| `--fixedExclusive`       | Use a two-layer encoding with exclusive scan        |
| `--baseline`             | Use baseline Thrust-based encoding                  |
| `--baselineSetupLess`    | Baseline encoding using on-the-fly `copy_if`        |

⚠ **Note:** These options are mutually exclusive; only one can be selected at a time.

### Example Usage

```sh
./bin/project -s 1024 --benchmark --fixedExclusive
```

## Performance Benchmarking

To measure execution time and throughput, use `--benchmark` to produce additional measurement outputs:

```sh
Setup Time: X ms
Setup Bandwidth: Y GB/s
Apply Time: Z ms
Apply Bandwidth: W GB/s
```

Excluded from the measurements are memory allocations and data movement.

## Validation

To ensure correctness, use `--validate`, which compares GPU results against a CPU-generated reference permutation. This option shouldn't be used together with `--pack` / `--unpack`.

