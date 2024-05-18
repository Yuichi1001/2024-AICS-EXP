# How to run

```
$ source env.sh
$ echo $NEUWARE_HOME
$ which cncc
$ which cnas
$ bash test.sh
```

# Reference baseline

cntoolkit version:

```
$ cat /usr/local/neuware/version.txt
Neuware Version 3.9.0
```

## MLU370-X8@1000MHz & Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz

cnmon info:

```
    Product Name                   : MLU370-X8
    Firmware                       : v1.1.6
    Driver                         : v5.10.26
    Compute Mode                   : Default
```

test log:

```
+ rm -rf build
+ mkdir build
+ cncc --bang-arch=compute_30 -O3 01_scalar.mlu -o build/01_scalar
+ ./build/01_scalar

M = 128, N = 256, K = 128

CPU Time taken: 5.857 ms

MLU Time taken: 2716.140 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 02_scalar_nram.mlu -o build/02_scalar_nram
+ ./build/02_scalar_nram

M = 128, N = 256, K = 128

CPU Time taken: 5.847 ms

MLU Time taken: 126.067 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 03_vector_nram.mlu -o build/03_vector_nram
+ ./build/03_vector_nram

M = 128, N = 256, K = 128

CPU Time taken: 5.904 ms

MLU Time taken: 0.050 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 04_vector_nram_blocks.mlu -o build/04_vector_nram_blocks
+ ./build/04_vector_nram_blocks

M = 524288, N = 256, K = 128

CPU Time taken: 26075.229 ms

MLU Time taken: 4.563 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 05_vector_nram_blocks_pipe3.mlu -o build/05_vector_nram_blocks_pipe3
+ ./build/05_vector_nram_blocks_pipe3

M = 524288, N = 256, K = 128

CPU Time taken: 26076.031 ms

MLU Time taken: 3.824 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 06_vector_sram_unions_pipe5.mlu -o build/06_vector_sram_unions_pipe5
+ ./build/06_vector_sram_unions_pipe5

M = 524288, N = 256, K = 128

CPU Time taken: 26081.332 ms

MLU Time taken: 4.073 ms

PASSED
```

## MLU370-X4@1000MHz & Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz

cnmon info:

```
    Product Name                   : MLU370-X4
    Firmware                       : v1.1.6
    Driver                         : v5.10.11
    Compute Mode                   : Default
```

test log:

```
+ rm -rf build
+ mkdir build
+ cncc --bang-arch=compute_30 -O3 01_scalar.mlu -o build/01_scalar
+ ./build/01_scalar

M = 128, N = 256, K = 128

CPU Time taken: 8.678 ms

MLU Time taken: 3261.580 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 02_scalar_nram.mlu -o build/02_scalar_nram
+ ./build/02_scalar_nram

M = 128, N = 256, K = 128

CPU Time taken: 8.895 ms

MLU Time taken: 126.071 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 03_vector_nram.mlu -o build/03_vector_nram
+ ./build/03_vector_nram

M = 128, N = 256, K = 128

CPU Time taken: 9.854 ms

MLU Time taken: 0.071 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 04_vector_nram_blocks.mlu -o build/04_vector_nram_blocks
+ ./build/04_vector_nram_blocks

M = 524288, N = 256, K = 128

CPU Time taken: 23470.078 ms

MLU Time taken: 3.703 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 05_vector_nram_blocks_pipe3.mlu -o build/05_vector_nram_blocks_pipe3
+ ./build/05_vector_nram_blocks_pipe3

M = 524288, N = 256, K = 128

CPU Time taken: 23473.186 ms

MLU Time taken: 3.180 ms

PASSED
+ cncc --bang-arch=compute_30 -O3 06_vector_sram_unions_pipe5.mlu -o build/06_vector_sram_unions_pipe5
+ ./build/06_vector_sram_unions_pipe5

M = 524288, N = 256, K = 128

CPU Time taken: 23990.900 ms

MLU Time taken: 3.158 ms

PASSED
```
