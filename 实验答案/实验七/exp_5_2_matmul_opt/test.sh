#!/bin/bash
set -ex

rm -rf build && mkdir build

cncc --bang-arch=compute_30 -O3 01_scalar.mlu -o build/01_scalar && ./build/01_scalar
cncc --bang-arch=compute_30 -O3 02_scalar_nram.mlu -o build/02_scalar_nram && ./build/02_scalar_nram
cncc --bang-arch=compute_30 -O3 03_vector_nram.mlu -o build/03_vector_nram && ./build/03_vector_nram
cncc --bang-arch=compute_30 -O3 04_vector_nram_blocks.mlu -o build/04_vector_nram_blocks && ./build/04_vector_nram_blocks
cncc --bang-arch=compute_30 -O3 05_vector_nram_blocks_pipe3.mlu -o build/05_vector_nram_blocks_pipe3 && ./build/05_vector_nram_blocks_pipe3
cncc --bang-arch=compute_30 -O3 06_vector_sram_unions_pipe5.mlu -o build/06_vector_sram_unions_pipe5 && ./build/06_vector_sram_unions_pipe5