/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef KERNEL_RELU_H_
#define KERNEL_RELU_H_
#include <stdio.h>
#include "cnrt.h"

//TODO: 声明算子实现函数，用于处理单精度浮点数数据
void ReluEnqueue(cnrtDim3_t dim,
                 cnrtFunctionType_t ktype,
                 cnrtQueue_t queue,
                 float *input_addr,
                 float *output_addr,
                 uint32_t count);
void ReluEnqueue(cnrtDim3_t dim,
                 cnrtFunctionType_t ktype,
                 cnrtQueue_t queue,
                 uint16_t *input_addr,
                 uint16_t *output_addr,
                 uint32_t count);
#endif  // KERNEL_RELU_H_
