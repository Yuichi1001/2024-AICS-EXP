/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include "kernel_relu.h"
#include "plugin_relu.h"
#include <iostream>

magicmind::Status PluginReLUKernel::SetLocalVar(magicmind::INodeResource *context) {
  // get input/output dtype and check
  context->GetTensorDataType("input", &input_dtype_);
  context->GetTensorDataType("output", &output_dtype_);

  if (input_dtype_ != magicmind::DataType::FLOAT32 &&
      input_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "Input data type is invalid，should be fp32 or fp16，but " +
                       magicmind::TypeEnumToString(input_dtype_) + "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  //TODO: 确定检查条件,如果输入数据类型与输出数据类型不匹配，则返回错误状态
  if (input_dtype_ != output_dtype_) {
    std::string temp = "Input data type is " + magicmind::TypeEnumToString(input_dtype_) +
                       "but output data type is " + magicmind::TypeEnumToString(output_dtype_) +
                       ".";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  // get input/output shape and check
  context->GetTensorShape("input", &input_shape_);
  context->GetTensorShape("output", &output_shape_);
  for (auto dim : input_shape_) {
    input_count_ *= dim;
  }
  uint64_t output_count = 1;
  for (auto dim : output_shape_) {
    output_count *= dim;
  }
  
  //TODO: 如果输入数据不是正整数，返回错误状态
  if (input_count_ <= 1) {
    std::string temp = "Input cout should be positive integer but " + std::to_string(input_count_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  //TODO: 如果输入和输出数据数量不匹配，返回错误状态
  if (input_count_ != output_count) {
    std::string temp = "Input count is " + std::to_string(input_count_) + "but output count is " +
                       std::to_string(output_count) + ".";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  
 std::vector<int64_t> dim_vec;
 context->GetAttr("Dim", &dim_vec);
 if (dim_vec.size() != 3) {
   magicmind::Status status(magicmind::error::Code::UNAVAILABLE, "Cnrtdim is invalid.");
   return status;
 }
  int64_t ktype_int = 1;
  //TODO: 使用 context 对象的 GetAttr 函数获取名为 "FuncType" 的属性值，并将其存储在 ktype_int 变量中
  context->GetAttr("FuncType", &ktype_int);

  // set dim
  ktype_ = (cnrtFunctionType_t)ktype_int;
  dim_.x = dim_vec[0];
  dim_.y = dim_vec[1];
  dim_.z = dim_vec[2];

  return magicmind::Status::OK();
}

size_t PluginReLUKernel::GetWorkspaceSize(magicmind::INodeResource *context) {
  size_t workspace_size = 0;
  return workspace_size;
}

magicmind::Status PluginReLUKernel::Enqueue(magicmind::INodeResource *context) {
  context->GetTensorDataPtr("input", &input_addr_);
  //TODO: // 从 context 对象中获取名为 "output" 的张量数据指针，将其存储在 output_addr_ 成员变量中
  context->GetTensorDataPtr("output", &output_addr_);
  context->GetQueue(&queue_);
  
  //TODO: 调用实现的relu算子对单精度浮点数数据进行处理
  if (input_dtype_ == magicmind::DataType::FLOAT32) {
    ReluEnqueue(dim_, ktype_, queue_, (float *)input_addr_, (float *)output_addr_, 
                input_count_);
  } else {
    ReluEnqueue(dim_, ktype_, queue_, (uint16_t *)input_addr_, (uint16_t *)output_addr_,
                input_count_);
  }
  return magicmind::Status::OK();
}
