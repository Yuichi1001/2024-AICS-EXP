/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include "cnnl.h"
#include "plugin_roll.h"
#include <iostream>

magicmind::Status CnnlRollKernel::SetLocalVar(magicmind::INodeResource *context) {
  // get input/output dtype and check
  context->GetTensorDataType("input", &input_dtype_);
  context->GetTensorDataType("output", &output_dtype_);
  context->GetAttr("shifts", &shifts_);
  context->GetAttr("dims", &dims_);
  std::cout<<"plugin roll success"<<std::endl;
  
  if (input_dtype_ != magicmind::DataType::FLOAT32 &&
      input_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "Input data type is invalid，should be fp32 or fp16，but " +
                       magicmind::TypeEnumToString(input_dtype_) + "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  //TODO:检查输入和输出数据类型是否一致
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

 //TODO: 检查输入张量的元素数量是否为正整数
  if (input_count_ < 1) {
    std::string temp = "Input cout should be positive integer but " + std::to_string(input_count_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  //TODO: 检查输入和输出张量的元素数量是否一致
  if (input_count_ != output_count) {
    std::string temp = "Input count is " + std::to_string(input_count_) + "but output count is " +
                       std::to_string(output_count) + ".";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  return magicmind::Status::OK();
}

size_t CnnlRollKernel::GetWorkspaceSize(magicmind::INodeResource *context) {
  size_t workspace_size = 0;
  return workspace_size;
}

template <typename T, typename SRC>
inline std::vector<T> ConvertVectorDtype(const SRC &data) {
  std::vector<T> dst(data.begin(), data.end());
  return dst;
}

static std::vector<int> int64_vec_to_int(std::vector<int64_t> &vec)
{
  std::vector<int> ret;
  for (auto &&val : vec)
  {
    ret.push_back(val);
  }
  return ret;
}

static cnnlDataType_t mm_dtype_to_cnnl_dtype(magicmind::DataType mm_dtype)
{
  switch (mm_dtype)
  {
  case magicmind::DataType::BOOL:
    return CNNL_DTYPE_BOOL;
  case magicmind::DataType::FLOAT16:
    return CNNL_DTYPE_HALF;
  case magicmind::DataType::FLOAT32:
    return CNNL_DTYPE_FLOAT;
  case magicmind::DataType::INT32:
    return CNNL_DTYPE_INT32;
  case magicmind::DataType::INT64:
    return CNNL_DTYPE_INT64;
  default:
    return CNNL_DTYPE_INVALID;
  }
}

magicmind::Status CnnlRollKernel::Enqueue(magicmind::INodeResource *context) {
  context->GetTensorDataPtr("input", &input_addr_);
  context->GetTensorDataPtr("output", &output_addr_);
  context->GetQueue(&queue_);
  cnnlHandle_t handle;
  cnnlCreate(&handle);
  cnnlSetQueue(handle,queue_);

  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlCreateTensorDescriptor(&input_desc);
  cnnlCreateTensorDescriptor(&output_desc);

  magicmind::DataType input_dtype;
  //TODO：通过context->GetTensorDataType获取输入张量的数据类型
  context->GetTensorDataType("input", &input_dtype);
  cnnlDataType_t input_dtype_;
  //TODO: 将 magicmind 数据类型转换为 cnnl 数据类型
  input_dtype_ = mm_dtype_to_cnnl_dtype(input_dtype);

  std::vector<int> in_shape_;
  //TODO: 将 int64_t 类型的向量 input_shape_ 转换为 int 类型的向量 in_shape_
  in_shape_ = int64_vec_to_int(input_shape_);
  cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_ARRAY, input_dtype_,
           static_cast<int>(in_shape_.size()), in_shape_.data());
  //TODO: 配置输出张量描述符
  cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_ARRAY, input_dtype_,
         static_cast<int>(in_shape_.size()), &in_shape_[0]);

  size_t workspace_size = 0;
  cnnlGetRollWorkspaceSize(handle, input_desc, &workspace_size);
  //TODO: 调用cnnlRoll算子
  cnnlRoll(handle, input_desc, input_addr_,
          ConvertVectorDtype<int>(shifts_).data(), static_cast<int>(shifts_.size()),
          ConvertVectorDtype<int>(dims_).data(), static_cast<int>(dims_.size()),
          nullptr, workspace_size,
          output_desc, output_addr_);
  return magicmind::Status::OK();
}
