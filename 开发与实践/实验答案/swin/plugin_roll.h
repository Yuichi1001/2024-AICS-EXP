/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include "mm_plugin.h"
// 1. register custom op
namespace magicmind {
Status CnnlRollDoShapeInfer(IShapeInferResource *context) {
  std::vector<int64_t> input_shape;
  //TODO:获取输入张量的形状
  context->GetShape("input", &input_shape);
  //TODO: 设置输出张量的形状
  context->SetShape("output", input_shape);
  return Status::OK();
}
PLUGIN_REGISTER_OP("MyPlugin::CnnlRoll")
    .Input("input").TypeConstraint("T")
    .Output("output").TypeConstraint("T")
    .Param("T").Type("type").Allowed({magicmind::DataType::FLOAT32, magicmind::DataType::FLOAT16})
    .Default(DataType::FLOAT32)
    .Param("shifts").TypeList("int")//.Default(std::vector<int64_t>{1, 2})
    .Param("dims").TypeList("int")//.Default(std::vector<int64_t>{0,1})
    //TODO: 注册形状推导函数
    .ShapeFn(CnnlRollDoShapeInfer);
}  // namespace magicmind
// 2.create plugin kernel
class CnnlRollKernel : public magicmind::IPluginKernel {
 public:
  // check kernel param
  magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
  // set plugin workspace
  size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
  magicmind::Status Enqueue(magicmind::INodeResource *context) override;
  ~CnnlRollKernel(){};

 private:
  uint64_t input_count_ = 1;
  void *input_addr_     = nullptr;
  void *output_addr_    = nullptr;
  magicmind::DataType input_dtype_;
  magicmind::DataType output_dtype_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;

  std::vector<int64_t> shifts_;
  std::vector<int64_t> dims_;

  cnrtQueue_t queue_;
  cnrtDim3_t dim_;
  cnrtFunctionType_t ktype_;
};

// 3.register kernel
class CnnlRollKernelFactory : public magicmind::IPluginKernelFactory {
 public:
  //TODO: 实现Create函数,在此处返回一个新的 CnnlRollKernel 对象
  magicmind::IPluginKernel *Create() override { return new CnnlRollKernel(); }
  ~CnnlRollKernelFactory() {}
};

namespace magicmind {
  //TODO: 注册算子Kernel,指定注册算子名字和设备类型
PLUGIN_REGISTER_KERNEL(CreatePluginKernelDefBuilder("MyPlugin::CnnlRoll").DeviceType("MLU"),
                       CnnlRollKernelFactory);
}  // namespace magicmind
