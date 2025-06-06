/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include "mm_plugin.h"

#define MM_PREDICT_FALSE(x) (__builtin_expect(x, 0))

#define MM_RETURN_IF_ERROR(...)                \
  do {                                         \
    magicmind::Status _status = (__VA_ARGS__); \
    if (MM_PREDICT_FALSE(!_status.ok()))       \
      return _status;                          \
  } while (0)

#define CORE_NAME "MyPlugin::PluginReLU"
// 1. register custom op
namespace magicmind {
Status DoShapeInfer(IShapeInferResource *context) {
  std::vector<int64_t> input_shape;
  MM_RETURN_IF_ERROR(context->GetShape("input", &input_shape));
  //TODO: 设置输出形状,使其形状与input的形状相同
  MM_RETURN_IF_ERROR(context->SetShape("output", input_shape));
  return Status::OK();
}
PLUGIN_REGISTER_OP(CORE_NAME)
    .Input("input")
    .TypeConstraint("T")
    //TODO:声明输出参数名称为 "output"
    .Output("output")
    //TODO:声明输出的类型约束为模板参数 "T"
    .TypeConstraint("T")
    .Param("T")
    .Type("type")
    .Default(DataType::FLOAT32)
    //TODO:声明参数名称为 "Dim"
    .Param("Dim")
    //TODO:声明Dim的参数类型为 "int" 的类型列表
    .TypeList("int")
    //TODO:设置参数 "Dim" 的默认值为 {1, 1, 1}
    .Default(std::vector<int64_t>{1, 1, 1})
    //TODO:声明参数名称为 "FuncType"
    .Param("FuncType")
    //TODO:声明参数类型为 "int"
    .Type("int")
    //TODO:设置参数 "FuncType" 的默认值
    .Default(1)
    //TODO: 声明形状推导函数
    .ShapeFn(DoShapeInfer);
}  // namespace magicmind
// 2.create plugin kernel
class PluginReLUKernel : public magicmind::IPluginKernel {
 public:
  // check kernel param
  magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
  // set plugin workspace
  size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
  magicmind::Status Enqueue(magicmind::INodeResource *context) override;
  ~PluginReLUKernel(){};

 private:
  uint64_t input_count_ = 1;
  void *input_addr_     = nullptr;
  void *output_addr_    = nullptr;
  magicmind::DataType input_dtype_;
  magicmind::DataType output_dtype_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  cnrtQueue_t queue_;
  cnrtDim3_t dim_;
  cnrtFunctionType_t ktype_;
};

// 3.register kernel
class PluginReLUKernelFactory : public magicmind::IPluginKernelFactory {
 public:
  //TODO: 实现Create函数
  magicmind::IPluginKernel *Create() override { return new PluginReLUKernel(); }
  ~PluginReLUKernelFactory() {}
};

namespace magicmind {
  //TODO: 注册算子Kernel
PLUGIN_REGISTER_KERNEL(CreatePluginKernelDefBuilder(CORE_NAME).DeviceType("MLU"),
                       PluginReLUKernelFactory);
}  // namespace magicmind
