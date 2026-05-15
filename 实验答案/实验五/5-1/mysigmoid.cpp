// file: mysigmoid.cpp
// Pytorch 扩展头文件的引用
#include <torch/extension.h>

#include <cmath>
#include <vector>

using namespace std;

// mysigmoid_cpu 函数的具体实现
torch::Tensor mysigmoid_cpu(const torch::Tensor &dets) {
  TORCH_CHECK(!dets.is_cuda(), "mysigmoid_cpu only supports CPU tensors");
  TORCH_CHECK(dets.scalar_type() == torch::kFloat32,
              "mysigmoid_cpu expects a float32 tensor");

  // TODO: 将输入的 tensor 转化为浮点类型的 vector
  auto input = dets.contiguous();
  auto input_data = input.data_ptr<float>();
  const auto input_size = input.numel();

  // TODO: 创建一个浮点类型的 output_data，output_data 为大小与输入相同的 vector
  vector<float> output_data(input_size);
  // TODO: 对于输入向量的每个元素计算 mysigmoid
  for (int64_t i = 0; i < input_size; ++i) {
    output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
  }

  // TODO: Create tensor options with dtype float32
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  // TODO: Create a tensor from the output vector
  auto flat = torch::from_blob(output_data.data(), {input_size}, opts).clone();
  // TODO: 将得到的 tensor 转换成所需的大小
  auto output = flat.view(input.sizes());
  return output;
}

// TODO: 算子绑定为 Pytorch 的模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mysigmoid_cpu", &mysigmoid_cpu, "MySigmoid activation function (CPU)");
}
