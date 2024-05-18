
#pragma once
#include <pybind11/pybind11.h>
#include <torch/extension.h>
torch::Tensor active_sigmoid_mlu(torch::Tensor x);
