## file: test_mysigmoid.py
import numpy as np
import torch
import torchvision  # noqa: F401 - 与 PDF 示例依赖保持一致。

# TODO：导入自定义连接库
import mysigmoid_extension


def mysigmoid_cpu(rand):
    rand = rand.contiguous().to(torch.float32)
    # TODO：调用 mysigmoid 函数对 rand 进行处理得到输出结果 output
    output = mysigmoid_extension.mysigmoid_cpu(rand)
    return output.contiguous()


def test_mysigmoid():
    torch.manual_seed(12345)
    rand = torch.randn(3, 512, 512, dtype=torch.float32).abs() + 1
    # TODO：调用 mysigmoid_cpu 函数对 rand 进行处理得到输出结果 output_cpu
    output_cpu = mysigmoid_cpu(rand)
    expected = torch.sigmoid(rand)

    print("------------------mysigmoid test completed----------------------")
    print("input: ", rand)
    print("input_size:", rand.size())
    print("output: ", output_cpu)
    print("output_size:", output_cpu.size())

    np.testing.assert_allclose(
        output_cpu.detach().numpy(),
        expected.detach().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    print("TEST mysigmoid PASS!\n")


# if __name__ == "__main__":
test_mysigmoid()
