import torch
import numpy as np
import torch_mlu
import copy
from mlu_custom_ext import mlu_functions
import unittest


class TestSigmoidBenchmark(unittest.TestCase):
    """
    test sigmoid benchmark
    """

    def test_forward_with_shapes(self, shapes=[(3, 4)]):
        # TODO: 为了计时能准确统计运算部分耗时，请补充预热代码
        for shape in shapes:
            x_cpu = torch.randn(shape)
            x_mlu = x_cpu.to('mlu')
            y_mlu = mlu_functions.sigmoid(x_mlu)
            x_cpu = x_mlu.to('cpu')

        for shape in shapes:
            event_start = torch.mlu.Event()
            event_end = torch.mlu.Event()

            event_start.record()
            x_cpu = torch.randn(shape)
            x_mlu = x_cpu.to('mlu')
            # TODO: 请补充mlu_custom_ext库的Sigmoid函数调用
            y_mlu = mlu_functions.sigmoid(x_mlu)
            y_cpu = x_cpu.sigmoid()
            np.testing.assert_array_almost_equal(y_mlu.cpu(), y_cpu, decimal=3)
            event_end.record()

            event_end.synchronize()
            print('forward time: ', event_start.hardware_time(event_end), 'ms')

    def test_backward_with_shapes(self, shapes=[(3, 4)]):
        # TODO: 为了计时能准确统计运算部分耗时，请补充预热代码
        for shape in shapes:
            x_cpu = torch.randn(shape)
            x_mlu = x_cpu.to('mlu')
            y_mlu = mlu_functions.sigmoid(x_mlu)
            x_cpu = x_mlu.to('cpu')

        for shape in shapes:
            event_start = torch.mlu.Event()
            event_end = torch.mlu.Event()

            event_start.record()
            x_mlu = torch.randn(shape, requires_grad=True, device='mlu')
            # TODO: 请补充mlu_custom_ext库的Sigmoid函数调用
            y_mlu = mlu_functions.sigmoid(x_mlu)
            z_mlu = torch.sum(y_mlu)
            z_mlu.backward()
            grad_mlu = x_mlu.grad
            with torch.no_grad():
                grad_cpu = (y_mlu * (1 - y_mlu)).cpu()
            np.testing.assert_array_almost_equal(grad_mlu.detach().cpu(),
                                                 grad_cpu,
                                                 decimal=3)
            event_end.record()

            event_end.synchronize()
            print('backward time: ', event_start.hardware_time(event_end), 'ms')

if __name__ == '__main__':
    unittest.main()
