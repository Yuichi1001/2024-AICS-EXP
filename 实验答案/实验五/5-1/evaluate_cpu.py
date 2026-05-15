from zipfile import ZipFile
import os
import time

import cv2
import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

# TODO：导入自定义动态链接库
import mysigmoid_extension


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(relative_path, *fallbacks):
    candidates = [os.path.join(BASE_DIR, relative_path)]
    for fallback in fallbacks:
        candidates.append(os.path.abspath(os.path.join(BASE_DIR, fallback)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


class COCODataSet(Dataset):
    def __init__(self):
        super(COCODataSet, self).__init__()
        data_path = resolve_path("../data/train2014_small.zip", "../../data/train2014_small.zip")
        self.zip_files = ZipFile(data_path)
        self.data_set = [
            file_name for file_name in self.zip_files.namelist() if file_name.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype="uint8")
        # TODO: 使用 cv2.imdecode() 函数从内存缓存中读取数据，并解码成彩色图像格式。
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # TODO: 使用 cv2.resize() 将图像缩放为 512*512，插值方式为区域插值。
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        # TODO: 使用 cv2.cvtColor 将图片从 BGR 格式转换成 RGB 格式。
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # TODO: 将 image 从 numpy 形式转换为 torch.float32，并归一化到 [0, 1]。
        image = torch.from_numpy(image).float() / 255.0
        # TODO: 用 permute 函数将 tensor 从 HxWxC 转换为 CxHxW。
        image = image.permute(2, 0, 1)
        return image


class ResBlock(nn.Module):
    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            # TODO: 进行卷积，卷积核为 3*1*1。
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            # TODO: 执行实例归一化。
            nn.InstanceNorm2d(c),
            # TODO: 执行 ReLU。
            nn.ReLU(),
            # TODO: 进行卷积，卷积核为 3*1*1。
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            # TODO: 执行实例归一化。
            nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        # TODO: 返回残差运算的结果。
        return x + self.layer(x)


class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            # TODO：构建图像转换网络，第一层卷积。
            nn.Conv2d(3, 32, kernel_size=9, padding=4, bias=False),
            nn.InstanceNorm2d(32),
            # TODO：创建激活函数 ReLU。
            nn.ReLU(),
            # TODO：第二层卷积。
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            # TODO：实例归一化。
            nn.InstanceNorm2d(64),
            # TODO：创建激活函数 ReLU。
            nn.ReLU(),
            # TODO：第三层卷积。
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            # TODO：实例归一化。
            nn.InstanceNorm2d(128),
            # TODO：创建激活函数 ReLU。
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            # TODO: 使用 torch.nn.Upsample 对特征图进行上采样。
            nn.Upsample(scale_factor=2, mode="nearest"),
            # TODO: 执行卷积操作。
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            # TODO: 实例归一化。
            nn.InstanceNorm2d(64),
            # TODO: 执行 ReLU 操作。
            nn.ReLU(),
            # TODO: 使用 torch.nn.Upsample 对特征图进行上采样。
            nn.Upsample(scale_factor=2, mode="nearest"),
            # TODO: 执行卷积操作。
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            # TODO: 实例归一化。
            nn.InstanceNorm2d(32),
            # TODO: 执行 ReLU 操作。
            nn.ReLU(),
            # TODO: 执行卷积操作。
            nn.Conv2d(32, 3, kernel_size=9, padding=4),
        )

    def forward(self, x):
        x = self.layer(x).contiguous()
        original_shape = x.shape
        # TODO: 调用自定义 mysigmoid 算子对 x 进行处理得到输出结果 out。
        out = mysigmoid_extension.mysigmoid_cpu(x.view(-1)).view(original_shape)
        return out


if __name__ == "__main__":
    # TODO: 使用 CPU 生成图像转换网络模型并保存在 g_net 中。
    g_net = TransNet().to("cpu")
    model_path = resolve_path("../models/fst.pth", "../../models/fst.pth")
    # TODO: 从 models 文件夹下加载网络参数到 g_net 中。
    g_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    g_net.eval()
    print("g_net build PASS!\n")

    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    data_group = DataLoader(data_set, batch_size=1, shuffle=True, drop_last=True)

    output_dir = resolve_path("../out/cpu")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, image in enumerate(data_group):
            image_c = image.cpu()
            start = time.time()
            # TODO: 计算 g_net，得到 image_g。
            image_g = g_net(image_c)
            delta_time = time.time() - start
            print("Inference (CPU) processing time: %s" % delta_time)
            # TODO: 利用 save_image 函数将生成图像 image_g 和输入图像 image_c 左右拼接后保存。
            save_image(torch.cat((image_c, image_g), -1), os.path.join(output_dir, f"result_{i}.jpg"))
            break
    print("TEST RESULT PASS!\n")
