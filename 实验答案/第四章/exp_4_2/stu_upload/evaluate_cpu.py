from torchvision.models import vgg19
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import cv2
import numpy
import time

class COCODataSet(Dataset):

    def __init__(self):
        super(COCODataSet, self).__init__()
        self.zip_files = ZipFile('./data/train2014_small.zip')
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        # TODO: 使用cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成彩色图像格式。
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) 
        # TODO: 使用cv2.resize()将图像缩放为512*512大小，其中所采用的插值方式为：区域插值
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA) 
        # TODO: 使用cv2.cvtColor将图片从BGR格式转换成RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # TODO: 将image从numpy形式转换为torch.float32,并将其归一化为[0,1]
        image = torch.from_numpy(image).float() / 255.0 
        # TODO: 用permute函数将tensor从HxWxC转换为CxHxW
        image = image.permute(2, 0, 1) 
        return image

class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            
            #TODO: 进行卷积，卷积核为3*1*1
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            #TODO: 执行实例归一化
            nn.InstanceNorm2d(c),
            #TODO: 执行ReLU
            nn.ReLU(),
            #TODO: 进行卷积，卷积核为3*1*1
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            #TODO: 执行实例归一化
            nn.InstanceNorm2d(c)
        )
        
    def forward(self, x):
        #TODO: 返回残差运算的结果
        return torch.nn.functional.relu(self.layer(x) + x)


class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            
            ###################下采样层################
            # TODO：构建图像转换网络，第一层卷积
             nn.Conv2d(3, 32, kernel_size=9, padding=4, bias=False),
            # TODO：实例归一化
            nn.InstanceNorm2d(32),
            # TODO：创建激活函数ReLU
            nn.ReLU(),
            # TODO：第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            # TODO：实例归一化
            nn.InstanceNorm2d(64),
            # TODO：创建激活函数ReLU
            nn.ReLU(),
            # TODO：第三层卷积
             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            # TODO：实例归一化
            nn.InstanceNorm2d(128),
            # TODO：创建激活函数ReLU
            nn.ReLU(),

            ##################残差层##################
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),

            ################上采样层##################
            #TODO: 使用torch.nn.Upsample对特征图进行上采样
            nn.Upsample(scale_factor=2, mode='nearest'),
            #TODO: 执行卷积操作
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            #TODO: 实例归一化
            nn.InstanceNorm2d(64),
            #TODO: 执行ReLU操作
            nn.ReLU(),

            #TODO: 使用torch.nn.Upsample对特征图进行上采样
            nn.Upsample(scale_factor=2, mode='nearest'),
            #TODO: 执行卷积操作
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            #TODO: 实例归一化
            nn.InstanceNorm2d(32),
            #TODO: 执行ReLU操作
            nn.ReLU(),
            
            ###############输出层#####################
            #TODO: 执行卷积操作
            nn.Conv2d(32, 3, kernel_size=9, padding=4),
            #TODO： sigmoid激活函数
            nn.Sigmoid()
        )

    def forward(self, x):
        return torch.nn.functional.relu(self.layer(x))
    


if __name__ == '__main__':
    # TODO: 使用cpu生成图像转换网络模型并保存在g_net中
    g_net = TransNet().to('cpu')
    # TODO:从/models文件夹下加载网络参数到g_net中
    g_net.load_state_dict(torch.load('models/fst.pth', map_location='cpu'))
    print("g_net build  PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")

    batch_size = 1
    data_group = DataLoader(data_set,batch_size,True,drop_last=True)

    for i, image in enumerate(data_group):
        image_c = image.cpu()
        #print(image_c.shape)
        start = time.time()
        # TODO: 计算 g_net,得到image_g
        image_g = g_net(image_c)
        end = time.time()
        delta_time = end - start
        print("Inference (CPU) processing time: %s" % delta_time)
        #TODO: 利用save_image函数将tensor形式的生成图像image_g以及输入图像image_c以jpg格式左右拼接的形式保存在/out/cpu/文件夹下
        save_image(torch.cat((image_c, image_g), -1), f'out/cpu/result_{i}.jpg')
    print("TEST RESULT PASS!\n")