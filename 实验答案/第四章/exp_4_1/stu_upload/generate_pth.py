import os
import scipy.io
import torch
import torch.nn as nn
from collections import OrderedDict

os.putenv('MLU_VISIBLE_DEVICES','')
cfgs = [64,'R', 64,'R', 'M', 128,'R', 128,'R', 'M',
       256,'R', 256,'R', 256,'R', 256,'R', 'M', 
       512,'R', 512,'R', 512,'R', 512,'R', 'M',
        512,'R', 512,'R', 512,'R', 512,'R', 'M']

IMAGE_PATH = 'data/strawberries.jpg'
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'

def vgg19():
    layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        'flatten', 'fc6', 'relu6','fc7', 'relu7', 'fc8', 'softmax'
    ]
    layer_container = nn.Sequential()
    in_channels = 3
    num_classes = 1000
    for i, layer_name in enumerate(layers):
        if layer_name.startswith('conv'):
            # TODO: 在时序容器中传入卷积运算
            out_channels = cfgs.pop(0)
            while isinstance(out_channels, str):  # 跳过'R'和'M'
                out_channels = cfgs.pop(0)
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            layer_container.add_module(layer_name, conv_layer)
            in_channels = out_channels
        elif layer_name.startswith('relu'):
            # TODO: 在时序容器中执行ReLU计算
            relu_layer = nn.ReLU(inplace=True)
            layer_container.add_module(layer_name, relu_layer)
        elif layer_name.startswith('pool'):
            # TODO: 在时序容器中执行maxpool计算
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            layer_container.add_module(layer_name, pool_layer)
        elif layer_name == 'flatten':
            # TODO: 在时序容器中执行flatten计算
            flatten_layer = nn.Flatten()
            layer_container.add_module(layer_name, flatten_layer)
        elif layer_name == 'fc6':
            # TODO: 在时序容器中执行全连接层计算
            fc_layer = nn.Linear(7*7*512, 4096)
            layer_container.add_module(layer_name, fc_layer)
            in_channels = 4096
        elif layer_name == 'fc7':
            # TODO: 在时序容器中执行全连接层计算
            fc_layer = nn.Linear(4096, 4096)
            layer_container.add_module(layer_name, fc_layer)
            in_channels = 4096
        elif layer_name == 'fc8':
            # TODO: 在时序容器中执行全连接层计算
            fc_layer = nn.Linear(4096, num_classes)
            layer_container.add_module(layer_name, fc_layer)
        elif layer_name == 'softmax':
            # TODO: 在时序容器中执行Softmax计算
            softmax_layer = nn.Softmax(dim=1)
            layer_container.add_module(layer_name, softmax_layer)
    return layer_container


if __name__ == '__main__':
    #TODO:使用scipy加载.mat格式的VGG19模型
    datas = scipy.io.loadmat(VGG_PATH)

    model = vgg19()
    new_state_dict = OrderedDict()
    for i, param_name in enumerate(model.state_dict()):
        name = param_name.split('.')
        if name[-1] == 'weight':
            new_state_dict[param_name] = torch.from_numpy(datas[str(i)]).float()
        else:
            new_state_dict[param_name] = torch.from_numpy(datas[str(i)][0]).float()
    #TODO:加载网络参数到model
    model.load_state_dict(new_state_dict)
    print("*** Start Saving pth ***")
    #TODO:保存模型的参数到models/vgg19.pth
    torch.save(model.state_dict(), 'models/vgg19.pth')
    print('Saving pth  PASS.')
    
