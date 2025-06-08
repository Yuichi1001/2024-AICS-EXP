# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import json
import argparse
import numpy as np

import torch

from src.config import get_config
from src.models import build_model

import magicmind.python.runtime as mm
import magicmind.python.runtime.parser as mm_parser

from torchvision import transforms
from PIL import Image

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', 
                        default='./src/configs/swinv2/swinv2_tiny_patch4_window8_256.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def trace_pt_model(config):
    # 根据config文件创建模型
    model = build_model(config)
    # 加载模型权重
    state_dict = torch.load('swinv2_tiny_patch4_window8_256.pth')['model']
    #TODO: 将模型的权重更新为预训练权重
    model.load_state_dict(state_dict)
    #TODO: 模型设置为推理模式，且权重数据类型设置为float
    model = model.float().eval()
    # 生成swin-transformer jit.trace 的pt模型
    input = torch.randn(1, 3, 256, 256).float()
    # 使用 torch.jit.trace() 函数对模型进行追踪（trace），以便生成 PyTorch JIT 模型
    trace_model = torch.jit.trace(model, input)
    torch.jit.save(trace_model, "trace_swinv2_tiny.pt")
    print('finish trace_pt_model')

def load_plugin_lib():
    roll_plugin_op_lib = "./add_plugin_roll/build_pytorch/libplugin_roll_with_pytorch.so"
    relu_plugin_op_lib = "./add_plugin_relu/build_pytorch/libplugin_relu_with_pytorch.so"
    loader_roll = mm.LoadPluginLibrary()
    loader_relu = mm.LoadPluginLibrary()
    assert loader_roll.load(roll_plugin_op_lib)
    assert loader_relu.load(relu_plugin_op_lib)
    print('finish load_plugin_lib')

def parser_pt_model():
    builder = mm.Builder()
    builder_config = mm.BuilderConfig()

    build_config = {
        "archs": ["mtp_372"],
        "graph_shape_mutable": True,
        "precision_config": {"precision_mode": "force_float32"},
        "opt_config": {"type64to32_conversion": True, "conv_scale_fold": True}
    }
    builder_config.parse_from_string(json.dumps(build_config)).ok()
    network = mm.Network()
    parser = mm_parser.Parser(mm.enums.ModelKind.kPytorch)
    parser.set_model_param("pytorch-input-dtypes", [mm.DataType.FLOAT32])
    assert parser.parse(network, "trace_swinv2_tiny.pt").ok()
    model = builder.build_model("./pytorch_converted", network, builder_config)
    assert model is not None

    offline_model_name = "pytorch_swinv2_model_plugin_roll_relu"
    model.serialize_to_file(offline_model_name)
    print("Generate model done, model save to %s" % offline_model_name)
    return offline_model_name

def img_preprocess():
    transform = transforms.Compose([
        #TODO: 将图像调整为指定大小（292x292），使用双三次插值方法
        transforms.Resize(292, interpolation=Image.BICUBIC),
        #TODO: 从图像中心裁剪出指定大小的区域（256x256）
        transforms.CenterCrop(size=(256, 256)),
        #TODO: 将图像转换为 PyTorch Tensor 格式
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    image_list = ['./course_images/ILSVRC2012_val_00000293.JPEG', 
                  './course_images/ILSVRC2012_val_00038455.JPEG']
    return transform, image_list 


def main(config):
    trace_pt_model(config)
    load_plugin_lib()
    mm_model = parser_pt_model()
    #TODO: 获取图像预处理转换和图像文件列表
    transform, image_list=img_preprocess()

    # 创建 Device
    dev = mm.Device()
    dev.id = 0  #设置 Device Id
    assert dev.active().ok()

    # 加载 MagicMind Swim-Transformer 模型
    model = mm.Model()
    model.deserialize_from_file(mm_model)

    # 创建 MagicMind engine, context
    engine = model.create_i_engine()
    context = engine.create_i_context()

    # 根据 Device 信息创建 queue 实例
    queue = dev.create_queue()
    
    # 准备 MagicMind 输入输出 Tensor 节点
    inputs = context.create_inputs()
    for i in range(len(image_list)):
        path = image_list[i] 
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        #TODO: 对图像进行预处理，转换为 PyTorch Tensor，并添加一个维度以匹配模型输入形状
        image = transform(img).unsqueeze(0)
        outputs = []
        images = np.float32(image)
        inputs[0].from_numpy(images)
        # 绑定context
        assert context.enqueue(inputs, outputs, queue).ok()
        # 执行推理
        assert queue.sync().ok()
        pred = torch.from_numpy(np.array(outputs[0].asnumpy()))
        
        ## 计算概率，以及输出top1、top5
        #TODO: 去掉维度为 1 的维度，以便后续计算
        pred = pred.squeeze(0)
        #TODO: 对推理结果进行 softmax 归一化，得到每个类别的概率
        pred = torch.softmax(pred, dim=0)
        #TODO: 将 PyTorch Tensor 转换为 NumPy 数组
        pred = pred.numpy()
        top_5 = pred.argsort()[-5:][::-1]
        top_1 = pred.argsort()[-1:][::-1]
        print('test img is ', path)
        print('top1_cls:', top_1)
        print('top1_prob:', pred[top_1])
        print('top5_cls:', top_5)
        print('top5_prob:', pred[top_5[0]], pred[top_5[1]], 
               pred[top_5[2]], pred[top_5[3]], pred[top_5[4]])
    print("swin_transformer_infer Pass!")

if __name__ == '__main__':
    args, config = parse_option()
    main(config)
