import torch
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from src.models.yolo import Model
from src.utils.yolov5_mm_utils import FixedCalibData,yolov5_add_DetectionOutput,letterbox, plot_images
from src.utils.general import check_img_size

import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser

batch_size = 1
IMAGE_PATH = "./course_images/images/000000110721.jpg"

def do_calibrate(network, calib_data, config):
    # 创建量化校准器
    calibrator = mm.Calibrator([calib_data])
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    assert config.parse_from_string(
        #TODO: 配置量化精度模式，以qint8作为输入精度，输出数据类型和中间结果均为FLOAT16
        """{"precision_config": {"precision_mode": "qint8_mixed_float16"}}""").ok()
    # 将校准器绑定Network
    calibrator.calibrate(network, config)
    print("Calibrator Pass!")

##################### 加载原始yolov5m pt 模型#####################################

# TODO:从文件中加载预训练的模型参数字典
net_dict = torch.load("yolov5_model/yolov5m.pt")['model'].state_dict()
net_modified = Model("src/models/yolov5m.yaml")
# TODO:将加载的预训练参数字典net_dict加载到自定义模型net_modified中，其中不加载不匹配的键。
net_modified.load_state_dict(net_dict, strict=False)

#################### 生成不带后处理的yolov5m pt 模型###############################
trace_input = torch.randn(1, 3, 640, 640).float()
#TODO: 使用 torch.jit.trace 将模型转为 Torch 脚本
model_mm = torch.jit.trace(net_modified.float().eval(), trace_input, check_trace=False)
#TODO: 将转换后的 Torch 脚本保存到文件
torch.jit.save(model_mm, "yolov5_model/yolov5_mm.pt")
print("Generate YOLOv5-mm model Pass!")

#TODO:  创建MagicMind Builder对象
builder = mm.Builder()
#TODO: 创建 MagicMind Network 对象
yolov5m_network = mm.Network()
#TODO: 创建 MagicMind Config 对象
config = mm.BuilderConfig()
# 创建 MagicMind Parser （pytorch后端）
parser = Parser(mm.ModelKind.kPytorch)

# 获取 Network 的输入节点及其维度
parser.set_model_param("pytorch-input-dtypes", [mm.DataType.FLOAT32])
#TODO: 将 yolov5_model/yolov5_mm.pt 与 MagicMind框架下的yolov5m_network绑定
assert parser.parse(yolov5m_network, "yolov5_model/yolov5_mm.pt").ok()
# 设置 MagicMind Network 参数：硬件平台、自动int64转int32、卷积折叠，可变输入开关等
config.parse_from_string('{"archs":["mtp_372"]}')
config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}')
config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}')
config.parse_from_string('{"graph_shape_mutable": false}')

# 设置输入节点维度
assert yolov5m_network.get_input(0).set_dimension(mm.Dims((batch_size, 3, 640, 640))).ok() 

# 设置输入的摆数
assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}').ok() 

# 设置输入归一化参数: 0，std: 255 (var=65025).  (input - mean)/std
assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean": [0, 0, 0], "var": [65025, 65025, 65025]}}}').ok() 

# 设置高性能 yolov5后处理算子相关参数
perms = [0, 2, 3, 1]
anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62,45, 59, 119, 116, 90, 156, 198, 373, 326 ]
confidence = 0.5
nms_thresh = 0.45
scale = 1.0
class_num = 80
img_shape = [640, 640]

# 将高性能 yolov5后处理算子添加至 yolov5m_networ
yolov5m_network = yolov5_add_DetectionOutput(yolov5m_network,
                                            perms,
                                            anchors, 
                                            int(batch_size),  # 正确位置的batch_size
                                            confidence,
                                            nms_thresh,
                                            scale,
                                            class_num,   
                                            img_shape)
print("Add yolov5-post-processing-op Pass!")

#TODO: 对量化校准器指定输入数据
calib_data = FixedCalibData(shape = mm.Dims([batch_size, 3, 640, 640]),
                            max_samples = 10,
                            img_dir = "./course_images/images/")
#TODO: 执行量化校准函数
do_calibrate(yolov5m_network, calib_data, config)

#TODO: 执行 MagicMind 模型生成
model = builder.build_model("pytorch_yolov5_inference_fusedOp_model", yolov5m_network, config)
assert model != None
#TODO: 将生成的MagicMind模型序列化并保存至本地
model.serialize_to_file("yolov5_model/pytorch_yolov5_inference_fusedOp_model")
print("Generate model done, model save to %s" % "yolov5_model/pytorch_yolov5_inference_fusedOp_model")

# 读取图片
img_pre = cv2.imread(IMAGE_PATH)  # BGR

stride = 32 
imgsz = [640, 640]
imgsz = check_img_size(imgsz[0], s= stride)  # 检测输入size
# 执行前处理
img_post, ratio,_,_ = letterbox(img_pre, (imgsz,imgsz), stride)
img_post = img_post[:, :, ::-1].transpose(2, 0, 1)  # BGR 转 RGB
img_post = np.ascontiguousarray(img_post)

with mm.System() as sys:
    
    #TODO: 创建 Device
    dev = mm.Device()
    dev.id = 0 # 设置 Device Id
    assert dev.active().ok()

    #TODO: 实例化一个MagicMind模型对象
    model = mm.Model()
    #TODO: 从文件中反序列化模型
    model.deserialize_from_file("yolov5_model/pytorch_yolov5_inference_fusedOp_model")
    
    #TODO: 创建 MagicMind engine
    engine = model.create_i_engine()
    #TODO: 创建MagicMind context 
    context = engine.create_i_context()
    
    #TODO: 准备 MagicMind 输入输出 Tensor 节点
    inputs = context.create_inputs()

    #TODO: 将img_post转换为 PyTorch 张量
    img_post = torch.from_numpy(img_post)
    img_post = img_post.permute(1,2,0)
    if img_post.ndimension() == 3:
        #TODO: 在img_post第0维度上添加一个维度
        img_post = img_post.unsqueeze(0)
    inputs_test = np.float32(img_post)
    inputs[0].from_numpy(inputs_test)
    outputs = []
    
    #TODO: 根据 Device 信息创建 queue 实例
    queue = dev.create_queue()
    #TODO: 绑定context
    assert context.enqueue(inputs, outputs, queue).ok()
    #TODO: 执行推理,同步执行队列中的任务
    context.enqueue(inputs, outputs, queue).ok()
    
    # 返回推理结果
    pred = torch.from_numpy(np.array(outputs[0].asnumpy()))
    detection_num = torch.from_numpy(np.array(outputs[1].asnumpy()))
    
    # 画图
    img_plot = plot_images(pred, detection_num, img_pre, ratio, imgsz)
    # cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
    cv2.imwrite("course_images/result.png", img_plot)
    print("The results are saved in ./course_images/")
    print("YOLOv5 infer Pass!")
