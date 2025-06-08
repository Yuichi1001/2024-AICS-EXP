import magicmind.python.runtime as mm
import torch
import yaml
import numpy as np
import cv2
import os
import glob
import math


def yolov5_add_DetectionOutput(network, #mm.builder
                               perms,   #[0, 2, 3, 1] nchw-nhwc
                               anchors, #[10, 13, 16, 30, 33, 23, 30, 61, 62,45, 59, 119, 116, 90, 156, 198, 373, 326 ]
                               batch_size,
                               confidence_thresh, # 0.5
                               nms_thresh, #0.45
                               scale,      #1.0
                               class_num,  #80
                               img_shape  #640 640
                               ):  

    yolov5m_network = network
    # 创建常量节点，表示坐标变换的顺序
    const_node = yolov5m_network.add_i_const_node(mm.DataType.INT32, mm.Dims(
            [len(perms)]), np.array(perms, dtype=np.int32))
    output_tensors = []
    for i in range(yolov5m_network.get_output_count()):
        #TODO: 获取yolov5m_network的当前输出节点
        tensor = yolov5m_network.get_output(i)
        # 添加 Permute 操作
        permute_node = yolov5m_network.add_i_permute_node(tensor, const_node.get_output(0))
        # 将 Permute 操作的输出节点添加到列表中
        output_tensors.append(permute_node.get_output(0))
    output_count = yolov5m_network.get_output_count()
    for i in range(output_count):
        # remove tensor of the original network
        yolov5m_network.unmark_output(yolov5m_network.get_output(0))

    #TODO: 创建常量节点，表示锚框的大小
    anchors_node = yolov5m_network.add_i_const_node(mm.DataType.FLOAT32, mm.Dims([len(anchors)]),
        np.array(anchors, dtype=np.float32))
    #TODO: 为yolov5m_network添加 DetectionOutput 节点，连接 Permute 操作的输出和锚框常量节点
    detect_out = yolov5m_network.add_i_detection_output_node(output_tensors, anchors_node.get_output(0))

    # 设置 DetectionOutput 节点的参数
    detect_out.set_algo(mm.IDetectionOutputAlgo.YOLOV5)
    detect_out.set_batch_size(batch_size)
    detect_out.set_confidence_thresh(confidence_thresh)
    detect_out.set_nms_thresh(nms_thresh)
    detect_out.set_scale(scale)
    detect_out.set_num_coord(4)
    detect_out.set_num_class(class_num)
    detect_out.set_num_entry(5)
    detect_out.set_num_anchor(3)
    detect_out.set_num_box_limit(1024)
    detect_out.set_image_shape(img_shape[0], img_shape[1])
    detect_out.set_layout(mm.Layout.NONE, mm.Layout.NONE)
    # mark layer detect_out as network output
    detection_output_count = detect_out.get_output_count()
    for i in range(detection_output_count):
        yolov5m_network.mark_output(detect_out.get_output(i))

    return yolov5m_network


def plot_images(pred, detection_num, img_src, ratio, imgsz):

    reshape_value = torch.reshape(pred, (-1, 1))
    src_h, src_w = img_src.shape[0],img_src.shape[1]
    scale_w = ratio * src_w 
    scale_h = ratio * src_h
    
    # yaml_dir = '.coco.yaml'
    # f1 = open(yaml_dir)
    # config_params = yaml.load(f1, Loader=yaml.FullLoader)
    # a = config_params['names']

    a =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names

    box_step = 7

    # 画图
    for k in range(detection_num):
    
        left = max(0, min(reshape_value[k * box_step + 3], imgsz)) 
        right = max(0, min(reshape_value[k * box_step + 5], imgsz))
        top = max(0, min(reshape_value[k * box_step + 4], imgsz))
        bottom = max(0, min(reshape_value[k * box_step + 6], imgsz))
        class_id = int(reshape_value[k * box_step + 1])
        score = float(reshape_value[k * box_step + 2])
        left = (left - (imgsz - scale_w) / 2)  
        right = (right - (imgsz - scale_w) / 2) 
        top = (top - (imgsz - scale_h) / 2) 
        bottom = (bottom - (imgsz - scale_h) / 2) 
        left = float(max(0, left))
        right = float(max(0, right))
        top = float(max(0, top))
        bottom = float(max(0, bottom))
        if (left <= 0 or right <= 0 or top <= 0 or bottom <= 0 ):
            continue

        cv2.rectangle(img_src,(int(left),int(top)),(int(right),int(bottom)),(0,255,0))
        cv2.putText(img_src, str(a[class_id]), (int(left), int(bottom - 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 0, 255), thickness=2)

    return img_src


def letterbox(img, dst_shape, stride = 32):
    src_h, src_w = img.shape[0], img.shape[1]
    dst_h, dst_w = dst_shape
    ratio = min(dst_h / src_h, dst_w / src_w)
    unpad_h, unpad_w = int(math.floor(src_h * ratio)), int(math.floor(src_w * ratio))
    if ratio != 1:
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (unpad_w, unpad_h), interp)
    # padding
    pad_t = int(math.floor((dst_h - unpad_h) / 2))
    pad_b = dst_h - unpad_h - pad_t
    pad_l = int(math.floor((dst_w - unpad_w) / 2))
    pad_r = dst_w - unpad_w - pad_l
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, ratio, pad_t, pad_l #增加返回pad


def preprocess_image(img, dst_shape) -> np.ndarray:
    # resize as letterbox
    img, ratio, _ , _ = letterbox(img, dst_shape)
    # BGR to RGB, HWC to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    # normalize
    img = img.astype(dtype = np.float32) / 255.0
    return img


class FixedCalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        print(img_dir)
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.jpg')
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.dst_shape_ = (self.shape_.GetDimValue(2), self.shape_.GetDimValue(3))

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_
    
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        for i in range(data_begin, data_end):
            img = cv2.imread(self.data_paths_[i])
            img = preprocess_image(img, self.dst_shape_)
            imgs.append(img[np.newaxis,:])
        # batch and normalize
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0))

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        data_begin = self.cur_data_index_
        data_end = data_begin + batch_size
        if data_end > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
        self.cur_sample_ = self.preprocess_images(data_begin, data_end)
        self.cur_data_index_ = data_end
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        return mm.Status.OK()