# coding:utf-8
from stu_upload.exp_3_3_style_transfer import *
from stu_upload.layers_2 import ConvolutionalLayer, MaxPoolingLayer
#from stu_upload.layers_3 import ContentLossLayer, StyleLossLayer
import numpy as np
import struct
import os
import scipy.io
import time

def computeMse(data1,data2):
    errors = []
    for i in range(len(data1)):
        errors.append(data1[i]-data2[i])

    squared_error = []
    for val in errors:
        squared_error.append(pow(val, 2))
    
    return sum(squared_error) / len(squared_error)

def test_speed_up():
    test_data = np.random.rand(1, 256, 24, 40)
    test_dloss = np.random.rand(1, 256, 24, 40)
    test_filter = np.random.rand(256, 3, 3, 256)
    test_bias = np.random.rand(256)

    conv = ConvolutionalLayer(3, 256, 256, 1, 1)
    conv.init_param()
    conv.load_param(test_filter, test_bias)
    stamp = time.time()
    conv_forward_result = conv.forward(test_data)
    conv_forward_time = time.time()-stamp
    print('conv forward raw time: %f ms'%(conv_forward_time*1000))
    stamp = time.time()
    conv_backward_result = conv.backward(test_dloss)
    conv_backward_time = time.time()-stamp
    print('conv backward raw time: %f ms'%(conv_backward_time*1000))

    speedup_conv = ConvolutionalLayer(3, 256, 256, 1, 1, 1)
    speedup_conv.init_param()
    speedup_conv.load_param(test_filter, test_bias)
    stamp = time.time()
    speedup_conv_forward_result = speedup_conv.forward(test_data)
    speedup_conv_forward_time = time.time()-stamp
    print('conv forward speedup time: %f ms'%(speedup_conv_forward_time*1000))
    stamp = time.time()
    speedup_conv_backward_result = speedup_conv.backward(test_dloss)
    speedup_conv_backward_time = time.time()-stamp
    print('conv backward speedup time: %f ms'%(speedup_conv_backward_time*1000))

    speedup_conv_forward_mse = computeMse(conv_forward_result.flatten(), speedup_conv_forward_result.flatten())
    speedup_conv_backward_mse = computeMse(conv_backward_result.flatten(), speedup_conv_backward_result.flatten())
    if speedup_conv_forward_mse < 0.003 and speedup_conv_backward_mse < 0.003:
        print('SPEEDUP CONV TEST PASS.')
    else:
        print('SPEEDUP CONV TEST FAILED.')
        exit()

    print('CONV FORWARD SPEEDUP RATIO: %f'%(conv_forward_time / speedup_conv_forward_time))
    print('CONV BACKWARD SPEEDUP RATIO: %f'%(conv_backward_time / speedup_conv_backward_time))

if __name__ == '__main__':
    np.random.seed(1234)
    print('-------------------------')
    test_speed_up()
    print('-------------------------')
    CONTENT_LOSS_LAYERS = ['relu4_2']
    STYLE_LOSS_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    NOISE = 0.5
    ALPHA, BETA = 1, 500
    TRAIN_STEP = 100
    LEARNING_RATE = 1.0
    IMAGE_HEIGHT, IMAGE_WIDTH = 192, 320

    vgg = VGG19(param_path='../imagenet-vgg-verydeep-19.mat')
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    content_loss_layer = ContentLossLayer()
    style_loss_layer = StyleLossLayer()
    adam_optimizer = AdamOptimizer(1.0, [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH])

    content_image, content_shape = vgg.load_image('../weinisi.jpg', IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image, _ = vgg.load_image('../style.jpg', IMAGE_HEIGHT, IMAGE_WIDTH)
    content_layers = vgg.forward(content_image, CONTENT_LOSS_LAYERS)
    style_layers = vgg.forward(style_image, STYLE_LOSS_LAYERS)
    transfer_image = get_random_img(content_image, NOISE)

    start = time.time()
    for step in range(TRAIN_STEP):
        transfer_layers = vgg.forward(transfer_image, CONTENT_LOSS_LAYERS + STYLE_LOSS_LAYERS)
        content_loss = np.array([])
        style_loss = np.array([])
        content_diff = np.zeros(transfer_image.shape)
        style_diff = np.zeros(transfer_image.shape)
        for layer in CONTENT_LOSS_LAYERS:
            # TODO： 计算内容损失的前向传播
            current_loss = ____________________________________
            content_loss = np.append(content_loss, current_loss)
            # TODO： 计算内容损失的反向传播
            dloss = content_loss_layer.backward(transfer_layers[layer], content_layers[layer])
            content_diff += ___________________________________
        for layer in STYLE_LOSS_LAYERS:
            # TODO： 计算风格损失的前向传播
            current_loss = ___________________________________
            style_loss = np.append(style_loss, current_loss)
            # TODO： 计算风格损失的反向传播
            dloss = style_loss_layer.backward(transfer_layers[layer], style_layers[layer])
            style_diff += ____________________________________
        total_loss = ALPHA * np.mean(content_loss) + BETA * np.mean(style_loss)
        image_diff = ALPHA * content_diff / len(CONTENT_LOSS_LAYERS) + BETA * style_diff / len(STYLE_LOSS_LAYERS)
        # TODO： 利用Adam优化器对风格迁移图像进行更新
        transfer_image = _____________________________________
        if step % 1 == 0:
            print('Step %d, loss = %f' % (step, total_loss), content_loss, style_loss)
            print('cost time: %f'%(time.time() - start))
            vgg.save_image(transfer_image, content_shape, 'output/output_' + str(step) + '.jpg')
            start = time.time()
