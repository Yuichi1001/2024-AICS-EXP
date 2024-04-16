import numpy as np
import struct
import os
import scipy.io
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
from layers_2 import ConvolutionalLayer, MaxPoolingLayer, FlattenLayer

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

class VGG19(object):
    def __init__(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path
        self.param_layer_name = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
        )        

    def build_model(self):
        # TODO：定义VGG19 的网络结构
        print('Building vgg-19 model...')

        self.layers = {}
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 64, 1, 1)
        self.layers['relu1_1'] = ReLULayer()
        _______________________________
        _______________________________
        _______________________________

        self.layers['fc8'] = FullyConnectedLayer(4096, 1000)

        self.layers['softmax'] = SoftmaxLossLayer()

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer in vgg-19...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        #print(params)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        print('Get image mean: ' + str(self.image_mean))

        for idx in range(43):
            if 'conv' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # matconvnet: weights dim [height, width, in_channel, out_channel]
                # ours: weights dim [in_channel, height, width, out_channel]
                # TODO：调整参数的形状
                weight = ______________________
                bias = ______________________
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)
            if idx >= 37 and 'fc' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx-1][0][0][0][0]
                weight = ___________________________________
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)

    def load_image(self, image_dir):
        print('Loading and preprocessing image from ' + image_dir)
        self.input_image = scipy.misc.imread(image_dir)
        self.input_image = scipy.misc.imresize(self.input_image,[224,224,3])
        self.input_image = np.array(self.input_image).astype(np.float32)
        self.input_image -= self.image_mean
        self.input_image = np.reshape(self.input_image, [1]+list(self.input_image.shape))
        # input dim [N, channel, height, width]
        # TODO：调整图片维度顺序
        self.input_image = ________________________

    def forward(self):   # TODO：神经网络的前向传播
        print('Inferencing...')
        start_time = time.time()
        current = self.input_image
        for idx in range(len(self.param_layer_name)):
            print('Inferencing layer: ' + self.param_layer_name[idx])
            current = __________________________________
        print('Inference time: %f' % (time.time()-start_time))
        return current

    def evaluate(self):
        # TODO：获取神经网络前向传播的结果
        prob = __________________________
        top1 = np.argmax(prob[0])
        print('Classification result: id = %d, prob = %f' % (top1, prob[0, top1]))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    vgg.load_image('../../cat1.jpg')
    prob = vgg.evaluate()
    
