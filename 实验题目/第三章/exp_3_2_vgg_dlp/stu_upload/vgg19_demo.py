# -*- coding: UTF-8 -*- 
import pycnnl
import time
import numpy as np
import os
import scipy.io

class VGG19(object):
    def __init__(self):
        # set up net
        
        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []
        self.filter_quant_params = []

   
    def build_model(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path

       
        # TODO: 使用net的createXXXLayer接口搭建VGG19网络
        # creating layers
        self.net.setInputShape(1, 3, 224, 224)
        # conv1_1
       
        input_shape1=pycnnl.IntVector(4)
        input_shape1[0]=1
        input_shape1[1]=3
        input_shape1[2]=224
        input_shape1[3]=224
        self.net.createConvLayer('conv1_1', input_shape1,64, 3, 1, 1, 1)
             
        # relu1_1
        self.net.createReLuLayer('relu1_1')
        # conv1_2
        
        input_shape12=pycnnl.IntVector(4)
        input_shape12[0]=1
        input_shape12[1]=64
        input_shape12[2]=224
        input_shape12[3]=224

        self.net.createConvLayer('conv1_2',input_shape12, 64, 3, 1, 1, 1)
        
        # relu1_2
        self.net.createReLuLayer('relu1_2')
        
        __________________________________
        # fc8
        
        input_shapem3=pycnnl.IntVector(4)
        input_shapem3[0]=1
        input_shapem3[1]=1
        input_shapem3[2]=1
        input_shapem3[3]=4096
        weight_shapem3=pycnnl.IntVector(4)
        weight_shapem3[0]=1
        weight_shapem3[1]=1
        weight_shapem3[2]=4096
        weight_shapem3[3]=1000
        output_shapem3=pycnnl.IntVector(4)
        output_shapem3[0]=1
        output_shapem3[1]=1
        output_shapem3[2]=1
        output_shapem3[3]=1000

        self.net.createMlpLayer('fc8', input_shapem3,weight_shapem3,output_shapem3)
        
        # softmax
        
        input_shapes=pycnnl.IntVector(3)
        input_shapes[0]=1
        input_shapes[1]=1
        input_shapes[2]=1000
    

        self.net.createSoftmaxLayer('softmax',input_shapes ,1)
    
    def load_model(self):
        # loading params ... 
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        
        count = 0
        for idx in range(self.net.size()):
            if 'conv' in self.net.getLayerName(idx):
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # TODO：调整权重形状
                # matconvnet: weights dim [height, width, in_channel, out_channel]
                # ours: weights dim [out_channel, height, width,in_channel]
                weight = ______________________________
                bias = bias.reshape(-1).astype(np.float)
                self.net.loadParams(idx, weight, bias)
                count += 1
            if 'fc' in self.net.getLayerName(idx):
                # Loading params may take quite a while. Please be patient.
                weight, bias = params['layers'][0][idx][0][0][0][0]
               
                weight = weight.reshape([weight.shape[0]*weight.shape[1]*weight.shape[2], weight.shape[3]])
                weight = _______________________________
                bias = bias.reshape(-1).astype(np.float)
            
                self.net.loadParams(idx, weight, bias)
                count += 1

    def load_image(self, image_dir):
        # loading image
        self.image = image_dir
        image_mean = np.array([123.68, 116.779, 103.939])
        print('Loading and preprocessing image from ' + image_dir)
        input_image = scipy.misc.imread(image_dir)
        input_image = scipy.misc.imresize(input_image,[224,224,3])
        input_image = np.array(input_image).astype(np.float32)
        input_image -= image_mean
        input_image = np.reshape(input_image, [1]+list(input_image.shape))
        # input dim [N, height, width, channel] 2
        # TODO：调整输入数据
        input_data = ________________________
        
        self.net.setInputData(input_data)

    def forward(self):
        return self.net.forward()
    
    def get_top5(self, label):
        start = time.time()
        self.forward()
        end = time.time()

        result = self.net.getOutputData()

        # loading labels
        labels = []
        with open('../synset_words.txt', 'r') as f:
            labels = f.readlines()

        # print results
        top1 = False
        top5 = False
        print('------ Top 5 of ' + self.image + ' ------')
        prob = sorted(list(result), reverse=True)[:6]
        if result.index(prob[0]) == label:
            top1 = True
        for i in range(5):
            top = prob[i]
            idx = result.index(top)
            if idx == label:
                top5 = True
            print('%f - '%top + labels[idx].strip())

        print('inference time: %f'%(end - start))
        return top1,top5
    
    def evaluate(self, file_list):
        top1_num = 0
        top5_num = 0
        total_num = 0

        start = time.time()
        with open(file_list, 'r') as f:
            file_list = f.readlines()
            total_num = len(file_list)
            for line in file_list:
                image = line.split()[0].strip()
                label = int(line.split()[1].strip())
                vgg.load_image(image)
                top1,top5 = vgg.get_top5(label)
                if top1 :
                    top1_num += 1
                if top5 :
                    top5_num += 1
        end = time.time()

        print('Global accuracy : ')
        print('accuracy1: %f (%d/%d) '%(float(top1_num)/float(total_num), top1_num, total_num))
        print('accuracy5: %f (%d/%d) '%(float(top5_num)/float(total_num), top5_num, total_num))
        print('Total execution time: %f'%(end - start))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.load_model()
    vgg.evaluate('../file_list')
