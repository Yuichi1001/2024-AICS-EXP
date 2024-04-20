import numpy as np
import struct
import os
import scipy.io
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
from layers_2 import ConvolutionalLayer, MaxPoolingLayer, FlattenLayer
from layers_3 import ContentLossLayer, StyleLossLayer

class VGG19(object):
    def __init__(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path
        self.param_layer_name = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
        ]

    def build_model(self):
        # TODO： 建立VGG19网络结构
        # 可以通过设置 type=1 来使用优化后的卷积和池化层，如 ConvolutionalLayer(3, 3, 64, 1, 1, type=1)
        print('Building vgg-19 model...')
        type = 1
        self.layers = {}
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 64, 1, 1, type)
        self.layers['relu1_1'] = ReLULayer()
        self.layers['conv1_2'] = ConvolutionalLayer(3, 64, 64, 1, 1, type)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(2, 2, type)

        self.layers['conv2_1'] = ConvolutionalLayer(3, 64, 128, 1, 1, type)
        self.layers['relu2_1'] = ReLULayer()
        self.layers['conv2_2'] = ConvolutionalLayer(3, 128, 128, 1, 1, type)
        self.layers['relu2_2'] = ReLULayer()
        self.layers['pool2'] = MaxPoolingLayer(2, 2, type)
        self.layers['conv3_1'] = ConvolutionalLayer(3, 128, 256, 1, 1, type)
        self.layers['relu3_1'] = ReLULayer()
        self.layers['conv3_2'] = ConvolutionalLayer(3, 256, 256, 1, 1, type)
        self.layers['relu3_2'] = ReLULayer()
        self.layers['conv3_3'] = ConvolutionalLayer(3, 256, 256, 1, 1, type)
        self.layers['relu3_3'] = ReLULayer()
        self.layers['conv3_4'] = ConvolutionalLayer(3, 256, 256, 1, 1, type)
        self.layers['relu3_4'] = ReLULayer()
        self.layers['pool3'] = MaxPoolingLayer(2, 2, type)
        self.layers['conv4_1'] = ConvolutionalLayer(3, 256, 512, 1, 1, type)
        self.layers['relu4_1'] = ReLULayer()
        self.layers['conv4_2'] = ConvolutionalLayer(3, 512, 512, 1, 1, type)
        self.layers['relu4_2'] = ReLULayer()
        self.layers['conv4_3'] = ConvolutionalLayer(3, 512, 512, 1, 1, type)
        self.layers['relu4_3'] = ReLULayer()
        self.layers['conv4_4'] = ConvolutionalLayer(3, 512, 512, 1, 1, type)
        self.layers['relu4_4'] = ReLULayer()
        self.layers['pool4'] = MaxPoolingLayer(2, 2, type)
        self.layers['conv5_1'] = ConvolutionalLayer(3, 512, 512, 1, 1, type)
        self.layers['relu5_1'] = ReLULayer()
        self.layers['conv5_2'] = ConvolutionalLayer(3, 512, 512, 1, 1, type)
        self.layers['relu5_2'] = ReLULayer()
        self.layers['conv5_3'] = ConvolutionalLayer(3, 512, 512, 1, 1, type)
        self.layers['relu5_3'] = ReLULayer()

        self.layers['conv5_4'] = ConvolutionalLayer(3, 512, 512, 1, 1, type)
        self.layers['relu5_4'] = ReLULayer()
        self.layers['pool5'] = MaxPoolingLayer(2, 2, type)

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer in vgg-19...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        print('Get image mean: ' + str(self.image_mean))
        for idx in range(37):
            if 'conv' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # matconvnet: weights dim [height, width, in_channel, out_channel]
                # ours: weights dim [in_channel, height, width, out_channel]
                weight = np.transpose(weight,[2,0,1,3])
                bias = bias.reshape(-1)
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)

    def load_image(self, image_dir, image_height, image_width):
        print('Loading and preprocessing image from ' + image_dir)
        self.input_image = scipy.misc.imread(image_dir)
        image_shape = self.input_image.shape
        self.input_image = scipy.misc.imresize(self.input_image,[image_height, image_width,3])
        self.input_image = np.array(self.input_image).astype(np.float32)
        self.input_image -= self.image_mean
        self.input_image = np.reshape(self.input_image, [1]+list(self.input_image.shape))
        # input dim [N, channel, height, width]
        # TODO: 调整输入数据的形状
        self.input_image = np.transpose(self.input_image, (0, 3, 1, 2))
        return self.input_image, image_shape

    def save_image(self, input_image, image_shape, image_dir):
        #print('Save image at ' + image_dir)
        # TODO：调整输出图片的形状
        input_image = np.transpose(input_image, (0, 2, 3, 1))
        input_image = input_image[0] + self.image_mean
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)
        input_image = scipy.misc.imresize(input_image, image_shape)
        scipy.misc.imsave(image_dir, input_image)

    def forward(self, input_image, layer_list):
        start_time = time.time()
        current = input_image
        layer_forward = {}
        for idx in range(len(self.param_layer_name)):
            # TODO： 计算VGG19网络的前向传播
            current = self.layers[self.param_layer_name[idx]].forward(current)
            if self.param_layer_name[idx] in layer_list:
                layer_forward[self.param_layer_name[idx]] = current
        #print('Forward time: %f' % (time.time()-start_time))
        return layer_forward

    def backward(self, dloss, layer_name):
        start_time = time.time()
        layer_idx = list.index(self.param_layer_name, layer_name)
        for idx in range(layer_idx, -1, -1):
            # TODO： 计算VGG19网络的反向传播
            dloss = self.layers[self.param_layer_name[idx]].backward(dloss)

        #print('Backward time: %f' % (time.time()-start_time))
        return dloss

def get_random_img(content_image, noise):
    # 生成风格迁移初始化图片
    noise_image = np.random.uniform(-20, 20, content_image.shape)
    random_img = noise_image * noise + content_image * (1 - noise)
    return random_img

class AdamOptimizer(object):
    def __init__(self, lr, diff_shape):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.lr = lr
        self.mt = np.zeros(diff_shape)
        self.vt = np.zeros(diff_shape)
        self.step = 0
    def update(self, input, grad):
        # TODO：补全参数更新过程
        self.step += 1
        self.mt = self.beta1*self.mt + (1-self.beta1)*grad
        self.vt = self.beta2*self.vt + (1-self.beta2)*(grad**2)
        mt_hat = self.mt / (1-self.beta1**self.step)
        vt_hat = self.vt / (1-self.beta2**self.step)
        # TODO： 利用梯度的一阶矩和二阶矩的无偏估计更新风格迁移图像
        output = input - self.lr * mt_hat / (np.sqrt(vt_hat + self.eps))
        return output


if __name__ == '__main__':

    CONTENT_LOSS_LAYERS = ['relu4_2']
    STYLE_LOSS_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    NOISE = 0.5
    ALPHA, BETA = 1, 500
    TRAIN_STEP = 1
    LEARNING_RATE = 1.0
    IMAGE_HEIGHT, IMAGE_WIDTH = 192, 320

    vgg = VGG19()
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    content_loss_layer = ContentLossLayer()
    style_loss_layer = StyleLossLayer()
    adam_optimizer = AdamOptimizer(1.0, [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH])

    content_image, content_shape = vgg.load_image('../../weinisi.jpg', IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image, _ = vgg.load_image('../../style.jpg', IMAGE_HEIGHT, IMAGE_WIDTH)
    content_layers = vgg.forward(content_image, CONTENT_LOSS_LAYERS)
    style_layers = vgg.forward(style_image, STYLE_LOSS_LAYERS)
    transfer_image = get_random_img(content_image, NOISE)

    for step in range(TRAIN_STEP):
        transfer_layers = vgg.forward(transfer_image, CONTENT_LOSS_LAYERS + STYLE_LOSS_LAYERS)
        content_loss = np.array([])
        style_loss = np.array([])
        content_diff = np.zeros(transfer_image.shape)
        style_diff = np.zeros(transfer_image.shape)
        for layer in CONTENT_LOSS_LAYERS:
            # TODO： 计算内容损失的前向传播
            current_loss = content_loss_layer.forward(transfer_layers[layer], content_layers[layer])
            content_loss = np.append(content_loss, current_loss)
            # TODO： 计算内容损失的反向传播
            dloss = content_loss_layer.backward(transfer_layers[layer], content_layers[layer])
            content_diff += vgg.backward(dloss, layer)
        for layer in STYLE_LOSS_LAYERS:
            # TODO： 计算风格损失的前向传播
            current_loss = style_loss_layer.forward(transfer_layers[layer], style_layers[layer])
            style_loss = np.append(style_loss, current_loss)
            # TODO： 计算风格损失的反向传播
            dloss = style_loss_layer.backward(transfer_layers[layer], style_layers[layer])
            style_diff += vgg.backward(dloss, layer)
        total_loss = ALPHA * np.mean(content_loss) + BETA * np.mean(style_loss)
        image_diff = ALPHA * content_diff / len(CONTENT_LOSS_LAYERS) + BETA * style_diff / len(STYLE_LOSS_LAYERS)
        # TODO： 利用Adam优化器对风格迁移图像进行更新
        transfer_image = adam_optimizer.update(transfer_image, image_diff)
        if step % 20 == 0:
            print('Step %d, loss = %f' % (step, total_loss), content_loss, style_loss)
            vgg.save_image(transfer_image, content_shape, '../output/output_' + str(step) + '.jpg')



