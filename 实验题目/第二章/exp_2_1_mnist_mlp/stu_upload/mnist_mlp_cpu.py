import sys
import numpy as np
import struct
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer

MNIST_DIR = "../mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass


class MNIST_MLP(object):
    def __init__(self, batch_size=100, input_size=784,hidden1=32,hidden2=16,out_classes=10,lr=0.01, max_epoch=1,print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter


    def load_mnist(self,file_dir, is_images = 'True'):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()


        if is_images:
           fmt_header = '>iiii'
           magic,num_images,num_rows,num_cols=struct.unpack_from(fmt_header,bin_data,0)
        else:
           fmt_header = '>ii'
           magic,num_images=struct.unpack_from(fmt_header,bin_data,0)
           num_rows, num_cols= 1,1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))

        mat_data = np.reshape(mat_data,[num_images,num_rows*num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
        return mat_data

    def load_data(self):
        # TODO: 调用函数 load_mnist 读取和预处理 MNIST 中训练数据和测试数据的图像和标记
        print('Loading MNIST data from files...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = _______________________________ 
        test_images = ________________________________
        test_labels = _________________________________

        self.train_data=np.append(train_images,train_labels, axis=1)
        self.test_data=np.append(test_images, test_labels, axis=1)



    def shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)

    def build_model(self):  # 建立网络结构
        # TODO：建立三层神经网络结构
        print('Building multi-layer perception model...')
        self.fc1=FullyConnectedLayer(self.input_size, self.hidden1) 
        self.relu1=ReLULayer()
        ______________________
        ______________________
        self.fc3=FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax=SoftmaxLossLayer()
        self.update_layer_list=[self.fc1,self.fc2,self.fc3]

    def init_model(self):
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer_list:
            layer.init_param()
    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params=np.load(param_dir,allow_pickle=True).item()
        #####weight参数
        self.fc1.load_param(params['w1'],params['b1'])
        self.fc2.load_param(params['w2'],params['b2'])
        self.fc3.load_param(params['w3'],params['b3'])


    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        print( params)
        np.save(param_dir, params)


    def forward(self, input):  # 神经网络的前向传播
        # TODO：神经网络的前向传播
        h1=self.fc1.forward(input)
        h1=self.relu1.forward(h1)
        ________________________
        prob=self.softmax.forward(h3)
        return prob



    def backward(self):   # 神经网络的反向传播
        # TODO：神经网络的反向传播
        dloss = self.softmax.backward()
        ________________________
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self,lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)
    



    def train(self):
        max_batch=self.train_data.shape[0] // self.batch_size ###python3

        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size,:-1] ##batchsize ,最后1列
                batch_labels = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size,-1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                   print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))
       



    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0]/self.batch_size)):
            batch_images=self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :1]
            prob = self.forward(batch_images)
            pred_labels=np.argmax(prob,axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size]=pred_labels
        accuracy = np.mean(pred_results==self.test_data[:,-1])
        print('Accuracy in test  set:%f' % accuracy)


def build_mnist_mlp(param_dir='weight.npy'):
    h1,h2,e=32,16,10
    mlp=MNIST_MLP(hidden1=h1,hidden2=h2,max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.train()
    mlp.save_model('mlp-%d-%d-%depoch.npy'%(h1,h2,e))
    mlp.load_model('mlp-%d-%d-%depoch.npy'%(h1,h2,e))
    return mlp


if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
