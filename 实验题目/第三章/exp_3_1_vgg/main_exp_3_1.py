from stu_upload.vgg_cpu import VGG19
import numpy as np
import struct
import os
import scipy.io
import time
#import os
import sys



def computeMse(data1,data2):
    errors = []
    for i in range(len(data1)):
        errors.append(data1[i]-data2[i])

    squared_error = []
    for val in errors:
        squared_error.append(pow(val, 2))
    
    return sum(squared_error) / len(squared_error)
def forward(vgg):
    print('Inferencing...')
    start_time = time.time()
    current = vgg.input_image
    pool5 = np.array([])
    for idx in range(len(vgg.param_layer_name)):
        print('Inferencing layer: ' + vgg.param_layer_name[idx])
        
        current = vgg.layers[vgg.param_layer_name[idx]].forward(current)
       

        if 'pool5' in vgg.param_layer_name[idx]:
            pool5 = current
    print('Inference time: %f' % (time.time()-start_time))
    return current, pool5

def check_pool5(stu_pool5):
    data = np.load('pool5_dump.npy')
    pool5_mse = computeMse(stu_pool5.flatten(), data.flatten())
    print('test pool5 mse: %f'%pool5_mse)

    if pool5_mse < 0.003:
        print('CHECK POOL5 PASS.')
    else:
        print('CHECK POOL5 FAILED.')
        exit()

def evaluate(vgg):
    prob, pool5 = forward(vgg)
    print('--------------检测结果------------------------')
    
    top1 = np.argmax(prob[0])
    print('Classification result: id = %d, prob = %f' % (top1, prob[0, top1]))
    return pool5

if __name__ == '__main__':
    test_conv_and_pool_layer()
    print('-------------------------------')
    vgg = VGG19(param_path='../imagenet-vgg-verydeep-19.mat')
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    vgg.load_image('../cat1.jpg')
    pool5 = evaluate(vgg)
    print('-------------------------------')
    check_pool5(pool5)
