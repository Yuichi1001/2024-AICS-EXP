from stu_upload.vgg19_demo import VGG19
import time
import numpy as np
import os
import scipy.io

def evaluate(vgg):
    start = time.time()
    vgg.forward()
    end = time.time()
    print('inference time: %f'%(end - start))
    result = vgg.net.getOutputData()
    prob = max(result)
    top1 = result.index(prob)
    print('Classification result: id = %d, prob = %f'%(top1, prob))


if __name__ == '__main__':
    vgg = VGG19()
   
    vgg.build_model(param_path='../imagenet-vgg-verydeep-19.mat')
    vgg.load_model()
    vgg.load_image('../cat1.jpg')
    #evaluate(vgg)
    for i in range(10):
        evaluate(vgg)
