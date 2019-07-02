# -*- coding: utf-8 -*-#

"""
File         :      mnist1.py
Description  :  
Author       :      赵金朋
Modify Time  :      2019/7/2 11:21
"""
"""
显示训练测试验证集大小
"""
from tensorflow.examples.tutorials.mnist import  input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

"""
softmax
"""
import  tensorflow as tf
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784])
w=tf.Variable(tf.zeros([784,10]))

